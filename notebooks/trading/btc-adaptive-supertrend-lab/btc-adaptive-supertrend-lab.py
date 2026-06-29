# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "altair==6.1.0",
#     "marimo==0.23.9",
#     "numpy==2.3.5",
#     "pandas==2.3.3",
#     "polars==1.36.1",
#     "yfinance==1.4.1",
# ]
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import marimo as mo
    import polars as pl
    import numpy as np
    import pandas as pd
    import warnings
    from datetime import date

    warnings.filterwarnings("ignore")
    return date, mo, np, pd, pl


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # BTC Adaptive Supertrend Lab

    Self-optimizing research notebook for the **Modern Adaptive Supertrend [GBB]** on BTC/USD daily data.

    **How it works (no manual tuning):**
    1. Search a bounded grid of L1/L2/L3 layer toggles and continuous parameters on the **train** window.
    2. Pick the winner on the **validation** window (Sharpe after trading costs vs buy-and-hold).
    3. Freeze parameters and report **final out-of-sample** results — no re-tuning on OOS.

    **Lookahead controls:** signals use same-bar close; positions apply from the *next* bar; flip costs charged on turnover.

    This is research, not live trading. Yahoo Finance daily BTC history can be noisy before ~2014.
    """)
    return


@app.cell
def _(date):
    TRAIN_START = date(2014, 1, 1)
    TRAIN_END = date(2019, 12, 31)
    VAL_START = date(2020, 1, 1)
    VAL_END = date(2022, 12, 31)
    OOS_START = date(2023, 1, 1)

    TRADING_COST_BPS = 10.0
    OPTIMIZER_BUDGET = 36
    MIN_VAL_DAYS = 126
    OOS_SHARPE_MARGIN = 0.05

    DEFAULT_PARAMS = {
        "atr_period": 10,
        "mult_fallback": 3.0,
        "use_l1": True,
        "min_period": 5,
        "max_period": 50,
        "use_l2": True,
        "ker_len": 20,
        "ker_pctile": True,
        "ker_pct_win": 500,
        "pivot": 0.5,
        "trend_gain": 0.8,
        "chop_gain": 0.5,
        "mult_min": 1.0,
        "mult_max": 6.0,
        "use_l3": True,
        "hyst_atr": 0.5,
        "hyst_bars": 1,
    }

    CLASSIC_PARAMS = {
        **DEFAULT_PARAMS,
        "use_l1": False,
        "use_l2": False,
        "use_l3": False,
        "atr_period": 10,
        "mult_fallback": 3.0,
    }
    return (
        CLASSIC_PARAMS,
        DEFAULT_PARAMS,
        MIN_VAL_DAYS,
        OOS_SHARPE_MARGIN,
        OOS_START,
        OPTIMIZER_BUDGET,
        TRAIN_END,
        TRAIN_START,
        TRADING_COST_BPS,
        VAL_END,
        VAL_START,
    )


@app.cell
def _(np):
    def ehlers_dominant_cycle(src, min_period=5, max_period=50):
        n = len(src)
        period = np.full(n, 10.0)
        sp = np.full(n, 10.0)
        smooth = np.zeros(n)
        detrend = np.zeros(n)
        q1 = np.zeros(n)
        i1 = np.zeros(n)
        jI = np.zeros(n)
        jQ = np.zeros(n)
        i2 = np.zeros(n)
        q2 = np.zeros(n)
        re = np.zeros(n)
        im = np.zeros(n)
        for i in range(3, n):
            pp = period[i - 1]
            smooth[i] = (4 * src[i] + 3 * src[i - 1] + 2 * src[i - 2] + src[i - 3]) / 10.0

            def _quad(a, idx, p):
                if idx < 6:
                    return 0.0
                return (0.0962 * a[idx] + 0.5769 * a[idx - 2] - 0.5769 * a[idx - 4] - 0.0962 * a[idx - 6]) * (
                    0.075 * p + 0.54
                )

            detrend[i] = _quad(smooth, i, pp)
            q1[i] = _quad(detrend, i, pp)
            i1[i] = detrend[i - 3] if i >= 3 else 0.0
            jI[i] = _quad(i1, i, pp)
            jQ[i] = _quad(q1, i, pp)
            i2r = i1[i] - jQ[i]
            q2r = q1[i] + jI[i]
            i2[i] = 0.2 * i2r + 0.8 * i2[i - 1]
            q2[i] = 0.2 * q2r + 0.8 * q2[i - 1]
            re_r = i2[i] * i2[i - 1] + q2[i] * q2[i - 1]
            im_r = i2[i] * q2[i - 1] - q2[i] * i2[i - 1]
            re[i] = 0.2 * re_r + 0.8 * re[i - 1]
            im[i] = 0.2 * im_r + 0.8 * im[i - 1]
            if im[i] != 0.0 and re[i] != 0.0:
                p_raw = 360.0 / np.degrees(np.arctan(im[i] / re[i]))
            else:
                p_raw = pp
            p_lim = max(min(p_raw, 1.5 * pp), 0.67 * pp) if pp > 0 else p_raw
            p_cl = min(max(p_lim, 6.0), 50.0)
            period[i] = 0.2 * p_cl + 0.8 * pp
            sp[i] = 0.33 * period[i] + 0.67 * sp[i - 1]
        return np.clip(np.round(sp), min_period, max_period).astype(float)

    def true_range(high, low, close):
        n = len(high)
        if n < 2:
            return np.array([high[0] - low[0]])
        prev_close = close[:-1]
        high_low = high[1:] - low[1:]
        high_pc = np.abs(high[1:] - prev_close)
        low_pc = np.abs(low[1:] - prev_close)
        return np.concatenate([[high[0] - low[0]], np.maximum(np.maximum(high_low, high_pc), low_pc)])

    def rma(series, length):
        if len(series) == 0:
            return series
        result = np.zeros_like(series, dtype=float)
        result[0] = series[0]
        for i in range(1, min(length, len(series))):
            result[i] = series[: i + 1].mean()
        if len(series) > length:
            result[length - 1] = series[:length].mean()
            alpha = 1.0 / length
            for i in range(length, len(series)):
                result[i] = alpha * series[i] + (1 - alpha) * result[i - 1]
        return result

    def adaptive_wilder_atr(tr, period_arr, seed):
        n = len(tr)
        atr = np.empty(n, dtype=float)
        atr[0] = seed
        for i in range(1, n):
            per = max(2, int(round(period_arr[i])))
            atr[i] = atr[i - 1] + (1.0 / per) * (tr[i] - atr[i - 1])
        return atr

    return adaptive_wilder_atr, ehlers_dominant_cycle, rma, true_range


@app.cell
def _(adaptive_wilder_atr, ehlers_dominant_cycle, np, rma, true_range):
    def modern_adaptive_supertrend(
        high,
        low,
        close,
        atr_period=10,
        mult_fallback=3.0,
        use_l1=False,
        min_period=5,
        max_period=50,
        use_l2=False,
        ker_len=20,
        ker_pctile=True,
        ker_pct_win=500,
        pivot=0.5,
        trend_gain=0.8,
        chop_gain=0.5,
        mult_min=1.0,
        mult_max=6.0,
        use_l3=True,
        hyst_atr=0.5,
        hyst_bars=1,
    ):
        n = len(close)
        hl2 = (high + low) / 2.0

        adapt_period = ehlers_dominant_cycle(hl2, min_period, max_period)
        period_used = np.where(use_l1 & (np.arange(n) >= 6), adapt_period, atr_period).astype(float)

        tr = true_range(high, low, close)
        atr_fixed = rma(tr, atr_period)
        atr_adaptive = adaptive_wilder_atr(tr, period_used, atr_fixed[0])
        atr = atr_adaptive if use_l1 else atr_fixed

        ker = np.zeros(n)
        for i in range(ker_len, n):
            ker_dir = abs(close[i] - close[i - ker_len])
            ker_vol = np.sum(np.abs(close[i - ker_len + 1 : i + 1] - close[i - ker_len : i]))
            ker[i] = ker_dir / ker_vol if ker_vol > 0 else 0.0

        ker_rank = np.zeros(n)
        for i in range(ker_pct_win - 1, n):
            window = ker[i - ker_pct_win + 1 : i + 1]
            ker_rank[i] = np.sum(window <= ker[i]) / len(window)
        ker_sig = ker_rank if ker_pctile else ker

        f_trend = np.maximum(0.0, (ker_sig - pivot) / (1.0 - pivot))
        f_chop = np.maximum(0.0, (pivot - ker_sig) / pivot)
        mult_hinge = mult_fallback * (1.0 + trend_gain * f_trend + chop_gain * f_chop)
        mult_eff = np.clip(mult_hinge, mult_min, mult_max) if use_l2 else np.full(n, mult_fallback)

        upper_basic = hl2 + mult_eff * atr
        lower_basic = hl2 - mult_eff * atr

        upper_band = np.zeros(n)
        lower_band = np.zeros(n)
        direction = np.ones(n, dtype=int)
        supertrend = np.zeros(n)
        flip = np.zeros(n, dtype=bool)

        cand_dir = 0
        cand_count = 0

        for i in range(n):
            if i == 0:
                upper_band[0] = upper_basic[0]
                lower_band[0] = lower_basic[0]
                direction[0] = 1
                supertrend[0] = lower_band[0]
                continue

            ub_prev = upper_band[i - 1]
            lb_prev = lower_band[i - 1]

            upper_band[i] = (
                upper_basic[i] if (upper_basic[i] < ub_prev or close[i - 1] > ub_prev) else ub_prev
            )
            lower_band[i] = (
                lower_basic[i] if (lower_basic[i] > lb_prev or close[i - 1] < lb_prev) else lb_prev
            )

            prev_dir = direction[i - 1]
            new_dir = prev_dir

            if not use_l3:
                if close[i] > ub_prev:
                    new_dir = 1
                elif close[i] < lb_prev:
                    new_dir = -1
            else:
                buf = hyst_atr * atr[i]
                if prev_dir == 1:
                    if close[i] < lb_prev - buf:
                        cand_count = cand_count + 1 if cand_dir == -1 else 1
                        cand_dir = -1
                        if cand_count >= hyst_bars:
                            new_dir = -1
                            cand_dir = 0
                            cand_count = 0
                    else:
                        cand_dir = 0
                        cand_count = 0
                else:
                    if close[i] > ub_prev + buf:
                        cand_count = cand_count + 1 if cand_dir == 1 else 1
                        cand_dir = 1
                        if cand_count >= hyst_bars:
                            new_dir = 1
                            cand_dir = 0
                            cand_count = 0
                    else:
                        cand_dir = 0
                        cand_count = 0

            direction[i] = new_dir
            flip[i] = new_dir != prev_dir
            supertrend[i] = lower_band[i] if new_dir == 1 else upper_band[i]

        return {
            "supertrend": supertrend,
            "direction": direction,
            "flip": flip,
            "upper_band": upper_band,
            "lower_band": lower_band,
            "atr": atr,
            "ker": ker_sig,
            "mult_eff": mult_eff,
            "period_used": period_used.astype(int),
        }

    return (modern_adaptive_supertrend,)


@app.cell
def _(np, pd):
    def backtest_long_flat(close: np.ndarray, direction: np.ndarray, cost_bps: float = 0.0) -> pd.DataFrame:
        """Long when direction==1, flat otherwise. Position from prior bar; costs on turnover."""
        close = np.asarray(close, dtype=float)
        direction = np.asarray(direction, dtype=int)
        asset_ret = np.zeros(len(close))
        asset_ret[1:] = close[1:] / close[:-1] - 1.0

        position = np.zeros(len(close))
        position[1:] = np.where(direction[:-1] == 1, 1.0, 0.0)

        turnover = np.zeros(len(close))
        turnover[1:] = np.abs(position[1:] - position[:-1])

        cost = turnover * (cost_bps / 10_000.0)
        strategy_ret = position * asset_ret - cost

        out = pd.DataFrame(
            {
                "asset_return": asset_ret,
                "position": position,
                "turnover": turnover,
                "strategy_return": strategy_ret,
                "buy_hold_return": asset_ret,
            }
        )
        out["strategy_equity"] = (1.0 + out["strategy_return"]).cumprod()
        out["buy_hold_equity"] = (1.0 + out["buy_hold_return"]).cumprod()
        return out

    def summarize_returns(returns: pd.Series) -> dict:
        r = returns.dropna()
        if len(r) < 5:
            return {
                "days": len(r),
                "cagr": np.nan,
                "ann_vol": np.nan,
                "sharpe": np.nan,
                "max_drawdown": np.nan,
                "hit_rate": np.nan,
            }
        ann_return = float(r.mean() * 252)
        ann_vol = float(r.std(ddof=0) * np.sqrt(252))
        sharpe = float(ann_return / ann_vol) if ann_vol > 0 else 0.0
        equity = (1.0 + r).cumprod()
        running_max = equity.cummax()
        max_dd = float(((equity - running_max) / running_max).min())
        years = len(r) / 252.0
        cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 else np.nan
        hit_rate = float((r > 0).mean())
        return {
            "days": len(r),
            "cagr": cagr,
            "ann_vol": ann_vol,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "hit_rate": hit_rate,
        }

    return backtest_long_flat, summarize_returns


@app.cell
def _(
  DEFAULT_PARAMS,
  MIN_VAL_DAYS,
  OPTIMIZER_BUDGET,
  backtest_long_flat,
  modern_adaptive_supertrend,
  np,
  pd,
  pl,
  summarize_returns,
):
    def slice_btc(df: pl.DataFrame, start, end=None) -> pl.DataFrame:
        out = df.filter(pl.col("datetime").dt.date() >= start)
        if end is not None:
            out = out.filter(pl.col("datetime").dt.date() <= end)
        return out.sort("open_time")

    def run_indicator(df: pl.DataFrame, params: dict) -> dict:
        return modern_adaptive_supertrend(
            high=df["high"].to_numpy(),
            low=df["low"].to_numpy(),
            close=df["close"].to_numpy(),
            **params,
        )

    def _clamp_int(value: int, lo: int, hi: int) -> int:
        return int(max(lo, min(hi, value)))

    def _clamp_float(value: float, lo: float, hi: float) -> float:
        return float(max(lo, min(hi, value)))

    def _candidate_key(params: dict) -> tuple:
        return tuple(sorted((k, params[k]) for k in params))

    def build_candidate_grid(base: dict, budget: int) -> list[dict]:
        candidates: list[dict] = []
        seen: set[tuple] = set()

        def add(**overrides) -> None:
            cand = {**base, **overrides}
            cand["atr_period"] = _clamp_int(cand["atr_period"], 5, 21)
            cand["mult_fallback"] = round(_clamp_float(cand["mult_fallback"], 1.5, 5.0), 2)
            cand["min_period"] = _clamp_int(cand["min_period"], 4, 12)
            cand["max_period"] = _clamp_int(cand["max_period"], 30, 60)
            cand["ker_len"] = _clamp_int(cand["ker_len"], 10, 40)
            cand["ker_pct_win"] = _clamp_int(cand["ker_pct_win"], 126, 756)
            cand["pivot"] = round(_clamp_float(cand["pivot"], 0.3, 0.7), 2)
            cand["trend_gain"] = round(_clamp_float(cand["trend_gain"], 0.2, 1.2), 2)
            cand["chop_gain"] = round(_clamp_float(cand["chop_gain"], 0.2, 1.0), 2)
            cand["mult_min"] = round(_clamp_float(cand["mult_min"], 0.8, 2.0), 2)
            cand["mult_max"] = round(_clamp_float(cand["mult_max"], 3.0, 8.0), 2)
            cand["hyst_atr"] = round(_clamp_float(cand["hyst_atr"], 0.1, 1.5), 2)
            cand["hyst_bars"] = _clamp_int(cand["hyst_bars"], 1, 3)
            key = _candidate_key(cand)
            if key not in seen:
                seen.add(key)
                candidates.append(cand)

        layer_seeds = [
            {"use_l1": False, "use_l2": False, "use_l3": True},
            {"use_l1": True, "use_l2": False, "use_l3": True},
            {"use_l1": False, "use_l2": True, "use_l3": True},
            {"use_l1": True, "use_l2": True, "use_l3": True},
        ]
        for layers in layer_seeds:
            add(**layers)
            for mult in [2.5, 3.0, 3.5]:
                for hyst in [0.35, 0.5, 0.75]:
                    add(**layers, mult_fallback=mult, hyst_atr=hyst)
            for ker_len in [14, 20, 30]:
                for ker_win in [252, 500]:
                    add(**layers, ker_len=ker_len, ker_pct_win=ker_win, pivot=0.5)
            for pivot in [0.4, 0.5, 0.6]:
                add(**layers, pivot=pivot, trend_gain=0.8, chop_gain=0.5)
            for atr_p in [7, 10, 14]:
                add(**layers, atr_period=atr_p)
            for hyst_bars in [1, 2]:
                add(**layers, hyst_bars=hyst_bars, hyst_atr=0.5)

        return candidates[:budget]

    def score_on_window(
        full_df: pl.DataFrame,
        val_df: pl.DataFrame,
        params: dict,
        cost_bps: float,
    ) -> dict | None:
        if val_df.shape[0] < MIN_VAL_DAYS:
            return None
        ind = run_indicator(full_df, params)
        bt = backtest_long_flat(full_df["close"].to_numpy(), ind["direction"], cost_bps=cost_bps)
        bt = bt.copy()
        bt["date"] = full_df["datetime"].to_pandas()

        val_dates = val_df["datetime"].to_pandas()
        mask = bt["date"].isin(val_dates)
        val_bt = bt.loc[mask]
        if len(val_bt) < MIN_VAL_DAYS:
            return None

        strat = summarize_returns(val_bt["strategy_return"])
        bh = summarize_returns(val_bt["buy_hold_return"])
        if not np.isfinite(strat["sharpe"]):
            return None

        flips = int(ind["flip"][mask.to_numpy()].sum()) if mask.any() else 0
        score = strat["sharpe"] - 0.25 * max(0.0, bh["sharpe"] - strat["sharpe"])
        score -= 0.02 * max(0, flips - 30)

        return {
            "score": float(score),
            "val_sharpe": strat["sharpe"],
            "val_cagr": strat["cagr"],
            "val_max_drawdown": strat["max_drawdown"],
            "bh_val_sharpe": bh["sharpe"],
            "flips": flips,
            "params": params,
        }

    def optimize_supertrend(
        train_df: pl.DataFrame,
        val_df: pl.DataFrame,
        full_df: pl.DataFrame,
        cost_bps: float,
        budget: int,
        base_params: dict,
    ) -> tuple[dict, pd.DataFrame]:
        candidates = build_candidate_grid(base_params, budget)
        rows = []
        for params in candidates:
            scored = score_on_window(full_df, val_df, params, cost_bps)
            if scored is None:
                continue
            row = {**{f"param_{k}": v for k, v in params.items()}, **scored}
            rows.append(row)

        if not rows:
            raise RuntimeError("Optimizer found no valid candidates on the validation window.")

        leaderboard = pd.DataFrame(rows).sort_values(
            ["score", "val_sharpe", "val_max_drawdown"],
            ascending=[False, False, False],
        )
        best = leaderboard.iloc[0]["params"]
        if isinstance(best, str):
            import ast

            best = ast.literal_eval(best)
        return best, leaderboard

    return build_candidate_grid, optimize_supertrend, run_indicator, slice_btc


@app.cell
def _(pl):
    import yfinance as yf

    _yf_raw = yf.download("BTC-USD", period="max", interval="1d", progress=False)
    if hasattr(_yf_raw.columns, "levels") and len(_yf_raw.columns.levels) > 1:
        _yf_raw.columns = _yf_raw.columns.get_level_values(0)
    btc_long = (
        pl.DataFrame(
            {
                "datetime": _yf_raw.index.tolist(),
                "open": _yf_raw["Open"].values.astype(float),
                "high": _yf_raw["High"].values.astype(float),
                "low": _yf_raw["Low"].values.astype(float),
                "close": _yf_raw["Close"].values.astype(float),
                "volume": _yf_raw["Volume"].values.astype(float),
            }
        )
        .with_columns(
            (pl.col("datetime").cast(pl.Int64) / 1_000_000).alias("open_time"),
            ((pl.col("datetime").cast(pl.Int64) / 1_000_000) + 86_400_000).alias("close_time"),
            pl.col("datetime").cast(pl.Datetime),
        )
        .sort("open_time")
    )
    _fetched_at = pl.datetime_now().cast(pl.Utf8)
    print(f"Fetched {btc_long.shape[0]} daily candles from Yahoo Finance at {_fetched_at}")
    print(f"Date range: {btc_long['datetime'][0]} to {btc_long['datetime'][-1]}")
    return (btc_long,)


@app.cell
def _(
    DEFAULT_PARAMS,
    OPTIMIZER_BUDGET,
    TRADING_COST_BPS,
    TRAIN_END,
    TRAIN_START,
    VAL_END,
    VAL_START,
    btc_long,
    optimize_supertrend,
    slice_btc,
):
    train_df = slice_btc(btc_long, TRAIN_START, TRAIN_END)
    val_df = slice_btc(btc_long, VAL_START, VAL_END)
    opt_df = slice_btc(btc_long, TRAIN_START, VAL_END)

    print(
        f"Train {train_df.shape[0]} rows | Val {val_df.shape[0]} rows | "
        f"Optimizer budget {OPTIMIZER_BUDGET} candidates"
    )

    best_params, optimizer_leaderboard = optimize_supertrend(
        train_df=train_df,
        val_df=val_df,
        full_df=opt_df,
        cost_bps=TRADING_COST_BPS,
        budget=OPTIMIZER_BUDGET,
        base_params=DEFAULT_PARAMS,
    )
    print("Self-optimized parameters (frozen before OOS):")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    return best_params, optimizer_leaderboard, opt_df, train_df, val_df


@app.cell
def _(mo, optimizer_leaderboard):
    _top = optimizer_leaderboard.head(12).copy()
    for col in list(_top.columns):
        if col == "params":
            _top[col] = _top[col].astype(str)
    mo.vstack([
        mo.md("### Optimizer leaderboard (validation window, top 12)"),
        mo.ui.table(_top.round(4)),
    ])
    return


@app.cell
def _(
    CLASSIC_PARAMS,
    OOS_SHARPE_MARGIN,
    OOS_START,
    TRADING_COST_BPS,
    VAL_END,
    VAL_START,
    backtest_long_flat,
    best_params,
    btc_long,
    mo,
    pd,
    pl,
    run_indicator,
    summarize_returns,
):
    optimized_ind = run_indicator(btc_long, best_params)
    classic_ind = run_indicator(btc_long, CLASSIC_PARAMS)

    opt_bt = backtest_long_flat(
        btc_long["close"].to_numpy(), optimized_ind["direction"], cost_bps=TRADING_COST_BPS
    )
    classic_bt = backtest_long_flat(
        btc_long["close"].to_numpy(), classic_ind["direction"], cost_bps=TRADING_COST_BPS
    )

    opt_bt["date"] = btc_long["datetime"].to_pandas()
    classic_bt["date"] = btc_long["datetime"].to_pandas()

    oos_mask = opt_bt["date"].dt.date >= OOS_START
    oos_opt = opt_bt.loc[oos_mask]
    oos_classic = classic_bt.loc[oos_mask]
    oos_bh = oos_opt["buy_hold_return"]

    oos_summary = pd.DataFrame(
        [
            {"strategy": "Self-optimized adaptive", **summarize_returns(oos_opt["strategy_return"])},
            {"strategy": "Classic supertrend", **summarize_returns(oos_classic["strategy_return"])},
            {"strategy": "Buy & hold BTC", **summarize_returns(oos_bh)},
        ]
    )

    opt_oos_sharpe = float(oos_summary.loc[oos_summary["strategy"] == "Self-optimized adaptive", "sharpe"].iloc[0])
    bh_oos_sharpe = float(oos_summary.loc[oos_summary["strategy"] == "Buy & hold BTC", "sharpe"].iloc[0])
    passed_gate = bool(opt_oos_sharpe >= bh_oos_sharpe + OOS_SHARPE_MARGIN)

    st_df = btc_long.with_columns(
        [
            pl.Series("supertrend", optimized_ind["supertrend"]),
            pl.Series("direction", optimized_ind["direction"]),
            pl.Series("flip", optimized_ind["flip"].astype(int)),
            pl.Series("ker", optimized_ind["ker"]),
            pl.Series("mult_eff", optimized_ind["mult_eff"]),
            pl.Series("period_used", optimized_ind["period_used"]),
        ]
    )

    verdict_md = (
        f"**OOS gate {'PASSED' if passed_gate else 'FAILED'}** — "
        f"optimized Sharpe {opt_oos_sharpe:.2f} vs buy-and-hold {bh_oos_sharpe:.2f} "
        f"(margin {OOS_SHARPE_MARGIN:.2f}) on {OOS_START}+ after {TRADING_COST_BPS:.0f} bps costs. "
        f"Validation window for selection: {VAL_START}–{VAL_END}."
    )

    mo.vstack([
        mo.md(f"### Out-of-sample scoreboard ({OOS_START}+)"),
        mo.ui.table(oos_summary.round(4)),
        mo.md(verdict_md),
    ])
    return classic_bt, opt_bt, passed_gate, st_df, verdict_md


@app.cell
def _(btc_long, classic_bt, opt_bt, pl, st_df):
    import altair as alt

    chart_df = st_df.with_columns(
        [
            pl.when(pl.col("direction") == 1)
            .then(pl.lit("Uptrend"))
            .otherwise(pl.lit("Downtrend"))
            .alias("trend"),
            (pl.col("direction") != pl.col("direction").shift(1))
            .fill_null(True)
            .cum_sum()
            .alias("segment"),
        ]
    )

    base = alt.Chart(chart_df).encode(x="datetime:T")
    price = base.mark_line(color="gray", strokeWidth=1).encode(y="close:Q")
    st_line = base.mark_line(strokeWidth=2.5).encode(
        y="supertrend:Q",
        color=alt.Color(
            "trend:N",
            scale=alt.Scale(domain=["Uptrend", "Downtrend"], range=["teal", "red"]),
        ),
        detail="segment:N",
    )
    fill = base.mark_area(opacity=0.06, color="gray").encode(y="close:Q", y2="supertrend:Q")
    flip_df = chart_df.filter(pl.col("flip") == 1)
    flip_up = alt.Chart(flip_df.filter(pl.col("direction") == 1)).mark_point(
        shape="triangle-up", color="teal", size=60, filled=True
    ).encode(x="datetime:T", y="supertrend:Q")
    flip_dn = alt.Chart(flip_df.filter(pl.col("direction") == -1)).mark_point(
        shape="triangle-down", color="red", size=60, filled=True
    ).encode(x="datetime:T", y="supertrend:Q")

    price_chart = (price + fill + st_line + flip_up + flip_dn).properties(
        title=f"Self-optimized Adaptive Supertrend — BTC/USD ({btc_long.shape[0]} daily candles)",
        width=950,
        height=420,
    ).interactive()

    diag = chart_df.select(["datetime", "ker", "mult_eff", "period_used"]).to_pandas()
    ker_chart = alt.Chart(diag).mark_line(color="#6366f1").encode(
        x="datetime:T", y=alt.Y("ker:Q", title="KER percentile")
    ).properties(title="KER signal", width=950, height=140)
    mult_chart = alt.Chart(diag).mark_line(color="#f59e0b").encode(
        x="datetime:T", y=alt.Y("mult_eff:Q", title="Effective multiplier")
    ).properties(title="Adaptive multiplier", width=950, height=140)

    price_chart & (ker_chart & mult_chart)

    eq = opt_bt[["date", "strategy_equity", "buy_hold_equity"]].copy()
    eq = eq.rename(columns={"strategy_equity": "Self-optimized", "buy_hold_equity": "Buy & hold"})
    eq["Classic supertrend"] = classic_bt["strategy_equity"].values
    eq_long = eq.melt(id_vars="date", var_name="series", value_name="equity")
    eq_chart = alt.Chart(eq_long).mark_line().encode(
        x="date:T",
        y=alt.Y("equity:Q", scale=alt.Scale(type="log")),
        color="series:N",
    ).properties(title="Equity curves (log scale, full history)", width=950, height=360).interactive()

    eq_chart
    return


@app.cell
def _(mo, pl, st_df):
    flip_rows = (
        st_df.with_row_index("idx")
        .filter(pl.col("flip") == 1)
        .select(["datetime", "direction", "close", "supertrend", "ker", "mult_eff", "idx"])
        .tail(25)
    )
    mo.vstack([
        mo.md("### Recent trend flips (optimized parameters)"),
        mo.ui.table(flip_rows),
    ])
    return


if __name__ == "__main__":
    app.run()
