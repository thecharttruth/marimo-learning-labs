# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "altair==6.1.0",
#     "anywidget==0.11.0",
#     "autogluon.tabular[all]==1.5.0",
#     "numpy==2.3.5",
#     "pandas==2.3.3",
#     "polars==1.41.2",
#     "scipy==1.16.3",
#     "traitlets==5.15.1",
#     "yfinance==1.4.1",
# ]
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(
    width="medium",
    auto_download=["html"],
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # AutoGluon Portfolio Allocation Learning Lab

    This notebook teaches a complete machine-learning trading workflow for ETF portfolios.

    The big question is simple: **can AutoGluon beat a simple SPY benchmark well enough to justify the extra work?**

    You will learn four linked ideas:

    1. **Machine learning framing**: AutoGluon predicts each ETF's future return from past-only clues using one shared cross-asset model.
    2. **AutoGluon workflow**: `TabularPredictor` tests model types for us, while we choose the data, target, time budget, and safety rules.
    3. **Trading discipline**: walk-forward validation trains only on the past, then tests on the future.
    4. **Benchmark discipline**: predictions only matter if the ML strategy beats simple choices like buying SPY.

    This is a research and education notebook, not a live trading system. Treat every result as a hypothesis. SPY is the default benchmark. The ML overlay only earns a role if it improves SPY after the notebook's deployment gate.

    **Prerequisite:** run the [Global Macro Opportunity Set Lab](../global-macro-opportunity-set-lab/notebook.py) first and promote only universes that pass its material benchmark gate.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Plain-English Map

    Think of the notebook as a science experiment:

    1. Download ETF price history.
    2. Turn prices into clues, called **features**.
    3. Ask AutoGluon to learn patterns from old data.
    4. Make predictions only for dates that come after the training data.
    5. Convert those predictions into portfolio weights.
    6. Compare the result with simple benchmarks like SPY and 60/40 SPY/IEF.

    Helpful words:

    - **Ticker**: the short market symbol for an ETF, like `SPY`.
    - **Portfolio**: the basket of ETFs we hold.
    - **Weight**: the percent of the portfolio put into one ETF.
    - **Feature**: a clue the model can use, such as recent return or volatility.
    - **Target**: the answer the model is trying to predict, such as next-month return.
    - **Backtest**: a historical replay that asks, "what would have happened?"
    - **Lookahead bias**: accidentally using future information that would not have been known at the time.
    """)
    return


@app.cell(hide_code=True)
def _():
    import warnings
    warnings.filterwarnings("ignore")

    from datetime import date, datetime, timedelta
    from itertools import combinations
    import tempfile
    from zoneinfo import ZoneInfo

    import altair as alt
    import anywidget
    import traitlets
    import marimo as mo
    import numpy as np
    import pandas as pd
    import polars as pl
    import inspect as _inspect
    try:
        import torch
    except ImportError:
        torch = None
    import yfinance as yf
    from autogluon.tabular import TabularPredictor
    from scipy.optimize import minimize

    # Material benchmark margins (aligned with the Global Macro opportunity-set gate).
    GATE_MIN_SHARPE_EDGE = 0.10
    GATE_MIN_CAGR_EDGE = 0.01
    GATE_MIN_DRAWDOWN_EDGE = 0.02
    GATE_MAX_SHARPE_GAP_VS_SIMPLE = 0.05
    GATE_MIN_HIT_RATE_VS_SPY = 0.50
    OVERLAY_MIN_RISK_OFF_SCORE = 0.15

    RUN_PRESETS = {
        "Custom": {},
        "Smoke test": {
            "evaluation_protocol": "Final OOS: selected start onward",
            "max_splits": 6,
            "feature_depth": 1,
            "model_time_limit": 15,
            "include_ml_research": False,
            "run_parameter_optimizer": False,
        },
        "Final OOS research": {
            "evaluation_protocol": "Final OOS: selected start onward",
            "max_splits": 40,
            "feature_depth": 2,
            "model_time_limit": 15,
            "include_ml_research": False,
            "run_parameter_optimizer": False,
        },
        "Full history debug": {
            "evaluation_protocol": "All walk-forward splits",
            "max_splits": 100,
            "feature_depth": 2,
            "model_time_limit": 15,
            "include_ml_research": True,
            "run_parameter_optimizer": False,
        },
    }

    # AutoGluon 1.5 still calls pandas fillna(..., downcast=...).
    # Newer pandas versions removed that keyword, so keep MoLab runtimes compatible.
    _pd_ndframe_fillna = pd.core.generic.NDFrame.fillna

    def _compat_pandas_fillna(self, *args, downcast=None, _orig=_pd_ndframe_fillna, **kwargs):
        kwargs.pop("downcast", None)
        return _orig(self, *args, **kwargs)

    if "downcast" not in _inspect.signature(_pd_ndframe_fillna).parameters:
        pd.DataFrame.fillna = _compat_pandas_fillna
        pd.Series.fillna = _compat_pandas_fillna

    RANDOM_SEED = 42
    np.random.seed(RANDOM_SEED)
    return (
        GATE_MAX_SHARPE_GAP_VS_SIMPLE,
        GATE_MIN_CAGR_EDGE,
        GATE_MIN_DRAWDOWN_EDGE,
        GATE_MIN_HIT_RATE_VS_SPY,
        GATE_MIN_SHARPE_EDGE,
        OVERLAY_MIN_RISK_OFF_SCORE,
        RANDOM_SEED,
        RUN_PRESETS,
        TabularPredictor,
        ZoneInfo,
        alt,
        anywidget,
        combinations,
        date,
        datetime,
        minimize,
        mo,
        np,
        pd,
        pl,
        tempfile,
        timedelta,
        torch,
        traitlets,
        yf,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Configuration & Controls

    Start here. These controls decide the experiment before any models run.

    - **Tickers** choose the ETFs the portfolio is allowed to hold.
    - **Start / end date** choose the history the notebook downloads.
    - **Evaluation protocol** chooses the report window: practice data, final out-of-sample data, or all walk-forward splits for debugging.
    - **Research / validation / final OOS dates** split time into three jobs. Research is for building ideas. Validation is for choosing settings. Final OOS is the honest test after settings are frozen. The final OOS report end lets you score a fixed window, such as exactly 2024-2025, while still downloading later data to measure the last forward returns.
    - **Forward return horizon** is what AutoGluon predicts. A 21-day horizon means, "what might this ETF return over about one month?"
    - **Training window** controls how much old data AutoGluon can study at each step.
    - **Rebalance frequency** controls how often the notebook retrains models and updates weights.
    - **Embargo** adds a waiting gap after training before predictions begin. It helps keep training data and test data separated.
    - **Max rebalance splits** controls how many walk-forward tests run. Start small, then raise it.
    - **Run preset** applies a safe starting configuration. Choose **Custom** to use every manual control directly.
    - **Include ML-MC / ML research strategies** keeps the older full-tilt research comparisons in the backtest. Leave it off for the default deployable overlay workflow.
    - **AutoGluon time limit** is the model-search budget for each model fit.
    - **Max single-asset weight** prevents the portfolio from putting too much money into one ETF.
    - **Feature set depth** controls how many clues are built for the model.

    The optional parameter optimizer tries a few settings on the validation window only. It does not get to see the final OOS window before making its choice.

    Learning move: run a small final-OOS test first. Inspect the tables and charts. Then raise `max_splits` after the workflow makes sense.
    """)
    return


@app.cell
def _(RUN_PRESETS, ZoneInfo, date, datetime, mo):
    default_tickers = "SPY,QQQ,IWM,TLT,GLD,VGT,XLF,XLE,VEA,VWO"

    # Controls
    run_preset = mo.ui.dropdown(
        options=list(RUN_PRESETS.keys()),
        value="Smoke test",
        label="Run preset",
    )
    ticker_input = mo.ui.text(value=default_tickers, label="Tickers (comma-separated)")
    start_date = mo.ui.date(value=date(2018, 1, 1), label="Start date")
    _default_end_date = datetime.now(ZoneInfo("America/New_York")).date()
    end_date = mo.ui.date(value=_default_end_date, label="End date")
    evaluation_protocol = mo.ui.dropdown(
        options=[
            "Final OOS: selected start onward",
            "Validation window only",
            "All walk-forward splits",
        ],
        value="Final OOS: selected start onward",
        label="Evaluation protocol",
    )
    research_end_date = mo.ui.date(value=date(2024, 12, 31), label="Research/tuning cutoff")
    validation_end_date = mo.ui.date(value=date(2025, 12, 31), label="Validation cutoff")
    final_oos_start_date = mo.ui.date(value=date(2026, 1, 1), label="Final OOS start")
    final_oos_end_date = mo.ui.date(value=_default_end_date, label="Final OOS/report end")
    rebalance_days = mo.ui.slider(5, 63, value=21, step=1, label="Rebalance frequency (days)")
    train_years = mo.ui.slider(1, 5, value=3, step=1, label="Training window (years)")
    forward_days = mo.ui.slider(5, 63, value=21, step=1, label="Forward return horizon (days)")
    embargo_days = mo.ui.slider(0, 21, value=1, step=1, label="Embargo after training window (days)")
    mc_draws = mo.ui.slider(50, 1000, value=200, step=50, label="Monte-Carlo draws")
    model_time_limit = mo.ui.slider(5, 120, value=15, step=5, label="AutoGluon time limit per fit (sec)")
    max_splits = mo.ui.slider(1, 100, value=6, step=1, label="Max rebalance splits to run")
    model_mode = mo.ui.dropdown(
        options=[
            "Auto GPU rich if available",
            "Fast CPU trees",
            "GPU rich models",
        ],
        value="Auto GPU rich if available",
        label="AutoGluon model mode",
    )
    allow_short = mo.ui.checkbox(value=False, label="Allow short positions")
    max_wt = mo.ui.slider(0.1, 1.0, value=0.35, step=0.05, label="Max single-asset weight")
    trading_cost_bps = mo.ui.slider(0.0, 10.0, value=0.0, step=0.5, label="Trading cost (bps per $ traded)")
    feature_depth = mo.ui.slider(1, 3, value=2, step=1, label="Feature set depth: 1 core, 2 expanded, 3 research")
    include_pair_factors = mo.ui.checkbox(value=False, label="Include cross-asset pair factors")
    pair_factor_limit = mo.ui.slider(5, 60, value=20, step=5, label="Max ticker pairs for pair factors")
    run_parameter_optimizer = mo.ui.checkbox(value=False, label="Run bounded parameter optimizer")
    apply_optimizer_settings = mo.ui.checkbox(value=False, label="Apply optimizer suggestion")
    include_ml_research = mo.ui.checkbox(value=False, label="Include ML-MC / ML research strategies")
    optimizer_candidate_budget = mo.ui.slider(1, 8, value=3, step=1, label="Optimizer candidate budget")
    optimizer_validation_splits = mo.ui.slider(1, 6, value=2, step=1, label="Optimizer validation split cap")
    optimizer_time_limit = mo.ui.slider(1, 15, value=3, step=1, label="Optimizer time limit per fit (sec)")

    mo.vstack([
        mo.hstack([run_preset]),
        mo.hstack([ticker_input]),
        mo.hstack([start_date, end_date]),
        mo.hstack([evaluation_protocol]),
        mo.hstack([research_end_date, validation_end_date]),
        mo.hstack([final_oos_start_date, final_oos_end_date]),
        mo.hstack([rebalance_days, forward_days]),
        mo.hstack([train_years, embargo_days]),
        mo.hstack([mc_draws]),
        mo.hstack([model_time_limit, max_splits]),
        mo.hstack([model_mode]),
        mo.hstack([allow_short, max_wt]),
        mo.hstack([trading_cost_bps]),
        mo.md("**Feature set**"),
        mo.hstack([feature_depth, include_pair_factors]),
        mo.hstack([pair_factor_limit]),
        mo.md("**Optional parameter optimizer**"),
        mo.hstack([run_parameter_optimizer, apply_optimizer_settings, include_ml_research]),
        mo.hstack([optimizer_candidate_budget, optimizer_validation_splits]),
        mo.hstack([optimizer_time_limit]),
    ])
    return (
        allow_short,
        apply_optimizer_settings,
        embargo_days,
        end_date,
        evaluation_protocol,
        feature_depth,
        final_oos_end_date,
        final_oos_start_date,
        forward_days,
        include_ml_research,
        include_pair_factors,
        max_splits,
        max_wt,
        mc_draws,
        model_mode,
        model_time_limit,
        optimizer_candidate_budget,
        optimizer_time_limit,
        optimizer_validation_splits,
        pair_factor_limit,
        rebalance_days,
        research_end_date,
        run_parameter_optimizer,
        run_preset,
        start_date,
        ticker_input,
        trading_cost_bps,
        train_years,
        validation_end_date,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. Market Data

    The notebook downloads ETF prices with `yfinance`, then keeps only tickers with enough history.

    It uses adjusted prices. Adjusted prices try to account for dividends and splits, so they are a better fit for portfolio research than raw prices.

    Trading caveats:

    - `yfinance` is good for learning, but it is not professional trading data.
    - ETF histories can have missing days, holiday gaps, and later data fixes.
    - This notebook does **not** backward-fill prices before an ETF's real inception date.
    - A real trading system should use a stable data vendor and save exact data snapshots.
    """)
    return


@app.cell
def _(datetime, pd, pl, yf):
    def download_prices(
        tickers: list[str],
        start: datetime,
        end: datetime,
        min_obs: int = 20,
    ) -> tuple[pl.DataFrame | None, list[str]]:
        """Download inception-safe adjusted close prices from yfinance."""
        if not tickers:
            return None, []

        price_parts: list[pd.Series] = []
        valid_tickers: list[str] = []
        try:
            raw = yf.download(
                tickers,
                start=start,
                end=end,
                progress=False,
                auto_adjust=True,
                group_by="column",
            )
            if not raw.empty and isinstance(raw.columns, pd.MultiIndex):
                if "Close" in raw.columns.get_level_values(0):
                    close_panel = raw["Close"]
                elif "Adj Close" in raw.columns.get_level_values(0):
                    close_panel = raw["Adj Close"]
                else:
                    close_panel = None
                if close_panel is not None:
                    for ticker in tickers:
                        if ticker not in close_panel.columns:
                            print(f"Skipping {ticker}: no data")
                            continue
                        series = close_panel[ticker].dropna()
                        if len(series) < min_obs:
                            print(f"Skipping {ticker}: insufficient history ({len(series)} rows)")
                            continue
                        price_parts.append(series.rename(ticker))
                        valid_tickers.append(ticker)
        except Exception as exc:
            print(f"Batch download failed ({exc}); falling back to per-ticker download.")

        if not price_parts:
            for ticker in tickers:
                try:
                    raw = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
                    if raw.empty:
                        print(f"Skipping {ticker}: no data")
                        continue
                    if isinstance(raw.columns, pd.MultiIndex):
                        raw.columns = raw.columns.get_level_values(0)
                    col = raw["Close"] if "Close" in raw.columns else raw["Adj Close"]
                    series = col.dropna()
                    if len(series) < min_obs:
                        print(f"Skipping {ticker}: insufficient history ({len(series)} rows)")
                        continue
                    price_parts.append(series.rename(ticker))
                    valid_tickers.append(ticker)
                except Exception as exc:
                    print(f"Skipping {ticker}: {exc}")

        if not price_parts:
            return None, valid_tickers

        wide = pd.concat(price_parts, axis=1).sort_index()
        joined = pl.DataFrame({
            "date": [d.date() for d in pd.to_datetime(wide.index)],
            **{ticker: wide[ticker].to_numpy(dtype=float) for ticker in wide.columns},
        }).with_columns(pl.col("date").cast(pl.Date))
        return joined.sort("date"), list(wide.columns)


    def inception_safe_fill(prices: pl.DataFrame, tickers: list[str]) -> pl.DataFrame:
        """Forward-fill gaps after inception only. Leading nulls stay null."""
        return prices.with_columns([pl.col(ticker).forward_fill().alias(ticker) for ticker in tickers])


    def build_inception_table(prices: pl.DataFrame, tickers: list[str]) -> pl.DataFrame:
        rows = []
        for ticker in tickers:
            series = prices.select(["date", ticker]).drop_nulls(subset=[ticker])
            if len(series) == 0:
                rows.append({
                    "Ticker": ticker,
                    "First Valid Date": None,
                    "Last Valid Date": None,
                    "Observations": 0,
                })
                continue
            rows.append({
                "Ticker": ticker,
                "First Valid Date": series["date"].min(),
                "Last Valid Date": series["date"].max(),
                "Observations": int(len(series)),
            })
        return pl.DataFrame(rows).sort(["First Valid Date", "Ticker"])

    def prices_to_returns(prices: pl.DataFrame, tickers: list[str], horizon: int) -> pl.DataFrame:
        """Convert prices to returns and next-bar-executable forward log returns."""
        log_prices = prices.with_columns([pl.col(t).log() for t in tickers])
        execution_lag = 1
        returns = log_prices.with_columns([
            (
                pl.col(t).shift(-(horizon + execution_lag))
                - pl.col(t).shift(-execution_lag)
            ).alias(f"fwd_return_{t}")
            for t in tickers
        ]).with_columns([
            (pl.col(t) - pl.col(t).shift(1)).alias(f"ret_{t}") for t in tickers
        ])
        return returns

    return build_inception_table, download_prices, inception_safe_fill, prices_to_returns


@app.cell
def _(build_inception_table, download_prices, end_date, inception_safe_fill, mo, pl, start_date, ticker_input):
    tickers = list(dict.fromkeys(t.strip().upper() for t in ticker_input.value.split(",") if t.strip()))
    _requested_start = start_date.value
    _requested_end = end_date.value
    _cache_tickers = "-".join(tickers)
    _cache_key = f"market_data_v2_{_cache_tickers}_{_requested_start}_{_requested_end}"

    # Cache yfinance downloads by requested universe and date range. The explicit
    # filter below is a second guard so stale broad caches cannot leak past controls.
    with mo.persistent_cache(name=_cache_key):
        prices, valid_tickers = download_prices(tickers, _requested_start, _requested_end)
        benchmark_prices, benchmark_valid_tickers = download_prices(["SPY", "IEF"], _requested_start, _requested_end)

    if prices is not None:
        prices = prices.filter(
            (pl.col("date") >= _requested_start) & (pl.col("date") <= _requested_end)
        )
        prices = inception_safe_fill(prices, valid_tickers)
    if benchmark_prices is not None:
        benchmark_prices = benchmark_prices.filter(
            (pl.col("date") >= _requested_start) & (pl.col("date") <= _requested_end)
        )
        benchmark_prices = inception_safe_fill(benchmark_prices, benchmark_valid_tickers)

    if prices is None or len(valid_tickers) < 2:
        raise ValueError(
            f"Need at least two tickers with sufficient history. "
            f"Requested: {', '.join(tickers)}; valid: {', '.join(valid_tickers)}."
        )

    inception_df = build_inception_table(prices, valid_tickers)

    if benchmark_prices is None:
        benchmark_prices = pl.DataFrame({"date": prices["date"]})
        benchmark_valid_tickers = []
    else:
        benchmark_prices = inception_safe_fill(benchmark_prices, benchmark_valid_tickers)

    _benchmark_note = (
        f"Standalone benchmarks loaded: {', '.join(benchmark_valid_tickers)}."
        if benchmark_valid_tickers
        else "Standalone benchmark data was not available from yfinance for this run."
    )

    mo.vstack([
        mo.md(f"""
        Loaded **{len(valid_tickers)}** tradable assets from `{prices['date'].min()}` to `{prices['date'].max()}` (`{len(prices)}` rows).

        Valid tradable tickers: {', '.join(valid_tickers)}.

        {_benchmark_note}

        The inception table below is a leak check. Newer ETFs should have newer first valid dates.
        """),
        mo.ui.table(
            inception_df.to_pandas(),
            pagination=False,
            selection=None,
            show_column_summaries=False,
            show_data_types=False,
            show_download=False,
            max_height=300,
        ),
    ])
    return benchmark_prices, inception_df, prices, valid_tickers


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Optional Parameter Optimizer

    The optimizer is intentionally small.

    It tries a short list of possible settings, scores them only on the validation window, and then freezes the winner before the final OOS report.

    The score is SPY-aware. That matters because a machine-learning strategy is not useful just because it is complicated. It has to beat the simple benchmark that an investor could buy instead.

    It suggests settings. It does not prove the strategy works. The final OOS gate still decides whether the SPY+ML overlay earned the extra complexity.
    """)
    return


@app.cell
def _(
    OVERLAY_MIN_RISK_OFF_SCORE,
    allow_short,
    backtest_allocations,
    build_features,
    build_walk_forward_splits,
    embargo_days,
    feature_depth,
    forward_days,
    include_pair_factors,
    max_wt,
    mc_draws,
    mo,
    np,
    optimizer_candidate_budget,
    optimizer_time_limit,
    optimizer_validation_splits,
    pair_factor_limit,
    pd,
    pl,
    prices,
    rebalance_days,
    research_end_date,
    run_parameter_optimizer,
    trading_cost_bps,
    train_and_allocate_split,
    train_years,
    valid_tickers,
    validation_end_date,
    model_mode,
):
    manual_settings = {
        "forward_days": int(forward_days.value),
        "rebalance_days": int(rebalance_days.value),
        "train_years": int(train_years.value),
        "max_wt": float(max_wt.value),
        "mc_draws": int(mc_draws.value),
    }


    def _clamp_int(value: int, lower: int, upper: int) -> int:
        return int(max(lower, min(upper, value)))


    def _clamp_float(value: float, lower: float, upper: float) -> float:
        return float(max(lower, min(upper, value)))


    def _add_candidate(candidates: list[dict], seen: set[tuple], base: dict, **overrides) -> None:
        candidate = {**base, **overrides}
        candidate["forward_days"] = _clamp_int(candidate["forward_days"], 5, 63)
        candidate["rebalance_days"] = _clamp_int(candidate["rebalance_days"], 5, 63)
        candidate["train_years"] = _clamp_int(candidate["train_years"], 1, 5)
        candidate["max_wt"] = round(_clamp_float(candidate["max_wt"], 0.10, 1.00), 2)
        candidate["mc_draws"] = _clamp_int(candidate["mc_draws"], 50, 1000)
        key = (
            candidate["forward_days"],
            candidate["rebalance_days"],
            candidate["train_years"],
            candidate["max_wt"],
            candidate["mc_draws"],
        )
        if key not in seen:
            seen.add(key)
            candidates.append(candidate)


    def _candidate_grid(base: dict, budget: int) -> list[dict]:
        candidates: list[dict] = []
        seen: set[tuple] = set()
        _add_candidate(candidates, seen, base)
        for horizon, rebalance in [(10, 10), (21, 21), (42, 21), (63, 21), (5, 5)]:
            _add_candidate(candidates, seen, base, forward_days=horizon, rebalance_days=rebalance)
        for train_window in [base["train_years"] - 1, base["train_years"] + 1, 2, 4]:
            _add_candidate(candidates, seen, base, train_years=train_window)
        for weight_cap in [0.25, 0.35, 0.50]:
            _add_candidate(candidates, seen, base, max_wt=weight_cap)
        for draws in [100, 200, 400]:
            _add_candidate(candidates, seen, base, mc_draws=draws)
        return candidates[:budget]


    def _score_candidate(backtest_df: pl.DataFrame, alloc_df: pl.DataFrame) -> dict | None:
        pdf = backtest_df.to_pandas().set_index("date")
        strategy_key = "spy_overlay" if "spy_overlay_return" in pdf.columns else "ml_mc"
        return_col = f"{strategy_key}_return"
        if return_col not in pdf.columns:
            return None
        returns = pdf[return_col].dropna()
        if len(returns) < 5:
            return None

        ann_return = float(returns.mean() * 252)
        ann_vol = float(returns.std(ddof=0) * np.sqrt(252))
        sharpe = float(ann_return / ann_vol) if ann_vol > 0 else 0.0
        equity_curve = (1 + returns).cumprod()
        running_max = equity_curve.cummax()
        max_drawdown = float(((equity_curve - running_max) / running_max).min())

        spy_returns = pdf["spy_return"].dropna() if "spy_return" in pdf.columns else pd.Series(dtype=float)
        if len(spy_returns) >= 5:
            spy_ann_return = float(spy_returns.mean() * 252)
            spy_ann_vol = float(spy_returns.std(ddof=0) * np.sqrt(252))
            spy_sharpe = float(spy_ann_return / spy_ann_vol) if spy_ann_vol > 0 else 0.0
            spy_equity_curve = (1 + spy_returns).cumprod()
            spy_running_max = spy_equity_curve.cummax()
            spy_max_drawdown = float(((spy_equity_curve - spy_running_max) / spy_running_max).min())
            common_index = returns.index.intersection(spy_returns.index)
            spy_period_hit_rate = (
                float((returns.loc[common_index] > spy_returns.loc[common_index]).mean())
                if len(common_index)
                else np.nan
            )
        else:
            spy_ann_return = 0.0
            spy_sharpe = 0.0
            spy_max_drawdown = 0.0
            spy_period_hit_rate = np.nan

        weight_col = f"{strategy_key}_weight"
        if weight_col not in alloc_df.columns:
            weight_col = "ml_mc_weight"
        weight_pdf = (
            alloc_df.pivot(values=weight_col, index="date", on="ticker")
            .sort("date")
            .to_pandas()
            .set_index("date")
            .fillna(0.0)
        )
        if len(weight_pdf) > 1:
            turnover = float(weight_pdf.diff().abs().sum(axis=1).iloc[1:].mean())
        else:
            turnover = float(weight_pdf.iloc[0].abs().sum()) if len(weight_pdf) else 0.0
        concentration = float(weight_pdf.abs().max(axis=1).mean()) if len(weight_pdf) else 0.0

        excess_ann_return = ann_return - spy_ann_return
        excess_sharpe = sharpe - spy_sharpe
        drawdown_edge = max_drawdown - spy_max_drawdown
        hit_rate_reward = 0.0 if np.isnan(spy_period_hit_rate) else spy_period_hit_rate - 0.50
        score = (
            (1.50 * excess_sharpe)
            + (1.00 * excess_ann_return)
            + (1.00 * drawdown_edge)
            + (0.25 * hit_rate_reward)
            - (0.10 * turnover)
            - (0.25 * concentration)
        )
        return {
            "score": float(score),
            "scored_strategy": strategy_key,
            "ann_return": ann_return,
            "ann_vol": ann_vol,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown,
            "spy_ann_return": spy_ann_return,
            "spy_sharpe": spy_sharpe,
            "spy_max_drawdown": spy_max_drawdown,
            "excess_ann_return": excess_ann_return,
            "excess_sharpe": excess_sharpe,
            "drawdown_edge": drawdown_edge,
            "spy_period_hit_rate": spy_period_hit_rate,
            "turnover": turnover,
            "max_position": concentration,
        }


    optimizer_result = {
        "status": "off",
        "manual_settings": manual_settings,
        "feature_settings": {
            "feature_depth": int(feature_depth.value),
            "include_pair_factors": bool(include_pair_factors.value),
            "pair_factor_limit": int(pair_factor_limit.value),
        },
        "suggestion": None,
        "selection_end": None,
        "candidate_rows": [],
        "validation_window": {
            "start_after": research_end_date.value,
            "end_on_or_before": validation_end_date.value,
        },
    }

    if not run_parameter_optimizer.value:
        _view = mo.md("""
        Parameter optimizer is off. Keep it off for the first manual run; turn it on when you are ready to spend extra runtime on a bounded no-lookahead search.
        """)
    else:
        candidates = _candidate_grid(manual_settings, int(optimizer_candidate_budget.value))
        validation_budget = int(optimizer_validation_splits.value)
        time_budget = int(optimizer_time_limit.value)
        rows = []

        for candidate_id, params in enumerate(candidates, start=1):
            try:
                candidate_feat_df = build_features(
                    prices,
                    valid_tickers,
                    params["forward_days"],
                    int(feature_depth.value),
                    bool(include_pair_factors.value),
                    int(pair_factor_limit.value),
                )
                _base_cols = ["date", "month", "year", "dow"]
                _target_cols = [f"fwd_return_{t}" for t in valid_tickers]
                candidate_feature_cols = [
                    c for c in candidate_feat_df.columns if c not in _base_cols + _target_cols
                ]
                _feature_count = len(candidate_feature_cols)
                _pair_feature_count = sum(c.startswith("pair_") for c in candidate_feature_cols)
                candidate_splits = build_walk_forward_splits(
                    candidate_feat_df,
                    params["train_years"],
                    params["rebalance_days"],
                    params["forward_days"],
                    int(embargo_days.value),
                )
                validation_splits = [
                    s for s in candidate_splits
                    if s["pred_start"] > research_end_date.value
                    and s["pred_end"] <= validation_end_date.value
                ]
                inner_count = min(validation_budget, len(validation_splits))
                if inner_count < 1:
                    rows.append({
                        "candidate": candidate_id,
                        **params,
                        "status": "skipped: no validation-window splits",
                        "feature_count": _feature_count,
                        "pair_features": _pair_feature_count,
                        "score": np.nan,
                    })
                    continue

                inner_splits = validation_splits[:inner_count]
                _selection_start = inner_splits[0]["pred_start"]
                _selection_end = inner_splits[-1]["pred_end"]
                _allocation_results = []
                for _split in inner_splits:
                    _allocation_results.extend(
                        train_and_allocate_split(
                            _split,
                            candidate_feat_df,
                            valid_tickers,
                            candidate_feature_cols,
                            bool(allow_short.value),
                            params["max_wt"],
                            params["mc_draws"],
                            time_budget,
                            params["forward_days"],
                            str(model_mode.value),
                            False,
                            OVERLAY_MIN_RISK_OFF_SCORE,
                        )
                    )

                if not _allocation_results:
                    rows.append({
                        "candidate": candidate_id,
                        **params,
                        "status": "skipped: no allocation records",
                        "selection_start": _selection_start,
                        "selection_end": _selection_end,
                        "feature_count": _feature_count,
                        "pair_features": _pair_feature_count,
                        "score": np.nan,
                    })
                    continue

                candidate_alloc_df = pl.DataFrame(_allocation_results).with_columns(pl.col("date").cast(pl.Date))
                candidate_backtest = backtest_allocations(
                    prices,
                    candidate_alloc_df,
                    valid_tickers,
                    None,
                    float(trading_cost_bps.value),
                    ["spy_overlay", "ew", "rp"],
                ).filter(pl.col("date") <= _selection_end)
                score_parts = _score_candidate(candidate_backtest, candidate_alloc_df)
                if score_parts is None:
                    rows.append({
                        "candidate": candidate_id,
                        **params,
                        "status": "skipped: too few validation returns",
                        "selection_start": _selection_start,
                        "selection_end": _selection_end,
                        "feature_count": _feature_count,
                        "pair_features": _pair_feature_count,
                        "score": np.nan,
                    })
                    continue

                rows.append({
                    "candidate": candidate_id,
                    **params,
                    **score_parts,
                    "selection_start": _selection_start,
                    "selection_end": _selection_end,
                    "feature_count": _feature_count,
                    "pair_features": _pair_feature_count,
                    "status": "scored",
                })
            except Exception as exc:
                rows.append({
                    "candidate": candidate_id,
                    **params,
                    "status": f"error: {type(exc).__name__}",
                    "score": np.nan,
                })

        _valid_rows = [
            row for row in rows
            if row.get("status") == "scored" and np.isfinite(row.get("score", np.nan))
        ]
        if not _valid_rows:
            optimizer_result = {
                **optimizer_result,
                "status": "failed",
                "candidate_rows": rows,
            }
            _view = mo.md(f"""
            The optimizer ran, but no candidate produced enough validation-window returns after `{research_end_date.value}` and through `{validation_end_date.value}`. Try a longer date range, fewer training years, or fewer optimizer validation splits.
            """)
        else:
            _ranked_rows = sorted(_valid_rows, key=lambda row: row["score"], reverse=True)
            best = _ranked_rows[0]
            _suggestion = {
                "forward_days": int(best["forward_days"]),
                "rebalance_days": int(best["rebalance_days"]),
                "train_years": int(best["train_years"]),
                "max_wt": float(best["max_wt"]),
                "mc_draws": int(best["mc_draws"]),
            }
            optimizer_result = {
                "status": "ready",
                "manual_settings": manual_settings,
                "feature_settings": {
                    "feature_depth": int(feature_depth.value),
                    "include_pair_factors": bool(include_pair_factors.value),
                    "pair_factor_limit": int(pair_factor_limit.value),
                },
                "suggestion": _suggestion,
                "selection_start": best["selection_start"],
                "selection_end": best["selection_end"],
                "candidate_rows": rows,
                "validation_window": {
                    "start_after": research_end_date.value,
                    "end_on_or_before": validation_end_date.value,
                },
            }

            _display_df = pd.DataFrame(_ranked_rows).head(8).copy()
            _display_df["Score"] = _display_df["score"].map(lambda v: f"{v:.2f}")
            _display_df["Sharpe"] = _display_df["sharpe"].map(lambda v: f"{v:.2f}")
            _display_df["Excess Sharpe vs SPY"] = _display_df["excess_sharpe"].map(lambda v: f"{v:.2f}")
            _display_df["Excess return vs SPY"] = _display_df["excess_ann_return"].map(lambda v: f"{v:.1%}")
            _display_df["Drawdown edge vs SPY"] = _display_df["drawdown_edge"].map(lambda v: f"{v:.1%}")
            _display_df["Max drawdown"] = _display_df["max_drawdown"].map(lambda v: f"{v:.1%}")
            _display_df["Turnover"] = _display_df["turnover"].map(lambda v: f"{v:.2f}")
            _display_df["Max position"] = _display_df["max_position"].map(lambda v: f"{v:.1%}")
            _display_df = _display_df.rename(columns={
                "candidate": "Candidate",
                "forward_days": "Forward",
                "rebalance_days": "Rebalance",
                "train_years": "Train years",
                "max_wt": "Max wt",
                "mc_draws": "MC draws",
                "selection_start": "Validation start",
                "selection_end": "Validation end",
                "feature_count": "Features",
                "pair_features": "Pair features",
            })
            _display_df = _display_df[[
                "Candidate",
                "Score",
                "Sharpe",
                "Excess Sharpe vs SPY",
                "Excess return vs SPY",
                "Drawdown edge vs SPY",
                "Max drawdown",
                "Turnover",
                "Max position",
                "Forward",
                "Rebalance",
                "Train years",
                "Max wt",
                "MC draws",
                "Features",
                "Pair features",
                "Validation start",
                "Validation end",
            ]]
            _view = mo.vstack([
                mo.md(f"""
                Optimizer selected **{_suggestion["forward_days"]}d forward / {_suggestion["rebalance_days"]}d rebalance / {_suggestion["train_years"]}y train / {_suggestion["max_wt"]:.0%} max weight / {_suggestion["mc_draws"]} MC draws**.

                The candidate score is SPY-aware: it rewards excess Sharpe, excess return, drawdown edge, and hit rate versus SPY, then penalizes turnover and concentration. It used validation-window returns from **`{best["selection_start"]}`** through **`{best["selection_end"]}`** only. Final OOS splits remain outside the optimizer scoring window.
                """),
                mo.ui.table(
                    _display_df,
                    pagination=False,
                    selection=None,
                    show_column_summaries=False,
                    show_data_types=False,
                    show_download=False,
                    max_height=320,
                ),
            ])

    _view
    return (optimizer_result,)


@app.cell
def _(
    RUN_PRESETS,
    allow_short,
    apply_optimizer_settings,
    embargo_days,
    evaluation_protocol,
    feature_depth,
    final_oos_end_date,
    final_oos_start_date,
    forward_days,
    include_ml_research,
    include_pair_factors,
    max_splits,
    max_wt,
    mc_draws,
    mo,
    model_mode,
    model_time_limit,
    optimizer_result,
    pair_factor_limit,
    rebalance_days,
    research_end_date,
    run_parameter_optimizer,
    run_preset,
    trading_cost_bps,
    train_years,
    validation_end_date,
):
    _preset_name = str(run_preset.value)
    _preset = RUN_PRESETS.get(_preset_name, {})
    _use_preset = _preset_name != "Custom"

    def _preset_value(key: str, control_value):
        return _preset.get(key, control_value) if _use_preset else control_value

    suggestion = optimizer_result.get("suggestion")
    use_optimizer_settings = bool(
        _preset_value("run_parameter_optimizer", run_parameter_optimizer.value)
        and apply_optimizer_settings.value
        and suggestion
    )
    active_settings = suggestion if use_optimizer_settings else {
        "forward_days": int(forward_days.value),
        "rebalance_days": int(rebalance_days.value),
        "train_years": int(train_years.value),
        "max_wt": float(max_wt.value),
        "mc_draws": int(mc_draws.value),
    }

    effective_forward_days = int(active_settings["forward_days"])
    effective_rebalance_days = int(active_settings["rebalance_days"])
    effective_train_years = int(active_settings["train_years"])
    effective_embargo_days = int(embargo_days.value)
    effective_max_wt = float(active_settings["max_wt"])
    effective_mc_draws = int(active_settings["mc_draws"])
    effective_allow_short = bool(allow_short.value)
    effective_model_time_limit = int(_preset_value("model_time_limit", model_time_limit.value))
    effective_model_mode = str(model_mode.value)
    effective_max_splits = int(_preset_value("max_splits", max_splits.value))
    effective_feature_depth = int(_preset_value("feature_depth", feature_depth.value))
    effective_include_pair_factors = bool(include_pair_factors.value)
    effective_pair_factor_limit = int(pair_factor_limit.value)
    effective_trading_cost_bps = float(trading_cost_bps.value)
    effective_research_end = research_end_date.value
    effective_validation_end = validation_end_date.value
    effective_final_oos_start = final_oos_start_date.value
    effective_final_oos_end = final_oos_end_date.value
    effective_include_ml_research = bool(_preset_value("include_ml_research", include_ml_research.value))
    evaluation_protocol_label = str(_preset_value("evaluation_protocol", evaluation_protocol.value))
    effective_evaluation_protocol = (
        "final_oos"
        if evaluation_protocol_label.startswith("Final OOS")
        else "validation"
        if evaluation_protocol_label.startswith("Validation")
        else "all"
    )
    try:
        import torch as _torch
        effective_cuda_available = bool(_torch.cuda.is_available())
    except Exception:
        effective_cuda_available = False

    _source = "optimizer suggestion" if use_optimizer_settings else (
        f"preset `{_preset_name}`" if _use_preset else "manual controls"
    )
    _holdout_note = ""
    if use_optimizer_settings and optimizer_result.get("selection_end") is not None:
        _holdout_note = f" Optimizer scoring ended `{optimizer_result['selection_end']}`; selected report splits must start later."

    mo.md(f"""
    ### Active Settings

    Source: **{_source}**. Protocol: **{evaluation_protocol_label}**.

    - Research ends: **`{effective_research_end}`**
    - Validation ends: **`{effective_validation_end}`**
    - Final OOS starts: **`{effective_final_oos_start}`**
    - Final OOS report ends: **`{effective_final_oos_end}`**
    - Prediction horizon: **{effective_forward_days} days**
    - Rebalance and retrain: every **{effective_rebalance_days} days**
    - Training window: **{effective_train_years} years**
    - Embargo after training: **{effective_embargo_days} days**
    - Max single-ETF weight: **{effective_max_wt:.0%}**
    - Monte Carlo draws: **{effective_mc_draws}**
    - Feature depth: **{effective_feature_depth}**
    - Forecast model: **shared cross-asset AutoGluon**
    - AutoGluon model mode: **{effective_model_mode}**
    - CUDA visible: **{effective_cuda_available}**
    - Trading cost: **{effective_trading_cost_bps:.1f} bps per dollar traded**
    - AutoGluon time limit: **{effective_model_time_limit}s per model fit**
    - ML research strategies: **{"on" if effective_include_ml_research else "off"}**

    {_holdout_note}
    """)
    return (
        effective_allow_short,
        effective_embargo_days,
        effective_evaluation_protocol,
        effective_feature_depth,
        effective_final_oos_start,
        effective_final_oos_end,
        effective_forward_days,
        effective_include_ml_research,
        effective_include_pair_factors,
        effective_max_splits,
        effective_max_wt,
        effective_mc_draws,
        effective_cuda_available,
        effective_model_time_limit,
        effective_model_mode,
        effective_pair_factor_limit,
        effective_rebalance_days,
        effective_research_end,
        effective_trading_cost_bps,
        effective_train_years,
        effective_validation_end,
        evaluation_protocol_label,
        use_optimizer_settings,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. Feature Engineering

    Machine learning begins by making a spreadsheet for the model.

    Each row is one date. Each column is either a clue or an answer.

    The target for each asset is:

    - `fwd_return_<ticker>`: the future return AutoGluon is trying to predict.

    The features are the clues AutoGluon can use:

    - recent returns: did this ETF go up or down recently?
    - momentum: has it been moving in the same direction for a while?
    - volatility: how jumpy has it been?
    - drawdown: how far is it below a recent high?
    - market breadth: how many ETFs in the group are rising?
    - relative strength: is one ETF doing better or worse than the group?

    Why use clues from other ETFs? A model predicting SPY can still learn from bonds, gold, sectors, or international ETFs. Markets are connected, so the model gets to test those links.

    Pair factors compare ETFs against each other. They can grow quickly, so keep the pair limit small until the notebook is running smoothly.

    Lookahead rule: clues must come from information known on or before the decision date. The answer can look into the future because it is what the model learns from, but those future answers are hidden when the model makes new predictions.
    """)
    return


@app.cell
def _(combinations, pl, prices_to_returns):
    def build_features(
        prices: pl.DataFrame,
        tickers: list[str],
        horizon: int,
        feature_depth: int = 1,
        include_pair_factors: bool = False,
        pair_factor_limit: int = 20,
    ) -> pl.DataFrame:
        """Build a wide feature matrix for ML."""
        returns = prices_to_returns(prices, tickers, horizon)

        if feature_depth >= 3:
            lag_windows = [1, 2, 3, 5, 10, 21, 42, 63, 126, 252]
            vol_windows = [5, 10, 21, 42, 63, 126]
            mom_windows = [5, 10, 21, 42, 63, 126, 252]
            ma_windows = [21, 63, 126, 252]
        elif feature_depth == 2:
            lag_windows = [1, 2, 3, 5, 10, 21, 42, 63, 126]
            vol_windows = [5, 10, 21, 63, 126]
            mom_windows = [5, 10, 21, 63, 126]
            ma_windows = [21, 63, 126]
        else:
            lag_windows = [5, 10, 21, 63]
            vol_windows = [21, 63]
            mom_windows = [21, 63]
            ma_windows = []

        df = returns.with_columns([
            pl.col("date").dt.month().alias("month"),
            pl.col("date").dt.year().alias("year"),
            pl.col("date").dt.weekday().alias("dow"),
        ])

        ret_cols = [f"ret_{t}" for t in tickers]
        if feature_depth >= 2:
            df = df.with_columns([
                pl.mean_horizontal([pl.col(c) for c in ret_cols]).alias("mkt_ret"),
                pl.mean_horizontal([
                    pl.when(pl.col(c) > 0).then(1.0).otherwise(0.0)
                    for c in ret_cols
                ]).alias("breadth_pos"),
            ])
            df = df.with_columns([
                pl.col("mkt_ret").shift(1).alias("mkt_lag1"),
                pl.col("mkt_ret").rolling_mean(window_size=21).alias("mkt_mom21"),
                pl.col("mkt_ret").rolling_mean(window_size=63).alias("mkt_mom63"),
                pl.col("mkt_ret").rolling_std(window_size=21).alias("mkt_vol21"),
                pl.col("mkt_ret").rolling_std(window_size=63).alias("mkt_vol63"),
                pl.col("breadth_pos").rolling_mean(window_size=21).alias("breadth_pos_21"),
            ])

        base_feature_exprs = []
        for t in tickers:
            ret_col = f"ret_{t}"
            for w in lag_windows:
                base_feature_exprs.append(pl.col(ret_col).shift(w).alias(f"lag{w}_{t}"))
            for w in vol_windows:
                base_feature_exprs.append(pl.col(ret_col).rolling_std(window_size=w).alias(f"vol{w}_{t}"))
            for w in mom_windows:
                base_feature_exprs.append(pl.col(ret_col).rolling_mean(window_size=w).alias(f"mom{w}_{t}"))

        if base_feature_exprs:
            df = df.with_columns(base_feature_exprs)

        if feature_depth >= 2:
            expanded_exprs = []
            for t in tickers:
                ret_col = f"ret_{t}"
                expanded_exprs.extend([
                    pl.when(pl.col(ret_col) < 0)
                    .then(pl.col(ret_col))
                    .otherwise(0.0)
                    .rolling_std(window_size=21)
                    .alias(f"downside_vol21_{t}"),
                    (
                        pl.col(f"vol21_{t}") / (pl.col(f"vol63_{t}") + 1e-9)
                    ).alias(f"vol_ratio21_63_{t}"),
                    (
                        100.0
                        - (
                            100.0
                            / (
                                1.0
                                + (
                                    pl.when(pl.col(ret_col) > 0)
                                    .then(pl.col(ret_col))
                                    .otherwise(0.0)
                                    .rolling_mean(window_size=14)
                                    / (
                                        pl.when(pl.col(ret_col) < 0)
                                        .then(-pl.col(ret_col))
                                        .otherwise(0.0)
                                        .rolling_mean(window_size=14)
                                        + 1e-9
                                    )
                                )
                            )
                        )
                    ).alias(f"rsi14_{t}"),
                    (pl.col(ret_col) - pl.col("mkt_ret")).alias(f"rel_mkt_ret_{t}"),
                    (pl.col(ret_col).rolling_mean(window_size=21) - pl.col("mkt_mom21")).alias(f"rel_mkt_mom21_{t}"),
                ])
                for w in ma_windows:
                    expanded_exprs.extend([
                        (pl.col(t) / (pl.col(t).rolling_mean(window_size=w) + 1e-9) - 1.0).alias(f"dist_ma{w}_{t}"),
                        (pl.col(t) / (pl.col(t).rolling_max(window_size=w) + 1e-9) - 1.0).alias(f"drawdown{w}_{t}"),
                        (
                            pl.col(ret_col).rolling_mean(window_size=w)
                            / (pl.col(ret_col).rolling_std(window_size=w) + 1e-9)
                        ).alias(f"ret_z{w}_{t}"),
                    ])
            if expanded_exprs:
                df = df.with_columns(expanded_exprs)

        if include_pair_factors and feature_depth >= 2:
            pair_limit = max(0, int(pair_factor_limit))
            pair_exprs = []
            for a, b in list(combinations(tickers, 2))[:pair_limit]:
                safe_pair = f"{a}_{b}".replace(".", "_").replace("-", "_")
                pair_exprs.extend([
                    (pl.col(a) - pl.col(b)).alias(f"pair_log_spread_{safe_pair}"),
                    (pl.col(f"ret_{a}") - pl.col(f"ret_{b}")).alias(f"pair_ret_spread_{safe_pair}"),
                    (
                        pl.col(f"ret_{a}").rolling_mean(window_size=21)
                        - pl.col(f"ret_{b}").rolling_mean(window_size=21)
                    ).alias(f"pair_mom21_{safe_pair}"),
                ])
                if feature_depth >= 3:
                    pair_exprs.extend([
                        (
                            pl.col(f"ret_{a}").rolling_mean(window_size=63)
                            - pl.col(f"ret_{b}").rolling_mean(window_size=63)
                        ).alias(f"pair_mom63_{safe_pair}"),
                        (
                            (pl.col(f"ret_{a}") - pl.col(f"ret_{b}")).rolling_std(window_size=21)
                        ).alias(f"pair_spread_vol21_{safe_pair}"),
                    ])
            if pair_exprs:
                df = df.with_columns(pair_exprs)

        df = df.drop_nulls()
        return df


    return (build_features,)


@app.cell
def _(
    build_features,
    effective_feature_depth,
    effective_forward_days,
    effective_include_pair_factors,
    effective_pair_factor_limit,
    mo,
    prices,
    valid_tickers,
):
    feat_df = build_features(
        prices,
        valid_tickers,
        effective_forward_days,
        effective_feature_depth,
        effective_include_pair_factors,
        effective_pair_factor_limit,
    )

    # Feature columns visible to AutoGluon (exclude date and future-return target columns)
    _base_cols = ["date", "month", "year", "dow"]
    _target_cols = [f"fwd_return_{t}" for t in valid_tickers]
    asset_feature_cols = [c for c in feat_df.columns if c not in _base_cols + _target_cols]
    _pair_feature_count = sum(c.startswith("pair_") for c in asset_feature_cols)

    mo.md(f"""
    Feature matrix has **`{len(feat_df)}`** rows and **`{len(feat_df.columns)}`** columns using a **{effective_forward_days}-day** forward return target and feature depth **{effective_feature_depth}**.

    Model feature count: **{len(asset_feature_cols)}**. Pair features: **{_pair_feature_count}**.
    """)
    return asset_feature_cols, feat_df


@app.cell
def _(
    asset_feature_cols,
    effective_cuda_available,
    effective_feature_depth,
    effective_include_pair_factors,
    effective_max_splits,
    effective_model_mode,
    effective_model_time_limit,
    mo,
    optimizer_candidate_budget,
    optimizer_time_limit,
    optimizer_validation_splits,
    run_parameter_optimizer,
):
    _feature_count = len(asset_feature_cols)
    _pair_feature_count = sum(c.startswith("pair_") for c in asset_feature_cols)
    final_fit_count = max(1, int(effective_max_splits))
    optimizer_fit_count = (
        int(optimizer_candidate_budget.value)
        * int(optimizer_validation_splits.value)
        if run_parameter_optimizer.value
        else 0
    )
    estimated_fit_seconds = (
        final_fit_count * int(effective_model_time_limit)
        + optimizer_fit_count * int(optimizer_time_limit.value)
    )

    heavy_features = (
        _feature_count >= 250
        or _pair_feature_count >= 60
        or (effective_include_pair_factors and effective_feature_depth >= 3)
    )
    heavy_training = (
        final_fit_count + optimizer_fit_count >= 80
        or estimated_fit_seconds >= 900
    )

    if heavy_features or heavy_training:
        guidance = f"""
        **MoLab runtime suggestion:** this is now a heavy workload.

        Forecast model: **shared cross-asset AutoGluon**. Model mode: **{effective_model_mode}**. CUDA visible: **{effective_cuda_available}**.

        If CUDA is false in MoLab, attach the GPU runtime before raising candidate budgets, split counts, or pair factors. Auto GPU-rich mode asks for GPU-capable AutoGluon models only when CUDA is visible.
        """
    else:
        guidance = f"""
        Current workload looks suitable for a normal first cloud run.

        Forecast model: **shared cross-asset AutoGluon**. Model mode: **{effective_model_mode}**. CUDA visible: **{effective_cuda_available}**.

        Use GPU when features, pair factors, optimizer candidates, or split count grow.
        """

    mo.md(f"""
    **Feature workload check**

    Features: **{_feature_count}** total. Pair features: **{_pair_feature_count}**. Estimated model fits: **{final_fit_count + optimizer_fit_count}**.

    The notebook trains one cross-asset model per split, then predicts each ticker with a ticker indicator.

    {guidance}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. Walk-Forward Validation

    Walk-forward validation is the trading version of a fair test.

    The notebook never shuffles dates randomly. It always moves forward through time:

    1. Train AutoGluon on past data.
    2. Wait through the embargo gap.
    3. Predict the next rebalance period.
    4. Build portfolio weights from those predictions and past risk data.
    5. Move forward, retrain, and repeat.

    So the current backtest uses **fixed scheduled retraining**. If rebalance frequency is 21 days, the notebook retrains around every 21 calendar days. It does not reuse one model across the full backtest.

    The calendar protocol separates the experiment into three roles:

    - **Research/tuning period**: build ideas.
    - **Validation period**: choose among a small number of settings.
    - **Final OOS period**: test the frozen rules. Do not tune on it.

    The notebook also uses purge and embargo rules:

    - **Purge**: drop training labels whose future outcomes would not be known yet.
    - **Embargo**: leave a short gap between training and prediction.

    A strong student habit: whenever you see a trading model, ask, "Could this have been known at the time?"
    """)
    return


@app.cell
def _(pl):
    def build_walk_forward_splits(
        df: pl.DataFrame,
        train_years: int,
        rebalance_days: int,
        forward_days: int,
        embargo_days: int = 0,
    ) -> list[dict]:
        """Generate trading-row walk-forward splits."""
        dates = df.sort("date")["date"].to_list()
        n = len(dates)
        if n < 2:
            return []

        train_rows = max(126, int(train_years * 252))
        rebalance_step = max(1, int(rebalance_days))
        embargo_rows = max(0, int(embargo_days))
        forward_rows = max(1, int(forward_days))

        splits = []
        train_start_idx = 0
        train_end_idx = min(train_rows, n - 1)

        while train_end_idx + embargo_rows + forward_rows < n:
            pred_start_idx = train_end_idx + embargo_rows
            pred_end_idx = min(pred_start_idx + rebalance_step - 1, n - 1)
            if pred_start_idx >= n:
                break
            splits.append({
                "train_start": dates[train_start_idx],
                "train_end": dates[train_end_idx],
                "embargo_days": embargo_rows,
                "pred_start": dates[pred_start_idx],
                "pred_end": dates[pred_end_idx],
            })
            train_start_idx += rebalance_step
            train_end_idx = min(train_end_idx + rebalance_step, n - 1)

        return splits

    return (build_walk_forward_splits,)


@app.cell
def _(
    build_walk_forward_splits,
    effective_embargo_days,
    effective_evaluation_protocol,
    effective_final_oos_end,
    effective_final_oos_start,
    effective_forward_days,
    effective_max_splits,
    effective_rebalance_days,
    effective_research_end,
    effective_train_years,
    effective_validation_end,
    evaluation_protocol_label,
    feat_df,
    mo,
    optimizer_result,
    pl,
    use_optimizer_settings,
):
    all_splits = build_walk_forward_splits(
        feat_df,
        effective_train_years,
        effective_rebalance_days,
        effective_forward_days,
        effective_embargo_days,
    )


    def _split_phase(split: dict) -> str:
        if split["pred_end"] <= effective_research_end:
            return "Research/tuning"
        if split["pred_start"] > effective_research_end and split["pred_end"] <= effective_validation_end:
            return "Validation"
        if split["pred_start"] >= effective_final_oos_start and split["pred_end"] <= effective_final_oos_end:
            return "Final OOS"
        if split["pred_start"] >= effective_final_oos_start:
            return "After selected OOS"
        return "Boundary/mixed"


    _phase_rows = []
    for _split in all_splits:
        _phase_rows.append({
            "Phase": _split_phase(_split),
            "Train Start": _split["train_start"],
            "Train End": _split["train_end"],
            "Embargo Days": _split.get("embargo_days", effective_embargo_days),
            "Prediction Start": _split["pred_start"],
            "Prediction End": _split["pred_end"],
        })
    split_phase_counts = (
        pl.DataFrame(_phase_rows)
        .group_by("Phase")
        .len()
        .rename({"len": "Available Splits"})
        .sort("Phase")
        if _phase_rows
        else pl.DataFrame({"Phase": [], "Available Splits": []})
    )

    if effective_evaluation_protocol == "validation":
        report_splits = [s for s in all_splits if _split_phase(s) == "Validation"]
        split_note = f"inside the validation window after `{effective_research_end}` and through `{effective_validation_end}`"
    elif effective_evaluation_protocol == "final_oos":
        report_splits = [s for s in all_splits if _split_phase(s) == "Final OOS"]
        split_note = f"inside the final OOS window from `{effective_final_oos_start}` through `{effective_final_oos_end}`"
    else:
        report_splits = all_splits
        split_note = "using all available walk-forward splits"

    _selection_end = optimizer_result.get("selection_end") if use_optimizer_settings else None
    if _selection_end is not None:
        report_splits = [s for s in report_splits if s["pred_start"] > _selection_end]
        split_note += f", after optimizer scoring ended `{_selection_end}`"

    splits = report_splits[:effective_max_splits]
    if not splits:
        raise ValueError(
            f"No walk-forward report splits are available for {evaluation_protocol_label}. "
            "Extend the end date, reduce the training window, reduce optimizer validation splits, "
            "or choose a broader evaluation protocol."
        )

    report_window_summary = {
        "protocol": effective_evaluation_protocol,
        "label": evaluation_protocol_label,
        "phase_note": split_note,
        "available_report_splits": len(report_splits),
        "running_report_splits": len(splits),
        "embargo_days": effective_embargo_days,
        "first_prediction_start": splits[0]["pred_start"],
        "last_prediction_end": splits[-1]["pred_end"],
        "final_oos_report_end": effective_final_oos_end,
    }

    mo.vstack([
        mo.md(f"""
        Created **{len(all_splits)}** walk-forward rebalance splits. Running **{len(splits)}** of **{len(report_splits)}** eligible report splits {split_note}.

        First report prediction window: **`{splits[0]["pred_start"]}`** to **`{splits[0]["pred_end"]}`**. Last selected report window: **`{splits[-1]["pred_start"]}`** to **`{splits[-1]["pred_end"]}`**.
        """),
        mo.ui.table(
            split_phase_counts.to_pandas(),
            pagination=False,
            selection=None,
            show_column_summaries=False,
            show_data_types=False,
            show_download=False,
            max_height=180,
        ),
    ])
    return report_window_summary, splits


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Allocation Engine

    The allocation engine turns model predictions into portfolio weights.

    Example: if the model likes SPY, GLD, and TLT, the allocator decides how much of the portfolio should go into each one.

    ### 5.1 Mean-Variance Optimizer

    The optimizer receives three things:

    - expected returns from AutoGluon forecasts
    - recent risk and correlation from historical returns
    - rules, such as max weight per ETF

    Plain-English goal: prefer ETFs with better predicted returns, but avoid putting too much weight into risky or highly similar ETFs.

    Math note for readers who want it:

    $$U(w) = w^T \mu - \frac{\lambda}{2} w^T \Sigma w$$

    Intuition:

    - Higher predicted return increases a weight.
    - Higher risk or higher correlation with other ETFs reduces a weight.
    - Constraints prevent the optimizer from putting everything into one asset.

    The notebook treats forecasts as a tilt, not a command. It starts from a diversified anchor that blends equal weight and risk parity, then lets AutoGluon move the weights modestly.

    Why so cautious? In portfolio allocation, a noisy forecast can look precise enough to push the optimizer into crowded positions. Shrinking the forecast and tilting around a simple anchor usually gives a fairer test of whether the model adds value.
    """)
    return


@app.cell
def _(RANDOM_SEED, minimize, np):
    def optimize_weights(
        mu: np.ndarray,
        sigma: np.ndarray,
        allow_short: bool,
        max_wt: float,
        risk_aversion: float = 1.0,
    ) -> np.ndarray:
        """Mean-variance weights via quadratic utility minimization."""
        n = len(mu)
        lower = -max_wt if allow_short else 0.0
        upper = max(max_wt, 1.0 / n if not allow_short else max_wt)
        w0 = np.ones(n) / n

        bounds = [(lower, upper) for _ in range(n)]
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

        def objective(w):
            return -w @ mu + (risk_aversion / 2.0) * w @ (sigma @ w)

        result = minimize(
            objective,
            w0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 1000, "ftol": 1e-9},
        )
        if not result.success or not np.all(np.isfinite(result.x)):
            return w0
        weights = np.clip(result.x, lower, upper)
        return weights / weights.sum()


    def mc_optimize_weights(
        mu: np.ndarray,
        sigma: np.ndarray,
        allow_short: bool,
        max_wt: float,
        n_draws: int,
        risk_aversion: float = 1.0,
        epsilon: float = 0.05,
    ) -> np.ndarray:
        """Resample means, optimize each draw, and average normalized weights."""
        rng = np.random.default_rng(RANDOM_SEED)
        jitter_cov = epsilon**2 * np.diag(np.maximum(np.diag(sigma), 1e-12))
        means = rng.multivariate_normal(mu, jitter_cov, size=n_draws)
        weights = np.array([
            optimize_weights(m, sigma, allow_short, max_wt, risk_aversion)
            for m in means
        ])
        avg = weights.mean(axis=0)
        return avg / avg.sum()

    return mc_optimize_weights, optimize_weights


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 5.2 AutoGluon Model Training and Allocation

    AutoGluon is an AutoML toolkit. That means it can try several model types and pick useful ones without us hand-building every model.

    Here, AutoGluon is used for a portfolio task: predict each ETF's future return from the feature spreadsheet.

    The notebook uses a shared cross-asset model:

    1. Stack the ETF targets into one longer table.
    2. Add a ticker column so the model knows which ETF each row is predicting.
    3. Fit one `TabularPredictor` per rebalance split.
    4. Predict each ETF's next-window return, then use the middle prediction as that ETF's expected return.

    The model-mode control chooses how ambitious AutoGluon should be. Fast CPU mode uses tree models. GPU-rich mode can try neural-network and XGBoost models when the cloud runtime has CUDA.

    Machine-learning lesson:

    - AutoGluon helps fit models, but it does not decide whether the experiment is fair.
    - The validation design matters as much as the model.
    - You still choose the target, features, universe, time budget, and trading rules.
    - A strong AutoGluon score can still become a weak trading strategy if costs, turnover, or lookahead bias are ignored.
    """)
    return


@app.cell
def _(
    TabularPredictor,
    mc_optimize_weights,
    np,
    optimize_weights,
    pd,
    pl,
    tempfile,
    torch,
):
    def train_and_allocate_split(
        split: dict,
        feat_df: pl.DataFrame,
        tickers: list[str],
        asset_feature_cols: list[str],
        allow_short: bool,
        max_wt: float,
        mc_draws: int,
        model_time_limit: int,
        forward_days_value: int,
        model_mode_value: str,
        include_ml_research: bool = False,
        overlay_min_risk_off_score: float = 0.15,
    ) -> list[dict]:
        """Train AutoGluon models on the split and return allocation records."""
        train_mask = (feat_df["date"] >= split["train_start"]) & (feat_df["date"] < split["train_end"])
        pred_mask = (feat_df["date"] >= split["pred_start"]) & (feat_df["date"] <= split["pred_end"])

        train_df = feat_df.filter(train_mask).sort("date")
        pred_df = feat_df.filter(pred_mask).sort("date")

        # The final horizon plus execution-lag rows in the training window have
        # target outcomes that would not be known at the rebalance date.
        label_lookahead_rows = int(forward_days_value) + 1
        if len(train_df) <= label_lookahead_rows:
            return []
        train_df = train_df.head(len(train_df) - label_lookahead_rows)

        if len(train_df) < 126 or len(pred_df) == 0:
            return []

        X_train = train_df.select(asset_feature_cols).to_pandas()
        X_pred = pred_df.select(asset_feature_cols).to_pandas()

        predicted_returns: list[float] = []
        fit_metadata: list[dict] = []
        fast_hyperparameters = {
            "RF": [{"ag_args": {"name_suffix": "RF"}, "ag_args_fit": {"num_gpus": 0}}],
            "XT": [{"ag_args": {"name_suffix": "XT"}, "ag_args_fit": {"num_gpus": 0}}],
        }
        _cuda_available = bool(torch is not None and torch.cuda.is_available())
        _mode = str(model_mode_value)
        _gpu_requested_by_mode = _mode == "GPU rich models" or (
            _mode == "Auto GPU rich if available" and _cuda_available
        )
        rich_candidates = {}
        if torch is not None:
            rich_candidates["NN_TORCH"] = [{
                "ag_args": {"name_suffix": "GPU"},
                "ag_args_fit": {"num_gpus": 1},
            }]
        try:
            import xgboost as _xgboost  # noqa: F401
            _xgb_available = True
        except Exception:
            _xgb_available = False
        if _xgb_available:
            rich_candidates["XGB"] = [{
                "ag_args": {"name_suffix": "GPU"},
                "ag_args_fit": {"num_gpus": 1},
            }]
        _use_gpu = bool(_gpu_requested_by_mode and rich_candidates)
        rich_hyperparameters = {**rich_candidates, **fast_hyperparameters}
        selected_hyperparameters = rich_hyperparameters if _use_gpu else fast_hyperparameters
        selected_num_gpus = 1 if _use_gpu else 0
        selected_model_plan = ", ".join(selected_hyperparameters.keys())
        available_rich_models = ", ".join(rich_candidates.keys()) if rich_candidates else "none"

        def _metadata(fit_backend: str, fit_status: str, fit_num_gpus: int, trained_models: str) -> dict:
            return {
                "ag_model_mode": _mode,
                "ag_cuda_available": _cuda_available,
                "ag_gpu_requested": _gpu_requested_by_mode,
                "ag_num_gpus": fit_num_gpus,
                "ag_model_plan": selected_model_plan,
                "ag_available_rich_models": available_rich_models,
                "ag_fit_backend": fit_backend,
                "ag_fit_status": fit_status,
                "ag_trained_models": trained_models,
            }

        def _fit_and_predict(train_data: pd.DataFrame, label_name: str, pred_data: pd.DataFrame):
            fit_backend = "gpu_rich" if _use_gpu else "cpu_fast"
            fit_status = "fit"
            fit_num_gpus = selected_num_gpus
            trained_models = ""
            with tempfile.TemporaryDirectory(prefix="ag_alloc_") as model_dir:
                predictor = TabularPredictor(
                    label=label_name,
                    path=model_dir,
                    verbosity=0,
                    problem_type="regression",
                )
                try:
                    predictor = predictor.fit(
                        train_data=train_data,
                        hyperparameters=selected_hyperparameters,
                        time_limit=model_time_limit,
                        num_gpus=selected_num_gpus,
                    )
                except Exception:
                    if not _use_gpu:
                        raise
                    fit_backend = "cpu_fallback"
                    fit_status = "gpu fallback"
                    fit_num_gpus = 0
                    with tempfile.TemporaryDirectory(prefix="ag_alloc_cpu_") as fallback_dir:
                        predictor = TabularPredictor(
                            label=label_name,
                            path=fallback_dir,
                            verbosity=0,
                            problem_type="regression",
                        ).fit(
                            train_data=train_data,
                            hyperparameters=fast_hyperparameters,
                            time_limit=model_time_limit,
                            num_gpus=0,
                        )
                        try:
                            trained_models = ", ".join(predictor.model_names())
                        except Exception:
                            trained_models = ""
                        preds = predictor.predict(pred_data)
                        return preds, fit_backend, fit_status, fit_num_gpus, trained_models
                try:
                    trained_models = ", ".join(predictor.model_names())
                except Exception:
                    trained_models = ""
                preds = predictor.predict(pred_data)
                return preds, fit_backend, fit_status, fit_num_gpus, trained_models

        ticker_id_map = {ticker: idx for idx, ticker in enumerate(tickers)}
        train_blocks = []
        for t in tickers:
            target = f"fwd_return_{t}"
            y_train = train_df[target].to_pandas()
            valid = ~y_train.isna()
            if int(valid.sum()) < 20:
                continue
            block = X_train.loc[valid].copy()
            block["target_ticker_id"] = ticker_id_map[t]
            block["target_return"] = y_train.loc[valid].to_numpy()
            train_blocks.append(block)

        min_shared_rows = max(100, 20 * len(tickers))
        if not train_blocks or sum(len(block) for block in train_blocks) < min_shared_rows:
            predicted_returns = [0.0 for _ in tickers]
            fit_metadata = [
                _metadata("none", "skipped: insufficient shared target rows", 0, "")
                for _ in tickers
            ]
        else:
            train_data = pd.concat(train_blocks, ignore_index=True)
            pred_blocks = []
            for t in tickers:
                block = X_pred.copy()
                block["target_ticker_id"] = ticker_id_map[t]
                block["_target_ticker_order"] = t
                pred_blocks.append(block)
            pred_data = pd.concat(pred_blocks, ignore_index=True)
            pred_tickers = pred_data.pop("_target_ticker_order")
            preds, fit_backend, fit_status, fit_num_gpus, trained_models = _fit_and_predict(
                train_data,
                "target_return",
                pred_data,
            )
            pred_summary = pd.DataFrame({
                "ticker": pred_tickers.to_numpy(),
                "prediction": np.asarray(preds, dtype=float),
            })
            medians = pred_summary.groupby("ticker")["prediction"].median()
            shared_metadata = _metadata(fit_backend, fit_status, fit_num_gpus, trained_models)
            for t in tickers:
                predicted_returns.append(float(medians.get(t, 0.0)))
                fit_metadata.append(dict(shared_metadata))

        cov_mask = feat_df["date"] < split["pred_start"]
        recent_returns = feat_df.filter(cov_mask).tail(252).select([f"ret_{t}" for t in tickers]).to_pandas().dropna()
        if recent_returns.empty or len(recent_returns) < 30:
            sigma = np.diag([0.0004] * len(tickers))
        else:
            sigma = recent_returns.cov().values * forward_days_value
            sigma = (sigma + sigma.T) / 2.0
            min_eig = np.min(np.linalg.eigvalsh(sigma))
            if min_eig < 1e-8:
                sigma += np.eye(len(tickers)) * (1e-8 - min_eig)

        w_ew = np.ones(len(tickers)) / len(tickers)
        inv_vol = 1.0 / (np.sqrt(np.diag(sigma)) + 1e-9)
        w_rp = inv_vol / inv_vol.sum()
        w_anchor = 0.5 * w_ew + 0.5 * w_rp

        mu = np.array(predicted_returns, dtype=float)
        horizon_vol = np.sqrt(np.maximum(np.diag(sigma), 1e-12))
        forecast_cap = max(0.02, float(np.nanmedian(horizon_vol)))
        mu_shrunk = np.clip(mu, -forecast_cap, forecast_cap)
        forecast_tilt_strength = 0.20

        if include_ml_research:
            w_ml_mc_forecast = mc_optimize_weights(
                mu_shrunk,
                sigma,
                allow_short,
                max_wt,
                n_draws=mc_draws,
                risk_aversion=10.0,
            )
            w_ml_forecast = optimize_weights(
                mu_shrunk,
                sigma,
                allow_short,
                max_wt,
                risk_aversion=10.0,
            )
            w_ml_mc = (1.0 - forecast_tilt_strength) * w_anchor + forecast_tilt_strength * w_ml_mc_forecast
            w_ml = (1.0 - forecast_tilt_strength) * w_anchor + forecast_tilt_strength * w_ml_forecast
            if allow_short:
                w_ml_mc = np.clip(w_ml_mc, -max_wt, max_wt)
                w_ml = np.clip(w_ml, -max_wt, max_wt)
            else:
                w_ml_mc = np.clip(w_ml_mc, 0.0, max_wt)
                w_ml = np.clip(w_ml, 0.0, max_wt)
            w_ml_mc = w_ml_mc / w_ml_mc.sum()
            w_ml = w_ml / w_ml.sum()
        else:
            w_ml_mc = w_anchor.copy()
            w_ml = w_anchor.copy()

        w_spy_overlay = np.zeros(len(tickers), dtype=float)
        overlay_spy_forecast = np.nan
        overlay_breadth = float(np.mean(mu_shrunk > 0.0)) if len(mu_shrunk) else np.nan
        overlay_risk_off_score = 0.0
        overlay_defensive_sleeve = 0.0
        if "SPY" in tickers:
            spy_idx = tickers.index("SPY")
            w_spy_overlay[spy_idx] = 1.0
            overlay_spy_forecast = float(mu_shrunk[spy_idx])
            defensive_tickers = [
                ticker for ticker in ["TLT", "GLD", "IEF", "SHY", "BIL"]
                if ticker in tickers and ticker != "SPY"
            ]
            if defensive_tickers:
                spy_vol = float(max(horizon_vol[spy_idx], 1e-6))
                risk_off_signal = float(np.clip(-overlay_spy_forecast / spy_vol, 0.0, 1.0))
                weak_breadth_signal = float(np.clip((0.50 - overlay_breadth) * 2.0, 0.0, 1.0))
                overlay_risk_off_score = float(
                    np.clip((0.70 * risk_off_signal) + (0.30 * weak_breadth_signal), 0.0, 1.0)
                )
                if overlay_risk_off_score >= overlay_min_risk_off_score:
                    max_defensive_sleeve = min(0.40, max(0.0, float(max_wt)))
                    overlay_defensive_sleeve = max_defensive_sleeve * overlay_risk_off_score
                if overlay_defensive_sleeve > 1e-6:
                    defensive_idx = np.array([tickers.index(ticker) for ticker in defensive_tickers], dtype=int)
                    defensive_vol = horizon_vol[defensive_idx]
                    defensive_scores = 1.0 / (defensive_vol + 1e-9)
                    defensive_alloc = defensive_scores / defensive_scores.sum()
                    w_spy_overlay[spy_idx] = 1.0 - overlay_defensive_sleeve
                    for _idx, _weight in zip(defensive_idx, defensive_alloc):
                        w_spy_overlay[_idx] += overlay_defensive_sleeve * float(_weight)
        else:
            w_spy_overlay = w_ml_mc.copy()
            overlay_risk_off_score = np.nan
            overlay_defensive_sleeve = np.nan

        overlay_sum = float(np.nansum(w_spy_overlay))
        if not np.isfinite(overlay_sum) or abs(overlay_sum) < 1e-9:
            w_spy_overlay = w_ew.copy()
        else:
            w_spy_overlay = w_spy_overlay / overlay_sum

        alloc_date = pred_df["date"].min()
        realized_end = pred_df["date"].max()
        records = []
        for idx, t in enumerate(tickers):
            records.append({
                "date": alloc_date,
                "train_start": split["train_start"],
                "train_end": split["train_end"],
                "pred_start": split["pred_start"],
                "pred_end": split["pred_end"],
                "realized_end": realized_end,
                "split": f"{split['pred_start']} - {split['pred_end']}",
                "ticker": t,
                "predicted_return": predicted_returns[idx],
                **fit_metadata[idx],
                "spy_overlay_weight": float(w_spy_overlay[idx]),
                "spy_overlay_spy_forecast": overlay_spy_forecast,
                "spy_overlay_breadth": overlay_breadth,
                "spy_overlay_risk_off_score": overlay_risk_off_score,
                "spy_overlay_defensive_sleeve": overlay_defensive_sleeve,
                "ml_mc_weight": float(w_ml_mc[idx]),
                "ml_weight": float(w_ml[idx]),
                "ew_weight": float(w_ew[idx]),
                "rp_weight": float(w_rp[idx]),
            })
        return records

    return (train_and_allocate_split,)


@app.cell
def _(
    OVERLAY_MIN_RISK_OFF_SCORE,
    asset_feature_cols,
    effective_allow_short,
    effective_forward_days,
    effective_include_ml_research,
    effective_max_wt,
    effective_mc_draws,
    effective_model_time_limit,
    effective_model_mode,
    feat_df,
    mo,
    pl,
    splits,
    train_and_allocate_split,
    valid_tickers,
):
    _allocation_results = []
    for _split in splits:
        _allocation_results.extend(
            train_and_allocate_split(
                _split,
                feat_df,
                valid_tickers,
                asset_feature_cols,
                effective_allow_short,
                effective_max_wt,
                effective_mc_draws,
                effective_model_time_limit,
                effective_forward_days,
                effective_model_mode,
                effective_include_ml_research,
                OVERLAY_MIN_RISK_OFF_SCORE,
            )
        )

    if not _allocation_results:
        raise ValueError("No allocation records were generated. Increase the date range, reduce the training window, or check data availability.")

    alloc_df = pl.DataFrame(_allocation_results).with_columns(pl.col("date").cast(pl.Date))
    mo.md(f"""
    Generated **{len(alloc_df)}** allocation records across **{alloc_df['date'].n_unique()}** rebalance dates.
    """)
    return (alloc_df,)


@app.cell
def _(alloc_df, effective_cuda_available, effective_model_mode, mo, pd, pl):
    _backend_cols = [
        "date",
        "ticker",
        "ag_model_mode",
        "ag_cuda_available",
        "ag_gpu_requested",
        "ag_num_gpus",
        "ag_model_plan",
        "ag_available_rich_models",
        "ag_fit_backend",
        "ag_fit_status",
        "ag_trained_models",
    ]
    _available_backend_cols = [c for c in _backend_cols if c in alloc_df.columns]
    autogluon_backend_diagnostics_df = pl.DataFrame()

    if len(_available_backend_cols) < len(_backend_cols):
        _backend_view = mo.md("AutoGluon backend diagnostics are not available for this run.")
    else:
        _backend_pdf = (
            alloc_df
            .select(_available_backend_cols)
            .unique()
            .sort(["date", "ticker"])
            .to_pandas()
        )
        _summary_pdf = (
            _backend_pdf
            .groupby(
                [
                    "ag_model_mode",
                    "ag_cuda_available",
                    "ag_gpu_requested",
                    "ag_num_gpus",
                    "ag_fit_backend",
                    "ag_fit_status",
                    "ag_model_plan",
                    "ag_available_rich_models",
                ],
                dropna=False,
            )
            .size()
            .reset_index(name="Asset fits")
            .sort_values(["ag_fit_backend", "ag_fit_status"])
        )
        _fallback_count = int((_backend_pdf["ag_fit_backend"] == "cpu_fallback").sum())
        _gpu_fit_count = int((_backend_pdf["ag_fit_backend"] == "gpu_rich").sum())
        _cpu_fit_count = int((_backend_pdf["ag_fit_backend"] == "cpu_fast").sum())
        _skipped_count = int((_backend_pdf["ag_fit_backend"] == "none").sum())
        _total_fits = int(len(_backend_pdf))
        _trained_models = sorted(
            {
                _model.strip()
                for _value in _backend_pdf["ag_trained_models"].dropna().astype(str)
                for _model in _value.split(",")
                if _model.strip()
            }
        )
        _trained_model_note = ", ".join(_trained_models[:12]) if _trained_models else "not reported"
        if len(_trained_models) > 12:
            _trained_model_note += f", plus {len(_trained_models) - 12} more"
        _rich_model_notes = sorted(
            str(_value)
            for _value in _backend_pdf["ag_available_rich_models"].dropna().unique()
        )
        _rich_model_note = "; ".join(_rich_model_notes) if _rich_model_notes else "none"

        autogluon_backend_diagnostics_df = pl.DataFrame(_summary_pdf)
        _backend_view = mo.vstack([
            mo.md(f"""
            ### AutoGluon Backend Diagnostics

            This table answers: **what did AutoGluon actually train?**

            - Requested model mode: **{effective_model_mode}**
            - CUDA visible: **{effective_cuda_available}**
            - Asset fits: **{_total_fits}** total
            - GPU-rich fits: **{_gpu_fit_count}**
            - CPU-fast fits: **{_cpu_fit_count}**
            - GPU-to-CPU fallbacks: **{_fallback_count}**
            - Skipped fits: **{_skipped_count}**
            - Rich optional models available: `{_rich_model_note}`
            - AutoGluon trained models: `{_trained_model_note}`
            """),
            mo.ui.table(
                _summary_pdf,
                pagination=False,
                selection=None,
                show_column_summaries=False,
                show_data_types=False,
                show_download=False,
                max_height=320,
            ),
        ])

    _backend_view
    return (autogluon_backend_diagnostics_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 6. Backtest

    The backtest replays history using the weights the model would have chosen.

    It waits one trading day before applying new weights. That avoids pretending we could trade at the same close used to create the signal.

    Strategies compared:

    - **SPY+ML Overlay**: starts as 100% SPY. ML can only move a capped sleeve into defensive assets such as TLT and GLD when the SPY forecast and market breadth weaken.
    - **ML-MC**: AutoGluon forecasts used as a conservative Monte Carlo tilt around a diversified anchor.
    - **ML**: AutoGluon forecasts used as the same conservative direct tilt.
    - **EW**: equal weight, meaning every ETF gets the same share.
    - **RP**: risk parity style, meaning lower-volatility ETFs usually get more weight.
    - **SPY**: buy and hold SPY.
    - **60/40 SPY/IEF**: a simple stock/bond benchmark.

    Metrics:

    - **CAGR**: average yearly growth rate.
    - **Volatility**: how bumpy the ride was.
    - **Sharpe**: return compared with bumpiness.
    - **Max drawdown**: the worst peak-to-trough drop.
    - **Turnover**: how much the portfolio changed at rebalances.
    - **Hit rate vs SPY**: how often the strategy beat SPY over rebalance windows.

    Execution and cost assumptions:

    - Signals are formed after the rebalance date's adjusted close.
    - Weights start earning returns on the next trading day, approximated with a one-bar delay.
    - The default cost is **0 bps** because many retail ETF platforms offer commission-free trading.
    - Slippage is intentionally omitted; this is a slower allocation workflow, not an intraday execution notebook.
    """)
    return


@app.cell(hide_code=True)
def _(effective_include_ml_research):
    strategy_order = ["spy_overlay"]
    if effective_include_ml_research:
        strategy_order.extend(["ml_mc", "ml"])
    strategy_order.extend(["ew", "rp", "spy", "spy_ief_6040"])
    strategy_label_map = {
        "spy_overlay": "SPY+ML Overlay",
        "ml_mc": "ML-MC",
        "ml": "ML",
        "ew": "Equal Weight",
        "rp": "Risk Parity",
        "spy": "SPY",
        "spy_ief_6040": "60/40 SPY/IEF",
    }
    strategy_color_map = {
        "spy_overlay": "#0f766e",
        "ml_mc": "#2563eb",
        "ml": "#16a34a",
        "ew": "#f59e0b",
        "rp": "#dc2626",
        "spy": "#111827",
        "spy_ief_6040": "#7c3aed",
    }
    alloc_strategy_keys = ["spy_overlay"]
    if effective_include_ml_research:
        alloc_strategy_keys.extend(["ml_mc", "ml"])
    alloc_strategy_keys.extend(["ew", "rp"])
    return alloc_strategy_keys, strategy_color_map, strategy_label_map, strategy_order


@app.cell
def _(alloc_strategy_keys, np, pd, pl):
    def backtest_allocations(
        prices: pl.DataFrame,
        alloc_df: pl.DataFrame,
        tickers: list[str],
        benchmark_prices: pl.DataFrame | None = None,
        trading_cost_bps: float = 0.0,
        alloc_strategy_keys: list[str] | None = None,
    ) -> pl.DataFrame:
        """Compute daily net returns, turnover, costs, and equity curves with one-bar execution lag."""
        active_alloc_strategies = alloc_strategy_keys or ["spy_overlay", "ew", "rp"]
        price_df = prices.select(["date"] + tickers)
        ret_df = price_df.with_columns([
            (pl.col(t) / pl.col(t).shift(1) - 1).alias(f"sret_{t}") for t in tickers
        ]).drop_nulls()

        first_alloc_date = alloc_df["date"].min()
        ret_df = ret_df.filter(pl.col("date") >= first_alloc_date)
        if len(ret_df) == 0:
            raise ValueError("No return rows are available after the first allocation date.")

        weights_dict = {}
        for strategy in active_alloc_strategies:
            weight_col = f"{strategy}_weight"
            if weight_col not in alloc_df.columns:
                continue
            weights_dict[strategy] = alloc_df.pivot(
                values=weight_col,
                index="date",
                on="ticker",
            ).sort("date")

        ret_pandas = ret_df.to_pandas().set_index("date").sort_index()
        ret_pandas.index = pd.to_datetime(ret_pandas.index)
        asset_returns = ret_pandas.loc[:, [f"sret_{t}" for t in tickers]]
        cost_rate = max(0.0, float(trading_cost_bps)) / 10_000.0

        _series_parts = []

        def _turnover_from_exposure(exposure: pd.DataFrame | pd.Series, name: str) -> tuple[pd.Series, pd.Series]:
            if isinstance(exposure, pd.Series):
                turnover = exposure.diff().abs()
                if len(turnover):
                    turnover.iloc[0] = abs(float(exposure.iloc[0]))
            else:
                turnover = exposure.diff().abs().sum(axis=1)
                if len(turnover):
                    turnover.iloc[0] = exposure.iloc[0].abs().sum()
            turnover = turnover.fillna(0.0).rename(f"{name}_turnover")
            cost = (turnover * cost_rate).rename(f"{name}_cost")
            return turnover, cost

        def _strategy_series(strategy: str) -> tuple[pd.Series, pd.Series, pd.Series]:
            wdf = weights_dict[strategy].to_pandas().set_index("date").sort_index()
            wdf.index = pd.to_datetime(wdf.index)
            wdf = wdf.reindex(ret_pandas.index, method="ffill")
            wdf = wdf.reindex(columns=tickers).ffill().fillna(0.0)
            # Shift by one trading day so signals formed after a close do not earn
            # that same close-to-close return.
            wdf = wdf.shift(1).fillna(0.0)
            warr = wdf.values
            rarr = asset_returns.values
            gross_ret = np.nansum(warr * rarr, axis=1)
            turnover, cost = _turnover_from_exposure(wdf, strategy)
            net_ret = (pd.Series(gross_ret, index=ret_pandas.index) - cost).rename(f"{strategy}_return")
            return net_ret, turnover, cost

        for _strategy in active_alloc_strategies:
            if _strategy not in weights_dict:
                continue
            _series_parts.extend(_strategy_series(_strategy))

        _benchmark_source = benchmark_prices
        if _benchmark_source is None or "SPY" not in _benchmark_source.columns:
            _benchmark_source = prices if "SPY" in prices.columns else None

        if _benchmark_source is not None and "SPY" in _benchmark_source.columns:
            _bench_cols = ["date"] + [c for c in ["SPY", "IEF"] if c in _benchmark_source.columns]
            _bench_returns = (
                _benchmark_source
                .select(_bench_cols)
                .with_columns([
                    (pl.col(c) / pl.col(c).shift(1) - 1).alias(f"bret_{c}")
                    for c in _bench_cols if c != "date"
                ])
                .drop_nulls()
                .to_pandas()
                .set_index("date")
                .sort_index()
            )
            _bench_returns.index = pd.to_datetime(_bench_returns.index)
            _bench_returns = _bench_returns.reindex(ret_pandas.index).fillna(0.0)

            _spy_exposure = pd.Series(1.0, index=ret_pandas.index).shift(1).fillna(0.0)
            _spy_turnover, _spy_cost = _turnover_from_exposure(_spy_exposure, "spy")
            _spy_return = (_bench_returns["bret_SPY"] * _spy_exposure - _spy_cost).rename("spy_return")
            _series_parts.extend([_spy_return, _spy_turnover, _spy_cost])

            if "bret_IEF" in _bench_returns.columns:
                _balanced_exposure = pd.Series(1.0, index=ret_pandas.index).shift(1).fillna(0.0)
                _balanced_turnover, _balanced_cost = _turnover_from_exposure(_balanced_exposure, "spy_ief_6040")
                _balanced_gross = 0.60 * _bench_returns["bret_SPY"] + 0.40 * _bench_returns["bret_IEF"]
                _balanced_return = (_balanced_gross * _balanced_exposure - _balanced_cost).rename("spy_ief_6040_return")
                _series_parts.extend([_balanced_return, _balanced_turnover, _balanced_cost])

        backtest = pd.concat(_series_parts, axis=1)

        if "realized_end" in alloc_df.columns:
            _windows = (
                alloc_df
                .select(["date", "realized_end"])
                .unique()
                .sort("date")
                .to_pandas()
            )
            _window_mask = pd.Series(False, index=backtest.index)
            for _row in _windows.itertuples(index=False):
                _start = pd.to_datetime(_row.date)
                _end = pd.to_datetime(_row.realized_end)
                _window_mask = _window_mask | ((backtest.index >= _start) & (backtest.index <= _end))
            backtest = backtest.loc[_window_mask]
            if backtest.empty:
                raise ValueError("No backtest return rows overlap the selected prediction windows.")

        _return_cols = [c for c in backtest.columns if c.endswith("_return")]
        equity = (1 + backtest[_return_cols]).cumprod() * 100
        equity.columns = [c.replace("_return", "_equity") for c in equity.columns]

        backtest = pd.concat([backtest, equity], axis=1)
        backtest.index.name = "date"
        return pl.DataFrame(backtest.reset_index())

    return (backtest_allocations,)


@app.cell
def _(
    alloc_strategy_keys,
    alloc_df,
    backtest_allocations,
    benchmark_prices,
    effective_trading_cost_bps,
    prices,
    valid_tickers,
):
    backtest_df = backtest_allocations(
        prices,
        alloc_df,
        valid_tickers,
        benchmark_prices,
        effective_trading_cost_bps,
        alloc_strategy_keys,
    )
    _return_cols = [c for c in backtest_df.columns if c.endswith("_return")]
    _equity_cols = [c for c in backtest_df.columns if c.endswith("_equity")]
    _turnover_cost_cols = [c for c in backtest_df.columns if c.endswith("_turnover") or c.endswith("_cost")]
    portfolio_returns_df = backtest_df.select(["date"] + _return_cols)
    equity_df = backtest_df.select(["date"] + _equity_cols)
    turnover_cost_df = backtest_df.select(["date"] + _turnover_cost_cols)
    equity_df
    return backtest_df, equity_df


@app.cell
def _(
    alloc_df,
    backtest_df,
    effective_trading_cost_bps,
    mo,
    np,
    pd,
    pl,
    strategy_label_map,
    strategy_order,
):
    def compute_metrics(
        backtest_df: pl.DataFrame,
        alloc_df: pl.DataFrame,
        strategy_order: list[str],
        strategy_label_map: dict,
    ) -> pl.DataFrame:
        """Compute report-window performance, cost, turnover, and SPY-relative stats."""
        pdf = backtest_df.to_pandas().set_index("date").sort_index()
        pdf.index = pd.to_datetime(pdf.index)
        _rebalance_dates = pd.to_datetime(sorted(alloc_df["date"].unique().to_list()))
        metrics = []
        _return_cols = [f"{s}_return" for s in strategy_order if f"{s}_return" in pdf.columns]

        def _period_hit_rate(col: str) -> tuple[float, int]:
            if col == "spy_return" or "spy_return" not in pdf.columns or len(_rebalance_dates) == 0:
                return np.nan, 0
            hits = []
            for _idx, _start in enumerate(_rebalance_dates):
                _end = _rebalance_dates[_idx + 1] if _idx + 1 < len(_rebalance_dates) else pdf.index.max() + pd.Timedelta(days=1)
                _mask = (pdf.index >= _start) & (pdf.index < _end)
                if _mask.sum() < 2:
                    continue
                _strategy_period = float((1 + pdf.loc[_mask, col]).prod() - 1)
                _spy_period = float((1 + pdf.loc[_mask, "spy_return"]).prod() - 1)
                hits.append(_strategy_period > _spy_period)
            if not hits:
                return np.nan, 0
            return float(np.mean(hits)), len(hits)

        for col in _return_cols:
            strategy = col.removesuffix("_return")
            rets = pdf[col].dropna()
            if rets.empty:
                continue
            total_return = float((1 + rets).prod() - 1)
            years = max(len(rets) / 252.0, 1 / 252.0)
            cagr = float((1 + total_return) ** (1 / years) - 1) if total_return > -1 else np.nan
            ann_vol = float(rets.std(ddof=0) * np.sqrt(252))
            sharpe = float((rets.mean() * 252) / ann_vol) if ann_vol > 0 else np.nan
            equity_curve = (1 + rets).cumprod()
            running_max = equity_curve.cummax()
            drawdown = (equity_curve - running_max) / running_max
            max_drawdown = float(drawdown.min())
            calmar = float(cagr / abs(max_drawdown)) if max_drawdown < 0 and np.isfinite(cagr) else np.nan
            turnover_col = f"{strategy}_turnover"
            cost_col = f"{strategy}_cost"
            if turnover_col in pdf.columns:
                _turnover_points = pdf.loc[pdf[turnover_col] > 1e-12, turnover_col]
                avg_turnover = float(_turnover_points.mean()) if len(_turnover_points) else 0.0
            else:
                avg_turnover = np.nan
            total_cost = float(pdf[cost_col].sum()) if cost_col in pdf.columns else 0.0
            hit_rate, hit_periods = _period_hit_rate(col)
            metrics.append({
                "Strategy": strategy_label_map.get(strategy, strategy.upper().replace("_", "-")),
                "Key": strategy,
                "Total Return": total_return,
                "CAGR": cagr,
                "Ann. Vol": ann_vol,
                "Sharpe": sharpe,
                "Max Drawdown": max_drawdown,
                "Calmar": calmar,
                "Avg Rebalance Turnover": avg_turnover,
                "Total Cost Drag": total_cost,
                "Hit Rate vs SPY": hit_rate,
                "Hit Periods": hit_periods,
            })
        return pl.DataFrame(metrics)

    metrics_df = compute_metrics(backtest_df, alloc_df, strategy_order, strategy_label_map)
    _display_df = metrics_df.to_pandas().copy()
    for _col in ["Total Return", "CAGR", "Ann. Vol", "Max Drawdown", "Avg Rebalance Turnover", "Total Cost Drag", "Hit Rate vs SPY"]:
        if _col in _display_df.columns:
            _display_df[_col] = _display_df[_col].map(lambda _v: "--" if pd.isna(_v) else f"{_v:.1%}")
    for _col in ["Sharpe", "Calmar"]:
        if _col in _display_df.columns:
            _display_df[_col] = _display_df[_col].map(lambda _v: "--" if pd.isna(_v) else f"{_v:.2f}")
    _display_df = _display_df.drop(columns=["Key"], errors="ignore")

    mo.vstack([
        mo.md(f"""
        ### Performance Summary

        Returns are net of the selected trading-cost sensitivity: **{effective_trading_cost_bps:.1f} bps per dollar traded**. With the default **0 bps**, this reflects a commission-free ETF workflow and intentionally excludes intraday slippage.
        """),
        mo.ui.table(
            _display_df,
            pagination=False,
            selection=None,
            show_column_summaries=False,
            show_data_types=False,
            show_download=False,
            max_height=360,
        ),
    ])
    return (metrics_df,)


@app.cell
def _(
    GATE_MAX_SHARPE_GAP_VS_SIMPLE,
    GATE_MIN_CAGR_EDGE,
    GATE_MIN_DRAWDOWN_EDGE,
    GATE_MIN_HIT_RATE_VS_SPY,
    GATE_MIN_SHARPE_EDGE,
    alloc_df,
    effective_embargo_days,
    effective_evaluation_protocol,
    effective_final_oos_end,
    effective_final_oos_start,
    effective_rebalance_days,
    effective_research_end,
    effective_trading_cost_bps,
    effective_validation_end,
    evaluation_protocol_label,
    metrics_df,
    mo,
    np,
    pd,
    optimizer_result,
    pl,
    report_window_summary,
    use_optimizer_settings,
):
    _metrics_pdf = metrics_df.to_pandas()
    _rebalance_count = int(alloc_df["date"].n_unique())
    _selection_end = optimizer_result.get("selection_end") if use_optimizer_settings else None
    _overlay_row = _metrics_pdf[_metrics_pdf["Key"] == "spy_overlay"]
    _spy_row = _metrics_pdf[_metrics_pdf["Key"] == "spy"]
    _rows = []

    _protocol_detail = report_window_summary.get("phase_note", evaluation_protocol_label)
    _rows.append({"Check": "Protocol", "Status": "INFO", "Detail": _protocol_detail})
    _rows.append({"Check": "Research cutoff", "Status": "INFO", "Detail": f"Research/tuning runs through {effective_research_end}; validation runs through {effective_validation_end}; final OOS runs from {effective_final_oos_start} through {effective_final_oos_end}."})
    _rows.append({"Check": "Report window", "Status": "PASS", "Detail": f"{report_window_summary['first_prediction_start']} through {report_window_summary['last_prediction_end']} across {_rebalance_count} rebalance date(s)."})
    _rows.append({"Check": "Purged/embargoed validation", "Status": "PASS", "Detail": f"Walk-forward splits train on past rows only, purge the last forward-horizon plus execution-lag labels from each training window, and start predictions after a {effective_embargo_days}-day embargo."})
    _rows.append({"Check": "Report rebalance dates", "Status": "PASS" if _rebalance_count >= 3 else "REVIEW", "Detail": f"{_rebalance_count} report split(s); use 3+ before judging performance."})
    _rows.append({"Check": "Retraining policy", "Status": "INFO", "Detail": f"Fixed walk-forward retraining every {effective_rebalance_days} days; adaptive retraining is monitored separately, not used to change this report."})
    _rows.append({"Check": "Trading cost assumption", "Status": "INFO", "Detail": f"{effective_trading_cost_bps:.1f} bps per dollar traded; default 0 bps matches commission-free ETF platforms."})

    if effective_evaluation_protocol == "final_oos":
        _inside_oos = bool(
            report_window_summary["first_prediction_start"] >= effective_final_oos_start
            and report_window_summary["last_prediction_end"] <= effective_final_oos_end
        )
        _rows.append({"Check": "Final OOS isolation", "Status": "PASS" if _inside_oos else "REVIEW", "Detail": f"Selected report runs {report_window_summary['first_prediction_start']} through {report_window_summary['last_prediction_end']}; final OOS bounds are {effective_final_oos_start} through {effective_final_oos_end}."})
    elif effective_evaluation_protocol == "validation":
        _inside_validation = bool(
            report_window_summary["first_prediction_start"] > effective_research_end
            and report_window_summary["last_prediction_end"] <= effective_validation_end
        )
        _rows.append({"Check": "Validation isolation", "Status": "PASS" if _inside_validation else "REVIEW", "Detail": f"Selected report is checked against {effective_research_end} to {effective_validation_end}."})
    else:
        _rows.append({"Check": "Calendar isolation", "Status": "REVIEW", "Detail": "All-splits mode is useful for debugging, but not for a final OOS claim."})

    if _selection_end is not None:
        _report_after_selection = bool(report_window_summary["first_prediction_start"] > _selection_end)
        _rows.append({"Check": "Optimizer selection window", "Status": "PASS" if _report_after_selection else "REVIEW", "Detail": f"Candidate selection ended {_selection_end}; report starts {report_window_summary['first_prediction_start']}."})
    elif optimizer_result.get("status") == "ready":
        _rows.append({"Check": "Optimizer selection window", "Status": "REVIEW", "Detail": "Optimizer suggestion exists but is not applied, so this is a manual report."})
    else:
        _rows.append({"Check": "Optimizer selection window", "Status": "INFO", "Detail": "Optimizer is off; no parameter search touched the report window."})

    if _overlay_row.empty or _spy_row.empty:
        _rows.append({"Check": "Overlay vs SPY", "Status": "BLOCKED", "Detail": "Need both SPY+ML Overlay and SPY metrics to compare."})
        _spy_hurdle_pass = False
        _simple_hurdle_pass = False
        _spy_hurdle_blocked = True
    else:
        _overlay = _overlay_row.iloc[0]
        _spy = _spy_row.iloc[0]
        _ew_row = _metrics_pdf[_metrics_pdf["Key"] == "ew"]
        _rp_row = _metrics_pdf[_metrics_pdf["Key"] == "rp"]
        _sharpe_edge = float(_overlay["Sharpe"] - _spy["Sharpe"]) if pd.notna(_overlay["Sharpe"]) and pd.notna(_spy["Sharpe"]) else np.nan
        _cagr_edge = float(_overlay["CAGR"] - _spy["CAGR"]) if pd.notna(_overlay["CAGR"]) and pd.notna(_spy["CAGR"]) else np.nan
        _dd_edge = float(_overlay["Max Drawdown"] - _spy["Max Drawdown"]) if pd.notna(_overlay["Max Drawdown"]) and pd.notna(_spy["Max Drawdown"]) else np.nan
        _material_vs_spy = bool(
            (pd.notna(_sharpe_edge) and _sharpe_edge >= GATE_MIN_SHARPE_EDGE)
            or (pd.notna(_cagr_edge) and _cagr_edge >= GATE_MIN_CAGR_EDGE)
            or (pd.notna(_dd_edge) and _dd_edge >= GATE_MIN_DRAWDOWN_EDGE)
        )
        _dd_pass = bool(pd.notna(_dd_edge) and _dd_edge >= 0.0)
        _spy_hurdle_pass = bool(_material_vs_spy and _dd_pass)
        _spy_hurdle_blocked = False
        _overlay_sharpe = float(_overlay["Sharpe"]) if pd.notna(_overlay["Sharpe"]) else np.nan
        _ew_sharpe = float(_ew_row.iloc[0]["Sharpe"]) if not _ew_row.empty and pd.notna(_ew_row.iloc[0]["Sharpe"]) else np.nan
        _rp_sharpe = float(_rp_row.iloc[0]["Sharpe"]) if not _rp_row.empty and pd.notna(_rp_row.iloc[0]["Sharpe"]) else np.nan
        _ew_competitive = bool(pd.isna(_ew_sharpe) or (pd.notna(_overlay_sharpe) and _overlay_sharpe + GATE_MAX_SHARPE_GAP_VS_SIMPLE >= _ew_sharpe))
        _rp_competitive = bool(pd.isna(_rp_sharpe) or (pd.notna(_overlay_sharpe) and _overlay_sharpe + GATE_MAX_SHARPE_GAP_VS_SIMPLE >= _rp_sharpe))
        _simple_hurdle_pass = bool(_ew_competitive and _rp_competitive)
        _hit_rate = float(_overlay["Hit Rate vs SPY"]) if pd.notna(_overlay["Hit Rate vs SPY"]) else np.nan
        _hit_rate_pass = bool(pd.isna(_hit_rate) or _hit_rate >= GATE_MIN_HIT_RATE_VS_SPY)
        _rows.extend([
            {"Check": "Material Sharpe edge vs SPY", "Status": "PASS" if pd.notna(_sharpe_edge) and _sharpe_edge >= GATE_MIN_SHARPE_EDGE else "REVIEW", "Detail": f"Need >= {GATE_MIN_SHARPE_EDGE:.2f}; observed {_sharpe_edge:.2f}." if pd.notna(_sharpe_edge) else "Sharpe edge unavailable."},
            {"Check": "Material CAGR edge vs SPY", "Status": "PASS" if pd.notna(_cagr_edge) and _cagr_edge >= GATE_MIN_CAGR_EDGE else "REVIEW", "Detail": f"Need >= {GATE_MIN_CAGR_EDGE:.1%}; observed {_cagr_edge:.1%}." if pd.notna(_cagr_edge) else "CAGR edge unavailable."},
            {"Check": "Drawdown vs SPY", "Status": "PASS" if _dd_pass else "FAIL", "Detail": f"SPY+ML Overlay {_overlay['Max Drawdown']:.1%} vs SPY {_spy['Max Drawdown']:.1%}."},
            {"Check": "Sharpe vs Equal Weight", "Status": "PASS" if _ew_competitive else "FAIL", "Detail": f"Overlay Sharpe {_overlay_sharpe:.2f} vs EW {_ew_sharpe:.2f} with {GATE_MAX_SHARPE_GAP_VS_SIMPLE:.2f} tolerance." if pd.notna(_ew_sharpe) else "Equal Weight benchmark unavailable."},
            {"Check": "Sharpe vs Risk Parity", "Status": "PASS" if _rp_competitive else "FAIL", "Detail": f"Overlay Sharpe {_overlay_sharpe:.2f} vs RP {_rp_sharpe:.2f} with {GATE_MAX_SHARPE_GAP_VS_SIMPLE:.2f} tolerance." if pd.notna(_rp_sharpe) else "Risk Parity benchmark unavailable."},
            {"Check": "Hit rate vs SPY", "Status": "PASS" if _hit_rate_pass else "REVIEW", "Detail": f"Need >= {GATE_MIN_HIT_RATE_VS_SPY:.0%}; observed {_hit_rate:.1%}." if pd.notna(_hit_rate) else "Hit rate unavailable."},
            {
                "Check": "SPY deployment hurdle",
                "Status": "PASS" if _spy_hurdle_pass else "FAIL",
                "Detail": (
                    f"The overlay cleared a material SPY margin (>= {GATE_MIN_SHARPE_EDGE:.2f} Sharpe, >= {GATE_MIN_CAGR_EDGE:.0%} CAGR, or >= {GATE_MIN_DRAWDOWN_EDGE:.0%} drawdown) without worse max drawdown."
                    if _spy_hurdle_pass
                    else "The overlay did not clear a material SPY margin with acceptable drawdown."
                ),
            },
            {
                "Check": "Simple benchmark hurdle",
                "Status": "PASS" if _simple_hurdle_pass else "FAIL",
                "Detail": "The overlay stayed competitive with equal weight and risk parity." if _simple_hurdle_pass else "Equal weight or risk parity still looks stronger than the overlay.",
            },
        ])

    _deployment_pass = bool(_spy_hurdle_pass and _simple_hurdle_pass)
    oos_gate_df = pl.DataFrame(_rows)
    _gate_statuses = set(oos_gate_df["Status"].to_list())
    if _spy_hurdle_blocked:
        _gate_label = "BLOCKED"
        _gate_explanation = "The notebook could not compare the SPY+ML overlay with SPY, so the ML strategy cannot be judged."
    elif not _deployment_pass:
        _gate_label = "DO NOT DEPLOY OVERLAY"
        _gate_explanation = "The overlay did not clear the material SPY hurdle and/or simple benchmark hurdle. Keep SPY or the stronger simple rule."
    elif "REVIEW" in _gate_statuses or "BLOCKED" in _gate_statuses:
        _gate_label = "REVIEW"
        _gate_explanation = "The overlay cleared the main hurdles, but at least one supporting check still needs attention."
    else:
        _gate_label = "OVERLAY EARNED COMPLEXITY"
        _gate_explanation = "The overlay cleared the material SPY hurdle and stayed competitive with equal weight and risk parity in this report window."

    mo.vstack([
        mo.md(f"""
        ### Out-of-Sample Gate

        Gate verdict: **{_gate_label}**.

        {_gate_explanation}
        """),
        mo.ui.table(
            oos_gate_df.to_pandas(),
            pagination=False,
            selection=None,
            show_column_summaries=False,
            show_data_types=False,
            show_download=False,
            max_height=360,
        ),
    ])
    return


@app.cell
def _(alloc_df, alt, mo, np, pd, pl, prices, valid_tickers):
    def _build_prediction_diagnostics(
        alloc_df: pl.DataFrame,
        prices: pl.DataFrame,
        tickers: list[str],
    ) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        _alloc_pdf = alloc_df.to_pandas().sort_values(["date", "ticker"])
        _price_pdf = prices.select(["date"] + tickers).to_pandas().sort_values("date")
        _price_pdf["date"] = pd.to_datetime(_price_pdf["date"])
        _price_pdf = _price_pdf.set_index("date")
        _rows = []

        for _row in _alloc_pdf.itertuples(index=False):
            _ticker = _row.ticker
            if _ticker not in _price_pdf.columns:
                continue
            _signal_date = pd.to_datetime(_row.date)
            try:
                _end = pd.to_datetime(str(_row.split).split(" - ")[-1])
            except Exception:
                _end = _signal_date
            _start_pos = _price_pdf.index.searchsorted(_signal_date, side="right")
            _end_pos = _price_pdf.index.searchsorted(_end, side="right") - 1
            if _start_pos < 0 or _end_pos < 0 or _start_pos >= len(_price_pdf) or _end_pos >= len(_price_pdf) or _end_pos <= _start_pos:
                continue
            _start_px = float(_price_pdf.iloc[_start_pos][_ticker])
            _end_px = float(_price_pdf.iloc[_end_pos][_ticker])
            if not np.isfinite(_start_px) or not np.isfinite(_end_px) or _start_px <= 0 or _end_px <= 0:
                continue
            _realized = float(np.log(_end_px / _start_px))
            _prediction = float(_row.predicted_return)
            _rows.append({
                "signal_date": _signal_date.date(),
                "execution_start": _price_pdf.index[_start_pos].date(),
                "ticker": _ticker,
                "split": _row.split,
                "predicted_return": _prediction,
                "realized_return": _realized,
                "absolute_error": abs(_prediction - _realized),
                "direction_hit": bool(np.sign(_prediction) == np.sign(_realized)) if _prediction != 0 and _realized != 0 else False,
            })

        if not _rows:
            return pl.DataFrame(), pl.DataFrame(), pl.DataFrame()

        _diag_pdf = pd.DataFrame(_rows)
        _rank_rows = []
        for _date, _group in _diag_pdf.groupby("signal_date"):
            if len(_group) < 2:
                continue
            _spearman = _group["predicted_return"].rank().corr(_group["realized_return"].rank())
            _q = max(1, int(np.ceil(len(_group) * 0.20)))
            _ranked = _group.sort_values("predicted_return", ascending=False)
            _top_realized = float(_ranked.head(_q)["realized_return"].mean())
            _bottom_realized = float(_ranked.tail(_q)["realized_return"].mean())
            _rank_rows.append({
                "signal_date": _date,
                "assets": int(len(_group)),
                "spearman_rank_corr": float(_spearman) if pd.notna(_spearman) else np.nan,
                "top_quintile_realized": _top_realized,
                "bottom_quintile_realized": _bottom_realized,
                "top_minus_bottom": _top_realized - _bottom_realized,
            })

        _ticker_rows = []
        for _ticker, _group in _diag_pdf.groupby("ticker"):
            _ticker_rows.append({
                "ticker": _ticker,
                "observations": int(len(_group)),
                "mean_predicted": float(_group["predicted_return"].mean()),
                "mean_realized": float(_group["realized_return"].mean()),
                "direction_hit_rate": float(_group["direction_hit"].mean()),
                "mean_abs_error": float(_group["absolute_error"].mean()),
            })

        return (
            pl.DataFrame(_diag_pdf),
            pl.DataFrame(pd.DataFrame(_rank_rows)),
            pl.DataFrame(pd.DataFrame(_ticker_rows).sort_values("mean_realized", ascending=False)),
        )

    prediction_diagnostics_df, rank_diagnostics_df, ticker_diagnostics_df = _build_prediction_diagnostics(
        alloc_df,
        prices,
        valid_tickers,
    )

    if prediction_diagnostics_df.is_empty():
        _diagnostics_view = mo.md("Prediction diagnostics are not available yet. Run at least one valid allocation split with realized forward prices.")
    else:
        _diag_pdf = prediction_diagnostics_df.to_pandas()
        _rank_pdf = rank_diagnostics_df.to_pandas() if not rank_diagnostics_df.is_empty() else pd.DataFrame()
        _ticker_pdf = ticker_diagnostics_df.to_pandas() if not ticker_diagnostics_df.is_empty() else pd.DataFrame()

        _scatter = (
            alt.Chart(_diag_pdf)
            .mark_circle(size=72, opacity=0.72)
            .encode(
                x=alt.X("predicted_return:Q", title="Predicted next-bar forward log return"),
                y=alt.Y("realized_return:Q", title="Realized next-bar forward log return"),
                color=alt.Color("ticker:N", title="Ticker"),
                tooltip=[
                    alt.Tooltip("signal_date:T", title="Signal"),
                    alt.Tooltip("execution_start:T", title="Execution start"),
                    alt.Tooltip("ticker:N", title="Ticker"),
                    alt.Tooltip("predicted_return:Q", title="Predicted", format=".2%"),
                    alt.Tooltip("realized_return:Q", title="Realized", format=".2%"),
                    alt.Tooltip("absolute_error:Q", title="Abs error", format=".2%"),
                ],
            )
            .properties(height=320, title="Prediction vs realized next-bar forward returns")
            .interactive()
        )

        if not _rank_pdf.empty:
            _rank_display = _rank_pdf.copy()
            for _col in ["spearman_rank_corr"]:
                _rank_display[_col] = _rank_display[_col].map(lambda _v: "--" if pd.isna(_v) else f"{_v:.2f}")
            for _col in ["top_quintile_realized", "bottom_quintile_realized", "top_minus_bottom"]:
                _rank_display[_col] = _rank_display[_col].map(lambda _v: "--" if pd.isna(_v) else f"{_v:.1%}")
        else:
            _rank_display = pd.DataFrame()

        if not _ticker_pdf.empty:
            _ticker_display = _ticker_pdf.copy()
            for _col in ["mean_predicted", "mean_realized", "direction_hit_rate", "mean_abs_error"]:
                _ticker_display[_col] = _ticker_display[_col].map(lambda _v: "--" if pd.isna(_v) else f"{_v:.1%}")
        else:
            _ticker_display = pd.DataFrame()

        _diagnostics_view = mo.vstack([
            mo.md("""
            ### Model Usefulness Diagnostics

            These diagnostics ask a simple question:

            Did the model rank better ETFs above worse ETFs after the one-trading-day execution delay?
            """),
            _scatter,
            mo.ui.table(
                _rank_display,
                pagination=False,
                selection=None,
                show_column_summaries=False,
                show_data_types=False,
                show_download=False,
                max_height=260,
            ),
            mo.ui.table(
                _ticker_display,
                pagination=False,
                selection=None,
                show_column_summaries=False,
                show_data_types=False,
                show_download=False,
                max_height=320,
            ),
        ])

    _diagnostics_view
    return (prediction_diagnostics_df,)


@app.cell
def _(
    alloc_df,
    backtest_df,
    effective_rebalance_days,
    mo,
    pd,
    pl,
    prediction_diagnostics_df,
):
    _retraining_rows = []
    _rebalance_dates = pd.to_datetime(sorted(alloc_df["date"].unique().to_list()))
    _rebalance_count = len(_rebalance_dates)
    _retraining_rows.append({
        "Check": "Current policy",
        "Status": "INFO",
        "Detail": f"Fixed walk-forward retraining every {effective_rebalance_days} days for the selected report splits.",
    })
    if _rebalance_count >= 2:
        _gaps = _rebalance_dates.to_series().diff().dropna().dt.days
        _retraining_rows.append({
            "Check": "Observed retrain spacing",
            "Status": "INFO",
            "Detail": f"Median selected spacing is {float(_gaps.median()):.0f} calendar days across {_rebalance_count} rebalance dates.",
        })
    else:
        _retraining_rows.append({
            "Check": "Observed retrain spacing",
            "Status": "REVIEW",
            "Detail": "Only one rebalance date is selected; raise max_splits to inspect retraining stability.",
        })

    _diag_pdf = prediction_diagnostics_df.to_pandas() if len(prediction_diagnostics_df) else pd.DataFrame()
    if not _diag_pdf.empty and {"realized_return", "predicted_return"}.issubset(_diag_pdf.columns):
        _abs_error = (_diag_pdf["realized_return"] - _diag_pdf["predicted_return"]).abs()
        _realized_vol = float(_diag_pdf["realized_return"].std(ddof=0))
        _mae = float(_abs_error.mean())
        _error_status = "PASS" if _realized_vol == 0 or _mae <= 1.5 * _realized_vol else "REVIEW"
        _retraining_rows.append({
            "Check": "Prediction error drift",
            "Status": _error_status,
            "Detail": f"Mean absolute forecast error is {_mae:.2%}; realized-return volatility is {_realized_vol:.2%}.",
        })
    else:
        _retraining_rows.append({
            "Check": "Prediction error drift",
            "Status": "REVIEW",
            "Detail": "Prediction diagnostics need realized forward returns from at least one valid split.",
        })

    _bt_pdf = backtest_df.to_pandas().set_index("date").sort_index()
    _bt_pdf.index = pd.to_datetime(_bt_pdf.index)
    if {"spy_overlay_return", "spy_return"}.issubset(_bt_pdf.columns) and len(_bt_pdf) >= 20:
        _window = min(63, len(_bt_pdf))
        _recent_overlay = float((1 + _bt_pdf["spy_overlay_return"].tail(_window)).prod() - 1)
        _recent_spy = float((1 + _bt_pdf["spy_return"].tail(_window)).prod() - 1)
        _relative_status = "PASS" if _recent_overlay >= _recent_spy else "REVIEW"
        _retraining_rows.append({
            "Check": "Recent SPY-relative health",
            "Status": _relative_status,
            "Detail": f"Last {_window} trading rows: SPY+ML Overlay {_recent_overlay:.1%} vs SPY {_recent_spy:.1%}.",
        })
    else:
        _retraining_rows.append({
            "Check": "Recent SPY-relative health",
            "Status": "REVIEW",
            "Detail": "Need SPY+ML Overlay and SPY returns before checking recent relative performance.",
        })

    retraining_diagnostics_df = pl.DataFrame(_retraining_rows)

    mo.vstack([
        mo.md("""
        ### Adaptive Retraining Diagnostics

        This table does not change the backtest rules. It shows warning signs a future live system could watch before deciding whether to retrain earlier than the fixed schedule.
        """),
        mo.ui.table(
            retraining_diagnostics_df.to_pandas(),
            pagination=False,
            selection=None,
            show_column_summaries=False,
            show_data_types=False,
            show_download=False,
            max_height=260,
        ),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 7. Interactive Visualizations

    The charts use TradingView Lightweight Charts so you can inspect the strategy interactively:

    - mouse wheel to zoom
    - drag to pan
    - crosshair hover for exact values
    - reset view to return to the full sample

    The first chart shows the full downloaded market history, so you can see where the selected report window sits inside the larger sample.

    The strategy equity and drawdown charts use the selected report window that feeds the metrics table. That keeps the picture honest: if the report protocol is final OOS, the strategy chart shows final OOS rather than pretending the model traded periods that were not run.

    Use them as the first reality check. If the ML path does not beat a simple benchmark after costs, the model needs more evidence before it deserves more runtime.
    """)
    return


@app.cell
def _(anywidget, traitlets):
    class TradingViewLineChart(anywidget.AnyWidget):
        series = traitlets.List().tag(sync=True)
        title = traitlets.Unicode("").tag(sync=True)
        subtitle = traitlets.Unicode("").tag(sync=True)
        height = traitlets.Int(360).tag(sync=True)
        percent = traitlets.Bool(False).tag(sync=True)

        _esm = r"""
    import { createChart, LineSeries } from "https://esm.sh/lightweight-charts@5.0.8";

    function formatValue(value, percent) {
      if (value === undefined || value === null || Number.isNaN(Number(value))) return "--";
      return percent ? `${(Number(value) * 100).toFixed(2)}%` : Number(value).toFixed(2);
    }

    function render({ model, el }) {
      const root = document.createElement("div");
      root.className = "tv-card";

      const toolbar = document.createElement("div");
      toolbar.className = "tv-toolbar";

      const titleWrap = document.createElement("div");
      const title = document.createElement("div");
      title.className = "tv-title";
      title.textContent = model.get("title") || "Chart";
      const subtitle = document.createElement("div");
      subtitle.className = "tv-subtitle";
      subtitle.textContent = model.get("subtitle") || "Mouse wheel to zoom. Drag to pan.";
      titleWrap.append(title, subtitle);

      const resetButton = document.createElement("button");
      resetButton.className = "tv-reset";
      resetButton.type = "button";
      resetButton.textContent = "Reset view";

      toolbar.append(titleWrap, resetButton);

      const chartEl = document.createElement("div");
      chartEl.className = "tv-chart";
      chartEl.style.height = `${model.get("height") || 360}px`;

      const legend = document.createElement("div");
      legend.className = "tv-legend";

      const footer = document.createElement("div");
      footer.className = "tv-footer";
      footer.innerHTML = 'Lightweight Charts&trade; by <a href="https://www.tradingview.com/" target="_blank" rel="noopener noreferrer">TradingView</a>';

      root.append(toolbar, chartEl, legend, footer);
      el.replaceChildren(root);

      const percent = Boolean(model.get("percent"));
      const chart = createChart(chartEl, {
        autoSize: true,
        height: model.get("height") || 360,
        layout: {
          background: { color: "#ffffff" },
          textColor: "#1f2937",
          fontFamily: "Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif",
        },
        grid: {
          vertLines: { color: "#eef2f7" },
          horzLines: { color: "#eef2f7" },
        },
        crosshair: { mode: 1 },
        handleScroll: {
          mouseWheel: true,
          pressedMouseMove: true,
          horzTouchDrag: true,
          vertTouchDrag: false,
        },
        handleScale: {
          axisPressedMouseMove: true,
          mouseWheel: true,
          pinch: true,
        },
        rightPriceScale: {
          borderVisible: false,
          scaleMargins: { top: 0.14, bottom: 0.16 },
        },
        timeScale: {
          borderVisible: false,
          rightOffset: 8,
          barSpacing: 7,
          fixLeftEdge: false,
          fixRightEdge: false,
          timeVisible: false,
          secondsVisible: false,
        },
        localization: percent
          ? { priceFormatter: (price) => formatValue(price, true) }
          : undefined,
      });

      const seriesSpecs = model.get("series") || [];
      const handles = [];
      for (const spec of seriesSpecs) {
        const line = chart.addSeries(LineSeries, {
          title: spec.name,
          color: spec.color || "#2563eb",
          lineWidth: spec.lineWidth || 2,
          crosshairMarkerVisible: true,
          priceLineVisible: false,
          lastValueVisible: true,
          priceFormat: percent
            ? { type: "custom", minMove: 0.0001, formatter: (price) => formatValue(price, true) }
            : { type: "price", precision: 2, minMove: 0.01 },
        });
        line.setData(spec.data || []);
        handles.push({ name: spec.name, color: spec.color || "#2563eb", line, data: spec.data || [] });
      }

      function legendHtmlForLatest() {
        return handles.map((h) => {
          const last = h.data.length ? h.data[h.data.length - 1] : null;
          return `<span class="tv-legend-item"><span class="tv-dot" style="background:${h.color}"></span>${h.name}: <strong>${last ? formatValue(last.value, percent) : "--"}</strong></span>`;
        }).join("");
      }

      legend.innerHTML = legendHtmlForLatest();

      chart.subscribeCrosshairMove((param) => {
        if (!param || !param.time || !param.seriesData) {
          legend.innerHTML = legendHtmlForLatest();
          return;
        }
        const parts = handles.map((h) => {
          const point = param.seriesData.get(h.line);
          const value = point ? (point.value ?? point.close) : undefined;
          return `<span class="tv-legend-item"><span class="tv-dot" style="background:${h.color}"></span>${h.name}: <strong>${formatValue(value, percent)}</strong></span>`;
        });
        legend.innerHTML = `<span class="tv-time">${param.time}</span>${parts.join("")}`;
      });

      resetButton.addEventListener("click", () => chart.timeScale().fitContent());
      chart.timeScale().fitContent();

      return () => {
        chart.remove();
      };
    }

    export default { render };
    """

        _css = r"""
    .tv-card {
      border: 1px solid #d8dee9;
      border-radius: 8px;
      background: #ffffff;
      padding: 12px;
      color: #111827;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif;
    }
    .tv-toolbar {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 8px;
    }
    .tv-title {
      font-size: 15px;
      font-weight: 700;
      line-height: 1.25;
    }
    .tv-subtitle {
      margin-top: 2px;
      color: #667085;
      font-size: 12px;
      line-height: 1.3;
    }
    .tv-reset {
      border: 1px solid #cbd5e1;
      background: #f8fafc;
      color: #111827;
      border-radius: 6px;
      font-size: 12px;
      line-height: 1;
      padding: 7px 9px;
      cursor: pointer;
      white-space: nowrap;
    }
    .tv-reset:hover { background: #eef2f7; }
    .tv-chart {
      width: 100%;
      min-height: 240px;
    }
    .tv-legend {
      display: flex;
      flex-wrap: wrap;
      gap: 8px 12px;
      min-height: 24px;
      margin-top: 8px;
      font-size: 12px;
      color: #344054;
    }
    .tv-legend-item {
      display: inline-flex;
      align-items: center;
      gap: 4px;
    }
    .tv-dot {
      width: 8px;
      height: 8px;
      border-radius: 999px;
      display: inline-block;
    }
    .tv-time {
      color: #667085;
      font-weight: 600;
      margin-right: 2px;
    }
    .tv-footer {
      margin-top: 6px;
      color: #98a2b3;
      font-size: 11px;
    }
    .tv-footer a { color: #64748b; text-decoration: none; }
    .tv-footer a:hover { text-decoration: underline; }
    """

    return (TradingViewLineChart,)


@app.cell
def _(
    TradingViewLineChart,
    benchmark_prices,
    effective_final_oos_end,
    effective_research_end,
    effective_validation_end,
    equity_df,
    mo,
    pd,
    prices,
    report_window_summary,
    strategy_color_map,
    strategy_label_map,
    strategy_order,
    valid_tickers,
):
    _equity_pdf = equity_df.to_pandas()
    _equity_series = []
    _full_history_series = []
    _stage_colors = {
        "Research": "#64748b",
        "Validation": "#f59e0b",
        "Final OOS": strategy_color_map.get("ml_mc", "#2563eb"),
        "After selected OOS": "#94a3b8",
    }

    def _stage_for_date(_date) -> str:
        _d = pd.Timestamp(_date).date()
        if _d <= effective_research_end:
            return "Research"
        if _d <= effective_validation_end:
            return "Validation"
        if _d <= effective_final_oos_end:
            return "Final OOS"
        return "After selected OOS"

    def _points_from_series(_series: pd.Series) -> list[dict]:
        return [
            {"time": pd.Timestamp(_date).strftime("%Y-%m-%d"), "value": float(_value)}
            for _date, _value in zip(_series.index, _series.values)
            if pd.notna(_value)
        ]

    def _stage_points_from_series(_series: pd.Series, _stage: str) -> list[dict]:
        return [
            {"time": pd.Timestamp(_date).strftime("%Y-%m-%d"), "value": float(_value)}
            for _date, _value in zip(_series.index, _series.values)
            if pd.notna(_value) and _stage_for_date(_date) == _stage
        ]

    _price_pdf = (
        prices
        .select(["date"] + valid_tickers)
        .to_pandas()
        .set_index("date")
        .sort_index()
    )
    _price_pdf.index = pd.to_datetime(_price_pdf.index)
    _asset_returns = _price_pdf.pct_change().dropna(how="all").fillna(0.0)

    if not _asset_returns.empty:
        _full_ew_equity = (1.0 + _asset_returns.mean(axis=1)).cumprod() * 100.0
        _full_history_series.append({
            "name": "Equal Weight · full history",
            "color": strategy_color_map.get("ew", "#f59e0b"),
            "lineWidth": 2,
            "data": _points_from_series(_full_ew_equity),
        })

    _benchmark_pdf = (
        benchmark_prices
        .to_pandas()
        .set_index("date")
        .sort_index()
        if benchmark_prices is not None and "date" in benchmark_prices.columns
        else pd.DataFrame()
    )
    if not _benchmark_pdf.empty:
        _benchmark_pdf.index = pd.to_datetime(_benchmark_pdf.index)
        _benchmark_returns = _benchmark_pdf.pct_change().dropna(how="all").fillna(0.0)
    else:
        _benchmark_returns = pd.DataFrame()

    _spy_context = None
    if "SPY" in _benchmark_returns.columns:
        _spy_context = (1.0 + _benchmark_returns["SPY"]).cumprod() * 100.0
    elif "SPY" in _asset_returns.columns:
        _spy_context = (1.0 + _asset_returns["SPY"]).cumprod() * 100.0

    if _spy_context is not None:
        for _stage, _stage_color in _stage_colors.items():
            _points = _stage_points_from_series(_spy_context, _stage)
            if _points:
                _full_history_series.append({
                    "name": f"SPY · {_stage}",
                    "color": _stage_color,
                    "lineWidth": 3,
                    "data": _points,
                })

    if {"SPY", "IEF"}.issubset(set(_benchmark_returns.columns)):
        _balanced_return = 0.60 * _benchmark_returns["SPY"] + 0.40 * _benchmark_returns["IEF"]
        _full_balanced_equity = (1.0 + _balanced_return).cumprod() * 100.0
        _full_history_series.append({
            "name": "60/40 SPY/IEF · full history",
            "color": strategy_color_map.get("spy_ief_6040", "#7c3aed"),
            "lineWidth": 2,
            "data": _points_from_series(_full_balanced_equity),
        })

    for _strategy in strategy_order:
        _col = f"{_strategy}_equity"
        if _col not in _equity_pdf.columns:
            continue
        if _strategy == "spy_overlay":
            for _stage, _stage_color in _stage_colors.items():
                _points = []
                for _date, _value in zip(_equity_pdf["date"], _equity_pdf[_col]):
                    if pd.notna(_value) and _stage_for_date(_date) == _stage:
                        _points.append({"time": pd.Timestamp(_date).strftime("%Y-%m-%d"), "value": float(_value)})
                if _points:
                    _equity_series.append({
                        "name": f"{strategy_label_map.get(_strategy, _strategy.upper())} · {_stage}",
                        "color": _stage_color,
                        "lineWidth": 3,
                        "data": _points,
                    })
            continue
        _points = []
        for _date, _value in zip(_equity_pdf["date"], _equity_pdf[_col]):
            if pd.notna(_value):
                _points.append({"time": pd.Timestamp(_date).strftime("%Y-%m-%d"), "value": float(_value)})
        _equity_series.append({
            "name": strategy_label_map.get(_strategy, _strategy.upper()),
            "color": strategy_color_map.get(_strategy, "#475569"),
            "lineWidth": 3 if _strategy in {"spy_overlay", "spy"} else 2,
            "data": _points,
        })

    _full_history_chart = TradingViewLineChart(
        series=_full_history_series,
        title="Full downloaded market history",
        subtitle="Shows the whole loaded sample. SPY changes color by research, validation, final OOS, and after-selected-OOS stage. Strategy metrics still come from the selected report window below.",
        height=360,
        percent=False,
    )

    _report_start = report_window_summary.get("first_prediction_start")
    _report_end = report_window_summary.get("last_prediction_end")
    _report_chart = TradingViewLineChart(
        series=_equity_series,
        title="Selected report-window strategy equity",
        subtitle=f"This chart feeds the metrics table: {_report_start} through {_report_end}. The SPY+ML overlay changes color by stage inside the selected report window.",
        height=390,
        percent=False,
    )
    mo.vstack([_full_history_chart, _report_chart])
    return


@app.cell
def _(
    TradingViewLineChart,
    equity_df,
    pd,
    strategy_color_map,
    strategy_label_map,
    strategy_order,
):
    _dd_pdf = equity_df.to_pandas().set_index("date")
    _drawdown_series = []
    for _strategy in strategy_order:
        _col = f"{_strategy}_equity"
        if _col not in _dd_pdf.columns:
            continue
        _equity_curve = _dd_pdf[_col]
        _running_max = _equity_curve.cummax()
        _dd = (_equity_curve - _running_max) / _running_max
        _points = []
        for _date, _value in zip(_dd.index, _dd.values):
            if pd.notna(_value):
                _points.append({"time": pd.Timestamp(_date).strftime("%Y-%m-%d"), "value": float(_value)})
        _drawdown_series.append({
            "name": strategy_label_map.get(_strategy, _strategy.upper()),
            "color": strategy_color_map.get(_strategy, "#475569"),
            "lineWidth": 3 if _strategy in {"spy_overlay", "spy"} else 2,
            "data": _points,
        })

    TradingViewLineChart(
        series=_drawdown_series,
        title="Drawdowns",
        subtitle="Values are percent drawdown. Lower is worse; compare the SPY+ML overlay to SPY before celebrating the equity curve.",
        height=320,
        percent=True,
    )
    return


@app.cell
def _(TradingViewLineChart, alloc_df, alt, mo, pd, valid_tickers):
    _alloc_pdf = alloc_df.to_pandas().sort_values(["date", "ticker"])
    _alloc_colors = [
        "#2563eb", "#16a34a", "#f59e0b", "#dc2626", "#7c3aed",
        "#0891b2", "#db2777", "#65a30d", "#9333ea", "#475569",
    ]

    _weight_col = "spy_overlay_weight" if "spy_overlay_weight" in _alloc_pdf.columns else "ml_mc_weight"
    _allocation_label = "SPY+ML Overlay" if _weight_col == "spy_overlay_weight" else "ML-MC"

    if _alloc_pdf.empty:
        _allocation_view = mo.md(f"No {_allocation_label} allocations are available yet. Run at least one walk-forward split to create weights.")
    else:
        _alloc_pdf["date"] = pd.to_datetime(_alloc_pdf["date"])
        _latest_date = _alloc_pdf["date"].max()
        _latest_alloc = (
            _alloc_pdf[_alloc_pdf["date"] == _latest_date]
            .loc[:, ["ticker", _weight_col]]
            .rename(columns={_weight_col: "weight"})
            .fillna({"weight": 0.0})
            .sort_values("weight", ascending=False)
            .reset_index(drop=True)
        )
        _latest_alloc["rank"] = _latest_alloc.index + 1
        _latest_alloc["direction"] = _latest_alloc["weight"].map(lambda _w: "Long" if _w >= 0 else "Short")
        _latest_alloc["weight_label"] = _latest_alloc["weight"].map(lambda _w: f"{_w:.1%}")
        _latest_alloc["dollar_label"] = _latest_alloc["weight"].map(lambda _w: f"${_w * 10_000:,.0f}")

        _top_allocations = _latest_alloc.head(3)
        _top_text = ", ".join(
            f"**{_row.ticker} {_row.weight:.1%}**"
            for _row in _top_allocations.itertuples(index=False)
        )
        _rebalance_count = int(_alloc_pdf["date"].nunique())
        _split_note = (
            "Only one rebalance date is shown right now; raise `max_splits` to reveal the allocation path."
            if _rebalance_count == 1
            else f"The path below shows {_rebalance_count} rebalance dates."
        )
        _total_weight = _latest_alloc["weight"].sum()

        _latest_table = _latest_alloc.rename(columns={
            "rank": "Rank",
            "ticker": "Ticker",
            "direction": "Side",
            "weight_label": "Portfolio share",
            "dollar_label": "Approx dollars per $10k",
        }).loc[:, ["Rank", "Ticker", "Side", "Portfolio share", "Approx dollars per $10k"]]

        _bar_chart = (
            alt.Chart(_latest_alloc)
            .mark_bar(cornerRadiusEnd=4)
            .encode(
                y=alt.Y("ticker:N", sort="-x", title="Ticker"),
                x=alt.X(
                    "weight:Q",
                    axis=alt.Axis(format=".0%"),
                    title="Portfolio share",
                ),
                color=alt.Color(
                    "ticker:N",
                    legend=None,
                    scale=alt.Scale(range=_alloc_colors),
                ),
                tooltip=[
                    alt.Tooltip("ticker:N", title="Ticker"),
                    alt.Tooltip("weight:Q", title="Portfolio share", format=".1%"),
                ],
            )
            .properties(
                title=f"Current {_allocation_label} portfolio: {_latest_date:%Y-%m-%d}",
                height=max(260, 28 * len(_latest_alloc)),
            )
        )

        _path_view = mo.md("Run at least two rebalance splits to show the allocation path chart.")
        if _rebalance_count > 1:
            _alloc_series = []
            for _idx, _ticker in enumerate(valid_tickers):
                _sub = _alloc_pdf[_alloc_pdf["ticker"] == _ticker].sort_values("date")
                _points = []
                for _date, _value in zip(_sub["date"], _sub[_weight_col]):
                    if pd.notna(_value):
                        _points.append({"time": pd.Timestamp(_date).strftime("%Y-%m-%d"), "value": float(_value)})
                _alloc_series.append({
                    "name": _ticker,
                    "color": _alloc_colors[_idx % len(_alloc_colors)],
                    "data": _points,
                })

            _path_view = TradingViewLineChart(
                series=_alloc_series,
                title=f"{_allocation_label} allocation path",
                subtitle="Each line is one ticker's portfolio share at each rebalance. Zoom or pan to inspect changes over time.",
                height=340,
                percent=True,
            )

        _allocation_view = mo.vstack([
            mo.md(f"""
            ## {_allocation_label} Allocation Weights

            Read this as the overlay's current portfolio, not a price chart. A **25%** weight means about **$2,500 of every $10,000** is assigned to that ticker.

            Latest rebalance: **{_latest_date:%Y-%m-%d}**. Largest positions: {_top_text}. Total displayed weight: **{_total_weight:.1%}**. {_split_note}
            """),
            _bar_chart,
            mo.ui.table(
                _latest_table,
                pagination=False,
                selection=None,
                show_column_summaries=False,
                show_data_types=False,
                show_download=False,
                max_height=360,
            ),
            _path_view,
        ], gap=1.0)

    _allocation_view
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 8. Student Exercises and Next Experiments

    Try these in order:

    1. Run one manual split first. Confirm the data, allocations, and charts make sense.
    2. Turn on the bounded parameter optimizer with a tiny budget. Compare its suggested settings with your manual settings.
    3. Apply the optimizer suggestion and confirm the report splits start after the optimizer selection window.
    4. Raise `max_splits` from 1 to 3, then 10. Watch runtime and whether results remain stable.
    5. Compare universes: broad ETFs only, sector ETFs only, or risk assets plus defensive assets.
    6. Add transaction costs and turnover. A model that trades too much may lose its edge after costs.
    7. On MoLab, attach GPU for richer AutoGluon runs, then confirm the backend diagnostics table shows GPU-rich fits instead of CPU fallback.
    8. Make SPY the hurdle. If the SPY+ML overlay cannot beat SPY on return and Sharpe without worse drawdown, write down why SPY is still the better answer.

    Core lesson: trading ML is not about finding a chart that looks good once. It is about proving that the extra complexity beats a simple benchmark after skeptical tests.
    """)
    return


if __name__ == "__main__":
    app.run()
