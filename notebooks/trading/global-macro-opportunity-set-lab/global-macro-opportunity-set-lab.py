# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "altair==6.1.0",
#     "marimo==0.23.9",
#     "numpy==2.3.5",
#     "pandas==2.3.3",
#     "yfinance==1.4.1",
# ]
# ///

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Global Macro Opportunity Set Lab

    This notebook asks a clean question before we spend more time on machine learning:

    **Does broadening beyond SPY create a better opportunity set?**

    The idea is simple. If plain allocation rules cannot improve the benchmark stack, AutoGluon probably does not have enough useful raw material. If a broader universe already improves risk, drawdown, or return, then ML may have something real to work with later.

    This is an education and research notebook, not financial advice. Treat every result as a hypothesis that needs fresh validation.

    **Prerequisite for ML:** run this notebook before the [AutoGluon Portfolio Allocation Learning Lab](../autogluon-allocation-learning-lab/notebook.py). Only universes that pass the material benchmark gate below should be sent to AutoGluon.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Plain-English Map

    1. Pick one or more ETF universes.
    2. Download adjusted price history.
    3. Respect each ETF's real start date.
    4. Test simple allocation rules first.
    5. Compare everything with SPY.
    6. Decide which universes deserve ML later.

    Key words:

    - **Opportunity set**: the assets a strategy is allowed to choose from.
    - **Inception date**: the first date an ETF actually has tradable history.
    - **Backward-fill leak**: accidentally filling a newer ETF's missing early prices with future prices.
    - **Equal weight**: split the portfolio evenly across eligible assets.
    - **Risk parity style**: give lower-volatility assets more weight.
    - **Drawdown**: the fall from a previous high point.
    """)
    return


@app.cell
def _():
    import warnings
    warnings.filterwarnings("ignore")

    import marimo as mo
    from datetime import date, datetime
    from zoneinfo import ZoneInfo

    import altair as alt
    import numpy as np
    import pandas as pd
    import yfinance as yf

    pd.options.display.float_format = "{:,.4f}".format

    UNIVERSE_PRESETS = {
        "SPY core": ["SPY", "QQQ", "IWM", "TLT", "GLD"],
        "Global macro": ["SPY", "QQQ", "IWM", "VEA", "VWO", "TLT", "IEF", "GLD", "DBC"],
        "Rates + inflation": ["SPY", "TLT", "IEF", "SHY", "TIP", "GLD", "DBC"],
        "Managed futures sleeve": ["SPY", "TLT", "GLD", "DBMF", "KMLM"],
        "Trend alternatives research": ["SPY", "TLT", "GLD", "DBC", "DBMF", "KMLM"],
    }

    # Material benchmark margins (aligned with the AutoGluon OOS gate).
    GATE_MIN_SHARPE_EDGE = 0.10
    GATE_MIN_CAGR_EDGE = 0.01
    GATE_MIN_DRAWDOWN_EDGE = 0.02
    GATE_MAX_SHARPE_GAP_VS_SIMPLE = 0.05

    def build_crisis_windows(sample_end: date) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
        return {
            "Covid shock": (pd.Timestamp("2020-02-19"), pd.Timestamp("2020-03-23")),
            "2022 inflation/rates": (pd.Timestamp("2022-01-03"), pd.Timestamp("2022-10-14")),
            "Recent sample": (pd.Timestamp("2024-01-01"), pd.Timestamp(sample_end)),
        }

    mo.md("Setup complete.")
    return (
        GATE_MAX_SHARPE_GAP_VS_SIMPLE,
        GATE_MIN_CAGR_EDGE,
        GATE_MIN_DRAWDOWN_EDGE,
        GATE_MIN_SHARPE_EDGE,
        UNIVERSE_PRESETS,
        ZoneInfo,
        alt,
        build_crisis_windows,
        date,
        datetime,
        mo,
        np,
        pd,
        yf,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. Configuration

    Safe beginner move: run all presets with simple benchmark rules. Do not add ML until a universe has a reason to exist.

    Important safety rule: this notebook does **not** backward-fill prices. A newer ETF cannot magically exist before its real first price.
    """)
    return


@app.cell
def _(UNIVERSE_PRESETS, ZoneInfo, date, datetime, mo):
    _today = datetime.now(ZoneInfo("America/New_York")).date()
    preset_choice = mo.ui.dropdown(
        options=list(UNIVERSE_PRESETS.keys()),
        value="Global macro",
        label="Selected universe preset",
    )
    run_all_presets = mo.ui.checkbox(value=True, label="Scan all universe presets")
    custom_tickers = mo.ui.text(
        value="",
        label="Optional custom universe (comma-separated)",
        placeholder="SPY,TLT,GLD,DBMF,KMLM",
    )
    start_date = mo.ui.date(value=date(2018, 1, 1), label="Start date")
    end_date = mo.ui.date(value=_today, label="End date")
    min_history_days = mo.ui.slider(63, 756, value=252, step=21, label="Minimum history before asset can enter")
    rebalance_days = mo.ui.slider(5, 63, value=21, step=1, label="Rebalance frequency (trading rows)")
    volatility_lookback = mo.ui.slider(21, 252, value=63, step=21, label="Risk-parity volatility lookback")
    trading_cost_bps = mo.ui.slider(0.0, 10.0, value=0.0, step=0.5, label="Trading cost (bps per $ traded)")

    mo.vstack([
        mo.hstack([preset_choice, run_all_presets]),
        mo.hstack([custom_tickers]),
        mo.hstack([start_date, end_date]),
        mo.hstack([min_history_days, rebalance_days]),
        mo.hstack([volatility_lookback, trading_cost_bps]),
    ])
    return (
        custom_tickers,
        end_date,
        min_history_days,
        preset_choice,
        rebalance_days,
        run_all_presets,
        start_date,
        trading_cost_bps,
        volatility_lookback,
    )


@app.cell
def _(UNIVERSE_PRESETS, custom_tickers, mo, preset_choice, run_all_presets):
    selected_universes = (
        {name: tickers for name, tickers in UNIVERSE_PRESETS.items()}
        if run_all_presets.value
        else {preset_choice.value: UNIVERSE_PRESETS[preset_choice.value]}
    )

    _custom = [t.strip().upper() for t in custom_tickers.value.split(",") if t.strip()]
    if _custom:
        selected_universes["Custom"] = list(dict.fromkeys(_custom))

    download_tickers = sorted({
        ticker
        for tickers in selected_universes.values()
        for ticker in tickers
    } | {"SPY", "IEF"})

    mo.md(f"""
    Scanning **{len(selected_universes)}** universe(s).

    Download tickers: `{", ".join(download_tickers)}`.
    """)
    return download_tickers, selected_universes


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 2. Data Loading and Inception Safety

    The loader downloads adjusted close prices from yfinance.

    Inception safety rules:

    - Leading missing prices stay missing.
    - No backward-fill is used.
    - An asset needs enough past observations before it can receive a weight.
    - The first valid date for every ticker is shown.

    This matters most for newer managed-futures ETFs. If a notebook fills their missing early history with later prices, the backtest becomes fiction.
    """)
    return


@app.cell
def _(datetime, download_tickers, end_date, mo, pd, start_date, yf):
    def download_adjusted_prices(tickers: list[str], start: datetime, end: datetime) -> pd.DataFrame:
        if not tickers:
            raise ValueError("No tickers provided.")
        price_parts: list[pd.Series] = []
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
                        series = close_panel[ticker].rename(ticker)
                        series.index = pd.to_datetime(series.index).tz_localize(None)
                        price_parts.append(series)
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
                    close = raw["Close"] if "Close" in raw.columns else raw["Adj Close"]
                    close = close.rename(ticker)
                    close.index = pd.to_datetime(close.index).tz_localize(None)
                    price_parts.append(close)
                except Exception as exc:
                    print(f"Skipping {ticker}: {exc}")

        if not price_parts:
            raise ValueError("No price data downloaded.")
        prices = pd.concat(price_parts, axis=1).sort_index()
        prices.index.name = "date"
        return prices


    def build_inception_table(prices: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for ticker in prices.columns:
            valid = prices[ticker].dropna()
            rows.append({
                "Ticker": ticker,
                "First Valid Date": valid.index.min().date() if len(valid) else None,
                "Last Valid Date": valid.index.max().date() if len(valid) else None,
                "Observations": int(len(valid)),
            })
        return pd.DataFrame(rows).sort_values(["First Valid Date", "Ticker"])


    with mo.persistent_cache(name=f"global_macro_prices_v2_{','.join(download_tickers)}_{start_date.value}_{end_date.value}"):
        prices = download_adjusted_prices(download_tickers, start_date.value, end_date.value)

    inception_df = build_inception_table(prices)
    mo.vstack([
        mo.md(f"""
        Loaded adjusted prices from **{prices.index.min().date()}** through **{prices.index.max().date()}**.

        The table below is a leak check. Newer ETFs should have newer first valid dates.
        """),
        mo.ui.table(
            inception_df,
            pagination=False,
            selection=None,
            show_column_summaries=False,
            show_data_types=False,
            show_download=False,
            max_height=300,
        ),
    ])
    return build_inception_table, download_adjusted_prices, inception_df, prices


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3. Benchmark Allocation Rules

    We test simple rules before ML:

    - **SPY**: buy and hold SPY.
    - **60/40 SPY/IEF**: basic stock/bond benchmark.
    - **Equal weight**: same weight for each eligible asset.
    - **Risk parity style**: inverse-volatility weights for eligible assets.

    Execution timing:

    - Weights are formed after a rebalance close.
    - They start earning returns on the next trading row.
    - This one-row delay helps avoid same-close lookahead.
    """)
    return


@app.cell
def _(np, pd):
    def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
        # Forward-fill after inception only. Leading NaNs remain NaN.
        return prices.ffill().pct_change()


    def eligible_tickers(
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        tickers: list[str],
        row_idx: int,
        min_history: int,
    ) -> list[str]:
        eligible = []
        for ticker in tickers:
            if ticker not in prices.columns:
                continue
            current_price = prices[ticker].iloc[row_idx]
            history = returns[ticker].iloc[: row_idx + 1].dropna()
            if pd.notna(current_price) and len(history) >= min_history:
                eligible.append(ticker)
        return eligible


    def make_weight_frame(
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        tickers: list[str],
        strategy: str,
        min_history: int,
        rebalance_step: int,
        vol_lookback: int,
    ) -> pd.DataFrame:
        weight_rows = []
        dates = []
        for row_idx in range(max(1, min_history), len(prices), int(rebalance_step)):
            eligible = eligible_tickers(prices, returns, tickers, row_idx, min_history)
            if not eligible:
                continue
            weights = pd.Series(0.0, index=tickers, dtype=float)
            if strategy == "equal_weight":
                weights.loc[eligible] = 1.0 / len(eligible)
            elif strategy == "risk_parity":
                window = returns.loc[:, eligible].iloc[max(0, row_idx - int(vol_lookback) + 1): row_idx + 1]
                vol = window.std(ddof=0).replace(0.0, np.nan)
                inv_vol = 1.0 / vol
                inv_vol = inv_vol.replace([np.inf, -np.inf], np.nan).dropna()
                if inv_vol.empty:
                    weights.loc[eligible] = 1.0 / len(eligible)
                else:
                    weights.loc[inv_vol.index] = inv_vol / inv_vol.sum()
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            weight_rows.append(weights)
            dates.append(prices.index[row_idx])
        if not weight_rows:
            return pd.DataFrame(columns=tickers, dtype=float)
        out = pd.DataFrame(weight_rows, index=pd.DatetimeIndex(dates), columns=tickers)
        out.index.name = "date"
        return out


    def backtest_weights(
        returns: pd.DataFrame,
        signal_weights: pd.DataFrame,
        trading_cost_bps: float,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        if signal_weights.empty:
            empty = pd.Series(dtype=float)
            return empty, empty, empty
        exposure = signal_weights.reindex(returns.index).ffill().fillna(0.0)
        exposure = exposure.shift(1).fillna(0.0)
        aligned_returns = returns.reindex(columns=exposure.columns).fillna(0.0)
        gross = (exposure * aligned_returns).sum(axis=1)
        turnover = exposure.diff().abs().sum(axis=1).fillna(0.0)
        if len(turnover):
            first_active = exposure.abs().sum(axis=1) > 0
            if first_active.any():
                first_idx = first_active.idxmax()
                turnover.loc[first_idx] = exposure.loc[first_idx].abs().sum()
        cost = turnover * (max(0.0, float(trading_cost_bps)) / 10_000.0)
        net = gross - cost
        return net, turnover, cost


    def buy_and_hold_returns(
        returns: pd.DataFrame,
        weights: dict[str, float],
        trading_cost_bps: float,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        cols = [ticker for ticker in weights if ticker in returns.columns]
        if not cols:
            empty = pd.Series(dtype=float)
            return empty, empty, empty
        w = pd.Series({ticker: weights[ticker] for ticker in cols}, dtype=float)
        w = w / w.sum()
        gross = returns.loc[:, cols].fillna(0.0).mul(w, axis=1).sum(axis=1)
        turnover = pd.Series(0.0, index=returns.index)
        if len(turnover):
            turnover.iloc[0] = float(w.abs().sum())
        cost = turnover * (max(0.0, float(trading_cost_bps)) / 10_000.0)
        return gross - cost, turnover, cost

    return backtest_weights, buy_and_hold_returns, compute_returns, make_weight_frame


@app.cell
def _(
    backtest_weights,
    buy_and_hold_returns,
    compute_returns,
    make_weight_frame,
    min_history_days,
    pd,
    prices,
    rebalance_days,
    selected_universes,
    trading_cost_bps,
    volatility_lookback,
):
    returns = compute_returns(prices)
    series_parts = {}
    turnover_parts = {}
    cost_parts = {}
    weight_book = {}

    spy_ret, spy_turnover, spy_cost = buy_and_hold_returns(
        returns,
        {"SPY": 1.0},
        trading_cost_bps.value,
    )
    series_parts["SPY"] = spy_ret
    turnover_parts["SPY"] = spy_turnover
    cost_parts["SPY"] = spy_cost

    if {"SPY", "IEF"}.issubset(returns.columns):
        balanced_ret, balanced_turnover, balanced_cost = buy_and_hold_returns(
            returns,
            {"SPY": 0.60, "IEF": 0.40},
            trading_cost_bps.value,
        )
        series_parts["60/40 SPY/IEF"] = balanced_ret
        turnover_parts["60/40 SPY/IEF"] = balanced_turnover
        cost_parts["60/40 SPY/IEF"] = balanced_cost

    for universe_name, tickers in selected_universes.items():
        available = [ticker for ticker in tickers if ticker in returns.columns]
        if len(available) < 2:
            continue
        for strategy_key, strategy_label in [
            ("equal_weight", "Equal Weight"),
            ("risk_parity", "Risk Parity"),
        ]:
            weights = make_weight_frame(
                prices,
                returns,
                available,
                strategy_key,
                min_history_days.value,
                rebalance_days.value,
                volatility_lookback.value,
            )
            ret, turnover, cost = backtest_weights(returns, weights, trading_cost_bps.value)
            label = f"{universe_name} | {strategy_label}"
            series_parts[label] = ret
            turnover_parts[label] = turnover
            cost_parts[label] = cost
            weight_book[label] = weights

    portfolio_returns = pd.DataFrame(series_parts).sort_index()
    portfolio_turnover = pd.DataFrame(turnover_parts).reindex(portfolio_returns.index).fillna(0.0)
    portfolio_cost = pd.DataFrame(cost_parts).reindex(portfolio_returns.index).fillna(0.0)
    equity_curves = (1.0 + portfolio_returns.fillna(0.0)).cumprod() * 100.0
    return portfolio_cost, portfolio_returns, portfolio_turnover, returns, weight_book, equity_curves


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4. Scoreboard

    Read this table from left to right:

    - Higher total return, CAGR, Sharpe, and Calmar are better.
    - Lower drawdown is better, but it is shown as a negative number.
    - SPY-relative columns tell us whether a broader universe is actually helping.
    - The material gate below decides which universes may advance to AutoGluon.
    """)
    return


@app.cell
def _(np, pd, portfolio_cost, portfolio_returns, portfolio_turnover, rebalance_days):
    def max_drawdown(equity: pd.Series) -> float:
        running_max = equity.cummax()
        dd = (equity - running_max) / running_max
        return float(dd.min())


    def hit_rate_vs_spy(strategy: pd.Series, spy: pd.Series, rebalance_step: int) -> tuple[float, int]:
        common = pd.concat([strategy, spy], axis=1).dropna()
        if common.empty:
            return np.nan, 0
        hits = []
        for start in range(0, len(common), int(rebalance_step)):
            window = common.iloc[start: start + int(rebalance_step)]
            if len(window) < 2:
                continue
            s_ret = float((1.0 + window.iloc[:, 0]).prod() - 1.0)
            spy_ret = float((1.0 + window.iloc[:, 1]).prod() - 1.0)
            hits.append(s_ret > spy_ret)
        if not hits:
            return np.nan, 0
        return float(np.mean(hits)), len(hits)


    def compute_scoreboard(returns: pd.DataFrame, turnover: pd.DataFrame, costs: pd.DataFrame, rebalance_step: int) -> pd.DataFrame:
        rows = []
        spy = returns["SPY"] if "SPY" in returns.columns else pd.Series(dtype=float)
        for name in returns.columns:
            r = returns[name].dropna()
            if len(r) < 20:
                continue
            total = float((1.0 + r).prod() - 1.0)
            years = max(len(r) / 252.0, 1 / 252.0)
            cagr = float((1.0 + total) ** (1.0 / years) - 1.0) if total > -1.0 else np.nan
            ann_vol = float(r.std(ddof=0) * np.sqrt(252.0))
            sharpe = float((r.mean() * 252.0) / ann_vol) if ann_vol > 0 else np.nan
            equity = (1.0 + r).cumprod()
            dd = max_drawdown(equity)
            calmar = float(cagr / abs(dd)) if dd < 0 and np.isfinite(cagr) else np.nan
            avg_turnover = float(turnover[name][turnover[name] > 1e-12].mean()) if name in turnover.columns and (turnover[name] > 1e-12).any() else 0.0
            total_cost = float(costs[name].sum()) if name in costs.columns else 0.0
            if name != "SPY" and len(spy):
                common = pd.concat([r.rename("strategy"), spy.rename("spy")], axis=1).dropna()
                spy_total = float((1.0 + common["spy"]).prod() - 1.0) if len(common) else np.nan
                strategy_total = float((1.0 + common["strategy"]).prod() - 1.0) if len(common) else np.nan
                excess_total = strategy_total - spy_total
                hr, hit_periods = hit_rate_vs_spy(common["strategy"], common["spy"], rebalance_step)
            else:
                excess_total = np.nan
                hr = np.nan
                hit_periods = 0
            rows.append({
                "Strategy": name,
                "Total Return": total,
                "CAGR": cagr,
                "Ann. Vol": ann_vol,
                "Sharpe": sharpe,
                "Max Drawdown": dd,
                "Calmar": calmar,
                "Avg Turnover": avg_turnover,
                "Total Cost Drag": total_cost,
                "Excess Total vs SPY": excess_total,
                "Hit Rate vs SPY": hr,
                "Hit Periods": hit_periods,
            })
        return pd.DataFrame(rows).sort_values(["Sharpe", "CAGR"], ascending=False)


    scoreboard_df = compute_scoreboard(
        portfolio_returns,
        portfolio_turnover,
        portfolio_cost,
        rebalance_days.value,
    )


    def compute_material_gates(
        scoreboard: pd.DataFrame,
        min_sharpe_edge: float,
        min_cagr_edge: float,
        min_drawdown_edge: float,
        max_sharpe_gap_vs_simple: float,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        spy_row = scoreboard[scoreboard["Strategy"] == "SPY"]
        if spy_row.empty:
            return pd.DataFrame(), pd.DataFrame()
        spy = spy_row.iloc[0]

        universe_rows = scoreboard[
            scoreboard["Strategy"].str.contains(r"\|", regex=True, na=False)
        ].copy()
        if universe_rows.empty:
            return pd.DataFrame(), pd.DataFrame()

        gate_rows = []
        for _, row in universe_rows.iterrows():
            universe_name, strategy_name = [part.strip() for part in row["Strategy"].split("|", 1)]
            sharpe_edge = float(row["Sharpe"] - spy["Sharpe"]) if pd.notna(row["Sharpe"]) and pd.notna(spy["Sharpe"]) else np.nan
            cagr_edge = float(row["CAGR"] - spy["CAGR"]) if pd.notna(row["CAGR"]) and pd.notna(spy["CAGR"]) else np.nan
            dd_edge = float(row["Max Drawdown"] - spy["Max Drawdown"]) if pd.notna(row["Max Drawdown"]) and pd.notna(spy["Max Drawdown"]) else np.nan
            material_vs_spy = bool(
                (pd.notna(sharpe_edge) and sharpe_edge >= min_sharpe_edge)
                or (pd.notna(cagr_edge) and cagr_edge >= min_cagr_edge)
                or (pd.notna(dd_edge) and dd_edge >= min_drawdown_edge)
            )
            gate_rows.append({
                "Universe": universe_name,
                "Strategy": strategy_name,
                "Sharpe Edge vs SPY": sharpe_edge,
                "CAGR Edge vs SPY": cagr_edge,
                "Drawdown Edge vs SPY": dd_edge,
                "Material vs SPY": material_vs_spy,
            })

        gate_df = pd.DataFrame(gate_rows)
        summary_rows = []
        for universe_name, group in gate_df.groupby("Universe"):
            best = group.sort_values(["Material vs SPY", "Sharpe Edge vs SPY"], ascending=[False, False]).iloc[0]
            ew_row = scoreboard[scoreboard["Strategy"] == f"{universe_name} | Equal Weight"]
            rp_row = scoreboard[scoreboard["Strategy"] == f"{universe_name} | Risk Parity"]
            ew_sharpe = float(ew_row.iloc[0]["Sharpe"]) if not ew_row.empty and pd.notna(ew_row.iloc[0]["Sharpe"]) else np.nan
            rp_sharpe = float(rp_row.iloc[0]["Sharpe"]) if not rp_row.empty and pd.notna(rp_row.iloc[0]["Sharpe"]) else np.nan
            best_sharpe = float(best["Sharpe Edge vs SPY"] + spy["Sharpe"]) if pd.notna(best["Sharpe Edge vs SPY"]) else np.nan
            competitive_vs_simple = True
            if pd.notna(best_sharpe):
                if pd.notna(ew_sharpe) and best_sharpe + max_sharpe_gap_vs_simple < ew_sharpe:
                    competitive_vs_simple = False
                if pd.notna(rp_sharpe) and best_sharpe + max_sharpe_gap_vs_simple < rp_sharpe:
                    competitive_vs_simple = False
            promote = bool(best["Material vs SPY"] and competitive_vs_simple)
            summary_rows.append({
                "Universe": universe_name,
                "Best Strategy": best["Strategy"],
                "Material vs SPY": bool(best["Material vs SPY"]),
                "Competitive vs EW/RP": competitive_vs_simple,
                "Promote to ML": promote,
            })
        summary_df = pd.DataFrame(summary_rows).sort_values(
            ["Promote to ML", "Material vs SPY"],
            ascending=[False, False],
        )
        return gate_df, summary_df

    return compute_material_gates, compute_scoreboard, hit_rate_vs_spy, max_drawdown, scoreboard_df


@app.cell
def _(mo, pd, scoreboard_df):
    _display = scoreboard_df.copy()
    for _col in [
        "Total Return",
        "CAGR",
        "Ann. Vol",
        "Max Drawdown",
        "Avg Turnover",
        "Total Cost Drag",
        "Excess Total vs SPY",
        "Hit Rate vs SPY",
    ]:
        if _col in _display.columns:
            _display[_col] = _display[_col].map(lambda value: "--" if pd.isna(value) else f"{value:.1%}")
    for _col in ["Sharpe", "Calmar"]:
        if _col in _display.columns:
            _display[_col] = _display[_col].map(lambda value: "--" if pd.isna(value) else f"{value:.2f}")

    mo.ui.table(
        _display,
        pagination=True,
        selection=None,
        show_column_summaries=False,
        show_data_types=False,
        show_download=True,
        max_height=480,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 4b. Material Benchmark Gate

    A universe is promoted to the AutoGluon notebook only when it clears a **material** edge versus SPY and stays competitive with its own simple equal-weight / risk-parity baselines.
    """)
    return


@app.cell
def _(
    GATE_MAX_SHARPE_GAP_VS_SIMPLE,
    GATE_MIN_CAGR_EDGE,
    GATE_MIN_DRAWDOWN_EDGE,
    GATE_MIN_SHARPE_EDGE,
    compute_material_gates,
    mo,
    pd,
    scoreboard_df,
):
    gate_detail_df, gate_summary_df = compute_material_gates(
        scoreboard_df,
        GATE_MIN_SHARPE_EDGE,
        GATE_MIN_CAGR_EDGE,
        GATE_MIN_DRAWDOWN_EDGE,
        GATE_MAX_SHARPE_GAP_VS_SIMPLE,
    )
    if gate_summary_df.empty:
        _gate_view = mo.md("Need SPY and at least one universe strategy before running the material gate.")
    else:
        _summary_display = gate_summary_df.copy()
        for _col in ["Material vs SPY", "Competitive vs EW/RP", "Promote to ML"]:
            _summary_display[_col] = _summary_display[_col].map(lambda value: "PASS" if value else "FAIL")
        _detail_display = gate_detail_df.copy()
        for _col in ["Sharpe Edge vs SPY", "CAGR Edge vs SPY", "Drawdown Edge vs SPY"]:
            if _col in _detail_display.columns:
                _detail_display[_col] = _detail_display[_col].map(
                    lambda value: "--" if pd.isna(value) else f"{value:.2f}" if "Sharpe" in _col else f"{value:.1%}"
                )
        _detail_display["Material vs SPY"] = _detail_display["Material vs SPY"].map(
            lambda value: "PASS" if value else "FAIL"
        )
        _gate_view = mo.vstack([
            mo.md(f"""
            ### Material Benchmark Gate

            A universe earns promotion only if it clears SPY by at least **{GATE_MIN_SHARPE_EDGE:.2f} Sharpe**, **{GATE_MIN_CAGR_EDGE:.0%} CAGR**, or **{GATE_MIN_DRAWDOWN_EDGE:.0%} drawdown**, and stays within **{GATE_MAX_SHARPE_GAP_VS_SIMPLE:.2f} Sharpe** of its own equal-weight / risk-parity baselines.
            """),
            mo.ui.table(
                _summary_display,
                pagination=False,
                selection=None,
                show_column_summaries=False,
                show_data_types=False,
                show_download=True,
                max_height=260,
            ),
            mo.ui.table(
                _detail_display,
                pagination=True,
                selection=None,
                show_column_summaries=False,
                show_data_types=False,
                show_download=True,
                max_height=320,
            ),
        ])
    _gate_view
    return gate_detail_df, gate_summary_df


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 5. Visual Check

    A table can hide the path. The chart shows whether the result came from steady improvement, one lucky burst, or just tracking SPY.
    """)
    return


@app.cell
def _(alt, equity_curves, pd, scoreboard_df):
    _top_names = ["SPY"]
    _ranked = [
        name for name in scoreboard_df["Strategy"].head(8).to_list()
        if name not in _top_names
    ]
    _top_names.extend(_ranked[:7])
    _chart_df = (
        equity_curves.loc[:, [name for name in _top_names if name in equity_curves.columns]]
        .reset_index()
        .melt("date", var_name="Strategy", value_name="Equity")
        .dropna()
    )
    _chart_df["date"] = pd.to_datetime(_chart_df["date"])
    _chart_df = (
        _chart_df
        .set_index("date")
        .groupby("Strategy", group_keys=False)
        .resample("W")
        .last()
        .drop(columns=["Strategy"], errors="ignore")
        .reset_index()
    )
    equity_chart = (
        alt.Chart(_chart_df)
        .mark_line()
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("Equity:Q", title="Growth of $100"),
            color=alt.Color("Strategy:N"),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip("Strategy:N"),
                alt.Tooltip("Equity:Q", format=".2f"),
            ],
        )
        .properties(height=420, title="Top strategies by Sharpe, with SPY included")
        .interactive()
    )
    equity_chart
    return (equity_chart,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 6. Crisis Windows

    This section asks a different question:

    **Which universes helped when SPY was under stress?**

    A strategy does not need to beat SPY every month to be useful. Sometimes the value is losing less during bad regimes.
    """)
    return


@app.cell
def _(build_crisis_windows, end_date, mo, pd, portfolio_returns, scoreboard_df):
    crisis_windows = build_crisis_windows(end_date.value)
    _candidates = ["SPY"]
    _candidates.extend([
        name for name in scoreboard_df["Strategy"].head(7).to_list()
        if name != "SPY"
    ])
    _rows = []
    for _period_name, (_start, _end) in crisis_windows.items():
        _window = portfolio_returns.loc[
            (portfolio_returns.index >= _start) & (portfolio_returns.index <= _end),
            [name for name in _candidates if name in portfolio_returns.columns],
        ]
        if len(_window) < 2:
            continue
        for _strategy in _window.columns:
            _ret = float((1.0 + _window[_strategy].fillna(0.0)).prod() - 1.0)
            _rows.append({
                "Window": _period_name,
                "Strategy": _strategy,
                "Return": _ret,
                "Start": _window.index.min().date(),
                "End": _window.index.max().date(),
            })
    crisis_df = pd.DataFrame(_rows)
    _display = crisis_df.copy()
    if not _display.empty:
        _display["Return"] = _display["Return"].map(lambda value: f"{value:.1%}")

    mo.ui.table(
        _display,
        pagination=False,
        selection=None,
        show_column_summaries=False,
        show_data_types=False,
        show_download=True,
        max_height=360,
    )
    return (crisis_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 7. Verdict

    Use this notebook as a gate:

    - If a universe cannot beat or improve simple benchmarks with simple allocation rules, do not spend GPU time on ML yet.
    - If a universe improves drawdown or Sharpe, it may deserve a second notebook pass with AutoGluon.
    - If the best result comes from a newer ETF, check whether the test window became too short.
    """)
    return


@app.cell
def _(gate_summary_df, mo, pd, scoreboard_df, selected_universes):
    _spy_row = scoreboard_df[scoreboard_df["Strategy"] == "SPY"]
    _non_spy = scoreboard_df[scoreboard_df["Strategy"] != "SPY"].copy()
    if _spy_row.empty or _non_spy.empty:
        _verdict = "Need SPY and at least one non-SPY strategy before judging."
    else:
        _spy = _spy_row.iloc[0]
        _best_sharpe = _non_spy.sort_values("Sharpe", ascending=False).iloc[0]
        _best_cagr = _non_spy.sort_values("CAGR", ascending=False).iloc[0]
        _best_drawdown = _non_spy.sort_values("Max Drawdown", ascending=False).iloc[0]
        _promoted = gate_summary_df[gate_summary_df["Promote to ML"]] if not gate_summary_df.empty else pd.DataFrame()
        if _promoted.empty:
            _promotion_note = "No universe cleared the material gate yet. Do not send this run to AutoGluon."
        else:
            _names = ", ".join(_promoted["Universe"].tolist())
            _promotion_note = f"Promote to ML: **{_names}**."
        _verdict = f"""
        Best Sharpe: **{_best_sharpe['Strategy']}** at **{_best_sharpe['Sharpe']:.2f}** versus SPY at **{_spy['Sharpe']:.2f}**.

        Best CAGR: **{_best_cagr['Strategy']}** at **{_best_cagr['CAGR']:.1%}** versus SPY at **{_spy['CAGR']:.1%}**.

        Best drawdown control: **{_best_drawdown['Strategy']}** at **{_best_drawdown['Max Drawdown']:.1%}** versus SPY at **{_spy['Max Drawdown']:.1%}**.

        {_promotion_note}
        """

    mo.md(f"""
    ### Research Verdict

    {_verdict}
    """)
    return


@app.cell
def _(UNIVERSE_PRESETS, gate_summary_df, mo, selected_universes):
    if gate_summary_df.empty or not gate_summary_df["Promote to ML"].any():
        _handoff = "No promoted universe yet. Re-run with different presets or dates after reviewing the gate table."
        _ticker_copy = None
    else:
        _lines = []
        _copy_lines = []
        for _, row in gate_summary_df[gate_summary_df["Promote to ML"]].iterrows():
            _tickers = selected_universes.get(row["Universe"], UNIVERSE_PRESETS.get(row["Universe"], []))
            if _tickers:
                _ticker_str = ", ".join(_tickers)
                _lines.append(f"- **{row['Universe']}** (`{_ticker_str}`) via **{row['Best Strategy']}**")
                _copy_lines.append(_ticker_str)
        _handoff = "\n".join(_lines) if _lines else "Promoted universes did not resolve to a ticker list."
        _ticker_copy = mo.ui.text(
            value=_copy_lines[0] if len(_copy_lines) == 1 else " | ".join(_copy_lines),
            label="Copy into AutoGluon Tickers control",
            full_width=True,
        ) if _copy_lines else None

    mo.vstack([
        mo.md(f"""
        ## 8. Promote to AutoGluon

        Copy a promoted ticker string into the [AutoGluon Portfolio Allocation Learning Lab](../autogluon-allocation-learning-lab/notebook.py) **Tickers** control.

        {_handoff}
        """),
        *([_ticker_copy] if _ticker_copy is not None else []),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 9. Student Exercises and Next Experiments

    1. Turn off "Scan all universe presets" and inspect one universe at a time.
    2. Raise the minimum history requirement and see which newer ETFs disappear.
    3. Compare equal weight and risk parity during 2022.
    4. Add one new ETF, then check its first valid date before trusting results.
    5. Only after the benchmark scan finds a promising universe, feed that universe into the AutoGluon allocation notebook.

    Core lesson: ML cannot rescue a weak opportunity set. First find assets that behave differently enough to matter.
    """)
    return


if __name__ == "__main__":
    app.run()
