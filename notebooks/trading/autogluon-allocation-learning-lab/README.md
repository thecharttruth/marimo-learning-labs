# AutoGluon Portfolio Allocation Learning Lab

This marimo notebook teaches a complete machine-learning trading research workflow:

- build ETF price features
- train AutoGluon tabular return forecasters
- use walk-forward validation to reduce lookahead bias
- convert return forecasts into a SPY-aware overlay allocation
- compare ML-driven allocation against equal-weight, risk-parity, SPY, and 60/40 SPY/IEF baselines
- inspect turnover, optional bps cost drag, a hard SPY overlay deployment hurdle, retraining diagnostics, and model usefulness diagnostics
- review results with interactive TradingView-style charts

## Notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thecharttruth/marimo-learning-labs/blob/main/notebooks/trading/autogluon-allocation-learning-lab/google-colab-autogluon-portfolio-allocation-learning-lab.ipynb)

- Local source: `notebook.py`
- Google Colab notebook: `google-colab-autogluon-portfolio-allocation-learning-lab.ipynb`
- Cloud notebook: https://molab.marimo.io/notebooks/nb_dqdkCXtD93dSyuxLXgnQJD
- Latest cloud save verified: 2026-06-21
- Latest local verification: 2026-06-21 (`py_compile`, `marimo check --strict`, cloud strict check, secret/random-split scan)
- Current known MoLab catalog title: `green-anteater`
- Intended title: `AutoGluon Portfolio Allocation Learning Lab`
- Local repair note: the notebook uses a normal marimo import cell, not `with app.setup(...)`. A later AI pass introduced an `app.setup` block that `marimo check` flagged; it has been reverted locally.
- Cloud sync note: local `notebook.py` is the durable source of truth. The 2026-06-21 MoLab pair session was updated through code-mode and the saved cloud artifact contains the latest cache, OOS-end, and chart-stage fixes. MoLab may normalize script dependency metadata and wrapper ordering, so compare cell bodies/markers rather than raw file hashes when local is authoritative.

## Execution Model

- Treat this local folder as the durable source of truth for edits and docs.
- Treat MoLab as the normal execution and sharing surface.
- Use local execution only if the notebook becomes too large, slow, or cloud-hostile enough to justify a local runtime setup.
- Local verification can stay limited to syntax, secret scans, and file consistency unless a local run is explicitly needed.

## Current Status

The notebook has been converted into a learning notebook and includes:

- AutoGluon lessons and trading caveats
- high-school-friendly plain-English map and glossary for portfolio, feature, target, backtest, and lookahead bias
- corrected forward-return target construction
- purged and embargoed walk-forward validation; no random train/test split
- explicit research, validation, and final out-of-sample calendar controls
- a final OOS/report-end control so the notebook can score a fixed period, such as 2024-2025, while still downloading later prices to measure the last forward-return outcomes
- one-trading-day execution lag to avoid same-close entry assumptions
- configurable embargo gap after each training window, with training-label purge for outcomes not known at the training cutoff
- configurable feature depth with expanded technical, breadth, relative-market, and bounded pair factors
- vectorized Polars feature construction so larger factor sets build with fewer table passes
- a shared cross-asset AutoGluon forecaster that trains one model per rebalance split
- safer AutoGluon hyperparameters for marimo Cloud plus an Auto GPU-rich model mode that uses CUDA-backed AutoGluon backends when available
- AutoGluon backend diagnostics showing requested mode, CUDA availability, GPU-rich fits, CPU fallback fits, available rich models, and trained model names
- conservative ML allocation that caps raw forecasts and keeps the older ML-MC tilt visible as a research comparison
- SPY-aware overlay strategy where SPY is the default portfolio and ML can only move a capped sleeve into defensive assets such as TLT and GLD when the SPY forecast and cross-asset breadth weaken
- optional bounded no-lookahead parameter optimizer that scores candidates inside the validation window using the same purged/embargoed split builder
- SPY-aware optimizer scoring that rewards the overlay's excess return, excess Sharpe, drawdown edge, and hit rate versus SPY before penalizing turnover and concentration
- standalone SPY and 60/40 SPY/IEF benchmarks
- trading cost sensitivity with a 0 bps default for commission-free ETF platforms
- performance summary table with CAGR, volatility, Sharpe, drawdown, Calmar, turnover, cost drag, and hit rate vs SPY
- final-OOS gate with a hard SPY hurdle: the SPY+ML overlay must beat SPY on CAGR and Sharpe without worse max drawdown, otherwise the notebook says the ML complexity did not earn its place
- simplified teaching copy for AutoGluon model training, walk-forward validation, allocation weights, backtest metrics, and diagnostics
- prediction-window masking so selected splits do not score returns beyond their report windows
- MoLab GPU/runtime guidance and a model-mode selector for large feature and optimizer workloads
- interactive TradingView-style chart widgets, including a full downloaded-history context chart plus a selected report-window strategy chart where the SPY+ML overlay changes color by research, validation, final-OOS, and after-selected-OOS stage
- clearer SPY+ML overlay allocation weight readout
- a normal marimo import/setup cell with all runtime imports and shared constants
- `mo.persistent_cache` wrapping the yfinance download with ticker/date-aware cache keys and an explicit active-date filter, so stale broad caches cannot leak past the selected controls
- complete PEP 723 script header listing all runtime dependencies for one-pass `uv run` installs, including `autogluon.tabular[all]` so richer tabular backends are available on MoLab hardware

## Next Work

The latest cloud analysis tested a two-year final OOS report window from 2024-01-01 through 2025-12-31, using data through 2026-06-18 only to measure late-2025 forward outcomes. Auto GPU-rich mode used CUDA-backed AutoGluon fits. ML-MC returned 44.58% with Sharpe 1.51 and max drawdown -15.04%, but it did not beat SPY, equal weight, or risk parity over this window. See `reports/2026-06-21-two-year-final-oos.md`.

The first SPY+ML overlay smoke run on the active 2026 final-OOS slice beat SPY on CAGR, Sharpe, and drawdown, but equal weight still slightly beat the overlay and the SPY-relative hit rate was weak. See `reports/2026-06-21-spy-overlay-redesign.md`.

A longer 2024-2026 final-OOS run covered 40 rebalance splits from 2024-01-15 through 2026-05-03. The overlay barely beat SPY on CAGR and Sharpe but had effectively the same drawdown, weak SPY-relative hit rate, and weaker risk-adjusted performance than equal weight and risk parity. See `reports/2026-06-21-long-oos-2024-2026.md`.

Next improve only on pre-2024 validation data, then rerun the same frozen 2024-2026 final-OOS window unchanged. The 2026-06-24 pass aligned both notebooks on inception-safe loading, material benchmark gates, EW/RP competitiveness checks, run presets, trading-row walk-forward splits, overlay no-trade bands, and an explicit macro-to-ML handoff.

See `PLAN.md`.
