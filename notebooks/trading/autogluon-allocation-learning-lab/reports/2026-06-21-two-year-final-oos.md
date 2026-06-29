# Two-Year Final OOS Test

Run date: 2026-06-21

## Setup

- Data downloaded: 2018-01-02 through 2026-06-18
- Scored final OOS report window: 2024-01-01 through 2025-12-31
- First selected prediction window: 2024-01-15 through 2024-02-04
- Last selected prediction window: 2025-12-08 through 2025-12-28
- Rebalance windows: 34
- Allocation rows: 340
- Backtest rows: 490
- Feature count: 358
- Trading cost: 0 bps
- AutoGluon mode: Auto GPU rich if available
- CUDA visible: true
- AutoGluon backend: gpu_rich
- Rich model plan: NN_TORCH, XGB, RF, XT

The notebook used data after 2025 only to measure late-2025 forward-return outcomes. The report split selector excluded windows after the selected OOS end date.

## Performance

| Strategy | Total Return | CAGR | Ann. Vol | Sharpe | Max Drawdown | Avg Rebalance Turnover | Hit Rate vs SPY |
|---|---:|---:|---:|---:|---:|---:|---:|
| ML-MC | 44.58% | 20.88% | 13.11% | 1.51 | -15.04% | 0.22 | 44.12% |
| ML | 44.99% | 21.05% | 13.07% | 1.53 | -14.92% | 0.22 | 47.06% |
| Equal Weight | 47.57% | 22.15% | 13.42% | 1.56 | -14.72% | 1.00 | 44.12% |
| Risk Parity | 46.97% | 21.90% | 12.80% | 1.61 | -13.73% | 0.04 | 44.12% |
| SPY | 48.94% | 22.74% | 16.48% | 1.33 | -18.76% | 1.00 | n/a |
| 60/40 SPY/IEF | 32.28% | 15.47% | 10.13% | 1.47 | -10.61% | 1.00 | 32.35% |

## Prediction Quality

- Prediction rows: 340
- Mean predicted horizon return: 0.68%
- Mean realized horizon return: 0.95%
- Mean absolute forecast error: 3.24%
- Direction hit rate: 50.29%

Best directional ticker: GLD at 67.65%.

Weakest directional tickers: IWM and VGT at 41.18%.

## Interpretation

The two-year OOS run is healthy as a notebook/runtime test: no cell errors, GPU-rich AutoGluon fits were used, and the report window stayed inside 2024-2025.

As a strategy result, ML-MC is improved versus the earlier unstable optimizer behavior, but it still does not beat the simple allocation baselines over this window. SPY has the highest total return and CAGR, while risk parity has the best Sharpe and lowest drawdown among the allocation strategies. The model forecast signal is only slightly better than a coin flip directionally, so the current ML layer should be treated as educational and exploratory rather than proven alpha.

Next useful tests:

- run the same two-year OOS window with feature depth 1 vs 2 vs 3
- test a smaller, less sector-heavy universe
- tune the conservative tilt strength on validation only, then rerun this fixed final OOS window
- add a no-trade band or turnover penalty to make the ML allocation compete more directly with risk parity
