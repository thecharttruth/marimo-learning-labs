# SPY-Aware Overlay Redesign Smoke Run

Date: 2026-06-21

Source of truth: local `notebook.py`

Cloud execution surface: MoLab paired runtime

## What Changed

The deployable ML strategy is now `SPY+ML Overlay`.

- Default position is 100% SPY.
- ML does not get to replace SPY with a broad optimized basket by default.
- ML can only move a capped defensive sleeve into assets such as TLT and GLD when the SPY forecast and cross-asset breadth weaken.
- The optimizer score, backtest, allocation readout, recent-health diagnostic, charts, and final OOS gate now evaluate the overlay against SPY.

## Live Smoke Run

Active protocol: 2026 final OOS

Selected report window: 2026-01-19 through 2026-05-03

Rebalance splits: 5

Trading cost: 0.0 bps

| Strategy | Total Return | CAGR | Sharpe | Max Drawdown | Hit Rate vs SPY |
| --- | ---: | ---: | ---: | ---: | ---: |
| SPY+ML Overlay | 7.12% | 27.22% | 1.79 | -8.33% | 20% |
| SPY | 6.65% | 25.26% | 1.65 | -8.89% | n/a |
| ML-MC | 5.88% | 22.13% | 1.46 | -7.85% | 60% |
| Equal Weight | 7.25% | 27.76% | 1.80 | -7.05% | 60% |
| Risk Parity | 6.51% | 24.71% | 1.70 | -7.10% | 60% |
| 60/40 SPY/IEF | 4.18% | 15.42% | 1.58 | -6.00% | 40% |

## Gate Result

Gate verdict: `OVERLAY EARNED COMPLEXITY` for this selected smoke window.

The overlay beat SPY on CAGR, Sharpe, and max drawdown in the selected 2026 final-OOS slice.

## Caveat

This is not yet a final research win. The window is short, the hit rate versus SPY is only 20%, and equal weight still slightly beat the overlay on return, Sharpe, and drawdown. The next real test is the fixed two-year 2024-2025 final-OOS window plus robustness checks across feature depth, universe choice, and defensive-sleeve rules.
