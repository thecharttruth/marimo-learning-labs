# Plan: No-Lookahead Parameter Optimizer

## Status

Implemented in local `notebook.py` and saved to MoLab on 2026-06-20, including configurable feature depth, expanded factor sets, bounded pair factors, workload-based MoLab GPU guidance, SPY and 60/40 benchmarks, cost sensitivity, one-trading-day execution lag, out-of-sample gate checks, and prediction diagnostics. Local `notebook.py` is now the durable source of truth because cloud sessions can disconnect. Treat MoLab as the execution and sharing surface, and push local to cloud when a live pair session is available.

A follow-up pass on 2026-06-20 wrapped the yfinance download in `mo.persistent_cache(name="market_data_v1")`, completed the PEP 723 script header so `uv run notebook.py` can install runtime dependencies, and restored Polars rolling standard deviation calls to the cloud-compatible `rolling_std` API. A later `with app.setup(hide_code=True)` rewrite was rejected by `marimo check` and has been reverted to a normal marimo import cell.

The 2026-06-21 analysis used the live MoLab runtime before the sandbox terminated. A depth-2 validation optimizer selected 10-day forward returns, 10-day rebalancing, a 3-year training window, 35% max weight, and 200 MC draws. Applying that frozen setting to all eligible 2026 final-OOS splits available through 2026-05-28 produced stronger return and Sharpe than SPY, equal weight, risk parity, and 60/40, but drawdown was slightly worse than SPY and turnover remained high.

After the sandbox terminated, the local notebook received a GPU-aware AutoGluon model-mode repair: Auto GPU-rich if available, Fast CPU trees, and GPU rich models. A follow-up autonomous pass made the GPU path inspectable by recording per-asset backend metadata, adding an AutoGluon backend diagnostics table, installing `autogluon.tabular[all]` for one-pass cloud runtime setup, and building the rich model plan only from optional backends available in the runtime.

A 2026-06-21 validation-safety pass added a configurable embargo gap after each training window and routes both the optimizer and final report through the same purged, embargoed walk-forward split builder. The notebook still purges the final forward-horizon plus execution-lag rows from every training window so labels whose outcomes would not be known at the training cutoff are excluded. No random train/test split is used.

A teaching pass on 2026-06-21 added a plain-English map and glossary, simplified the controls, feature engineering, validation, AutoGluon, allocation, backtest, and diagnostics explanations, and kept the notebook focused on AutoGluon for portfolio allocation rather than generic AutoML.

The fresh 2026-06-21 MoLab pair session was updated from the local notebook through marimo code-mode. Cloud and local cell/config hashes matched (`102cace278f3b5efb510aadba76ae7d2d6ce90d2d677cdbcafbb0cfdccb51efd`), and cloud-side `py_compile` plus `marimo check --strict` passed. Raw file hashes can differ because MoLab may normalize the PEP 723 dependency header.

A 2026-06-21 speed and cleanup pass added vectorized Polars feature construction and made the shared cross-asset AutoGluon forecaster the only forecast path. The notebook now trains one long-form model per rebalance split with a ticker indicator. The change keeps the notebook cleaner while preserving chronological splits, purge/embargo rules, and next-trading-day execution timing.

A 2026-06-21 strategy-stability pass changed the ML allocation from a full-strength mean-variance bet to a conservative forecast tilt. Raw AutoGluon forecasts are capped to a horizon-volatility scale, optimized with higher risk aversion, and blended as a modest tilt around a diversified equal-weight/risk-parity anchor. This is meant to reduce concentration, turnover, and drawdown when forecasts are noisy.

A 2026-06-21 visualization pass raised the split cap so the existing `All walk-forward splits` protocol can show the full available walk-forward history, and changed the equity chart so the ML-MC line is colored by research, validation, and final-OOS stage.

A 2026-06-21 two-year final-OOS test found and fixed a cache/window issue: the yfinance cache could reuse a broader data range than the active end-date control. The notebook now uses ticker/date-aware cache keys plus an explicit active-date filter. It also separates data end date from final OOS report end, so a fixed 2024-2025 report can use later data only to measure late-2025 forward outcomes. The cloud test ran 34 final-OOS rebalance windows from 2024-01-15 through 2025-12-28 with Auto GPU-rich mode and CUDA-backed AutoGluon fits. ML-MC returned 44.58% with Sharpe 1.51 and max drawdown -15.04%, but SPY, equal weight, and risk parity were stronger over this window. Results are saved in `reports/2026-06-21-two-year-final-oos.md`.

A 2026-06-21 benchmark-discipline pass made SPY the hard hurdle. That first version pointed the hurdle at ML-MC; the later overlay redesign superseded it by making SPY+ML Overlay the deployable strategy and the gate target.

A 2026-06-21 overlay redesign made SPY the default strategy rather than just a benchmark. The notebook now adds `SPY+ML Overlay`, which starts at 100% SPY and lets ML move only a capped sleeve into defensive assets such as TLT and GLD when the SPY forecast and cross-asset breadth weaken. Optimizer scoring, allocation charts, recent-health diagnostics, and the final OOS gate now evaluate the overlay versus SPY.

The first overlay smoke run on the active 2026 final-OOS slice covered 2026-01-19 through 2026-05-03 across 5 rebalance splits. SPY+ML Overlay returned 7.12% with 27.22% CAGR, 1.79 Sharpe, and -8.33% max drawdown versus SPY at 6.65%, 25.26% CAGR, 1.65 Sharpe, and -8.89% max drawdown. Equal weight still slightly beat the overlay, so the result is a smoke-test pass rather than a robust research win. See `reports/2026-06-21-spy-overlay-redesign.md`.

A longer 2024-2026 final-OOS run moved research/tuning to 2022-12-31, validation to 2023-12-31, and final OOS to 2024-01-01 through 2026-06-21. The cloud run covered 40 rebalance splits from 2024-01-15 through 2026-05-03 with GPU-rich AutoGluon fits. SPY+ML Overlay returned 56.34% with 21.59% CAGR, 1.31 Sharpe, and -18.76% max drawdown versus SPY at 55.91%, 21.45% CAGR, 1.29 Sharpe, and -18.76% max drawdown. The edge was too small to justify the complexity, and equal weight/risk parity were stronger. See `reports/2026-06-21-long-oos-2024-2026.md`.

## Goal

Improve the Configuration & Controls section so students can run a background-safe optimizer that suggests robust strategy parameters without curve fitting or lookahead bias.

## Scope

Tune strategy parameters around the existing AutoGluon forecasting workflow:

- forward return horizon
- rebalance frequency
- training window
- max single-asset weight
- Monte Carlo draw count
- optional shorting setting, if runtime remains practical
- feature set depth and pair-factor budget as manual research controls
- standalone benchmark comparison against SPY and 60/40 SPY/IEF
- hard SPY deployment hurdle: the SPY+ML overlay must beat SPY on CAGR and Sharpe without worse max drawdown before the notebook treats ML as useful
- realistic cost sensitivity with a 0 bps default for commission-free ETF platforms
- strict execution timing that avoids same-close entries
- purged/embargoed walk-forward validation; no random train/test split
- AutoGluon model mode: auto GPU-rich, fast CPU trees, or forced GPU-rich backends, with backend diagnostics for CUDA visibility, optional rich models, trained models, and CPU fallbacks
- AutoGluon forecast structure: shared cross-asset model only, to keep the learning path fast and simple
- allocation behavior: SPY is the default deployable portfolio; ML only earns a capped defensive overlay when forecasts justify reducing risk
- high-school-friendly teaching explanations for the AutoGluon portfolio allocation workflow

Do not tune on future periods that are later reported as out-of-sample performance.

## Ordered Steps

1. Add a teaching note explaining manual controls vs optimized controls. Done.
2. Add an optimizer toggle or button in the controls section. Done.
3. Implement a small candidate grid with conservative defaults. Done.
4. Add nested walk-forward evaluation. Done:
   - inner window chooses parameters
   - outer future window reports performance
5. Score candidates with a robust objective. Done:
   - Sharpe reward
   - max drawdown penalty
   - turnover penalty
   - concentration penalty
6. Display selected parameters with why they were selected. Done.
7. Let the user apply suggested parameters or keep manual controls. Done.
8. Keep cloud runtime bounded with a max-candidate and max-split budget. Done.
9. Add SPY and 60/40 SPY/IEF benchmarks to metrics and charts. Done.
10. Add turnover and optional bps cost drag with a 0 bps default. Done.
11. Add one-trading-day execution lag and next-bar forward-return targets. Done.
12. Add performance summary, out-of-sample gate, and prediction diagnostics. Done.

## Stop Conditions

- Runtime becomes too slow for marimo Cloud.
- The optimizer needs unavailable model packages.
- Any implementation would evaluate a parameter set on the same future data used to select it.
- Any backtest path would require entering at the same close used to form the signal.
- The notebook cannot explain the validation split clearly to a student.

## Verification Steps

- Confirm no cell errors in the live marimo runtime.
- Confirm the optimizer uses only past data for selection.
- Confirm selected parameters are frozen before reporting future performance.
- Confirm validation and final report splits are chronological, purged, and embargoed.
- Confirm weights are shifted by one trading day before returns are applied.
- Confirm SPY and 60/40 benchmark columns are present when yfinance data is available.
- Confirm the optimizer score uses only validation-window SPY-relative metrics when choosing settings.
- Confirm the final OOS gate fails the SPY+ML overlay when SPY beats it on CAGR, Sharpe, or max drawdown.
- Confirm the default trading cost remains 0 bps and any nonzero bps cost is visible in the metrics table.
- Confirm local `notebook.py` remains the source of truth after edits.
- Confirm cloud cell/config hash matches local after pushing to a fresh MoLab pair session.
- Confirm no secrets are written.
- Keep local verification to syntax, secret scans, and file consistency unless a local runtime is explicitly needed.
- Confirm the normal import/setup cell runs before dependent cells and contains all module imports.
- Confirm `mo.persistent_cache` invalidates when ticker or date widgets change and hits on kernel restart with unchanged inputs.
- Confirm the PEP 723 header lists all runtime dependencies so `uv run notebook.py` installs everything in one pass.
- Confirm `autogluon_backend_diagnostics_df` shows GPU-rich fits when MoLab GPU is attached; if it shows CPU fallback, inspect available rich models and package/runtime status before trusting performance differences.
- Confirm the shared cross-asset model estimates one AutoGluon fit per split.
- Confirm vectorized feature construction produces the same columns and remains lookahead-safe: lagged values use `shift`, rolling values use current-or-past rows, and future-return targets stay excluded from feature columns.
- Confirm the SPY+ML overlay turnover and drawdown are reasonable versus SPY and the older ML-MC research comparison on the same selected split set.
- Confirm the equity chart colors the SPY+ML overlay by research, validation, and final-OOS stages when multiple stages are selected.
- Confirm the final OOS report-end control excludes post-report splits while allowing later downloaded data to measure forward outcomes.
- Confirm yfinance cache keys include ticker and date controls, and the loaded data is filtered to the active requested range even when a broad cache exists.
- Confirm the equity chart distinguishes after-selected-OOS points from selected final-OOS points in all-splits mode.

## Expected Artifacts

- Updated `notebook.py`
- Updated `README.md` if the user-facing workflow changes
- `reports/2026-06-21-two-year-final-oos.md`
- `reports/2026-06-21-spy-overlay-redesign.md`
- `reports/2026-06-21-long-oos-2024-2026.md`
- Optional `reports/optimizer_validation.md`
- Optional blog notes in `blog/draft.md`

## Next Recommended Task

Re-run the frozen 2024-2026 final-OOS window on MoLab after the 2026-06-24 gate and inception-safety pass. Confirm whether any universe promoted by the Global Macro lab can clear the new material overlay hurdle.
