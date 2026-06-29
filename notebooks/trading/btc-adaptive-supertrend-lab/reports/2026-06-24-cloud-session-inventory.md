# Cloud session inventory (pre-pull)

MoLab URL: https://sb-8e1f27f556b2f563.sb.molab.run/

Observed on 2026-06-24 via marimo-pair cell inspection.

## Cell themes

- Coinbase kline fetch (`fetch_coinbase_klines`, `new_fetch_coinbase_klines`)
- Ehlers homodyne dominant-cycle detector
- `modern_adaptive_supertrend` implementation
- Multiple duplicate supertrend runs and Altair chart cells
- yfinance install / BTC-USD long-history cell using undefined `_yf`

## Errors to fix after pull

| Issue | Detail |
|-------|--------|
| Graph: multiply-defined | `alt`, `subprocess`, `sys`, `warnings`, chart locals (`dir_txt`, `flips`, `st_df_new`, etc.) |
| Runtime | `NameError: name '_yf' is not defined` in yfinance download cell |
| Structure | Duplicate experimental cells from iterative pairing (cells 05–22) |

## Action

Run `scripts/pull-molab-notebook.sh`, then dedupe cells and re-run `marimo check --strict`.
