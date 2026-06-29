# Plan: BTC Adaptive Supertrend Lab

## Goal

Keep a durable local backup of the MoLab BTC supertrend research notebook and clean it into a maintainable teaching notebook.

## Scope

Include:

- Coinbase daily BTC candles
- adaptive supertrend indicator
- Altair visualization of trend and flips
- optional yfinance long-history path

Exclude for now:

- live trading
- broker keys in the repo

## Ordered Steps

1. Create local project folder and registry entry. Done.
2. Pull cloud `notebook.py` with `scripts/pull-molab-notebook.sh`.
3. Fix marimo graph errors (duplicate public defs, missing `_yf`).
4. Run `marimo check --strict` and a MoLab smoke run.
5. Save a report under `reports/`.

## Stop Conditions

- Tokens or secrets would be written to disk.
- Cloud URL does not match `btc-adaptive-supertrend-lab` in NOTEBOOKS.yaml.

## Verification Steps

- `uvx marimo@latest check --strict notebook.py`
- No multiply-defined public variables across cells.
- Cloud and local cell bodies match after pull.
