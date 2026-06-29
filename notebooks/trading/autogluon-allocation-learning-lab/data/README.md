# Data Notes

This notebook currently uses `yfinance` adjusted ETF price data for education and prototyping.

## Current Data Source

- Provider: Yahoo Finance via `yfinance`
- Data type: adjusted daily ETF prices
- Private credentials required: no

## Caveats

- `yfinance` is convenient but not institutional-grade market data.
- Data can be revised, delayed, missing, or inconsistent across tickers.
- A serious trading workflow should pin data snapshots and document survivorship rules.

## Storage Rule

Do not commit large raw data files or private datasets into this project unless explicitly intended. Use this file to document how data was fetched and refreshed.
