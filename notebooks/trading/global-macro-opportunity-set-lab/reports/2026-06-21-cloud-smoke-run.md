# Cloud Smoke Run

Date: 2026-06-21

Cloud session: https://sb-2d2c1af36d62b595.sb.molab.run/

Source of truth: local `notebook.py`

## Verification

- Local `python3 -m py_compile notebook.py`: pass
- Local `uvx marimo@latest check --strict notebook.py`: pass
- Cloud cell errors: none
- Cloud saved `/marimo/notebook.py` `py_compile`: pass
- Cloud saved `/marimo/notebook.py` `marimo check --strict`: pass
- Secret scan: token value not present in project files

## Data Window

Adjusted yfinance prices loaded from 2018-01-02 through 2026-06-18.

Newer ETF inception dates were preserved:

- DBMF first valid date: 2019-05-08
- KMLM first valid date: 2020-12-02
- CTA first valid date: 2022-03-08

No backward-fill before inception was used.

## First Scoreboard Read

| Strategy | Total Return | CAGR | Sharpe | Max Drawdown | Hit Rate vs SPY |
| --- | ---: | ---: | ---: | ---: | ---: |
| Managed futures sleeve - Equal Weight | 125.05% | 10.09% | 1.25 | -13.83% | 39.22% |
| Trend alternatives research - Equal Weight | 122.50% | 9.94% | 1.20 | -16.50% | 43.14% |
| Managed futures sleeve - Risk Parity | 105.78% | 8.93% | 1.17 | -13.24% | 40.20% |
| SPY core - Equal Weight | 183.46% | 13.14% | 0.99 | -25.55% | 42.16% |
| 60/40 SPY/IEF | 116.66% | 9.59% | 0.85 | -21.02% | 34.31% |
| SPY | 216.48% | 14.62% | 0.81 | -33.72% | n/a |

## Interpretation

This is the result we hoped to see from an opportunity-set lab:

- SPY still dominated raw total return and CAGR.
- Managed-futures sleeves did not beat SPY on raw return.
- Managed-futures sleeves materially improved Sharpe and drawdown.
- That means the broader universe may be useful for risk control, crisis behavior, or as a sleeve in a later SPY-plus-alpha system.

The next notebook step should not be AutoGluon yet. First add a material benchmark gate:

- beat SPY on Sharpe by at least 0.20, or
- reduce max drawdown by at least 10 percentage points, or
- beat 60/40 on both Sharpe and CAGR.

Only universes that pass a clear gate should be sent to the ML notebook.
