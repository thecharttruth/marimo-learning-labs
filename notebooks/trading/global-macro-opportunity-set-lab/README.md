# Global Macro Opportunity Set Lab

This marimo notebook tests whether broadening beyond SPY creates a better opportunity set before adding machine learning.

The first version is intentionally benchmark-only:

- universe presets for SPY core, global macro, rates/inflation, managed futures, and trend alternatives
- inception-safe yfinance data loading
- no backward-fill before an ETF exists
- equal-weight and risk-parity style allocation scans
- SPY and 60/40 SPY/IEF benchmarks
- SPY-relative metrics and crisis-window checks
- teaching copy aimed at curious beginners

## Notebook

- Local source: `notebook.py`
- Cloud execution surface: https://sb-5c296059d691d8f3.sb.molab.run/
- Token handling: use the marimo-pair token through environment/session auth only; do not save it in files.

## Source of Truth

Treat this local folder as the durable source of truth. Treat MoLab as the execution and sharing surface.

## Current Status

Created on 2026-06-21 as a sibling project to `autogluon-allocation-learning-lab`.

The notebook answers:

> Does broadening beyond SPY create a better opportunity set before we spend GPU time on AutoGluon?

Cloud smoke run passed on 2026-06-21. Managed-futures sleeves did not beat SPY on raw return, but they materially improved Sharpe and drawdown. See `reports/2026-06-21-cloud-smoke-run.md`.

2026-06-24 pass added material benchmark gates, batch yfinance download, dynamic crisis windows, EW/RP competitiveness checks, and an explicit promote-to-AutoGluon handoff linked to the ML notebook.

## Next Work

Re-run the cloud smoke test after gate changes and confirm promoted universes still match the 2026-06-21 managed-futures result.
