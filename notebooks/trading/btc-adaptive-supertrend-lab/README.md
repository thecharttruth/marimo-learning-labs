# BTC Adaptive Supertrend Lab

Experimental marimo notebook for BTC daily research:

- Coinbase kline fetch helpers
- Ehlers homodyne dominant-cycle detector (L1)
- Modern adaptive supertrend with optional L3 hysteresis filter
- Altair charts for trend flips and band visualization
- yfinance fallback path for long BTC-USD history

## Notebook

- Local source: `notebook.py` (pull from cloud if missing or stale)
- Cloud execution surface: https://sb-8e1f27f556b2f563.sb.molab.run/
- Pairing auth: use `MARIMO_TOKEN` in the environment only; never commit tokens.

## Source of Truth

This project started on MoLab. Until the first successful pull, treat the cloud session as primary and this folder as the backup target.

Pull the latest cloud notebook:

```bash
cd /Volumes/CORSAIR/Marimo
MARIMO_TOKEN=your_session_token ./scripts/pull-molab-notebook.sh \
  'https://sb-8e1f27f556b2f563.sb.molab.run/' \
  'notebooks/trading/btc-adaptive-supertrend-lab'
uvx marimo@0.23.9 check --strict notebooks/trading/btc-adaptive-supertrend-lab/notebook.py
```

## Current Status

- 2026-06-24: Deduplicated `notebook.py` (1119 → ~600 lines). `marimo check --strict` passes.
- One Coinbase path (recent ~1000 daily candles) and one Yahoo Finance path (full history).
- Local is source of truth; push to MoLab when the sandbox session is open.

## Next Work

1. Re-run on MoLab and confirm charts render with live Coinbase + Yahoo data.
2. Optional: add markdown intro and parameter sliders for L1/L2/L3 toggles.
