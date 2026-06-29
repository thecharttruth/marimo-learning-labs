# Plan: Global Macro Opportunity Set Lab

## Goal

Build a clean teaching notebook that tests non-SPY opportunity sets before adding ML.

## Scope

Include:

- global macro ETF universe presets
- managed futures and trend-alternative sleeves
- inception-safe data loading
- benchmark-only allocation scans
- SPY-relative metrics
- crisis-period diagnostics
- clear verdict on which universes deserve ML

Exclude for now:

- AutoGluon
- parameter optimization
- live trading
- broker integration

## Ordered Steps

1. Create local project folder. Done.
2. Create `notebook.py` with teaching sections and benchmark scan. Done.
3. Add README, PLAN, and metadata. Done.
4. Verify local syntax and marimo strict checks.
5. Push the notebook to the fresh MoLab session with marimo-pair. Done.
6. Run the cloud notebook and inspect results. Done.
7. Save a report under `reports/`. Done.

## Stop Conditions

- Any ticker appears before its real first valid price.
- The notebook uses backward-fill across ETF inception.
- The cloud session cannot be verified as saved.
- The benchmark scan shows no meaningful non-SPY improvement.

## Verification Steps

- Run `python3 -m py_compile notebook.py`.
- Run `uvx marimo@latest check --strict notebook.py`.
- Scan for secrets.
- Confirm the cloud notebook imports and renders.
- Confirm the saved cloud artifact contains the same intended notebook.
- Confirm no token values are stored.

## Expected Artifacts

- `notebook.py`
- `README.md`
- `PLAN.md`
- `metadata.yaml`
- `reports/2026-06-21-cloud-smoke-run.md`

## Next Recommended Task

Re-run the cloud smoke test after the 2026-06-24 gate pass and confirm promoted universes still match the managed-futures result before feeding them into AutoGluon.
