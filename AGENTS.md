# AGENTS.md

Guidance for Codex and other coding agents working in this Marimo library.

## Source of Truth

- Treat `/Volumes/CORSAIR/Marimo` as the durable local source of truth.
- Treat MoLab and other cloud marimo sessions as execution and sharing surfaces.
- When editing a live marimo session, use `marimo._code_mode` or the marimo pair tooling. Do not edit a live notebook file behind the kernel.
- Prefer making durable edits locally, then push local `notebook.py` into a live cloud session for execution.
- After cloud-originated edits, export or pull the latest notebook back into the matching local project folder before calling the work complete.
- Use `python3 scripts/molab-sync.py status|pull|push|sync --slug <slug>` to keep local and MoLab copies aligned. See `notebooks/trading/SYNC.md`.
- When local is authoritative and MoLab normalizes script metadata, compare cloud/local cell bodies and config hashes rather than raw file hashes.

## Organization

- One notebook project gets one folder under `notebooks/<domain>/<slug>/`.
- Keep different notebooks separate even when they share a parent directory (for example `notebooks/trading/`).
- Read `notebooks/trading/NOTEBOOKS.yaml` (and `notebooks/trading/README.md`) before pairing on cloud to confirm URL ↔ slug ↔ local folder.
- Every project should have `notebook.py`, `README.md`, `PLAN.md`, and `metadata.yaml`.
- After cloud-originated work on a cloud-primary project, pull `notebook.py` with `scripts/pull-molab-notebook.sh` into the matching local folder before calling sync complete.
- Keep blog drafts beside the notebook while drafting.
- Keep reusable chart helpers or utilities in `shared/` only after at least two notebooks need them.
- Do not create broad abstractions for one notebook.

## Safety

- Never print or save secrets.
- Use `.env.example` for required environment variable names.
- Prefer explicit data provenance, timestamps, and refresh rules.
- For trading notebooks, document lookahead-bias controls, transaction-cost assumptions, and out-of-sample gates.

## Verification

Before calling a notebook project ready:

- Confirm the marimo notebook imports and renders.
- Confirm the local `notebook.py` is the latest intended version.
- If a cloud session is used, confirm the cloud notebook matches local by cell/config evidence or explain the remaining sync gate.
- Confirm README and metadata cloud links are current.
- Confirm no secret values are present.
- If a MoLab session was used, run `molab-sync.py status` for the slug and pull if drift is shown.
