# Local ↔ MoLab sync policy for notebooks/trading/

## Principle

**Local folders in this repo are the durable backup.** MoLab is where notebooks run and are shared.

MoLab **does** persist `notebook.py` on the running kernel at `/marimo/notebook.py` while a session is alive, but:

- **Sandbox URLs** (`https://sb-*.sb.molab.run/`) can expire when the sandbox stops.
- **Catalog notebooks** (`https://molab.marimo.io/notebooks/nb_*`) are more durable, but you should still pull back to local after edits.
- MoLab may **normalize** PEP 723 headers or wrapper order, so compare **cell bodies**, not raw file hashes.

Treat “saved in the cloud” as: **open session + written `/marimo/notebook.py` + pull verified locally**.

## One notebook = one folder = one MoLab URL

See [NOTEBOOKS.yaml](NOTEBOOKS.yaml). Never pair on a URL without checking the slug first.

| Slug | Local folder | Cloud |
|------|--------------|-------|
| `global-macro-opportunity-set-lab` | `global-macro-opportunity-set-lab/` | sandbox MoLab |
| `autogluon-allocation-learning-lab` | `autogluon-allocation-learning-lab/` | molab.marimo.io catalog |
| `btc-adaptive-supertrend-lab` | `btc-adaptive-supertrend-lab/` | sandbox MoLab |

## Daily workflow

### After editing **locally**

```bash
cd /Volumes/CORSAIR/Marimo
MARIMO_TOKEN=your_token python3 scripts/molab-sync.py push --slug global-macro-opportunity-set-lab
```

Reload the MoLab browser tab if cells do not update immediately.

### After editing **on MoLab** (or before closing the tab)

```bash
MARIMO_TOKEN=your_token python3 scripts/molab-sync.py pull --slug global-macro-opportunity-set-lab
uvx marimo@0.23.9 check --strict notebooks/trading/global-macro-opportunity-set-lab/notebook.py
```

### End of session (recommended)

```bash
MARIMO_TOKEN=your_token python3 scripts/molab-sync.py sync --slug global-macro-opportunity-set-lab
```

This pushes local → cloud, pulls cloud → local, and verifies cell-body hashes match.

### Check all projects for drift

```bash
MARIMO_TOKEN=your_token python3 scripts/molab-sync.py status
```

## MoLab profile (catalog nb_* URLs)

Sandbox URLs (`sb-*.sb.molab.run`) are **not** saved in your MoLab profile. Use the profile publish tool once per notebook:

```bash
cd /Volumes/CORSAIR/Marimo
./scripts/molab-profile.sh login      # one-time Google sign-in in Chrome
./scripts/molab-profile.sh status   # catalog vs needs_publish
./scripts/molab-profile.sh publish  # import all unpublished registry notebooks
./scripts/molab-profile.sh publish --slug btc-adaptive-supertrend-lab
```

The tool reads `NOTEBOOKS.yaml`, validates with `marimo check --strict`, uploads via MoLab Import UI (automated), and writes catalog URLs back to `NOTEBOOKS.yaml` + project `metadata.yaml`.

### New notebook checklist

1. Add project folder under `notebooks/trading/<slug>/` with `notebook.py`.
2. Register in `NOTEBOOKS.yaml` with `cloud_kind: unpublished`.
3. Run `./scripts/molab-profile.sh publish --slug <slug>`.
4. Use `molab-sync.py push/pull` against the new `nb_*` URL.

## MoLab manual save checklist

In the MoLab UI, before closing a session:

1. Run all cells (or confirm no stale errors).
2. Wait for the kernel to finish writing (no active runs).
3. Run **pull** locally so this repo has the latest copy.
4. Optionally use MoLab export/download if the UI offers it.

## Agent rule

An agent must not call a notebook “synced” until:

1. `molab-sync.py status` shows `match` for that slug, **or**
2. A pull/push round-trip completed in this session and `.sync/state.json` was updated.

## Tokens

- Set `MARIMO_TOKEN` in the environment only.
- Never commit tokens to git.
- See repo-root `.env.example`.
