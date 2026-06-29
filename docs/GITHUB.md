# GitHub sync

This library uses **GitHub as the durable public backup** for notebook source. MoLab catalog notebooks can mirror from GitHub for execution.

## Repository

- **GitHub:** https://github.com/thecharttruth/marimo-learning-labs
- **Local root:** `/Volumes/CORSAIR/Marimo`
- **Canonical notebook files:** `notebooks/trading/<slug>/notebook.py`

## What is safe to publish

These notebooks use **public market data only** (Yahoo Finance, etc.). No broker keys, API keys, or `MARIMO_TOKEN` values belong in the repo.

- `MARIMO_TOKEN` — environment variable only (see `.env.example`)
- `.env` — gitignored; never commit
- Notebook cells — no `os.environ` secret reads in current trading labs

Before each push, quick scan:

```bash
rg -i 'api[_-]?key|secret|password|ghp_|gho_|sk-[a-zA-Z0-9]{10,}' notebooks/ scripts/ --glob '!**/.venv/**'
```

## Daily workflow

```bash
cd /Volumes/CORSAIR/Marimo
./scripts/github-sync-slug-files.sh   # refresh {slug}.py copies from notebook.py
git add notebooks/trading/*/*.py
git commit -m "Update notebooks"
git push
```

Then in MoLab: **New notebook → Mirror from GitHub** (or re-sync an existing mirrored notebook) pointing at:

`https://github.com/thecharttruth/marimo-learning-labs/blob/main/notebooks/trading/<slug>/notebook.py`

## MoLab display names

GitHub mirrors use `{slug}.py` (a real copy of `notebook.py`) so MoLab cards are not all labeled `notebook.py`. Run `./scripts/github-sync-slug-files.sh` before each push to keep copies in sync.

**Rename to these titles in MoLab** (or run `./scripts/molab-rename.sh`):

| Slug | MoLab title |
|------|-------------|
| `global-macro-opportunity-set-lab` | **Global Macro Opportunity Set Lab** |
| `autogluon-allocation-learning-lab` | **AutoGluon Portfolio Allocation Learning Lab** |
| `btc-adaptive-supertrend-lab` | **BTC Adaptive Supertrend Lab** |

Delete duplicate `notebook.py` cards in MoLab if you mirrored twice — keep one per slug.

## MoLab GitHub mirror (auto-sync)

One-time setup for each notebook:

```bash
./scripts/molab-github-mirror.sh login    # once, if Chrome session expired
./scripts/molab-github-mirror.sh mirror   # all three trading labs
```

After that, **only `git push` is required** — MoLab synced notebooks pull from GitHub automatically.

Stable share links (no login, tracks `main`):

`https://molab.marimo.io/github/thecharttruth/marimo-learning-labs/blob/main/notebooks/trading/<slug>/notebook.py`

## MoLab profile URLs

Catalog URLs (`nb_*`) live in `notebooks/trading/NOTEBOOKS.yaml` and each project's `metadata.yaml`. GitHub is source of truth for **code**; MoLab is the **execution surface**.

Optional live push while a sandbox is open:

```bash
MARIMO_TOKEN=... python3 scripts/molab-sync.py push --slug <slug> --url https://sb-....sb.molab.run/
```
