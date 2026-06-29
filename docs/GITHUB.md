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
git status
git add notebooks/trading/<slug>/notebook.py   # or git add -A for full library changes
git commit -m "Update <slug> notebook"
git push
```

Then in MoLab: **New notebook → Mirror from GitHub** (or re-sync an existing mirrored notebook) pointing at:

`https://github.com/thecharttruth/marimo-learning-labs/blob/main/notebooks/trading/<slug>/notebook.py`

## MoLab profile URLs

Catalog URLs (`nb_*`) live in `notebooks/trading/NOTEBOOKS.yaml` and each project's `metadata.yaml`. GitHub is source of truth for **code**; MoLab is the **execution surface**.

Optional live push while a sandbox is open:

```bash
MARIMO_TOKEN=... python3 scripts/molab-sync.py push --slug <slug> --url https://sb-....sb.molab.run/
```
