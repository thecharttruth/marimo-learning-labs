# Marimo Notebook Library

This folder is the local source-of-truth library for marimo notebooks, related apps, exports, blog drafts, and GitHub-ready publishing assets.

Cloud notebooks are for execution, demos, collaboration, and sharing. Local notebook project folders are for version control, recovery, editing history, blog writing, and reproducible publishing.

## Launch Notebooks

[![Open AutoGluon Lab In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/thecharttruth/marimo-learning-labs/blob/main/notebooks/trading/autogluon-allocation-learning-lab/google-colab-autogluon-portfolio-allocation-learning-lab.ipynb) [![Open AutoGluon Lab In marimo](https://marimo.io/shield.svg)](https://molab.marimo.io/notebooks/nb_dqdkCXtD93dSyuxLXgnQJD)

- AutoGluon Portfolio Allocation Learning Lab: [`google-colab-autogluon-portfolio-allocation-learning-lab.ipynb`](notebooks/trading/autogluon-allocation-learning-lab/google-colab-autogluon-portfolio-allocation-learning-lab.ipynb)

## Layout

```text
notebooks/
  trading/
    NOTEBOOKS.yaml
    README.md
    global-macro-opportunity-set-lab/
    autogluon-allocation-learning-lab/
    btc-adaptive-supertrend-lab/
shared/
  charts/
  utils/
templates/
  notebook-project/
docs/
  posts/
```

## Rules

- Keep real API keys, secrets, account tokens, broker credentials, and private data out of this folder.
- Keep each notebook in its own project folder, even when notebooks are small.
- Prefer `notebook.py` as the canonical marimo source file.
- Use `metadata.yaml` to track cloud URLs, GitHub URLs, tags, status, and publishing notes.
- Use `notebooks/trading/NOTEBOOKS.yaml` as the machine-readable catalog when multiple cloud sessions exist.
- Pull cloud notebooks into the matching local folder with `scripts/pull-molab-notebook.sh` or `python3 scripts/molab-sync.py pull --slug <slug>` (requires `MARIMO_TOKEN` in the environment).
- Keep local and MoLab aligned with `python3 scripts/molab-sync.py sync --slug <slug>` at the end of each session. See `notebooks/trading/SYNC.md`.
- Put long-form writing in `blog/` while it is attached to a notebook, then copy finished posts to `docs/posts/`.
- Put generated HTML, IPYNB, PDF, and image exports in `exports/`.
- Put run summaries, evaluation tables, and screenshots in `reports/`.
- Use `data/README.md` to document data sources and refresh rules; do not commit private datasets unless explicitly intended.

## GitHub Publishing

**Repo:** https://github.com/thecharttruth/marimo-learning-labs (public source of truth for notebook code)

See `docs/GITHUB.md` for push workflow and secret checks. MoLab can mirror notebooks from this repo.

Start with one GitHub repo for the whole library. Split a notebook into its own repo only if it becomes a standalone product, course, or deployable app.
