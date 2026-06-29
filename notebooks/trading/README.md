# Trading Notebooks

Each notebook lives in its own project folder. Local `notebook.py` is the durable source of truth unless a project's `metadata.yaml` marks the cloud session as primary during active research.

## Projects

| Slug | Folder | MoLab / cloud |
|------|--------|----------------|
| `global-macro-opportunity-set-lab` | [global-macro-opportunity-set-lab/](global-macro-opportunity-set-lab/) | [MoLab session](https://sb-5c296059d691d8f3.sb.molab.run/) |
| `autogluon-allocation-learning-lab` | [autogluon-allocation-learning-lab/](autogluon-allocation-learning-lab/) | [MoLab notebook](https://molab.marimo.io/notebooks/nb_dqdkCXtD93dSyuxLXgnQJD) |
| `btc-adaptive-supertrend-lab` | [btc-adaptive-supertrend-lab/](btc-adaptive-supertrend-lab/) | [MoLab session](https://sb-8e1f27f556b2f563.sb.molab.run/) |

Machine-readable registry: [NOTEBOOKS.yaml](NOTEBOOKS.yaml)

Sync policy and MoLab profile publish: [SYNC.md](SYNC.md)

```bash
# Save notebooks to your MoLab profile (catalog nb_* URLs)
../../scripts/molab-profile.sh login
../../scripts/molab-profile.sh publish

# Check drift (needs MARIMO_TOKEN + open MoLab session per URL)
python3 ../../scripts/molab-sync.py status

# Push local -> cloud / pull cloud -> local / round-trip verify
python3 ../../scripts/molab-sync.py push --slug global-macro-opportunity-set-lab
python3 ../../scripts/molab-sync.py pull --slug global-macro-opportunity-set-lab
python3 ../../scripts/molab-sync.py sync --slug global-macro-opportunity-set-lab
```

## Agent workflow

1. Read `NOTEBOOKS.yaml` and open the matching local folder.
2. Confirm the MoLab URL matches the intended slug before pairing.
3. Edit locally when the project is local-authoritative; use marimo-pair + `cm` on cloud when the live session is ahead.
4. Publish new notebooks to MoLab profile: `../../scripts/molab-profile.sh publish --slug <slug>`.
5. After cloud edits, pull back with:

```bash
MARIMO_TOKEN=your_session_token ../../scripts/pull-molab-notebook.sh \
  'https://sb-example.sb.molab.run/' \
  'notebooks/trading/your-notebook-lab'
```

Never save tokens in this repository.

## Required files per project

- `notebook.py`
- `README.md`
- `PLAN.md`
- `metadata.yaml`

Optional: `blog/`, `reports/`, `exports/`, `data/`
