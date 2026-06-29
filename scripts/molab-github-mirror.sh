#!/usr/bin/env bash
# Mirror marimo notebooks from GitHub into MoLab workspace (auto-sync on git push).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec python3 "$ROOT/scripts/molab-github-mirror.py" "$@"
