#!/usr/bin/env bash
# Publish local marimo notebooks to your MoLab profile (catalog nb_* URLs).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec python3 "$ROOT/scripts/molab-profile.py" "$@"
