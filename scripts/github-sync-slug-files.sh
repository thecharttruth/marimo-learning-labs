#!/usr/bin/env bash
# Refresh {slug}.py copies from notebook.py before git push (MoLab mirror filenames).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
for dir in "$ROOT"/notebooks/trading/*/; do
  [[ -f "$dir/notebook.py" ]] || continue
  slug="$(basename "$dir")"
  cp "$dir/notebook.py" "$dir/$slug.py"
  echo "updated $slug/$slug.py"
done
