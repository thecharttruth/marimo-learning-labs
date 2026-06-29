#!/usr/bin/env bash
# Pull a saved notebook.py from a running MoLab/marimo session into a local project folder.
#
# Usage:
#   MARIMO_TOKEN=your_token ./scripts/pull-molab-notebook.sh \
#     'https://sb-example.sb.molab.run/' \
#     'notebooks/trading/my-notebook-lab' \
#     [session_id]
#
# Never commit tokens. Use the environment variable only.
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: MARIMO_TOKEN=... $0 <molab_url> <local_project_dir> [session_id]" >&2
  exit 1
fi

url="${1%/}"
out_dir="$2"
session_id="${3:-}"
token="${MARIMO_TOKEN:-}"
repo_root="$(cd "$(dirname "$0")/.." && pwd)"
execute_code="${repo_root}/.agents/skills/marimo-pair/scripts/execute-code.sh"

if [[ -z "$token" ]]; then
  echo "Set MARIMO_TOKEN in the environment before running this script." >&2
  exit 1
fi

mkdir -p "$out_dir/reports"
tmp_file="$(mktemp)"
trap 'rm -f "$tmp_file"' EXIT

http_code=$(curl -sS -o "$tmp_file" -w '%{http_code}' \
  -H "Authorization: Bearer ${token}" \
  "${url}/marimo/notebook.py" || echo "000")

if [[ "$http_code" == "200" ]] && head -1 "$tmp_file" | grep -q 'import marimo'; then
  cp "$tmp_file" "${out_dir}/notebook.py"
  echo "Saved ${out_dir}/notebook.py via HTTP ($(wc -l < "${out_dir}/notebook.py") lines)"
  exit 0
fi

if [[ ! -x "$execute_code" ]]; then
  echo "HTTP download failed (HTTP ${http_code}) and marimo-pair script is unavailable." >&2
  exit 1
fi

session_args=()
if [[ -n "$session_id" ]]; then
  session_args=(--session "$session_id")
fi

MARIMO_TOKEN="$token" bash "$execute_code" --url "$url" "${session_args[@]}" <<'PY' > "$tmp_file"
from pathlib import Path
print(Path('/marimo/notebook.py').read_text(), end='')
PY

if ! head -1 "$tmp_file" | grep -q 'import marimo'; then
  echo "Pair fallback did not return a marimo notebook." >&2
  head -c 400 "$tmp_file" >&2 || true
  echo >&2
  exit 1
fi

cp "$tmp_file" "${out_dir}/notebook.py"
echo "Saved ${out_dir}/notebook.py via marimo-pair ($(wc -l < "${out_dir}/notebook.py") lines)"
