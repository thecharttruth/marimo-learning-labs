#!/usr/bin/env bash
# Deprecated wrapper — use ./scripts/molab-profile.sh instead.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
exec "$ROOT/scripts/molab-profile.sh" publish "$@"
