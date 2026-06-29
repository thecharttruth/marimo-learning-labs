#!/usr/bin/env bash
# Launch Google Chrome with remote debugging for molab-profile automation.
# Log into MoLab once in this window; publish commands attach via CDP (port 9222).
set -euo pipefail

PROFILE_DIR="${MOLAB_CHROME_PROFILE:-$HOME/Library/Application Support/molab-playwright}"
CDP_PORT="${MOLAB_CDP_PORT:-9222}"
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

if [[ ! -x "$CHROME" ]]; then
  echo "Google Chrome not found at $CHROME" >&2
  exit 1
fi

if curl -sf "http://127.0.0.1:${CDP_PORT}/json/version" >/dev/null 2>&1; then
  echo "MoLab Chrome already running on port ${CDP_PORT}."
  open "https://molab.marimo.io/notebooks"
  exit 0
fi

echo "Starting Chrome (profile: ${PROFILE_DIR}, CDP port: ${CDP_PORT})"
echo "Sign into MoLab in the window that opens, then run: ./scripts/molab-profile.sh publish"

"$CHROME" \
  --remote-debugging-port="${CDP_PORT}" \
  --user-data-dir="${PROFILE_DIR}" \
  "https://molab.marimo.io/notebooks" \
  >/dev/null 2>&1 &

sleep 2
echo "Chrome started. Complete Google sign-in if prompted."
