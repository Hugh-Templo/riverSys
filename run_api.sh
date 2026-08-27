#!/usr/bin/env bash
# Start FastAPI backend (lightweight web UI — recommended on Pi 5).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

if [[ ! -d .venv ]]; then
  echo "Missing .venv — run: bash scripts/setup_env.sh"
  exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PLASTIC_USB_ROOT="${PLASTIC_USB_ROOT:-/media/river/6A1E-2BCD}"

exec python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
