#!/usr/bin/env bash
# Launch plastic OpenCV display inside a reusable tmux session.
set -euo pipefail

ROOT="/home/river/projectRiver"
SESSION="plastic-display"
export DISPLAY="${DISPLAY:-:0}"
export PLASTIC_USB_ROOT="${PLASTIC_USB_ROOT:-/media/river/6A1E-2BCD}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
export GDK_BACKEND="${GDK_BACKEND:-x11}"

cd "$ROOT"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  # Already running — attach so the user can see logs / press keys in tmux.
  exec tmux attach-session -t "$SESSION"
fi

exec tmux new-session -s "$SESSION" "bash '$ROOT/run_display.sh'; echo; echo 'Display stopped. Press Enter to close.'; read"
