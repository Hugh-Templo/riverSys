#!/usr/bin/env bash
# Start local OpenCV HDMI display (Python UI).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PLASTIC_USB_ROOT="${PLASTIC_USB_ROOT:-/media/river/6A1E-2BCD}"
export DISPLAY="${DISPLAY:-:0}"
# Pi OS labwc is Wayland; OpenCV/Qt is more stable on XWayland and quieter on GLib.
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"
export GDK_BACKEND="${GDK_BACKEND:-x11}"
exec python display/local_display.py "$@"
