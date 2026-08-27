#!/usr/bin/env bash
# Create venv with system-site-packages (picamera2 + apt torch/opencv) and install light deps.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export DEBIAN_FRONTEND=noninteractive

echo "Installing system packages (OpenCV, camera, GPIO, CPU torch)..."
sudo apt-get update
sudo apt-get install -y \
  python3-opencv \
  python3-picamera2 \
  python3-lgpio \
  python3-torch \
  python3-torchvision \
  python3-venv \
  python3-pip \
  python3-yaml \
  python3-pil \
  python3-numpy

# Fresh venv so a failed CUDA torch install cannot linger.
rm -rf .venv
python3 -m venv --system-site-packages .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip

export PIP_CACHE_DIR="${PLASTIC_USB_ROOT:-/media/river/6A1E-2BCD}/plastic_system/cache/pip"
mkdir -p "$PIP_CACHE_DIR"

# Verify system torch is visible BEFORE ultralytics so pip does not fetch CUDA wheels.
python - <<'PY'
import torch
print("system torch", torch.__version__, "cuda?", torch.cuda.is_available())
import cv2
print("system cv2", cv2.__version__)
PY

# Install API stack; block accidental torch/opencv reinstalls from PyPI.
pip install \
  --no-deps \
  "fastapi>=0.115.0" \
  "uvicorn[standard]>=0.30.0" \
  "ultralytics>=8.3.0" \
  "pydantic>=2.0.0" \
  "python-multipart>=0.0.9"

# Lightweight pure-python deps ultralytics/uvicorn need (no torch/opencv).
pip install \
  "starlette>=0.40.0" \
  "anyio>=4.0.0" \
  "typing-inspection>=0.4.0" \
  "annotated-types>=0.6.0" \
  "annotated-doc>=0.0.2" \
  "click>=8.0" \
  "h11>=0.16" \
  "httptools>=0.6" \
  "python-dotenv>=1.0" \
  "uvloop>=0.20; platform_system == 'Linux'" \
  "watchfiles>=0.20" \
  "websockets>=13.0" \
  "matplotlib>=3.8" \
  "scipy>=1.11" \
  "pandas>=2.0" \
  "seaborn>=0.13" \
  "requests>=2.31" \
  "tqdm>=4.66" \
  "psutil>=5.9" \
  "pyyaml>=6.0" \
  "pillow>=10.0"

python - <<'PY'
from backend import config
config.ensure_directories()
import torch, cv2
from ultralytics import YOLO
print("OK torch", torch.__version__)
print("OK cv2", cv2.__version__)
print("OK ultralytics import")
print("USB system root:", config.SYSTEM_ROOT)
print("Models dir:", config.MODELS_DIR)
PY

chmod +x run_api.sh run_display.sh scripts/*.sh scripts/*.py display/local_display.py

echo
echo "Setup complete."
echo "Next:"
echo "  1) python scripts/prepare_dataset.py"
echo "  2) label images (recyclable=0, non_recyclable=1)"
echo "  3) python scripts/download_weights.py   # optional smoke-test base model"
echo "  4) bash run_api.sh                     # web UI at http://<pi-ip>:8000"
echo "     or: bash run_display.sh             # OpenCV window on HDMI"
