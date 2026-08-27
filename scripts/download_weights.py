#!/usr/bin/env python3
"""Download base YOLOv8n weights onto the USB models directory."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend import config


def main() -> int:
    config.ensure_directories()
    out = config.BASE_WEIGHTS
    if out.exists():
        print(f"Already present: {out}")
        return 0

    from ultralytics import YOLO

    # Download into USB models dir (ultralytics writes yolov8n.pt into CWD).
    prev = Path.cwd()
    try:
        import os

        os.chdir(config.MODELS_DIR)
        YOLO("yolov8n.pt")
    finally:
        os.chdir(prev)

    local = config.MODELS_DIR / "yolov8n.pt"
    if local.exists():
        if local.resolve() != out.resolve():
            shutil.copy2(local, out)
        print(f"Base weights ready: {out}")
        return 0

    cached = list(config.CACHE_DIR.rglob("yolov8n.pt"))
    if cached:
        shutil.copy2(cached[0], out)
        print(f"Base weights copied from cache → {out}")
        return 0

    print("Download finished but yolov8n.pt was not found on disk.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
