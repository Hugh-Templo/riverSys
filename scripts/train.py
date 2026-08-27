#!/usr/bin/env python3
"""Train YOLOv8n on the USB-hosted plastic dataset (keep artifacts off the SD card)."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend import config


def main() -> int:
    parser = argparse.ArgumentParser(description="Train plastic YOLO model on USB data")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--model", type=str, default="yolov8n.pt")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    config.ensure_directories()
    if not config.DATA_YAML.exists():
        print("data.yaml missing — run scripts/prepare_dataset.py first")
        return 1

    label_count = len(list((config.YOLO_DATASET / "labels" / "train").glob("*.txt")))
    if label_count == 0:
        print("No label files found under labels/train.")
        print("Annotate images before training (class 0=recyclable, 1=non_recyclable).")
        return 1

    from ultralytics import YOLO

    # Download / load base weights into USB models dir when possible.
    base = config.MODELS_DIR / Path(args.model).name
    model = YOLO(args.model if not base.exists() else str(base))

    results = model.train(
        data=str(config.DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(config.RUNS_DIR),
        name="plastic_yolov8n",
        exist_ok=True,
        workers=2,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    if best.exists():
        target = config.CUSTOM_WEIGHTS
        shutil.copy2(best, target)
        print(f"Copied best weights → {target}")
        print("Restart the API or call POST /api/detector/reload")
    else:
        print("Training finished but best.pt was not found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
