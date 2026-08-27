#!/usr/bin/env python3
"""Prepare YOLO dataset folders on the USB drive from the raw Plastic Recognition set."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend import config


DATA_YAML_TEMPLATE = """# Auto-generated for floating plastic segregation
path: {path}
train: images/train
val: images/val

names:
  0: recyclable
  1: non_recyclable

# Labeling guide (YOLO txt next to each image name under labels/):
#   class_id  x_center  y_center  width  height   (all normalized 0-1)
# class 0 = recyclable (bottles / hard plastic)
# class 1 = non_recyclable (bags / film / fragments)
"""


def _copy_images(src_dir: Path, dst_dir: Path) -> int:
    dst_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for img in sorted(src_dir.glob("*.jpg")) + sorted(src_dir.glob("*.JPG")):
        target = dst_dir / img.name
        if not target.exists():
            shutil.copy2(img, target)
        count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw",
        type=Path,
        default=config.RAW_DATASET,
        help="Raw dataset root containing Train/Test/Predict",
    )
    args = parser.parse_args()

    config.ensure_directories()
    raw = args.raw
    train_src = raw / "Train"
    val_src = raw / "Test"

    if not train_src.exists():
        print(f"Missing train folder: {train_src}")
        return 1

    n_train = _copy_images(train_src, config.YOLO_DATASET / "images" / "train")
    n_val = _copy_images(val_src, config.YOLO_DATASET / "images" / "val") if val_src.exists() else 0

    # Also stage Predict images into val if Test is tiny.
    predict = raw / "Predict"
    if predict.exists() and n_val < 5:
        n_val += _copy_images(predict, config.YOLO_DATASET / "images" / "val")

    yaml_text = DATA_YAML_TEMPLATE.format(path=str(config.YOLO_DATASET))
    config.DATA_YAML.write_text(yaml_text, encoding="utf-8")

    label_train = list((config.YOLO_DATASET / "labels" / "train").glob("*.txt"))
    label_val = list((config.YOLO_DATASET / "labels" / "val").glob("*.txt"))

    print(f"Images ready on USB:")
    print(f"  train images: {n_train}")
    print(f"  val images:   {n_val}")
    print(f"  train labels: {len(label_train)}")
    print(f"  val labels:   {len(label_val)}")
    print(f"  data.yaml:    {config.DATA_YAML}")
    if not label_train:
        print("\nIMPORTANT: images have no labels yet.")
        print("Label them with Label Studio / CVAT / labelImg into:")
        print(f"  {config.YOLO_DATASET / 'labels' / 'train'}")
        print(f"  {config.YOLO_DATASET / 'labels' / 'val'}")
        print("Then run: python scripts/train.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
