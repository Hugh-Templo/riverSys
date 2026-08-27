"""Save live labeled frames into the USB YOLO training set."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from . import config
from .detector import Detection

logger = logging.getLogger(__name__)

CLASS_ID = {"recyclable": 0, "non_recyclable": 1}


def save_training_sample(frame_rgb: np.ndarray, det: Detection, label: str) -> Path | None:
    """Write image + YOLO label for one human-confirmed detection."""
    if label not in CLASS_ID:
        return None

    config.ensure_directories()
    img_dir = config.YOLO_DATASET / "images" / "train"
    lbl_dir = config.YOLO_DATASET / "labels" / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    name = f"live_{stamp}_{label}"
    img_path = img_dir / f"{name}.jpg"
    lbl_path = lbl_dir / f"{name}.txt"

    h, w = frame_rgb.shape[:2]
    x1 = max(0, min(w - 1, det.x1))
    y1 = max(0, min(h - 1, det.y1))
    x2 = max(0, min(w - 1, det.x2))
    y2 = max(0, min(h - 1, det.y2))
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    xc = (x1 + x2) / 2.0 / w
    yc = (y1 + y2) / 2.0 / h
    nw = bw / w
    nh = bh / h

    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(img_path), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    lbl_path.write_text(f"{CLASS_ID[label]} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}\n", encoding="utf-8")
    logger.info("Saved training sample %s (%s)", img_path.name, label)
    return img_path
