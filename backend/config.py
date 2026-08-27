"""Central configuration for the plastic segregation system."""

from __future__ import annotations

import os
from pathlib import Path

# Keep heavy artifacts on the USB stick so the 16GB SD card stays free.
USB_ROOT = Path(os.getenv("PLASTIC_USB_ROOT", "/media/river/6A1E-2BCD"))
RAW_DATASET = USB_ROOT / "Plastic Recognition"
SYSTEM_ROOT = USB_ROOT / "plastic_system"

MODELS_DIR = SYSTEM_ROOT / "models"
RUNS_DIR = SYSTEM_ROOT / "runs"
LOGS_DIR = SYSTEM_ROOT / "logs"
CACHE_DIR = SYSTEM_ROOT / "cache"
YOLO_DATASET = SYSTEM_ROOT / "yolo_dataset"
DATA_YAML = YOLO_DATASET / "data.yaml"

# Prefer a fine-tuned weight when present; otherwise fall back to YOLOv8n.
CUSTOM_WEIGHTS = MODELS_DIR / "plastic_yolov8n.pt"
BASE_WEIGHTS = MODELS_DIR / "yolov8n.pt"

CLASS_NAMES = {
    0: "recyclable",
    1: "non_recyclable",
}

# Recyclable plastics → servo left; non-recyclable → servo right.
RECYCLABLE_CLASSES = {"recyclable"}
NON_RECYCLABLE_CLASSES = {"non_recyclable"}

# Camera Module 3 capture size (good speed/quality balance on Pi 5).
CAMERA_WIDTH = int(os.getenv("CAMERA_WIDTH", "1280"))
CAMERA_HEIGHT = int(os.getenv("CAMERA_HEIGHT", "720"))
CAMERA_FPS = int(os.getenv("CAMERA_FPS", "15"))
CAMERA_BRIGHTNESS = float(os.getenv("CAMERA_BRIGHTNESS", "0.12"))
CAMERA_CONTRAST = float(os.getenv("CAMERA_CONTRAST", "1.15"))
CAMERA_SATURATION = float(os.getenv("CAMERA_SATURATION", "1.1"))

# Inference
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.30"))
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.45"))
INFER_IMGSZ = int(os.getenv("INFER_IMGSZ", "640"))
USE_OPENCV_FALLBACK = os.getenv("USE_OPENCV_FALLBACK", "1") == "1"
OPENCV_MIN_AREA_FRAC = float(os.getenv("OPENCV_MIN_AREA_FRAC", "0.0006"))
OPENCV_MAX_AREA_FRAC = float(os.getenv("OPENCV_MAX_AREA_FRAC", "0.35"))
OPENCV_MAX_DETS = int(os.getenv("OPENCV_MAX_DETS", "8"))

# Servo (signal wire on this BCM pin; power the 20kg servo from an external 5–6V supply).
SERVO_PIN = int(os.getenv("SERVO_PIN", "18"))
SERVO_RECYCLABLE_ANGLE = float(os.getenv("SERVO_RECYCLABLE_ANGLE", "-60"))
SERVO_NON_RECYCLABLE_ANGLE = float(os.getenv("SERVO_NON_RECYCLABLE_ANGLE", "60"))
SERVO_HOME_ANGLE = float(os.getenv("SERVO_HOME_ANGLE", "0"))
SERVO_HOLD_SECONDS = float(os.getenv("SERVO_HOLD_SECONDS", "1.2"))
SERVO_ENABLED = os.getenv("SERVO_ENABLED", "1") == "1"
SERVO_MIN_PULSE = float(os.getenv("SERVO_MIN_PULSE", "0.0005"))
SERVO_MAX_PULSE = float(os.getenv("SERVO_MAX_PULSE", "0.0025"))

# API / UI
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
AUTO_SORT = os.getenv("AUTO_SORT", "1") == "1"
SORT_COOLDOWN_SECONDS = float(os.getenv("SORT_COOLDOWN_SECONDS", "2.5"))

# Keep torch / ultralytics caches off the SD card.
os.environ.setdefault("TORCH_HOME", str(CACHE_DIR / "torch"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(CACHE_DIR / "ultralytics"))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR / "xdg"))


ZONE_BOX_SIZE = float(os.getenv("ZONE_BOX_SIZE", "0.22"))
TRAIN_PROMPT_COOLDOWN = float(os.getenv("TRAIN_PROMPT_COOLDOWN", "1.5"))


def ensure_directories() -> None:
    for path in (
        MODELS_DIR,
        RUNS_DIR,
        LOGS_DIR,
        CACHE_DIR,
        YOLO_DATASET / "images" / "train",
        YOLO_DATASET / "images" / "val",
        YOLO_DATASET / "labels" / "train",
        YOLO_DATASET / "labels" / "val",
        CACHE_DIR / "torch",
        CACHE_DIR / "ultralytics",
        CACHE_DIR / "xdg",
        SYSTEM_ROOT,
    ):
        path.mkdir(parents=True, exist_ok=True)
