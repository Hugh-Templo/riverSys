"""Detection: YOLOv8 when trained weights exist, OpenCV fallback otherwise."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from . import config

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    label: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int
    source: str  # "yolo" | "opencv"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PlasticDetector:
    def __init__(self) -> None:
        self.model = None
        self.mode = "opencv"
        self.weights_path: Optional[Path] = None
        self._load_model()

    def _load_model(self) -> None:
        config.ensure_directories()
        weights = None
        if config.CUSTOM_WEIGHTS.exists():
            weights = config.CUSTOM_WEIGHTS
        elif config.BASE_WEIGHTS.exists():
            weights = config.BASE_WEIGHTS

        if weights is None:
            logger.warning(
                "No YOLO weights in %s — using OpenCV fallback until you train.",
                config.MODELS_DIR,
            )
            self.mode = "opencv"
            return

        try:
            from ultralytics import YOLO

            self.model = YOLO(str(weights))
            self.weights_path = weights
            # Custom 2-class model vs COCO base model.
            names = getattr(self.model, "names", {}) or {}
            name_values = {str(v).lower() for v in names.values()}
            if "recyclable" in name_values or "non_recyclable" in name_values:
                self.mode = "yolo_custom"
            else:
                self.mode = "yolo_coco"
            logger.info("Loaded YOLO weights %s (mode=%s)", weights, self.mode)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load YOLO (%s); OpenCV fallback", exc)
            self.model = None
            self.mode = "opencv"

    def reload(self) -> str:
        self.model = None
        self._load_model()
        return self.mode

    def detect(self, frame_rgb: np.ndarray) -> list[Detection]:
        if frame_rgb is None:
            return []
        if self.mode.startswith("yolo") and self.model is not None:
            return self._detect_yolo(frame_rgb)
        if config.USE_OPENCV_FALLBACK:
            return self._detect_opencv(frame_rgb)
        return []

    def _detect_yolo(self, frame_rgb: np.ndarray) -> list[Detection]:
        import cv2

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        results = self.model.predict(
            source=frame_bgr,
            conf=config.CONF_THRESHOLD,
            iou=config.IOU_THRESHOLD,
            imgsz=config.INFER_IMGSZ,
            verbose=False,
        )
        detections: list[Detection] = []
        if not results:
            return detections

        result = results[0]
        names = result.names or {}
        if result.boxes is None:
            return detections

        for box in result.boxes:
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            raw_name = str(names.get(cls_id, cls_id)).lower()
            label = self._map_label(raw_name)
            if label is None:
                continue
            detections.append(
                Detection(
                    label=label,
                    confidence=conf,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    source="yolo",
                )
            )
        return detections

    def _map_label(self, raw_name: str) -> Optional[str]:
        raw = raw_name.lower().strip().replace(" ", "_").replace("-", "_")
        if raw in config.RECYCLABLE_CLASSES | config.NON_RECYCLABLE_CLASSES:
            return raw

        # Useful when only base COCO yolov8n.pt is present (smoke-test mode).
        recyclable_aliases = {
            "bottle",
            "cup",
            "wine_glass",
            "plastic_bottle",
            "recyclable",
        }
        non_recyclable_aliases = {
            "plastic_bag",
            "bag",
            "cell_phone",  # often mis-detected; treat conservatively
            "non_recyclable",
            "book",  # paper-like distractor → non path for demo sorting
        }
        if raw in recyclable_aliases:
            return "recyclable"
        if raw in non_recyclable_aliases:
            return "non_recyclable"
        # Custom floating-plastic class names from literature-style datasets.
        if raw in {"plastic_bottle", "bottle", "hard_plastic", "common_plastic"}:
            return "recyclable"
        if raw in {"plastic_film", "vinyl", "fragment", "fragmented_plastic", "film"}:
            return "non_recyclable"
        if "bottle" in raw:
            return "recyclable"
        if any(k in raw for k in ("bag", "film", "vinyl", "fragment", "wrapper")):
            return "non_recyclable"
        if self.mode == "yolo_custom":
            return raw if raw in ("recyclable", "non_recyclable") else None
        return None

    def _detect_opencv(self, frame_rgb: np.ndarray) -> list[Detection]:
        """Heuristic floating-plastic candidates — tuned for dim Pi camera scenes."""
        import cv2

        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        h, w = frame_bgr.shape[:2]
        blurred = cv2.GaussianBlur(frame_bgr, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

        # 1) Brighter / more saturated patches than typical dark water.
        bright = cv2.inRange(hsv, (0, 0, 70), (180, 120, 255))
        colorful = cv2.inRange(hsv, (0, 35, 40), (180, 255, 255))
        # Near-white / pale plastics (low sat, mid-high value).
        pale = cv2.inRange(hsv, (0, 0, 90), (180, 50, 255))

        # 2) Local contrast: pixels standing out from neighborhood (works in dim light).
        local = cv2.GaussianBlur(gray, (31, 31), 0)
        diff = cv2.absdiff(gray, local)
        _, contrast = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)

        # 3) Edges help bags/bottles with weak color.
        edges = cv2.Canny(gray, 40, 120)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

        mask = bright | colorful | pale | contrast
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)

        # Reinforce with edge-closed contrast regions (bags/bottles with weak color).
        edge_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=1)
        mask = cv2.bitwise_or(mask, cv2.bitwise_and(edge_closed, contrast))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = (h * w) * config.OPENCV_MIN_AREA_FRAC
        max_area = (h * w) * config.OPENCV_MAX_AREA_FRAC

        detections: list[Detection] = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue
            x, y, bw, bh = cv2.boundingRect(contour)
            # Ignore ultra-thin border strips (often lighting gradients).
            if bw < 18 or bh < 18:
                continue
            if x <= 2 or y <= 2 or x + bw >= w - 2 or y + bh >= h - 2:
                if area < min_area * 4:
                    continue

            aspect = bw / max(bh, 1)
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
            solidity = area / max(float(bw * bh), 1.0)

            # Elongated / floppy shapes → film/bag; compact → bottle-like.
            if aspect > 2.0 or aspect < 0.45 or (solidity < 0.35 and len(approx) > 6):
                label = "non_recyclable"
            else:
                label = "recyclable"

            roi = gray[y : y + bh, x : x + bw]
            contrast_score = float(np.clip(roi.std() / 40.0, 0.0, 1.0)) if roi.size else 0.0
            size_score = float(np.clip(area / max_area, 0.0, 1.0))
            conf = float(np.clip(0.28 + 0.35 * contrast_score + 0.30 * size_score, 0.28, 0.95))

            detections.append(
                Detection(
                    label=label,
                    confidence=conf,
                    x1=x,
                    y1=y,
                    x2=x + bw,
                    y2=y + bh,
                    source="opencv",
                )
            )

        detections = self._nms(detections, iou_thresh=0.35)
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections[: config.OPENCV_MAX_DETS]

    @staticmethod
    def _nms(dets: list[Detection], iou_thresh: float = 0.35) -> list[Detection]:
        if not dets:
            return []
        boxes = sorted(dets, key=lambda d: d.confidence, reverse=True)
        keep: list[Detection] = []

        def iou(a: Detection, b: Detection) -> float:
            x1 = max(a.x1, b.x1)
            y1 = max(a.y1, b.y1)
            x2 = min(a.x2, b.x2)
            y2 = min(a.y2, b.y2)
            inter = max(0, x2 - x1) * max(0, y2 - y1)
            if inter <= 0:
                return 0.0
            area_a = max(1, (a.x2 - a.x1) * (a.y2 - a.y1))
            area_b = max(1, (b.x2 - b.x1) * (b.y2 - b.y1))
            return inter / float(area_a + area_b - inter)

        while boxes:
            best = boxes.pop(0)
            keep.append(best)
            boxes = [b for b in boxes if iou(best, b) < iou_thresh]
        return keep


def draw_detections(frame_rgb: np.ndarray, detections: list[Detection]) -> np.ndarray:
    import cv2

    out = frame_rgb.copy()
    for det in detections:
        color = (46, 204, 113) if det.label == "recyclable" else (231, 76, 60)
        cv2.rectangle(out, (det.x1, det.y1), (det.x2, det.y2), color, 2)
        caption = f"{det.label} {det.confidence:.2f}"
        cv2.putText(
            out,
            caption,
            (det.x1, max(20, det.y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )
    return out
