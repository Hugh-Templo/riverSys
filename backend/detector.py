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
            custom_markers = {
                "recyclable",
                "non_recyclable",
                "biodegradable",
                "non_biodegradable",
            }
            if name_values & custom_markers:
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
            dets = self._detect_yolo(frame_rgb)
        elif config.USE_OPENCV_FALLBACK:
            dets = self._detect_opencv(frame_rgb)
        else:
            dets = []
        return self._apply_memory(frame_rgb, dets)

    def set_memory(self, memory) -> None:
        """Attach ObjectMemory used to remember TRAIN 1/2 labels."""
        self.memory = memory

    def _apply_memory(self, frame_rgb: np.ndarray, dets: list[Detection]) -> list[Detection]:
        memory = getattr(self, "memory", None)
        if memory is None or len(memory) == 0:
            return dets

        import cv2

        h, w = frame_rgb.shape[:2]
        out: list[Detection] = []
        for det in dets:
            x1, y1 = max(0, det.x1), max(0, det.y1)
            x2, y2 = min(w, det.x2), min(h, det.y2)
            roi = frame_rgb[y1:y2, x1:x2]
            hit = memory.match(roi)
            if hit is not None:
                out.append(
                    Detection(
                        label=hit.label,
                        confidence=max(det.confidence, float(hit.score)),
                        x1=det.x1,
                        y1=det.y1,
                        x2=det.x2,
                        y2=det.y2,
                        source="memory",
                    )
                )
            else:
                out.append(det)

        # If nothing else fired, still recognize a remembered object in the center box.
        if not out:
            from .object_memory import center_train_box

            x1, y1, x2, y2 = center_train_box(w, h)
            hit = memory.match(frame_rgb[y1:y2, x1:x2])
            if hit is not None:
                out.append(
                    Detection(
                        label=hit.label,
                        confidence=float(hit.score),
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                        source="memory",
                    )
                )
        _ = cv2
        return out

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
        if raw in config.BIODEGRADABLE_CLASSES | config.NON_BIODEGRADABLE_CLASSES:
            if raw in {"recyclable"}:
                return "biodegradable"
            if raw in {"non_recyclable"}:
                return "non_biodegradable"
            return raw

        # Useful when only base COCO yolov8n.pt is present (smoke-test mode).
        biodegradable_aliases = {
            "banana",
            "apple",
            "orange",
            "carrot",
            "broccoli",
            "biodegradable",
            "organic",
            "paper",
            "cardboard",
            "recyclable",  # legacy
        }
        non_biodegradable_aliases = {
            "bottle",
            "cup",
            "wine_glass",
            "plastic_bottle",
            "plastic_bag",
            "bag",
            "cell_phone",
            "book",
            "non_biodegradable",
            "non_recyclable",  # legacy
        }
        if raw in biodegradable_aliases:
            return "biodegradable"
        if raw in non_biodegradable_aliases:
            return "non_biodegradable"
        # Custom floating-plastic / waste class names.
        if raw in {"food", "leaf", "wood", "plant", "compost"}:
            return "biodegradable"
        if raw in {"plastic_film", "vinyl", "fragment", "fragmented_plastic", "film", "hard_plastic", "common_plastic"}:
            return "non_biodegradable"
        if any(k in raw for k in ("bio", "organic", "food", "paper", "cardboard")):
            return "biodegradable"
        if any(k in raw for k in ("plastic", "bottle", "bag", "film", "vinyl", "fragment", "wrapper")):
            return "non_biodegradable"
        if self.mode == "yolo_custom":
            if raw in ("biodegradable", "non_biodegradable"):
                return raw
            if raw == "recyclable":
                return "biodegradable"
            if raw == "non_recyclable":
                return "non_biodegradable"
            return None
        return None

    def _detect_opencv(self, frame_rgb: np.ndarray) -> list[Detection]:
        """Heuristic detector on the full frame (no digital zoom — that only shrank boxes)."""
        import cv2

        # Ignore covered / near-black frames (they create noisy false positives).
        gray_full = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        if float(gray_full.mean()) < config.OPENCV_MIN_MEAN_LUMA or float(gray_full.std()) < config.OPENCV_MIN_LUMA_STD:
            return []

        # Ignore blurry frames (common while continuous AF is hunting).
        # Measure sharpness on a fixed-size center patch so the threshold is stable.
        fh, fw = gray_full.shape[:2]
        cw, ch = int(fw * 0.45), int(fh * 0.45)
        cx0, cy0 = (fw - cw) // 2, (fh - ch) // 2
        center = gray_full[cy0 : cy0 + ch, cx0 : cx0 + cw]
        center_small = cv2.resize(center, (320, 180), interpolation=cv2.INTER_AREA)
        sharpness = float(cv2.Laplacian(center_small, cv2.CV_64F).var())
        if sharpness < config.OPENCV_MIN_SHARPNESS:
            return []

        zoomed, offset_x, offset_y, scale = self._center_zoom(frame_rgb, config.DETECT_ZOOM)
        frame_bgr = cv2.cvtColor(zoomed, cv2.COLOR_RGB2BGR)
        h, w = frame_bgr.shape[:2]

        # CLAHE boosts weak contrast of plastics against background.
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.4, tileGridSize=(8, 8))
        l_ch = clahe.apply(l_ch)
        enhanced = cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)

        blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

        # Background-relative masks (absolute "bright" thresholds match whole indoor frames).
        med = float(np.median(gray))
        bright_rel = np.zeros_like(gray)
        bright_rel[gray > med + 22] = 255
        dark_rel = np.zeros_like(gray)
        dark_rel[gray < med - 28] = 255

        sat = hsv[:, :, 1]
        colorful = np.zeros_like(gray)
        colorful[(sat > 45) & (gray > med + 8)] = 255

        local = cv2.GaussianBlur(gray, (31, 31), 0)
        diff = cv2.absdiff(gray, local)
        _, contrast = cv2.threshold(diff, 14, 255, cv2.THRESH_BINARY)

        edges = cv2.Canny(gray, 35, 110)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

        mask = bright_rel | colorful | contrast
        mask = cv2.bitwise_or(mask, cv2.bitwise_and(dark_rel, edges))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
        edge_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=1)
        mask = cv2.bitwise_or(mask, cv2.bitwise_and(edge_closed, contrast))

        coverage = float(np.count_nonzero(mask)) / float(mask.size)
        if coverage > 0.45:
            return []

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = (h * w) * config.OPENCV_MIN_AREA_FRAC
        max_area = (h * w) * config.OPENCV_MAX_AREA_FRAC
        min_px = config.OPENCV_MIN_BOX_PX

        detections: list[Detection] = []
        full_h, full_w = frame_rgb.shape[:2]
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > max_area:
                continue
            x, y, bw, bh = cv2.boundingRect(contour)
            if bw < min_px or bh < min_px:
                continue
            if x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1:
                if area < min_area * 6:
                    continue

            aspect = bw / max(bh, 1)
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
            solidity = area / max(float(bw * bh), 1.0)

            if aspect > 2.0 or aspect < 0.45 or (solidity < 0.35 and len(approx) > 6):
                # Thin / irregular → often plastic film / bags (non-biodegradable).
                label = "non_biodegradable"
            else:
                # More compact blobs → treat as biodegradable candidate until trained.
                label = "biodegradable"

            roi = gray[y : y + bh, x : x + bw]
            contrast_score = float(np.clip(roi.std() / 35.0, 0.0, 1.0)) if roi.size else 0.0
            ideal = (h * w) * 0.01
            size_score = float(np.clip(1.0 - abs(area - ideal) / max(ideal, 1.0), 0.0, 1.0))
            conf = float(np.clip(0.30 + 0.40 * contrast_score + 0.25 * size_score, 0.28, 0.95))

            fx1 = int(offset_x + x / scale)
            fy1 = int(offset_y + y / scale)
            fx2 = int(offset_x + (x + bw) / scale)
            fy2 = int(offset_y + (y + bh) / scale)

            cx = (fx1 + fx2) / 2.0
            cy = (fy1 + fy2) / 2.0
            if not (full_w * 0.10 <= cx <= full_w * 0.90 and full_h * 0.08 <= cy <= full_h * 0.92):
                continue

            detections.append(
                Detection(
                    label=label,
                    confidence=conf,
                    x1=max(0, fx1),
                    y1=max(0, fy1),
                    x2=min(full_w - 1, fx2),
                    y2=min(full_h - 1, fy2),
                    source="opencv",
                )
            )

        detections = self._nms(detections, iou_thresh=0.35)
        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections[: config.OPENCV_MAX_DETS]

    @staticmethod
    def _center_zoom(frame_rgb: np.ndarray, zoom: float) -> tuple[np.ndarray, int, int, float]:
        """Crop the center and upscale so mid-range objects look larger to the detector."""
        import cv2

        zoom = max(1.0, float(zoom))
        h, w = frame_rgb.shape[:2]
        if zoom <= 1.01:
            return frame_rgb, 0, 0, 1.0
        cw = max(32, int(w / zoom))
        ch = max(32, int(h / zoom))
        x0 = (w - cw) // 2
        y0 = (h - ch) // 2
        crop = frame_rgb[y0 : y0 + ch, x0 : x0 + cw]
        zoomed = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
        scale = w / float(cw)
        return zoomed, x0, y0, scale

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
        color = (46, 204, 113) if det.label == "biodegradable" else (231, 76, 60)
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
