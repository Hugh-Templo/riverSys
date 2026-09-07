"""Persistent object memory so TRAIN labels are remembered across sessions."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from . import config

logger = logging.getLogger(__name__)

MEMORY_DIR = config.SYSTEM_ROOT / "object_memory"
MEMORY_INDEX = MEMORY_DIR / "index.json"


@dataclass
class MemoryHit:
    label: str
    score: float
    name: str


class ObjectMemory:
    """HSV-histogram templates learned from the yellow TRAIN box (1/2 keys)."""

    def __init__(self, match_threshold: float = 0.72) -> None:
        self.match_threshold = match_threshold
        self.entries: list[dict] = []
        self._hists: list[np.ndarray] = []
        config.ensure_directories()
        MEMORY_DIR.mkdir(parents=True, exist_ok=True)
        self.load()

    def __len__(self) -> int:
        return len(self.entries)

    def load(self) -> None:
        self.entries = []
        self._hists = []
        if not MEMORY_INDEX.exists():
            return
        try:
            data = json.loads(MEMORY_INDEX.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Memory index load failed: %s", exc)
            return
        for item in data.get("items", []):
            hist_path = MEMORY_DIR / item.get("hist", "")
            if not hist_path.exists():
                continue
            try:
                hist = np.load(str(hist_path))
            except Exception:  # noqa: BLE001
                continue
            # Migrate legacy class names from earlier builds.
            label = str(item.get("label", ""))
            if label == "recyclable":
                item = {**item, "label": "biodegradable"}
            elif label == "non_recyclable":
                item = {**item, "label": "non_biodegradable"}
            self.entries.append(item)
            self._hists.append(hist)
        logger.info("Object memory loaded: %s templates", len(self.entries))

    def _save_index(self) -> None:
        MEMORY_INDEX.write_text(
            json.dumps({"items": self.entries}, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def feature_hist(roi_rgb: np.ndarray) -> Optional[np.ndarray]:
        if roi_rgb is None or roi_rgb.size == 0:
            return None
        roi = cv2.resize(roi_rgb, (96, 96), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [32], [0, 180])
        hist_s = cv2.calcHist([hsv], [1], None, [32], [0, 256])
        hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256])
        hist = np.concatenate([hist_h.flatten(), hist_s.flatten(), hist_v.flatten()]).astype(np.float32)
        norm = float(np.linalg.norm(hist)) + 1e-6
        return hist / norm

    def remember(self, roi_rgb: np.ndarray, label: str) -> str:
        """Store a new template for biodegradable / non_biodegradable."""
        if label in {"recyclable"}:
            label = "biodegradable"
        elif label in {"non_recyclable"}:
            label = "non_biodegradable"
        if label not in {"biodegradable", "non_biodegradable"}:
            raise ValueError(label)
        hist = self.feature_hist(roi_rgb)
        if hist is None:
            raise ValueError("empty ROI")

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        name = f"{label}_{stamp}"
        hist_path = MEMORY_DIR / f"{name}.npy"
        thumb_path = MEMORY_DIR / f"{name}.jpg"
        np.save(str(hist_path), hist)
        bgr = cv2.cvtColor(cv2.resize(roi_rgb, (128, 128)), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(thumb_path), bgr)

        item = {
            "name": name,
            "label": label,
            "hist": hist_path.name,
            "thumb": thumb_path.name,
            "created": stamp,
        }
        self.entries.append(item)
        self._hists.append(hist)
        self._save_index()
        logger.info("Remembered %s (%s total)", name, len(self.entries))
        return name

    def match(self, roi_rgb: np.ndarray) -> Optional[MemoryHit]:
        if not self._hists:
            return None
        hist = self.feature_hist(roi_rgb)
        if hist is None:
            return None
        best_i = -1
        best_score = -1.0
        for i, ref in enumerate(self._hists):
            score = float(np.dot(hist, ref))
            if score > best_score:
                best_score = score
                best_i = i
        if best_i < 0 or best_score < self.match_threshold:
            return None
        entry = self.entries[best_i]
        return MemoryHit(label=entry["label"], score=best_score, name=entry["name"])


def center_train_box(frame_w: int, frame_h: int, frac: float = 0.36) -> tuple[int, int, int, int]:
    """Yellow TRAIN capture square in the middle of the frame."""
    side = int(min(frame_w, frame_h) * frac)
    side = max(80, side)
    x1 = (frame_w - side) // 2
    y1 = (frame_h - side) // 2
    return x1, y1, x1 + side, y1 + side
