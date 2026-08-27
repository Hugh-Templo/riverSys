"""Persist servo angles and left/right drop-bin mapping on the USB stick."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from . import config

logger = logging.getLogger(__name__)

CALIBRATION_PATH = config.SYSTEM_ROOT / "calibration.json"
Side = Literal["left", "right"]


@dataclass
class Calibration:
    """Physical drop bins are always LEFT and RIGHT on the camera view."""

    recyclable_angle: float = -60.0
    non_recyclable_angle: float = 60.0
    home_angle: float = 0.0
    # Which screen side receives recyclable plastics (the other side is non-recyclable).
    recyclable_side: Side = "left"
    # Normalized square drop areas: (x, y, size) — left and right bins.
    left_zone: tuple[float, float, float] = (0.04, 0.28, 0.28)
    right_zone: tuple[float, float, float] = (0.68, 0.28, 0.28)

    def apply_to_config(self) -> None:
        config.SERVO_RECYCLABLE_ANGLE = float(self.recyclable_angle)
        config.SERVO_NON_RECYCLABLE_ANGLE = float(self.non_recyclable_angle)
        config.SERVO_HOME_ANGLE = float(self.home_angle)

    def side_for_label(self, label: str) -> Side:
        if label == "recyclable":
            return self.recyclable_side
        return "right" if self.recyclable_side == "left" else "left"

    def label_for_side(self, side: Side) -> str:
        if side == self.recyclable_side:
            return "recyclable"
        return "non_recyclable"

    def angle_for_side(self, side: Side) -> float:
        label = self.label_for_side(side)
        return self.recyclable_angle if label == "recyclable" else self.non_recyclable_angle

    def angle_for_label(self, label: str) -> float:
        return self.recyclable_angle if label == "recyclable" else self.non_recyclable_angle

    def zone_for_side(self, side: Side) -> tuple[float, float, float]:
        return self.left_zone if side == "left" else self.right_zone

    def swap_sides(self) -> None:
        self.recyclable_side = "right" if self.recyclable_side == "left" else "left"
        self.recyclable_angle, self.non_recyclable_angle = (
            self.non_recyclable_angle,
            self.recyclable_angle,
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["left_zone"] = list(self.left_zone)
        data["right_zone"] = list(self.right_zone)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Calibration":
        # Backward compatible with older recyclable_zone / non_recyclable_zone keys.
        left = data.get("left_zone") or data.get("recyclable_zone") or [0.04, 0.28, 0.28]
        right = data.get("right_zone") or data.get("non_recyclable_zone") or [0.68, 0.28, 0.28]
        side = str(data.get("recyclable_side", "left")).lower()
        if side not in ("left", "right"):
            side = "left"
        return cls(
            recyclable_angle=float(data.get("recyclable_angle", -60.0)),
            non_recyclable_angle=float(data.get("non_recyclable_angle", 60.0)),
            home_angle=float(data.get("home_angle", 0.0)),
            recyclable_side=side,  # type: ignore[arg-type]
            left_zone=(float(left[0]), float(left[1]), float(left[2])),
            right_zone=(float(right[0]), float(right[1]), float(right[2])),
        )


def load_calibration() -> Calibration:
    config.ensure_directories()
    if not CALIBRATION_PATH.exists():
        cal = Calibration(
            recyclable_angle=config.SERVO_RECYCLABLE_ANGLE,
            non_recyclable_angle=config.SERVO_NON_RECYCLABLE_ANGLE,
            home_angle=config.SERVO_HOME_ANGLE,
        )
        save_calibration(cal)
        cal.apply_to_config()
        return cal
    try:
        data = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
        cal = Calibration.from_dict(data)
        # Migrate old bottom-corner layout to left/right drop bins if needed.
        if "left_zone" not in data:
            cal.left_zone = (0.04, 0.28, 0.28)
            cal.right_zone = (0.68, 0.28, 0.28)
            cal.recyclable_side = "left"
            save_calibration(cal)
        cal.apply_to_config()
        logger.info("Loaded calibration from %s", CALIBRATION_PATH)
        return cal
    except Exception as exc:  # noqa: BLE001
        logger.warning("Calibration load failed (%s); using defaults", exc)
        cal = Calibration()
        cal.apply_to_config()
        return cal


def save_calibration(cal: Calibration) -> Path:
    config.ensure_directories()
    CALIBRATION_PATH.write_text(json.dumps(cal.to_dict(), indent=2), encoding="utf-8")
    cal.apply_to_config()
    logger.info("Saved calibration → %s", CALIBRATION_PATH)
    return CALIBRATION_PATH


def zone_pixels(
    zone: tuple[float, float, float], frame_w: int, frame_h: int
) -> tuple[int, int, int, int]:
    """Return square drop area as (x1, y1, x2, y2) in pixels."""
    x_n, y_n, size_n = zone
    side = int(min(frame_w, frame_h) * size_n)
    x1 = int(frame_w * x_n)
    y1 = int(frame_h * y_n)
    x2 = min(frame_w - 1, x1 + side)
    y2 = min(frame_h - 1, y1 + side)
    return x1, y1, x2, y2


def point_in_zone(cx: int, cy: int, zone_xyxy: tuple[int, int, int, int]) -> bool:
    x1, y1, x2, y2 = zone_xyxy
    return x1 <= cx <= x2 and y1 <= cy <= y2
