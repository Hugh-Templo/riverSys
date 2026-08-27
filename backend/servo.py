"""GPIO servo control for recyclable / non-recyclable segregation."""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional

from . import config

logger = logging.getLogger(__name__)


class ServoSorter:
    """One 20kg servo: left = recyclable, right = non-recyclable, center = home."""

    def __init__(self) -> None:
        self._servo = None
        self._lock = threading.Lock()
        self._busy = False
        self._last_action = "home"
        self._last_error: Optional[str] = None
        self._enabled = config.SERVO_ENABLED
        self._init_servo()

    def _init_servo(self) -> None:
        if not self._enabled:
            logger.info("Servo disabled via SERVO_ENABLED=0")
            return
        try:
            from gpiozero import AngularServo
            from gpiozero.pins.lgpio import LGPIOFactory

            factory = LGPIOFactory()
            self._servo = AngularServo(
                config.SERVO_PIN,
                min_angle=-90,
                max_angle=90,
                min_pulse_width=config.SERVO_MIN_PULSE,
                max_pulse_width=config.SERVO_MAX_PULSE,
                pin_factory=factory,
            )
            self._servo.angle = config.SERVO_HOME_ANGLE
            logger.info("Servo ready on GPIO %s", config.SERVO_PIN)
        except Exception as exc:  # noqa: BLE001
            self._last_error = str(exc)
            self._servo = None
            logger.warning("Servo init failed (%s) — running in dry-run mode", exc)

    @property
    def status(self) -> dict:
        return {
            "enabled": self._enabled,
            "connected": self._servo is not None,
            "busy": self._busy,
            "last_action": self._last_action,
            "pin": config.SERVO_PIN,
            "error": self._last_error,
        }

    def home(self) -> dict:
        return self._move(config.SERVO_HOME_ANGLE, "home")

    def sort_recyclable(self) -> dict:
        return self._actuate(config.SERVO_RECYCLABLE_ANGLE, "recyclable")

    def sort_non_recyclable(self) -> dict:
        return self._actuate(config.SERVO_NON_RECYCLABLE_ANGLE, "non_recyclable")

    def sort_label(self, label: str) -> dict:
        if label == "recyclable":
            return self.sort_recyclable()
        if label == "non_recyclable":
            return self.sort_non_recyclable()
        return {"ok": False, "error": f"unknown label: {label}"}

    def set_angle(self, angle: float, action: str = "manual") -> dict:
        """Hold an absolute angle without auto-returning home (for calibration)."""
        return self._move(float(angle), action)

    def swap_bins(self) -> dict:
        """Swap which physical side is recyclable vs non-recyclable."""
        config.SERVO_RECYCLABLE_ANGLE, config.SERVO_NON_RECYCLABLE_ANGLE = (
            config.SERVO_NON_RECYCLABLE_ANGLE,
            config.SERVO_RECYCLABLE_ANGLE,
        )
        return {
            "ok": True,
            "recyclable_angle": config.SERVO_RECYCLABLE_ANGLE,
            "non_recyclable_angle": config.SERVO_NON_RECYCLABLE_ANGLE,
        }

    def _actuate(self, angle: float, action: str) -> dict:
        with self._lock:
            if self._busy:
                return {"ok": False, "busy": True, "action": self._last_action}
            self._busy = True

        try:
            moved = self._move(angle, action)
            time.sleep(config.SERVO_HOLD_SECONDS)
            self._move(config.SERVO_HOME_ANGLE, "home")
            return moved
        finally:
            with self._lock:
                self._busy = False

    def _move(self, angle: float, action: str) -> dict:
        angle = max(-90.0, min(90.0, float(angle)))
        self._last_action = action
        if self._servo is None:
            logger.info("Dry-run servo → %s (%.1f°)", action, angle)
            return {
                "ok": True,
                "dry_run": True,
                "action": action,
                "angle": angle,
            }
        try:
            self._servo.angle = angle
            return {"ok": True, "dry_run": False, "action": action, "angle": angle}
        except Exception as exc:  # noqa: BLE001
            self._last_error = str(exc)
            logger.error("Servo move failed: %s", exc)
            return {"ok": False, "error": str(exc), "action": action}
