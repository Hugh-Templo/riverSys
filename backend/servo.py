"""GPIO servo control for biodegradable / non-biodegradable segregation."""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Optional

from . import config

logger = logging.getLogger(__name__)


class ServoSorter:
    """One 20kg servo: left = biodegradable, right = non-biodegradable, center = home."""

    def __init__(self) -> None:
        self._servo = None
        self._lock = threading.Lock()
        self._busy = False
        self._last_action = "home"
        self._last_error: Optional[str] = None
        self._enabled = config.SERVO_ENABLED
        self._current_angle = float(config.SERVO_HOME_ANGLE)
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
            self._current_angle = float(config.SERVO_HOME_ANGLE)
            logger.info(
                "Servo ready on GPIO %s (move=%.1fs hold=%.1fs)",
                config.SERVO_PIN,
                config.SERVO_MOVE_SECONDS,
                config.SERVO_HOLD_SECONDS,
            )
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
            "angle": self._current_angle,
            "pin": config.SERVO_PIN,
            "error": self._last_error,
        }

    def home(self) -> dict:
        return self._move_smooth(config.SERVO_HOME_ANGLE, "home")

    def sort_biodegradable(self) -> dict:
        return self._start_actuate(config.SERVO_RECYCLABLE_ANGLE, "biodegradable")

    def sort_non_biodegradable(self) -> dict:
        return self._start_actuate(config.SERVO_NON_RECYCLABLE_ANGLE, "non_biodegradable")

    # Back-compat wrappers
    def sort_recyclable(self) -> dict:
        return self.sort_biodegradable()

    def sort_non_recyclable(self) -> dict:
        return self.sort_non_biodegradable()

    def sort_label(self, label: str) -> dict:
        if label in {"biodegradable", "recyclable"}:
            return self.sort_biodegradable()
        if label in {"non_biodegradable", "non_recyclable"}:
            return self.sort_non_biodegradable()
        return {"ok": False, "error": f"unknown label: {label}"}

    def set_angle(self, angle: float, action: str = "manual") -> dict:
        """Hold an absolute angle immediately (for calibration nudges)."""
        return self._apply_angle(float(angle), action)

    def swap_bins(self) -> dict:
        """Swap which physical side is biodegradable vs non-biodegradable."""
        config.SERVO_RECYCLABLE_ANGLE, config.SERVO_NON_RECYCLABLE_ANGLE = (
            config.SERVO_NON_RECYCLABLE_ANGLE,
            config.SERVO_RECYCLABLE_ANGLE,
        )
        return {
            "ok": True,
            "biodegradable_angle": config.SERVO_RECYCLABLE_ANGLE,
            "non_biodegradable_angle": config.SERVO_NON_RECYCLABLE_ANGLE,
        }

    def _start_actuate(self, angle: float, action: str) -> dict:
        """Start a drop cycle in the background so the UI stays responsive."""
        with self._lock:
            if self._busy:
                return {"ok": False, "busy": True, "action": self._last_action}
            self._busy = True

        def worker() -> None:
            try:
                self._actuate(angle, action)
            finally:
                with self._lock:
                    self._busy = False

        threading.Thread(target=worker, daemon=True, name=f"servo-{action}").start()
        return {
            "ok": True,
            "started": True,
            "action": action,
            "angle": float(angle),
            "move_seconds": config.SERVO_MOVE_SECONDS,
            "hold_seconds": config.SERVO_HOLD_SECONDS,
        }

    def _actuate(self, angle: float, action: str) -> dict:
        # Smooth travel out (5s) → hold at drop (3s) → smooth return home (5s).
        moved = self._move_smooth(angle, action, config.SERVO_MOVE_SECONDS)
        if not moved.get("ok", False):
            return moved
        time.sleep(config.SERVO_HOLD_SECONDS)
        self._move_smooth(config.SERVO_HOME_ANGLE, "home", config.SERVO_MOVE_SECONDS)
        return moved

    def _move_smooth(
        self,
        angle: float,
        action: str,
        duration: Optional[float] = None,
    ) -> dict:
        """Ease from the current angle to the target over ``duration`` seconds."""
        duration = float(config.SERVO_MOVE_SECONDS if duration is None else duration)
        target = max(-90.0, min(90.0, float(angle)))
        start = float(self._current_angle)
        delta = target - start

        if duration <= 0 or abs(delta) < 0.05:
            return self._apply_angle(target, action)

        # ~40 Hz updates keep PWM smooth without flooding the pin factory.
        steps = max(1, int(duration * 40))
        dt = duration / steps
        logger.info(
            "Servo smooth %s: %.1f° → %.1f° over %.1fs",
            action,
            start,
            target,
            duration,
        )
        last_result: dict = {"ok": True, "action": action, "angle": target}
        for i in range(1, steps + 1):
            t = i / steps
            # Cosine ease-in/out — slower at the ends, steadier in the middle.
            eased = 0.5 - 0.5 * math.cos(math.pi * t)
            step_angle = start + delta * eased
            label = action if i == steps else "moving"
            last_result = self._apply_angle(step_angle, label)
            if not last_result.get("ok", False):
                return last_result
            time.sleep(dt)

        self._last_action = action
        last_result["action"] = action
        last_result["angle"] = target
        return last_result

    def _apply_angle(self, angle: float, action: str) -> dict:
        angle = max(-90.0, min(90.0, float(angle)))
        self._last_action = action
        self._current_angle = angle
        if self._servo is None:
            if action != "moving":
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
