#!/usr/bin/env python3
"""Local OpenCV display with training prompts, drop zones, and servo calibration."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from backend import config
from backend.calibration import (
    load_calibration,
    point_in_zone,
    save_calibration,
    zone_pixels,
)
from backend.camera import CameraService
from backend.detector import Detection, PlasticDetector, draw_detections
from backend.servo import ServoSorter
from backend.train_capture import save_training_sample


@dataclass
class UiButton:
    name: str
    x1: int
    y1: int
    x2: int
    y2: int
    label: str
    color: tuple[int, int, int]


class DisplayApp:
    def __init__(self, auto_sort: bool) -> None:
        self.auto_sort = auto_sort
        self.training = False
        self.calibrating = False
        self.cal_step = 0
        self.cal_angle = 0.0
        self.pending: Optional[tuple[np.ndarray, Detection]] = None
        self.last_train_prompt = 0.0
        self.last_sort = 0.0
        self.status_msg = ""
        self.status_until = 0.0
        self.click_action: Optional[str] = None
        self.last_click = (-1, -1)
        self.buttons: list[UiButton] = []
        self.left_zone_px = (0, 0, 0, 0)
        self.right_zone_px = (0, 0, 0, 0)
        self.cal = load_calibration()
        self.camera = CameraService()
        self.detector = PlasticDetector()
        self.sorter = ServoSorter()
        self.window = "Plastic Segregation"

    def set_status(self, msg: str, seconds: float = 2.5) -> None:
        self.status_msg = msg
        self.status_until = time.monotonic() + seconds

    def on_mouse(self, event: int, x: int, y: int, _flags: int, _param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        self.last_click = (x, y)
        for btn in self.buttons:
            if btn.x1 <= x <= btn.x2 and btn.y1 <= y <= btn.y2:
                self.click_action = btn.name
                return
        if self.pending is not None:
            if point_in_zone(x, y, self.left_zone_px):
                self.click_action = "zone_left"
            elif point_in_zone(x, y, self.right_zone_px):
                self.click_action = "zone_right"

    def _build_buttons(self, w: int, h: int) -> None:
        bar_h = 56
        y1 = h - bar_h - 8
        y2 = h - 8
        gap = 8
        labels = [
            ("train", f"TRAIN {'ON' if self.training else 'OFF'}", (40, 180, 90) if self.training else (70, 70, 70)),
            ("calibrate", "CALIBRATE", (40, 140, 220)),
            ("home", "HOME", (120, 120, 120)),
            ("auto", f"AUTO {'ON' if self.auto_sort else 'OFF'}", (40, 180, 90) if self.auto_sort else (70, 70, 70)),
            ("swap", "SWAP BINS", (180, 120, 40)),
        ]
        n = len(labels)
        bw = (w - gap * (n + 1)) // n
        self.buttons = []
        for i, (name, text, color) in enumerate(labels):
            x1 = gap + i * (bw + gap)
            x2 = x1 + bw
            self.buttons.append(UiButton(name, x1, y1, x2, y2, text, color))

    def _draw_buttons(self, frame_bgr: np.ndarray) -> None:
        for btn in self.buttons:
            cv2.rectangle(frame_bgr, (btn.x1, btn.y1), (btn.x2, btn.y2), btn.color, -1)
            cv2.rectangle(frame_bgr, (btn.x1, btn.y1), (btn.x2, btn.y2), (255, 255, 255), 1)
            tw = cv2.getTextSize(btn.label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)[0][0]
            tx = btn.x1 + max(4, (btn.x2 - btn.x1 - tw) // 2)
            cv2.putText(
                frame_bgr,
                btn.label,
                (tx, btn.y1 + 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    def _draw_zones(self, frame_bgr: np.ndarray) -> None:
        """Draw LEFT and RIGHT physical drop bins on the camera view."""
        h, w = frame_bgr.shape[:2]
        self.left_zone_px = zone_pixels(self.cal.left_zone, w, h)
        self.right_zone_px = zone_pixels(self.cal.right_zone, w, h)

        for side, box in (("left", self.left_zone_px), ("right", self.right_zone_px)):
            label = self.cal.label_for_side(side)  # type: ignore[arg-type]
            angle = self.cal.angle_for_side(side)  # type: ignore[arg-type]
            color = (46, 204, 113) if label == "recyclable" else (60, 76, 231)
            x1, y1, x2, y2 = box
            overlay = frame_bgr.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.addWeighted(overlay, 0.20, frame_bgr, 0.80, 0, frame_bgr)
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 3)
            cv2.putText(
                frame_bgr,
                f"{side.upper()} DROP",
                (x1 + 10, y1 + 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame_bgr,
                label.replace("_", " ").upper(),
                (x1 + 10, y1 + 64),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame_bgr,
                f"servo {angle:+.0f} deg",
                (x1 + 10, y1 + 94),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    def _draw_train_prompt(self, frame_bgr: np.ndarray, det: Detection) -> None:
        h, w = frame_bgr.shape[:2]
        panel = np.zeros_like(frame_bgr)
        cv2.rectangle(panel, (w // 8, h // 5), (7 * w // 8, 3 * h // 5), (20, 20, 20), -1)
        cv2.addWeighted(panel, 0.75, frame_bgr, 0.25, 0, frame_bgr)
        cv2.rectangle(frame_bgr, (det.x1, det.y1), (det.x2, det.y2), (0, 255, 255), 3)
        lines = [
            "TRAINING LABEL",
            "Choose class for detected object",
            "1 = recyclable     2 = non-recyclable",
            "or click LEFT / RIGHT drop box",
            "ESC = skip",
        ]
        y = h // 5 + 45
        for i, line in enumerate(lines):
            cv2.putText(
                frame_bgr,
                line,
                (w // 8 + 24, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0 if i == 0 else 0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            y += 40

    def _draw_cal_prompt(self, frame_bgr: np.ndarray) -> None:
        h, w = frame_bgr.shape[:2]
        target = "RECYCLABLE" if self.cal_step == 0 else "NON-RECYCLABLE"
        lines = [
            f"SERVO CALIBRATION — {target}",
            f"Angle now: {self.cal_angle:+.0f} deg  (servo holds this position)",
            "[ / ] = +/-5 deg     - / = = +/-1 deg",
            "ENTER = save this side     S = swap bins",
            "ESC = cancel",
        ]
        cv2.rectangle(frame_bgr, (20, 20), (w - 20, 30 + 38 * len(lines)), (0, 0, 0), -1)
        y = 55
        for line in lines:
            cv2.putText(frame_bgr, line, (36, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2, cv2.LINE_AA)
            y += 38

    def _accept_label(self, label: str) -> None:
        if self.pending is None:
            return
        frame, det = self.pending
        path = save_training_sample(frame, det, label)
        self.pending = None
        self.last_train_prompt = time.monotonic()
        count = len(list((config.YOLO_DATASET / "labels" / "train").glob("*.txt")))
        name = path.name if path else label
        self.set_status(f"Saved {label} ({count} labels) · {name}")
        self.sorter.sort_label(label)

    def _start_calibration(self) -> None:
        self.calibrating = True
        self.cal_step = 0
        self.cal_angle = float(self.cal.recyclable_angle)
        self.pending = None
        self.sorter.set_angle(self.cal_angle, "calibrate")
        self.set_status("Set RECYCLABLE servo angle, then ENTER")

    def _finish_cal_step(self) -> None:
        if self.cal_step == 0:
            self.cal.recyclable_angle = self.cal_angle
            self.cal_step = 1
            self.cal_angle = float(self.cal.non_recyclable_angle)
            self.sorter.set_angle(self.cal_angle, "calibrate")
            self.set_status("Set NON-RECYCLABLE servo angle, then ENTER")
            return
        self.cal.non_recyclable_angle = self.cal_angle
        save_calibration(self.cal)
        self.calibrating = False
        self.sorter.home()
        self.set_status(
            f"Calibration saved: LEFT={self.cal.label_for_side('left')} "
            f"{self.cal.angle_for_side('left'):+.0f}° / "
            f"RIGHT={self.cal.label_for_side('right')} "
            f"{self.cal.angle_for_side('right'):+.0f}°"
        )

    def _nudge_cal(self, delta: float) -> None:
        self.cal_angle = max(-90.0, min(90.0, self.cal_angle + delta))
        self.sorter.set_angle(self.cal_angle, "calibrate")

    def _swap_bins(self) -> None:
        self.cal.swap_sides()
        self.cal.apply_to_config()
        save_calibration(self.cal)
        if self.calibrating:
            self.cal_angle = self.cal.recyclable_angle if self.cal_step == 0 else self.cal.non_recyclable_angle
            self.sorter.set_angle(self.cal_angle, "calibrate")
        self.set_status(
            f"LEFT={self.cal.label_for_side('left')} @ {self.cal.angle_for_side('left'):+.0f}°  |  "
            f"RIGHT={self.cal.label_for_side('right')} @ {self.cal.angle_for_side('right'):+.0f}°"
        )

    def _handle(self, action: Optional[str], key: int) -> bool:
        if action == "quit" or key == ord("q"):
            return False

        if self.calibrating:
            if key == 27:
                self.calibrating = False
                self.sorter.home()
                self.set_status("Calibration cancelled")
            elif key in (13, 10):
                self._finish_cal_step()
            elif key == ord("["):
                self._nudge_cal(-5)
            elif key == ord("]"):
                self._nudge_cal(5)
            elif key == ord("-"):
                self._nudge_cal(-1)
            elif key in (ord("="), ord("+")):
                self._nudge_cal(1)
            elif key == ord("s") or action == "swap":
                self._swap_bins()
            return True

        if self.pending is not None:
            if key in (ord("1"), ord("r")):
                self._accept_label("recyclable")
            elif key in (ord("2"), ord("n")):
                self._accept_label("non_recyclable")
            elif action == "zone_left":
                self._accept_label(self.cal.label_for_side("left"))
            elif action == "zone_right":
                self._accept_label(self.cal.label_for_side("right"))
            elif key == 27:
                self.pending = None
                self.last_train_prompt = time.monotonic()
                self.set_status("Skipped label")
            return True

        if action == "train" or key == ord("t"):
            self.training = not self.training
            self.pending = None
            self.set_status(f"Training {'ON' if self.training else 'OFF'}")
        elif action == "calibrate" or key == ord("c"):
            self._start_calibration()
        elif action == "home" or key == ord("h"):
            self.sorter.home()
            self.set_status("Servo home")
        elif action == "auto" or key == ord("a"):
            self.auto_sort = not self.auto_sort
            self.set_status(f"Auto-sort {'ON' if self.auto_sort else 'OFF'}")
        elif action == "swap" or key == ord("s"):
            self._swap_bins()
        elif key == ord("r"):
            self.sorter.sort_recyclable()
        elif key == ord("n"):
            self.sorter.sort_non_recyclable()
        return True

    def run(self) -> int:
        self.camera.start()
        # Prefer a simple Qt window; avoids some Wayland/GLib teardown glitches on Pi OS.
        try:
            cv2.namedWindow(self.window, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        except Exception:
            cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        try:
            cv2.resizeWindow(self.window, config.CAMERA_WIDTH, config.CAMERA_HEIGHT)
        except Exception:
            pass
        cv2.setMouseCallback(self.window, self.on_mouse)

        print(
            "Controls:\n"
            "  TRAIN button / t   toggle training labels\n"
            "  CALIBRATE / c      set recyclable & non-recyclable servo angles\n"
            "  SWAP BINS / s      flip which side is which\n"
            "  AUTO / a           auto drop with servo\n"
            "  r / n              manual drop    h home    q quit\n"
            "  Training prompt: 1=recyclable  2=non-recyclable  or click LEFT/RIGHT box\n"
            "  Calibration: [ ] +/-5°  - = +/-1°  ENTER save side"
        )

        try:
            while True:
                frame = self.camera.get_frame()
                if frame is None:
                    time.sleep(0.02)
                    continue

                h, w = frame.shape[:2]
                self._build_buttons(w, h)

                action = self.click_action
                self.click_action = None

                detections = [] if self.calibrating else self.detector.detect(frame)
                annotated = draw_detections(frame, detections)
                show = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
                self._draw_zones(show)

                best: Optional[Detection] = None
                decision = None
                if detections:
                    best = max(detections, key=lambda d: d.confidence)
                    if best.confidence >= config.CONF_THRESHOLD:
                        decision = best.label

                now = time.monotonic()
                if (
                    self.training
                    and not self.calibrating
                    and self.pending is None
                    and best is not None
                    and best.confidence >= config.CONF_THRESHOLD
                    and now - self.last_train_prompt >= config.TRAIN_PROMPT_COOLDOWN
                ):
                    self.pending = (frame.copy(), best)

                if self.pending is not None:
                    self._draw_train_prompt(show, self.pending[1])
                if self.calibrating:
                    self._draw_cal_prompt(show)

                hud = (
                    f"dets={len(detections)}  "
                    f"LEFT={self.cal.label_for_side('left')}@{self.cal.angle_for_side('left'):+.0f}°  "
                    f"RIGHT={self.cal.label_for_side('right')}@{self.cal.angle_for_side('right'):+.0f}°  "
                    f"train={'ON' if self.training else 'OFF'}  auto={'ON' if self.auto_sort else 'OFF'}"
                )
                if decision and self.pending is None and not self.calibrating:
                    hud += f"  |  {decision}"
                cv2.putText(show, hud, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
                if self.status_msg and now < self.status_until:
                    cv2.putText(
                        show,
                        self.status_msg,
                        (16, 56),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                self._draw_buttons(show)

                if (
                    self.auto_sort
                    and not self.training
                    and not self.calibrating
                    and self.pending is None
                    and decision
                    and not self.sorter.status["busy"]
                    and now - self.last_sort >= config.SORT_COOLDOWN_SECONDS
                ):
                    side = self.cal.side_for_label(decision)
                    angle = self.cal.angle_for_label(decision)
                    self.sorter.sort_label(decision)
                    self.last_sort = now
                    self.set_status(f"Dropped {decision} → {side.upper()} @ {angle:+.0f}°")

                cv2.imshow(self.window, show)
                key = cv2.waitKey(1) & 0xFF
                if not self._handle(action, key):
                    break
        finally:
            self._shutdown_ui()
        return 0

    def _shutdown_ui(self) -> None:
        """Tear down camera/servo/window in an order that avoids Qt/GLib unref noise."""
        try:
            cv2.setMouseCallback(self.window, lambda *_args: None)
        except Exception:
            pass
        try:
            self.camera.stop()
        except Exception:
            pass
        try:
            self.sorter.home()
        except Exception:
            pass
        try:
            cv2.waitKey(1)
            cv2.destroyWindow(self.window)
            cv2.waitKey(1)
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        except Exception:
            pass


def main() -> int:
    parser = argparse.ArgumentParser(description="Plastic segregation local display")
    parser.add_argument("--no-servo", action="store_true")
    parser.add_argument("--auto-sort", action="store_true", default=True)
    parser.add_argument("--no-auto-sort", action="store_true")
    args = parser.parse_args()

    config.ensure_directories()
    if args.no_servo:
        config.SERVO_ENABLED = False

    app = DisplayApp(auto_sort=args.auto_sort and not args.no_auto_sort)
    return app.run()


if __name__ == "__main__":
    raise SystemExit(main())
