"""Picamera2 capture helpers for Camera Module 3."""

from __future__ import annotations

import logging
import threading
from typing import Optional

import numpy as np

from . import config

logger = logging.getLogger(__name__)


class CameraService:
    """Thread-safe latest-frame camera capture."""

    def __init__(self) -> None:
        self._picam = None
        self._lock = threading.Lock()
        self._latest: Optional[np.ndarray] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._mock = False

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def using_mock(self) -> bool:
        return self._mock

    def start(self) -> None:
        if self._running:
            return

        try:
            from picamera2 import Picamera2

            picam = Picamera2()
            cam_config = picam.create_preview_configuration(
                main={
                    "size": (config.CAMERA_WIDTH, config.CAMERA_HEIGHT),
                    "format": "RGB888",
                },
                controls={"FrameRate": config.CAMERA_FPS},
            )
            picam.configure(cam_config)
            picam.start()
            # Brighten dim indoor/water scenes so OpenCV heuristics can see plastics.
            try:
                picam.set_controls(
                    {
                        "AeEnable": True,
                        "AwbEnable": True,
                        "Brightness": float(getattr(config, "CAMERA_BRIGHTNESS", 0.12)),
                        "Contrast": float(getattr(config, "CAMERA_CONTRAST", 1.15)),
                        "Saturation": float(getattr(config, "CAMERA_SATURATION", 1.1)),
                    }
                )
            except Exception as ctrl_exc:  # noqa: BLE001
                logger.warning("Camera image controls not applied: %s", ctrl_exc)
            self._picam = picam
            self._mock = False
            logger.info(
                "Camera Module 3 started at %sx%s",
                config.CAMERA_WIDTH,
                config.CAMERA_HEIGHT,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Camera unavailable (%s); using mock frames", exc)
            self._picam = None
            self._mock = True

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        if self._picam is not None:
            try:
                self._picam.stop()
                self._picam.close()
            except Exception:  # noqa: BLE001
                pass
            self._picam = None

    def get_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._latest is None:
                return None
            return self._latest.copy()

    def _capture_loop(self) -> None:
        while self._running:
            try:
                if self._picam is not None:
                    frame = self._picam.capture_array()
                    # Picamera2 RGB888 is often delivered as BGR-ish naming; keep RGB for OpenCV draw helpers that expect BGR later.
                    if frame.ndim == 3 and frame.shape[2] == 4:
                        frame = frame[:, :, :3]
                else:
                    frame = self._mock_frame()

                with self._lock:
                    self._latest = frame
            except Exception as exc:  # noqa: BLE001
                logger.error("Capture error: %s", exc)
                with self._lock:
                    self._latest = self._mock_frame()

    @staticmethod
    def _mock_frame() -> np.ndarray:
        h, w = config.CAMERA_HEIGHT, config.CAMERA_WIDTH
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:, :] = (30, 90, 140)  # water-like blue-green
        cv2_available = True
        try:
            import cv2

            cv2.rectangle(frame, (w // 3, h // 3), (w // 2, h // 2), (200, 200, 240), -1)
            cv2.putText(
                frame,
                "MOCK CAMERA",
                (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        except Exception:  # noqa: BLE001
            cv2_available = False
            frame[h // 3 : h // 2, w // 3 : w // 2] = (240, 200, 200)
        _ = cv2_available
        return frame
