"""FastAPI backend: live detection stream + servo segregation API."""

from __future__ import annotations

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config
from .camera import CameraService
from .detector import PlasticDetector, draw_detections
from .servo import ServoSorter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("plastic")

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend" / "static"

camera = CameraService()
detector = PlasticDetector()
servo = ServoSorter()

_state_lock = asyncio.Lock()
_latest_detections: list[dict] = []
_latest_decision: Optional[str] = None
_latest_jpeg: bytes = b""
_frame_count = 0
_last_sort_ts = 0.0
_auto_sort = config.AUTO_SORT
_running_pipeline = True


class SortRequest(BaseModel):
    label: str


class SettingsUpdate(BaseModel):
    auto_sort: Optional[bool] = None
    conf_threshold: Optional[float] = None


def _encode_jpeg(frame_rgb: np.ndarray, quality: int = 75) -> bytes:
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return b""
    return buf.tobytes()


def _choose_label(detections: list) -> Optional[str]:
    if not detections:
        return None
    best = max(detections, key=lambda d: d.confidence)
    if best.confidence < config.CONF_THRESHOLD:
        return None
    return best.label


async def _pipeline_loop() -> None:
    global _latest_detections, _latest_decision, _latest_jpeg, _frame_count, _last_sort_ts
    while _running_pipeline:
        frame = camera.get_frame()
        if frame is None:
            await asyncio.sleep(0.05)
            continue

        detections = await asyncio.to_thread(detector.detect, frame)
        annotated = draw_detections(frame, detections)
        jpeg = _encode_jpeg(annotated)
        decision = _choose_label(detections)
        _frame_count += 1

        async with _state_lock:
            _latest_detections = [d.to_dict() for d in detections]
            _latest_decision = decision
            if jpeg:
                _latest_jpeg = jpeg

        if _auto_sort and decision and not servo.status["busy"]:
            now = time.monotonic()
            if now - _last_sort_ts >= config.SORT_COOLDOWN_SECONDS:
                _last_sort_ts = now
                await asyncio.to_thread(servo.sort_label, decision)

        await asyncio.sleep(0.03)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global _running_pipeline
    config.ensure_directories()
    camera.start()
    _running_pipeline = True
    task = asyncio.create_task(_pipeline_loop())
    logger.info("Plastic segregation API ready on %s:%s", config.HOST, config.PORT)
    try:
        yield
    finally:
        _running_pipeline = False
        task.cancel()
        camera.stop()
        servo.home()


app = FastAPI(title="Plastic Segregation System", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if FRONTEND.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    index_path = FRONTEND / "index.html"
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.get("/api/health")
async def health() -> dict:
    return {
        "ok": True,
        "camera_running": camera.is_running,
        "camera_mock": camera.using_mock,
        "detector_mode": detector.mode,
        "weights": str(detector.weights_path) if detector.weights_path else None,
        "servo": servo.status,
        "auto_sort": _auto_sort,
        "frame_count": _frame_count,
        "dataset": str(config.RAW_DATASET),
        "models_dir": str(config.MODELS_DIR),
    }


@app.get("/api/status")
async def status() -> dict:
    async with _state_lock:
        detections = list(_latest_detections)
        decision = _latest_decision
    return {
        "detections": detections,
        "decision": decision,
        "auto_sort": _auto_sort,
        "detector_mode": detector.mode,
        "servo": servo.status,
        "frame_count": _frame_count,
        "conf_threshold": config.CONF_THRESHOLD,
    }


@app.post("/api/settings")
async def update_settings(body: SettingsUpdate) -> dict:
    global _auto_sort
    if body.auto_sort is not None:
        _auto_sort = body.auto_sort
    if body.conf_threshold is not None:
        config.CONF_THRESHOLD = float(body.conf_threshold)
    return {"ok": True, "auto_sort": _auto_sort, "conf_threshold": config.CONF_THRESHOLD}


@app.post("/api/sort")
async def sort_manual(body: SortRequest) -> dict:
    result = await asyncio.to_thread(servo.sort_label, body.label)
    return result


@app.post("/api/servo/home")
async def servo_home() -> dict:
    return await asyncio.to_thread(servo.home)


@app.post("/api/detector/reload")
async def reload_detector() -> dict:
    mode = await asyncio.to_thread(detector.reload)
    return {"ok": True, "mode": mode, "weights": str(detector.weights_path)}


async def _mjpeg_generator():
    last = b""
    while True:
        async with _state_lock:
            payload = _latest_jpeg
        if payload and payload is not last:
            last = payload
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + payload + b"\r\n"
        await asyncio.sleep(0.05)


@app.get("/api/stream")
async def stream():
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.websocket("/ws/status")
async def ws_status(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            async with _state_lock:
                payload = {
                    "detections": list(_latest_detections),
                    "decision": _latest_decision,
                    "auto_sort": _auto_sort,
                    "detector_mode": detector.mode,
                    "servo": servo.status,
                    "frame_count": _frame_count,
                }
            await websocket.send_json(payload)
            await asyncio.sleep(0.25)
    except WebSocketDisconnect:
        return
