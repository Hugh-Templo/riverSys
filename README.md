# riverSys — web branch

**FastAPI + HTML/CSS** dashboard for Raspberry Pi 5 + Camera Module 3.

## Run

```bash
bash scripts/setup_env.sh
bash run_api.sh
# open http://<pi-ip>:8000
```

## Includes

- MJPEG live stream + detection overlay
- WebSocket status / manual sort buttons
- Auto-sort servo control via API
- Shared `backend/` (camera, detector, servo)

Python HDMI window lives on the `python` branch (`bash run_display.sh`).
Full tree also on `main`.
