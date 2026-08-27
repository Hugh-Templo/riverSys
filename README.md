# riverSys — python branch

OpenCV / HDMI **Python display** for Raspberry Pi 5 + Camera Module 3.

## Run

```bash
bash scripts/setup_env.sh
bash run_display.sh
# or: desktop icon "Plastic Segregation" (tmux)
```

## Includes

- Live camera detection (OpenCV fallback / YOLO when trained)
- LEFT / RIGHT drop zones
- TRAIN toggle + label prompts
- Servo calibration / swap bins / auto-sort
- Shared `backend/` (camera, detector, servo)

Web dashboard lives on the `web` branch (`bash run_api.sh`).
Full tree also on `main`.
