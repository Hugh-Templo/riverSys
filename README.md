# riverSys

Floating plastic detection and segregation for **Raspberry Pi 5 + Camera Module 3**.

## Branches

| Branch | Focus | Start |
|--------|--------|--------|
| `main` | Full system (Python display + web API) | see below |
| `python` | OpenCV / HDMI local display + servo | `bash run_display.sh` |
| `web` | FastAPI + HTML dashboard + servo | `bash run_api.sh` |

## Hardware

- Raspberry Pi 5 (8 GB)
- Camera Module 3
- One 20 kg servo on **GPIO 18** (external 5–6 V power; common GND with Pi)
- Dataset / models on USB: `/media/river/6A1E-2BCD/`

## Setup

```bash
cd /home/river/projectRiver   # or your clone path
bash scripts/setup_env.sh
python scripts/prepare_dataset.py
```

Use Debian `python3-torch` via apt (setup script does this). Do **not** `pip install torch` on aarch64 — PyPI may pull huge CUDA wheels.

## Run

**Python display (HDMI):**

```bash
bash run_display.sh
# or desktop icon → Plastic Segregation (tmux)
```

**Web UI:**

```bash
bash run_api.sh
# open http://<pi-ip>:8000
```

## Features

- Recyclable vs non-recyclable sorting via servo (±60° by default, calibratable)
- LEFT / RIGHT drop zones on camera view
- Training mode (label live detections → USB YOLO dataset)
- Servo calibration + swap bins
- OpenCV fallback until a custom YOLO model is trained

## License

Project scaffolding for Hugh-Templo/riverSys.
