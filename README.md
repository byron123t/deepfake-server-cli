# Rope

Real-time face swap over a local network. A GPU server runs the GAN; a lightweight client streams webcam frames to it and displays the result.

---

## Quick start — client

If someone else is running the server, this is all you need.

**1. Clone and install**
```bash
git clone <repo>
cd Rope
pip install -r requirements_client.txt
```

**2. Create your `.env`**
```bash
cp .env.example .env
```
Edit `.env` and set:
```ini
SERVER_URL=ws://SERVER_IP:8765/ws
SOURCE_PATH=/path/to/your/source_face.jpg
```

**3. Run**
```bash
python client.py
```

A window opens with the face swap applied live. If `SOURCE_PATH` is not set in `.env`, a file picker will appear on launch.

**Controls**

| Key | Action |
|-----|--------|
| `q` | Quit |
| `s` | Resend source face to server |
| `r` | Cycle enhance mode: off → GPEN-256 → GPEN-512 → off |
| `m` | Toggle mouth mask (preserves original mouth) |
| `+` / `-` | Opacity up / down (0.05 steps) |

---

## Server setup

### Requirements
- NVIDIA GPU (CUDA 12, cuDNN 9)
- Python 3.10+
- conda recommended

### Install
```bash
conda create -n rope python=3.10
conda activate rope
pip install -r requirements.txt
```

### Model files

Place these in the `models/` directory:

| File | Required | Purpose |
|------|----------|---------|
| `inswapper_128.onnx` | Yes | Face swapper (FP32) |
| `GPEN-BFR-256.onnx` | Optional | Fast face enhancement |
| `GPEN-BFR-512.onnx` | Optional | High-quality face enhancement |

InsightFace detection models (`buffalo_l`) are downloaded automatically on first run to `~/.insightface/models/`.

### Configure
```bash
cp .env.example .env
```
Edit `.env`:
```ini
SOURCE_PATH=/path/to/source_face.jpg   # or a folder of images
HOST=0.0.0.0
PORT=8765
```

### Run
```bash
python server.py
```

With verbose per-frame timing:
```bash
python server.py --verbose
```

Verify it's up:
```bash
curl http://SERVER_IP:8765/health
# {"status":"ok","models_loaded":true,"source_face_loaded":true}
```

---

## Source face

`SOURCE_PATH` can be a single image or a folder of images of the same person. Multiple images give a more robust identity embedding across different lighting and angles.

The source face can be set at server startup (via `.env` or `--source`) or sent at runtime by the client (key `s`).

---

## Parameter test grid

`test_params.py` sends a frozen frame through 8 preset configurations and shows the results side-by-side so you can visually compare quality vs. speed trade-offs.

```bash
python test_params.py --source /path/to/source.jpg
# or use a static image instead of webcam:
python test_params.py --source /path/to/source.jpg --image /path/to/test_frame.jpg
```

Keys: `n`/`space` = new snapshot, `s` = save PNG, `q` = quit.

---

## Architecture

```
Client (CPU only)                    Server (GPU)
─────────────────                    ────────────────────────────
Webcam capture                       FastAPI + WebSocket
  ↓ JPEG encode (downscaled)
  ──── WebSocket ──────────────────► InsightFace detection
  ◄─── swapped JPEG ───────────────  inswapper_128 GAN
  ↓ decode + upscale                 GPEN-256 / GPEN-512 (optional)
  ↓ cross-fade display
```

Frames are downscaled before sending (default `--send-scale 0.5`) and upscaled after receipt to reduce bandwidth and server decode cost. The server does all face detection and inference; the client is pure I/O.

---

## Disclaimer

Rope is intended for responsible and ethical use only. Users are solely responsible for their actions when using this software.

Do not create or share content that could harm, defame, or harass individuals. Obtain proper consent from individuals before using their likeness. Do not use this technology for deceptive purposes. Respect applicable laws, regulations, and copyright restrictions.

By using this software you agree to use it ethically and legally.
