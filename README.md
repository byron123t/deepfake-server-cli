![image](https://github.com/Hillobar/Rope/assets/63615199/40f7397f-713c-4813-ac86-bab36f6bd5ba)

Rope implements the insightface inswapper_128 model as a **real-time client-server face swap system**. The server runs the GAN inference on GPU; a lightweight client handles face detection and streams video over WebSocket.

---

## Architecture

```
Client device (CPU only)               Server (GPU required)
────────────────────────               ──────────────────────────────────
Webcam → MediaPipe face detection      FastAPI + WebSocket (server.py)
  ↓ crop faces, extract landmarks        ├─ inswapper_128 GAN
  ──── WebSocket (LAN/VPN) ──────►       ├─ GFPGAN / GPEN / CodeFormer
  ◄─── swapped face crops ────────       └─ occluder, face parser
  ↓ composite back into frame
  ↓ display
```

The client sends padded face crops + 5-point keypoints to the server. The server swaps the face and returns the processed crop. No full frames are transmitted.

---

## Requirements

### Server machine (GPU)
- NVIDIA GPU with CUDA 11.8
- Python 3.10+

### Client machine
- Python 3.10+
- Webcam
- No GPU required

---

## Model Files

Place the following ONNX model files in the `models/` directory on the **server**:

| File | Required | Purpose |
|------|----------|---------|
| `inswapper_128.fp16.onnx` | Yes | Face swapper |
| `det_10g.onnx` | Yes | Face detection |
| `w600k_r50.onnx` | Yes | ArcFace recognition |
| `GFPGANv1.4.onnx` | Optional | Face restoration |
| `GPEN-BFR-512.onnx` | Optional | Face restoration |
| `GPEN-BFR-256.onnx` | Optional | Face restoration |
| `codeformer_fp16.onnx` | Optional | Face restoration |
| `occluder.onnx` | Optional | Occlusion masking |
| `faceparser_fp16.onnx` | Optional | Face parsing |

These are the same models used by the original Rope GUI.

---

## Installation

### Server
```bash
git clone <repo>
cd Rope
pip install -r requirements.txt
```

### Client (separate machine)
```bash
git clone <repo>
cd Rope
pip install -r requirements_client.txt
```

---

## Configuration

Both `server.py` and `client.py` read from a `.env` file. Copy the example and fill in your values:

```bash
cp .env.example .env
```

**`.env` on the server:**
```ini
HOST=0.0.0.0
PORT=8765
SOURCE_PATH=/path/to/source_face.jpg   # or a folder of images
```

**`.env` on the client:**
```ini
SERVER_URL=ws://SERVER_IP:8765/ws
SOURCE_PATH=/path/to/source_face.jpg   # or a folder of images
CAMERA_INDEX=0
CAPTURE_WIDTH=1280
CAPTURE_HEIGHT=720
```

> `.env` is gitignored. Never commit it.

CLI arguments override `.env` values if both are provided.

---

## Running

### 1. Start the server

```bash
python server.py
```

Or with explicit arguments:
```bash
python server.py --source /path/to/source.jpg --host 0.0.0.0 --port 8765
```

You will see:
```
Loading models …
Models ready.
Source face loaded: 1/1 image(s) used
Uvicorn running on http://0.0.0.0:8765
```

### 2. Test connectivity (from the client machine)

```bash
# HTTP health check
curl http://SERVER_IP:8765/health
# {"status":"ok","models_loaded":true,"source_face_loaded":true}

# Browser status page + WebSocket ping test
open http://SERVER_IP:8765/
```

### 3. Start the client

```bash
python client.py
```

Or with explicit arguments:
```bash
python client.py --server ws://SERVER_IP:8765/ws --source /path/to/source.jpg
```

A webcam preview window opens with the face swap applied in real time.

---

## Source Face

The source face (the identity to swap onto detected faces) can be:

- **A single image:** `--source /path/to/face.jpg`
- **A folder of images of the same person:** `--source /path/to/folder/`

When a folder is provided, the server computes an ArcFace embedding for each image and averages them, producing a more robust identity representation across different lighting and angles.

The source can be set at server startup via `SOURCE_PATH` in `.env` or `--source`, or sent at runtime by the client.

---

## Client Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `s` | Resend source face to server |
| `r` | Toggle face restorer (GFPGAN) on/off |
| `+` | Increase blend smoothing |
| `-` | Decrease blend smoothing |

---

## Performance

Server: 3090Ti (24GB), i5-13600K  
Benchmark: `benchmark/target-1080p.mp4`, 2048×1080, 25 fps

| Mode | Time (5 threads) |
|------|-----------------|
| Swap 128 | 4.4s |
| Swap 256 | 8.6s |
| Swap 512 | 28.6s |
| Swap + GFPGAN | 9.3s |
| Swap + CodeFormer | 11.3s |
| Swap + Occluder | 4.7s |
| Swap + MouthParser | 5.1s |

---

## Disclaimer

Rope is intended for responsible and ethical use only. Users are solely responsible for their actions when using this software.

Do not create or share content that could harm, defame, or harass individuals. Obtain proper consent from individuals before using their likeness. Do not use this technology for deceptive purposes. Respect applicable laws, regulations, and copyright restrictions.

By using this software you agree to use it ethically and legally.
