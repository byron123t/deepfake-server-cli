"""
Rope Face-Swap API Server
=========================
Receives face crops + 5-pt keypoints from a client over WebSocket,
runs the inswapper GAN + optional super-resolution restorer on GPU,
and streams swapped face crops back.

Usage:
    python server.py --source /path/to/source_face.jpg [--host 0.0.0.0] [--port 8765]
    python server.py --source /path/to/source_folder/

    Or set values in .env and run: python server.py

WebSocket endpoint: ws://<host>:<port>/ws

Protocol (msgpack-serialised dicts):

  Client -> Server:
    { "type": "set_source", "images": [<bytes>, ...] }  # 1-N JPEGs; embeddings averaged
    { "type": "swap",
      "frame_id": int,
      "params": { ... }                                  # optional, overrides defaults
      "faces": [
        { "crop":  <bytes>,                              # JPEG of padded face region
          "kps":   [[x,y]*5],                            # 5-pt landmarks in crop coords
          "bbox":  [cx1, cy1, cx2, cy2]                  # crop origin in full frame
        }, ...
      ]
    }

  Server -> Client:
    { "type": "source_set", "success": bool, "faces_used": int, "images_sent": int }
    { "type": "source_ready" }                           # if --source was given at startup
    { "type": "swapped",
      "frame_id": int,
      "faces": [
        { "swapped": <bytes>,                            # JPEG of swapped crop (same size)
          "bbox":    [cx1, cy1, cx2, cy2]
        }, ...
      ]
    }
    { "type": "error", "message": str }
"""

import argparse
import asyncio
import glob
import os
import sys

from dotenv import load_dotenv
load_dotenv()

import cv2
import msgpack
import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

sys.path.insert(0, os.path.dirname(__file__))
import rope.Models as ModelsModule
from rope.SwapCore import SwapCore, DEFAULT_PARAMS

# ------------------------------------------------------------------ #
# Globals populated at startup                                         #
# ------------------------------------------------------------------ #
_models: ModelsModule.Models | None = None
_swap_core: SwapCore | None = None
_startup_embedding: np.ndarray | None = None
_gpu_lock = asyncio.Lock()

# ------------------------------------------------------------------ #
# App                                                                  #
# ------------------------------------------------------------------ #
app = FastAPI()


@app.on_event("startup")
async def _startup():
    global _models, _swap_core, _startup_embedding
    print("Loading models …")
    _models = ModelsModule.Models()
    _swap_core = SwapCore(_models)
    print("Models ready.")

    if _args.source:
        raw_images = _load_source_images(_args.source)
        if not raw_images:
            print(f"[WARN] No readable images found at: {_args.source}")
        else:
            emb, n_ok = _compute_averaged_embedding(raw_images)
            if emb is None:
                print(f"[WARN] No face detected in {len(raw_images)} source image(s).")
            else:
                _startup_embedding = emb
                print(f"Source face loaded: {n_ok}/{len(raw_images)} image(s) used from {_args.source}")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "models_loaded": _swap_core is not None,
        "source_face_loaded": _startup_embedding is not None,
    }


@app.get("/")
async def index():
    html = """<!DOCTYPE html>
<html>
<head><title>Rope Server</title>
<style>body{font-family:monospace;padding:2em;background:#111;color:#eee}
  #log{margin-top:1em;white-space:pre;color:#0f0}</style>
</head>
<body>
<h2>Rope Face-Swap Server</h2>
<div id="status">Checking health…</div>
<button onclick="pingWS()">Ping WebSocket</button>
<div id="log"></div>
<script>
const log = s => document.getElementById('log').textContent += s + '\\n';
fetch('/health').then(r=>r.json()).then(d=>{
  document.getElementById('status').textContent = JSON.stringify(d, null, 2);
});
function pingWS() {
  const ws = new WebSocket('ws://' + location.host + '/ws');
  ws.binaryType = 'arraybuffer';
  ws.onopen = () => log('WS connected');
  ws.onmessage = e => log('WS message received (' + e.data.byteLength + ' bytes)');
  ws.onerror = e => log('WS error: ' + e);
  ws.onclose = () => log('WS closed');
}
</script>
</body>
</html>"""
    return HTMLResponse(html)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    source_embedding: np.ndarray | None = _startup_embedding

    if source_embedding is not None:
        await _send(websocket, {"type": "source_ready"})

    try:
        while True:
            raw = await websocket.receive_bytes()
            msg = msgpack.unpackb(raw, raw=False)
            mtype = msg.get("type")

            # -------------------------------------------------- #
            if mtype == "set_source":
                raw_list = msg.get("images") or ([msg["image"]] if "image" in msg else [])
                async with _gpu_lock:
                    emb, n_ok = _compute_averaged_embedding(raw_list)
                if emb is None:
                    await _send(websocket, {
                        "type": "source_set", "success": False,
                        "message": "No face detected in any source image."
                    })
                else:
                    source_embedding = emb
                    await _send(websocket, {
                        "type": "source_set", "success": True,
                        "faces_used": n_ok, "images_sent": len(raw_list),
                    })

            # -------------------------------------------------- #
            elif mtype == "swap":
                if source_embedding is None:
                    await _send(websocket, {
                        "type": "error",
                        "message": "No source face set. Send set_source first."
                    })
                    continue

                frame_id = msg.get("frame_id", 0)
                params = {**DEFAULT_PARAMS, **msg.get("params", {})}
                result_faces = []

                async with _gpu_lock:
                    for face in msg.get("faces", []):
                        swapped_bytes = _process_face(
                            face["crop"], face["kps"], face["bbox"],
                            source_embedding, params)
                        result_faces.append({
                            "swapped": swapped_bytes,
                            "bbox": face["bbox"],
                        })

                await _send(websocket, {
                    "type": "swapped",
                    "frame_id": frame_id,
                    "faces": result_faces,
                })

            # -------------------------------------------------- #
            else:
                await _send(websocket, {
                    "type": "error",
                    "message": f"Unknown message type: {mtype}"
                })

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        print(f"[ERROR] {exc}")
        try:
            await _send(websocket, {"type": "error", "message": str(exc)})
        except Exception:
            pass


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _load_source_images(path: str) -> list[bytes]:
    """Return JPEG-encoded bytes for each image found at path (file or directory)."""
    if os.path.isfile(path):
        img = cv2.imread(path)
        if img is None:
            return []
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return [buf.tobytes()] if ok else []

    if os.path.isdir(path):
        results = []
        for ext in _IMAGE_EXTS:
            for p in glob.glob(os.path.join(path, f"*{ext}")) + \
                     glob.glob(os.path.join(path, f"*{ext.upper()}")):
                img = cv2.imread(p)
                if img is None:
                    continue
                ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if ok:
                    results.append(buf.tobytes())
        return results

    return []


def _compute_averaged_embedding(raw_images: list) -> tuple[np.ndarray | None, int]:
    """Decode each JPEG, detect face, compute embedding. Average all valid embeddings."""
    embeddings = []
    for img_bytes in raw_images:
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        emb = _swap_core.get_embedding(img_rgb)
        if emb is not None:
            embeddings.append(emb)

    if not embeddings:
        return None, 0

    avg = np.mean(np.stack(embeddings, axis=0), axis=0).astype(np.float32)
    return avg, len(embeddings)


def _process_face(crop_bytes: bytes, kps_list: list, bbox: list,
                  source_embedding: np.ndarray, params: dict) -> bytes:
    nparr = np.frombuffer(crop_bytes, np.uint8)
    crop = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    kps = np.array(kps_list, dtype=np.float32)

    swapped_rgb = _swap_core.process(crop_rgb, kps, source_embedding, params)

    swapped_bgr = cv2.cvtColor(swapped_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", swapped_bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return buf.tobytes()


async def _send(ws: WebSocket, payload: dict):
    await ws.send_bytes(msgpack.packb(payload, use_bin_type=True))


# ------------------------------------------------------------------ #
# Entry point                                                          #
# ------------------------------------------------------------------ #
_args: argparse.Namespace | None = None


def main():
    global _args
    parser = argparse.ArgumentParser(
        description="Rope face-swap WebSocket server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"),
                        help="Bind address (env: HOST)")
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8765")),
                        help="Port to listen on (env: PORT)")
    parser.add_argument("--source", default=os.getenv("SOURCE_PATH", ""),
                        help="Source face image or folder (env: SOURCE_PATH)")
    _args = parser.parse_args()

    uvicorn.run(app, host=_args.host, port=_args.port, log_level="info")


if __name__ == "__main__":
    main()
