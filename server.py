"""
Rope Face-Swap API Server
=========================
Receives full JPEG frames from a client over WebSocket, runs face detection +
inswapper GAN + optional restorer on GPU, and streams swapped frames back.

Usage:
    python server.py --source /path/to/source_face.jpg [--host 0.0.0.0] [--port 8765]

WebSocket endpoint: ws://<host>:<port>/ws

Protocol (msgpack-serialised dicts):

  Client -> Server:
    { "type": "set_source", "images": [<bytes>, ...] }
    { "type": "frame",
      "frame_id": int,
      "frame":   <bytes>,           # JPEG of full video frame
      "params":  { ... }            # optional
    }

  Server -> Client:
    { "type": "source_set",   "success": bool, "faces_used": int, "images_sent": int }
    { "type": "source_ready" }
    { "type": "frame_result", "frame_id": int, "frame": <bytes> }  # JPEG of processed frame
    { "type": "error",        "message": str }
"""

import argparse
import asyncio
import concurrent.futures
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
# Globals                                                              #
# ------------------------------------------------------------------ #
_models: ModelsModule.Models | None = None
_swap_core: SwapCore | None = None
_startup_embedding: np.ndarray | None = None

# Single-threaded executor: serialises all GPU inference
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

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
            loop = asyncio.get_running_loop()
            emb, n_ok = await loop.run_in_executor(
                _executor, _compute_averaged_embedding, raw_images)
            if emb is None:
                print(f"[WARN] No face detected in {len(raw_images)} source image(s).")
            else:
                _startup_embedding = emb
                print(f"Source face loaded: {n_ok}/{len(raw_images)} image(s) from {_args.source}")


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

    # Depth-1 queue: always process the most recent frame, drop stale ones.
    # The client uses _in_flight semaphore so in practice only one frame is
    # ever queued, but the drop logic protects us if that changes.
    frame_queue: asyncio.Queue = asyncio.Queue(maxsize=1)
    loop = asyncio.get_running_loop()

    # ---------------------------------------------------------- #
    # Receiver: reads WebSocket messages as fast as they arrive   #
    # ---------------------------------------------------------- #
    async def _recv_loop():
        nonlocal source_embedding
        try:
            while True:
                raw = await websocket.receive_bytes()
                msg = msgpack.unpackb(raw, raw=False)
                mtype = msg.get("type")

                # -- set_source --------------------------------- #
                if mtype == "set_source":
                    raw_list = msg.get("images") or (
                        [msg["image"]] if "image" in msg else [])
                    emb, n_ok = await loop.run_in_executor(
                        _executor, _compute_averaged_embedding, raw_list)
                    if emb is None:
                        await _send(websocket, {
                            "type": "source_set", "success": False,
                            "message": "No face detected in any source image.",
                        })
                    else:
                        source_embedding = emb
                        await _send(websocket, {
                            "type": "source_set", "success": True,
                            "faces_used": n_ok, "images_sent": len(raw_list),
                        })

                # -- frame -------------------------------------- #
                elif mtype == "frame":
                    if source_embedding is None:
                        # Must still respond so the client releases _in_flight
                        await _send(websocket, {
                            "type": "error",
                            "message": "No source face set. Send set_source first.",
                        })
                        continue

                    item = (
                        msg.get("frame_id", 0),
                        msg.get("frame"),
                        source_embedding,
                        {**DEFAULT_PARAMS, **msg.get("params", {})},
                    )
                    # Replace stale queued frame with the latest one
                    if frame_queue.full():
                        try:
                            frame_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                    try:
                        frame_queue.put_nowait(item)
                    except asyncio.QueueFull:
                        pass

                else:
                    await _send(websocket, {
                        "type": "error",
                        "message": f"Unknown message type: {mtype}",
                    })

        except WebSocketDisconnect:
            pass
        except Exception as exc:
            print(f"[RECV] {exc}")
        finally:
            # Signal the processor to shut down
            try:
                frame_queue.put_nowait(None)
            except asyncio.QueueFull:
                frame_queue.get_nowait()
                frame_queue.put_nowait(None)

    # ---------------------------------------------------------- #
    # Processor: GPU inference, one frame at a time               #
    # ---------------------------------------------------------- #
    async def _proc_loop():
        while True:
            item = await frame_queue.get()
            if item is None:
                break
            frame_id, frame_bytes, s_e, params = item
            try:
                result_bytes = await loop.run_in_executor(
                    _executor, _process_frame_sync, frame_bytes, s_e, params)
                if result_bytes is not None:
                    await _send(websocket, {
                        "type": "frame_result",
                        "frame_id": frame_id,
                        "frame": result_bytes,
                    })
                else:
                    # Decode failed — client must still release _in_flight
                    await _send(websocket, {
                        "type": "error",
                        "message": "Failed to decode frame.",
                    })
            except Exception as exc:
                print(f"[PROC] {exc}")
                try:
                    await _send(websocket, {
                        "type": "error",
                        "message": str(exc),
                    })
                except Exception:
                    pass

    recv_task = asyncio.create_task(_recv_loop())
    proc_task = asyncio.create_task(_proc_loop())

    done, pending = await asyncio.wait(
        [recv_task, proc_task], return_when=asyncio.FIRST_COMPLETED)
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)


# ------------------------------------------------------------------ #
# Synchronous GPU processing (runs inside the thread-pool executor)   #
# ------------------------------------------------------------------ #

def _process_frame_sync(frame_bytes: bytes, source_embedding: np.ndarray,
                        params: dict) -> bytes | None:
    """Decode JPEG, detect + swap faces on GPU, re-encode to JPEG."""
    nparr = np.frombuffer(frame_bytes, np.uint8)
    frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        return None

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    result_rgb = _swap_core.process_frame(frame_rgb, source_embedding, params)
    result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)

    ok, buf = cv2.imencode('.jpg', result_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes() if ok else None


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _load_source_images(path: str) -> list[bytes]:
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
