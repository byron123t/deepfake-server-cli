"""
Rope Face-Swap Client
=====================
Captures webcam frames, sends them as JPEG to the server, and displays
the fully-composited swapped frame that comes back.

All face detection and compositing runs on the server GPU — the client
does nothing heavier than JPEG encode/decode.

Usage:
    python client.py --server ws://SERVER_IP:8765/ws --source /path/to/source.jpg
    python client.py --server ws://SERVER_IP:8765/ws --source /path/to/source_folder/

    Or set values in .env and run: python client.py

Keys:
    q    - quit
    s    - resend source to server
    r    - toggle face restoration on server (GFPGAN)
    +/-  - increase/decrease blend smoothing
"""

import argparse
import asyncio
import glob
import os
import queue
import sys
import threading
import time
import tkinter as tk
from tkinter import filedialog

from dotenv import load_dotenv
load_dotenv()

import cv2
import msgpack
import numpy as np
import websockets

_SOURCE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ------------------------------------------------------------------ #
# Source helpers                                                       #
# ------------------------------------------------------------------ #

def _collect_source_images(path: str) -> list[bytes]:
    if os.path.isfile(path):
        img = cv2.imread(path)
        if img is None:
            print(f"[WARN] Cannot read: {path}")
            return []
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return [buf.tobytes()] if ok else []

    if os.path.isdir(path):
        results = []
        for ext in _SOURCE_EXTS:
            for p in sorted(glob.glob(os.path.join(path, f"*{ext}")) +
                            glob.glob(os.path.join(path, f"*{ext.upper()}"))):
                img = cv2.imread(p)
                if img is None:
                    continue
                ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if ok:
                    results.append(buf.tobytes())
        return results

    print(f"[WARN] Source path not found: {path}")
    return []


# ------------------------------------------------------------------ #
# Shared display state                                                 #
# ------------------------------------------------------------------ #
_frame_q: queue.Queue = queue.Queue(maxsize=2)

_display_lock = threading.Lock()
_display_frame: np.ndarray | None = None  # RGB


def _set_display(frame: np.ndarray) -> None:
    global _display_frame
    with _display_lock:
        _display_frame = frame


def _get_display() -> np.ndarray | None:
    with _display_lock:
        return _display_frame


# ------------------------------------------------------------------ #
# Capture thread — raw BGR frames only, no processing                 #
# ------------------------------------------------------------------ #

def _capture_thread(camera_id: int, width: int, height: int):
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {camera_id}")
        return

    print(f"Camera {camera_id} opened at {width}x{height}")
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        if _frame_q.full():
            try:
                _frame_q.get_nowait()
            except queue.Empty:
                pass
        _frame_q.put(frame_bgr)

    cap.release()


# ------------------------------------------------------------------ #
# Async WebSocket client                                               #
# ------------------------------------------------------------------ #

# Limits to one frame in-flight at a time so we always send the freshest
# frame and never build up a stale backlog.
_in_flight: asyncio.Semaphore | None = None


async def _ws_client(uri: str, source_path: str, params_ref: dict):
    global _in_flight
    _in_flight = asyncio.Semaphore(1)

    print(f"Connecting to {uri} …")
    async with websockets.connect(uri, max_size=20 * 1024 * 1024) as ws:
        print("Connected.")
        _ws_ref.append(ws)

        if source_path:
            await _send_source(ws, source_path)

        send_task = asyncio.create_task(_sender(ws, params_ref))
        recv_task = asyncio.create_task(_receiver(ws))
        await asyncio.gather(send_task, recv_task)


async def _send_source(ws, path: str):
    if not path:
        return
    images = _collect_source_images(path)
    if not images:
        print(f"[WARN] No readable images found at: {path}")
        return
    await ws.send(msgpack.packb({"type": "set_source", "images": images}, use_bin_type=True))
    label = f"{len(images)} image(s)" if len(images) > 1 else "1 image"
    print(f"Source face sent: {label} from {path}")


async def _sender(ws, params_ref: dict):
    loop = asyncio.get_event_loop()
    frame_id = 0
    while True:
        # Wait for the server to finish the previous frame before grabbing
        # the next one — this keeps the queue flushed to the freshest frame.
        await _in_flight.acquire()

        try:
            frame_bgr = await loop.run_in_executor(
                None, lambda: _frame_q.get(timeout=1.0))
        except queue.Empty:
            _in_flight.release()
            continue

        # Show raw frame immediately so the preview never freezes.
        _set_display(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

        ok, buf = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if not ok:
            _in_flight.release()
            continue

        payload = msgpack.packb({
            "type": "frame",
            "frame_id": frame_id,
            "params": params_ref.copy(),
            "frame": buf.tobytes(),
        }, use_bin_type=True)
        await ws.send(payload)
        frame_id += 1


async def _receiver(ws):
    async for raw in ws:
        msg = msgpack.unpackb(raw, raw=False)
        mtype = msg.get("type")

        if mtype == "frame_result":
            frame_bytes = msg.get("frame")
            if frame_bytes:
                nparr = np.frombuffer(frame_bytes, np.uint8)
                result_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if result_bgr is not None:
                    _set_display(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB))
            if _in_flight:
                _in_flight.release()

        elif mtype == "source_set":
            if msg.get("success"):
                n_ok = msg.get("faces_used", "?")
                n_sent = msg.get("images_sent", "?")
                print(f"[source_set] OK — {n_ok}/{n_sent} image(s) had a detectable face")
            else:
                print(f"[source_set] FAILED: {msg.get('message', 'no face found')}")

        elif mtype == "source_ready":
            print("[server] Source face ready (loaded at server startup).")

        elif mtype == "error":
            print(f"[server error] {msg.get('message')}")
            if _in_flight:
                _in_flight.release()


# ------------------------------------------------------------------ #
# Display loop (main thread)                                           #
# ------------------------------------------------------------------ #

def _display_loop(params_ref: dict, source_path_ref: list):
    fps_t = time.time()
    fps_count = 0
    fps_display = 0.0

    print("Display started. Press 'q' to quit, 's' to resend source, 'r' to toggle restorer.")

    while True:
        frame = _get_display()

        if frame is not None:
            display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            fps_count += 1
            now = time.time()
            if now - fps_t >= 1.0:
                fps_display = fps_count / (now - fps_t)
                fps_count = 0
                fps_t = now

            restorer_on = params_ref.get("RestorerSwitch", False)
            cv2.putText(display, f"FPS: {fps_display:.1f}", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display, f"Restorer: {'ON' if restorer_on else 'OFF'}", (10, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            cv2.imshow("Rope — Face Swap", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            if source_path_ref[0] and _ws_loop and _ws_ref:
                asyncio.run_coroutine_threadsafe(
                    _send_source(_ws_ref[0], source_path_ref[0]),
                    _ws_loop)
        elif key == ord('r'):
            params_ref["RestorerSwitch"] = not params_ref.get("RestorerSwitch", False)
            print(f"Restorer: {'ON' if params_ref['RestorerSwitch'] else 'OFF'}")
        elif key == ord('+'):
            params_ref["BlendSlider"] = min(params_ref.get("BlendSlider", 5) + 1, 20)
        elif key == ord('-'):
            params_ref["BlendSlider"] = max(params_ref.get("BlendSlider", 5) - 1, 0)

    cv2.destroyAllWindows()


# ------------------------------------------------------------------ #
# Source-face picker                                                   #
# ------------------------------------------------------------------ #

def _pick_source_face() -> str:
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(
        title="Select source face image",
        filetypes=[
            ("Images", "*.jpg *.jpeg *.png *.webp *.bmp"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    if not path:
        print("No source face selected — exiting.")
        sys.exit(0)
    return path


# ------------------------------------------------------------------ #
# Entry point                                                          #
# ------------------------------------------------------------------ #
_ws_loop: asyncio.AbstractEventLoop | None = None
_ws_ref: list = []


def main():
    global _ws_loop, _ws_ref

    parser = argparse.ArgumentParser(
        description="Rope face-swap client",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--server", default=os.getenv("SERVER_URL", "ws://127.0.0.1:8765/ws"),
                        help="WebSocket server URL (env: SERVER_URL)")
    parser.add_argument("--source", default=os.getenv("SOURCE_PATH", ""),
                        help="Source face image or folder (env: SOURCE_PATH)")
    parser.add_argument("--camera", type=int, default=int(os.getenv("CAMERA_INDEX", "0")),
                        help="Camera device index (env: CAMERA_INDEX)")
    parser.add_argument("--width", type=int, default=int(os.getenv("CAPTURE_WIDTH", "1280")),
                        help="Capture width (env: CAPTURE_WIDTH)")
    parser.add_argument("--height", type=int, default=int(os.getenv("CAPTURE_HEIGHT", "720")),
                        help="Capture height (env: CAPTURE_HEIGHT)")
    args = parser.parse_args()

    if not args.source:
        args.source = _pick_source_face()

    params_ref: dict = {}
    source_path_ref: list = [args.source]

    cap_thread = threading.Thread(target=_capture_thread,
                                  args=(args.camera, args.width, args.height),
                                  daemon=True)
    cap_thread.start()

    def _run_ws():
        global _ws_loop
        _ws_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_ws_loop)
        try:
            _ws_loop.run_until_complete(_ws_client(args.server, args.source, params_ref))
        except Exception as e:
            print(f"[WS] Disconnected: {e}")

    ws_thread = threading.Thread(target=_run_ws, daemon=True)
    ws_thread.start()

    _display_loop(params_ref, source_path_ref)


if __name__ == "__main__":
    main()
