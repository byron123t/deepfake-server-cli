"""
Rope Face-Swap Client
=====================
Captures webcam frames, detects faces locally using MediaPipe,
sends face crops + 5-pt keypoints to the server, and displays the
swapped result composited back onto the live video.

Usage:
    python client.py --server ws://SERVER_IP:8765/ws --source /path/to/source.jpg
    python client.py --server ws://SERVER_IP:8765/ws --source /path/to/source_folder/

    Or set values in .env and run: python client.py

Keys:
    q    - quit
    s    - resend source to server (re-reads --source file or folder)
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

from dotenv import load_dotenv
load_dotenv()

import cv2
import mediapipe as mp
import msgpack
import numpy as np
import websockets

# ------------------------------------------------------------------ #
# MediaPipe setup                                                      #
# ------------------------------------------------------------------ #
_mp_fm = mp.solutions.face_mesh

_face_mesh = _mp_fm.FaceMesh(
    static_image_mode=False,
    max_num_faces=4,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# MediaPipe FaceMesh landmark indices -> ArcFace 5-point format
# [left_eye, right_eye, nose_tip, left_mouth_corner, right_mouth_corner]
_ARCFACE_LM = [33, 263, 1, 61, 291]

_CROP_PAD = 0.4   # padding ratio around detected face bbox

_SOURCE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def detect_faces(frame_rgb: np.ndarray) -> list[dict]:
    """
    Returns a list of dicts:
      { 'crop_rgb': HxWx3, 'kps': 5x2 float32, 'bbox': [cx1,cy1,cx2,cy2] }
    kps are in crop-relative coordinates.
    """
    h, w = frame_rgb.shape[:2]
    results = _face_mesh.process(frame_rgb)
    if not results.multi_face_landmarks:
        return []

    faces = []
    for face_lm in results.multi_face_landmarks:
        lms = face_lm.landmark

        pts = np.array([[lms[i].x * w, lms[i].y * h] for i in _ARCFACE_LM], dtype=np.float32)

        all_x = np.array([lm.x * w for lm in lms])
        all_y = np.array([lm.y * h for lm in lms])
        fx1, fy1 = np.min(all_x), np.min(all_y)
        fx2, fy2 = np.max(all_x), np.max(all_y)

        face_w = fx2 - fx1
        face_h = fy2 - fy1
        pad_x = face_w * _CROP_PAD
        pad_y = face_h * _CROP_PAD

        cx1 = max(int(fx1 - pad_x), 0)
        cy1 = max(int(fy1 - pad_y), 0)
        cx2 = min(int(fx2 + pad_x), w)
        cy2 = min(int(fy2 + pad_y), h)

        crop_rgb = frame_rgb[cy1:cy2, cx1:cx2].copy()
        if crop_rgb.size == 0:
            continue

        kps_local = pts - np.array([[cx1, cy1]], dtype=np.float32)

        faces.append({
            "crop_rgb": crop_rgb,
            "kps": kps_local,
            "bbox": [cx1, cy1, cx2, cy2],
        })

    return faces


def _collect_source_images(path: str) -> list[bytes]:
    """Return JPEG-encoded bytes for each source image found at path (file or dir)."""
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
# Shared state between capture thread and async WS loop               #
# ------------------------------------------------------------------ #
_frame_q: queue.Queue = queue.Queue(maxsize=2)
_result_q: queue.Queue = queue.Queue(maxsize=8)


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
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        faces = detect_faces(frame_rgb)

        if _frame_q.full():
            try:
                _frame_q.get_nowait()
            except queue.Empty:
                pass
        _frame_q.put((frame_rgb, faces))

    cap.release()


# ------------------------------------------------------------------ #
# Async WebSocket client                                               #
# ------------------------------------------------------------------ #

async def _ws_client(uri: str, source_path: str, params_ref: dict):
    print(f"Connecting to {uri} …")
    async with websockets.connect(uri, max_size=20 * 1024 * 1024) as ws:
        print("Connected.")

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
        try:
            frame_rgb, faces = await loop.run_in_executor(
                None, lambda: _frame_q.get(timeout=1.0))
        except queue.Empty:
            continue

        if not faces:
            _result_q.put({"frame_rgb": frame_rgb, "faces": []})
            continue

        msg_faces = []
        for f in faces:
            ok, buf = cv2.imencode(".jpg", cv2.cvtColor(f["crop_rgb"], cv2.COLOR_RGB2BGR),
                                   [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ok:
                continue
            msg_faces.append({
                "crop": buf.tobytes(),
                "kps": f["kps"].tolist(),
                "bbox": f["bbox"],
            })

        if not msg_faces:
            _result_q.put({"frame_rgb": frame_rgb, "faces": []})
            continue

        payload = msgpack.packb({
            "type": "swap",
            "frame_id": frame_id,
            "params": params_ref.copy(),
            "faces": msg_faces,
        }, use_bin_type=True)
        await ws.send(payload)

        _result_q.put({"frame_rgb": frame_rgb.copy(), "frame_id": frame_id, "pending": True})
        frame_id += 1
        await asyncio.sleep(0)


async def _receiver(ws):
    async for raw in ws:
        msg = msgpack.unpackb(raw, raw=False)
        mtype = msg.get("type")

        if mtype == "source_set":
            if msg.get("success"):
                n_ok = msg.get("faces_used", "?")
                n_sent = msg.get("images_sent", "?")
                print(f"[source_set] OK — {n_ok}/{n_sent} image(s) had a detectable face")
            else:
                detail = msg.get("message", "no face found")
                print(f"[source_set] FAILED: {detail}")

        elif mtype == "source_ready":
            print("[server] Source face ready (loaded at server startup).")

        elif mtype == "swapped":
            fid = msg.get("frame_id")
            pending = None
            tmp = []
            try:
                while True:
                    item = _result_q.get_nowait()
                    if item.get("frame_id") == fid and item.get("pending"):
                        pending = item
                    else:
                        tmp.append(item)
            except queue.Empty:
                pass
            for t in tmp:
                try:
                    _result_q.put_nowait(t)
                except queue.Full:
                    pass

            if pending is None:
                continue

            frame_rgb = pending["frame_rgb"]
            for face_result in msg.get("faces", []):
                swapped_bytes = face_result["swapped"]
                bbox = face_result["bbox"]
                cx1, cy1, cx2, cy2 = bbox

                nparr = np.frombuffer(swapped_bytes, np.uint8)
                swapped_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                swapped_rgb = cv2.cvtColor(swapped_bgr, cv2.COLOR_BGR2RGB)

                crop_h, crop_w = cy2 - cy1, cx2 - cx1
                if swapped_rgb.shape[:2] != (crop_h, crop_w):
                    swapped_rgb = cv2.resize(swapped_rgb, (crop_w, crop_h))

                frame_rgb[cy1:cy2, cx1:cx2] = swapped_rgb

            try:
                _result_q.put_nowait({"frame_rgb": frame_rgb, "faces": [], "ready": True})
            except queue.Full:
                pass

        elif mtype == "error":
            print(f"[server error] {msg.get('message')}")


# ------------------------------------------------------------------ #
# Display loop (runs in main thread)                                   #
# ------------------------------------------------------------------ #

def _display_loop(params_ref: dict, source_path_ref: list):
    latest_frame = None
    fps_t = time.time()
    fps_count = 0
    fps_display = 0.0

    print("Display started. Press 'q' to quit, 's' to resend source, 'r' to toggle restorer.")

    while True:
        try:
            while True:
                item = _result_q.get_nowait()
                if item.get("ready") or not item.get("pending"):
                    latest_frame = item["frame_rgb"]
        except queue.Empty:
            pass

        if latest_frame is not None:
            display = cv2.cvtColor(latest_frame, cv2.COLOR_RGB2BGR)

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
            if source_path_ref[0] and _ws_loop:
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
