"""
test_params.py — Side-by-side parameter comparison grid.

Freezes one frame from the webcam (or a static --image), sends it to the
Rope server with each preset below, then displays all results in a labeled
grid so you can visually judge quality differences.

Usage:
    python test_params.py
    python test_params.py --image /path/to/face.jpg
    python test_params.py --server ws://1.2.3.4:8765/ws

Keys:
    n / space  — capture new snapshot from webcam
    s          — save current grid to PNG
    q          — quit
"""

import argparse
import asyncio
import math
import os
import time

import cv2
import msgpack
import numpy as np
import websockets
from dotenv import load_dotenv

load_dotenv()

# ------------------------------------------------------------------ #
# Presets                                                              #
# ------------------------------------------------------------------ #
PRESETS = [
    {"label": "swap only",          "params": {}},
    {"label": "mouth mask",         "params": {"mouth_mask": True}},
    {"label": "opacity 0.75",       "params": {"opacity": 0.75}},
    {"label": "mask + opacity 0.75","params": {"mouth_mask": True, "opacity": 0.75}},
    {"label": "GPEN-256",           "params": {"enhance": "gpen256"}},
    {"label": "GPEN-256 + mask",    "params": {"enhance": "gpen256", "mouth_mask": True}},
    {"label": "GPEN-512",           "params": {"enhance": "gpen512"}},
    {"label": "GPEN-512 + mask",    "params": {"enhance": "gpen512", "mouth_mask": True}},
]

CELL_W = 480
CELL_H = 270
LABEL_H = 28
GRID_COLS = 4
FONT = cv2.FONT_HERSHEY_SIMPLEX


# ------------------------------------------------------------------ #
# Grid rendering                                                       #
# ------------------------------------------------------------------ #

def _make_grid(results: list[tuple[str, np.ndarray | None, float]]) -> np.ndarray:
    """results: list of (label, bgr_frame_or_None, elapsed_ms)"""
    n = len(results)
    cols = GRID_COLS
    rows = math.ceil(n / cols)
    cell_h_full = CELL_H + LABEL_H
    canvas = np.zeros((rows * cell_h_full, cols * CELL_W, 3), dtype=np.uint8)

    for i, (label, frame, elapsed) in enumerate(results):
        row, col = divmod(i, cols)
        y0 = row * cell_h_full
        x0 = col * CELL_W

        # Label bar
        cv2.rectangle(canvas, (x0, y0), (x0 + CELL_W, y0 + LABEL_H), (30, 30, 30), -1)
        timing = f"  {elapsed:.0f}ms" if elapsed > 0 else ""
        cv2.putText(canvas, label + timing, (x0 + 6, y0 + 19),
                    FONT, 0.5, (200, 220, 255), 1, cv2.LINE_AA)

        # Frame
        fy0 = y0 + LABEL_H
        if frame is not None:
            thumb = cv2.resize(frame, (CELL_W, CELL_H), interpolation=cv2.INTER_AREA)
            canvas[fy0:fy0 + CELL_H, x0:x0 + CELL_W] = thumb
        else:
            # Placeholder while waiting
            cv2.rectangle(canvas, (x0, fy0), (x0 + CELL_W, fy0 + CELL_H), (40, 40, 40), -1)
            cv2.putText(canvas, "waiting…", (x0 + CELL_W // 2 - 45, fy0 + CELL_H // 2),
                        FONT, 0.6, (120, 120, 120), 1, cv2.LINE_AA)

        # Border
        cv2.rectangle(canvas, (x0, y0), (x0 + CELL_W - 1, y0 + cell_h_full - 1),
                      (80, 80, 80), 1)

    return canvas


# ------------------------------------------------------------------ #
# Async runner                                                         #
# ------------------------------------------------------------------ #

async def _run_comparison(uri: str, source_images: list[bytes],
                          frame_bytes: bytes) -> list[tuple[str, np.ndarray, float]]:
    """Send the frame with each preset sequentially; return (label, bgr, ms) list."""
    results = []
    async with websockets.connect(uri, max_size=20 * 1024 * 1024) as ws:
        # Send source
        await ws.send(msgpack.packb(
            {"type": "set_source", "images": source_images}, use_bin_type=True))

        # Wait for source_set ack
        while True:
            raw = await asyncio.wait_for(ws.recv(), timeout=15.0)
            msg = msgpack.unpackb(raw, raw=False)
            if msg.get("type") == "source_set":
                if not msg.get("success"):
                    raise RuntimeError(f"source_set failed: {msg.get('message')}")
                break
            if msg.get("type") == "source_ready":
                break

        # Run each preset
        for i, preset in enumerate(PRESETS):
            label = preset["label"]
            params = preset["params"].copy()
            payload = msgpack.packb({
                "type": "frame",
                "frame_id": i,
                "frame": frame_bytes,
                "params": params,
            }, use_bin_type=True)

            t0 = time.perf_counter()
            await ws.send(payload)

            # Wait for result
            while True:
                raw = await asyncio.wait_for(ws.recv(), timeout=30.0)
                msg = msgpack.unpackb(raw, raw=False)
                if msg.get("type") == "frame_result" and msg.get("frame_id") == i:
                    elapsed = (time.perf_counter() - t0) * 1000
                    fb = msg.get("frame")
                    nparr = np.frombuffer(fb, np.uint8)
                    bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    results.append((label, bgr, elapsed))
                    print(f"  [{i+1}/{len(PRESETS)}] {label}: {elapsed:.0f} ms")
                    break
                if msg.get("type") == "error":
                    print(f"  [{i+1}/{len(PRESETS)}] {label}: server error — {msg.get('message')}")
                    results.append((label, None, 0.0))
                    break

    return results


# ------------------------------------------------------------------ #
# Source helpers                                                       #
# ------------------------------------------------------------------ #

_SOURCE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _load_source_images(path: str) -> list[bytes]:
    if os.path.isfile(path):
        img = cv2.imread(path)
        if img is None:
            return []
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return [buf.tobytes()] if ok else []

    if os.path.isdir(path):
        out = []
        for ext in _SOURCE_EXTS:
            import glob
            for p in sorted(glob.glob(os.path.join(path, f"*{ext}")) +
                            glob.glob(os.path.join(path, f"*{ext.upper()}"))):
                img = cv2.imread(p)
                if img is None:
                    continue
                ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                if ok:
                    out.append(buf.tobytes())
        return out
    return []


def _capture_frame(camera_id: int, width: int, height: int) -> np.ndarray | None:
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # Read a few frames so the camera AE settles
    for _ in range(5):
        cap.read()
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def _frame_to_jpeg(bgr: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes() if ok else b""


# ------------------------------------------------------------------ #
# Entry point                                                          #
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="Rope parameter comparison grid",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--server", default=os.getenv("SERVER_URL", "ws://127.0.0.1:8765/ws"))
    parser.add_argument("--source", default=os.getenv("SOURCE_PATH", ""),
                        help="Source face image or folder")
    parser.add_argument("--image", default="",
                        help="Static test image instead of webcam snapshot")
    parser.add_argument("--camera", type=int, default=int(os.getenv("CAMERA_INDEX", "0")))
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    args = parser.parse_args()

    if not args.source:
        print("ERROR: --source is required (source face image or folder)")
        raise SystemExit(1)

    source_images = _load_source_images(args.source)
    if not source_images:
        print(f"ERROR: No images found at source path: {args.source}")
        raise SystemExit(1)

    # Build placeholder grid while we wait
    placeholders = [(p["label"], None, 0.0) for p in PRESETS]
    grid = _make_grid(placeholders)
    cv2.imshow("Rope — Parameter Comparison", grid)
    cv2.waitKey(1)

    def _snapshot() -> bytes:
        if args.image:
            img = cv2.imread(args.image)
            if img is None:
                print(f"ERROR: cannot read image: {args.image}")
                raise SystemExit(1)
            return _frame_to_jpeg(img)
        print(f"Capturing snapshot from camera {args.camera} …")
        frame = _capture_frame(args.camera, args.width, args.height)
        if frame is None:
            print(f"ERROR: failed to capture from camera {args.camera}")
            raise SystemExit(1)
        return _frame_to_jpeg(frame)

    def _run():
        frame_bytes = _snapshot()
        print(f"Running {len(PRESETS)} presets …")
        results = asyncio.run(
            _run_comparison(args.server, source_images, frame_bytes))
        return results

    results = _run()
    grid = _make_grid(results)
    cv2.imshow("Rope — Parameter Comparison", grid)

    save_idx = 0
    print("Grid ready.  n/space=new snapshot  s=save PNG  q=quit")
    while True:
        key = cv2.waitKey(50) & 0xFF
        if key == ord('q'):
            break
        elif key in (ord('n'), ord(' ')):
            print("Re-running …")
            results = _run()
            grid = _make_grid(results)
            cv2.imshow("Rope — Parameter Comparison", grid)
        elif key == ord('s'):
            path = f"rope_test_{save_idx:03d}.png"
            cv2.imwrite(path, grid)
            print(f"Saved: {path}")
            save_idx += 1

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
