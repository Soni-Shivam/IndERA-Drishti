"""
app.py
------
Flask web server for the ESP32-CAM + PaliGemma offline inference dashboard.

Endpoints
---------
GET  /                  → Serve the main dashboard HTML
GET  /video_feed        → MJPEG stream from ESP32-CAM (full frame)
GET  /crop_feed         → MJPEG stream of the cropped ROI (real-time)
POST /set_roi           → Set crop region {x,y,w,h} as 0-1 normalised floats
GET  /get_roi           → Return current ROI
GET  /events            → SSE stream of inference results (JSON)
POST /capture           → Snapshot cropped ROI → run inference → broadcast result
GET  /status            → JSON: model load status, last result, camera status
POST /settings          → Update ESP32 stream URL, inventory groups
"""

import io
import json
import queue
import threading
import time
from datetime import datetime

import cv2
import numpy as np
import requests
from flask import Flask, Response, jsonify, render_template, request, stream_with_context
from PIL import Image

import model_engine

# ---------------------------------------------------------------------------
# App + shared state
# ---------------------------------------------------------------------------
app = Flask(__name__)

# Thread-safe queue for SSE broadcast
_result_queue: queue.Queue = queue.Queue(maxsize=50)

# Shared camera frame (bytes, JPEG) — always the FULL raw frame
_frame_lock = threading.Lock()
_current_frame: bytes | None = None

# Region-of-interest: normalised [0,1] coordinates within the full frame
# {x, y, w, h}  where (x,y) is top-left corner
_roi_lock = threading.Lock()
_roi: dict = {"x": 0.1, "y": 0.1, "w": 0.8, "h": 0.8}  # sensible default

# Last inference result (thread-safe via lock)
_result_lock = threading.Lock()
_last_result: dict = {}

# Last image sent to the model (JPEG bytes)
_last_capture_lock = threading.Lock()
_last_capture_jpeg: bytes | None = None

# Camera thread state
_camera_thread: threading.Thread | None = None
_camera_running = threading.Event()

# Model loading state
_model_ready = threading.Event()
_model_error: str = ""

# Settings (mutable at runtime)
_settings_lock = threading.Lock()
_settings = {
    "stream_url": "http://10.53.7.152:81/stream",
    "snapshot_url": "http://10.53.7.152:80/capture",
    "inventory_groups": model_engine.INVENTORY_GROUPS[:],
    "use_snapshot_endpoint": True,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _apply_roi(frame_bgr: np.ndarray) -> np.ndarray:
    """Crop a BGR numpy frame using the current _roi (normalised coords)."""
    with _roi_lock:
        roi = dict(_roi)

    h, w = frame_bgr.shape[:2]
    x1 = max(0, int(roi["x"] * w))
    y1 = max(0, int(roi["y"] * h))
    x2 = min(w, int((roi["x"] + roi["w"]) * w))
    y2 = min(h, int((roi["y"] + roi["h"]) * h))

    if x2 <= x1 or y2 <= y1:
        return frame_bgr  # degenerate ROI → return full frame

    return frame_bgr[y1:y2, x1:x2]


def _decode_current_frame() -> np.ndarray | None:
    """Decode the latest JPEG frame buffer into a BGR numpy array."""
    with _frame_lock:
        data = _current_frame
    if data is None:
        return None
    arr = np.frombuffer(data, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


# ---------------------------------------------------------------------------
# Background: model loader
# ---------------------------------------------------------------------------

def _load_model_background():
    global _model_error
    try:
        model_engine.load_model()
        _model_ready.set()
    except Exception as exc:
        _model_error = str(exc)
        _model_ready.set()


# ---------------------------------------------------------------------------
# Background: camera reader
# ---------------------------------------------------------------------------

def _camera_loop():
    global _current_frame

    while _camera_running.is_set():
        with _settings_lock:
            url = _settings["stream_url"]

        cap = cv2.VideoCapture(url)

        if not cap.isOpened():
            print(f"[Camera] Cannot open stream: {url}. Retrying in 3 s…")
            time.sleep(3)
            continue

        print(f"[Camera] Stream opened: {url}")

        while _camera_running.is_set():
            ret, frame = cap.read()
            if not ret:
                print("[Camera] Frame grab failed — reconnecting…")
                break

            # Rotate -90° (counterclockwise) to correct ESP32-CAM orientation
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

            ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                with _frame_lock:
                    _current_frame = buf.tobytes()

        cap.release()

    print("[Camera] Thread exiting.")


# ---------------------------------------------------------------------------
# Inference helper — operates on the CROPPED region
# ---------------------------------------------------------------------------

def _do_inference() -> dict:
    """Grab the current ROI crop, run inference, broadcast result via SSE."""
    with _settings_lock:
        use_snapshot = _settings["use_snapshot_endpoint"]
        snap_url = _settings["snapshot_url"]
        # groups = _settings["inventory_groups"][:]
        groups = _settings["inventory_groups"][:] # <--- This is correct in your file

    image: Image.Image | None = None

    if use_snapshot:
        try:
            resp = requests.get(snap_url, timeout=5)
            resp.raise_for_status()
            full = Image.open(io.BytesIO(resp.content)).convert("RGB")
            # Apply the same -90° rotation used in the camera loop
            full = full.rotate(90, expand=True)   # PIL rotate(90) = counterclockwise 90°
            # Apply ROI to the now-rotated snapshot
            with _roi_lock:
                roi = dict(_roi)
            fw, fh = full.size
            x1 = max(0, int(roi["x"] * fw))
            y1 = max(0, int(roi["y"] * fh))
            x2 = min(fw, int((roi["x"] + roi["w"]) * fw))
            y2 = min(fh, int((roi["y"] + roi["h"]) * fh))
            if x2 > x1 and y2 > y1:
                image = full.crop((x1, y1, x2, y2))
            else:
                image = full
        except Exception as exc:
            print(f"[Inference] Snapshot endpoint failed ({exc}), falling back to frame buffer")

    if image is None:
        frame_bgr = _decode_current_frame()
        if frame_bgr is None:
            return {"error": "No frame available. Is the camera stream running?"}
        cropped_bgr = _apply_roi(frame_bgr)
        image = Image.fromarray(cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB))

    # ── Save the exact image sent to the model ────────────────
    global _last_capture_jpeg
    _buf = io.BytesIO()
    image.save(_buf, format="JPEG", quality=90)
    with _last_capture_lock:
        _last_capture_jpeg = _buf.getvalue()
    # ─────────────────────────────────────────────────────────

    # result = model_engine.run_inference(image, groups)
    result = model_engine.run_inference(image, inventory_groups=groups)
    result["timestamp"] = datetime.now().isoformat()

    with _result_lock:
        _last_result.update(result)

    try:
        _result_queue.put_nowait(result)
    except queue.Full:
        pass

    return result


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    """Full-frame MJPEG proxy of the ESP32 stream — draws the ROI rectangle."""

    def generate():
        while True:
            frame_bgr = _decode_current_frame()

            if frame_bgr is None:
                time.sleep(0.05)
                continue

            # Draw the current ROI as an overlay rectangle
            with _roi_lock:
                roi = dict(_roi)
            h, w = frame_bgr.shape[:2]
            x1 = int(roi["x"] * w)
            y1 = int(roi["y"] * h)
            x2 = int((roi["x"] + roi["w"]) * w)
            y2 = int((roi["y"] + roi["h"]) * h)

            vis = frame_bgr.copy()
            # Semi-transparent dark mask outside ROI
            mask = np.zeros_like(vis)
            mask[:, :] = [0, 0, 0]
            # Draw bright rectangle
            cv2.rectangle(vis, (x1, y1), (x2, y2), (80, 200, 255), 2)
            # Corner ticks
            tick = 16
            cv2.line(vis, (x1, y1), (x1 + tick, y1), (80, 200, 255), 3)
            cv2.line(vis, (x1, y1), (x1, y1 + tick), (80, 200, 255), 3)
            cv2.line(vis, (x2, y1), (x2 - tick, y1), (80, 200, 255), 3)
            cv2.line(vis, (x2, y1), (x2, y1 + tick), (80, 200, 255), 3)
            cv2.line(vis, (x1, y2), (x1 + tick, y2), (80, 200, 255), 3)
            cv2.line(vis, (x1, y2), (x1, y2 - tick), (80, 200, 255), 3)
            cv2.line(vis, (x2, y2), (x2 - tick, y2), (80, 200, 255), 3)
            cv2.line(vis, (x2, y2), (x2, y2 - tick), (80, 200, 255), 3)
            # Label
            cv2.putText(vis, "ROI", (x1 + 6, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 200, 255), 1, cv2.LINE_AA)

            ok, buf = cv2.imencode(".jpg", vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
                )
            time.sleep(0.04)

    return Response(
        stream_with_context(generate()),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/crop_feed")
def crop_feed():
    """Real-time MJPEG stream of the cropped ROI only."""

    def generate():
        while True:
            frame_bgr = _decode_current_frame()

            if frame_bgr is None:
                time.sleep(0.05)
                continue

            cropped = _apply_roi(frame_bgr)
            ok, buf = cv2.imencode(".jpg", cropped, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if ok:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
                )
            time.sleep(0.04)

    return Response(
        stream_with_context(generate()),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/set_roi", methods=["POST"])
def set_roi():
    """Set the crop region. Body: {x, y, w, h} as floats in [0, 1]."""
    data = request.get_json(force=True)
    try:
        x = float(data["x"])
        y = float(data["y"])
        w = float(data["w"])
        h = float(data["h"])
    except (KeyError, ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid ROI params: {e}"}), 400

    # Clamp & validate
    x = max(0.0, min(0.99, x))
    y = max(0.0, min(0.99, y))
    w = max(0.01, min(1.0 - x, w))
    h = max(0.01, min(1.0 - y, h))

    with _roi_lock:
        _roi.update({"x": x, "y": y, "w": w, "h": h})

    print(f"[ROI] Updated → x={x:.3f} y={y:.3f} w={w:.3f} h={h:.3f}")
    return jsonify({"ok": True, "roi": _roi})


@app.route("/last_capture")
def last_capture():
    """Return the exact JPEG image that was sent to PaliGemma for the last inference."""
    with _last_capture_lock:
        data = _last_capture_jpeg
    if data is None:
        return Response(status=204)
    return Response(
        data,
        mimetype="image/jpeg",
        headers={"Cache-Control": "no-store, no-cache", "Pragma": "no-cache"},
    )


@app.route("/get_roi")
def get_roi():
    with _roi_lock:
        return jsonify(dict(_roi))


@app.route("/events")
def events():
    """SSE endpoint — each event is a JSON inference result."""

    def generate():
        yield 'data: {"type": "connected"}\n\n'
        while True:
            try:
                item = _result_queue.get(timeout=1.0)
                payload = json.dumps({"type": "result", **item})
                yield f"data: {payload}\n\n"
            except queue.Empty:
                yield ": heartbeat\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/capture", methods=["POST"])
def capture():
    """Trigger a snapshot (cropped) + inference. Returns result as JSON."""
    if not _model_ready.is_set():
        return jsonify({"error": "Model is still loading…"}), 503
    if _model_error:
        return jsonify({"error": f"Model failed to load: {_model_error}"}), 500

    result_holder = {}
    error_holder = {}

    def _run():
        try:
            result_holder.update(_do_inference())
        except Exception as exc:
            error_holder["error"] = str(exc)

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t.join(timeout=60)

    if error_holder:
        return jsonify(error_holder), 500

    return jsonify(result_holder)


@app.route("/status")
def status():
    with _result_lock:
        last = dict(_last_result)
    with _settings_lock:
        settings = dict(_settings)
    with _roi_lock:
        roi = dict(_roi)

    return jsonify({
        "model_ready": _model_ready.is_set(),
        "model_error": _model_error,
        "camera_running": _camera_running.is_set(),
        "stream_url": settings["stream_url"],
        "inventory_groups": settings["inventory_groups"],
        "last_result": last,
        "roi": roi,
    })


@app.route("/settings", methods=["POST"])
def update_settings():
    data = request.get_json(force=True)
    with _settings_lock:
        if "stream_url" in data:
            _settings["stream_url"] = data["stream_url"]
        if "snapshot_url" in data:
            _settings["snapshot_url"] = data["snapshot_url"]
        if "inventory_groups" in data and isinstance(data["inventory_groups"], list):
            _settings["inventory_groups"] = data["inventory_groups"]
        if "use_snapshot_endpoint" in data:
            _settings["use_snapshot_endpoint"] = bool(data["use_snapshot_endpoint"])

    if "stream_url" in data:
        _camera_running.clear()
        time.sleep(0.5)
        _camera_running.set()
        global _camera_thread
        _camera_thread = threading.Thread(target=_camera_loop, daemon=True)
        _camera_thread.start()

    return jsonify({"ok": True, "settings": _settings})


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    threading.Thread(target=_load_model_background, daemon=True).start()

    _camera_running.set()
    _camera_thread = threading.Thread(target=_camera_loop, daemon=True)
    _camera_thread.start()

    print("=" * 60)
    print("  INDERA Vision Dashboard")
    print("  Open: http://127.0.0.1:5000")
    print("=" * 60)

    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
