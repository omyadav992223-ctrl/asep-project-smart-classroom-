"""
app.py  --  Flask Server for Smart Classroom AI (Phase 1.5)
===========================================================
Routes:
  GET  /                 -> Dashboard UI
  GET  /video_feed       -> MJPEG stream
  GET  /data_stream      -> JSON metrics snapshot
  GET  /session_report   -> JSON session summary report
  POST /export_session   -> Save CSV log
  POST /export_report    -> Save JSON report
  POST /reset_session    -> Clear in-memory log

Author  : Smart Classroom AI
Version : 2.0.0  (Phase 1.5)
"""

import json
import time
import threading
from flask import Flask, Response, render_template, jsonify, stream_with_context
from camera import VideoCamera, list_available_cameras

app    = Flask(__name__)

try:
    camera = VideoCamera(camera_index=0)   # index 0 = currently active webcam
except RuntimeError as e:
    print(f"\n{'='*55}")
    print(f" ERROR: {e}")
    print(f" Connect your webcam and restart the server.")
    print(f"{'='*55}\n")
    raise

_lock  = threading.Lock()


# ---------------------------------------------------------------------------
# Frame generator
# ---------------------------------------------------------------------------

def _gen():
    while True:
        frame = camera.get_frame()
        if frame is None:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(_gen(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/data_stream")
def data_stream():
    """
    Server-Sent Events stream that pushes live classroom metrics every 1 s.

    The browser connects once via EventSource('/data_stream') and receives
    a continuous push of JSON metric objects, eliminating the need for
    repeated polling requests.

    Event format (parsed by script.js applyMetrics):
      event: metrics
      data: {face_count, focus_score, emotion, gaze_state, boredom, fps, ...}
    """
    def _generate():
        while True:
            try:
                metrics = camera.get_metrics()
                payload = json.dumps(metrics)
                # SSE format: event name + data line + blank line terminator
                yield f"event: metrics\ndata: {payload}\n\n"
            except Exception as e:
                yield f"event: error\ndata: {{\"msg\": \"{e}\"}}\n\n"
            time.sleep(1.0)

    return Response(
        stream_with_context(_generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control" : "no-cache",
            "X-Accel-Buffering": "no",    # disable nginx buffering if proxied
        },
    )


@app.route("/cameras")
def cameras():
    """List available webcam indices on this machine."""
    available = list_available_cameras()
    return jsonify({"available_cameras": available})


@app.route("/session_report")
def session_report():
    """
    On-demand session summary for the Research Paper Results section.
    Returns JSON with focus stats, drowsiness alerts, gaze distribution,
    emotion distribution, boredom metrics, and attendance roster.
    """
    return jsonify(camera.get_session_report())


@app.route("/attendance_live")
def attendance_live():
    """Lightweight endpoint: returns only the live attendance roster.

    Polled by the frontend every ~3 s to update the roster panel without
    triggering the full session-report computation.
    """
    roster = camera.logger.attendance_roster
    return jsonify({
        "total_present": len(roster),
        "students":      roster,
    })


@app.route("/export_session", methods=["POST"])
def export_session():
    path = camera.export_session()
    return jsonify({"status": "ok", "file": path})


@app.route("/export_report", methods=["POST"])
def export_report():
    """Export the session summary report as a JSON file."""
    path = camera.export_report()
    return jsonify({"status": "ok", "file": path})


@app.route("/reset_session", methods=["POST"])
def reset_session():
    camera.logger.clear()
    return jsonify({"status": "ok", "message": "Session cleared."})


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 55)
    print(" Smart Classroom AI - Phase 1.5")
    print(" Dashboard -> http://127.0.0.1:5000")
    print("=" * 55)
    try:
        app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
    finally:
        camera.release()
