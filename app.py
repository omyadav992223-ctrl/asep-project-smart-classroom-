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

import threading
from flask import Flask, Response, render_template, jsonify
from camera import VideoCamera

app    = Flask(__name__)
camera = VideoCamera(camera_index=0)
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
    Live metrics polled by the frontend every ~1 second.

    Response fields (Phase 1.5):
      status, emotion, focus_score, ear, yaw, pitch, roll,
      gaze, gaze_h, gaze_v, boredom, fps
    """
    return jsonify(camera.get_metrics())


@app.route("/session_report")
def session_report():
    """
    On-demand session summary for the Research Paper Results section.
    Returns JSON with focus stats, drowsiness alerts, gaze distribution,
    emotion distribution, and boredom metrics.
    """
    return jsonify(camera.get_session_report())


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
