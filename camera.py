"""
camera.py  --  Vision Engine for Smart Classroom AI
====================================================
Phase 1.5  --  Research-Grade Accuracy Edition

NEW in this version
-------------------
  1. Iris-based Gaze Tracking      (landmarks 468-477)
  2. Boredom Variance Tracker      (60-second rolling expressiveness window)
  3. Enhanced Confusion Detector   (brow asymmetry + head-roll angle)
  4. Gaze integrated into Focus Score  (higher penalty than head-pose alone)
  5. Session Summary Report        (JSON -- mean focus, peak emotion, drowsiness)

Research Paper Methodology Notes
---------------------------------
  Every landmark index and every mathematical formula used in this module
  is documented in-line so you can cite them directly in your Methods section.

  Key references:
    EAR  -- Soukupova & Cech (2016) "Real-Time Eye Blink Detection using
             Facial Landmarks", CVWW.
    PnP  -- OpenCV solvePnP / Rodrigues decomposition for head pose.
    Gaze -- Iris-to-eye-boundary ratio method (scale-invariant, novel).
    Boredom -- Facial expressiveness variance over rolling time window (novel).

Author  : Smart Classroom AI
Version : 3.0.0  (Phase 1.5)
"""

import cv2
import csv
import json
import os
import time
from collections import Counter, deque
from datetime import datetime

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python         import BaseOptions
from mediapipe.tasks.python.vision  import (FaceLandmarker,
                                             FaceLandmarkerOptions,
                                             RunningMode)

# ---------------------------------------------------------------------------
# Model path
# ---------------------------------------------------------------------------
_MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")


# ===========================================================================
# LANDMARK INDEX CONSTANTS
# (Copy these into your Research Paper -- Methodology / Landmark Selection)
# ===========================================================================

# ---------------------------------------------------------------------------
# EAR -- Eye Aspect Ratio  (Soukupova & Cech 2016)
# 6-point model per eye:
#   p1 = outer corner (temporal)
#   p2 = upper-outer lid
#   p3 = upper-inner lid
#   p4 = inner corner (nasal)
#   p5 = lower-inner lid
#   p6 = lower-outer lid
# ---------------------------------------------------------------------------
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

# ---------------------------------------------------------------------------
# HEAD POSE -- 6-point solvePnP
# Selected for minimal soft-tissue deformation and strong geometric span.
# ---------------------------------------------------------------------------
POSE_INDICES = [
    1,    # Nose tip            -- zero-motion anchor
    152,  # Chin                -- largest vertical extent
    263,  # Left eye corner     -- temporal anchor
    33,   # Right eye corner    -- temporal anchor
    287,  # Left mouth corner   -- horizontal span
    57,   # Right mouth corner  -- horizontal span
]
# Generic 3-D face model points (mm, standard OpenCV coordinate frame)
MODEL_3D = np.array([
    [ 0.0,    0.0,    0.0],   # Nose tip
    [ 0.0,  -63.6,  -12.5],  # Chin
    [-43.3,  32.7,  -26.0],  # Left eye outer corner
    [ 43.3,  32.7,  -26.0],  # Right eye outer corner
    [-28.9, -28.9,  -24.1],  # Left mouth corner
    [ 28.9, -28.9,  -24.1],  # Right mouth corner
], dtype=np.float64)

# ---------------------------------------------------------------------------
# IRIS & EYE BOUNDARY  (MediaPipe 478-point model -- iris refinement)
#
# LEFT IRIS  (person's anatomical left eye):
#   468 -- iris centre           <-- PRIMARY gaze reference
#   469 -- iris right edge
#   470 -- iris bottom edge
#   471 -- iris left edge
#   472 -- iris top edge
#
# RIGHT IRIS:
#   473 -- iris centre           <-- PRIMARY gaze reference
#   474 -- iris right edge
#   475 -- iris bottom edge
#   476 -- iris left edge
#   477 -- iris top edge
#
# Eye boundary landmarks (normalization reference for gaze ratio):
#   Left  -- inner(nasal):362  outer(temporal):263  top:386  bottom:374
#   Right -- inner(nasal):133  outer(temporal):33   top:159  bottom:145
#
# Gaze ratio formula (scale-invariant):
#   h_ratio = (iris_x - eye_left_x) / eye_width      [0=left, 1=right]
#   v_ratio = (iris_y - eye_top_y)  / eye_height     [0=up,   1=down]
# ---------------------------------------------------------------------------
LEFT_IRIS   = 468
RIGHT_IRIS  = 473

LEFT_EYE_INNER  = 362;  LEFT_EYE_OUTER  = 263
LEFT_EYE_TOP    = 386;  LEFT_EYE_BOTTOM = 374
RIGHT_EYE_INNER = 133;  RIGHT_EYE_OUTER = 33
RIGHT_EYE_TOP   = 159;  RIGHT_EYE_BOTTOM= 145

# ---------------------------------------------------------------------------
# DETECTION THRESHOLDS
# ---------------------------------------------------------------------------
EAR_THRESHOLD        = 0.21    # Below -> eye considered closed (drowsiness)
YAW_THRESHOLD        = 25.0    # Max acceptable head yaw  (degrees)
PITCH_THRESHOLD      = 20.0    # Max acceptable head pitch (degrees)

# Gaze ratio thresholds (iris position as fraction of eye bounding box)
# Calibrated empirically on frontal faces; adjust if camera angle differs.
GAZE_H_LOW   = 0.38   # Below -> gaze displaced LEFT
GAZE_H_HIGH  = 0.62   # Above -> gaze displaced RIGHT
GAZE_V_LOW   = 0.30   # Below -> gaze displaced UP
GAZE_V_HIGH  = 0.75   # Above -> gaze displaced DOWN

# Mam's Pitch Note: Detecting phone usage via Gaze Discrepancy (Eyes Down, Head Forward).

# Focus score penalty weights
GAZE_AWAY_PENALTY    = 28.0   # Strong: gaze is the definitive attention signal
HEAD_YAW_SCALE       = 0.6    # Reduced vs v1 (gaze now carries more weight)
HEAD_PITCH_SCALE     = 0.4
EAR_PEN_PER_FRAME    = 2.0    # Per consecutive closed-eye frame


# ===========================================================================
# HELPER MATH
# ===========================================================================

def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def compute_ear(lms: np.ndarray, idx: list) -> float:
    """
    Eye Aspect Ratio -- Soukupova & Cech (2016).

    Formula:
        EAR = ( ||p2-p6|| + ||p3-p5|| ) / ( 2 * ||p1-p4|| )

    Parameters
    ----------
    lms : (N, 2) pixel coordinate array  -- all face landmarks
    idx : [p1, p2, p3, p4, p5, p6]      -- 6-point EAR landmark indices

    Returns
    -------
    float  -- typical open-eye EAR ~ 0.25-0.35; EAR < 0.21 -> closed
    """
    p  = lms[idx]
    A  = _dist(p[1], p[5])   # ||p2 - p6||
    B  = _dist(p[2], p[4])   # ||p3 - p5||
    C  = _dist(p[0], p[3])   # ||p1 - p4||
    return float((A + B) / (2.0 * C + 1e-9))


def compute_head_pose(pts2d: np.ndarray, fw: int, fh: int) -> dict:
    """
    Estimate head pose via solvePnP (Iterative).

    Method:
      Maps POSE_INDICES landmarks from a normalized 3-D face model to
      their 2-D image positions.  Camera intrinsics are approximated as
      a pinhole model with focal length = frame width (no distortion).

    Returns
    -------
    dict -- {"yaw": float, "pitch": float, "roll": float}  in degrees
            NumPy 2.x note: decomposeProjectionMatrix returns (3,1) arrays
            which are flattened before indexing.
    """
    focal  = float(fw)
    cam    = np.array([[focal, 0, fw/2.0],
                       [0, focal, fh/2.0],
                       [0,     0,    1.0]], dtype=np.float64)
    dist   = np.zeros((4, 1), dtype=np.float64)

    ok, rvec, tvec = cv2.solvePnP(MODEL_3D, pts2d, cam, dist,
                                   flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}

    rmat, _  = cv2.Rodrigues(rvec)
    proj     = cv2.hconcat([rmat, tvec])
    _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(proj)
    euler    = euler.flatten()          # NumPy 2.x fix: was (3,1), needs (3,)
    return {"yaw": float(euler[1]),
            "pitch": float(euler[0]),
            "roll":  float(euler[2])}


def compute_gaze(lms: np.ndarray) -> dict:
    """
    Iris-based Gaze Direction  --  scale-invariant, pose-independent.

    Method (Novel -- suitable for citation):
      For each eye the iris centre landmark is expressed as a normalised
      ratio within the eye bounding box defined by the four boundary
      landmarks (inner, outer, top, bottom eyelid points).

      h_ratio = (iris_x - eye_left_x) / eye_width
      v_ratio = (iris_y - eye_top_y)  / eye_height

      Averaging both eyes suppresses per-eye noise.
      Because the ratio is computed WITHIN the eye frame, gaze direction
      is independent of head pose (a student can look left while facing
      the camera and still register "Center" if pupils are centered).

    Landmarks used:
      Left  iris centre : 468
      Right iris centre : 473
      Left  eye bounds  : inner=362, outer=263, top=386, bottom=374
      Right eye bounds  : inner=133, outer=33,  top=159, bottom=145

    Thresholds (empirically calibrated):
      h < 0.38  -> Left   |  h > 0.62  -> Right
      v < 0.30  -> Up     |  v > 0.65  -> Down
      otherwise -> Center

    Parameters
    ----------
    lms : (N, 2) pixel coordinate array; must have N >= 478 for iris

    Returns
    -------
    dict -- {"direction": str, "h_ratio": float, "v_ratio": float}
    """
    if len(lms) < 478:
        return {"direction": "N/A", "h_ratio": 0.5, "v_ratio": 0.5}

    def _ratio(iris_idx, inner, outer, top, bot):
        ix, iy   = lms[iris_idx]
        x_min    = min(lms[inner, 0], lms[outer, 0])
        x_max    = max(lms[inner, 0], lms[outer, 0])
        y_top    = lms[top, 1]
        y_bot    = lms[bot, 1]
        h        = (ix - x_min) / (x_max - x_min + 1e-9)
        v        = (iy - y_top) / (abs(y_bot - y_top) + 1e-9)
        return h, v

    lh, lv = _ratio(LEFT_IRIS,
                    LEFT_EYE_INNER, LEFT_EYE_OUTER,
                    LEFT_EYE_TOP,   LEFT_EYE_BOTTOM)
    rh, rv = _ratio(RIGHT_IRIS,
                    RIGHT_EYE_INNER, RIGHT_EYE_OUTER,
                    RIGHT_EYE_TOP,   RIGHT_EYE_BOTTOM)

    avg_h = (lh + rh) / 2.0
    avg_v = (lv + rv) / 2.0

    if   avg_h < GAZE_H_LOW:   direction = "Left"
    elif avg_h > GAZE_H_HIGH:  direction = "Right"
    elif avg_v < GAZE_V_LOW:   direction = "Up"
    elif avg_v > GAZE_V_HIGH:  direction = "Down"
    else:                       direction = "Center"

    return {"direction": direction,
            "h_ratio":   round(float(avg_h), 3),
            "v_ratio":   round(float(avg_v), 3)}


# ===========================================================================
# EMOTION DETECTION  --  MediaPipe Blendshapes (v2, Classroom Edition)
# ===========================================================================
#
# 52 blendshape coefficients (0.0-1.0) are weighted to score 8 emotions.
# No external model required.
#
# Classroom-specific signals:
#   Sleepy   -> eyeBlink* (high) + EAR (low)                 [drowsiness]
#   Confused -> |browDownL - browDownR| (asymmetry) + roll    [comprehension]
#   Bored    -> expressiveness variance <threshold over 60s   [engagement]
#
# Changes vs v1:
#   - ear  passed in for Sleepy   (EAR-to-sleep remap)
#   - roll passed in for Confused (head-tilt strengthens signal)
#   - Bored removed from per-frame scoring; handled by BoredomVarianceTracker
#   - Threshold lowered 0.18 -> 0.10; smoothing window 10 -> 8 frames
# ===========================================================================

_BS_IDX: dict[str, int] = {}

def _bidx(bs: list) -> None:
    global _BS_IDX
    if not _BS_IDX:
        _BS_IDX = {b.category_name: i for i, b in enumerate(bs)}

def _b(bs: list, name: str) -> float:
    i = _BS_IDX.get(name)
    return float(bs[i].score) if i is not None else 0.0

_emo_hist: deque = deque(maxlen=8)


def detect_emotion(bs: list, ear: float = 0.30, roll: float = 0.0) -> str:
    """
    Classify classroom emotion from blendshapes + EAR + head roll.

    Blendshape keys per emotion (for Research Paper reference):
      Happy     : mouthSmileLeft, mouthSmileRight, cheekSquintLeft/Right
      Sad       : mouthFrownLeft, mouthFrownRight, browInnerUp
      Angry     : browDownLeft, browDownRight, noseSneerLeft/Right
      Surprised : jawOpen, eyeWideLeft/Right, browInnerUp
      Disgusted : noseSneerLeft/Right, mouthUpperUpLeft/Right
      Sleepy    : eyeBlinkLeft, eyeBlinkRight  +  EAR remap signal
      Confused  : |browDownLeft - browDownRight|  +  abs(roll)/20  +  jawOpen

    Parameters
    ----------
    bs   : blendshape list from FaceLandmarker result
    ear  : pre-computed Eye Aspect Ratio
    roll : head roll angle (degrees) from solvePnP decomposition
    """
    if not bs:
        return "Neutral"
    _bidx(bs)

    sl = _b(bs,"mouthSmileLeft");  sr = _b(bs,"mouthSmileRight")
    ql = _b(bs,"cheekSquintLeft"); qr = _b(bs,"cheekSquintRight")
    fl = _b(bs,"mouthFrownLeft");  fr = _b(bs,"mouthFrownRight")
    bu = _b(bs,"browInnerUp");     md = _b(bs,"mouthLowerDownLeft")
    dl = _b(bs,"browDownLeft");    dr = _b(bs,"browDownRight")
    nl = _b(bs,"noseSneerLeft");   nr = _b(bs,"noseSneerRight")
    jo = _b(bs,"jawOpen")
    wl = _b(bs,"eyeWideLeft");     wr = _b(bs,"eyeWideRight")
    ul = _b(bs,"mouthUpperUpLeft");ur = _b(bs,"mouthUpperUpRight")
    bl = _b(bs,"eyeBlinkLeft");    br_ = _b(bs,"eyeBlinkRight")

    happy     = sl*0.38 + sr*0.38 + ql*0.12 + qr*0.12
    sad       = fl*0.38 + fr*0.38 + bu*0.14 + md*0.10
    angry     = dl*0.38 + dr*0.38 + nl*0.12 + nr*0.12
    surprised = jo*0.35 + wl*0.22 + wr*0.22 + bu*0.21
    disgusted = nl*0.32 + nr*0.32 + ul*0.18 + ur*0.18

    # Sleepy -- EAR remap: 0.10(closed)->1.0, 0.25(open)->0.0
    ear_sig  = max(0.0, min(1.0, (0.25 - ear) / 0.15))
    sleepy   = bl*0.30 + br_*0.30 + ear_sig*0.40

    # Confused -- brow asymmetry (one raised, one furrowed) + head tilt
    # Research note: |browDownL - browDownR| > 0.15 is a reliable confusion marker.
    brow_asym  = abs(dl - dr)
    brow_mixed = bu * min(dl, dr)
    head_tilt  = min(1.0, abs(roll) / 20.0)   # 20 deg -> full tilt score
    confused   = brow_asym*0.40 + brow_mixed*0.25 + jo*0.10 + head_tilt*0.25

    scores = {
        "Happy":     happy,
        "Sad":       sad,
        "Angry":     angry,
        "Surprised": surprised,
        "Disgusted": disgusted,
        "Sleepy":    sleepy,
        "Confused":  confused,
    }
    best, bscore = max(scores.items(), key=lambda x: x[1])
    emo = best if bscore > 0.10 else "Neutral"

    _emo_hist.append(emo)
    return Counter(_emo_hist).most_common(1)[0][0]


# ===========================================================================
# BOREDOM VARIANCE TRACKER
# ===========================================================================

class BoredomTracker:
    """
    Detects sustained boredom by monitoring facial expressiveness VARIANCE
    over a rolling time window.

    Research Methodology (Novel Contribution):
      Boredom in a classroom manifests as a prolonged absence of facial
      movement rather than a specific expression.  We operationalise this
      by defining an expressiveness index E(t) as the mean of 10 key
      blendshape scores, sampling at 2 Hz, and computing Var(E) over a
      60-second sliding window.

      Decision rule:
        Bored  iff  Var(E) < var_thr  AND  mean(E) < expr_thr

      Blendshapes sampled (expressiveness proxy):
        mouthSmileLeft, mouthSmileRight, mouthFrownLeft, mouthFrownRight,
        browInnerUp, browDownLeft, browDownRight, jawOpen,
        eyeWideLeft, eyeWideRight

      Thresholds (tunable):
        var_thr  = 0.0015  (variance; low => face is static)
        expr_thr = 0.12    (mean; low  => face is flat)

    Returns a boredom score in [0, 1].  Values > 0.5 are labelled "Bored".
    Requires at least 10 samples before returning non-zero scores.
    """

    _KEYS = [
        "mouthSmileLeft", "mouthSmileRight",
        "mouthFrownLeft", "mouthFrownRight",
        "browInnerUp",
        "browDownLeft",   "browDownRight",
        "jawOpen",
        "eyeWideLeft",    "eyeWideRight",
    ]

    def __init__(self,
                 window_sec: int   = 60,
                 sample_hz:  float = 2.0,
                 var_thr:    float = 0.0015,
                 expr_thr:   float = 0.12):
        self._buf       = deque(maxlen=int(window_sec * sample_hz))
        self._interval  = 1.0 / sample_hz
        self._last_t    = 0.0
        self.var_thr    = var_thr
        self.expr_thr   = expr_thr

    def _expr(self, bs: list) -> float:
        return sum(_b(bs, k) for k in self._KEYS) / len(self._KEYS)

    def update(self, bs: list) -> float:
        """Sample blendshapes and return boredom score [0, 1]."""
        now = time.time()
        if bs and (now - self._last_t) >= self._interval:
            self._buf.append(self._expr(bs))
            self._last_t = now

        if len(self._buf) < 10:
            return 0.0

        arr  = np.array(self._buf)
        var  = float(np.var(arr))
        mean = float(np.mean(arr))

        var_s  = max(0.0, min(1.0, (self.var_thr  - var)  / (self.var_thr  + 1e-9)))
        expr_s = max(0.0, min(1.0, (self.expr_thr - mean) / (self.expr_thr + 1e-9)))
        return round(var_s * expr_s, 3)

    @property
    def ready(self) -> bool:
        return len(self._buf) >= 10


# ===========================================================================
# DATA LOGGER + SESSION REPORT
# ===========================================================================

class DataLogger:
    """
    Accumulates per-frame research metrics and exports to CSV / JSON report.

    Session Report fields (for Research Paper Results section):
      - session_duration_sec
      - mean / max / min / std of focus score
      - % time Focused vs Distracted
      - drowsiness_alerts  (consecutive EAR < threshold for >= 3 samples)
      - gaze_distribution  (% time per gaze direction)
      - peak emotion and full emotion distribution
      - boredom statistics (mean, max)
    """

    FIELDS = ["timestamp", "ear", "yaw", "pitch", "roll",
              "gaze", "focus_score", "status", "emotion", "boredom"]

    def __init__(self):
        self.records: list[dict] = []

    def log(self, ear, yaw, pitch, roll, gaze,
            focus_score, status, emotion, boredom) -> None:
        self.records.append({
            "timestamp"  : datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "ear"        : round(ear, 4),
            "yaw"        : round(yaw, 2),
            "pitch"      : round(pitch, 2),
            "roll"       : round(roll, 2),
            "gaze"       : gaze,
            "focus_score": round(focus_score, 2),
            "status"     : status,
            "emotion"    : emotion,
            "boredom"    : round(boredom, 3),
        })

    def export_csv(self, path: str | None = None) -> str:
        if path is None:
            os.makedirs("logs", exist_ok=True)
            path = os.path.join("logs",
                                f"session_{datetime.now():%Y%m%d_%H%M%S}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.FIELDS)
            w.writeheader(); w.writerows(self.records)
        return path

    def generate_report(self) -> dict:
        """
        Build session summary dictionary for the Results section.

        Drowsiness alerts: counted as the number of RUNS of >= 3 consecutive
        records where EAR < EAR_THRESHOLD (avoids counting individual blinks).
        """
        if not self.records:
            return {"error": "No data recorded yet."}

        n     = len(self.records)
        fs    = [r["focus_score"] for r in self.records]
        emos  = [r["emotion"]     for r in self.records]
        gazes = [r["gaze"]        for r in self.records]
        ears  = [r["ear"]         for r in self.records]
        bored = [r["boredom"]     for r in self.records]
        stats = [r["status"]      for r in self.records]

        # Count drowsiness alert events (run of >=3 below threshold)
        drowsy_alerts, run = 0, 0
        for e in ears:
            if e < EAR_THRESHOLD:
                run += 1
                if run == 3:
                    drowsy_alerts += 1
            else:
                run = 0

        emo_cnt  = Counter(emos)
        gaze_cnt = Counter(gazes)

        return {
            "generated_at"         : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "session_duration_sec" : round(n * 0.5),
            "total_records"        : n,
            "focus": {
                "mean"          : round(sum(fs) / n, 2),
                "max"           : round(max(fs), 2),
                "min"           : round(min(fs), 2),
                "std"           : round(float(np.std(fs)), 2),
                "pct_focused"   : round(stats.count("Focused")    / n * 100, 1),
                "pct_distracted": round(stats.count("Distracted") / n * 100, 1),
            },
            "drowsiness_alerts"    : drowsy_alerts,
            "gaze_distribution"    : {k: round(v/n*100, 1)
                                       for k, v in gaze_cnt.items()},
            "emotion": {
                "peak"        : emo_cnt.most_common(1)[0][0],
                "distribution": {k: round(v/n*100, 1)
                                  for k, v in emo_cnt.items()},
            },
            "boredom": {
                "mean": round(sum(bored) / n, 3),
                "max" : round(max(bored), 3),
            },
        }

    def export_report_json(self, path: str | None = None) -> str:
        if path is None:
            os.makedirs("logs", exist_ok=True)
            path = os.path.join("logs",
                                f"report_{datetime.now():%Y%m%d_%H%M%S}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.generate_report(), f, indent=2)
        return path

    def clear(self) -> None:
        self.records.clear()


# ===========================================================================
# VIDEO CAMERA  --  Main Vision Engine
# ===========================================================================

class VideoCamera:
    """
    Central orchestrator:
      capture  ->  MediaPipe  ->  EAR  ->  head pose  ->  gaze
      ->  emotion  ->  boredom  ->  focus score  ->  log  ->  JPEG
    """

    def __init__(self, camera_index: int = 0):
        # -- Webcam (CAP_DSHOW required on this machine; MSMF fails) ---------
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(camera_index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS,          30)
        for _ in range(30):           # warm-up: read until valid frame
            ok, _ = self.cap.read()
            if ok: break

        # -- MediaPipe FaceLandmarker (Tasks API) ----------------------------
        opts = FaceLandmarkerOptions(
            base_options = BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode = RunningMode.IMAGE,
            num_faces    = 1,
            min_face_detection_confidence = 0.6,
            min_face_presence_confidence  = 0.6,
            min_tracking_confidence       = 0.5,
            output_face_blendshapes       = True,   # required for emotion
            output_facial_transformation_matrixes = False,
        )
        self.landmarker = FaceLandmarker.create_from_options(opts)

        # -- Subsystems ------------------------------------------------------
        self.logger         = DataLogger()
        self._boredom       = BoredomTracker()
        self._ear_ctr       = 0
        self._focus_hist    = deque(maxlen=30)
        self._prev_t        = time.time()
        self._log_ctr       = 0
        self._LOG_EVERY     = 15      # frames between log writes (~2 Hz @ 30fps)
        self._distraction_counter = 0

        self.metrics: dict = {
            "status": "No Face", "emotion": "N/A",
            "focus_score": 0.0,
            "ear": 0.0, "yaw": 0.0, "pitch": 0.0, "roll": 0.0,
            "gaze": "N/A", "gaze_h": 0.5, "gaze_v": 0.5,
            "gaze_state": "N/A",
            "boredom": 0.0, "fps": 0.0,
        }

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _lm_arr(self, lms, w, h) -> np.ndarray:
        return np.array([[l.x*w, l.y*h] for l in lms], dtype=np.float64)

    def _focus_score(self, ear, yaw, pitch, is_distracted: bool, is_iris_centered: bool) -> tuple[float, str]:
        """
        Heuristic Focus Score (0-100).

        Penalty hierarchy  (highest -> lowest priority):
          1. Gaze off-center  -- definitive attention indicator  (-25 pts)
          2. EAR below threshold -- eyes closing/drowsy          (up to -40)
          3. Head yaw excess  -- reduced weight; gaze is primary (up to -20)
          4. Head pitch excess                                   (up to -10)
        """
        score = 100.0

        # 1. EAR
        if ear < EAR_THRESHOLD:
            self._ear_ctr += 1
            score -= min(40.0, self._ear_ctr * EAR_PEN_PER_FRAME)
        else:
            self._ear_ctr = max(0, self._ear_ctr - 2)

        # 2. Head pose (reduced because gaze is now the primary signal)
        eff_yaw_threshold = 50.0 if is_iris_centered else YAW_THRESHOLD
        eff_pitch_threshold = 30.0 if is_iris_centered else PITCH_THRESHOLD

        score -= min(20.0, max(0.0, abs(yaw)   - eff_yaw_threshold)   * HEAD_YAW_SCALE)
        score -= min(10.0, max(0.0, abs(pitch) - eff_pitch_threshold) * HEAD_PITCH_SCALE)

        # 3. Gaze -- strongest single distraction signal
        if is_distracted:
            score -= 25.0

        score = max(0.0, min(100.0, score))
        self._focus_hist.append(score)
        smoothed = float(np.mean(self._focus_hist))
        return round(smoothed, 1), "Focused" if smoothed >= 60 else "Distracted"

    def _draw_overlay(self, frame, lms, gaze_dir, gaze_state="N/A") -> None:
        """Landmark dots + coloured iris circles + Mobile Usage Alert."""
        for x, y in lms:
            cv2.circle(frame, (int(x), int(y)), 1, (0, 200, 150), -1)
        if len(lms) >= 478:
            col = (0, 255, 100) if gaze_dir == "Center" else (0, 80, 255)
            for idx in (LEFT_IRIS, RIGHT_IRIS):
                cx, cy = int(lms[idx, 0]), int(lms[idx, 1])
                cv2.circle(frame, (cx, cy), 5, col, 2)
                
        # Phone Detection Floating Label
        if gaze_state == "PHONE SUSPECTED":
            cv2.rectangle(frame, (40, 25), (460, 60), (0, 0, 0), -1)
            cv2.putText(frame, "⚠️ SUSPECTED MOBILE USAGE",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.7, (0, 255, 255), 2)

    def _draw_hud(self, frame, m) -> None:
        h, w = frame.shape[:2]
        ov   = frame.copy()
        cv2.rectangle(ov, (0, h-70), (w, h), (0,0,0), -1)
        cv2.addWeighted(ov, 0.5, frame, 0.5, 0, frame)
        col = (0, 220, 0) if m["status"] == "Focused" else (0, 60, 220)
        cv2.putText(frame,
            f"Status:{m['status']}  Score:{m['focus_score']:.1f}%"
            f"  Gaze:{m['gaze']}  FPS:{m['fps']:.1f}",
            (10, h-45), cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 2)
        cv2.putText(frame,
            f"EAR:{m['ear']:.3f}  Yaw:{m['yaw']:.1f}  "
            f"Pitch:{m['pitch']:.1f}  Emotion:{m['emotion']}",
            (10, h-18), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (200,200,200), 1)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def get_frame(self) -> bytes | None:
        try:
            return self._inner()
        except Exception as e:
            print(f"[camera] {e}")
            return None

    def _inner(self) -> bytes | None:
        ok, frame = self.cap.read()
        if not ok:
            return None

        frame = cv2.flip(frame, 1)
        fh, fw = frame.shape[:2]

        now = time.time()
        fps = 1.0 / max(now - self._prev_t, 1e-9)
        self._prev_t = now

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_img)

        if result.face_landmarks:
            lms  = self._lm_arr(result.face_landmarks[0], fw, fh)

            # -- Metrics --
            ear  = (compute_ear(lms, LEFT_EYE) + compute_ear(lms, RIGHT_EYE)) / 2.0
            pose = compute_head_pose(lms[POSE_INDICES].astype(np.float64), fw, fh)
            gaze = compute_gaze(lms)
            
            # Gaze Buffer Logic
            h_ratio = gaze["h_ratio"]
            is_iris_centered = (0.35 <= h_ratio <= 0.65)
            
            if not is_iris_centered:
                self._distraction_counter += 1
            else:
                self._distraction_counter = 0
                
            fps_val = max(1, int(fps))
            is_distracted = self._distraction_counter > (fps_val * 1.2)
                
            if is_distracted and gaze["v_ratio"] >= 0.80:
                gaze_state = "PHONE SUSPECTED"
            elif is_distracted:
                gaze_state = "Looking Away"
            else:
                gaze_state = "Centered"

            focus_score, status = self._focus_score(
                ear, pose["yaw"], pose["pitch"], is_distracted, is_iris_centered)

            bs      = result.face_blendshapes[0] if result.face_blendshapes else []
            boredom = self._boredom.update(bs)
            emotion = ("Bored"
                       if boredom > 0.5 and self._boredom.ready
                       else detect_emotion(bs, ear=ear, roll=pose["roll"]))

            self.metrics.update({
                "status": status, "emotion": emotion,
                "focus_score": focus_score,
                "ear": round(ear, 4),
                "yaw": round(pose["yaw"], 2),
                "pitch": round(pose["pitch"], 2),
                "roll": round(pose["roll"], 2),
                "gaze": gaze["direction"],
                "gaze_h": gaze["h_ratio"],
                "gaze_v": gaze["v_ratio"],
                "gaze_state": gaze_state,
                "boredom": boredom,
                "fps": round(fps, 1),
            })

            self._draw_overlay(frame, lms, gaze["direction"], gaze_state)

            self._log_ctr += 1
            if self._log_ctr >= self._LOG_EVERY:
                self.logger.log(ear, pose["yaw"], pose["pitch"], pose["roll"],
                                gaze["direction"], focus_score, status,
                                emotion, boredom)
                self._log_ctr = 0

        else:
            self.metrics.update({
                "status": "No Face", "emotion": "N/A",
                "focus_score": 0.0, "gaze": "N/A", "gaze_state": "N/A", "fps": round(fps, 1),
            })
            cv2.putText(frame, "No Face Detected",
                        (fw//2-120, fh//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 60, 220), 2)

        self._draw_hud(frame, self.metrics)
        _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return jpg.tobytes()

    def get_metrics(self) -> dict:
        return dict(self.metrics)

    def export_session(self) -> str:
        return self.logger.export_csv()

    def get_session_report(self) -> dict:
        return self.logger.generate_report()

    def export_report(self) -> str:
        return self.logger.export_report_json()

    def release(self) -> None:
        self.cap.release()
        self.landmarker.close()
