import csv
import cv2
import json
import os
import threading
import time
from collections import Counter, deque
from datetime import datetime

try:
    from ultralytics import YOLO as _YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False
    print("[phone] ultralytics not installed – phone detection disabled.")
    print("[phone] Run: python -m pip install ultralytics")

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python        import BaseOptions
from mediapipe.tasks.python.vision import (FaceLandmarker,
                                            FaceLandmarkerOptions,
                                            RunningMode)

# ---------------------------------------------------------------------------
# Model path & profile directory
# ---------------------------------------------------------------------------
_MODEL_PATH   = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
_PROFILES_DIR = os.path.join(os.path.dirname(__file__), "static", "profiles")

# ---------------------------------------------------------------------------
# Phone detection constants  (YOLOv8-nano, COCO class 67 = cell phone)
# ---------------------------------------------------------------------------
_PHONE_CLASS_ID         = 67      # COCO label index for "cell phone"
_PHONE_CONF             = 0.40    # YOLO confidence threshold
_PHONE_DETECT_EVERY     = 10      # run YOLO every N frames (keeps FPS smooth)
_PHONE_GAZE_DOWN_V      = 0.68    # gaze_v above this = looking DOWN at a device
_PHONE_SPATIAL_MARGIN   = 0.30    # horizontal margin (fraction of face-box width)
                                  # for associating a phone box to a face
_PHONE_FOCUS_PENALTY    = 30.0    # extra focus-score penalty per frame, phone-distracted

# LBPH confidence threshold: lower score = better match; 0.0 = perfect.
# Measured live-webcam confidence ranges (from noise simulation):
#   Correct match under normal noise:  0 – 139  ← accept these
#   Wrong cross-person match:          157+      ← reject these
# 140 sits cleanly between the two populations.
_LBPH_THRESHOLD = 140.0

# Run LBPH recognition only every N frames to keep the video smooth.
# In between, the previously cached name/PRN is returned instantly.
_RECOG_EVERY_N = 8

# Haar cascade shipped with every opencv build – used only for profile loading.
_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_face_cascade  = cv2.CascadeClassifier(_CASCADE_PATH)

# CLAHE instance shared by loader and recognizer (same params = consistent preprocessing)
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def _augment_face(crop: np.ndarray) -> list:
    """Return a list of augmented variants of a 200×200 grayscale face crop.

    Augmentations: original + horizontal flip + 3 brightness shifts +
    2 small rotations.  This gives LBPH 7× more training samples per photo,
    dramatically reducing label confusion when only 1 reference image exists.
    """
    h, w   = crop.shape[:2]
    cx, cy = w // 2, h // 2
    out    = [crop]                          # original

    # Horizontal flip
    out.append(cv2.flip(crop, 1))

    # Brightness shifts (±15, ±30 pixel values)
    for delta in (+20, -20, +35):
        shifted = np.clip(crop.astype(np.int16) + delta, 0, 255).astype(np.uint8)
        out.append(shifted)

    # Small rotations (±8 degrees)
    for angle in (+8, -8):
        M   = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        rot = cv2.warpAffine(crop, M, (w, h),
                             borderMode=cv2.BORDER_REFLECT)
        out.append(rot)

    return out


def _load_known_faces() -> tuple[list[str], list[str], "cv2.face.LBPHFaceRecognizer", bool]:
    """Scan *_PROFILES_DIR* for Name_PRN images, train an LBPH recognizer.

    File naming convention:  ``vivek_1251130166.jpg``
    Supports .jpg / .jpeg / .png

    Pipeline per reference image:
      BGR → Gray → Haar face-detect → 200×200 crop → CLAHE equalize
      → 7× augmented variants → all fed into LBPH training.

    Returns:
        names      -- list of display names (title-cased)
        prns       -- list of PRN strings (parallel to names)
        recognizer -- trained cv2.face.LBPHFaceRecognizer (or untrained)
        trained    -- True if at least one face was enrolled
    """
    names:  list[str] = []
    prns:   list[str] = []
    faces:  list      = []    # grayscale 200×200 crops (augmented)
    labels: list[int] = []    # label per crop

    os.makedirs(_PROFILES_DIR, exist_ok=True)
    supported = {".jpg", ".jpeg", ".png"}

    for fname in sorted(os.listdir(_PROFILES_DIR)):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in supported:
            continue

        stem  = os.path.splitext(fname)[0]         # e.g. "vivek_1251130166"
        parts = stem.rsplit("_", 1)                # split on LAST underscore only
        if len(parts) != 2:
            print(f"[enroll] Skipping '{fname}' – expected Name_PRN format.")
            continue
        name, prn = parts[0].strip(), parts[1].strip()

        img_path = os.path.join(_PROFILES_DIR, fname)
        bgr      = cv2.imread(img_path)
        if bgr is None:
            print(f"[enroll] Cannot read '{fname}' – skipping.")
            continue

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # ── Detect the primary face in the reference photo ────────────────
        detected = _face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))

        if len(detected) == 0:
            print(f"[enroll] No Haar face in '{fname}'; using full-image crop.")
            face_crop = cv2.resize(gray, (200, 200))
        else:
            # Sort by area descending → take the largest face in the photo
            detected = sorted(detected, key=lambda r: r[2]*r[3], reverse=True)
            x, y, w, h = detected[0]
            face_crop  = cv2.resize(gray[y:y+h, x:x+w], (200, 200))

        # ── CLAHE equalise (same transform applied at predict time) ───────
        face_crop = _CLAHE.apply(face_crop)

        # ── Augment → 7 training samples per reference photo ─────────────
        label    = len(names)            # unique integer for this student
        variants = _augment_face(face_crop)
        faces.extend(variants)
        labels.extend([label] * len(variants))

        names.append(name.title())       # "vivek" → "Vivek"
        prns.append(prn)
        print(f"[enroll] {name.title()} (PRN {prn}) – label {label}, "
              f"{len(variants)} training samples")

    # ── Train LBPH ────────────────────────────────────────────────────
    # radius=2, neighbors=16 give a richer histogram than the default (1,8)
    # and are less sensitive to small lighting / pose differences.
    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=2, neighbors=16, grid_x=8, grid_y=8)

    trained = len(faces) > 0
    if trained:
        recognizer.train(faces, np.array(labels, dtype=np.int32))
        print(f"[enroll] LBPH trained: {len(names)} student(s), "
              f"{len(faces)} total samples.")
    else:
        print("[enroll] WARNING: No profiles found – recognition disabled.")

    return names, prns, recognizer, trained


# ---------------------------------------------------------------------------
# Phone detection helper functions
# ---------------------------------------------------------------------------

def _init_phone_detector():
    """Load YOLOv8-nano for phone detection.

    The nano model (~6 MB) is auto-downloaded on first use to the
    ultralytics cache directory.  Returns None if ultralytics is not
    installed so the rest of the pipeline continues to work.
    """
    if not _YOLO_AVAILABLE:
        return None
    try:
        model = _YOLO("yolov8n.pt")
        # Warm up: one dummy pass so the first real frame is not slow
        import numpy as _np
        model.predict(
            source=_np.zeros((64, 64, 3), dtype=_np.uint8),
            classes=[_PHONE_CLASS_ID],
            conf=_PHONE_CONF,
            verbose=False,
        )
        print("[phone] YOLOv8-nano loaded and warmed up.")
        return model
    except Exception as exc:
        print(f"[phone] YOLO init failed: {exc} – phone detection disabled.")
        return None


def _detect_phones(
    model,
    infer_frame,
    fw: int, fh: int,
    infer_w: int, infer_h: int,
) -> list[tuple[int, int, int, int]]:
    """Run YOLO on *infer_frame*, return phone boxes in full-res coords.

    Returns a list of (x1, y1, x2, y2) tuples in the coordinate space of
    the original high-res render frame.
    """
    results = model.predict(
        source=infer_frame,
        classes=[_PHONE_CLASS_ID],
        conf=_PHONE_CONF,
        verbose=False,
    )
    boxes = []
    scale_x = fw / infer_w
    scale_y = fh / infer_h
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            boxes.append((
                int(x1 * scale_x),
                int(y1 * scale_y),
                int(x2 * scale_x),
                int(y2 * scale_y),
            ))
    return boxes


def _correlate_phone_face(
    phone_boxes: list[tuple[int, int, int, int]],
    face_box: tuple[int, int, int, int],
    gaze_v: float,
    frame_h: int,
) -> bool:
    """Return True when BOTH spatial AND gaze conditions are met:

    Spatial — a detected phone box must overlap the student's body zone
    (the region directly below their face, extending to the bottom of the
    frame, widened by _PHONE_SPATIAL_MARGIN × face width).

    Gaze    — the student's vertical gaze ratio must exceed
    _PHONE_GAZE_DOWN_V (i.e., they are looking downward, toward their lap
    or desk where a phone would typically be held).

    Requiring both conditions eliminates false positives such as a phone
    lying on a desk that the student is not actively looking at.
    """
    if not phone_boxes:
        return False

    # Gaze gate — must be looking down first
    if gaze_v < _PHONE_GAZE_DOWN_V:
        return False

    fx1, fy1, fx2, fy2 = face_box
    face_w  = max(1, fx2 - fx1)
    margin  = int(face_w * _PHONE_SPATIAL_MARGIN)

    # Body zone: region below the face extended to frame bottom
    bz_x1 = fx1 - margin
    bz_y1 = fy2              # starts right below the chin
    bz_x2 = fx2 + margin
    bz_y2 = frame_h

    for (px1, py1, px2, py2) in phone_boxes:
        # Intersection-over-phone: phone must overlap the body zone
        ix1 = max(bz_x1, px1)
        iy1 = max(bz_y1, py1)
        ix2 = min(bz_x2, px2)
        iy2 = min(bz_y2, py2)
        if ix2 > ix1 and iy2 > iy1:
            return True   # spatial + gaze both confirmed

    return False


# ---------------------------------------------------------------------------
# Landmark index groups
# ---------------------------------------------------------------------------
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

POSE_INDICES = [
    1,    # Nose tip
    152,  # Chin
    263,  # Left eye corner
    33,   # Right eye corner
    287,  # Left mouth corner
    57,   # Right mouth corner
]

MODEL_3D = np.array([
    [ 0.0,    0.0,    0.0],
    [ 0.0,  -63.6,  -12.5],
    [-43.3,  32.7,  -26.0],
    [ 43.3,  32.7,  -26.0],
    [-28.9, -28.9,  -24.1],
    [ 28.9, -28.9,  -24.1],
], dtype=np.float64)

LEFT_IRIS   = 468
RIGHT_IRIS  = 473

LEFT_EYE_INNER  = 362;  LEFT_EYE_OUTER  = 263
LEFT_EYE_TOP    = 386;  LEFT_EYE_BOTTOM = 374
RIGHT_EYE_INNER = 133;  RIGHT_EYE_OUTER = 33
RIGHT_EYE_TOP   = 159;  RIGHT_EYE_BOTTOM= 145

# ---------------------------------------------------------------------------
# Detection thresholds
# ---------------------------------------------------------------------------
EAR_THRESHOLD   = 0.21
YAW_THRESHOLD   = 25.0
PITCH_THRESHOLD = 20.0

GAZE_H_LOW  = 0.38
GAZE_H_HIGH = 0.62
GAZE_V_LOW  = 0.30
GAZE_V_HIGH = 0.75

# Focus score penalty weights
GAZE_AWAY_PENALTY = 28.0
HEAD_YAW_SCALE    = 0.6
HEAD_PITCH_SCALE  = 0.4
EAR_PEN_PER_FRAME = 2.0


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------
def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


def compute_ear(lms: np.ndarray, idx: list) -> float:
    p = lms[idx]
    A = _dist(p[1], p[5])
    B = _dist(p[2], p[4])
    C = _dist(p[0], p[3])
    return float((A + B) / (2.0 * C + 1e-9))


def compute_head_pose(pts2d: np.ndarray, fw: int, fh: int) -> dict:
    focal = float(fw)
    cam   = np.array([[focal, 0, fw / 2.0],
                      [0, focal, fh / 2.0],
                      [0,     0,      1.0]], dtype=np.float64)
    dist  = np.zeros((4, 1), dtype=np.float64)

    ok, rvec, tvec = cv2.solvePnP(MODEL_3D, pts2d, cam, dist,
                                   flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}

    rmat, _ = cv2.Rodrigues(rvec)
    proj    = cv2.hconcat([rmat, tvec])
    _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(proj)
    euler   = euler.flatten()
    return {"yaw": float(euler[1]), "pitch": float(euler[0]), "roll": float(euler[2])}


def compute_gaze(lms: np.ndarray) -> dict:
    if len(lms) < 478:
        return {"direction": "N/A", "h_ratio": 0.5, "v_ratio": 0.5}

    def _ratio(iris_idx, inner, outer, top, bot):
        ix, iy = lms[iris_idx]
        x_min  = min(lms[inner, 0], lms[outer, 0])
        x_max  = max(lms[inner, 0], lms[outer, 0])
        y_top  = lms[top, 1]
        y_bot  = lms[bot, 1]
        h = (ix - x_min) / (x_max - x_min + 1e-9)
        v = (iy - y_top) / (abs(y_bot - y_top) + 1e-9)
        return h, v

    lh, lv = _ratio(LEFT_IRIS,  LEFT_EYE_INNER,  LEFT_EYE_OUTER,  LEFT_EYE_TOP,  LEFT_EYE_BOTTOM)
    rh, rv = _ratio(RIGHT_IRIS, RIGHT_EYE_INNER, RIGHT_EYE_OUTER, RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM)

    avg_h = (lh + rh) / 2.0
    avg_v = (lv + rv) / 2.0

    if   avg_h < GAZE_H_LOW:  direction = "Left"
    elif avg_h > GAZE_H_HIGH: direction = "Right"
    elif avg_v < GAZE_V_LOW:  direction = "Up"
    elif avg_v > GAZE_V_HIGH: direction = "Down"
    else:                      direction = "Center"

    return {
        "direction": direction,
        "h_ratio":   round(float(avg_h), 3),
        "v_ratio":   round(float(avg_v), 3),
    }


# ---------------------------------------------------------------------------
# Emotion Detection
# ---------------------------------------------------------------------------
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
    if not bs:
        return "Neutral"
    _bidx(bs)

    sl = _b(bs, "mouthSmileLeft");  sr  = _b(bs, "mouthSmileRight")
    ql = _b(bs, "cheekSquintLeft"); qr  = _b(bs, "cheekSquintRight")
    fl = _b(bs, "mouthFrownLeft");  fr  = _b(bs, "mouthFrownRight")
    bu = _b(bs, "browInnerUp");     md  = _b(bs, "mouthLowerDownLeft")
    dl = _b(bs, "browDownLeft");    dr  = _b(bs, "browDownRight")
    nl = _b(bs, "noseSneerLeft");   nr  = _b(bs, "noseSneerRight")
    jo = _b(bs, "jawOpen")
    wl = _b(bs, "eyeWideLeft");     wr  = _b(bs, "eyeWideRight")
    ul = _b(bs, "mouthUpperUpLeft");ur  = _b(bs, "mouthUpperUpRight")
    bl = _b(bs, "eyeBlinkLeft");    br_ = _b(bs, "eyeBlinkRight")

    happy     = sl*0.38 + sr*0.38 + ql*0.12 + qr*0.12
    sad       = fl*0.38 + fr*0.38 + bu*0.14 + md*0.10
    angry     = dl*0.38 + dr*0.38 + nl*0.12 + nr*0.12
    surprised = jo*0.35 + wl*0.22 + wr*0.22 + bu*0.21
    disgusted = nl*0.32 + nr*0.32 + ul*0.18 + ur*0.18

    ear_sig = max(0.0, min(1.0, (0.25 - ear) / 0.15))
    sleepy  = bl*0.30 + br_*0.30 + ear_sig*0.40

    brow_asym  = abs(dl - dr)
    brow_mixed = bu * min(dl, dr)
    head_tilt  = min(1.0, abs(roll) / 20.0)
    confused   = brow_asym*0.40 + brow_mixed*0.25 + jo*0.10 + head_tilt*0.25

    # Focused: calm brows, mouth closed, stable EAR (not sleepy), no strong expression
    calm_brow  = max(0.0, 1.0 - (dl + dr + bu) / 0.6)
    calm_mouth = max(0.0, 1.0 - (jo + sl + sr + fl + fr) / 0.8)
    calm_eyes  = max(0.0, 1.0 - ear_sig)          # high EAR = eyes open = not sleepy
    focused    = calm_brow*0.40 + calm_mouth*0.30 + calm_eyes*0.30

    scores = {
        "Focused":   focused,
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


# ---------------------------------------------------------------------------
# Boredom Tracker
# ---------------------------------------------------------------------------
class BoredomTracker:
    _KEYS = [
        "mouthSmileLeft", "mouthSmileRight",
        "mouthFrownLeft", "mouthFrownRight",
        "browInnerUp",
        "browDownLeft",   "browDownRight",
        "jawOpen",
        "eyeWideLeft",    "eyeWideRight",
    ]

    def __init__(self, window_sec: int = 60, sample_hz: float = 2.0,
                 var_thr: float = 0.0015, expr_thr: float = 0.12):
        self._buf      = deque(maxlen=int(window_sec * sample_hz))
        self._interval = 1.0 / sample_hz
        self._last_t   = 0.0
        self.var_thr   = var_thr
        self.expr_thr  = expr_thr

    def _expr(self, bs: list) -> float:
        return sum(_b(bs, k) for k in self._KEYS) / len(self._KEYS)

    def update(self, bs: list) -> float:
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


# ---------------------------------------------------------------------------
# Data Logger
# ---------------------------------------------------------------------------

# Columns written to the master attendance CSV on disk.
_ATTENDANCE_FIELDS = ["name", "prn", "marked_at"]

# Permanent accumulating file – one row per student per session, never
# overwritten.  All DataLogger instances share the same path and lock so
# that concurrent flask threads can never corrupt the file.
_MASTER_ATTENDANCE_PATH = os.path.join("logs", "master_attendance.csv")
_MASTER_LOCK = threading.Lock()   # process-wide lock for the master CSV


def _ensure_master_csv() -> None:
    """Create logs/ directory and write the CSV header if the file is new."""
    os.makedirs("logs", exist_ok=True)
    if not os.path.exists(_MASTER_ATTENDANCE_PATH):
        with open(_MASTER_ATTENDANCE_PATH, "w", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=_ATTENDANCE_FIELDS).writeheader()
        print(f"[attendance] Master CSV created: {_MASTER_ATTENDANCE_PATH}")


class DataLogger:
    FIELDS = ["timestamp", "ear", "yaw", "pitch", "roll",
              "gaze", "focus_score", "status", "emotion", "boredom"]

    def __init__(self):
        self.records: list[dict] = []
        # Attendance roster: keyed by PRN, value is the full attendance entry.
        # Using a dict (not a set) so duplicate marks are silently ignored.
        self._attendance: dict[str, dict] = {}

        # Guarantee the master CSV exists with a header before any writes.
        _ensure_master_csv()

    def mark_present(self, name: str, prn: str) -> None:
        """Record a student as present and instantly persist to disk.

        The in-memory roster is updated first.  If this PRN has not been seen
        before, the entry is also appended to the master attendance CSV under
        a process-wide lock so concurrent face-recognition threads cannot
        interleave writes and corrupt the file.

        Safe to call repeatedly – only the *first* confirmation is stored
        in memory or on disk.
        """
        if prn not in self._attendance:
            entry = {
                "name":      name,
                "prn":       prn,
                "marked_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            # ── 1. Update in-memory roster ────────────────────────────────
            self._attendance[prn] = entry

            # ── 2. Instantly append to disk (thread-safe) ─────────────────
            with _MASTER_LOCK:
                with open(_MASTER_ATTENDANCE_PATH, "a",
                          newline="", encoding="utf-8") as fh:
                    csv.DictWriter(fh, fieldnames=_ATTENDANCE_FIELDS).writerow(entry)
            print(f"[attendance] Persisted to disk: {name} (PRN {prn}) "
                  f"@ {entry['marked_at']}")

    @property
    def attendance_roster(self) -> list[dict]:
        """Return the current attendance list sorted by name."""
        return sorted(self._attendance.values(), key=lambda r: r["name"])

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
            path = os.path.join("logs", f"session_{datetime.now():%Y%m%d_%H%M%S}.csv")
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=self.FIELDS)
            w.writeheader(); w.writerows(self.records)
        return path

    def generate_report(self) -> dict:
        if not self.records:
            return {"error": "No data recorded yet."}

        n     = len(self.records)
        fs    = [r["focus_score"] for r in self.records]
        emos  = [r["emotion"]     for r in self.records]
        gazes = [r["gaze"]        for r in self.records]
        ears  = [r["ear"]         for r in self.records]
        bored = [r["boredom"]     for r in self.records]
        stats = [r["status"]      for r in self.records]

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
            "drowsiness_alerts" : drowsy_alerts,
            "gaze_distribution" : {k: round(v/n*100, 1) for k, v in gaze_cnt.items()},
            "emotion": {
                "peak"        : emo_cnt.most_common(1)[0][0],
                "distribution": {k: round(v/n*100, 1) for k, v in emo_cnt.items()},
            },
            "boredom": {
                "mean": round(sum(bored) / n, 3),
                "max" : round(max(bored), 3),
            },
            # ── Attendance Roster ─────────────────────────────────────────
            "attendance": {
                "total_present": len(self._attendance),
                "students":      self.attendance_roster,
            },
        }

    def export_report_json(self, path: str | None = None) -> str:
        if path is None:
            os.makedirs("logs", exist_ok=True)
            path = os.path.join("logs", f"report_{datetime.now():%Y%m%d_%H%M%S}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.generate_report(), f, indent=2)
        return path

    def clear(self) -> None:
        """Reset the in-memory session state.

        NOTE: The on-disk master_attendance.csv is intentionally NOT cleared
        here.  It is a permanent, accumulating record across all sessions and
        should only be managed manually by the administrator.
        """
        self.records.clear()
        self._attendance.clear()


# ===========================================================================
# VIDEO CAMERA  --  Main Vision Engine
# ===========================================================================

def list_available_cameras(max_test: int = 5) -> list[int]:
    """Scan camera indices 0..max_test-1 and return those that open."""
    available = []
    for idx in range(max_test):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            ok, _ = cap.read()
            if ok:
                available.append(idx)
        cap.release()
    return available


class VideoCamera:

    # Capture resolution fed to OpenCV (must match what the USB cam supports well)
    _CAP_W = 1280
    _CAP_H = 720

    # Inference downscale target – MediaPipe sees this smaller frame only
    _INFER_W = 640
    _INFER_H = 480

    def __init__(self, camera_index: int = 1):
        """Open the external USB webcam at *camera_index* (default: 1).

        Capture at 1920×1080 for crisp visuals; MediaPipe inference runs on
        a 1280×720 downscaled copy for performance.
        """
        print(f"[camera] Trying to open camera index {camera_index} ...")

        # 1st attempt: DirectShow backend (best for Windows / USB webcams)
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            print(f"[camera] CAP_DSHOW failed, falling back to default backend ...")
            self.cap = cv2.VideoCapture(camera_index)

        # 2nd attempt: auto-scan if the requested index still does not work
        if not self.cap.isOpened():
            print(f"[camera] Index {camera_index} not available. Scanning for any webcam ...")
            available = list_available_cameras()
            if available:
                fallback = available[0]
                print(f"[camera] Found cameras at indices: {available}. Using index {fallback}.")
                self.cap = cv2.VideoCapture(fallback, cv2.CAP_DSHOW)
                if not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(fallback)
            else:
                raise RuntimeError(
                    "[camera] No webcam detected. "
                    "Please connect your webcam and restart the application."
                )

        # ── Stable 720p capture for the external USB webcam ──────────────
        # 1080p caused a hard 1 FPS bottleneck on this hardware;
        # 1280×720 is widely supported and keeps the feed smooth.
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self._CAP_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._CAP_H)
        self.cap.set(cv2.CAP_PROP_FPS,          30)
        # Keep the OS-level capture buffer at 1 frame so cap.read() always
        # returns the *latest* frame instead of one that is several frames old.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Warm-up: flush initial dark/blank frames
        opened = False
        for _ in range(30):
            ok, _ = self.cap.read()
            if ok:
                opened = True
                break

        if not opened:
            self.cap.release()
            raise RuntimeError(
                "[camera] Webcam opened but could not read a frame. "
                "Check that no other application is using the camera."
            )

        actual_w   = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h   = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
        print(f"[camera] Webcam ready: {actual_w}x{actual_h} @ {actual_fps:.1f} fps")
        print(f"[camera] Inference downscale target: {self._INFER_W}x{self._INFER_H}")

        # ── Face Recognition: train LBPH on enrolled profiles at startup ──
        (self._known_names,
         self._known_prns,
         self._recognizer,
         self._recog_ready) = _load_known_faces()

        # ── Phone Detector (YOLOv8-nano) ───────────────────────────────
        self._phone_model: object | None = _init_phone_detector()
        # Cached phone boxes from last YOLO run (full-res pixel coords):
        # each entry = (x1, y1, x2, y2)
        self._phone_boxes: list[tuple[int,int,int,int]] = []
        self._phone_frame_ctr: int = 0   # counts frames since last YOLO run

        # ── MediaPipe: allow up to 10 simultaneous faces (classroom mode) ──
        opts = FaceLandmarkerOptions(
            base_options = BaseOptions(model_asset_path=_MODEL_PATH),
            running_mode = RunningMode.IMAGE,
            num_faces    = 10,
            min_face_detection_confidence = 0.5,
            min_face_presence_confidence  = 0.5,
            min_tracking_confidence       = 0.4,
            output_face_blendshapes       = True,
            output_facial_transformation_matrixes = False,
        )
        self.landmarker = FaceLandmarker.create_from_options(opts)

        self.logger   = DataLogger()
        self._prev_t  = time.time()
        self._log_ctr = 0
        self._LOG_EVERY = 15

        # ── Per-face state dicts (keyed by face index 0..N-1) ──
        # Each entry holds an independent boredom tracker, EAR counter,
        # focus-score history, and last-known identity cache.
        self._face_state: dict[int, dict] = {}

        # ── Name-keyed attendance state (CAMERA level, not face_id level) ──
        # MediaPipe's face_id ordering is unstable between frames, so
        # tracking streaks per-name is the only reliable approach.
        # _name_streak[name] counts consecutive recog-frames that confirmed
        # that identity; _name_attended is a set of already-marked names.
        self._name_streak:   dict[str, int] = {}
        self._name_attended: set[str]       = set()

        # ── Producer / consumer decoupling ────────────────────────────────
        # A background daemon thread runs the full capture+inference pipeline
        # and stores the finished JPEG in _latest_jpg.  The Flask /video_feed
        # route reads _latest_jpg under a lock – never blocking on ML work.
        self._latest_jpg: bytes | None = None
        self._frame_lock  = threading.Lock()
        self._capture_thread = threading.Thread(
            target=self._capture_loop, daemon=True, name="CaptureThread")
        self._capture_thread.start()
        print("[camera] Background capture thread started.")

        # ── Class-level (aggregated) metrics exposed to the API ──
        self.metrics: dict = {
            # Classroom aggregates
            "face_count":    0,
            "status":        "No Face",
            "focus_score":   0.0,
            "ear":           0.0,
            "yaw":           0.0,
            "pitch":         0.0,
            "roll":          0.0,
            "gaze":          "N/A",
            "gaze_h":        0.5,
            "gaze_v":        0.5,
            "gaze_state":    "N/A",
            "boredom":       0.0,
            "emotion":       "N/A",
            "fps":           0.0,
            "phone_alerts":  0,    # how many students are phone-distracted
            "face_data":     [],
        }

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------
    def _lm_arr(self, lms, w, h) -> np.ndarray:
        return np.array([[l.x*w, l.y*h] for l in lms], dtype=np.float64)

    def _get_face_state(self, face_id: int) -> dict:
        """Return (creating if needed) the per-face mutable state dict."""
        if face_id not in self._face_state:
            self._face_state[face_id] = {
                "boredom":   BoredomTracker(),
                "ear_ctr":   0,
                "focus_hist": deque(maxlen=30),
                # Last known identity for this face slot (cached between recog frames)
                "name":      "Unknown",
                "prn":       "—",
                # Frame counter: recognition runs every _RECOG_EVERY_N frames
                "recog_ctr": 0,
            }
        return self._face_state[face_id]

    def _identify_face(self, face_id: int,
                       full_frame: np.ndarray,
                       lms: np.ndarray,
                       scale_x: float, scale_y: float) -> tuple[str, str]:
        """Crop the face, run LBPH every *_RECOG_EVERY_N* frames, and track
        attendance using camera-level name-keyed streak counters.

        Using name-keyed streaks (not face_id-keyed) is critical because
        MediaPipe re-orders face indices every frame, so face_id=0 in one
        frame may be a different person in the next.
        """
        state = self._get_face_state(face_id)

        if not self._recog_ready:
            return state["name"], state["prn"]

        # ── Frame-skip: run LBPH only every N frames per face slot ────────
        state["recog_ctr"] = (state["recog_ctr"] + 1) % _RECOG_EVERY_N
        if state["recog_ctr"] != 0:
            return state["name"], state["prn"]

        # ── Crop face ROI from the full-res frame ─────────────────────────
        fh, fw = full_frame.shape[:2]
        xs = np.clip((lms[:, 0] * scale_x).astype(int), 0, fw - 1)
        ys = np.clip((lms[:, 1] * scale_y).astype(int), 0, fh - 1)
        x1, y1 = int(xs.min()), int(ys.min())
        x2, y2 = int(xs.max()), int(ys.max())

        pad_x = max(20, int((x2 - x1) * 0.20))
        pad_y = max(20, int((y2 - y1) * 0.20))
        x1 = max(0, x1 - pad_x);  y1 = max(0, y1 - pad_y)
        x2 = min(fw, x2 + pad_x); y2 = min(fh, y2 + pad_y)

        crop_bgr = full_frame[y1:y2, x1:x2]
        if crop_bgr.size == 0 or (x2 - x1) < 50 or (y2 - y1) < 50:
            return state["name"], state["prn"]

        # ── CLAHE + LBPH predict ─────────────────────────────────────
        gray_crop = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        gray_crop = cv2.resize(gray_crop, (200, 200))
        gray_crop = _CLAHE.apply(gray_crop)

        label, confidence = self._recognizer.predict(gray_crop)

        if confidence <= _LBPH_THRESHOLD and 0 <= label < len(self._known_names):
            matched_name = self._known_names[label]
            matched_prn  = self._known_prns[label]
        else:
            matched_name = "Unknown"
            matched_prn  = "—"

        # Cache in the face slot so overlay shows the right label
        state["name"] = matched_name
        state["prn"]  = matched_prn

        # ── Name-keyed attendance streak (survives face_id reshuffling) ────
        if matched_name != "Unknown":
            self._name_streak[matched_name] = \
                self._name_streak.get(matched_name, 0) + 1

            # Mark present after 3 consecutive recognition-frames – fast but
            # reliable enough given CLAHE + augmentation pipeline.
            if (self._name_streak[matched_name] >= 3
                    and matched_name not in self._name_attended):
                self._name_attended.add(matched_name)
                self.logger.mark_present(matched_name, matched_prn)
                print(f"[attendance] Marked: {matched_name} "
                      f"(PRN {matched_prn}) conf={confidence:.1f}")

        return state["name"], state["prn"]

    def _focus_score(self, face_id: int,
                     ear: float, yaw: float, pitch: float,
                     is_distracted: bool,
                     phone_distracted: bool = False) -> tuple[float, str]:
        """Compute a smoothed focus score for a specific face.

        When *phone_distracted* is True an extra penalty is applied on top of
        the normal distraction penalties, dragging the class average down on
        the teacher’s dashboard chart.
        """
        state = self._get_face_state(face_id)
        score = 100.0

        if ear < EAR_THRESHOLD:
            state["ear_ctr"] += 1
            score -= min(40.0, state["ear_ctr"] * EAR_PEN_PER_FRAME)
        else:
            state["ear_ctr"] = max(0, state["ear_ctr"] - 2)

        score -= min(20.0, max(0.0, abs(yaw)   - YAW_THRESHOLD)   * HEAD_YAW_SCALE)
        score -= min(10.0, max(0.0, abs(pitch)  - PITCH_THRESHOLD) * HEAD_PITCH_SCALE)

        if is_distracted:
            score -= GAZE_AWAY_PENALTY

        # Phone-distraction penalty stacks on top of gaze distraction
        if phone_distracted:
            score -= _PHONE_FOCUS_PENALTY

        score = max(0.0, min(100.0, score))
        state["focus_hist"].append(score)
        smoothed = float(np.mean(state["focus_hist"]))
        return round(smoothed, 1), "Focused" if smoothed >= 60 else "Distracted"

    def _draw_face_overlay(self, frame: np.ndarray, lms: np.ndarray,
                            face_id: int, face_info: dict,
                            scale_x: float, scale_y: float) -> None:
        """Draw iris dots, bounding box, identity label, and phone-alert banner."""
        gaze_dir        = face_info["gaze"]
        status          = face_info["status"]
        score           = face_info["focus_score"]
        name            = face_info.get("name", "Unknown")
        prn             = face_info.get("prn",  "—")
        phone_distracted= face_info.get("phone_distracted", False)

        iris_col = (0, 255, 100) if gaze_dir == "Center" else (0, 80, 255)

        # Box colour: red for phone-alert, green focused, blue distracted, orange unknown
        if phone_distracted:
            box_col = (0, 30, 220)          # bright red (BGR)
        elif name == "Unknown":
            box_col = (0, 140, 255)         # orange
        elif status == "Focused":
            box_col = (0, 220, 0)           # green
        else:
            box_col = (0, 60, 220)          # blue

        # Iris circles
        if len(lms) >= 478:
            for idx in (LEFT_IRIS, RIGHT_IRIS):
                cx = int(lms[idx, 0] * scale_x)
                cy = int(lms[idx, 1] * scale_y)
                cv2.circle(frame, (cx, cy), 6, iris_col, 2)

        # Bounding box
        xs = (lms[:, 0] * scale_x).astype(int)
        ys = (lms[:, 1] * scale_y).astype(int)
        x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
        pad = 12
        x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
        x2, y2 = min(frame.shape[1]-1, x2+pad), min(frame.shape[0]-1, y2+pad)
        thickness = 3 if phone_distracted else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_col, thickness)

        # Identity label (two lines)
        label1 = "Unknown Student" if name == "Unknown" else f"{name} | {prn}"
        label2 = f"Focus: {score:.0f}%"

        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.50
        line_gap   = 20
        label_y1   = max(y1 - line_gap - 4, 20)
        label_y2   = max(y1 - 4, label_y1 + line_gap)
        cv2.putText(frame, label1, (x1, label_y1), font, font_scale, box_col, 2)
        cv2.putText(frame, label2, (x1, label_y2), font, font_scale - 0.05,
                    (200, 200, 200), 1)

        # Phone-distraction banner below the box
        if phone_distracted:
            banner_y = min(y2 + 22, frame.shape[0] - 4)
            cv2.putText(frame, "PHONE DETECTED",
                        (x1, banner_y), font, 0.52, (0, 30, 255), 2)

    def _draw_hud(self, frame: np.ndarray, m: dict) -> None:
        """Render the classroom-aggregate HUD strip at the bottom of the frame."""
        h, w = frame.shape[:2]
        # Simple semi-opaque bar: draw a filled black rect, then blend only the
        # ROI — avoids copying the entire full-res frame just for the HUD.
        roi = frame[h - 70 : h, 0 : w]
        roi[:] = (roi * 0.45).astype(np.uint8)

        n     = m["face_count"]
        alerts= m.get("phone_alerts", 0)
        col   = (0, 220, 0) if m["status"] == "Focused" else (0, 60, 220)
        p_col = (0, 30, 255) if alerts > 0 else (200, 200, 200)

        line1 = (f"Faces:{n}  Focus:{m['focus_score']:.1f}%"
                 f"  Status:{m['status']}  Gaze:{m['gaze']}"
                 f"  FPS:{m['fps']:.1f}")
        line2 = (f"EAR:{m['ear']:.3f}  Yaw:{m['yaw']:.1f}"
                 f"  Pitch:{m['pitch']:.1f}  Emotion:{m['emotion']}"
                 f"  Boredom:{m['boredom']*100:.0f}%"
                 f"  PhoneAlerts:{alerts}")

        cv2.putText(frame, line1, (10, h - 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, col, 2)
        cv2.putText(frame, line2, (10, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, p_col, 1)

    # -----------------------------------------------------------------------
    # Background capture loop  (runs in its own daemon thread)
    # -----------------------------------------------------------------------
    def _capture_loop(self) -> None:
        """Continuously capture and process frames in the background.

        Results are stored in *_latest_jpg* under *_frame_lock*.  The Flask
        streaming thread reads from that cache without ever blocking on ML.
        """
        while True:
            try:
                jpg = self._process_frame()
                if jpg is not None:
                    with self._frame_lock:
                        self._latest_jpg = jpg
            except Exception as exc:
                print(f"[camera] capture loop error: {exc}")

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------
    def get_frame(self) -> bytes | None:
        """Return the most recently processed JPEG frame.

        This is deliberately non-blocking: it just reads the cached bytes that
        the background capture thread deposited.  If no frame is ready yet
        (first ~100 ms of startup) it returns None and the caller skips.
        """
        with self._frame_lock:
            return self._latest_jpg

    def _process_frame(self) -> bytes | None:
        ok, frame = self.cap.read()
        if not ok:
            return None

        # Mirror so the feed feels natural (external cameras are not mirrored)
        frame  = cv2.flip(frame, 1)
        fh, fw = frame.shape[:2]          # high-res dimensions (e.g. 1920×1080)

        now = time.time()
        fps = 1.0 / max(now - self._prev_t, 1e-9)
        self._prev_t = now

        # ── ① Downscale for inference ─────────────────────────────────────
        # Keep the original full-res `frame` for drawing; run MediaPipe on a
        # lighter copy so landmark detection stays performant.
        infer_w, infer_h = self._INFER_W, self._INFER_H
        # Respect actual capture resolution (camera may not support 1080p)
        if fw > infer_w or fh > infer_h:
            small = cv2.resize(frame, (infer_w, infer_h),
                               interpolation=cv2.INTER_LINEAR)
        else:
            small = frame          # already small enough – no copy needed
            infer_w, infer_h = fw, fh

        # Scale factors: convert inference coords → full-res render coords
        scale_x = fw / infer_w
        scale_y = fh / infer_h

        # ── ② Run MediaPipe on the downscaled frame ───────────────────────
        rgb    = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(mp_img)

        # ── ②b Phone detection (YOLO, runs every _PHONE_DETECT_EVERY frames) ─
        self._phone_frame_ctr = (self._phone_frame_ctr + 1) % _PHONE_DETECT_EVERY
        if self._phone_frame_ctr == 0 and self._phone_model is not None:
            self._phone_boxes = _detect_phones(
                self._phone_model, small, fw, fh, infer_w, infer_h)
            # Draw phone boxes on full-res frame
            for (px1, py1, px2, py2) in self._phone_boxes:
                cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 30, 255), 2)
                cv2.putText(frame, "PHONE", (px1, max(py1-6, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 30, 255), 2)
        elif self._phone_boxes:
            # Redraw cached boxes so they persist between YOLO runs
            for (px1, py1, px2, py2) in self._phone_boxes:
                cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 30, 255), 2)
                cv2.putText(frame, "PHONE", (px1, max(py1-6, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 30, 255), 2)

        # ── ③ Per-face inference loop ─────────────────────────────────────
        face_data: list[dict] = []

        if result.face_landmarks:
            n_faces = len(result.face_landmarks)

            for face_id, face_lms in enumerate(result.face_landmarks):
                # Map landmarks to inference-frame pixel coordinates
                lms  = self._lm_arr(face_lms, infer_w, infer_h)

                # --- EAR (Eye-Aspect-Ratio) ---
                ear = (compute_ear(lms, LEFT_EYE) +
                       compute_ear(lms, RIGHT_EYE)) / 2.0

                # --- Head Pose (runs on inference-frame dims) ---
                pose = compute_head_pose(
                    lms[POSE_INDICES].astype(np.float64), infer_w, infer_h)

                # --- Gaze ---
                gaze = compute_gaze(lms)

                # --- Distraction decision ---
                yaw_distracted   = abs(pose["yaw"])   > YAW_THRESHOLD
                pitch_distracted = abs(pose["pitch"])  > PITCH_THRESHOLD
                iris_distracted  = gaze["direction"] != "Center"
                is_distracted    = yaw_distracted or pitch_distracted or iris_distracted
                gaze_state       = "Distracted" if is_distracted else "Centered"

                # --- Blendshapes & Emotion ---
                bs = (result.face_blendshapes[face_id]
                      if result.face_blendshapes else [])
                face_state = self._get_face_state(face_id)
                boredom    = face_state["boredom"].update(bs)
                emotion    = ("Bored"
                              if boredom > 0.5 and face_state["boredom"].ready
                              else detect_emotion(bs, ear=ear, roll=pose["roll"]))

                # ── ③a Identify face & mark attendance ───────────────────
                name, prn = self._identify_face(
                    face_id, frame, lms, scale_x, scale_y)

                # ── ③b Phone + gaze spatial correlation ──────────────────
                # Build full-res face bbox for spatial test
                xs_fr = (lms[:, 0] * scale_x).astype(int)
                ys_fr = (lms[:, 1] * scale_y).astype(int)
                face_box_fr = (xs_fr.min(), ys_fr.min(),
                               xs_fr.max(), ys_fr.max())
                phone_distracted = _correlate_phone_face(
                    self._phone_boxes, face_box_fr,
                    gaze["v_ratio"], fh)

                # --- Focus Score (single call, phone penalty included) ---
                focus_score, status = self._focus_score(
                    face_id, ear, pose["yaw"], pose["pitch"],
                    is_distracted, phone_distracted=phone_distracted)

                face_info = {
                    "face_id":          face_id,
                    "name":             name,
                    "prn":              prn,
                    "ear":              round(ear, 4),
                    "yaw":              round(pose["yaw"],   2),
                    "pitch":            round(pose["pitch"],  2),
                    "roll":             round(pose["roll"],   2),
                    "gaze":             gaze["direction"],
                    "gaze_h":           gaze["h_ratio"],
                    "gaze_v":           gaze["v_ratio"],
                    "gaze_state":       gaze_state,
                    "focus_score":      focus_score,
                    "status":           status,
                    "emotion":          emotion,
                    "boredom":          boredom,
                    "phone_distracted": phone_distracted,
                }
                face_data.append(face_info)

                # Draw per-face overlay on the HIGH-RES frame
                self._draw_face_overlay(frame, lms, face_id, face_info,
                                        scale_x, scale_y)

            # ── ④ Aggregate classroom-level metrics ───────────────────────
            n = len(face_data)

            avg_ear     = round(sum(f["ear"]         for f in face_data) / n, 4)
            avg_yaw     = round(sum(f["yaw"]         for f in face_data) / n, 2)
            avg_pitch   = round(sum(f["pitch"]       for f in face_data) / n, 2)
            avg_roll    = round(sum(f["roll"]        for f in face_data) / n, 2)
            avg_focus   = round(sum(f["focus_score"] for f in face_data) / n, 1)
            avg_boredom = round(sum(f["boredom"]     for f in face_data) / n, 3)
            avg_gaze_h  = round(sum(f["gaze_h"]      for f in face_data) / n, 3)
            avg_gaze_v  = round(sum(f["gaze_v"]      for f in face_data) / n, 3)

            # Majority-vote on gaze direction and status
            gaze_vote    = Counter(f["gaze"]    for f in face_data).most_common(1)[0][0]
            status_vote  = Counter(f["status"]  for f in face_data).most_common(1)[0][0]
            emotion_vote = Counter(f["emotion"] for f in face_data).most_common(1)[0][0]
            gaze_state_v = Counter(f["gaze_state"] for f in face_data).most_common(1)[0][0]
            phone_alerts = sum(1 for f in face_data if f.get("phone_distracted"))

            self.metrics.update({
                "face_count":   n,
                "status":       status_vote,
                "emotion":      emotion_vote,
                "focus_score":  avg_focus,
                "ear":          avg_ear,
                "yaw":          avg_yaw,
                "pitch":        avg_pitch,
                "roll":         avg_roll,
                "gaze":         gaze_vote,
                "gaze_h":       avg_gaze_h,
                "gaze_v":       avg_gaze_v,
                "gaze_state":   gaze_state_v,
                "boredom":      avg_boredom,
                "fps":          round(fps, 1),
                "phone_alerts": phone_alerts,
                "face_data":    face_data,
            })

            # ── ⑤ Log classroom averages every N frames ───────────────────
            self._log_ctr += 1
            if self._log_ctr >= self._LOG_EVERY:
                self.logger.log(
                    avg_ear, avg_yaw, avg_pitch, avg_roll,
                    gaze_vote, avg_focus, status_vote,
                    emotion_vote, avg_boredom)
                self._log_ctr = 0

        else:
            # ── No faces detected ─────────────────────────────────────────
            self.metrics.update({
                "face_count":  0,
                "status":      "No Face",
                "emotion":     "N/A",
                "focus_score": 0.0,
                "gaze":        "N/A",
                "gaze_state":  "N/A",
                "fps":         round(fps, 1),
                "face_data":   [],
            })
            cv2.putText(frame, "No Faces Detected — Waiting for Students",
                        (max(0, fw // 2 - 280), fh // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 60, 220), 2)

        self._draw_hud(frame, self.metrics)
        # Quality 75 cuts encode time by ~30 % versus 85 with imperceptible
        # visual difference at 1280×720 streamed over a local network.
        _, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
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
