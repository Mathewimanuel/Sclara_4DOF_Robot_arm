"""
vision_gui.py
=============
Robot Arm Vision Control GUI — Phase 2/3 Implementation
Architecture reference: architecture.docx v1.0

OVERVIEW
--------
Desktop GUI that tracks a hand-mounted ArUco marker (ID 0) via a Logitech
C920 camera, computes 3D hand position in the robot world frame, solves
inverse kinematics across all 4 axes (BASE, SHOULDER, ELBOW, WRIST), and
sends coordinated step commands to the ESP32 firmware at 10Hz over serial.
A separate MediaPipe pipeline measures pinch distance and maps it to gripper
PWM in the same serial command.

WINDOW LAYOUT
-------------
  Left   — Live camera feed (640×480 captured, 480×360 displayed)
           Overlays: ArUco detection boxes, hand crosshair, XYZ readout,
           pinch mm→PWM label, speed banner, calibration prompts
  Centre — 3D arm preview (matplotlib, live at ~10fps)
           Shows: arm segments, joint dots, target crosshair, collision
           zones, workspace boundary. Green crosshair = reachable,
           red = clamped. View buttons: FRONT / SIDE / TOP / ISO
  Right  — Control panel + serial monitor
           IK joint deltas and step counts (B/S/E/W)
           Gripper indicator (real mm + PWM)
           Reachability status + suppress reason
           Hand speed bar (3D magnitude)
           Hand calibration (START / CAPTURE / RESET)
           Command stats + last TX
           Serial monitor with TX entry

SIGNAL PIPELINE (per frame, architecture §5.6)
----------------------------------------------
  1. ArUco detection (CLAHE-enhanced greyscale)
  2. Ref marker (ID 10) → world frame transform (R_wc, t_wc)
  3. Hand marker (ID 0) → raw tvec in ref marker space
  4. Median filter N=3 per axis (spike removal, §5.2)
  5. One Euro Filter per axis (adaptive smoothing, §5.3)
  6. Coordinate transform → IK world frame (§3.3–3.4):
       arm_x = -(vision_x + REF_OFFSET_X_MM)   [ArUco X is mirrored]
       arm_y =   vision_y + REF_OFFSET_Y_MM
       arm_z =   vision_z + REF_OFFSET_Z_MM
  7. Dead band check: 8mm 3D distance in hand space BEFORE scaling (§6.5)
     → skips frame if hand hasn't moved enough from last sent position
  8. Hand calibration scale applied (X, Y, Z independently)
  9. Workspace clamp: X ±300mm, Y 80–425mm, Z 20–415mm  (recomputed from FK with TOOL_MM=122)
 10. IKSolver.solve(x, y, z) → IKResult (B/S/E/W deltas + steps)
 11. Safety gate — all must pass or command is dropped:
       a. 3D velocity magnitude < 400mm/s (warn >250mm/s)
       b. IKResult.reachable == True
       c. Per-axis step delta < 150 steps (jerk limit)
 12. Serial command sent at max 10Hz:  B:<n> S:<n> E:<n> W:<n> G:<n>

PINCH PIPELINE (per frame, architecture §5.5 + §5.6)
------------------------------------------------------
  1. MediaPipe Hands detects thumb tip (LM4) and index tip (LM8)
  2. Hand ArUco marker used as ruler:
       pixels_per_mm = corner_tl→tr distance / HAND_SIZE_MM (25.6mm)
       pinch_mm      = raw_pixel_distance / pixels_per_mm
     → real physical mm, scale-invariant with camera distance
  3. Clamped to 8–80mm physical range
  4. Gripper dead band: 44µs (~5°) — holds last value if change < threshold
  5. Mapped to PWM: closed(8mm)=2500µs, open(80mm)=900µs
  6. Included as G:<pulse_us> in every serial command

COORDINATE FRAMES (architecture §3)
-------------------------------------
  Vision frame origin : ref marker centre on table surface
  IK world frame origin: table surface directly below shoulder pivot
  Offset : REF_OFFSET_X/Y/Z_MM — measure once per setup
  X axis : mirrored (operator faces camera, natural mirror in X)
  Y axis : forward (toward arm)
  Z axis : up from table surface

HAND CALIBRATION (architecture §4.4, scale-only mode)
------------------------------------------------------
  Two captured points define the operator's comfortable hand range.
  Scale = workspace_span / hand_range per axis (X, Y, Z independently).
  10% padding NOT yet applied (Phase 2 remaining item).
  Dead band applied before scaling so threshold is constant in hand space.
  Calibration state machine: IDLE → WAIT_P1 → WAIT_P2 → DONE

SAFETY (architecture §6)
-------------------------
  ENABLE button    — master on/off; must be ON for commands to send
  Speed suppression— 3D magnitude computed over rolling position history
  Reachability gate— IKSolver returns reachable=False for clamped/collision
  Jerk limit       — max 150 steps/axis/command prevents runaway on glitch
  Tracking loss    — explicit stop + filter reset on marker disappearance
  Dead band        — 8mm minimum movement prevents tremor-induced drift
  Watchdog (ESP32) — firmware stops motors if no command within 500ms

REMAINING GAPS
--------------
  - 10% padding on calibration hand range (§4.4)
  - aruco_config.json save/load GUI fields for REF_OFFSET + workspace bounds (§12.1)
  - Phase 5: ik_gui.py manual joint jog integration

DEPENDENCIES
------------
    pip install pyserial opencv-contrib-python numpy pillow mediapipe matplotlib

FILES REQUIRED IN SAME DIRECTORY
----------------------------------
    camera_calibration.pkl  — from checkerboard calibration at 640×480
    IKSolver.py             — exposes solve(), IKResult, arm_geometry()
"""

import tkinter as tk
from tkinter import ttk
import threading
import time
import os
from scipy.interpolate import CubicSpline
import sys
import math
import datetime
import pickle
import json
from collections import deque

import cv2
import numpy as np
from PIL import Image, ImageTk
import mediapipe as mp

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import platform
import serial
import serial.tools.list_ports

# IKSolver must be in the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from IKSolver import solve, IKResult, arm_geometry
try:
    from IKSolver import forward as _ik_forward
except ImportError:
    # Older IKSolver without forward() — define fallback using FKResult structure
    def _ik_forward(base_delta, shoulder_delta, elbow_delta):
        from IKSolver import solve as _s
        class _FKR: x=0; y=0; z=0
        return _FKR()
forward = _ik_forward

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════

REFERENCE_ID       = 10
HAND_ID            = 0
REFERENCE_SIZE_MM  = 102.8
HAND_SIZE_MM       = 25.6

CALIB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "camera_calibration.pkl")

BAUD_RATE          = 115200
UPDATE_RATE_HZ     = 10     # architecture spec §7.2 — max 10Hz serial commands

# ── Reference marker → IK world frame offset ─────────────────
# Measure once: physical distance from ref marker centre to the
# point on the table directly below the shoulder pivot.
# +X = ref marker needs to move right to reach shoulder base
# +Y = ref marker needs to move forward (toward arm) to reach shoulder base
REF_OFFSET_X_MM =   0.0   # ← set after physical measurement
REF_OFFSET_Y_MM =   250.0   # ← set after physical measurement
REF_OFFSET_Z_MM =   0.0   # ← set if marker is raised off table (e.g. on a stand)


# Hand marker position is mapped directly into IKSolver world frame.
# These clamp the hand to a safe sub-volume of the full reachable space.
WS_X_MIN = -300.0;  WS_X_MAX =  300.0   # mm left/right
WS_Y_MIN =   80.0;  WS_Y_MAX =  425.0   # mm forward (FK max ~427mm at SH=90 EL=-110)
WS_Z_MIN =   20.0;  WS_Z_MAX =  415.0   # mm up (FK max ~417mm at SH=0 EL=-114)

# ── Dead band (architecture spec §6.5) ───────────────────────
# Applied in hand space BEFORE scaling — guarantees same physical
# hand movement threshold regardless of scale factor.
DEADBAND_MM        =  8.0   # mm — position must move this far from last sent
# Gripper dead band: 5 degrees servo equivalent
# pulse range = 1600µs over 180 degrees → 5° = 44µs
DEADBAND_GRIPPER_US = round(5 / 180 * 1600)   # = 44 µs


MAX_SPEED_3D    = 400.0   # mm/s  — 3D velocity magnitude; suppress above this

# §9 Speed matching — hand velocity → V: field in serial command
SPEED_MATCH_ENABLED  = True   # False → always send V:100 (full speed)
SPEED_MATCH_MIN      = 15     # % floor — arm speed when hand barely moves past deadband
SPEED_MATCH_MAX      = 100    # % ceiling — arm speed when hand hits HAND_MAX_SPEED_MMS
HAND_MAX_SPEED_MMS   = 70.0  # mm/s — hand speed that maps to SPEED_MATCH_MAX
                               # Set to your realistic top-end hand speed, NOT the safety cutoff
                               # Speeds above this still clamp to SPEED_MATCH_MAX (no harm)
WARNING_SPEED   = 50.0   # mm/s  — show warning above this
MAX_STEP_JUMP   = 150     # steps — max per-axis delta per command (jerk limit)

# §8.1 Trajectory spline smoother
TRAJ_BUFFER_N   =   15     # number of positions to fit spline through (tune: more=smoother/laggier)
TRAJ_MIN_SPAN   =   0.08  # seconds — minimum time span in buffer before spline fires

# ── Pinch detection (MediaPipe) ───────────────────────────────
# Pinch measured in real physical mm using hand ArUco marker as ruler (§5.5).
# ArUco corner top-left → top-right distance gives pixels_per_mm.
# No palm-size proxy used.
PINCH_CLOSED_MM  =  8.0   # mm — fully closed (fingers touching)
PINCH_OPEN_MM    = 80.0   # mm — fully open (comfortable max spread)

# ── Gripper PWM range ─────────────────────────────────────────
GRIPPER_MIN_PWM = 900    # µs — fully open  (103mm)
GRIPPER_MAX_PWM = 2500   # µs — fully closed


MEDIAN_WINDOW      = 1     # hand marker: OEF handles all smoothing; median=1 is intentional pass-through
REF_MEDIAN_WINDOW  = 3     # ref marker tvec/rvec spike rejection (3 frames ≈ 100ms at 30fps)
ONE_EURO_FREQ      = 20.0
ONE_EURO_MINCUTOFF = 0.5
ONE_EURO_BETA      = 0.5
ONE_EURO_DCUTOFF   = 1.0

# §5.6 Pinch filter — separate params per architecture spec
PINCH_MEDIAN_WINDOW  = 3
PINCH_OEF_MINCUTOFF  = 1.5   # slightly higher for faster gripper response
PINCH_OEF_BETA       = 0.05  # lower for more stable gripper

# §4.4 Calibration rolling buffer — always captures on press, uses mean of buffer
CALIB_ROLLING_FRAMES  = 30     # rolling window size (30 frames ≈ 1s at 30fps)
CALIB_STABILITY_GOOD  = 8.0    # mm spread — "good" stability indicator threshold
CALIB_STABILITY_OK    = 20.0   # mm spread — "ok" stability indicator threshold

# §6.1 WAITING state — both markers must be visible for this long before STANDBY
WAITING_VISIBLE_SECS  = 1.0

MONITOR_MAX_LINES  = 1000

# ── Phase 5: Manual Jog constants ────────────────────────────
JOG_SIZES = [1, 5, 10, 25, 50]   # mm step options
JOG_AXIS_COLORS = {"X": "#ff6666", "Y": "#66ff88", "Z": "#6699ff"}
JOG_PRESETS = [
    # label,       x,      y,      z
    ("IK ZERO",   0.0,  230.5,  191.4),
    ("MAX FWD",   0.0,  427.1,   60.6),
    ("MAX HIGH",  0.0,  250.1,  417.1),
    ("NEAR TBL",  0.0,  304.4,   40.0),
    ("LEFT 45", -222.1,  222.1,  218.1),
    ("RIGHT 45",222.1, 222.1,  218.1),
]

# ═══════════════════════════════════════════════════════════════
# §12.1 CONFIG LOAD / SAVE  (aruco_config.json)
# ═══════════════════════════════════════════════════════════════

CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "aruco_config.json")

# Module-level constants persisted to JSON.
# Calibration state (scale/offset/P1/P2) is saved separately by the instance.
_CONFIG_KEYS = {
    "REF_OFFSET_X_MM":     "float",
    "REF_OFFSET_Y_MM":     "float",
    "REF_OFFSET_Z_MM":     "float",
    "WS_X_MIN":            "float",
    "WS_X_MAX":            "float",
    "WS_Y_MIN":            "float",
    "WS_Y_MAX":            "float",
    "WS_Z_MIN":            "float",
    "WS_Z_MAX":            "float",
    "DEADBAND_MM":         "float",
    "MAX_SPEED_3D":        "float",
    "WARNING_SPEED":       "float",
    "SPEED_MATCH_ENABLED":  "int",
    "SPEED_MATCH_MIN":      "int",
    "SPEED_MATCH_MAX":      "int",
    "HAND_MAX_SPEED_MMS":   "float",
    "MAX_STEP_JUMP":       "int",
    "PINCH_CLOSED_MM":     "float",
    "PINCH_OPEN_MM":       "float",
    "GRIPPER_MIN_PWM":     "int",
    "GRIPPER_MAX_PWM":     "int",
    "ONE_EURO_MINCUTOFF":  "float",
    "ONE_EURO_BETA":       "float",
    "PINCH_OEF_MINCUTOFF": "float",
    "PINCH_OEF_BETA":      "float",
    "UPDATE_RATE_HZ":      "int",
}

def load_config():
    """Load module-level constants from aruco_config.json at import time."""
    global REF_OFFSET_X_MM, REF_OFFSET_Y_MM, REF_OFFSET_Z_MM
    global WS_X_MIN, WS_X_MAX, WS_Y_MIN, WS_Y_MAX, WS_Z_MIN, WS_Z_MAX
    global DEADBAND_MM, MAX_SPEED_3D, WARNING_SPEED, MAX_STEP_JUMP
    global SPEED_MATCH_ENABLED, SPEED_MATCH_MIN, SPEED_MATCH_MAX, HAND_MAX_SPEED_MMS
    global PINCH_CLOSED_MM, PINCH_OPEN_MM, GRIPPER_MIN_PWM, GRIPPER_MAX_PWM
    global ONE_EURO_MINCUTOFF, ONE_EURO_BETA, PINCH_OEF_MINCUTOFF, PINCH_OEF_BETA
    global UPDATE_RATE_HZ

    if not os.path.exists(CONFIG_FILE):
        return

    try:
        with open(CONFIG_FILE, "r") as f:
            data = json.load(f)
        g = globals()
        for key, typ in _CONFIG_KEYS.items():
            if key in data:
                g[key] = float(data[key]) if typ == "float" else int(data[key])
        print(f"[config] Loaded constants from {CONFIG_FILE}")
    except Exception as ex:
        print(f"[config] WARNING: failed to load {CONFIG_FILE}: {ex}")

# Load constants at import — before any class is constructed
load_config()


# ── Palette (matches ik_gui.py) ───────────────────────────────
BG       = "#0a0f0a"
BG2      = "#0d1a0d"
BG3      = "#050a05"
FG       = "#00ff88"
FG_DIM   = "#336633"
FG_ERR   = "#ff4444"
FG_WARN  = "#ffaa00"
FG_TX    = "#557755"

FONT_MONO = ("Courier New", 10)
FONT_HEAD = ("Courier New", 12, "bold")
FONT_BIG  = ("Courier New", 36, "bold")
FONT_TINY = ("Courier New",  8)



# ═══════════════════════════════════════════════════════════════
# FILTERS  (architecture spec §5.2 / §5.3)
# ═══════════════════════════════════════════════════════════════

class MedianFilter:
    """Window-N median filter — removes single-frame spike outliers.
    Uses deque(maxlen=n) for O(1) append+discard instead of list.pop(0) O(N).
    9 active instances × 30fps = 270 discards/second — deque matters here.
    """
    def __init__(self, n=3):
        self.n   = n
        self.buf = deque(maxlen=n)  # auto-discards oldest; O(1)

    def filter(self, x):
        self.buf.append(x)          # no manual pop needed
        s = sorted(self.buf)
        return s[len(s) // 2]

    def reset(self):
        self.buf.clear()


class OneEuroFilter:
    """
    One Euro Filter — adaptive low-pass for real-time tracking.
    Heavy smoothing at low speed, minimal lag at high speed.
    Reference: Casiez et al. 2012, CHI.
    """
    def __init__(self, freq, mincutoff=1.0, beta=0.0, dcutoff=1.0):
        self.freq      = freq
        self.mincutoff = mincutoff
        self.beta      = beta
        self.dcutoff   = dcutoff
        self._x_prev   = None
        self._dx_prev  = 0.0

    def _alpha(self, cutoff):
        te  = 1.0 / self.freq
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def filter(self, x):
        if self._x_prev is None:
            self._x_prev = x
            return x
        # Derivative estimate
        dx     = (x - self._x_prev) * self.freq
        # Smooth derivative with fixed dcutoff
        a_d    = self._alpha(self.dcutoff)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev
        # Adaptive signal cutoff
        cutoff = self.mincutoff + self.beta * abs(dx_hat)
        a      = self._alpha(cutoff)
        x_hat  = a * x + (1.0 - a) * self._x_prev
        self._x_prev  = x_hat
        self._dx_prev = dx_hat
        return x_hat

    def reset(self):
        self._x_prev  = None
        self._dx_prev = 0.0


# ═══════════════════════════════════════════════════════════════
# ARUCO TRACKER
# ═══════════════════════════════════════════════════════════════

class ArUcoTracker:

    def __init__(self):
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100)
        self.params = cv2.aruco.DetectorParameters()
        self.params.minMarkerPerimeterRate = 0.02
        self._clahe = cv2.createCLAHE(2.0, (8, 8))   # cached — created once, reused every frame

        self.marker_sizes = {
            REFERENCE_ID: REFERENCE_SIZE_MM / 1000.0,
            HAND_ID:       HAND_SIZE_MM      / 1000.0,
        }

        with open(CALIB_FILE, "rb") as f:
            calib = pickle.load(f)
        self.K = calib["camera_matrix"]
        self.D = calib["dist_coeffs"]

        self.R_wc = None
        self.t_wc = None
        self._ref_last_seen = 0.0   # timestamp of last successful ref marker update
        # Median filters for ref marker raw tvec and rvec (6 scalars)
        # Absorbs per-frame detection noise without EMA warmup/drift issues
        self._ref_med_tvec = [MedianFilter(REF_MEDIAN_WINDOW) for _ in range(3)]
        self._ref_med_rvec = [MedianFilter(REF_MEDIAN_WINDOW) for _ in range(3)]

        # Two-stage filter pipeline per axis (x, y, z)
        # Stage 1 — Median pre-filter: removes ArUco pose spike outliers
        self._median = [MedianFilter(MEDIAN_WINDOW) for _ in range(3)]
        # Stage 2 — One Euro Filter: adaptive smoothing with minimal lag
        self._oef    = [OneEuroFilter(ONE_EURO_FREQ,
                                      mincutoff=ONE_EURO_MINCUTOFF,
                                      beta=ONE_EURO_BETA,
                                      dcutoff=ONE_EURO_DCUTOFF)
                        for _ in range(3)]

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = self._clahe.apply(gray)   # reuse cached CLAHE object — no per-frame allocation
        corners, ids, _ = cv2.aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.params)
        poses = {}
        if ids is not None:
            for i, mid in enumerate(ids.flatten()):
                mid = int(mid)
                if mid not in self.marker_sizes:
                    continue
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners[i], self.marker_sizes[mid], self.K, self.D)
                poses[mid] = {
                    "rvec":    rvec[0][0],
                    "tvec":    tvec[0][0],
                    "corners": corners[i],
                }
        return poses

    def update_extrinsics(self, ref_pose):
        """Recompute world frame transform from ref marker pose.
        Applies a small median filter to raw tvec/rvec to suppress single-frame
        detection spikes (e.g. from motion blur, CLAHE flicker).
        No warmup/drift problem — median returns valid output from frame 1.
        """
        raw_rvec = ref_pose["rvec"].reshape(3)
        raw_tvec = ref_pose["tvec"].reshape(3)
        # Median-filter each scalar independently
        filt_rvec = np.array([self._ref_med_rvec[i].filter(raw_rvec[i]) for i in range(3)])
        filt_tvec = np.array([self._ref_med_tvec[i].filter(raw_tvec[i]) for i in range(3)])
        R_cm, _ = cv2.Rodrigues(filt_rvec)
        t_cm    = filt_tvec.reshape(3, 1)
        self.R_wc = R_cm.T
        self.t_wc = -self.R_wc @ t_cm
        self._ref_last_seen = time.monotonic()

    def world_position(self, pose):
        """Return filtered 3D position in world frame (cm)."""
        if self.R_wc is None:
            return None
        raw = (self.R_wc @ pose["tvec"].reshape(3, 1) + self.t_wc).flatten()
        # Stage 1: Median pre-filter (spike removal)
        median_out = np.array([self._median[i].filter(raw[i]) for i in range(3)])
        # Stage 2: One Euro Filter (adaptive smoothing)
        filtered   = np.array([self._oef[i].filter(median_out[i]) for i in range(3)])
        return filtered   # metres → caller converts to cm

    def reset_filters(self):
        """Call when tracking is lost to avoid stale filter state."""
        for f in self._median:
            f.reset()
        for f in self._oef:
            f.reset()

    def update_filter_params(self, median_n, oef_freq, oef_mincutoff, oef_beta, oef_dcutoff):
        """Public method to rebuild position filters with new parameters.
        Replaces direct _median/_oef field access from VisionGUI._apply_filter_params.
        """
        self._median = [MedianFilter(median_n) for _ in range(3)]
        self._oef    = [OneEuroFilter(oef_freq,
                                      mincutoff=oef_mincutoff,
                                      beta=oef_beta,
                                      dcutoff=oef_dcutoff)
                        for _ in range(3)]

# ═══════════════════════════════════════════════════════════════
# PINCH DETECTOR  (MediaPipe Hands)
# ═══════════════════════════════════════════════════════════════

class PinchDetector:
    """
    Measures thumb-index pinch in real physical mm using the hand ArUco
    marker as a ruler (architecture §5.5).

    pixels_per_mm = aruco_corner_tl→tr distance / HAND_SIZE_MM
    pinch_mm      = raw_pixel_distance / pixels_per_mm

    Clamped to PINCH_CLOSED_MM–PINCH_OPEN_MM then mapped to gripper PWM.
    Requires update_aruco_ruler() to be called each frame with the current
    hand marker corners so the px→mm scale stays valid as hand moves.
    """

    THUMB_TIP = 4
    INDEX_TIP = 8

    def __init__(self):
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
        )
        self.pinch_mm       = PINCH_OPEN_MM
        self.landmarks      = None
        self.hand_seen      = False
        self._pixels_per_mm = None

        # §5.6 — two-stage filter on pinch distance (same pattern as position)
        self._pinch_median = MedianFilter(PINCH_MEDIAN_WINDOW)
        self._pinch_oef    = OneEuroFilter(ONE_EURO_FREQ,
                                           mincutoff=PINCH_OEF_MINCUTOFF,
                                           beta=PINCH_OEF_BETA,
                                           dcutoff=ONE_EURO_DCUTOFF)

    def update_aruco_ruler(self, corners):
        """
        Call each frame with hand marker corners (shape 4×2, pixels).
        Computes pixels_per_mm from top-left → top-right edge (§5.5).
        """
        tl = corners[0]   # top-left
        tr = corners[1]   # top-right
        edge_px = np.linalg.norm(tr - tl)
        if edge_px > 1.0:
            self._pixels_per_mm = edge_px / HAND_SIZE_MM

    def process(self, bgr_frame):
        """Process one BGR frame. Updates pinch_mm and landmarks."""
        rgb    = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)

        if not result.multi_hand_landmarks:
            self.landmarks = None
            self.hand_seen = False
            self.pinch_mm  = PINCH_OPEN_MM
            # Reset filters on hand loss so stale state doesn't bleed in on re-detection
            self._pinch_median.reset()
            self._pinch_oef.reset()
            return

        lm             = result.multi_hand_landmarks[0].landmark
        self.landmarks = lm
        self.hand_seen = True

        if self._pixels_per_mm is None or self._pixels_per_mm < 1e-3:
            return

        h, w = bgr_frame.shape[:2]
        def px(idx):
            return np.array([lm[idx].x * w, lm[idx].y * h])

        raw_px  = np.linalg.norm(px(self.THUMB_TIP) - px(self.INDEX_TIP))
        raw_mm  = raw_px / self._pixels_per_mm

        # §5.6 — Median spike removal then One Euro adaptive smoothing
        med_mm       = self._pinch_median.filter(raw_mm)
        filtered_mm  = self._pinch_oef.filter(med_mm)

        self.pinch_mm = max(PINCH_CLOSED_MM, min(PINCH_OPEN_MM, filtered_mm))

    @property
    def gripper_pwm(self) -> int:
        """Map pinch_mm → PWM µs. Closed(8mm)=2500µs, Open(80mm)=900µs."""
        t = (self.pinch_mm - PINCH_CLOSED_MM) / (PINCH_OPEN_MM - PINCH_CLOSED_MM)
        t = max(0.0, min(1.0, t))
        # t=0 → closed → GRIPPER_MAX_PWM; t=1 → open → GRIPPER_MIN_PWM
        return round(GRIPPER_MAX_PWM - t * (GRIPPER_MAX_PWM - GRIPPER_MIN_PWM))

    def draw_overlay(self, frame):
        """Draw hand skeleton and pinch indicator onto frame (in-place)."""
        if self.landmarks is None:
            return

        h, w = frame.shape[:2]
        lm   = self.landmarks

        def pt(idx):
            return (int(lm[idx].x * w), int(lm[idx].y * h))

        # Skeleton
        for a, b in mp.solutions.hands.HAND_CONNECTIONS:
            cv2.line(frame, pt(a), pt(b), (40, 100, 40), 1)

        # Thumb and index tips
        t = pt(self.THUMB_TIP)
        i = pt(self.INDEX_TIP)

        # Colour: green = open, orange = closed
        ratio = 1.0 - (self.pinch_mm - PINCH_CLOSED_MM) / (PINCH_OPEN_MM - PINCH_CLOSED_MM)
        ratio = max(0.0, min(1.0, ratio))
        tip_col = (
            0,
            int(ratio * 170 + (1 - ratio) * 255),
            int(ratio * 255 + (1 - ratio) * 136),
        )
        cv2.circle(frame, t, 7, tip_col, -1)
        cv2.circle(frame, i, 7, tip_col, -1)
        cv2.line(frame, t, i, tip_col, 2)

        # Label shows real mm and PWM
        mid = ((t[0] + i[0]) // 2, (t[1] + i[1]) // 2 - 12)
        cv2.putText(frame, f"{self.pinch_mm:.0f}mm → G:{self.gripper_pwm}",
                    mid, cv2.FONT_HERSHEY_SIMPLEX, 0.4, tip_col, 1)

    def update_filter_params(self, median_n, oef_freq, oef_mincutoff, oef_beta, oef_dcutoff):
        """Public method to rebuild pinch filters with new parameters."""
        self._pinch_median = MedianFilter(median_n)
        self._pinch_oef    = OneEuroFilter(oef_freq,
                                           mincutoff=oef_mincutoff,
                                           beta=oef_beta,
                                           dcutoff=oef_dcutoff)

    def close(self):
        self._hands.close()


# ═══════════════════════════════════════════════════════════════
# SERIAL MANAGER
# ═══════════════════════════════════════════════════════════════

class SerialManager:
    def __init__(self, on_rx):
        self.ser      = None
        self.on_rx    = on_rx
        self._thread  = None
        self._running = False

    def connect(self, port):
        try:
            self.ser = serial.Serial(port, BAUD_RATE, timeout=0.1)
            self._running = True
            self._thread  = threading.Thread(target=self._read_loop, daemon=True)
            self._thread.start()
            return True
        except Exception as ex:
            return str(ex)

    def disconnect(self):
        self._running = False
        if self.ser:
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None

    def send(self, msg):
        if self.ser and self.ser.is_open:
            try:
                self.ser.write((msg + "\n").encode())
                return True
            except Exception:
                return False
        return False

    @property
    def is_connected(self):
        return self.ser is not None and self.ser.is_open

    def _read_loop(self):
        buf = ""
        while self._running:
            try:
                if self.ser and self.ser.in_waiting:
                    buf += self.ser.read(self.ser.in_waiting).decode(errors="replace")
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        line = line.strip()
                        if line:
                            self.on_rx(line)
                else:
                    time.sleep(0.01)
            except Exception:
                time.sleep(0.1)


# ═══════════════════════════════════════════════════════════════
# MAIN GUI
# ═══════════════════════════════════════════════════════════════

class VisionGUI:

    def __init__(self, root):
        self.root = root
        self.root.title("Robot Arm — Vision Control")
        self.root.configure(bg=BG)
        self.root.minsize(1440, 720)

        self.serial    = SerialManager(self.on_rx)
        self.connected = False
        self.homing    = False
        self._hand_tracked = False   # updated each camera frame; guards stale _clear_tracking calls

        self.tracker   = None
        self._load_tracker()

        # Pinch detector (MediaPipe) — maps to gripper PWM
        self._pinch    = PinchDetector()

        self.cap       = None
        self._cam_running = False

        # State
        self.enabled       = False
        self._manual_mode  = False     # Phase 5: when True, jog panel controls arm; vision sends suppressed
        self._jog_x        = 0.0      # current jog target in IK world mm
        self._jog_y        = 230.5
        self._jog_z        = 191.4
        self._jog_result   = None     # last IKResult from jog solve
        self.current_pos   = None      # (x_mm, y_mm, z_mm) world frame
        self.current_speed = 0.0       # mm/s 3D magnitude
        self.speed_status  = "NO_TRACKING"
        self.last_result   = None      # last IKResult
        self.last_steps    = None      # (b, s, e, w) last sent steps
        self.command_count = 0
        self.last_send_t        = 0.0
        self._preset_move_until = 0.0   # keepalive blocked until preset move completes
        self._last_cmd_str  = ""      # last TX string — read by _update_displays at 10Hz
        self._3d_last_draw  = 0.0     # timestamp of last 3D preview redraw
        self.suppress_reason = ""      # why last command was suppressed
        self._ids_present     = []        # last detected marker IDs — read by _update_displays

        # ── §6.1–6.2 Control state machine ───────────────────────
        # WAITING  — neither/one marker visible; no commands sent
        # STANDBY  — both markers visible; waiting for 2s stillness
        # TRACKING — actively sending commands at 10Hz
        # HOLDING  — speed exceeded; last position held
        self.control_state     = "WAITING"
        self._both_visible_since   = None   # time both markers first seen together
        self._stable_since         = None
        self._holding_stable_since = None
        ACTIVATION_HOLD_SECS   = 2.0    # §6.2 still for this long to enter TRACKING
        RETRACK_HOLD_SECS      = 0.5    # §6.2 stable for this long to leave HOLDING
        HOLD_SPEED_THRESHOLD   = 80.0   # mm/s — enter HOLDING above this (§6.3 8cm/s)
        self._activation_secs  = ACTIVATION_HOLD_SECS
        self._retrack_secs     = RETRACK_HOLD_SECS
        self._hold_threshold   = HOLD_SPEED_THRESHOLD

        self._pos_hist  = deque(maxlen=30)   # (t, np.array xyz_mm)

        # ── Dead band state ───────────────────────────────────────
        self._last_sent_raw_xyz = None
        self._last_sent_g       = None

        # ── §8.1 Trajectory spline buffer ──────────────────────────
        # Stores (t, xyz_mm) after calibration remap, before clamp.
        # CubicSpline fitted each frame; evaluated at t[-1] for smooth output.
        self._traj_buf = deque(maxlen=TRAJ_BUFFER_N)

        # ── Perf throttles ────────────────────────────────────────
        self._pinch_last_t      = 0.0   # last time MediaPipe ran (throttled to 10Hz)
        self._display_timer_id  = None  # recurring 10Hz display update timer

        # ── Hand calibration state ────────────────────────────────
        self._calib_state  = "IDLE"
        self._calib_p1     = None
        self._calib_p2     = None
        self._calib_scale  = np.array([1.0, 1.0, 1.0])
        self._calib_offset = np.array([0.0, 0.0, 0.0])
        self._calib_R_wc   = None   # R_wc snapshot at calibration time
        self._calib_t_wc   = None   # t_wc snapshot at calibration time (for translation drift)
        self._last_raw_xyz = None
        # §4.4 — rolling buffer: always accepts on press, mean suppresses tremor
        self._calib_rolling = deque(maxlen=CALIB_ROLLING_FRAMES)

        # ── §4.3 Ref marker alignment state ──────────────────────
        # "ALIGNING" on startup — overlay guides operator to place ref marker.
        # "READY"    after SPACE confirm — normal tracking begins.
        self._align_state    = "ALIGNING"
        self._align_ok       = False   # True when pos error <5mm AND rot error <3°
        self._align_pos_err  = None    # mm, updated each frame
        self._align_rot_err  = None    # degrees, updated each frame

        self._build_ui()
        self._refresh_ports()
        self._apply_filter_params()   # re-init filters with loaded constants
        self._load_calib()            # restore saved calibration if present

        # Bind SPACE to confirm alignment (§4.3)
        self.root.bind("<space>", self._confirm_alignment)

        # Start camera
        self._start_camera()

        # Start 10Hz recurring display-update timer (fix 3 — decouples GUI updates from cam thread)
        self._schedule_display_timer()

    # ── Tracker init ─────────────────────────────────────────

    def _schedule_display_timer(self):
        """Recurring 10Hz timer that drives all GUI widget updates.
        Also sends keepalives so the firmware watchdog doesn't fire when:
        - hand marker is not visible (manual mode, hand put down)
        - commands suppressed for any reason but Python is still connected
        Keepalive is independent of camera thread and hand visibility.
        """
        self._update_displays()
        self._send_keepalive(time.time())
        self._display_timer_id = self.root.after(
            int(1000 / UPDATE_RATE_HZ), self._schedule_display_timer)

    def _load_tracker(self):
        try:
            self.tracker = ArUcoTracker()   
            self._tracker_ok = True
        except FileNotFoundError:
            self._tracker_ok = False
            self._tracker_err = f"camera_calibration.pkl not found:\n{CALIB_FILE}"
        except Exception as ex:
            self._tracker_ok = False
            self._tracker_err = str(ex)

    # ═══════════════════════════════════════════════════════
    # UI BUILD
    # ═══════════════════════════════════════════════════════

    def _build_ui(self):
        self._build_topbar()
        self._build_statusbar()
        self._build_main_area()

    def _build_topbar(self):
        bar = tk.Frame(self.root, bg=BG2, pady=8, padx=12)
        bar.pack(fill="x")

        tk.Label(bar, text="VISION CONTROL  —  FULL 3D IK",
                 bg=BG2, fg=FG, font=FONT_HEAD).pack(side="left")

        conn = tk.Frame(bar, bg=BG2)
        conn.pack(side="right")

        tk.Label(conn, text="CAM:", bg=BG2, fg=FG_DIM,
                 font=FONT_MONO).pack(side="left")
        self.cam_index_var = tk.IntVar(value=0)  # default 0 = default camera
        tk.Spinbox(conn, from_=0, to=5, width=3,
                   textvariable=self.cam_index_var,
                   bg=BG3, fg=FG, font=FONT_MONO,
                   buttonbackground=BG2, relief="flat",
                   command=self._restart_camera).pack(side="left", padx=(2, 12))

        tk.Label(conn, text="PORT:", bg=BG2, fg=FG_DIM,
                 font=FONT_MONO).pack(side="left")
        self.port_var   = tk.StringVar()
        self.port_combo = ttk.Combobox(conn, textvariable=self.port_var,
                                        width=12, font=FONT_MONO)
        self.port_combo.pack(side="left", padx=4)

        tk.Button(conn, text="↻", bg=BG2, fg=FG,
                  font=("Courier New", 12), relief="flat", cursor="hand2",
                  command=self._refresh_ports).pack(side="left")

        self.btn_connect = tk.Button(
            conn, text="CONNECT", bg="#001a08", fg=FG,
            font=FONT_MONO, relief="flat", padx=10, cursor="hand2",
            command=self._toggle_connect)
        self.btn_connect.pack(side="left", padx=(8, 0))

        # HOME ALL button
        tk.Button(conn, text="HOME ALL", bg="#002200", fg=FG,
                  font=FONT_MONO, relief="flat", padx=8, cursor="hand2",
                  command=self._home_all).pack(side="left", padx=(12, 0))

    def _build_statusbar(self):
        self.status_var = tk.StringVar(value="● DISCONNECTED")
        self.status_lbl = tk.Label(
            self.root, textvariable=self.status_var,
            bg=BG, fg=FG_ERR, font=FONT_MONO, anchor="w", padx=12)
        self.status_lbl.pack(fill="x")

    def _build_main_area(self):
        main = tk.Frame(self.root, bg=BG)
        main.pack(fill="both", expand=True, padx=12, pady=8)

        # Right: controls + monitor — scrollable canvas wrapper
        right_outer = tk.Frame(main, bg=BG, width=396)
        right_outer.pack(side="right", fill="y", padx=(12, 0))
        right_outer.pack_propagate(False)

        right_scroll = tk.Scrollbar(right_outer, orient="vertical", bg=BG, troughcolor=BG2)
        right_scroll.pack(side="right", fill="y")

        right_canvas = tk.Canvas(right_outer, bg=BG, bd=0, highlightthickness=0,
                                  yscrollcommand=right_scroll.set, width=376)
        right_canvas.pack(side="left", fill="both", expand=True)
        right_scroll.config(command=right_canvas.yview)

        right = tk.Frame(right_canvas, bg=BG, width=376)
        right_win = right_canvas.create_window((0, 0), window=right, anchor="nw")

        _scroll_initialised = [False]
        def _on_right_configure(event):
            bb = right_canvas.bbox("all")
            if bb:
                right_canvas.configure(scrollregion=bb)
                if not _scroll_initialised[0]:
                    right_canvas.yview_moveto(0.0)  # scroll to top on first layout only
                    _scroll_initialised[0] = True
        def _on_canvas_resize(event):
            right_canvas.itemconfig(right_win, width=event.width)
        right.bind("<Configure>", _on_right_configure)
        right_canvas.bind("<Configure>", _on_canvas_resize)

        # Mouse wheel scroll
        def _on_mousewheel(event):
            right_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        right_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        self._build_control_panel(right)
        self._build_jog_panel(right)
        self._build_monitor_panel(right)

        # Centre: 3D preview (fixed width)
        centre = tk.Frame(main, bg=BG, width=420)
        centre.pack(side="right", fill="y", padx=(8, 0))
        centre.pack_propagate(False)
        self._build_3d_panel(centre)

        # Left: camera feed (expands)
        left = tk.Frame(main, bg=BG)
        left.pack(side="left", fill="both", expand=True)
        self._build_camera_panel(left)

    def _build_camera_panel(self, parent):
        frame = tk.LabelFrame(parent, text=" CAMERA FEED ",
                               bg=BG2, fg=FG, font=FONT_HEAD,
                               relief="solid", bd=1)
        frame.pack(fill="x", pady=4)

        self.cam_label = tk.Label(frame, bg=BG3,
                                   text="Camera not started",
                                   fg=FG_DIM, font=FONT_MONO,
                                   width=480, height=360)
        self.cam_label.pack(padx=4, pady=4)

        # Camera controls bar
        ctrl = tk.Frame(frame, bg=BG2, pady=4)
        ctrl.pack(fill="x", padx=4)

        self.enable_btn = tk.Button(
            ctrl, text="▶  ENABLE CONTROL",
            bg="#002200", fg=FG,
            font=("Courier New", 11, "bold"),
            relief="flat", padx=16, pady=4, cursor="hand2",
            command=self._toggle_enable)
        self.enable_btn.pack(side="left")

        self._speed_match_on = tk.BooleanVar(value=SPEED_MATCH_ENABLED)
        self.speed_match_btn = tk.Button(
            ctrl, text="⚡ SPD MATCH: ON",
            bg="#001a2e", fg="#00aaff",
            font=("Courier New", 10, "bold"),
            relief="flat", padx=10, pady=4, cursor="hand2",
            command=self._toggle_speed_match)
        self.speed_match_btn.pack(side="left", padx=(6, 0))

        self.detect_var = tk.StringVar(value="NO MARKERS")
        tk.Label(ctrl, textvariable=self.detect_var,
                 bg=BG2, fg=FG_DIM, font=FONT_MONO).pack(side="right")

    def _build_3d_panel(self, parent):
        frame = tk.LabelFrame(parent, text=" 3D ARM PREVIEW (LIVE) ",
                               bg=BG2, fg=FG, font=FONT_HEAD,
                               relief="solid", bd=1)
        frame.pack(fill="both", expand=True, pady=4)

        fig = plt.Figure(figsize=(4.2, 4.8), dpi=88, facecolor=BG2)
        self.ax3d = fig.add_subplot(111, projection="3d")
        self.ax3d.set_facecolor(BG)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        self._style_3d_axes()

        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)
        self._3d_canvas = canvas
        self._3d_fig    = fig

        # ── Persistent artists — created once, data updated in-place ──
        # Static geometry (never changes)
        # (numpy and math already imported at module level as np and math)
        r = 400
        xx, yy = np.meshgrid([-r, r], [-r, r])
        self.ax3d.plot_surface(xx, yy, np.zeros_like(xx),
                               alpha=0.06, color="#336633", linewidth=0)
        theta = np.linspace(math.radians(-110), math.radians(110), 60)
        self.ax3d.plot(380 * np.sin(theta), 380 * np.cos(theta), np.zeros(60),
                       color="#1a3a1a", linewidth=0.8, linestyle="dotted")
        th = np.linspace(0, 2 * math.pi, 48)
        xc, yc = 40 * np.cos(th), 40 * np.sin(th)
        for z_v in (0, 95):
            self.ax3d.plot(xc, yc, z_v, color="#cc3333", alpha=0.2, linewidth=0.8)
        xs, ys = 84 * np.cos(th), 84 * np.sin(th)
        for z_v in (95, 150):
            self.ax3d.plot(xs, ys, z_v, color="#cc8822", alpha=0.2, linewidth=0.8)

        # Dynamic arm segments (4 lines)
        z = [0, 0]
        self._ln_mount,  = self.ax3d.plot(z, z, z, color="#446644", linewidth=3)
        self._ln_upper,  = self.ax3d.plot(z, z, z, color=FG,        linewidth=4)
        self._ln_fore,   = self.ax3d.plot(z, z, z, color="#00cc66", linewidth=4)
        self._ln_tool,   = self.ax3d.plot(z, z, z, color="#ffaa00", linewidth=2.5,
                                           linestyle="dashed")
        # Crosshair lines (3 lines)
        self._ln_cx, = self.ax3d.plot(z, z, z, color=FG, linewidth=1.5)
        self._ln_cy, = self.ax3d.plot(z, z, z, color=FG, linewidth=1.5)
        self._ln_cz, = self.ax3d.plot(z, z, z, color=FG, linewidth=1.5)
        self._ln_cdrop, = self.ax3d.plot(z, z, z, color=FG, linewidth=0.5,
                                          linestyle="dotted", alpha=0.5)

        # Joint dots (5 scatter — one per joint)
        joint_cols = ["#446644", FG, FG, "#00cc66", "#ffaa00"]
        joint_sz   = [35, 55, 55, 45, 35]
        self._sc_joints = [
            self.ax3d.scatter([0], [0], [0], color=c, s=s, depthshade=False, zorder=5)
            for c, s in zip(joint_cols, joint_sz)
        ]
        self._sc_target = self.ax3d.scatter([0], [0], [0], color=FG, s=70,
                                             marker="x", linewidths=2, depthshade=False)

        # Joint labels (4 text objects)
        self._txt_joints = [
            self.ax3d.text(0, 0, 0, lbl, color=col, fontsize=6, fontweight="bold")
            for lbl, col in [("SH", FG), ("EL", FG), ("WR", "#00cc66"), ("TIP", "#ffaa00")]
        ]
        self._txt_status = self.ax3d.text(0, 0, 520, "IK ZERO — awaiting tracking",
                                           color=FG_DIM, fontsize=7, ha="center")

        # View angle buttons
        ctrl = tk.Frame(frame, bg=BG2, pady=3)
        ctrl.pack(fill="x", padx=4)
        tk.Label(ctrl, text="VIEW:", bg=BG2, fg=FG_DIM,
                 font=FONT_TINY).pack(side="left")
        for label, elev, azim in [("FRONT", 10, 0), ("SIDE", 10, 90),
                                   ("TOP", 89, 0), ("ISO", 25, -60)]:
            tk.Button(ctrl, text=label, bg=BG2, fg=FG_DIM,
                      font=FONT_TINY, relief="flat", cursor="hand2",
                      command=lambda e=elev, a=azim: self._set_view(e, a)
                      ).pack(side="left", padx=2)

        # Draw idle pose immediately
        self._draw_idle_pose()

    def _style_3d_axes(self):
        ax = self.ax3d
        ax.set_xlabel("X", color=FG_DIM, fontsize=7, labelpad=-8)
        ax.set_ylabel("Y", color=FG_DIM, fontsize=7, labelpad=-8)
        ax.set_zlabel("Z", color=FG_DIM, fontsize=7, labelpad=-8)
        ax.tick_params(colors=FG_DIM, labelsize=6, pad=-4)
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor("#1a2a1a")
        ax.xaxis.line.set_color("#1a2a1a")
        ax.yaxis.line.set_color("#1a2a1a")
        ax.zaxis.line.set_color("#1a2a1a")
        ax.set_xlim(-400, 400)
        ax.set_ylim(-400, 400)
        ax.set_zlim(   0, 500)
        ax.view_init(elev=25, azim=-60)

    def _set_view(self, elev, azim):
        self.ax3d.view_init(elev=elev, azim=azim)
        self._3d_canvas.draw_idle()

    def _draw_idle_pose(self):
        """Set persistent artists to IK-zero pose on startup."""
        geom = arm_geometry(0, 0, 0, 0, 183.5, 191.4)
        self._set_arm_artists(geom, None, None, None, status="IK ZERO")
        self._3d_canvas.draw_idle()

    # ═══════════════════════════════════════════════════════
    # §4.3 REF MARKER ALIGNMENT OVERLAY
    # ═══════════════════════════════════════════════════════

    def _confirm_alignment(self, event=None):
        """SPACE key handler — confirm alignment and enter normal operation."""
        if self._align_state != "ALIGNING":
            return
        self._align_state = "READY"
        self.root.after(0, self._log,
                        "Ref marker alignment confirmed — tracking active.", "ok")

    def _draw_alignment_overlay(self, display, poses):
        """
        §4.3 — Draw ref marker alignment guide on camera frame.
        Shows target crosshair, ghost outline, position/rotation errors,
        directional arrows, and ALIGNED indicator.
        Called every frame while _align_state == 'ALIGNING'.
        """
        h, w = display.shape[:2]
        cx_target = w // 2          # target ref marker centre (screen centre)
        cy_target = h // 2 + 40    # slightly below centre — natural table position

        # Semi-transparent dark banner at top
        cv2.rectangle(display, (0, 0), (w, 80), (0, 0, 0), -1)
        cv2.putText(display, "ALIGN REF MARKER (ID 10) — PRESS SPACE TO CONFIRM",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 100), 1)

        # Ghost outline — expected ref marker size and position
        # Draw a square showing where the marker should sit
        ghost_half = 55   # pixels, approximate expected ref marker on screen
        ghost_col = (80, 80, 80)
        pts = np.array([
            [cx_target - ghost_half, cy_target - ghost_half],
            [cx_target + ghost_half, cy_target - ghost_half],
            [cx_target + ghost_half, cy_target + ghost_half],
            [cx_target - ghost_half, cy_target + ghost_half],
        ], dtype=np.int32)
        cv2.polylines(display, [pts], True, ghost_col, 1, cv2.LINE_AA)

        # Target crosshair (always shown)
        cv2.line(display, (cx_target - 20, cy_target), (cx_target + 20, cy_target),
                 ghost_col, 1, cv2.LINE_AA)
        cv2.line(display, (cx_target, cy_target - 20), (cx_target, cy_target + 20),
                 ghost_col, 1, cv2.LINE_AA)

        if REFERENCE_ID not in poses:
            cv2.putText(display, "Ref marker not detected — show ID 10 to camera",
                        (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 255), 1)
            self._align_ok      = False
            self._align_pos_err = None
            self._align_rot_err = None
            return

        # Ref marker detected — compute centre and errors
        corners = poses[REFERENCE_ID]["corners"][0]   # (4, 2)
        mx = int(corners[:, 0].mean())
        my = int(corners[:, 1].mean())

        # Position error in pixels, convert to approximate mm
        # At typical table distance (~500mm), 1px ≈ 0.5mm for C920 at 640×480
        PIX_PER_MM = 1.6   # rough estimate; good enough for alignment guide
        pos_err_px = float(np.sqrt((mx - cx_target)**2 + (my - cy_target)**2))
        pos_err_mm = pos_err_px / PIX_PER_MM

        # Rotation error — angle of marker top edge from horizontal
        top_edge = corners[1] - corners[0]   # corner 0→1 = top edge
        rot_err_deg = float(abs(np.degrees(np.arctan2(top_edge[1], top_edge[0]))))
        if rot_err_deg > 90:
            rot_err_deg = 180 - rot_err_deg

        self._align_pos_err = pos_err_mm
        self._align_rot_err = rot_err_deg
        self._align_ok = (pos_err_mm < 5.0 and rot_err_deg < 3.0)

        # Draw detected marker centre
        cv2.circle(display, (mx, my), 6, (255, 255, 0), -1)

        # Directional arrow from detected centre toward target
        if pos_err_px > 12:
            dx = cx_target - mx
            dy = cy_target - my
            mag = max(float(np.sqrt(dx*dx + dy*dy)), 1)
            ax_end = int(mx + dx / mag * 40)
            ay_end = int(my + dy / mag * 40)
            cv2.arrowedLine(display, (mx, my), (ax_end, ay_end),
                            (0, 200, 255), 2, tipLength=0.4)

        # Rotation arrow if misaligned rotationally
        if rot_err_deg > 3.0:
            label_rot = f"ROTATE {rot_err_deg:.0f}deg"
            cv2.putText(display, label_rot, (mx + 10, my + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 180, 255), 1)

        # Error readouts
        pos_col = (0, 255, 136) if pos_err_mm < 5.0 else (0, 180, 255)
        rot_col = (0, 255, 136) if rot_err_deg < 3.0 else (0, 180, 255)
        cv2.putText(display, f"pos err: {pos_err_mm:.1f}mm",
                    (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, pos_col, 1)
        cv2.putText(display, f"rot err: {rot_err_deg:.1f}deg",
                    (170, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.45, rot_col, 1)

        # ALIGNED banner
        if self._align_ok:
            cv2.rectangle(display, (w//2 - 100, cy_target - 80),
                          (w//2 + 100, cy_target - 55), (0, 60, 0), -1)
            cv2.putText(display, "ALIGNED — SPACE to confirm",
                        (w//2 - 98, cy_target - 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 136), 1)
        else:
            cv2.putText(display, "SPACE to skip alignment",
                        (w - 200, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                        0.38, (80, 80, 80), 1)

    def _set_arm_artists(self, geom, tx, ty, tz, status=None):
        """Update all persistent 3D artists in-place — no cla(), no artist recreation."""
        pts = [geom.base, geom.shoulder, geom.elbow, geom.wrist, geom.tip]

        def _seg(ln, a, b):
            ln.set_data([a[0], b[0]], [a[1], b[1]])
            ln.set_3d_properties([a[2], b[2]])

        _seg(self._ln_mount, pts[0], pts[1])
        _seg(self._ln_upper, pts[1], pts[2])
        _seg(self._ln_fore,  pts[2], pts[3])
        _seg(self._ln_tool,  pts[3], pts[4])

        # Joint scatter dots
        for sc, pt in zip(self._sc_joints, pts):
            sc._offsets3d = ([pt[0]], [pt[1]], [pt[2]])

        # Crosshair
        if tx is not None:
            cr = 28
            self._ln_cx.set_data([tx-cr, tx+cr], [ty,    ty   ]); self._ln_cx.set_3d_properties([tz,    tz   ])
            self._ln_cy.set_data([tx,    tx   ], [ty-cr, ty+cr]); self._ln_cy.set_3d_properties([tz,    tz   ])
            self._ln_cz.set_data([tx,    tx   ], [ty,    ty   ]); self._ln_cz.set_3d_properties([tz-cr, tz+cr])
            self._ln_cdrop.set_data([tx, tx], [ty, ty]);          self._ln_cdrop.set_3d_properties([0, tz])
            self._sc_target._offsets3d = ([tx], [ty], [tz])
            for ln in (self._ln_cx, self._ln_cy, self._ln_cz, self._ln_cdrop):
                ln.set_visible(True)
            self._sc_target.set_visible(True)
        else:
            for ln in (self._ln_cx, self._ln_cy, self._ln_cz, self._ln_cdrop):
                ln.set_visible(False)
            self._sc_target.set_visible(False)

        # Joint labels
        label_pts = [pts[1], pts[2], pts[3], pts[4]]
        for txt, pt in zip(self._txt_joints, label_pts):
            txt.set_position((pt[0]+14, pt[1]+14))
            txt.set_3d_properties(pt[2]+14, zdir=None)

        # Status text
        if status is not None:
            self._txt_status.set_text(status)

    def _update_3d_plot(self, result: IKResult, target_xyz):
        """Update 3D arm preview using persistent artists — no cla() rebuild."""
        tx, ty, tz = target_xyz
        geom = arm_geometry(
            result.base_delta, result.shoulder_delta,
            result.elbow_delta, tx, ty, tz)

        cross_col = FG if result.reachable else FG_ERR
        for ln in (self._ln_cx, self._ln_cy, self._ln_cz, self._ln_cdrop):
            ln.set_color(cross_col)
        self._sc_target.set_facecolor(cross_col)
        self._sc_target.set_edgecolor(cross_col)

        self._set_arm_artists(geom, tx, ty, tz, status=None)
        self._3d_canvas.draw_idle()

    def _build_control_panel(self, parent):
        frame = tk.LabelFrame(parent, text=" IK SOLUTION ",
                               bg=BG2, fg=FG, font=FONT_HEAD,
                               relief="solid", bd=1)
        frame.pack(fill="x", pady=4)

        inner = tk.Frame(frame, bg=BG2, padx=10, pady=8)
        inner.pack(fill="x")

        # Hand position XYZ
        tk.Label(inner, text="HAND POSITION (mm)", bg=BG2, fg=FG_DIM,
                 font=FONT_TINY).pack(anchor="w")
        self.pos_var = tk.StringVar(value="X:─  Y:─  Z:─")
        tk.Label(inner, textvariable=self.pos_var,
                 bg=BG2, fg=FG, font=FONT_MONO).pack(anchor="w")

        tk.Frame(inner, bg=FG_DIM, height=1).pack(fill="x", pady=5)

        # Joint deltas
        tk.Label(inner, text="JOINT DELTAS", bg=BG2, fg=FG_DIM,
                 font=FONT_TINY).pack(anchor="w")

        JCOLORS = {"BASE": "#ff6666", "SHOULDER": "#66ff88",
                   "ELBOW": "#6699ff", "WRIST": "#ffaa44"}
        self._joint_vars = {}
        for name, color in JCOLORS.items():
            row = tk.Frame(inner, bg=BG2)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=name, bg=BG2, fg=color,
                     font=FONT_TINY, width=9, anchor="w").pack(side="left")
            v = tk.StringVar(value="─")
            self._joint_vars[name] = v
            tk.Label(row, textvariable=v, bg=BG2, fg=color,
                     font=("Courier New", 11, "bold"),
                     width=10, anchor="e").pack(side="left")

        tk.Frame(inner, bg=FG_DIM, height=1).pack(fill="x", pady=5)

        # Gripper indicator
        tk.Label(inner, text="GRIPPER (PINCH)", bg=BG2, fg=FG_DIM,
                 font=FONT_TINY).pack(anchor="w")
        self.pinch_var = tk.StringVar(value="─")
        self.pinch_lbl = tk.Label(inner, textvariable=self.pinch_var,
                                   bg=BG2, fg="#ffaa44",
                                   font=("Courier New", 13, "bold"))
        self.pinch_lbl.pack(anchor="w")
        self.pinch_dist_var = tk.StringVar(value="dist: ─")
        tk.Label(inner, textvariable=self.pinch_dist_var,
                 bg=BG2, fg=FG_DIM, font=FONT_TINY).pack(anchor="w")

        tk.Frame(inner, bg=FG_DIM, height=1).pack(fill="x", pady=5)

        # Reachability + suppress reason
        self.reach_var = tk.StringVar(value="─")
        self.reach_lbl = tk.Label(inner, textvariable=self.reach_var,
                                   bg=BG2, fg=FG, font=FONT_TINY)
        self.reach_lbl.pack(anchor="w")

        self.suppress_var = tk.StringVar(value="")
        self.suppress_lbl = tk.Label(inner, textvariable=self.suppress_var,
                                      bg=BG2, fg=FG_ERR, font=FONT_TINY,
                                      wraplength=330, justify="left")
        self.suppress_lbl.pack(anchor="w")

        tk.Frame(inner, bg=FG_DIM, height=1).pack(fill="x", pady=5)

        # Speed display
        tk.Label(inner, text="HAND SPEED (3D)", bg=BG2, fg=FG_DIM,
                 font=FONT_TINY).pack(anchor="w")
        self.speed_var = tk.StringVar(value="─")
        self.speed_lbl = tk.Label(inner, textvariable=self.speed_var,
                                   bg=BG2, fg=FG,
                                   font=("Courier New", 16, "bold"))
        self.speed_lbl.pack(anchor="w")

        self.speed_bar = tk.Canvas(inner, bg="#1a2a1a", height=8,
                                    bd=0, highlightthickness=0)
        self.speed_bar.pack(fill="x", pady=(2, 8))

        tk.Frame(inner, bg=FG_DIM, height=1).pack(fill="x", pady=4)

        # ── Hand calibration ──────────────────────────────────────
        tk.Label(inner, text="HAND CALIBRATION", bg=BG2, fg=FG_DIM,
                 font=FONT_TINY).pack(anchor="w")

        self.calib_status_var = tk.StringVar(value="─  not calibrated")
        self.calib_status_lbl = tk.Label(inner, textvariable=self.calib_status_var,
                                          bg=BG2, fg=FG_DIM,
                                          font=("Courier New", 10, "bold"))
        self.calib_status_lbl.pack(anchor="w")

        self.calib_scale_var = tk.StringVar(value="scale  1.00 × 1.00 × 1.00")
        tk.Label(inner, textvariable=self.calib_scale_var,
                 bg=BG2, fg=FG_DIM, font=FONT_TINY).pack(anchor="w", pady=(0, 4))

        calib_btns = tk.Frame(inner, bg=BG2)
        calib_btns.pack(fill="x")

        self.calib_start_btn = tk.Button(
            calib_btns, text="START CALIB",
            bg="#002200", fg=FG, font=FONT_TINY,
            relief="flat", padx=6, pady=3, cursor="hand2",
            command=self._calib_start)
        self.calib_start_btn.pack(side="left", padx=(0, 4))

        self.calib_cap_btn = tk.Button(
            calib_btns, text="CAPTURE",
            bg="#1a1a1a", fg=FG_DIM, font=FONT_TINY,
            relief="flat", padx=6, pady=3, cursor="hand2",
            state="disabled",
            command=self._calib_capture)
        self.calib_cap_btn.pack(side="left", padx=(0, 4))

        tk.Button(calib_btns, text="RESET",
                  bg="#1a1a1a", fg=FG_DIM, font=FONT_TINY,
                  relief="flat", padx=6, pady=3, cursor="hand2",
                  command=self._calib_reset).pack(side="left")

        tk.Button(calib_btns, text="SAVE CFG",
                  bg="#001a1a", fg=FG_DIM, font=FONT_TINY,
                  relief="flat", padx=6, pady=3, cursor="hand2",
                  command=self._save_config).pack(side="left", padx=(6, 0))

        tk.Frame(inner, bg=FG_DIM, height=1).pack(fill="x", pady=4)

        # ── Verification points ───────────────────────────────────
        tk.Label(inner, text="VERIFICATION POINTS", bg=BG2, fg=FG_DIM,
                 font=FONT_TINY).pack(anchor="w")
        vrow = tk.Frame(inner, bg=BG2)
        vrow.pack(fill="x", pady=(2, 0))
        _vpts = [
            ("IK ZERO",  "B:0 S:0 E:0 W:0 G:0"),
            ("MAX FWD",  "B:0 S:1000 E:-306 W:-18 G:0"),
            ("MAX HIGH", "B:0 S:0 E:-317 W:-101 G:0"),
        ]
        for label, cmd in _vpts:
            tk.Button(vrow, text=label,
                      bg="#0a1a0a", fg="#88ff88", font=FONT_TINY,
                      relief="flat", padx=6, pady=3, cursor="hand2",
                      command=lambda c=cmd: self._send_verification(c)
                      ).pack(side="left", padx=(0, 4))

        tk.Frame(inner, bg=FG_DIM, height=1).pack(fill="x", pady=4)

        # ── Config fields — REF_OFFSET + workspace bounds ─────────
        cfg_hdr = tk.Frame(inner, bg=BG2)
        cfg_hdr.pack(fill="x")
        tk.Label(cfg_hdr, text="CONFIG", bg=BG2, fg=FG_DIM,
                 font=FONT_TINY).pack(side="left")
        self._cfg_visible = False
        self._cfg_toggle_btn = tk.Button(
            cfg_hdr, text="▶ SHOW",
            bg=BG2, fg=FG_DIM, font=FONT_TINY,
            relief="flat", padx=4, pady=1, cursor="hand2",
            command=self._toggle_config_panel)
        self._cfg_toggle_btn.pack(side="left", padx=(4, 0))

        self._cfg_frame = tk.Frame(inner, bg=BG2)
        # (hidden until toggled)

        _cfg_fields = [
            ("REF X mm",  "REF_OFFSET_X_MM"),
            ("REF Y mm",  "REF_OFFSET_Y_MM"),
            ("REF Z mm",  "REF_OFFSET_Z_MM"),
            ("WS X min",  "WS_X_MIN"),
            ("WS X max",  "WS_X_MAX"),
            ("WS Y min",  "WS_Y_MIN"),
            ("WS Y max",  "WS_Y_MAX"),
            ("WS Z min",  "WS_Z_MIN"),
            ("WS Z max",  "WS_Z_MAX"),
        ]
        self._cfg_vars = {}   # name → StringVar
        for label, key in _cfg_fields:
            row = tk.Frame(self._cfg_frame, bg=BG2)
            row.pack(fill="x", pady=1)
            tk.Label(row, text=f"{label}:", bg=BG2, fg=FG_DIM,
                     font=FONT_TINY, width=10, anchor="e").pack(side="left")
            var = tk.StringVar(value=str(globals()[key]))
            self._cfg_vars[key] = var
            tk.Entry(row, textvariable=var,
                     bg=BG3, fg=FG, font=FONT_TINY,
                     insertbackground=FG, relief="flat",
                     width=10).pack(side="left", padx=(4, 0))

        tk.Button(self._cfg_frame, text="APPLY",
                  bg="#001a1a", fg=FG_DIM, font=FONT_TINY,
                  relief="flat", padx=8, pady=3, cursor="hand2",
                  command=self._apply_config_fields).pack(anchor="w", pady=(4, 2))

        tk.Frame(inner, bg=FG_DIM, height=1).pack(fill="x", pady=4)

        # Stats
        self.cmds_var = tk.StringVar(value="Commands sent: 0")
        tk.Label(inner, textvariable=self.cmds_var,
                 bg=BG2, fg=FG_DIM, font=FONT_TINY).pack(anchor="w")
        self.last_cmd_var = tk.StringVar(value="Last TX: ─")
        tk.Label(inner, textvariable=self.last_cmd_var,
                 bg=BG2, fg=FG_TX, font=FONT_TINY).pack(anchor="w")

    # ═══════════════════════════════════════════════════════
    # PHASE 5 — MANUAL JOG PANEL
    # ═══════════════════════════════════════════════════════

    def _build_jog_panel(self, parent):
        """XYZ sliders, jog buttons, presets, and SEND — Phase 5 manual control."""
        frame = tk.LabelFrame(parent, text=" MANUAL JOG ",
                               bg=BG2, fg=FG, font=FONT_HEAD,
                               relief="solid", bd=1)
        frame.pack(fill="x", pady=4)
        inner = tk.Frame(frame, bg=BG2, padx=10, pady=8)
        inner.pack(fill="x")

        # ── Manual mode toggle ────────────────────────────────
        hdr = tk.Frame(inner, bg=BG2)
        hdr.pack(fill="x", pady=(0, 6))
        self._manual_btn = tk.Button(
            hdr, text="● MANUAL OFF",
            bg="#1a0000", fg=FG_ERR, font=("Courier New", 10, "bold"),
            relief="flat", padx=10, pady=4, cursor="hand2",
            command=self._toggle_manual_mode)
        self._manual_btn.pack(side="left")
        tk.Label(hdr, text="  vision sends suppressed when ON",
                 bg=BG2, fg=FG_DIM, font=FONT_TINY).pack(side="left")

        tk.Frame(inner, bg=FG_DIM, height=1).pack(fill="x", pady=(0, 6))

        # ── XYZ numeric entries ───────────────────────────────
        self._jog_vars = {}
        xyz_defaults = {"X": 0.0, "Y": 230.5, "Z": 191.4}
        xyz_ranges    = {"X": (-300, 300), "Y": (80, 425), "Z": (20, 415)}
        for axis in ("X", "Y", "Z"):
            color = JOG_AXIS_COLORS[axis]
            lo, hi = xyz_ranges[axis]
            var = tk.DoubleVar(value=xyz_defaults[axis])
            self._jog_vars[axis] = var

            row = tk.Frame(inner, bg=BG2, pady=3)
            row.pack(fill="x")

            tk.Label(row, text=axis, bg=BG2, fg=color,
                     font=("Courier New", 20, "bold"), width=2).pack(side="left")
            ent = tk.Entry(row, textvariable=var, width=7,
                           bg=BG3, fg=color, insertbackground=color,
                           font=("Courier New", 13, "bold"),
                           relief="flat", justify="right")
            ent.pack(side="left", padx=4)
            ent.bind("<Return>",   lambda e, a=axis: self._jog_entry_changed(a))
            ent.bind("<FocusOut>", lambda e, a=axis: self._jog_entry_changed(a))
            tk.Label(row, text="mm", bg=BG2, fg=FG_DIM,
                     font=FONT_TINY).pack(side="left")

            tk.Label(row, text=f"{lo}", bg=BG2, fg=FG_DIM,
                     font=FONT_TINY).pack(side="left", padx=(8, 2))
            slider = tk.Scale(row, variable=var,
                              from_=lo, to=hi, resolution=0.5,
                              orient="horizontal", length=180,
                              bg=BG2, fg=color, troughcolor="#1a2a1a",
                              highlightthickness=0, sliderrelief="flat",
                              showvalue=False,
                              command=lambda v, a=axis: self._jog_slider_changed(a))
            slider.pack(side="left", padx=2)
            tk.Label(row, text=f"{hi}", bg=BG2, fg=FG_DIM,
                     font=FONT_TINY).pack(side="left", padx=(2, 0))

        tk.Frame(inner, bg=FG_DIM, height=1).pack(fill="x", pady=6)

        # ── Jog step size ─────────────────────────────────────
        step_row = tk.Frame(inner, bg=BG2)
        step_row.pack(fill="x", pady=(0, 4))
        tk.Label(step_row, text="STEP:", bg=BG2, fg=FG_DIM,
                 font=FONT_TINY).pack(side="left")
        self._jog_step = tk.IntVar(value=10)
        for sz in JOG_SIZES:
            tk.Radiobutton(step_row, text=f"{sz}mm", variable=self._jog_step, value=sz,
                           bg=BG2, fg=FG, selectcolor=BG3,
                           activebackground=BG2, font=FONT_TINY
                           ).pack(side="left", padx=3)

        # ── Jog buttons ───────────────────────────────────────
        for axis in ("X", "Y", "Z"):
            color = JOG_AXIS_COLORS[axis]
            row = tk.Frame(inner, bg=BG2)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=f"{axis}:", bg=BG2, fg=color,
                     font=("Courier New", 10, "bold"), width=3).pack(side="left")
            tk.Button(row, text=f"◀ -{axis}",
                      bg="#002200", fg=color, font=FONT_TINY, relief="flat",
                      padx=10, pady=3, cursor="hand2",
                      command=lambda a=axis: self._jog(a, -1)
                      ).pack(side="left", padx=(0, 4))
            tk.Button(row, text=f"+{axis} ▶",
                      bg="#002200", fg=color, font=FONT_TINY, relief="flat",
                      padx=10, pady=3, cursor="hand2",
                      command=lambda a=axis: self._jog(a, +1)
                      ).pack(side="left")

        tk.Frame(inner, bg=FG_DIM, height=1).pack(fill="x", pady=6)

        # ── Presets ───────────────────────────────────────────
        tk.Label(inner, text="PRESETS", bg=BG2, fg=FG_DIM,
                 font=FONT_TINY).pack(anchor="w")
        preset_grid = tk.Frame(inner, bg=BG2)
        preset_grid.pack(fill="x", pady=(2, 6))
        for idx, (label, px, py, pz) in enumerate(JOG_PRESETS):
            col = idx % 3
            row_n = idx // 3
            tk.Button(preset_grid, text=label,
                      bg="#002200", fg=FG, font=FONT_TINY,
                      relief="flat", padx=4, pady=4, cursor="hand2",
                      command=lambda x=px, y=py, z=pz: self._jog_goto(x, y, z)
                      ).grid(row=row_n, column=col, padx=3, pady=2, sticky="ew")
        for c in range(3):
            preset_grid.columnconfigure(c, weight=1)

        # ── IK readout + SEND ─────────────────────────────────
        self._jog_reach_var = tk.StringVar(value="─")
        self._jog_reach_lbl = tk.Label(inner, textvariable=self._jog_reach_var,
                                        bg=BG2, fg=FG_DIM,
                                        font=("Courier New", 10, "bold"))
        self._jog_reach_lbl.pack(anchor="w")

        self._jog_fk_err_var = tk.StringVar(value="FK err: ─")
        tk.Label(inner, textvariable=self._jog_fk_err_var,
                 bg=BG2, fg=FG_DIM, font=FONT_TINY).pack(anchor="w")

        self._jog_serial_var = tk.StringVar(value="─")
        tk.Label(inner, textvariable=self._jog_serial_var,
                 bg=BG2, fg=FG_TX, font=FONT_TINY).pack(anchor="w", pady=(0, 6))

        self._jog_send_btn = tk.Button(
            inner, text="▶  SEND TO ARM",
            bg="#004400", fg=FG, font=("Courier New", 11, "bold"),
            relief="flat", padx=16, pady=6, cursor="hand2",
            command=self._jog_send)
        self._jog_send_btn.pack(fill="x")

        # Initial solve — wrapped so a solver error never blocks GUI startup
        try:
            self._jog_solve()
        except Exception as _e:
            print(f"[jog] initial solve skipped: {_e}")

    def _toggle_manual_mode(self):
        self._manual_mode = not self._manual_mode
        if self._manual_mode:
            self._manual_btn.configure(text="● MANUAL ON",
                                        bg="#002200", fg=FG)
            self._log("Manual jog mode ON — vision tracking suppressed.", "warn")
        else:
            self._manual_btn.configure(text="● MANUAL OFF",
                                        bg="#1a0000", fg=FG_ERR)
            self._log("Manual jog mode OFF — vision tracking resumed.", "ok")

    def _jog_entry_changed(self, axis):
        try:
            self._jog_solve()
        except Exception:
            pass

    def _jog_slider_changed(self, axis):
        self._jog_solve()

    def _jog(self, axis: str, direction: int):
        step = self._jog_step.get() * direction
        ranges = {"X": (-300, 300), "Y": (80, 425), "Z": (20, 415)}
        lo, hi = ranges[axis]
        var = self._jog_vars[axis]
        var.set(round(max(lo, min(hi, var.get() + step)), 1))
        self._jog_solve()

    def _jog_goto(self, x, y, z):
        self._jog_vars["X"].set(x)
        self._jog_vars["Y"].set(y)
        self._jog_vars["Z"].set(z)
        self._jog_solve()

    def _jog_solve(self):
        """Solve IK for current jog XYZ and update readout widgets."""
        try:
            x = float(self._jog_vars["X"].get())
            y = float(self._jog_vars["Y"].get())
            z = float(self._jog_vars["Z"].get())
        except (ValueError, tk.TclError):
            return

        result = solve(x, y, z)
        self._jog_result = result

        if result.reachable:
            self._jog_reach_var.set("● REACHABLE")
            self._jog_reach_lbl.configure(fg=FG)
        else:
            self._jog_reach_var.set("● OUT OF RANGE")
            self._jog_reach_lbl.configure(fg=FG_ERR)

        # FK round-trip error
        try:
            fk = forward(result.base_delta, result.shoulder_delta, result.elbow_delta)
            err = math.sqrt((fk.x-x)**2 + (fk.y-y)**2 + (fk.z-z)**2)
            col = FG if err < 0.5 else FG_WARN
            self._jog_fk_err_var.set(f"FK err: {err:.3f} mm {'✓' if err < 0.5 else '⚠'}")
            # (no label configure needed — static label)
        except Exception:
            self._jog_fk_err_var.set("FK err: ─")

        self._jog_serial_var.set(result.serial_string())

        # Update shared 3D preview and IK panel if in manual mode
        if self._manual_mode:
            self.last_result = result
            self.current_pos = (x, y, z)

    def _jog_send(self):
        """Send current jog IK result to the arm immediately."""
        if not self._manual_mode:
            self._log("Manual mode is OFF — enable it before sending jog commands.", "warn")
            return
        if not self.connected:
            self._log("Not connected — cannot send jog command.", "err")
            return
        if self._jog_result is None:
            return
        if not self._jog_result.reachable:
            self._log("Jog position out of range — not sent.", "warn")
            return

        # No jerk limit here — manual presets are intentional large moves.
        # The firmware handles them with full trapezoidal ramping.
        # Jerk limiting only applies to 10Hz vision streaming (in _update_tracking).
        g = self._pinch.gripper_pwm if self._pinch else 900
        cmd = f"{self._jog_result.serial_string()} G:{g}"
        if self.serial.send(cmd):
            self._log(f"JOG → {cmd}", "ok")
            self._monitor_line(f"→ JOG {cmd}", "tx")
            now = time.time()
            self.last_send_t = now   # reset watchdog clock from jog send

            # Estimate move duration so keepalive doesn't interrupt the ramp.
            # Two components:
            #   1. Pure pacer: max(delta * motorCap) ticks * 100us  (all steps at max speed)
            #   2. Ramp overhead: the trapezoid ramp means steps near start/end are slower
            #      than max speed, adding real time beyond the pure pacer estimate.
            #      Overhead = ramp_steps * (MIN_SPEED_TICKS - pacer_maxTicks) * 100us
            #      (covers both accel and decel ramps combined)
            # Without ramp overhead: keepalive fires ~2s early and resets ramp mid-decel
            # causing the motor to re-accelerate to full speed then slam into target.
            motor_caps    = [50, 80, 50, 50]   # B, S, E, W — must match firmware
            MIN_SPEED_T   = 150                 # firmware MIN_SPEED_TICKS
            ACCEL_S       = 300                 # firmware ACCEL_STEPS

            new_steps = (self._jog_result.base_steps, self._jog_result.shoulder_steps,
                         self._jog_result.elbow_steps, self._jog_result.wrist_steps)
            if self.last_steps is not None:
                deltas = [abs(new_steps[i] - self.last_steps[i]) for i in range(4)]
            else:
                deltas = [abs(s) for s in new_steps]

            # Find pacer motor (longest duration)
            durations   = [deltas[i] * motor_caps[i] for i in range(4)]
            pacer       = max(durations)
            pacer_idx   = durations.index(pacer)
            pacer_delta = deltas[pacer_idx]
            pacer_cap   = motor_caps[pacer_idx]

            # Ramp overhead for pacer motor
            ramp_steps    = min(ACCEL_S, pacer_delta // 2) if pacer_delta > 0 else 0
            ramp_overhead = ramp_steps * (MIN_SPEED_T - pacer_cap) * 100e-6

            move_secs = pacer * 100e-6 + ramp_overhead + 0.5   # +0.5s safety margin
            self._preset_move_until = now + move_secs
            self._log(f"  (move ~{pacer*100e-6:.1f}s + {ramp_overhead:.1f}s ramp, "
                      f"keepalive held {move_secs:.1f}s)", "info")

            # Update last_steps so vision jerk gate is correct when tracking resumes
            self.last_steps = new_steps
        else:
            self._log("Jog send failed.", "err")

    def _toggle_config_panel(self):
        """Show/hide the REF_OFFSET + workspace config entry fields."""
        self._cfg_visible = not self._cfg_visible
        if self._cfg_visible:
            self._cfg_frame.pack(fill="x", pady=(2, 0))
            self._cfg_toggle_btn.configure(text="▼ HIDE")
        else:
            self._cfg_frame.pack_forget()
            self._cfg_toggle_btn.configure(text="▶ SHOW")

    def _apply_config_fields(self):
        """Parse entry fields and update module-level globals live.
        Invalid (non-float) values are highlighted red and skipped.
        On success logs applied values; caller should press SAVE CFG to persist.
        """
        global REF_OFFSET_X_MM, REF_OFFSET_Y_MM, REF_OFFSET_Z_MM
        global WS_X_MIN, WS_X_MAX, WS_Y_MIN, WS_Y_MAX, WS_Z_MIN, WS_Z_MAX
        name_to_global = {
            "REF_OFFSET_X_MM": "REF_OFFSET_X_MM",
            "REF_OFFSET_Y_MM": "REF_OFFSET_Y_MM",
            "REF_OFFSET_Z_MM": "REF_OFFSET_Z_MM",
            "WS_X_MIN": "WS_X_MIN", "WS_X_MAX": "WS_X_MAX",
            "WS_Y_MIN": "WS_Y_MIN", "WS_Y_MAX": "WS_Y_MAX",
            "WS_Z_MIN": "WS_Z_MIN", "WS_Z_MAX": "WS_Z_MAX",
        }
        errors = []
        applied = []
        g = globals()
        for key, var in self._cfg_vars.items():
            try:
                val = float(var.get())
                g[key] = val
                applied.append(f"{key}={val:.1f}")
            except ValueError:
                errors.append(key)
        if applied:
            self._log("Config applied (not saved): " + "  ".join(applied), "ok")
            self._log("Press SAVE CFG to persist to aruco_config.json.", "info")
        if errors:
            self._log(f"Invalid values (not applied): {', '.join(errors)}", "warn")

    def _send_verification(self, cmd):
        """Send a raw verification command directly, bypassing IK and state machine."""
        if not self._manual_mode:
            self._log("Manual mode is OFF — enable it before sending verification commands.", "warn")
            return
        if not self.connected:
            self._log("Not connected.", "err")
            return
        if self.serial.send(cmd):
            self._log(f"VERIFY: {cmd}", "info")
            self.root.after(0, self._update_stats, cmd)

    def _build_monitor_panel(self, parent):
        frame = tk.LabelFrame(parent, text=" SERIAL MONITOR ",
                               bg=BG2, fg=FG, font=FONT_HEAD,
                               relief="solid", bd=1)
        frame.pack(fill="both", expand=True, pady=4)

        self.monitor = tk.Text(
            frame, bg=BG3, fg=FG,
            font=FONT_TINY, relief="flat",
            wrap="none", state="disabled")
        self.monitor.pack(side="left", fill="both", expand=True)

        sb = tk.Scrollbar(frame, command=self.monitor.yview, bg=BG2)
        sb.pack(side="right", fill="y")
        self.monitor.configure(yscrollcommand=sb.set)

        self.monitor.tag_configure("rx_ok",   foreground=FG)
        self.monitor.tag_configure("rx_err",  foreground=FG_ERR)
        self.monitor.tag_configure("rx_warn", foreground=FG_WARN)
        self.monitor.tag_configure("rx_info", foreground=FG_DIM)
        self.monitor.tag_configure("tx",      foreground=FG_TX)
        self.monitor.tag_configure("ts",      foreground="#224422")

        self._monitor_lines = 0

        ctrl = tk.Frame(frame, bg=BG2, padx=4, pady=2)
        ctrl.pack(fill="x")
        tk.Button(ctrl, text="Clear", bg=BG2, fg=FG_DIM,
                  font=FONT_TINY, relief="flat", cursor="hand2",
                  command=self._clear_monitor).pack(side="left")

        # Send bar
        send = tk.Frame(frame, bg=BG2, padx=4, pady=4)
        send.pack(fill="x")
        tk.Label(send, text="TX:", bg=BG2, fg=FG_DIM,
                 font=FONT_MONO).pack(side="left")
        self.raw_var = tk.StringVar()
        ent = tk.Entry(send, textvariable=self.raw_var,
                       bg=BG3, fg=FG, insertbackground=FG,
                       font=FONT_MONO, relief="flat")
        ent.pack(side="left", fill="x", expand=True, padx=4)
        ent.bind("<Return>", lambda _: self._send_raw_entry())
        tk.Button(send, text="SEND", bg="#002200", fg=FG,
                  font=FONT_MONO, relief="flat", cursor="hand2",
                  command=self._send_raw_entry).pack(side="right")

    # ═══════════════════════════════════════════════════════
    # CAMERA LOOP
    # ═══════════════════════════════════════════════════════

    def _restart_camera(self):
        """Stop current camera and restart with new index (called by CAM spinbox)."""
        self._cam_running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        self.root.after(200, self._start_camera)  # brief delay for thread to exit

    def _start_camera(self):
        # On Windows use DirectShow backend — avoids silent timeout with default MSMF
        idx = self.cam_index_var.get()
        if platform.system() == "Windows":
            self.cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(idx)

        if not self.cap.isOpened():
            self.root.after(0, lambda: self.cam_label.configure(
                text="Camera index 0 not found", image=""))
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self._cam_running = True
        threading.Thread(target=self._cam_loop, daemon=True).start()

    def _cam_loop(self):
        # Warm-up: DirectShow needs a few grabs before delivering real frames.
        # Done here in the thread so the main thread (and GUI) never blocks.
        for _ in range(5):
            self.cap.read()

        _consecutive_failures = 0
        while self._cam_running:
            _frame_start = time.perf_counter()   # §7.3 deadline timing
            ret, frame = self.cap.read()
            if not ret:
                _consecutive_failures += 1
                if _consecutive_failures == 1:
                    self.root.after(0, lambda: self.cam_label.configure(
                        text="Camera read failed — check connection", image=""))
                time.sleep(0.05)
                continue
            _consecutive_failures = 0

            frame = cv2.flip(frame, 1)
            display = frame.copy()

            # ── Pinch detection (runs on every frame) ─────────────
            # ── Pinch detection throttled to 10Hz (MediaPipe neural net is expensive)
            _now_p = time.perf_counter()
            if _now_p - self._pinch_last_t >= 1.0 / UPDATE_RATE_HZ:
                self._pinch.process(frame)
                self._pinch_last_t = _now_p
            self._pinch.draw_overlay(display)  # always draw last result

            if self._tracker_ok:
                poses = self.tracker.detect(frame)

                # §4.3 — Alignment phase: draw overlay, block hand tracking
                if self._align_state == "ALIGNING":
                    self._draw_alignment_overlay(display, poses)
                    # Still update ref extrinsics so world frame is ready on confirm
                    if REFERENCE_ID in poses:
                        self.tracker.update_extrinsics(poses[REFERENCE_ID])
                    # Skip hand tracking until aligned
                    self._ids_present = list(poses.keys())   # read by _update_displays

                else:
                    # §3.2 — update world transform whenever ref is visible
                    if REFERENCE_ID in poses:
                        self.tracker.update_extrinsics(poses[REFERENCE_ID])

                        # Ref marker drift check — warn if marker moved after calibration
                        if (self._calib_state == "DONE" and
                                self._calib_R_wc is not None and
                                self._calib_t_wc is not None and
                                self.tracker.R_wc is not None):
                            # Rotation drift: Frobenius norm on R_wc change (~8° threshold)
                            r_diff = np.linalg.norm(
                                self.tracker.R_wc - self._calib_R_wc, 'fro')
                            # Translation drift: direct t_wc comparison (20mm threshold)
                            # (Previously t_snap used R@inv(R)@t = -t, which is always
                            # just the current t_wc — completely ignoring calib snapshot)
                            t_diff = np.linalg.norm(
                                self.tracker.t_wc.flatten() - self._calib_t_wc)
                            if r_diff > 0.15 or t_diff > 0.020:   # ~8° rotation OR 20mm shift
                                self.root.after(0, self._log,
                                    "⚠ Ref marker may have moved since calibration — "
                                    "recalibrate for accurate mapping.", "warn")
                                # Re-snap to avoid spamming the warning every frame
                                self._calib_R_wc = self.tracker.R_wc.copy()
                                self._calib_t_wc = self.tracker.t_wc.flatten().copy()

                    # Draw all detected markers
                    for mid, pose in poses.items():
                        cv2.aruco.drawDetectedMarkers(
                            display, [pose["corners"]], np.array([[mid]]))

                    tracking = False
                    if HAND_ID in poses and self.tracker.R_wc is not None:
                        hand_corners = poses[HAND_ID]["corners"][0]
                        self._pinch.update_aruco_ruler(hand_corners)

                        p_world = self.tracker.world_position(poses[HAND_ID])
                        if p_world is not None:
                            xyz_mm = p_world * 1000.0

                            # §6.4 Tilt check
                            rvec = poses[HAND_ID]["rvec"]
                            R_cm, _ = cv2.Rodrigues(rvec)
                            marker_normal = R_cm[:, 2]
                            # Tilt relative to world horizontal (not camera axis).
                            # World up = R_wc @ [0,0,1] — derived from ref marker,
                            # so "flat on table" = 0° regardless of camera angle.
                            world_up = self.tracker.R_wc @ np.array([0.0, 0.0, 1.0])
                            world_up = world_up / np.linalg.norm(world_up)
                            cos_a = float(np.clip(np.dot(marker_normal, world_up), -1, 1))
                            tilt_deg = float(np.degrees(np.arccos(abs(cos_a))))

                            self._update_tracking(xyz_mm, tilt_deg)
                            tracking = True

                            # Crosshair colour by control state
                            if self.control_state == "TRACKING":
                                color = (0, 255, 136)    # green
                            elif self.control_state == "HOLDING":
                                color = (0, 140, 255)    # orange
                            elif self.control_state == "STANDBY":
                                color = (180, 180, 180)  # grey
                            else:  # WAITING
                                color = (80, 80, 80)     # dark grey

                            corners = hand_corners
                            cx = int(corners[:, 0].mean())
                            cy = int(corners[:, 1].mean())
                            cv2.circle(display, (cx, cy), 18, color, 2)
                            cv2.drawMarker(display, (cx, cy), color,
                                           cv2.MARKER_CROSS, 20, 2)

                            cv2.putText(display, self.control_state,
                                        (cx - 30, cy - 24),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                            if tilt_deg > 30.0:
                                cv2.putText(display, f"TILT {tilt_deg:.0f}deg",
                                            (cx - 30, cy - 38),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 80, 255), 1)

                            x, y, z = xyz_mm
                            cv2.putText(display,
                                        f"vis X:{x:+.0f} Y:{y:.0f} Z:{z:.0f}mm",
                                        (cx + 22, cy - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                            ik_x = -(x + REF_OFFSET_X_MM)
                            ik_y =   y + REF_OFFSET_Y_MM
                            ik_z =   z + REF_OFFSET_Z_MM
                            cv2.putText(display,
                                        f"ik  X:{ik_x:+.0f} Y:{ik_y:.0f} Z:{ik_z:.0f}mm",
                                        (cx + 22, cy + 8),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

                    self._ids_present = list(poses.keys())   # read by _update_displays

                    # §6.1 — "ready" means hand visible + world frame known.
                    # Ref marker is only required if R_wc not yet established;
                    # once calibrated, hand alone is sufficient (ref may be
                    # occluded by the hand itself without breaking state).
                    if self.tracker.R_wc is None:
                        hand_ready = (REFERENCE_ID in poses
                                      and HAND_ID in poses)
                    else:
                        hand_ready = (HAND_ID in poses)

                    # §6.1 WAITING → STANDBY transition
                    if self.control_state == "WAITING":
                        if hand_ready:
                            if self._both_visible_since is None:
                                self._both_visible_since = time.time()
                            elif time.time() - self._both_visible_since >= WAITING_VISIBLE_SECS:
                                self.control_state       = "STANDBY"
                                self._both_visible_since = None
                                self._stable_since       = None
                        else:
                            self._both_visible_since = None

                    # Update shared flag
                    self._hand_tracked = tracking

                    if not tracking:
                        # Marker lost — change state but DO NOT clear everything.
                        # The arm holds its last position. Keepalive fires from
                        # _send_keepalive() below to prevent watchdog.
                        # _clear_tracking() is only called on deliberate stop/rehome.
                        self.tracker.reset_filters()
                        if self.control_state in ("TRACKING", "HOLDING", "STANDBY"):
                            self.control_state = "WAITING"
                            self._both_visible_since   = None
                            self._stable_since         = None
                            self._holding_stable_since = None
                        # Send keepalive so watchdog doesn't fire while marker is lost
                        self._send_keepalive(time.time())


            else:
                # Show calibration error
                cv2.putText(display, "CALIB FILE MISSING", (20, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            if self._calib_state in ("WAIT_P1", "WAIT_P2"):
                pt_num = "1" if self._calib_state == "WAIT_P1" else "2"
                cv2.rectangle(display, (0, 0), (640, 50), (0, 0, 0), -1)
                cv2.putText(display,
                            f"CALIBRATION P{pt_num} — move to extreme then press CAPTURE",
                            (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                            (255, 200, 0), 2)
                if self._last_raw_xyz is not None:
                    px, py, pz = self._last_raw_xyz
                    cv2.putText(display,
                                f"X:{px:+.0f} Y:{py:.0f} Z:{pz:.0f} mm",
                                (10, 43), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                                (200, 200, 200), 1)


            # Overlay: enabled/disabled + gripper value
            label = "CONTROL: ENABLED" if self.enabled else "CONTROL: PAUSED"
            col   = (0, 255, 136) if self.enabled else (80, 80, 80)
            cv2.putText(display, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
            if self._pinch.hand_seen:
                cv2.putText(display,
                            f"GRIPPER: {self._pinch.gripper_pwm}µs",
                            (10, 58), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 170, 68), 1)

            # Bottom banner — shows any active suppression reason
            if self.speed_status == "TOO_FAST":
                cv2.rectangle(display, (0, 440), (640, 480), (0, 0, 180), -1)
                cv2.putText(display, f"!!! TOO FAST {self.current_speed:.0f}mm/s — SLOW DOWN !!!",
                            (20, 468), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                            (255, 255, 255), 2)
            elif self.speed_status == "WARNING":
                cv2.rectangle(display, (0, 440), (640, 480), (0, 80, 160), -1)
                cv2.putText(display, f"Moving fast — {self.current_speed:.0f}mm/s",
                            (20, 468), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                            (255, 255, 255), 2)
            elif self.suppress_reason:
                cv2.rectangle(display, (0, 440), (640, 480), (0, 0, 100), -1)
                txt = self.suppress_reason[:60]
                cv2.putText(display, f"SUPPRESSED: {txt}",
                            (10, 468), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (200, 120, 120), 1)

            # Resize for display only — full FOV, smaller render
            rgb   = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            rgb   = cv2.resize(rgb, (480, 360), interpolation=cv2.INTER_LINEAR)
            img   = Image.fromarray(rgb)
            imgtk = ImageTk.PhotoImage(image=img)
            self.root.after(0, self._update_cam_label, imgtk)

            # §7.3 — deadline-based pacing: sleep only the remaining time in
            # the frame budget so detection cost doesn't accumulate as drift.
            _frame_end = time.perf_counter()
            _elapsed   = _frame_end - _frame_start
            _remaining = (1.0 / 30) - _elapsed
            if _remaining > 0:
                time.sleep(_remaining)

    def _update_cam_label(self, imgtk):
        self.cam_label.imgtk = imgtk   # keep reference
        self.cam_label.configure(image=imgtk)

    # ═══════════════════════════════════════════════════════
    # TRACKING + CONTROL
    # ═══════════════════════════════════════════════════════

    def _update_tracking(self, xyz_mm: np.ndarray, marker_tilt_deg: float = 0.0):
        """
        Called every camera frame with filtered hand position in world mm.
        Runs state machine (§6.1–6.2), tilt validity check (§6.4),
        IK solve, safety gates, and sends command if clear.
        """
        t = time.time()   # single timestamp for entire frame — spline + rate gate both use this
        x, y, z = float(xyz_mm[0]), float(xyz_mm[1]), float(xyz_mm[2])

        # ── Translate from vision frame → IK world frame ──────────
        x += REF_OFFSET_X_MM
        y += REF_OFFSET_Y_MM
        z += REF_OFFSET_Z_MM
        x  = -x   # ArUco X is mirrored relative to IKSolver world frame

        # ── Apply hand calibration affine remap ───────────────────
        # Maps raw hand range [P1, P2] → workspace [WS_MIN, WS_MAX]
        # offset = WS_MIN - P1*scale, so: out = raw*scale + offset
        raw_xyz = np.array([x, y, z])
        _in_deadband = (
            self._last_sent_raw_xyz is not None and
            np.linalg.norm(raw_xyz - self._last_sent_raw_xyz) < DEADBAND_MM
        )

        x = x * self._calib_scale[0] + self._calib_offset[0]
        y = y * self._calib_scale[1] + self._calib_offset[1]
        z = z * self._calib_scale[2] + self._calib_offset[2]

        self._last_raw_xyz = np.array([x, y, z])

        # ── §8.1 Cubic spline trajectory smoother ─────────────────────────────────
        # Always buffer positions at full frame rate for a dense sample set.
        # Only fit and evaluate the spline at 10Hz (command rate) to avoid
        # running a scipy matrix solve 30x/second in the camera thread.
        self._traj_buf.append((t, np.array([x, y, z])))
        _rate_due = (t - self.last_send_t >= 1.0 / UPDATE_RATE_HZ)
        if (_rate_due and len(self._traj_buf) >= 4 and
                self._traj_buf[-1][0] - self._traj_buf[0][0] >= TRAJ_MIN_SPAN):
            _ts  = np.array([e[0] for e in self._traj_buf])
            _pts = np.array([e[1] for e in self._traj_buf])  # (N, 3)
            try:
                _cs = CubicSpline(_ts, _pts)
                _smooth = _cs(_ts[-1])   # evaluate at latest time
                x, y, z = float(_smooth[0]), float(_smooth[1]), float(_smooth[2])
            except Exception:
                pass  # fall back to raw if spline fails (degenerate buffer)

        # ── 3D velocity ───────────────────────────────────────────
        self._pos_hist.append((t, xyz_mm.copy()))
        if len(self._pos_hist) >= 2:
            t0, p0 = self._pos_hist[0]
            dt = t - t0
            self.current_speed = float(np.linalg.norm(xyz_mm - p0)) / dt \
                                  if dt > 0.001 else 0.0
        else:
            self.current_speed = 0.0

        # Feed rolling buffer with PRE-scale, STILL-hand frames only.
        # Gated on speed < hold_threshold so transit frames never contaminate
        # the mean — prevents the bias that causes calibration drift.
        if self._calib_state in ("WAIT_P1", "WAIT_P2"):
            if self.current_speed < self._hold_threshold:
                self._calib_rolling.append(raw_xyz.copy())
            else:
                self._calib_rolling.clear()   # hand moved — discard stale frames

        # ── §6.1–6.2 State machine ────────────────────────────────
        hand_still = self.current_speed < self._hold_threshold

        if self.control_state == "WAITING":
            # WAITING→STANDBY is handled in camera loop (both-visible timer)
            # If we reach here still WAITING, suppress all commands
            pass

        elif self.control_state == "STANDBY":
            if hand_still:
                if self._stable_since is None:
                    self._stable_since = t
                elif t - self._stable_since >= self._activation_secs:
                    self.control_state = "TRACKING"
                    self._stable_since = None
                    self._holding_stable_since = None
            else:
                self._stable_since = None   # reset countdown if hand moves

        elif self.control_state == "TRACKING":
            if not hand_still:
                self.control_state = "HOLDING"
                self._holding_stable_since = None

        elif self.control_state == "HOLDING":
            if hand_still:
                if self._holding_stable_since is None:
                    self._holding_stable_since = t
                elif t - self._holding_stable_since >= self._retrack_secs:
                    self.control_state = "TRACKING"
                    self._holding_stable_since = None
            else:
                self._holding_stable_since = None

        # Speed display status (independent of state machine — for overlay)
        if   self.current_speed > MAX_SPEED_3D:  self.speed_status = "TOO_FAST"
        elif self.current_speed > WARNING_SPEED:  self.speed_status = "WARNING"
        else:                                     self.speed_status = "OK"

        # ── Clamp to safe workspace ───────────────────────────────
        x = max(WS_X_MIN, min(WS_X_MAX, x))
        y = max(WS_Y_MIN, min(WS_Y_MAX, y))
        z = max(WS_Z_MIN, min(WS_Z_MAX, z))

        # ── IK solve ──────────────────────────────────────────────
        result = solve(x, y, z)
        self.last_result  = result
        self.current_pos  = (x, y, z)

        # ── Safety gate — collect all suppress reasons ────────────
        reasons = []

        # 0. Manual mode — jog panel has control; suppress vision sends
        if self._manual_mode:
            reasons.append("MANUAL MODE")

        # 1. State machine — only TRACKING state sends commands
        if self.control_state != "TRACKING":
            reasons.append(self.control_state)

        # 2. §6.4 Tilt validity — reject if marker tilted > 30° from horizontal
        if marker_tilt_deg > 30.0:
            reasons.append(f"TILT {marker_tilt_deg:.0f}°")

        # 3. IK reachability
        if not result.reachable:
            reasons.append("IK:" + "; ".join(result.warnings))

        # 4. Jerk limit
        if self.last_steps is not None:
            new_steps = (result.base_steps, result.shoulder_steps,
                         result.elbow_steps, result.wrist_steps)
            names = ("B", "S", "E", "W")
            for nm, prev, cur in zip(names, self.last_steps, new_steps):
                delta = abs(cur - prev)
                if delta > MAX_STEP_JUMP:
                    reasons.append(f"JERK {nm}:{delta}steps")

        self.suppress_reason = " | ".join(reasons)

        # ── Send if clear ─────────────────────────────────────────
        rate_ok = (t - self.last_send_t >= 1.0 / UPDATE_RATE_HZ)   # reuse t from top of frame

        if self.enabled and self.connected and not reasons and rate_ok:
            g = self._pinch.gripper_pwm

            # Gripper deadband — independent of position deadband
            g_changed = (self._last_sent_g is None or
                         abs(g - self._last_sent_g) >= DEADBAND_GRIPPER_US)
            if not g_changed:
                g = self._last_sent_g  # hold last value

            # Speed matching — map hand velocity to V: percentage
            if SPEED_MATCH_ENABLED:
                # Map hand speed [0, HAND_MAX_SPEED_MMS] → V [SPEED_MATCH_MIN, SPEED_MATCH_MAX]
                # Clamp at top so speeds above HAND_MAX_SPEED_MMS still get full V
                ratio = self.current_speed / HAND_MAX_SPEED_MMS
                if ratio > 1.0: ratio = 1.0
                v = int(SPEED_MATCH_MIN + ratio * (SPEED_MATCH_MAX - SPEED_MATCH_MIN))
                if v < SPEED_MATCH_MIN: v = SPEED_MATCH_MIN
                if v > SPEED_MATCH_MAX: v = SPEED_MATCH_MAX
            else:
                v = 100

            # Send if position moved OR gripper changed (or both)
            if not _in_deadband or g_changed:
                b = result.base_steps
                s = result.shoulder_steps
                e = result.elbow_steps
                w = result.wrist_steps

                cmd = f"B:{b} S:{s} E:{e} W:{w} G:{g} V:{v}"
                if self.serial.send(cmd):
                    self.command_count      += 1
                    self.last_send_t         = t
                    self._last_sent_g        = g

                    if not _in_deadband:
                        self.last_steps          = (b, s, e, w)
                        self._last_sent_raw_xyz  = raw_xyz.copy()
                    self._last_cmd_str  = cmd   # display timer reads this at 10Hz
                    self.root.after(0, self._monitor_line, f"→ {cmd}", "tx")   # log to monitor only

        # Keepalive — keep watchdog alive when commands suppressed for any reason
        self._send_keepalive(t)

        # display updates are driven by the 10Hz recurring timer, not per-frame

    def _send_keepalive(self, t):
        """Send last known position to keep firmware watchdog alive.
        Called when commands are suppressed (deadband, HOLDING, TILT, STANDBY,
        unreachable) OR when the hand marker is temporarily lost.
        In both cases Python is alive and connected — the arm should hold position,
        not be stopped by the watchdog.

        Does NOT fire if:
        - not connected or not enabled
        - no last_steps (no real command ever sent)
        - sent too recently (400ms interval, well within 500ms watchdog)
        """
        KEEPALIVE_INTERVAL = 0.4
        if (self.connected and
                self.last_steps is not None and
                t - self.last_send_t >= KEEPALIVE_INTERVAL and
                t >= self._preset_move_until):
            # self.enabled deliberately NOT required — keepalive must fire even
            # when vision is paused or disabled, as long as the serial connection
            # is live and the firmware watchdog is armed.
            # _preset_move_until blocks keepalive during manual preset moves so
            # the firmware ramp is not interrupted and restarted every 400ms.
            b, s, e, w = self.last_steps
            g_ka = self._last_sent_g if self._last_sent_g is not None else self._pinch.gripper_pwm
            cmd = f"B:{b} S:{s} E:{e} W:{w} G:{g_ka} V:100"
            if self.serial.send(cmd):
                self.last_send_t = t   # reset watchdog clock — do NOT update last_steps

    # ═══════════════════════════════════════════════════════
    # HAND CALIBRATION
    # ═══════════════════════════════════════════════════════

    def _calib_start(self):
        """Begin two-point calibration sequence."""
        self._calib_state   = "WAIT_P1"
        self._calib_p1      = None
        self._calib_p2      = None
        self._calib_rolling.clear()
        self._update_calib_ui()
        self._log("Calibration started. Move hand to NEAR/LOW extreme, "
                  "hold as still as possible, then press CAPTURE.", "info")

    def _calib_capture(self):
        """§4.4 rolling-buffer capture — always accepts, mean of last N frames
        suppresses tremor. Stability indicator shown on overlay."""
        if self._calib_state == "IDLE":
            return
        if self._last_raw_xyz is None:
            self._log("No hand detected — move hand into view first.", "warn")
            return

        # Use rolling buffer if populated, fall back to single frame
        if len(self._calib_rolling) >= 3:
            buf      = np.array(self._calib_rolling)
            captured = np.mean(buf, axis=0)
            spread   = float(np.max(np.max(buf, axis=0) - np.min(buf, axis=0)))
            n        = len(buf)
        else:
            captured = self._last_raw_xyz.copy()
            spread   = 0.0
            n        = 1

        self._calib_rolling.clear()   # reset for next point

        quality = ("GOOD" if spread < CALIB_STABILITY_GOOD else
                   "OK"   if spread < CALIB_STABILITY_OK   else "ROUGH")

        if self._calib_state == "WAIT_P1":
            self._calib_p1    = captured
            self._calib_state = "WAIT_P2"
            self._log(f"Point 1 captured [{quality}, n={n}, spread={spread:.1f}mm]: "
                      f"X:{captured[0]:+.0f} Y:{captured[1]:.0f} Z:{captured[2]:.0f} mm", "ok")
            self._log("Now move hand to FAR/HIGH extreme and press CAPTURE.", "info")

        elif self._calib_state == "WAIT_P2":
            self._calib_p2    = captured
            self._calib_state = "DONE"
            self._compute_scale()
            # Snapshot R_wc and t_wc so we can detect if ref marker moves afterward
            if self.tracker and self.tracker.R_wc is not None:
                self._calib_R_wc = self.tracker.R_wc.copy()
                self._calib_t_wc = self.tracker.t_wc.flatten().copy()
            self._log(f"Point 2 captured [{quality}, n={n}, spread={spread:.1f}mm]: "
                      f"X:{captured[0]:+.0f} Y:{captured[1]:.0f} Z:{captured[2]:.0f} mm", "ok")
            self._log(f"Scale — X:{self._calib_scale[0]:.3f}  "
                      f"Y:{self._calib_scale[1]:.3f}  Z:{self._calib_scale[2]:.3f}", "ok")
            self._log("Ref marker position locked — do not move the ref marker "
                      "after calibration.", "info")

        self._update_calib_ui()

    def _compute_scale(self):
        """
        Derive per-axis affine remap from two captured raw hand positions.
        Maps [min(P1,P2), max(P1,P2)] → [WS_MIN, WS_MAX] on each axis.
        Capture order doesn't matter — scale is always positive.
            scale[i]  = (WS_MAX[i] - WS_MIN[i]) / (hi[i] - lo[i])
            offset[i] = WS_MIN[i] - lo[i] * scale[i]
        """
        WS_MIN = np.array([WS_X_MIN, WS_Y_MIN, WS_Z_MIN])
        WS_MAX = np.array([WS_X_MAX, WS_Y_MAX, WS_Z_MAX])
        lo = np.minimum(self._calib_p1, self._calib_p2)
        hi = np.maximum(self._calib_p1, self._calib_p2)
        span = hi - lo   # always positive

        MIN_SPAN_MM = 20.0
        for i in range(3):
            ax = 'XYZ'[i]
            if span[i] < MIN_SPAN_MM:
                self._calib_scale[i]  = 1.0
                self._calib_offset[i] = 0.0
                self._log(f"Axis {ax}: span too small ({span[i]:.1f}mm) "
                          f"— remap left at 1:1", "warn")
            else:
                s = (WS_MAX[i] - WS_MIN[i]) / span[i]
                self._calib_scale[i]  = s
                self._calib_offset[i] = WS_MIN[i] - lo[i] * s
                self._log(f"Axis {ax}: scale={s:.3f}  "
                          f"offset={self._calib_offset[i]:+.1f}mm  "
                          f"(span {span[i]:.0f}mm → "
                          f"{WS_MIN[i]:.0f}–{WS_MAX[i]:.0f}mm)", "info")

    def _calib_reset(self):
        """Reset calibration back to 1:1."""
        self._calib_state   = "IDLE"
        self._calib_p1      = None
        self._calib_p2      = None
        self._calib_scale   = np.array([1.0, 1.0, 1.0])
        self._calib_offset  = np.array([0.0, 0.0, 0.0])
        self._calib_R_wc    = None
        self._calib_t_wc    = None
        self._calib_rolling.clear()
        self._update_calib_ui()
        self._log("Calibration reset to 1:1.", "info")

    def _apply_filter_params(self):
        """Re-construct position and pinch filters using current globals.
        Called after load_config() so loaded filter params actually take effect.
        Uses public update_filter_params() methods — no private field access.
        """
        if self.tracker:
            self.tracker.update_filter_params(
                MEDIAN_WINDOW, ONE_EURO_FREQ, ONE_EURO_MINCUTOFF,
                ONE_EURO_BETA, ONE_EURO_DCUTOFF)
        self._pinch.update_filter_params(
            PINCH_MEDIAN_WINDOW, ONE_EURO_FREQ, PINCH_OEF_MINCUTOFF,
            PINCH_OEF_BETA, ONE_EURO_DCUTOFF)

    def _save_config(self):
        """Save module-level constants AND calibration state to aruco_config.json."""
        g = globals()
        data = {k: g[k] for k in _CONFIG_KEYS}

        # Calibration state — instance level
        if self._calib_state == "DONE":
            data["_calib"] = {
                "scale":  self._calib_scale.tolist(),
                "offset": self._calib_offset.tolist(),
                "p1":     self._calib_p1.tolist(),
                "p2":     self._calib_p2.tolist(),
            }
        else:
            data["_calib"] = None   # no valid calibration — don't restore stale data

        try:
            with open(CONFIG_FILE, "w") as f:
                json.dump(data, f, indent=2)
            self._log(f"Config saved → {CONFIG_FILE}", "ok")
        except Exception as ex:
            self._log(f"Failed to save config: {ex}", "warn")

    def _load_calib(self):
        """Restore calibration state from aruco_config.json after construction."""
        if not os.path.exists(CONFIG_FILE):
            return
        try:
            with open(CONFIG_FILE, "r") as f:
                data = json.load(f)
            calib = data.get("_calib")
            if not calib:
                return
            self._calib_scale  = np.array(calib["scale"],  dtype=float)
            self._calib_offset = np.array(calib["offset"], dtype=float)
            self._calib_p1     = np.array(calib["p1"],     dtype=float)
            self._calib_p2     = np.array(calib["p2"],     dtype=float)
            self._calib_state  = "DONE"
            self._update_calib_ui()
            self._log(f"Calibration restored from {CONFIG_FILE}  "
                      f"scale X:{self._calib_scale[0]:.3f} "
                      f"Y:{self._calib_scale[1]:.3f} "
                      f"Z:{self._calib_scale[2]:.3f}", "ok")
        except Exception as ex:
            self._log(f"Failed to restore calibration: {ex}", "warn")

    def _update_calib_ui(self):
        """Refresh calibration status labels and button states."""
        labels = {
            "IDLE":    ("─  not calibrated",  FG_DIM, "START CALIB",   True,  False),
            "WAIT_P1": ("● waiting for P1...", FG_WARN, "CAPTURE P1",   False, True),
            "WAIT_P2": ("● waiting for P2...", FG_WARN, "CAPTURE P2",   False, True),
            "DONE":    ("✓  calibrated",       FG,      "RECALIBRATE",  True,  False),
        }
        txt, col, btn_txt, start_en, cap_en = labels[self._calib_state]
        self.calib_status_var.set(txt)
        self.calib_status_lbl.configure(fg=col)
        self.calib_start_btn.configure(
            text=btn_txt,
            state="normal" if start_en else "disabled",
            bg="#002200" if start_en else "#1a1a1a")
        self.calib_cap_btn.configure(
            state="normal" if cap_en else "disabled",
            bg="#003300" if cap_en else "#1a1a1a")

        if self._calib_state == "DONE":
            self.calib_scale_var.set(
                f"scale  X:{self._calib_scale[0]:.2f}  "
                f"Y:{self._calib_scale[1]:.2f}  "
                f"Z:{self._calib_scale[2]:.2f}  |  "
                f"off  X:{self._calib_offset[0]:+.0f}  "
                f"Y:{self._calib_offset[1]:+.0f}  "
                f"Z:{self._calib_offset[2]:+.0f}mm")
        else:
            self.calib_scale_var.set("scale  1.00 × 1.00 × 1.00  |  off  0 0 0")

    def _clear_tracking(self):
        """Called when hand marker is lost — stop sending, reset state.
        Guards against stale root.after() calls: if the camera thread resumed
        tracking before this runs on the main thread, skip the reset entirely.
        """
        if self._hand_tracked:
            return   # tracking resumed between queueing this call and now
        self.current_pos   = None
        self.current_speed = 0.0
        self.speed_status  = "NO_TRACKING"
        self.last_result   = None
        # §6.2 — hand marker loss → WAITING (must see hand for 1s to reach STANDBY;
        #         ref marker only required for initial R_wc calibration)
        self.control_state          = "WAITING"
        self._both_visible_since    = None
        self._stable_since          = None
        self._holding_stable_since  = None
        self._pos_hist.clear()
        self._traj_buf.clear()   # §8.1 — flush spline buffer on marker loss
        self._last_sent_raw_xyz = None
        self._last_sent_g       = None
        self._draw_idle_pose()
        self._update_displays()

    # Throttle 3D redraws — matplotlib is slow; cap at ~10 fps

    def _update_displays(self):
        # Detected marker IDs label
        ids = self._ids_present
        self.detect_var.set(f"IDs: {ids}" if ids else "NO MARKERS")

        # Gripper indicator (always updated regardless of tracking state)
        if self._pinch.hand_seen:
            self.pinch_var.set(f"{self._pinch.gripper_pwm} µs")
            self.pinch_dist_var.set(
                f"pinch: {self._pinch.pinch_mm:.1f}mm  "
                f"(closed={PINCH_CLOSED_MM:.0f}  open={PINCH_OPEN_MM:.0f})")
        else:
            self.pinch_var.set("─  no hand")
            self.pinch_dist_var.set("dist: ─")

        if self.current_pos is not None and self.last_result is not None:
            x, y, z  = self.current_pos
            result   = self.last_result

            # Hand position
            self.pos_var.set(f"X:{x:+.0f}  Y:{y:.0f}  Z:{z:.0f}")

            # Joint deltas
            jvals = {
                "BASE":     f"{result.base_delta:+.1f}°  ({result.base_steps:+d})",
                "SHOULDER": f"{result.shoulder_delta:+.1f}°  ({result.shoulder_steps:+d})",
                "ELBOW":    f"{result.elbow_delta:+.1f}°  ({result.elbow_steps:+d})",
                "WRIST":    f"{result.wrist_delta:+.1f}°  ({result.wrist_steps:+d})",
            }
            for name, val in jvals.items():
                self._joint_vars[name].set(val)

            # Reachability + control state
            state_colors = {"TRACKING": FG, "HOLDING": FG_WARN,
                            "STANDBY": FG_DIM, "WAITING": FG_DIM}
            state_col = state_colors.get(self.control_state, FG_DIM)
            if result.reachable:
                self.reach_var.set(f"● REACHABLE  [{self.control_state}]")
                self.reach_lbl.configure(fg=state_col)
            else:
                self.reach_var.set(f"● OUT OF RANGE  [{self.control_state}]")
                self.reach_lbl.configure(fg=FG_ERR)

            # Suppress reason — hijacked during calibration to show stability
            if self._calib_state in ("WAIT_P1", "WAIT_P2"):
                n_buf = len(self._calib_rolling)
                if n_buf >= 3:
                    buf    = np.array(self._calib_rolling)
                    spread = float(np.max(np.max(buf, axis=0) - np.min(buf, axis=0)))
                    pt     = "P1" if self._calib_state == "WAIT_P1" else "P2"
                    if spread < CALIB_STABILITY_GOOD:
                        stab_txt = f"● STABLE  {spread:.0f}mm spread  n={n_buf}  [{pt}]"
                        stab_col = FG
                    elif spread < CALIB_STABILITY_OK:
                        stab_txt = f"◐ OK  {spread:.0f}mm spread  n={n_buf}  [{pt}]"
                        stab_col = FG_WARN
                    else:
                        stab_txt = f"○ ROUGH  {spread:.0f}mm — hold stiller  [{pt}]"
                        stab_col = FG_ERR
                else:
                    stab_txt = f"filling buffer…  {n_buf}/{CALIB_ROLLING_FRAMES}"
                    stab_col = FG_DIM
                self.suppress_var.set(stab_txt)
                self.suppress_lbl.configure(fg=stab_col)
            else:
                self.suppress_var.set(
                    f"⚠ SUPPRESSED: {self.suppress_reason}" if self.suppress_reason else "")
                self.suppress_lbl.configure(fg=FG_ERR)

            # Speed
            spd_col = FG_ERR if self.speed_status == "TOO_FAST" else \
                      FG_WARN if self.speed_status == "WARNING" else FG
            self.speed_var.set(f"{self.current_speed:.0f} mm/s")
            self.speed_lbl.configure(fg=spd_col)
            self.speed_bar.delete("all")
            ws       = self.speed_bar.winfo_width() or 330
            spd_norm = min(self.current_speed / MAX_SPEED_3D, 1.0)
            warn_x   = int((WARNING_SPEED / MAX_SPEED_3D) * ws)
            self.speed_bar.create_rectangle(0, 0, ws, 8, fill="#1a2a1a", outline="")
            self.speed_bar.create_line(warn_x, 0, warn_x, 8, fill=FG_WARN, width=1)
            self.speed_bar.create_rectangle(0, 0, int(spd_norm * ws), 8,
                                             fill=spd_col, outline="")

            # Command stats
            self._update_stats()

            # 3D preview — throttled to ~10 fps to keep UI responsive
            now = time.time()
            if now - self._3d_last_draw >= 0.10:
                self._3d_last_draw = now
                self._update_3d_plot(result, (x, y, z))
        else:
            self.pos_var.set("X:─  Y:─  Z:─")
            for v in self._joint_vars.values():
                v.set("─")
            self.reach_var.set("─")
            self.reach_lbl.configure(fg=FG_DIM)
            if self._calib_state in ("WAIT_P1", "WAIT_P2"):
                self.suppress_var.set("no hand detected — move hand into view")
                self.suppress_lbl.configure(fg=FG_WARN)
            else:
                self.suppress_var.set("")
                self.suppress_lbl.configure(fg=FG_ERR)
            self.speed_var.set("─")
            self.speed_lbl.configure(fg=FG_DIM)

    def _update_stats(self, cmd=None):
        """Update command stats widgets. Called from _update_displays at 10Hz.
        Also logs to monitor when cmd provided (on actual send).
        """
        self.cmds_var.set(f"Commands sent: {self.command_count}")
        self.last_cmd_var.set(f"Last TX: {self._last_cmd_str}")
        if cmd:
            self._monitor_line(f"→ {cmd}", "tx")

    # ═══════════════════════════════════════════════════════
    # SERIAL
    # ═══════════════════════════════════════════════════════

    def _refresh_ports(self):
        ports = [p.device for p in serial.tools.list_ports.comports()]
        self.port_combo["values"] = ports
        if ports:
            self.port_combo.set(ports[0])

    def _toggle_connect(self):
        if self.connected:
            self.serial.disconnect()
            self.connected = False
            self.btn_connect.configure(text="CONNECT", bg="#001a08")
            self._set_status("● DISCONNECTED", FG_ERR)
            self._log("Disconnected.", "info")
        else:
            port = self.port_var.get()
            if not port:
                self._log("No port selected.", "err")
                return
            result = self.serial.connect(port)
            if result is True:
                self.connected = True
                self.btn_connect.configure(text="DISCONNECT", bg="#003300")
                self._set_status(f"● CONNECTED  {port}  {BAUD_RATE}", FG)
                self._log(f"Connected to {port}. Waiting for boot...", "ok")
                # Auto-home after boot delay
                threading.Thread(target=self._auto_home, daemon=True).start()
            else:
                self._log(f"Connection failed: {result}", "err")

    def _auto_home(self):
        time.sleep(3)
        self.root.after(0, self._log, "Sending safety prompt 'y'...", "info")
        self.serial.send("y")
        self.root.after(0, self._log,
                        "Homing all motors — please wait (~35s)...", "info")
        self.root.after(0, self._set_status,
                        "● HOMING — please wait...", FG_WARN)
        self.homing = True
        # Event-driven wait: _handle_rx sets self.homing=False when it sees
        # ALL MOTORS HOMED. Poll rather than sleep a fixed 35 seconds.
        timeout = 120   # generous ceiling for slow homing or stall retry
        waited  = 0
        while self.homing and waited < timeout:
            time.sleep(0.5)
            waited += 0.5
        if not self.homing:
            self.root.after(0, self._log, "Homing complete — ready.", "ok")
            self.root.after(0, self._set_status,
                            f"● CONNECTED  {self.port_var.get()}  {BAUD_RATE}", FG)
        else:
            self.root.after(0, self._log,
                            "Homing timeout — check arm and retry.", "warn")
            self.root.after(0, self._set_status,
                            "● HOMING TIMEOUT", FG_ERR)

    def _home_all(self):
        if not self.connected:
            self._log("Not connected.", "err")
            return
        self.serial.send("a")
        self._monitor_line("→ a  (HOME ALL)", "tx")
        self._log("→ HOME ALL sent", "tx")

    def _toggle_speed_match(self):
        global SPEED_MATCH_ENABLED
        SPEED_MATCH_ENABLED = not SPEED_MATCH_ENABLED
        if SPEED_MATCH_ENABLED:
            self.speed_match_btn.configure(text="⚡ SPD MATCH: ON",  fg="#00aaff", bg="#001a2e")
            self._log("Speed matching ENABLED — arm mirrors hand pace.", "ok")
        else:
            self.speed_match_btn.configure(text="⚡ SPD MATCH: OFF", fg="#555555", bg="#111111")
            self._log("Speed matching OFF — arm runs at full speed.", "info")

    def _toggle_enable(self):
        if not self.connected:
            self._log("Not connected — cannot enable.", "err")
            return
        self.enabled = not self.enabled
        if self.enabled:
            self.enable_btn.configure(text="⏸  PAUSE CONTROL", bg="#004400")
            self._log("Vision control ENABLED.", "ok")
        else:
            self.enable_btn.configure(text="▶  ENABLE CONTROL", bg="#002200")
            self._log("Vision control PAUSED.", "info")
            # Do NOT clear last_steps — keepalive needs it to keep watchdog alive.
            # Clear only the vision-specific references that must reset on re-enable.
            self._last_sent_raw_xyz = None   # reset deadband reference on pause
            self._last_sent_g       = None
            # last_steps intentionally preserved for keepalive and jerk gate on resume

    def _send_raw_entry(self):
        msg = self.raw_var.get().strip()
        if msg and self.connected:
            self.serial.send(msg)
            self._monitor_line(f"→ {msg}", "tx")
            self.raw_var.set("")

    def on_rx(self, line):
        self.root.after(0, self._handle_rx, line)

    def _handle_rx(self, line):
        # ── BOOT: ESP32 RESET ───────────────────────────────────────────
        # Firmware restarted mid-session — arm is un-homed. Pause control
        # and trigger a fresh home sequence.
        if line.startswith("BOOT:"):
            self.enabled = False
            self.enable_btn.configure(text="▶  ENABLE CONTROL", bg="#002200")
            self._log("⚠ ESP32 RESET detected — arm un-homed. Re-homing...", "err")
            self._set_status("● ESP32 RESET — re-homing...", FG_ERR)
            threading.Thread(target=self._auto_home, daemon=True).start()
            self._monitor_line(f"← {line}", "rx_err")
            return

        # ── ALL MOTORS HOMED ────────────────────────────────────────────
        # Firmware homing complete — release the _auto_home wait loop.
        if "HOMED" in line:
            self.homing = False
            self._monitor_line(f"← {line}", "rx_ok")
            return

        # ── STATE: periodic broadcast ───────────────────────────────
        # Suppress from monitor (too noisy at 2Hz).
        # Arm position drift detection is handled by the ref marker drift check
        # in the camera loop — STATE: comparison against last_steps produces too
        # many false positives during streaming and initial catch-up to be useful.
        if line.startswith("STATE:"):
            return

        # ── WATCHDOG: TIMEOUT ───────────────────────────────────────────
        # Firmware stopped motors — no IK command arrived within 500ms.
        # In manual jog mode this fires normally after a preset move because
        # no streaming commands follow — the arm is stationary on purpose.
        # Only pause vision control; manual jog is unaffected either way.
        if line.startswith("WATCHDOG:"):
            if not self._manual_mode:
                self.enabled = False
                self.enable_btn.configure(text="▶  ENABLE CONTROL", bg="#002200")
                self._log("⚠ Watchdog timeout — arm stopped. Re-enable when ready.", "warn")
            else:
                self._log("Watchdog timeout (manual mode — arm idle, normal).", "info")
            self._monitor_line(f"← {line}", "rx_warn")
            return

        # ── Standard tag routing ────────────────────────────────────────
        if line.startswith("OK"):
            tag = "rx_ok"
        elif "ERR" in line or "STALL" in line or "EMERGENCY" in line:
            tag = "rx_err"
        elif "WARNING" in line or "LIMIT" in line:
            tag = "rx_warn"
        else:
            tag = "rx_info"
        self._monitor_line(f"← {line}", tag)



    # ═══════════════════════════════════════════════════════
    # LOGGING
    # ═══════════════════════════════════════════════════════

    def _monitor_line(self, text, tag="rx_info"):
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self.monitor.configure(state="normal")
        if self._monitor_lines >= MONITOR_MAX_LINES:
            self.monitor.delete("1.0", "40.0")
            self._monitor_lines -= 40
        self.monitor.insert("end", f"[{ts}] ", "ts")
        self.monitor.insert("end", text + "\n", tag)
        self._monitor_lines += 1
        self.monitor.see("end")
        self.monitor.configure(state="disabled")

    def _log(self, msg, tag="info"):
        self._monitor_line(msg, tag)

    def _clear_monitor(self):
        self.monitor.configure(state="normal")
        self.monitor.delete("1.0", "end")
        self.monitor.configure(state="disabled")
        self._monitor_lines = 0

    def _set_status(self, msg, color):
        self.status_var.set(msg)
        self.status_lbl.configure(fg=color)

    def on_close(self):
        self._cam_running = False
        self.serial.disconnect()
        self._pinch.close()
        if self.cap:
            self.cap.release()
        self.root.destroy()


# ═══════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    root = tk.Tk()
    app  = VisionGUI(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
