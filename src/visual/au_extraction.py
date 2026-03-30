"""
Behavioral Pipeline — Facial Action Unit Extraction  (v2.0)
=============================================================
Uses the **MediaPipe Tasks** FaceLandmarker (468+ 3-D landmarks)
to compute frame-by-frame behavioural micro-expression features.

Improvements over v1
---------------------
  • Adaptive blink threshold (calibrated from first 90 frames)
  • Lip-corner depression (AU15) — additional stress marker
  • Jaw clench indicator — chin-to-nose distance variance
  • Exponential-moving-average smoothing on raw metrics
  • Robust windowed statistics (median + IQR alongside mean/std)

Extracted features (per 10-second window)
------------------------------------------
  1. blink_rate         — blinks/min estimated from EAR
  2. ear_mean           — mean Eye Aspect Ratio
  3. ear_std            — EAR variability
  4. brow_furrow_mean   — normalised inner-brow distance (AU4)
  5. brow_furrow_std    — AU4 variability
  6. head_pose_variance — nose-tip displacement variance
  7. head_pose_mean_movement — mean frame-to-frame nose displacement
"""

import logging
import os
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)

# Resolve model path relative to project root
_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "models", "face_landmarker.task",
)

logger = logging.getLogger(__name__)


class AUExtractor:
    """
    Extracts behavioral signals from facial landmarks
    using MediaPipe FaceLandmarker (Tasks API, CPU-only).
    """

    # ── MediaPipe Face Mesh landmark indices ─────────────────────────
    # Right eye (6 pts for EAR — Soukupová & Čech 2016)
    RIGHT_EYE = [33, 160, 158, 133, 153, 144]
    # Left eye
    LEFT_EYE  = [362, 385, 387, 263, 373, 380]

    # Additional vertical eye pairs for improved EAR
    RIGHT_EYE_UPPER = [159, 145]   # upper-lid centre, lower-lid centre
    LEFT_EYE_UPPER  = [386, 374]

    # Inner-eyebrow reference points (AU4 — brow furrow)
    LEFT_INNER_BROW  = 107
    RIGHT_INNER_BROW = 336

    # Outer-eyebrow reference points
    LEFT_OUTER_BROW  = 70
    RIGHT_OUTER_BROW = 300

    # Inter-ocular normalisation anchors
    LEFT_EYE_OUTER  = 33
    RIGHT_EYE_OUTER = 263

    # Nose tip for head-pose tracking
    NOSE_TIP = 1

    # Lip corners (AU15 — lip corner depressor)
    LEFT_LIP_CORNER  = 61
    RIGHT_LIP_CORNER = 291
    UPPER_LIP_CENTER = 13
    LOWER_LIP_CENTER = 14

    # Chin (for jaw clench indicator)
    CHIN = 152

    # ─────────────────────────────────────────────────────────────────

    def __init__(
        self,
        fps: int = 30,
        blink_threshold: float = 0.21,
        model_path: str | None = None,
        adaptive_blink: bool = True,
    ):
        self.fps = fps
        self.blink_threshold = blink_threshold
        self.adaptive_blink = adaptive_blink
        self._calibration_ears: list[float] = []
        self._calibrated = False

        # Resolve model
        model = model_path or _MODEL_PATH
        if not os.path.exists(model):
            raise FileNotFoundError(
                f"FaceLandmarker model not found at {model}.\n"
                "Download with:\n"
                "  python -c \"import urllib.request; "
                "urllib.request.urlretrieve("
                "'https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/1/"
                "face_landmarker.task', 'models/face_landmarker.task')\""
            )

        # MediaPipe FaceLandmarker (CPU-only, local inference)
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model),
            running_mode=RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.landmarker = FaceLandmarker.create_from_options(options)

        # ── Rolling buffers ──
        self.ear_buffer: list[float] = []
        self.brow_buffer: list[float] = []
        self.nose_positions: list[np.ndarray] = []
        self.lip_depression_buffer: list[float] = []
        self.jaw_buffer: list[float] = []

        # ── Blink state machine ──
        self.blink_count = 0
        self.blink_state = False
        self.frame_count = 0

        # ── EMA smoothing ── (alpha = 2/(span+1), span=5)
        self._ema_alpha = 0.33
        self._ema_ear = None
        self._ema_brow = None

        # ── Timestamp for VIDEO mode ──
        self._ts_ms = 0

    # ── helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _distance(p1: np.ndarray, p2: np.ndarray) -> float:
        return float(np.linalg.norm(p1 - p2))

    @staticmethod
    def _lm_xyz(landmark) -> np.ndarray:
        return np.array([landmark.x, landmark.y, landmark.z])

    def _ema(self, prev: float | None, value: float) -> float:
        if prev is None:
            return value
        return self._ema_alpha * value + (1 - self._ema_alpha) * prev

    # ── feature extractors ───────────────────────────────────────────

    def _compute_ear(self, landmarks, eye_indices: list[int]) -> float:
        """
        Eye Aspect Ratio (Soukupová & Čech, 2016).
        EAR = (‖p2−p6‖ + ‖p3−p5‖) / (2 · ‖p1−p4‖)
        """
        pts = [self._lm_xyz(landmarks[i]) for i in eye_indices]
        v1 = self._distance(pts[1], pts[5])
        v2 = self._distance(pts[2], pts[4])
        h  = self._distance(pts[0], pts[3])
        return (v1 + v2) / (2.0 * h) if h > 0 else 0.0

    def _compute_brow_furrow(self, landmarks) -> float:
        """Normalised inner-eyebrow distance (AU4)."""
        lb = self._lm_xyz(landmarks[self.LEFT_INNER_BROW])
        rb = self._lm_xyz(landmarks[self.RIGHT_INNER_BROW])
        le = self._lm_xyz(landmarks[self.LEFT_EYE_OUTER])
        re = self._lm_xyz(landmarks[self.RIGHT_EYE_OUTER])

        iod = self._distance(le, re)
        bd  = self._distance(lb, rb)
        return bd / iod if iod > 0 else 0.0

    def _compute_lip_depression(self, landmarks) -> float:
        """
        Lip corner depression proxy — vertical distance between
        lip corners and upper lip center, normalised by face height.
        Lower values ↔ frown / stress.
        """
        lc = self._lm_xyz(landmarks[self.LEFT_LIP_CORNER])
        rc = self._lm_xyz(landmarks[self.RIGHT_LIP_CORNER])
        uc = self._lm_xyz(landmarks[self.UPPER_LIP_CENTER])
        nose = self._lm_xyz(landmarks[self.NOSE_TIP])
        chin = self._lm_xyz(landmarks[self.CHIN])

        face_h = self._distance(nose, chin)
        if face_h == 0:
            return 0.0

        avg_corner_y = (lc[1] + rc[1]) / 2.0
        depression = (avg_corner_y - uc[1]) / face_h
        return float(depression)

    def _compute_jaw_distance(self, landmarks) -> float:
        """Chin-to-nose distance (jaw clench indicator)."""
        chin = self._lm_xyz(landmarks[self.CHIN])
        nose = self._lm_xyz(landmarks[self.NOSE_TIP])
        le = self._lm_xyz(landmarks[self.LEFT_EYE_OUTER])
        re = self._lm_xyz(landmarks[self.RIGHT_EYE_OUTER])
        iod = self._distance(le, re)
        return self._distance(chin, nose) / iod if iod > 0 else 0.0

    # ── adaptive blink calibration ───────────────────────────────────

    def _calibrate_blink(self) -> None:
        """Set blink threshold as 75% of the median resting EAR."""
        if len(self._calibration_ears) >= 60:
            median_ear = float(np.median(self._calibration_ears))
            self.blink_threshold = max(0.15, median_ear * 0.75)
            self._calibrated = True
            logger.debug(
                "Blink threshold calibrated: %.3f (median EAR: %.3f)",
                self.blink_threshold, median_ear,
            )

    # ── per-frame processing ─────────────────────────────────────────

    def process_frame(self, frame_rgb: np.ndarray) -> dict | None:
        """
        Run FaceLandmarker on *frame_rgb*.
        Returns per-frame AU metrics dict, or None if no face detected.
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        self._ts_ms += int(1000 / self.fps)
        result = self.landmarker.detect_for_video(mp_image, self._ts_ms)

        if not result.face_landmarks:
            return None

        landmarks = result.face_landmarks[0]
        self.frame_count += 1

        # ── 1. EAR (AU45 proxy) ──
        left_ear  = self._compute_ear(landmarks, self.LEFT_EYE)
        right_ear = self._compute_ear(landmarks, self.RIGHT_EYE)
        avg_ear   = (left_ear + right_ear) / 2.0

        # EMA smoothing
        self._ema_ear = self._ema(self._ema_ear, avg_ear)
        smoothed_ear = self._ema_ear

        # Adaptive calibration (first 90 frames ≈ 3 s)
        if self.adaptive_blink and not self._calibrated:
            self._calibration_ears.append(avg_ear)
            if self.frame_count >= 90:
                self._calibrate_blink()

        self.ear_buffer.append(smoothed_ear)

        # Blink state machine
        if smoothed_ear < self.blink_threshold:
            if not self.blink_state:
                self.blink_count += 1
                self.blink_state = True
        else:
            self.blink_state = False

        # ── 2. Brow furrow (AU4) ──
        brow_furrow = self._compute_brow_furrow(landmarks)
        self._ema_brow = self._ema(self._ema_brow, brow_furrow)
        self.brow_buffer.append(self._ema_brow)

        # ── 3. Head pose (nose tip) ──
        nose_pos = self._lm_xyz(landmarks[self.NOSE_TIP])
        self.nose_positions.append(nose_pos)

        # ── 4. Lip depression ──
        lip_dep = self._compute_lip_depression(landmarks)
        self.lip_depression_buffer.append(lip_dep)

        # ── 5. Jaw distance ──
        jaw_dist = self._compute_jaw_distance(landmarks)
        self.jaw_buffer.append(jaw_dist)

        return {
            "ear": smoothed_ear,
            "brow_furrow": self._ema_brow,
            "nose_position": nose_pos,
            "blink_count": self.blink_count,
            "lip_depression": lip_dep,
            "jaw_distance": jaw_dist,
            "landmarks": landmarks,
        }

    # ── windowed aggregation ─────────────────────────────────────────

    def get_window_features(self, window_seconds: int = 10) -> dict | None:
        """
        Aggregate per-frame data over the last *window_seconds*.
        Returns a 7-element behavioural feature dict.
        """
        wf = int(window_seconds * self.fps)
        if len(self.ear_buffer) < wf:
            return None

        ear_w  = np.array(self.ear_buffer[-wf:])
        brow_w = np.array(self.brow_buffer[-wf:])
        nose_w = np.array(self.nose_positions[-wf:])

        # ── Blink rate (blinks / minute) ──
        blinks = 0
        prev = False
        for e in ear_w:
            if e < self.blink_threshold:
                if not prev:
                    blinks += 1
                    prev = True
            else:
                prev = False
        blink_rate = blinks * (60.0 / window_seconds)

        # ── Brow statistics ──
        brow_mean = float(np.mean(brow_w))
        brow_std  = float(np.std(brow_w))

        # ── Head-pose ──
        disp  = np.diff(nose_w, axis=0)
        norms = np.linalg.norm(disp, axis=1) if len(disp) > 0 else np.array([0.0])
        hp_var  = float(np.var(norms))
        hp_mean = float(np.mean(norms))

        return {
            "blink_rate":              blink_rate,
            "ear_mean":                float(np.mean(ear_w)),
            "ear_std":                 float(np.std(ear_w)),
            "brow_furrow_mean":        brow_mean,
            "brow_furrow_std":         brow_std,
            "head_pose_variance":      hp_var,
            "head_pose_mean_movement": hp_mean,
        }

    # ── housekeeping ─────────────────────────────────────────────────

    def reset_buffers(self) -> None:
        self.ear_buffer.clear()
        self.brow_buffer.clear()
        self.nose_positions.clear()
        self.lip_depression_buffer.clear()
        self.jaw_buffer.clear()
        self.blink_count = 0
        self.blink_state = False
        self.frame_count = 0
        self._ema_ear = None
        self._ema_brow = None
        self._calibration_ears.clear()
        self._calibrated = False

    def release(self) -> None:
        self.landmarker.close()
