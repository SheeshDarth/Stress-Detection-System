"""
Multimodal Stress Detection System — Real-Time Application  (v3.0)
====================================================================
Ties together the three pipelines (visual, physiological, fusion)
and drives a live webcam-based HUD.

Improvements over v2
---------------------
  • Face-mesh overlay (press M) — renders 468 landmark connections
  • Session logging (press L) — CSV export of all metrics
  • Stress history mini-graph — 60-reading confidence timeline
  • Improved HUD layout with additional data panels

Controls
────────
  Q  — Quit
  R  — Reset buffers
  M  — Toggle face-mesh overlay
  S  — Save screenshot
  L  — Toggle session logging (CSV export)
"""

import cv2
import os
import sys
import time
import math
import logging
import threading
import numpy as np
import winsound
from collections import deque
from datetime import datetime

from src.visual.au_extraction import AUExtractor
from src.physiological.rppg_extraction import RPPGExtractor
from src.fusion.classifier import StressClassifier
from src.session_logger import SessionLogger
from src.report_generator import generate_report

logger = logging.getLogger(__name__)

# ── Calibration Manager ──────────────────────────────────────────────

class CalibrationManager:
    """Handles the 15-second personalized baseline calibration phase."""
    def __init__(self, duration_sec: int = 15):
        self.duration_sec = duration_sec
        self.start_time = 0.0
        self.is_calibrating = False
        self.is_done = False
        
        # Buffers to collect baseline data
        self.history_bpm = []
        self.history_hrv = []
        self.history_ear = []
        self.history_brow = []
        
        # Final absolute baselines
        self.baseline = {
            "bpm": 0.0,
            "hrv": 0.0,
            "ear": 0.0,
            "brow": 0.0
        }

    def start(self):
        self.start_time = time.time()
        self.is_calibrating = True
        self.is_done = False
        self.history_bpm.clear()
        self.history_hrv.clear()
        self.history_ear.clear()
        self.history_brow.clear()

    def update(self, bpm, hrv, ear, brow) -> float:
        """Returns progress from 0.0 to 1.0"""
        if not self.is_calibrating: return 1.0
        
        elapsed = time.time() - self.start_time
        progress = min(1.0, elapsed / self.duration_sec)
        
        # Only collect valid readings
        if bpm > 0: self.history_bpm.append(bpm)
        if hrv > 0: self.history_hrv.append(hrv)
        if ear > 0: self.history_ear.append(ear)
        if brow > 0: self.history_brow.append(brow)

        if progress >= 1.0:
            self._finalize()
            
        return progress

    def _finalize(self):
        self.is_calibrating = False
        self.is_done = True
        
        if self.history_bpm: self.baseline["bpm"] = float(np.median(self.history_bpm))
        if self.history_hrv: self.baseline["hrv"] = float(np.median(self.history_hrv))
        if self.history_ear: self.baseline["ear"] = float(np.median(self.history_ear))
        if self.history_brow: self.baseline["brow"] = float(np.median(self.history_brow))
        
        logger.info(
            "[Calibration Complete] Personal Baselines: BPM=%.1f  HRV=%.1f  EAR=%.3f  Brow=%.3f",
            self.baseline['bpm'], self.baseline['hrv'],
            self.baseline['ear'], self.baseline['brow'],
        )


# ── MediaPipe FaceMesh connection pairs for overlay drawing ──────────
# Subset of key connections for a clean overlay (not all 468)
FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
    361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
    176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
    162, 21, 54, 103, 67, 109,
]
LEFT_EYE_CONN  = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
RIGHT_EYE_CONN = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
LIPS_CONN = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
]
LEFT_EYEBROW  = [70, 63, 105, 66, 107]
RIGHT_EYEBROW = [300, 293, 334, 296, 336]
NOSE_BRIDGE = [168, 6, 197, 195, 5, 4]


class StressDetectionSystem:
    """End-to-end real-time stress detection via laptop webcam."""

    def __init__(self, fps: int = 30, window_seconds: int = 10):
        self.fps = fps
        self.window_seconds = window_seconds

        # ── Pipeline modules ──
        self.au_extractor   = AUExtractor(fps=fps)
        self.rppg_extractor = RPPGExtractor(fps=fps, window_seconds=window_seconds)
        self.classifier     = StressClassifier(model_path="models/stress_model.pkl")
        self.calibration    = CalibrationManager(duration_sec=15)

        # ── Session logger ──
        self.session_logger = SessionLogger()

        # ── Shared state (guarded by self.lock) ──
        self.lock = threading.Lock()
        self.processing_frame: np.ndarray | None = None
        self.processing_active = True

        self.current_bpm:    float = 0.0
        self.current_hrv:    float = 0.0
        self.current_ear:    float = 0.0
        self.current_brow:   float = 0.0
        self.stress_label:   str   = "Initializing..."
        self.confidence:     float = 0.0
        self.signal_quality: str   = "Initializing..."
        self.signal_snr:     float = 0.0
        self.buffer_fill:    float = 0.0
        self.face_detected:  bool  = False
        self.blink_count:    int   = 0
        self.current_landmarks = None  # for face mesh overlay

        # ── Smoothing & Accuracy ──
        self.bpm_buf = deque(maxlen=15)
        self.hrv_buf = deque(maxlen=15)
        
        # Larger buffers for stronger hysteresis (less twitchy state changes)
        self.label_buf = deque(maxlen=40)
        self.conf_buf  = deque(maxlen=40)
        
        # Continuous stress score 0–100 (shown alongside binary label)
        self.stress_score: float = 0.0

        # ── Performance: throttle heavy ML to 1 Hz (not 30 Hz) ──
        self._last_classify_t: float = 0.0
        self._cached_vis: dict | None = None   # last computed window features

        # ── Model accuracy (set after load/train) ──
        self._model_accuracy: float = 0.0

        # ── Drowsiness detection ──
        self.is_drowsy: bool = False
        self._low_ear_frames: int = 0           # consecutive sub-threshold EAR frames
        self._last_drowsy_alert_t: float = 0.0  # separate cooldown from stress alert
        
        # ── Audio Alert State ──
        self.audio_alerts_enabled = True
        self.last_alert_time = 0.0
        self.alert_cooldown = 15.0  # seconds between beeps

        # ── AI Assistant State ──
        self.ai_tips = [
            "Take a deep breath in... and out slowly.",
            "Relax your shoulders and unclench your jaw.",
            "Look away from the screen for 20 seconds.",
            "Try the 4-7-8 breathing technique.",
            "Stand up and stretch for a moment.",
            "Drink some water to stay hydrated."
        ]
        self.current_tip_idx = 0
        self.last_tip_change = time.time()
        self.xai_trigger = ""

        # ── Pulse waveform (for display) ──
        self.pulse_display: np.ndarray = np.zeros(200)

        # ── Stress history (for mini-graph) ──
        self.stress_history: deque[float] = deque(maxlen=90)

        # ── FPS counter ──
        self.frame_times = deque(maxlen=30)
        self.actual_fps  = 0.0

        # ── UI state ──
        self.show_mesh     = False
        self.session_start = time.time()

        # ── Colour palette (Premium) ──
        self.CLR_GREEN   = (50, 220, 110)
        self.CLR_ORANGE  = (20, 150, 255)
        self.CLR_RED     = (60, 60, 255)
        self.CLR_BLUE    = (255, 180, 50)
        self.CLR_CYAN    = (220, 255, 50)
        self.CLR_WHITE   = (250, 250, 250)
        self.CLR_GRAY    = (170, 170, 170)
        self.CLR_DK_GRAY = (60, 60, 60)
        self.CLR_BG      = (20, 20, 25)
        self.CLR_PANEL   = (35, 35, 40)

    def __del__(self):
        """Safety net: flag shutdown if object is garbage-collected unexpectedly."""
        try:
            self.processing_active = False
        except Exception:
            pass

    # ── model bootstrap ──────────────────────────────────────────────

    def _ensure_model(self) -> None:
        if not self.classifier.load_model():
            logger.info("No pre-trained model found — training on synthetic data …")
            self.classifier.train(use_mock=True)
        # Cache accuracy for HUD display
        self._model_accuracy = self.classifier.training_metrics.get("accuracy", 0.0)
        if self._model_accuracy > 0:
            logger.info("Model accuracy: %.1f%%", self._model_accuracy * 100)

    # ── worker thread ────────────────────────────────────────────────

    def _worker(self) -> None:
        """Heavy computation: Face Landmarks → AU + rPPG → classification."""
        while self.processing_active:
            with self.lock:
                frame = (self.processing_frame.copy()
                         if self.processing_frame is not None else None)
                self.processing_frame = None

            if frame is None:
                time.sleep(0.005)
                continue

            try:
                au = self.au_extractor.process_frame(frame)

                if au is None:
                    with self.lock:
                        self.face_detected = False
                        self.current_landmarks = None
                    continue

                with self.lock:
                    self.face_detected = True
                    self.current_ear = au["ear"]
                    self.current_brow = au["brow_furrow"]
                    self.blink_count = au["blink_count"]
                    self.current_landmarks = au["landmarks"]

                rppg = self.rppg_extractor.process_frame(frame, au["landmarks"])
                if rppg is None:
                    continue

                # Smooth BPM / HRV
                if rppg["bpm"] > 0:
                    self.bpm_buf.append(rppg["bpm"])
                if rppg["hrv_rmssd"] > 0:
                    self.hrv_buf.append(rppg["hrv_rmssd"])

                with self.lock:
                    self.current_bpm    = float(np.median(self.bpm_buf)) if self.bpm_buf else 0.0
                    self.current_hrv    = float(np.median(self.hrv_buf)) if self.hrv_buf else 0.0
                    self.signal_quality = rppg["signal_quality"]
                    self.signal_snr     = rppg.get("snr", 0.0)
                    self.buffer_fill    = rppg["buffer_fill"]

                    # Update pulse waveform for display
                    pulse = self.rppg_extractor.pulse_signal
                    if len(pulse) > 0:
                        indices = np.linspace(0, len(pulse) - 1, 200).astype(int)
                        self.pulse_display = pulse[indices].copy()

                # Process baseline calibration or normal classification
                if self.calibration.is_calibrating:
                    with self.lock:
                        if self.current_bpm > 0:
                            self.calibration.update(self.current_bpm, self.current_hrv, self.current_ear, self.current_brow)
                            self.stress_label = "Calibrating..."
                            self.confidence = 0.0
                else:
                    # ── Continuous stress score (always, even with poor signal) ──
                    score = self._compute_stress_score(
                        self.current_bpm, self.current_hrv,
                        self.current_brow, self.current_ear,
                        self.signal_quality,
                    )
                    with self.lock:
                        self.stress_score = score

                    # ── Drowsiness detection (sustained low EAR = PERCLOS proxy) ──
                    # EAR < 0.21 for >2.5 s at 30 fps = drowsy (heavy eyelids)
                    ear_now = self.current_ear
                    if ear_now > 0:
                        if ear_now < 0.21:
                            self._low_ear_frames += 1
                        else:
                            # Recover faster than accumulate to avoid sticking
                            self._low_ear_frames = max(0, self._low_ear_frames - 2)
                        prev_drowsy = self.is_drowsy
                        self.is_drowsy = self._low_ear_frames > int(self.fps * 2.5)
                        if self.is_drowsy != prev_drowsy:
                            logger.info("Drowsiness %s (low-EAR frames=%d)",
                                        'DETECTED' if self.is_drowsy else 'cleared',
                                        self._low_ear_frames)

                    # ── ML classification throttled to 1 Hz ──
                    # get_window_features() is O(window_size) — far too expensive at 30 Hz
                    now_t = time.time()
                    if now_t - self._last_classify_t >= 1.0:
                        raw_vis = self.au_extractor.get_window_features(self.window_seconds)
                        if raw_vis is not None:
                            self._cached_vis = raw_vis
                        self._last_classify_t = now_t
                    vis = self._cached_vis

                    can_predict = (
                        vis is not None
                        and self.current_bpm > 0
                        and self.buffer_fill >= 0.50
                        and self.signal_quality in ("Fair", "Good")
                    )

                    if can_predict:
                        physio = {"bpm": self.current_bpm, "hrv_rmssd": self.current_hrv}
                        vec = self.classifier.create_feature_vector(vis, physio)
                        if vec is not None:
                            label, conf = self.classifier.predict(vec)
                            self.label_buf.append(label)
                            self.conf_buf.append(conf)

                    with self.lock:
                        # ── Handle NOT-predictable state first ──
                        if not can_predict:
                            # Actively DECAY stale votes when signal is Poor.
                            # Without decay the buffer stays permanently at the last
                            # classifiable state (the "stuck stressed" bug).
                            if self.signal_quality in ("Poor", "Initializing..."):
                                # Remove one stale vote per worker cycle (~30 Hz)
                                # so the buffer empties in ~1-2 s
                                if self.label_buf:
                                    self.label_buf.popleft()
                                if self.conf_buf:
                                    self.conf_buf.popleft()

                            # Score-based authoritative override:
                            # When the ML buffer is empty/stale, the rule-based
                            # stress score determines the displayed state.
                            if score < 28:
                                # Clearly relaxed — snap to Normal immediately
                                self.stress_label = "Normal"
                                self.label_buf.clear()
                                self.conf_buf.clear()
                            elif score > 68 and len(self.label_buf) >= 5:
                                self.stress_label = "Stressed"
                            elif not self.label_buf:
                                self.stress_label = "Measuring..."
                            # Else: keep whatever the hysteresis determined last

                        else:
                            # ── Normal ML hysteresis: require 75% majority (40-vote window) ──
                            recent_labels = list(self.label_buf)
                            total_preds   = len(recent_labels)

                            if total_preds >= 15:
                                stressed_pct = recent_labels.count("Stressed") / total_preds
                                normal_pct   = recent_labels.count("Normal")   / total_preds

                                if stressed_pct > 0.75:
                                    self.stress_label = "Stressed"
                                elif normal_pct > 0.75:
                                    self.stress_label = "Normal"
                                # Below 75% either way: keep current label (no flip)
                            elif total_preds > 0:
                                self.stress_label = recent_labels[-1]  # early phase

                            # Score safety valve — if physiology is clearly calm,
                            # override the ML even if buffer hasn't flipped yet
                            if score < 20 and self.stress_label == "Stressed":
                                self.stress_label = "Normal"
                                self.label_buf.clear()
                                self.conf_buf.clear()
                                logger.debug("Score override: Normal (score=%.0f)", score)

                        # XAI explanation when stressed post-calibration
                        if self.stress_label == "Stressed" and self.calibration.is_done:
                            current_vals = {
                                'bpm': self.current_bpm,
                                'hrv_rmssd': self.current_hrv,
                                'brow_furrow': self.current_brow,
                                'ear_std': vis.get('ear_std', 0) if vis else 0,
                            }
                            self.xai_trigger = self.classifier.explain_prediction(
                                current_vals, self.calibration.baseline
                            )
                        else:
                            self.xai_trigger = ""

                        # Confidence = mean conf of the dominating label
                        matching_confs = [
                            c for lbl, c in zip(self.label_buf, self.conf_buf)
                            if lbl == self.stress_label
                        ]
                        self.confidence = float(np.mean(matching_confs)) if matching_confs else 0.0

                    # Record to stress history for graph
                    score_val  = score / 100.0
                    stress_val = score_val if self.stress_label == "Stressed" else -score_val
                    self.stress_history.append(stress_val)

                    # ── Stress audio alert (1 kHz, 800 ms) ──
                    if (self.stress_label == "Stressed"
                            and score >= 55
                            and self.audio_alerts_enabled):
                        now_a = time.time()
                        if now_a - self.last_alert_time > self.alert_cooldown:
                            threading.Thread(
                                target=lambda: winsound.Beep(1000, 800), daemon=True
                            ).start()
                            self.last_alert_time = now_a

                    # ── Drowsiness audio alert (3 × 1500 Hz rapid beeps) ──
                    if self.is_drowsy and self.audio_alerts_enabled:
                        now_d = time.time()
                        if now_d - self._last_drowsy_alert_t > 8.0:  # 8 s cooldown
                            def _drowsy_beep():
                                for _ in range(3):
                                    winsound.Beep(1500, 300)
                                    time.sleep(0.12)
                            threading.Thread(target=_drowsy_beep, daemon=True).start()
                            self._last_drowsy_alert_t = now_d
                            logger.warning("Drowsiness alert triggered.")

                    # Log to session CSV if active
                    if self.session_logger.is_active:
                        self.session_logger.log(
                            bpm=self.current_bpm,
                            hrv_rmssd=self.current_hrv,
                            ear=self.current_ear,
                            brow_furrow=self.current_brow,
                            blink_count=self.blink_count,
                            stress_label=self.stress_label,
                            confidence=self.confidence,
                            signal_quality=self.signal_quality,
                            signal_snr=self.signal_snr,
                        )

            except Exception as exc:
                logger.exception("Worker thread error: %s", exc)

    # ── Continuous stress score ───────────────────────────────────────

    @staticmethod
    def _compute_stress_score(
        bpm: float,
        hrv: float,
        brow: float,
        ear: float,
        signal_quality: str,
    ) -> float:
        """Rule-based stress score 0–100 using physiological heuristics.

        This score runs independently of the ML classifier and is always
        visible, even when signal quality is Poor.  Higher = more stressed.

        Components (max points):
        - Heart Rate (30 pts): elevated BPM above resting norms
        - HRV RMSSD (30 pts): reduced HRV indicates stress
        - Brow Furrow (25 pts): facial tension (AU4)
        - Blink/EAR  (15 pts): prolonged eye-tension patterns

        Returns 0 if no physiological data is available yet.
        """
        if bpm == 0.0 and hrv == 0.0 and brow == 0.0:
            return 0.0

        score = 0.0

        # ── Heart Rate component (30 pts) ──
        if bpm > 0:
            if bpm >= 110:
                score += 30
            elif bpm >= 100:
                score += 22
            elif bpm >= 90:
                score += 14
            elif bpm >= 82:
                score += 6
            # ≤82 BPM → 0 pts (normal resting range)

        # ── HRV component (30 pts) — low HRV = high stress ──
        if hrv > 0:
            if hrv <= 15:
                score += 30
            elif hrv <= 25:
                score += 22
            elif hrv <= 35:
                score += 14
            elif hrv <= 50:
                score += 6
            # >50 ms HRV → very relaxed, 0 pts

        # ── Brow furrow component (25 pts) ──
        if brow > 0:
            if brow > 0.42:
                score += 25
            elif brow > 0.38:
                score += 16
            elif brow > 0.34:
                score += 8

        # ── EAR / blinking component (15 pts) ──
        if ear > 0:
            # Very low EAR = eyes tense/squinting
            if ear < 0.20:
                score += 15
            elif ear < 0.24:
                score += 8
            elif ear < 0.26:
                score += 3

        # Dampen score when signal is poor (noisy data → less confidence)
        if signal_quality == "Poor":
            score *= 0.60
        elif signal_quality == "Stabilizing…":
            score *= 0.75

        return min(100.0, score)

    # ── HUD drawing helpers ──────────────────────────────────────────

    def _draw_rounded_rect(self, img, pt1, pt2, color, thickness, radius=12):
        """Draw a rounded rectangle."""
        x1, y1 = pt1
        x2, y2 = pt2
        r = radius

        if thickness == -1:
            cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
            cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
            cv2.circle(img, (x1 + r, y1 + r), r, color, -1)
            cv2.circle(img, (x2 - r, y1 + r), r, color, -1)
            cv2.circle(img, (x1 + r, y2 - r), r, color, -1)
            cv2.circle(img, (x2 - r, y2 - r), r, color, -1)
        else:
            cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness)
            cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness)
            cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness)
            cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness)
            cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
            cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
            cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
            cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    def _status_color(self) -> tuple:
        """Return colour based on current stress state."""
        if self.stress_label == "Normal":
            return self.CLR_GREEN
        elif self.stress_label == "Stressed":
            return self.CLR_RED
        return self.CLR_GRAY

    def _draw_face_mesh(self, frame: np.ndarray) -> None:
        """Draw the face landmark mesh overlay on the frame."""
        with self.lock:
            landmarks = self.current_landmarks

        if landmarks is None:
            return

        h, w = frame.shape[:2]

        def _draw_contour(indices, color, closed=True):
            pts = []
            for idx in indices:
                if idx < len(landmarks):
                    lm = landmarks[idx]
                    pts.append((int(lm.x * w), int(lm.y * h)))
            if len(pts) < 2:
                return
            for i in range(len(pts) - 1):
                cv2.line(frame, pts[i], pts[i + 1], color, 1, cv2.LINE_AA)
            if closed and len(pts) > 2:
                cv2.line(frame, pts[-1], pts[0], color, 1, cv2.LINE_AA)

        mesh_color = (0, 255, 200)  # cyan-green
        eye_color = (0, 200, 255)   # warm yellow
        lip_color = (180, 100, 255) # pink-purple
        brow_color = (255, 200, 100) # light blue

        # Face oval
        _draw_contour(FACE_OVAL, mesh_color, closed=True)

        # Eyes
        _draw_contour(LEFT_EYE_CONN, eye_color, closed=True)
        _draw_contour(RIGHT_EYE_CONN, eye_color, closed=True)

        # Eyebrows
        _draw_contour(LEFT_EYEBROW, brow_color, closed=False)
        _draw_contour(RIGHT_EYEBROW, brow_color, closed=False)

        # Lips
        _draw_contour(LIPS_CONN, lip_color, closed=True)

        # Nose bridge
        _draw_contour(NOSE_BRIDGE, mesh_color, closed=False)

        # Additional landmark dots (key points only)
        key_points = [1, 33, 133, 362, 263, 61, 291, 199]
        for idx in key_points:
            if idx < len(landmarks):
                lm = landmarks[idx]
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 2, mesh_color, -1, cv2.LINE_AA)

    def _draw_pulse_waveform(self, frame, x, y, w, h):
        """Draw the live pulse waveform in a box."""
        overlay = frame.copy()
        self._draw_rounded_rect(overlay, (x, y), (x + w, y + h), self.CLR_PANEL, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        self._draw_rounded_rect(frame, (x, y), (x + w, y + h), self.CLR_DK_GRAY, 1)

        cv2.putText(frame, "PULSE", (x + 5, y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.CLR_GRAY, 1)

        pulse = self.pulse_display.copy()
        if len(pulse) < 2 or np.std(pulse) < 1e-10:
            cv2.putText(frame, "No signal", (x + w // 3, y + h // 2 + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.CLR_DK_GRAY, 1)
            return

        pmin, pmax = np.min(pulse), np.max(pulse)
        if pmax - pmin > 0:
            pulse_norm = (pulse - pmin) / (pmax - pmin)
        else:
            pulse_norm = np.zeros_like(pulse)

        n_pts = min(len(pulse_norm), w - 10)
        for i in range(1, n_pts):
            x1 = x + 5 + int((i - 1) / n_pts * (w - 10))
            x2 = x + 5 + int(i / n_pts * (w - 10))
            y1 = y + h - 5 - int(pulse_norm[i - 1] * (h - 22))
            y2 = y + h - 5 - int(pulse_norm[i] * (h - 22))
            color = self.CLR_GREEN if self.stress_label == "Normal" else self.CLR_RED
            cv2.line(frame, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

    def _draw_stress_history(self, frame, x, y, w, h):
        """Draw the stress confidence history as a mini timeline graph."""
        # Background
        overlay = frame.copy()
        self._draw_rounded_rect(overlay, (x, y), (x + w, y + h), self.CLR_PANEL, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        self._draw_rounded_rect(frame, (x, y), (x + w, y + h), self.CLR_DK_GRAY, 1)

        # Label
        cv2.putText(frame, "STRESS HISTORY", (x + 5, y + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.CLR_GRAY, 1)

        history = list(self.stress_history)
        if len(history) < 2:
            cv2.putText(frame, "Collecting...", (x + w // 3, y + h // 2 + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.CLR_DK_GRAY, 1)
            return

        # Draw center line (neutral)
        mid_y = y + h // 2
        cv2.line(frame, (x + 5, mid_y), (x + w - 5, mid_y),
                 self.CLR_DK_GRAY, 1, cv2.LINE_AA)

        # Draw bars — positive = stressed (red), negative = normal (green)
        bar_w = max(1, (w - 10) // len(history))
        usable_h = (h - 24) // 2  # half-height for each direction

        for i, val in enumerate(history):
            bx = x + 5 + i * bar_w
            if bx + bar_w > x + w - 5:
                break

            bar_h = int(abs(val) * usable_h)
            if bar_h < 1:
                bar_h = 1

            if val > 0:  # Stressed
                color = self.CLR_RED
                cv2.rectangle(frame, (bx, mid_y - bar_h),
                              (bx + bar_w - 1, mid_y), color, -1)
            else:  # Normal
                color = self.CLR_GREEN
                cv2.rectangle(frame, (bx, mid_y),
                              (bx + bar_w - 1, mid_y + bar_h), color, -1)

        # Labels
        cv2.putText(frame, "S", (x + w - 14, y + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, self.CLR_RED, 1)
        cv2.putText(frame, "N", (x + w - 14, y + h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, self.CLR_GREEN, 1)

    def _draw_ai_assistant(self, frame: np.ndarray, x: int, y: int, w: int, h: int) -> None:
        """Draw the AI assistant panel with contextual tips."""
        overlay = frame.copy()
        self._draw_rounded_rect(overlay, (x, y), (x + w, y + h), self.CLR_PANEL, -1)
        cv2.addWeighted(overlay, 0.9, frame, 0.1, 0, frame)
        
        # Border changes color based on stress
        border_clr = self.CLR_RED if self.stress_label == "Stressed" else self.CLR_DK_GRAY
        self._draw_rounded_rect(frame, (x, y), (x + w, y + h), border_clr, 1)

        # Header
        cv2.putText(frame, "AI ASSISTANT", (x + 12, y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.CLR_CYAN, 1, cv2.LINE_AA)
        
        # Rotate tips every 8 seconds
        now = time.time()
        if now - self.last_tip_change > 8.0:
            self.current_tip_idx = (self.current_tip_idx + 1) % len(self.ai_tips)
            self.last_tip_change = now

        tip = self.ai_tips[self.current_tip_idx]
        
        # Override tip if stressed
        if self.stress_label == "Stressed":
            if self.xai_trigger:
                tip = self.xai_trigger + ". Please take a deep breath."
            else:
                tip = "High stress detected! Please take a deep breath."
            
        # Breathing animation indicator (expanding/contracting circle)
        if self.stress_label == "Stressed":
            cycle = (math.sin(now * 2) + 1) / 2  # 0 to 1
            radius = int(4 + cycle * 4)
            cv2.circle(frame, (x + w - 20, y + 16), radius, self.CLR_RED, -1, cv2.LINE_AA)

        # Simple text wrapping for the tip
        words = tip.split(' ')
        lines = []
        current_line = ""
        for word in words:
            test_line = current_line + " " + word if current_line else word
            (tw, _), _ = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            if tw > w - 24:
                lines.append(current_line)
                current_line = word
            else:
                current_line = test_line
        lines.append(current_line)

        # Draw text lines
        ty = y + 45
        for line in lines:
            cv2.putText(frame, line, (x + 12, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.CLR_WHITE, 1, cv2.LINE_AA)
            ty += 18

    def _draw_calibration_overlay(self, frame: np.ndarray) -> None:
        """Draw a full-screen semi-transparent overlay indicating the calibration phase."""
        h, w = frame.shape[:2]
        
        # Dark overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (10, 10, 15), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        cx, cy = w // 2, h // 2
        
        # Progress math
        progress = self.calibration.update(0,0,0,0)  # Just read current time progress
        angle = int(360 * progress)
        
        # Draw glowing ring
        ring_r = 100
        axes = (ring_r, ring_r)
        # Background ring
        cv2.ellipse(frame, (cx, cy), axes, 0, 0, 360, self.CLR_DK_GRAY, 4, cv2.LINE_AA)
        # Progress ring
        cv2.ellipse(frame, (cx, cy), axes, -90, 0, angle, self.CLR_CYAN, 6, cv2.LINE_AA)

        # Draw text at center
        pct = int(progress * 100)
        cv2.putText(frame, f"{pct}%", (cx - 25, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.CLR_WHITE, 2, cv2.LINE_AA)
        
        cv2.putText(frame, "CALIBRATING BASELINE", (cx - 150, cy + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.CLR_CYAN, 2, cv2.LINE_AA)
        cv2.putText(frame, "Please relax and look at the camera naturally.", (cx - 180, cy + 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.CLR_GRAY, 1, cv2.LINE_AA)
        
        # Show mini vitals underneath
        cv2.putText(frame, f"BPM: {self.current_bpm:.0f}", (cx - 100, cy + 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.CLR_GRAY, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Face: {'Detected' if self.face_detected else 'Searching...'}", (cx + 10, cy + 220), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.CLR_GRAY, 1, cv2.LINE_AA)


    def _draw_hud(self, frame: np.ndarray) -> np.ndarray:
        """Overlay a comprehensive heads-up display."""
        import math
        h, w = frame.shape[:2]
        accent = self._status_color()

        # ═══════════════════════════════════════════════
        # ── Face mesh overlay (behind HUD) ──
        # ═══════════════════════════════════════════════
        if self.show_mesh:
            self._draw_face_mesh(frame)

        # ═══════════════════════════════════════════════
        # ── Drowsiness warning banner (top-centre) ──
        # ═══════════════════════════════════════════════
        if self.is_drowsy:
            # Flashing amber banner: visible on even seconds, dark on odd
            flash_on = int(time.time() * 2) % 2 == 0
            ban_h, ban_w = 46, 340
            ban_x = (w - ban_w) // 2
            ban_y = 10
            ban_clr = (0, 165, 255) if flash_on else (0, 100, 160)  # amber BGR
            overlay2 = frame.copy()
            self._draw_rounded_rect(overlay2, (ban_x, ban_y),
                                    (ban_x + ban_w, ban_y + ban_h), ban_clr, -1)
            cv2.addWeighted(overlay2, 0.80, frame, 0.20, 0, frame)
            self._draw_rounded_rect(frame, (ban_x, ban_y),
                                    (ban_x + ban_w, ban_y + ban_h), (0, 204, 255), 2)
            cv2.putText(frame, "[!] DROWSINESS DETECTED  --  TAKE A BREAK",
                        (ban_x + 12, ban_y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)

        # ═══════════════════════════════════════════════
        # ── Main panel (top-right) ──
        # ═══════════════════════════════════════════════
        pw, ph = 310, 305
        px, py = w - pw - 16, 16

        # Semi-transparent background
        overlay = frame.copy()
        self._draw_rounded_rect(overlay, (px, py), (px + pw, py + ph),
                                self.CLR_PANEL, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Accent border (premium subtle gradient-like look via thick edge)
        self._draw_rounded_rect(frame, (px, py), (px + pw, py + ph),
                                accent, 2)

        # ── Title bar ──
        cv2.putText(frame, "STRESS MONITOR", (px + 14, py + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, self.CLR_WHITE, 2, cv2.LINE_AA)

        # Session timer
        elapsed = int(time.time() - self.session_start)
        mins, secs = divmod(elapsed, 60)
        cv2.putText(frame, f"{mins:02d}:{secs:02d}", (px + pw - 65, py + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.CLR_GRAY, 1, cv2.LINE_AA)

        # Model accuracy badge (small, below timer)
        if self._model_accuracy > 0:
            acc_txt = f"Acc: {self._model_accuracy * 100:.1f}%"
            cv2.putText(frame, acc_txt, (px + pw - 80, py + 42),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.30, (120, 200, 120), 1, cv2.LINE_AA)

        cv2.line(frame, (px + 10, py + 48), (px + pw - 10, py + 48),
                 self.CLR_DK_GRAY, 1)

        # ── Face status ──
        face_clr = self.CLR_GREEN if self.face_detected else self.CLR_RED
        face_txt = "Face OK" if self.face_detected else "No Face"
        cv2.circle(frame, (px + 20, py + 55), 5, face_clr, -1, cv2.LINE_AA)
        cv2.putText(frame, face_txt, (px + 32, py + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, face_clr, 1, cv2.LINE_AA)

        # Blink count
        cv2.putText(frame, f"Blinks: {self.blink_count}",
                    (px + 140, py + 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.CLR_GRAY, 1, cv2.LINE_AA)

        # ── Heart Rate ──
        y_off = py + 85
        bpm_clr = self.CLR_GREEN if 55 <= self.current_bpm <= 100 else self.CLR_RED
        if self.current_bpm == 0: bpm_clr = self.CLR_GRAY
        cv2.putText(frame, "Heart Rate", (px + 14, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.CLR_GRAY, 1, cv2.LINE_AA)
        bpm_txt = f"{self.current_bpm:.0f}" if self.current_bpm > 0 else "--"
        cv2.putText(frame, bpm_txt, (px + 14, y_off + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, bpm_clr, 2, cv2.LINE_AA)
        cv2.putText(frame, "BPM", (px + 14 + len(bpm_txt) * 22, y_off + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.CLR_GRAY, 1, cv2.LINE_AA)

        # ── HRV ──
        hrv_clr = self.CLR_GREEN if self.current_hrv > 30 else self.CLR_ORANGE
        if self.current_hrv == 0: hrv_clr = self.CLR_GRAY
        cv2.putText(frame, "HRV (RMSSD)", (px + 165, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.CLR_GRAY, 1, cv2.LINE_AA)
        hrv_txt = f"{self.current_hrv:.1f}" if self.current_hrv > 0 else "--"
        cv2.putText(frame, hrv_txt, (px + 165, y_off + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, hrv_clr, 2, cv2.LINE_AA)
        cv2.putText(frame, "ms", (px + 165 + len(hrv_txt) * 22, y_off + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.CLR_GRAY, 1, cv2.LINE_AA)

        # ── Divider ──
        y_off += 50
        cv2.line(frame, (px + 10, y_off), (px + pw - 10, y_off),
                 self.CLR_DK_GRAY, 1)

        # ── Stress Status ──
        y_off += 20
        cv2.putText(frame, "Status", (px + 14, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.CLR_GRAY, 1, cv2.LINE_AA)

        status_clr = self._status_color()
        # Add slight pulsing effect if stressed
        if self.stress_label == "Stressed":
            pulse_alpha = (math.sin(time.time() * 5) + 1) / 2
            r, g, b = status_clr
            status_clr = (int(r + (255-r)*pulse_alpha*0.3), int(g*0.7), int(b*0.7))
            
        cv2.putText(frame, self.stress_label, (px + 14, y_off + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, status_clr, 2, cv2.LINE_AA)

        # Confidence bar
        if self.confidence > 0:
            bar_x = px + 14
            bar_y = y_off + 42
            bar_w = pw - 28
            fill  = int(bar_w * self.confidence)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 10),
                          self.CLR_DK_GRAY, -1)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + 10),
                          status_clr, -1)
            cv2.putText(frame, f"{self.confidence * 100:.0f}%",
                        (bar_x + bar_w + 5, bar_y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, self.CLR_GRAY, 1, cv2.LINE_AA)

        # ── Signal quality bar ──
        y_off += 65
        cv2.putText(frame, f"Signal: {self.signal_quality}", (px + 14, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.CLR_GRAY, 1)

        qual_bar_x = px + 14
        qual_bar_y = y_off + 8
        qual_bar_w = pw - 28
        snr_frac = min(1.0, max(0.0, self.signal_snr / 10.0))
        snr_fill = int(qual_bar_w * snr_frac)
        snr_clr = self.CLR_GREEN if snr_frac > 0.3 else (self.CLR_ORANGE if snr_frac > 0.1 else self.CLR_RED)
        cv2.rectangle(frame, (qual_bar_x, qual_bar_y),
                      (qual_bar_x + qual_bar_w, qual_bar_y + 6),
                      self.CLR_DK_GRAY, -1)
        cv2.rectangle(frame, (qual_bar_x, qual_bar_y),
                      (qual_bar_x + snr_fill, qual_bar_y + 6),
                      snr_clr, -1)

        # ── Buffer fill bar ──
        buf_y = qual_bar_y + 14
        buf_w = int(qual_bar_w * self.buffer_fill)
        cv2.rectangle(frame, (qual_bar_x, buf_y),
                      (qual_bar_x + qual_bar_w, buf_y + 6),
                      self.CLR_DK_GRAY, -1)
        cv2.rectangle(frame, (qual_bar_x, buf_y),
                      (qual_bar_x + buf_w, buf_y + 6),
                      self.CLR_BLUE, -1)
        cv2.putText(frame, f"Buffer: {self.buffer_fill*100:.0f}%",
                    (qual_bar_x + qual_bar_w + 5, buf_y + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, self.CLR_GRAY, 1)

        # ── Stress Score bar (0–100, always shown) ──
        sc_y = buf_y + 18
        sc_score = self.stress_score
        sc_frac  = sc_score / 100.0
        sc_fill  = int(qual_bar_w * sc_frac)
        # Colour gradient: green → orange → red
        if sc_frac < 0.40:
            sc_clr = self.CLR_GREEN
        elif sc_frac < 0.65:
            sc_clr = self.CLR_ORANGE
        else:
            sc_clr = self.CLR_RED
        cv2.putText(frame, "Stress Score", (qual_bar_x, sc_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.CLR_GRAY, 1)
        sc_bar_y = sc_y + 6
        cv2.rectangle(frame, (qual_bar_x, sc_bar_y),
                      (qual_bar_x + qual_bar_w, sc_bar_y + 8),
                      self.CLR_DK_GRAY, -1)
        cv2.rectangle(frame, (qual_bar_x, sc_bar_y),
                      (qual_bar_x + sc_fill, sc_bar_y + 8),
                      sc_clr, -1)
        cv2.putText(frame, f"{sc_score:.0f}",
                    (qual_bar_x + qual_bar_w + 5, sc_bar_y + 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, sc_clr, 1, cv2.LINE_AA)


        # ═══════════════════════════════════════════════
        # ── Pulse waveform (below main panel) ──
        # ═══════════════════════════════════════════════
        wave_w, wave_h = pw, 80
        wave_x = w - wave_w - 12
        wave_y = py + ph + 8
        self._draw_pulse_waveform(frame, wave_x, wave_y, wave_w, wave_h)

        # ═══════════════════════════════════════════════
        # ── Stress history (below pulse waveform) ──
        # ═══════════════════════════════════════════════
        hist_w, hist_h = pw, 65
        hist_x = w - hist_w - 12
        hist_y = wave_y + wave_h + 6
        self._draw_stress_history(frame, hist_x, hist_y, hist_w, hist_h)

        # ═══════════════════════════════════════════════
        # ── AI Assistant (bottom-right, below history) ──
        # ═══════════════════════════════════════════════
        ai_w, ai_h = pw, 90
        ai_x = w - ai_w - 16
        ai_y = hist_y + hist_h + 8
        self._draw_ai_assistant(frame, ai_x, ai_y, ai_w, ai_h)

        # ═══════════════════════════════════════════════
        # ── Bottom-left info ──
        # ═══════════════════════════════════════════════
        cv2.putText(frame, f"FPS: {self.actual_fps:.0f}", (16, h - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.CLR_WHITE, 1, cv2.LINE_AA)

        # ── Top-left controls ──
        controls = "ESC/Q=Quit  R=Reset  M=Mesh  S=Screenshot  L=Log  A=Audio"
        cv2.putText(frame, controls,
                    (16, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, self.CLR_WHITE, 1, cv2.LINE_AA)

        # ── Indicators ──
        iy = 50
        if self.session_logger.is_active:
            # Pulsating red dot
            alpha = 0.6 + 0.4 * math.sin(time.time() * 3)
            dot_clr = tuple(int(c * alpha) for c in self.CLR_RED)
            cv2.circle(frame, (20, iy), 6, dot_clr, -1, cv2.LINE_AA)
            cv2.putText(frame, f"REC ({self.session_logger.row_count} rows)",
                        (32, iy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        self.CLR_RED, 1, cv2.LINE_AA)
            iy += 25
            
        if self.audio_alerts_enabled:
            cv2.putText(frame, "Audio: ON", (16, iy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.CLR_CYAN, 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Audio: OFF", (16, iy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.CLR_GRAY, 1, cv2.LINE_AA)

        # ── Face-not-detected warning ──
        if not self.face_detected:
            alpha = 0.3 + 0.2 * np.sin(time.time() * 4)
            overlay2 = frame.copy()
            cv2.rectangle(overlay2, (0, 0), (w, h), self.CLR_RED, 8)
            cv2.addWeighted(overlay2, alpha, frame, 1 - alpha, 0, frame)
            cv2.putText(frame, "NO FACE DETECTED",
                        (w // 2 - 140, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.CLR_RED, 2)

        return frame

    # ── main loop ────────────────────────────────────────────────────

    def run(self, camera_index: int = 0) -> None:
        """Main capture loop.  Drives the webcam feed, HUD rendering and key handling.

        Args:
            camera_index: OpenCV camera device index (default 0).
        """
        logger.info("Multimodal Stress Detection System v3.0 — starting")
        logger.info("Controls: ESC/Q=Quit  R=Reset  M=Mesh  S=Screenshot  L=Log  A=Audio")

        # Privacy / biometric data notice
        logger.info("NOTICE: This app processes facial video for stress analysis.")
        logger.info("All data is processed on-device only. Session logs saved to: %s",
                    os.path.abspath(self.session_logger.log_dir))

        # Bootstrap model
        self._ensure_model()

        # Start worker
        worker = threading.Thread(target=self._worker, daemon=True)
        worker.start()

        # Open webcam
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            logger.error("Could not open webcam (index %d).", camera_index)
            return
        
        # Request higher resolution if available for a 16:9 modern look
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        logger.info("Webcam opened (index %d) — starting real-time detection …", camera_index)

        self.session_start = time.time()
        
        # Trigger calibration phase
        self.calibration.start()
        
        # ── Task 8: Webcam disconnect recovery state ──
        consecutive_failures = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    logger.warning("Frame read failed — attempting reconnect (failure #%d)…",
                                   consecutive_failures + 1)
                    cap.release()
                    time.sleep(1.0)
                    cap = cv2.VideoCapture(camera_index)
                    if not cap.isOpened():
                        logger.error("Camera reconnect failed. Exiting.")
                        break
                    consecutive_failures += 1
                    if consecutive_failures > 10:
                        logger.error("Too many consecutive frame failures (%d). Exiting.",
                                     consecutive_failures)
                        break
                    continue
                consecutive_failures = 0

                # Mirror the video feed for a natural user experience
                frame = cv2.flip(frame, 1)

                # FPS tracking
                now = time.time()
                self.frame_times.append(now)
                if len(self.frame_times) > 1:
                    dt = self.frame_times[-1] - self.frame_times[0]
                    self.actual_fps = len(self.frame_times) / dt if dt > 0 else 0

                # Send frame to worker (latest-only, non-blocking)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                with self.lock:
                    self.processing_frame = rgb

                # Draw HUD
                display = self._draw_hud(frame)
                cv2.imshow("Stress Detection System", display)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):  # Q or ESC to quit
                    break
                elif key in (ord("r"), ord("R")):
                    self.au_extractor.reset_buffers()
                    self.rppg_extractor.reset()
                    self.bpm_buf.clear()
                    self.hrv_buf.clear()
                    self.label_buf.clear()
                    self.conf_buf.clear()
                    self.stress_history.clear()
                    self.stress_label = "Initializing..."
                    self.confidence = 0.0
                    self.pulse_display = np.zeros(200)
                    logger.info("Buffers reset.")
                elif key in (ord("m"), ord("M")):
                    self.show_mesh = not self.show_mesh
                    logger.info("Mesh overlay: %s", 'ON' if self.show_mesh else 'OFF')
                elif key in (ord("a"), ord("A")):
                    self.audio_alerts_enabled = not self.audio_alerts_enabled
                    logger.info("Audio alerts: %s", 'ON' if self.audio_alerts_enabled else 'OFF')
                elif key in (ord("s"), ord("S")):
                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    fname = f"screenshot_{ts}.png"
                    cv2.imwrite(fname, display)
                    logger.info("Screenshot saved: %s", fname)
                elif key in (ord("l"), ord("L")):
                    if self.session_logger.is_active:
                        self.session_logger.stop()
                        logger.info("Logging stopped. File: %s", self.session_logger.filepath)
                    else:
                        path = self.session_logger.start()
                        logger.info("Logging started → %s", path)

        finally:
            self.processing_active = False
            if self.session_logger.is_active:
                self.session_logger.finalise()
                logger.info("Session log saved: %s", self.session_logger.filepath)
                logger.info("Generating HTML Dashboard...")
                report_path = generate_report(self.session_logger.filepath)
                if report_path:
                    logger.info("Interactive Report available at: %s", report_path)
            cap.release()
            cv2.destroyAllWindows()
            self.au_extractor.release()
            logger.info("System shut down cleanly.")


# ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multimodal Stress Detection System v3.0")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--no-calib", action="store_true", help="Skip calibration phase")
    args = parser.parse_args()

    # ── Ensure log directory exists ──
    os.makedirs("logs", exist_ok=True)

    # ── Configure root logger ──
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/stress_detection.log", encoding="utf-8"),
        ],
    )

    system = StressDetectionSystem(fps=30, window_seconds=10)

    if args.no_calib:
        system.calibration.is_done = True  # skip calibration
        system.calibration.is_calibrating = False
        logger.info("--no-calib: Calibration phase skipped.")

    system.run(camera_index=args.camera)
