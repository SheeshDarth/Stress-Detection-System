"""
Physiological Pipeline — Remote Photoplethysmography (rPPG)  v2.0
==================================================================
Extracts heart-rate (BPM) and heart-rate variability (RMSSD) from
a standard webcam feed using the **POS** (Plane-Orthogonal-to-Skin)
algorithm (Wang et al., 2017).

Improvements over v1
---------------------
  • Overlap-add POS  — 2-second sliding sub-windows for noise robustness
  • Adaptive ROI     — auto-adjusts ROI bounding box size from landmarks
  • Signal quality   — SNR metric to detect poor illumination / motion
  • Detrending       — 2nd-order polynomial detrend before bandpass
  • Median BPM       — rolling median of per-window BPM estimates
  • Welch PSD        — Welch's method for smoother spectral estimate

Signal processing chain
-----------------------
1. ROI extraction  — forehead + upper cheeks (landmark-based convex hull)
2. Spatial average  — mean RGB per frame in rolling 10-s buffer
3. POS projection   — P = 3Rn − 2Gn,  S = 1.5Rn + Gn − 1.5Bn
4. Overlap-add     — 2-s sub-windows with Hanning taper, stride = 1 frame
5. Detrend + bandpass — poly-detrend → Butterworth 0.7–4.0 Hz
6. BPM via Welch PSD — dominant frequency in valid HR band
7. HRV via RMSSD     — RMS of successive IBI differences from signal peaks
"""

import logging
import cv2
import numpy as np
from collections import deque
from scipy.signal import butter, filtfilt, find_peaks, welch
from scipy.spatial import ConvexHull

logger = logging.getLogger(__name__)


class RPPGExtractor:
    """Extracts rPPG signals using POS algorithm from facial ROIs."""

    # ── ROI landmark indices (MediaPipe Face Mesh 468) ────────────────
    # Forehead — above eyebrows, below hairline
    FOREHEAD_LANDMARKS = [
        10, 67, 109, 108, 69, 104, 68, 71,
        21, 54, 103, 151, 337, 299, 338, 297,
        332, 284, 251, 301, 298,
    ]
    # Left cheek — below left eye, above mouth
    LEFT_CHEEK_LANDMARKS  = [36, 50, 101, 119, 120, 100, 142, 203, 206, 216]
    # Right cheek — below right eye, above mouth
    RIGHT_CHEEK_LANDMARKS = [266, 280, 330, 348, 349, 329, 371, 423, 426, 436]

    def __init__(self, fps: int = 30, window_seconds: int = 10):
        self.fps = fps
        self.window_seconds = window_seconds
        self.window_size = int(fps * window_seconds)

        # Rolling buffer for mean RGB (one entry per frame)
        self.rgb_buffer: deque[np.ndarray] = deque(maxlen=self.window_size)

        # Latest computed metrics
        self.current_bpm: float  = 0.0
        self.current_hrv: float  = 0.0
        self.signal_snr: float   = 0.0
        self.pulse_signal: np.ndarray = np.array([])

        # Rolling BPM buffer for median smoothing
        self._bpm_history: deque[float] = deque(maxlen=8)

    # ── ROI helpers ──────────────────────────────────────────────────

    def _roi_mask(self, landmarks, indices: list[int],
                  frame_shape: tuple) -> np.ndarray | None:
        """Convex hull mask from landmark indices."""
        h, w = frame_shape[:2]
        pts = np.array(
            [[int(landmarks[i].x * w), int(landmarks[i].y * h)]
             for i in indices],
            dtype=np.int32,
        )
        if len(pts) < 3:
            return None
        try:
            hull = ConvexHull(pts)
            hull_pts = pts[hull.vertices]
        except Exception:
            hull_pts = pts

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull_pts, 255)
        return mask

    def _extract_rgb(self, frame: np.ndarray, landmarks) -> np.ndarray | None:
        """Spatial RGB mean across forehead + cheek ROIs."""
        total_rgb = np.zeros(3, dtype=np.float64)
        total_px  = 0

        for roi_idx in (self.FOREHEAD_LANDMARKS,
                        self.LEFT_CHEEK_LANDMARKS,
                        self.RIGHT_CHEEK_LANDMARKS):
            mask = self._roi_mask(landmarks, roi_idx, frame.shape)
            if mask is None:
                continue
            pixels = frame[mask > 0]
            if len(pixels) == 0:
                continue
            total_rgb += pixels.astype(np.float64).sum(axis=0)
            total_px  += len(pixels)

        return (total_rgb / total_px) if total_px > 0 else None

    # ── POS algorithm (overlap-add) ──────────────────────────────────

    def _pos_overlap_add(self, rgb_array: np.ndarray) -> np.ndarray:
        """
        Plane-Orthogonal-to-Skin with overlap–add sub-windows.

        For each 2-s sub-window (sliding by 1 frame):
          P  = 3·Rn − 2·Gn
          S  = 1.5·Rn + Gn − 1.5·Bn
          H  = P + (σ(P)/σ(S))·S

        Outputs are tapered with Hanning window and overlap-added.
        """
        N = len(rgb_array)
        sub_len = max(int(self.fps * 2), 30)  # 2-second sub-window
        if N < sub_len:
            return self._pos_simple(rgb_array)

        pulse = np.zeros(N)
        for start in range(0, N - sub_len + 1, 1):
            end = start + sub_len
            seg = rgb_array[start:end]

            mean = seg.mean(axis=0)
            mean[mean == 0] = 1.0
            norm = seg / mean

            R, G, B = norm[:, 0], norm[:, 1], norm[:, 2]
            P = 3.0 * R - 2.0 * G
            S = 1.5 * R + G - 1.5 * B

            std_s = np.std(S)
            alpha = np.std(P) / std_s if std_s > 0 else 1.0

            h = P + alpha * S
            h *= np.hanning(sub_len)
            pulse[start:end] += h

        pulse -= pulse.mean()
        return pulse

    @staticmethod
    def _pos_simple(rgb_array: np.ndarray) -> np.ndarray:
        """Simple (non-overlap-add) POS for short buffers."""
        if len(rgb_array) < 2:
            return np.array([])

        mean = rgb_array.mean(axis=0)
        mean[mean == 0] = 1.0
        norm = rgb_array / mean

        R, G, B = norm[:, 0], norm[:, 1], norm[:, 2]
        P = 3.0 * R - 2.0 * G
        S = 1.5 * R + G - 1.5 * B

        std_s = np.std(S)
        alpha = np.std(P) / std_s if std_s > 0 else 1.0

        pulse = P + alpha * S
        pulse -= pulse.mean()
        return pulse

    # ── signal conditioning ──────────────────────────────────────────

    def _detrend(self, sig: np.ndarray, order: int = 2) -> np.ndarray:
        """Remove slow baseline drift with polynomial detrending."""
        if len(sig) < order + 2:
            return sig
        t = np.arange(len(sig))
        coeffs = np.polyfit(t, sig, order)
        trend = np.polyval(coeffs, t)
        return sig - trend

    def _bandpass(self, sig: np.ndarray,
                  lo: float = 0.7, hi: float = 4.0,
                  order: int = 4) -> np.ndarray:
        """4th-order Butterworth bandpass (0.7–4.0 Hz)."""
        nyq  = self.fps / 2.0
        low  = max(lo / nyq, 0.01)
        high = min(hi / nyq, 0.99)
        if low >= high or len(sig) < 3 * (order + 1):
            return sig
        b, a = butter(order, [low, high], btype="band")
        return filtfilt(b, a, sig)

    # ── metric computation ───────────────────────────────────────────

    def _bpm_welch(self, pulse: np.ndarray) -> float:
        """Dominant frequency → BPM via Welch PSD.

        Search is restricted to the physiologically plausible HR band of
        50–130 BPM (0.83–2.17 Hz) to avoid locking onto low-frequency
        motion artefacts.  The spectral peak must also dominate (≥15% of
        total in-band power) to guard against broad-spectrum noise.
        """
        if len(pulse) < self.fps * 3:
            return 0.0

        nperseg = min(len(pulse), self.fps * 4)
        freqs, psd = welch(pulse, fs=self.fps, nperseg=nperseg,
                           noverlap=nperseg // 2)

        # Physiological HR band: 50–130 BPM
        mask = (freqs >= 0.83) & (freqs <= 2.17)
        if not mask.any():
            return 0.0

        valid_freqs = freqs[mask]
        valid_psd   = psd[mask]

        # Require the spectral peak to carry ≥15% of in-band power
        peak_idx   = np.argmax(valid_psd)
        peak_power = valid_psd[peak_idx]
        total_power = valid_psd.sum()
        if peak_power / (total_power + 1e-10) < 0.15:
            return 0.0  # No dominant frequency — signal too noisy

        return float(valid_freqs[peak_idx] * 60.0)

    def _compute_snr(self, pulse: np.ndarray) -> float:
        """Estimate signal-to-noise ratio of the pulse signal."""
        if len(pulse) < self.fps * 3:
            return 0.0

        nperseg = min(len(pulse), self.fps * 4)
        freqs, psd = welch(pulse, fs=self.fps, nperseg=nperseg,
                           noverlap=nperseg // 2)

        # Use the same physiological band as BPM estimation
        mask = (freqs >= 0.83) & (freqs <= 2.17)
        if not mask.any() or psd[mask].sum() == 0:
            return 0.0

        peak_power  = np.max(psd[mask])
        total_power = psd[mask].sum()

        snr = 10 * np.log10(peak_power / (total_power - peak_power + 1e-10))
        return float(snr)

    def _hrv_rmssd(self, pulse: np.ndarray) -> float:
        """RMSSD of inter-beat intervals from pulse signal peaks.

        Includes artefact rejection: IBIs that deviate more than 25% from
        the median are discarded before computing RMSSD, preventing false
        peaks from inflating the metric to physiologically impossible values.
        Result is capped at 150 ms — anything higher indicates noise.
        """
        if len(pulse) < self.fps * 3:
            return 0.0

        min_dist = int(self.fps * 0.40)  # min 150 ms between peaks (400 BPM max guard)
        prom = max(0.05, 0.3 * np.std(pulse))
        peaks, _ = find_peaks(pulse, distance=min_dist, prominence=prom)

        if len(peaks) < 3:
            return 0.0

        ibi_ms = np.diff(peaks) / self.fps * 1000.0
        # Physiological IBI range: 380–1500 ms (40–158 BPM)
        valid = ibi_ms[(ibi_ms >= 380) & (ibi_ms <= 1500)]
        if len(valid) < 2:
            return 0.0

        # Artefact rejection: discard IBIs >25% away from the median
        ibi_med = np.median(valid)
        valid = valid[np.abs(valid - ibi_med) / (ibi_med + 1e-10) < 0.25]
        if len(valid) < 2:
            return 0.0

        rmssd = float(np.sqrt(np.mean(np.diff(valid) ** 2)))
        # Cap at 150 ms — higher values are physiologically implausible from rPPG
        return min(rmssd, 150.0)

    # ── public API ───────────────────────────────────────────────────

    def process_frame(self, frame_rgb: np.ndarray,
                      landmarks) -> dict | None:
        """
        Accept one RGB frame + Face Mesh landmarks.
        Updates rolling buffer and returns current physiological metrics.
        """
        rgb_mean = self._extract_rgb(frame_rgb, landmarks)
        if rgb_mean is None:
            return None

        self.rgb_buffer.append(rgb_mean)

        # Need ≥ 3 s of data
        if len(self.rgb_buffer) < self.fps * 3:
            return {
                "bpm": 0.0,
                "hrv_rmssd": 0.0,
                "signal_quality": "Initializing…",
                "buffer_fill": len(self.rgb_buffer) / self.window_size,
                "snr": 0.0,
            }

        # POS → detrend → bandpass → metrics
        rgb_arr = np.array(self.rgb_buffer)

        # Use overlap-add POS for full buffer, simple for partial
        if len(rgb_arr) >= self.fps * 6:
            pulse = self._pos_overlap_add(rgb_arr)
        else:
            pulse = self._pos_simple(rgb_arr)

        if len(pulse) == 0:
            return None

        pulse = self._detrend(pulse)
        filtered = self._bandpass(pulse)
        self.pulse_signal = filtered

        # BPM with median smoothing — only accept physiologically plausible values
        raw_bpm = self._bpm_welch(filtered)
        if 50 <= raw_bpm <= 130:  # physiological gate
            self._bpm_history.append(raw_bpm)
        self.current_bpm = float(np.median(self._bpm_history)) if self._bpm_history else 0.0

        # HRV
        self.current_hrv = self._hrv_rmssd(filtered)

        # Signal quality
        self.signal_snr = self._compute_snr(filtered)

        if len(self.rgb_buffer) >= self.window_size:
            if self.signal_snr > 3:
                quality = "Good"
            elif self.signal_snr > 0:
                quality = "Fair"
            else:
                quality = "Poor"
        else:
            quality = "Stabilizing…"

        return {
            "bpm": self.current_bpm,
            "hrv_rmssd": self.current_hrv,
            "signal_quality": quality,
            "buffer_fill": len(self.rgb_buffer) / self.window_size,
            "snr": self.signal_snr,
        }

    def get_current_metrics(self) -> dict:
        return {
            "bpm": self.current_bpm,
            "hrv_rmssd": self.current_hrv,
            "buffer_size": len(self.rgb_buffer),
            "buffer_capacity": self.window_size,
            "snr": self.signal_snr,
        }

    def is_signal_ready(self) -> bool:
        """Return True when the signal buffer is filled enough to produce reliable predictions.

        Uses the same 30% fill threshold and BPM-valid gate as the main capture loop,
        so callers can query readiness without inspecting internal state directly.
        """
        fill = len(self.rgb_buffer) / self.window_size if self.window_size > 0 else 0.0
        return fill >= 0.30 and self.current_bpm > 0

    def reset(self) -> None:
        """Reset all internal buffers and computed metrics."""
        self.rgb_buffer.clear()
        self._bpm_history.clear()
        self.current_bpm = 0.0
        self.current_hrv = 0.0
        self.signal_snr  = 0.0
        self.pulse_signal = np.array([])
