"""
WESAD Dataset Loader - Cross-Dataset Validation  (Phase 1)
============================================================
Loads the WESAD (Wearable Stress and Affect Detection) dataset,
extracts physiological features (HR, HRV) from the wrist BVP signal,
and returns feature vectors compatible with the 11-D schema.

WESAD Reference
---------------
Philip Schmidt, Attila Reiss, Robert Duerichen, Claus Marberger, Kai Van Laerhoven.
"Introducing WESAD, a Multimodal Dataset for Wearable Stress and Affect Detection."
Proc. ACM ICMI 2018. DOI: 10.1145/3242969.3242985

Dataset Download
----------------
https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gu  (requires registration)
Extract so that the path looks like:
  <dataset_root>/S2/S2.pkl
  <dataset_root>/S3/S3.pkl
  ...  (subjects S2-S17, S12 excluded)

Label Mapping
-------------
WESAD label codes:
  0 = not defined / transient
  1 = baseline   - Normal  (label=0)
  2 = stress     - Stressed (label=1)
  3 = amusement  - excluded (not comparable to stress)
  4 = meditation - excluded

Feature Extraction
------------------
The wrist BVP signal (64 Hz) is windowed into 10-second segments.
Each segment produces 2 physiological features:
  - bpm        (Welch PSD on bandpass-filtered BVP)
  - hrv_rmssd  (IBI peak detection - RMSSD)

The 9 behavioral features (EAR, brow, head pose, lip, jaw) cannot be
extracted from WESAD (no video). They are set to 0.0 in the returned
feature vectors. This is explicitly acknowledged as a limitation.

Usage
-----
    loader = WESADLoader(dataset_root="/path/to/WESAD")
    X, y, subject_ids = loader.load_all()
    print(f"Loaded {len(y)} windows from {len(set(subject_ids))} subjects")
    # X.shape == (N, 11)   (9 behavioral=0, 2 physiological from BVP)
"""

import os
import pickle
import logging
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, welch

logger = logging.getLogger(__name__)

# WESAD label codes
_LABEL_BASELINE = 1
_LABEL_STRESS   = 2
_EXCLUDED_LABELS = {0, 3, 4}  # transient, amusement, meditation

# BVP sampling rate (wrist device)
_WRIST_BVP_FS = 64
# Window and step in seconds
_WINDOW_SEC  = 10
_STEP_SEC    = 5   # 50% overlap


class WESADLoader:
    """Load WESAD subjects and extract stress-compatible features."""

    def __init__(
        self,
        dataset_root: str,
        window_sec: int  = _WINDOW_SEC,
        step_sec: int    = _STEP_SEC,
        bvp_fs: int      = _WRIST_BVP_FS,
    ):
        self.dataset_root = dataset_root
        self.window_sec   = window_sec
        self.step_sec     = step_sec
        self.bvp_fs       = bvp_fs
        self.win_samples  = window_sec * bvp_fs
        self.step_samples = step_sec   * bvp_fs

    # -- BVP signal processing -----------------------------------------

    def _bandpass(self, sig: np.ndarray) -> np.ndarray:
        """Butterworth bandpass 0.7-4.0 Hz (42-240 BPM)."""
        nyq  = self.bvp_fs / 2.0
        low  = max(0.7 / nyq, 0.01)
        high = min(4.0 / nyq, 0.99)
        if low >= high or len(sig) < 15:
            return sig
        b, a = butter(4, [low, high], btype="band")
        return filtfilt(b, a, sig)

    def _extract_bpm(self, bvp: np.ndarray) -> float:
        """Welch PSD - dominant frequency - BPM."""
        if len(bvp) < self.bvp_fs * 3:
            return 0.0
        nperseg = min(len(bvp), self.bvp_fs * 4)
        freqs, psd = welch(bvp, fs=self.bvp_fs, nperseg=nperseg,
                           noverlap=nperseg // 2)
        mask = (freqs >= 0.75) & (freqs <= 2.5)   # 45-150 BPM
        if not mask.any():
            return 0.0
        valid_psd   = psd[mask]
        valid_freqs = freqs[mask]
        peak_idx    = np.argmax(valid_psd)
        if valid_psd[peak_idx] / (valid_psd.sum() + 1e-10) < 0.15:
            return 0.0
        return float(valid_freqs[peak_idx] * 60.0)

    def _extract_hrv(self, bvp: np.ndarray) -> float:
        """IBI peak detection - RMSSD (ms), capped at 150."""
        if len(bvp) < self.bvp_fs * 3:
            return 0.0
        min_dist = int(self.bvp_fs * 0.40)
        prom = max(0.05, 0.3 * np.std(bvp))
        peaks, _ = find_peaks(bvp, distance=min_dist, prominence=prom)
        if len(peaks) < 3:
            return 0.0
        ibi_ms = np.diff(peaks) / self.bvp_fs * 1000.0
        valid  = ibi_ms[(ibi_ms >= 380) & (ibi_ms <= 1500)]
        if len(valid) < 2:
            return 0.0
        ibi_med = np.median(valid)
        valid   = valid[np.abs(valid - ibi_med) / (ibi_med + 1e-10) < 0.25]
        if len(valid) < 2:
            return 0.0
        return float(min(np.sqrt(np.mean(np.diff(valid) ** 2)), 150.0))

    # -- per-subject loading -------------------------------------------

    def _load_subject(self, subject_id: str) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Load one subject pickle.
        Returns (X_windows, y_windows) where X is (N, 11) and y is (N,).
        """
        pkl_path = os.path.join(
            self.dataset_root, subject_id, f"{subject_id}.pkl"
        )
        if not os.path.exists(pkl_path):
            logger.warning("Subject %s not found at %s", subject_id, pkl_path)
            return None

        with open(pkl_path, "rb") as f:
            data = pickle.load(f, encoding="latin1")

        # Wrist BVP (64 Hz) and labels (700 Hz for chest, resampled for alignment)
        bvp    = data["signal"]["wrist"]["BVP"].flatten().astype(np.float64)
        labels = data["label"].flatten()  # 700 Hz

        # Downsample labels to 64 Hz to match BVP
        ratio = len(labels) / len(bvp)
        label_ds = np.array([
            labels[int(i * ratio)]
            for i in range(len(bvp))
        ])

        windows_X, windows_y = [], []

        for start in range(0, len(bvp) - self.win_samples + 1, self.step_samples):
            end    = start + self.win_samples
            seg    = bvp[start:end]
            seg_lb = label_ds[start:end]

            # Majority label in window; skip if contains excluded labels
            maj_label = np.bincount(seg_lb.astype(int)).argmax()
            if maj_label in _EXCLUDED_LABELS:
                continue
            # Map WESAD labels to binary
            stress_label = 1 if maj_label == _LABEL_STRESS else 0

            # Extract physiological features
            filtered = self._bandpass(seg)
            bpm      = self._extract_bpm(filtered)
            hrv      = self._extract_hrv(filtered)

            if bpm == 0.0:
                continue  # insufficient signal quality

            # Build 11-D vector (behavioral features = 0.0 - no video)
            feat = np.zeros(11, dtype=np.float64)
            feat[7] = bpm   # BPM index
            feat[8] = hrv   # HRV index

            windows_X.append(feat)
            windows_y.append(stress_label)

        if not windows_X:
            return None

        return np.array(windows_X), np.array(windows_y)

    # -- public API ----------------------------------------------------

    def load_all(
        self,
        subject_ids: list[str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Load all available subjects.

        Parameters
        ----------
        subject_ids : optional list of subject IDs (e.g. ['S2', 'S3']).
                      If None, auto-discovers all S* folders.

        Returns
        -------
        X            : (N, 11) feature array  (9 behavioral=0, 2 physiological)
        y            : (N,)    binary labels   0=Normal, 1=Stressed
        subject_ids_ : (N,)    subject index   (for leave-one-subject-out CV)
        """
        if subject_ids is None:
            subject_ids = sorted([
                d for d in os.listdir(self.dataset_root)
                if d.startswith("S") and
                   os.path.isdir(os.path.join(self.dataset_root, d))
            ])

        all_X, all_y, all_ids = [], [], []
        for sid in subject_ids:
            result = self._load_subject(sid)
            if result is None:
                continue
            X_s, y_s = result
            all_X.append(X_s)
            all_y.append(y_s)
            all_ids.extend([sid] * len(y_s))
            logger.info("  %s: %d windows (Normal=%d, Stressed=%d)",
                        sid, len(y_s),
                        int(np.sum(y_s == 0)), int(np.sum(y_s == 1)))

        if not all_X:
            raise RuntimeError(
                f"No WESAD data found in '{self.dataset_root}'. "
                "Download from: https://uni-siegen.sciebo.de/s/HGdUkoNlW1Ub0Gu"
            )

        X    = np.vstack(all_X)
        y    = np.concatenate(all_y)
        ids  = np.array(all_ids)
        logger.info(
            "WESAD total: %d windows from %d subjects  (Normal=%d, Stressed=%d)",
            len(y), len(set(ids)), int(np.sum(y == 0)), int(np.sum(y == 1))
        )
        return X, y, ids

    def load_cached(
        self,
        cache_dir: str = "data",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Load pre-extracted WESAD features from .npy cache if available."""
        xp = os.path.join(cache_dir, "X_wesad.npy")
        yp = os.path.join(cache_dir, "y_wesad.npy")
        ip = os.path.join(cache_dir, "ids_wesad.npy")
        if os.path.exists(xp) and os.path.exists(yp):
            X   = np.load(xp)
            y   = np.load(yp)
            ids = np.load(ip) if os.path.exists(ip) else np.zeros(len(y))
            if X.shape[1] != 11:
                logger.warning("Cached WESAD has %d features, expected 11 - ignoring.", X.shape[1])
                return None
            logger.info("Loaded cached WESAD: %d windows", len(y))
            return X, y, ids
        return None

    def save_cache(
        self,
        X: np.ndarray,
        y: np.ndarray,
        ids: np.ndarray,
        cache_dir: str = "data",
    ) -> None:
        """Save extracted WESAD features to .npy files."""
        os.makedirs(cache_dir, exist_ok=True)
        np.save(os.path.join(cache_dir, "X_wesad.npy"),   X)
        np.save(os.path.join(cache_dir, "y_wesad.npy"),   y)
        np.save(os.path.join(cache_dir, "ids_wesad.npy"), ids)
        logger.info("WESAD cache saved to %s/", cache_dir)
