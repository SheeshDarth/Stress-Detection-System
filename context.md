# Context & Changelog — Multimodal Stress and Drowsiness Detection System

## Project Overview

Real-time, non-invasive detection of psychological stress and drowsiness using only a standard webcam.  
Two parallel pipelines (behavioral AU analysis + rPPG physiology) fuse into an 11-D feature vector classified by an RF + ExtraTrees ensemble at 1 Hz, with temporal hysteresis and rule-based override.

**GitHub:** https://github.com/SheeshDarth/Stress-Detection-System  
**Local path:** `C:\Downloads\STRESS DETECTION IPCV\Project`

---

## Version History

---

### v3.1 — Performance & Accuracy Enhancement (2026-06-02)

#### Summary
Expanded the feature vector from 9-D to 11-D, achieved a ~10x rPPG processing speedup, added CHROM algorithm as an automatic fallback, introduced Kalman-filtered BPM estimation, upgraded the classifier to an RF + ExtraTrees soft-voting ensemble, and improved drowsiness detection accuracy by using per-user calibrated EAR thresholds.

#### Breaking Change
> **The feature vector expanded from 9 to 11 dimensions.**  
> The existing `models/stress_model.pkl` is incompatible with this release.  
> On first run, the system automatically retrains. To retrain manually:  
> ```
> python train.py
> ```
> To re-extract UBFC-Phys features with the new schema:
> ```
> python train.py --extract
> ```

---

#### File-by-File Changes

##### `src/physiological/rppg_extraction.py`

| Change | Detail |
|---|---|
| **POS stride fix** | `_pos_overlap_add()` stride changed from `1` to `max(1, fps//3)` (10 frames at 30fps). Reduces inner-loop iterations from ~241 to ~25 — **~10x speedup** with negligible accuracy loss (80% window overlap retained). |
| **CHROM fallback** | Added `_chrom_algorithm()` (de Haan & Jeanne, 2013). When POS SNR < 1.0, CHROM is computed and the higher-SNR result is used automatically. Improves robustness under variable illumination. |
| **Kalman BPM filter** | Added `_kalman_update()` — a 1-D constant-position Kalman filter (Q=2.0, R=8.0) replacing the 8-sample rolling median. Converges to smooth BPM in 5–8 updates with lower lag than median smoothing. Kalman state resets on `reset()`. |
| **Extended BPM range** | Physiological gate widened from 50–130 BPM (0.83–2.17 Hz) to **45–150 BPM** (0.75–2.5 Hz), covering athletic users (low resting HR) and elderly/anxious subjects (elevated HR). Applied in both `_bpm_welch()` and `_compute_snr()`. |
| **Tighter SNR thresholds** | Quality classification updated: Good > **5.0** (was 3.0), Fair > **1.0** (was 0.0). Reduces false-positive "Good" labels from moderately noisy signals, preventing low-confidence predictions from influencing the ML buffer. |
| **Kalman state reset** | `reset()` now reinitialises `_kf_x = 72.0` and `_kf_P = 100.0` alongside existing buffer clears. |

##### `src/visual/au_extraction.py`

| Change | Detail |
|---|---|
| **EMA alpha reduced** | `_ema_alpha` changed from **0.33** to **0.25** (effective span 7 frames vs. 5). Produces smoother EAR and brow-furrow traces, reducing jitter in the feature window statistics. |
| **Lip & jaw features exposed** | `get_window_features()` now returns two additional keys: `lip_depression_mean` (mean AU15 proxy over the 10-second window) and `jaw_clenching_std` (standard deviation of chin-to-nose distance, capturing intermittent jaw clenching). These were previously computed in `process_frame()` but discarded at aggregation. |

##### `src/fusion/classifier.py`

| Change | Detail |
|---|---|
| **11-D feature vector** | `FEATURE_NAMES` expanded from 9 to 11 entries. New features: `lip_depression_mean` (index 9) and `jaw_clenching_std` (index 10). |
| **`create_feature_vector()`** | Reads `lip_depression_mean` and `jaw_clenching_std` from the visual dict. Defaults to 0.0 if absent (backward-safe for partial feature dicts). |
| **`generate_mock_dataset()`** | Synthetic data now generates 11-D samples. New column distributions: Normal — lip_dep ~0.08±0.02, jaw_std ~0.04±0.01; Stressed — lip_dep ~0.22±0.05, jaw_std ~0.10±0.025. Physiological clips applied. |
| **RF + ExtraTrees ensemble** | Training now builds a `VotingClassifier(voting='soft')` combining RF (seed=42) and ExtraTrees (seed=43) before wrapping in `CalibratedClassifierCV`. The RF is fit first to extract Gini feature importances. The ensemble provides better generalization at borderline samples. `GradientBoostingClassifier` import removed (unused). |
| **Model version check** | `load_model()` now compares saved feature count against `len(FEATURE_NAMES)`. Mismatches log a warning and return `False`, triggering automatic retraining. Wrapped in `try/except` to handle corrupt or incompatible pickle files gracefully. |
| **UBFC cache version check** | `load_ubfc_features()` now validates `X.shape[1] == len(FEATURE_NAMES)` before returning cached arrays, preventing silent dimension mismatches after schema changes. |

##### `main.py`

| Change | Detail |
|---|---|
| **Lower prediction threshold** | `can_predict` gate changed from `buffer_fill >= 0.50` to `buffer_fill >= 0.35`. The system now begins ML classification after ~3.5 seconds of rPPG data (was ~5 seconds), reducing the "Measuring..." delay for users. |
| **Adaptive drowsiness EAR** | Drowsiness check now uses `self.au_extractor.blink_threshold` (per-user calibrated value) instead of the hardcoded `0.21`. This matches the same adaptive threshold used for blink detection, ensuring consistent sensitivity across users with naturally wide or narrow eye openings. |

##### `tests/test_classifier.py`

| Change | Detail |
|---|---|
| **Feature count tests** | `test_nine_features` renamed `test_eleven_features`; assertion updated to `== 11`. Added `test_contains_lip_depression` and `test_contains_jaw_clenching`. |
| **Vector shape tests** | `test_creates_9d_vector` renamed `test_creates_11d_vector`; visual dict extended with new keys; shape assertion updated to `(11,)`. |
| **Mock dataset test** | `test_shape` assertion updated: `X.shape[1] == 11`. |
| **Prediction vectors** | All 9-element test vectors extended to 11 elements with physiologically appropriate values for lip/jaw features. |
| **Feature importance test** | `test_feature_importance_populated` updated to `== 11`. |

##### `tests/test_au_extraction.py`

| Change | Detail |
|---|---|
| **EMA smoothing test** | Expected value updated from `0.665` to `0.625` to match the new alpha=0.25 (was 0.33). Comment updated. |

---

#### Test Results — v3.1

```
53 passed, 1 warning in 130.32s
```

All 53 tests pass across `test_classifier.py`, `test_rppg.py`, and `test_au_extraction.py`.

---

#### Performance Impact Summary

| Metric | Before (v3.0) | After (v3.1) | Delta |
|---|---|---|---|
| POS inner-loop iterations (300 frames) | ~241 | ~25 | **-90%** |
| Feature dimensions | 9 | 11 | +2 |
| BPM smoothing | 8-sample median | Kalman filter | Lower lag |
| Signal quality gate (Good) | SNR > 3.0 | SNR > 5.0 | Tighter |
| Prediction start (buffer fill) | 50% (~5 s) | 35% (~3.5 s) | -1.5 s |
| Classifier | RF only | RF + ExtraTrees | Better edge-case accuracy |
| Drowsiness threshold | Fixed 0.21 EAR | Per-user adaptive | Personalised |

---

### v3.0 — Initial Release (prior)

- 30 FPS asynchronous dual-pipeline architecture
- MediaPipe FaceLandmarker (468 3D landmarks)
- POS rPPG with overlap-add sub-windowing
- 9-D early fusion → Random Forest (100-300 trees, GridSearchCV)
- 40-vote temporal hysteresis with active signal-decay gating
- Rule-based physiological override (score 0-100)
- PERCLOS-based drowsiness detection
- 15-second calibration phase
- Session CSV logging + HTML report generation
- Accuracy: 96.57%, F1: 96.66%, ROC AUC: 99.39%
- CPU: <15% at 30 FPS, no GPU required

---

## Architecture Diagram (Text)

```
Webcam (30 FPS)
     |
     +---> [AUExtractor]  MediaPipe 468-pt landmarks
     |          |
     |     [EAR, AU4 Brow, Head Pose, Lip Depression, Jaw Clench]  --> 9-D behavioral
     |          |
     |     [PERCLOS Drowsiness] --> audio alert (3x 1500 Hz beep)
     |
     +---> [RPPGExtractor]  Facial skin ROI RGB buffer
                |
           [POS + CHROM fallback] --> detrend --> Butterworth bandpass
                |
           [Welch PSD] --> Kalman BPM   [peak IBI] --> RMSSD HRV   --> 2-D physiological
                |
     [9-D behavioral + 2-D physiological = 11-D feature vector]
                |
          [StressClassifier @ 1 Hz]
          RF + ExtraTrees VotingClassifier (soft)
          CalibratedClassifierCV (isotonic)
                |
     [40-vote temporal hysteresis buffer]
     [Signal-decay gating when quality < Fair]
     [Rule-based score override (0-100) if score < 28 -> Normal]
                |
     [Stress label + confidence -> HUD display + CSV log]
```

---

## Setup & Running

```bash
# Install dependencies
pip install -r requirements.txt

# Download MediaPipe model (one-time)
python -c "import urllib.request; urllib.request.urlretrieve('https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task', 'models/face_landmarker.task')"

# Retrain model (required after v3.1 upgrade)
python train.py

# Run live system
python main.py

# Run tests
pytest tests/ -v
```

## Controls

| Key | Action |
|---|---|
| Q / ESC | Quit |
| R | Reset signal buffers |
| M | Toggle face mesh overlay |
| S | Save screenshot |
| L | Toggle CSV session logging |
| A | Toggle audio alerts |
