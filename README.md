# Multimodal Stress Detection System v3.0

> 🧠 Real-time, non-invasive stress & drowsiness detection using only a standard webcam.
> Combines **facial action unit analysis** with **remote photoplethysmography (rPPG)** in a privacy-first, fully on-device pipeline.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green?logo=opencv)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-f89939)
![License](https://img.shields.io/badge/License-MIT-brightgreen)
![Tests](https://img.shields.io/badge/tests-66%20passed-success)

---

## ✨ What It Does

| Capability | How |
|------------|-----|
| 💓 **Live Heart Rate** | Remote PPG via webcam green channel (Welch PSD, 50–130 BPM physiological gate) |
| 📊 **HRV (RMSSD)** | Inter-beat interval analysis with IBI outlier rejection (capped 150 ms) |
| 👁️ **Drowsiness Detection** | PERCLOS proxy — sustained low Eye Aspect Ratio > 2.5 s → amber flashing banner + 3× alarm beep |
| 🧩 **9-D Multimodal Fusion** | 7 behavioral (EAR, brow, head pose, blinks) + 2 physiological (BPM, HRV) → Random Forest |
| 🎯 **Continuous Stress Score** | Rule-based 0–100 score (BPM + HRV + AU4 brow + EAR) — always live, independent of signal quality |
| 🔄 **State Recovery** | Label-buffer vote-decay + score safety valve prevents stuck "Stressed" label |
| 🌐 **Zero Cloud** | All inference on-device. No API keys. No data leaves your machine. |
| 🔌 **Webcam Reconnect** | Automatic reconnect with exponential failure cutoff (10 retries) |
| 📄 **HTML Reports** | Interactive post-session dashboard generated from CSV logs |

---

## 🖥️ Live Demo Screenshot

<!-- Add screenshot here -->
> Run `python main.py --no-calib` for a quick test without the 15-second calibration phase.

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                     main.py  (v3.0)                            │
│                                                                │
│   Webcam ──► Worker Thread (daemon)                            │
│              │                                                │
│        ┌─────┴─────────────────────┐                          │
│        ▼                           ▼                          │
│   AUExtractor               RPPGExtractor                     │
│   (MediaPipe 468-LM)        (rPPG — Green ch.)                │
│   • EAR / Blink             • Welch PSD → BPM                 │
│   • Brow furrow (AU4)       • IBI peaks → HRV                 │
│   • Head pose               • SNR quality gate                │
│   • PERCLOS drowsiness                                        │
│        │                           │                          │
│        └───────────┬───────────────┘                          │
│                    ▼                                          │
│            StressClassifier                                   │
│            9-D feature vector                                 │
│            RandomForest (acc 96.5%)                           │
│            + Rule-based score 0–100                           │
│                    │                                          │
│         ┌──────────┼──────────┐                               │
│         ▼          ▼          ▼                               │
│      HUD (cv2)  Session  HTML Report                          │
│                  CSV log  (post-session)                      │
└────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Webcam (built-in or USB)
- Windows (for audio alerts via `winsound`; Linux/Mac require minor modification)

### 1. Clone & Install

```bash
git clone https://github.com/SheeshDarth/Stress-Detection-System.git
cd Stress-Detection-System

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

pip install -r requirements.txt
```

### 2. Download MediaPipe Model

```bash
python -c "
import urllib.request, os
os.makedirs('models', exist_ok=True)
urllib.request.urlretrieve(
    'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
    'models/face_landmarker.task'
)
print('Downloaded models/face_landmarker.task')
"
```

### 3. Run

```bash
# Full run with 15-second calibration phase (recommended)
python main.py

# Skip calibration (quick test)
python main.py --no-calib

# Use a specific camera index
python main.py --camera 1
```

### 4. Controls

| Key | Action |
|-----|--------|
| **ESC / Q** | Quit |
| **R** | Reset all signal buffers |
| **M** | Toggle face mesh overlay |
| **S** | Save screenshot |
| **L** | Start / stop CSV session logging |
| **A** | Toggle audio alerts |

---

## 🧪 Testing

```bash
# Full test suite (66 tests)
pytest tests/ -v

# Unit tests only (fast, no webcam needed)
pytest tests/test_classifier.py tests/test_rppg.py tests/test_au_extraction.py -v

# Integration test (generates synthetic video, runs full pipeline headlessly)
pytest tests/test_integration.py -v -m integration

# Headless sanity check (no webcam required)
python sanity_check.py
```

---

## 📁 Project Structure

```
Stress-Detection-System/
├── main.py                      # Real-time webcam application (v3.0)
├── train.py                     # Training pipeline
├── sanity_check.py              # Headless pipeline verification
├── download_dataset.py          # UBFC-Phys dataset downloader
├── requirements.txt
├── README.md
│
├── models/
│   ├── face_landmarker.task     # MediaPipe model (download separately)
│   ├── stress_model.pkl         # Trained classifier (auto-generated)
│   └── stress_model_metrics.json
│
├── src/
│   ├── data_loader.py           # UBFC-Phys dataset loader
│   ├── session_logger.py        # CSV session logging with finalise
│   ├── report_generator.py      # HTML post-session report generator
│   ├── visual/
│   │   └── au_extraction.py     # MediaPipe AU extraction v2.0
│   ├── physiological/
│   │   └── rppg_extraction.py   # rPPG pipeline v2.0
│   └── fusion/
│       └── classifier.py        # Multimodal fusion classifier v2.0
│
├── tests/
│   ├── test_au_extraction.py
│   ├── test_rppg.py
│   ├── test_classifier.py
│   ├── test_session_logger.py
│   └── test_integration.py      # Synthetic video integration test
│
└── logs/                        # Auto-created; session CSVs + app log
```

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | 96.57% |
| **F1 Score** | 96.66% |
| **ROC AUC** | 99.39% |
| **CV F1 (5-fold)** | 96.32% ± 1.06% |

### Feature Importances (Random Forest)

| Feature | Importance |
|---------|-----------|
| `ear_std` | 0.258 |
| `brow_furrow_std` | 0.253 |
| `head_pose_variance` | 0.170 |
| `head_pose_mean_movement` | 0.092 |
| `blink_rate` | 0.084 |
| `bpm` | 0.069 |
| `hrv_rmssd` | 0.036 |
| `brow_furrow_mean` | 0.019 |
| `ear_mean` | 0.018 |

---

## 🔬 Signal Processing Details

### Behavioral Pipeline (Visual)
1. **MediaPipe FaceLandmarker** → 468 3-D landmarks at 25+ FPS (CPU-only)
2. **Eye Aspect Ratio (EAR)** → adaptive blink threshold (calibrated from first 90 frames)
3. **Brow furrow (AU4)** → normalized inner-eyebrow distance
4. **Head pose** → nose-tip displacement variance and mean movement
5. **PERCLOS** → consecutive frames with EAR < 0.21 → drowsiness flag at 2.5 s
6. **10-second windowed aggregation** → 7 statistical features

### Physiological Pipeline (rPPG)
1. **ROI extraction** → forehead + cheek regions via convex hull of landmarks
2. **Spatial RGB mean** → per-frame in a 10-second rolling buffer
3. **POS algorithm** → Plane-Orthogonal-to-Skin projection
4. **Butterworth bandpass** → 0.83–2.17 Hz (50–130 BPM physiological gate)
5. **Welch PSD** → dominant frequency → BPM (requires ≥15% peak dominance)
6. **IBI outlier rejection** → ±25% of median, RMSSD capped at 150 ms
7. **SNR quality gate** → "Poor / Fair / Good" quality label

### Stress Scoring (Dual-Mode)
| Mode | When Active | Basis |
|------|------------|-------|
| **Rule-based score (0–100)** | Always | Physiological thresholds on BPM, HRV, brow, EAR |
| **ML binary label** | Signal ≥ "Fair" + buffer ≥ 50% | Random Forest with 75% hysteresis over 40-vote window |
| **Score override** | Score < 28 | Snaps to "Normal", clears stale ML votes immediately |

---

## 🔒 Privacy & Security

- **100% on-device** — no video, biometric data, or stress readings ever transmitted
- Session logs stored locally in `logs/` (auto-created, owner-accessible)
- Model loaded from local `.pkl` file — no network calls at inference time
- One-time internet access: MediaPipe model download and optional UBFC dataset

---

## 📚 References

- Wang, W. et al. (2017). "Algorithmic principles of remote-PPG." *IEEE TBME, 64*(7), 1479–1491
- Soukupová, T. & Čech, J. (2016). "Real-time eye blink detection using facial landmarks." *CVWW*
- de Haan, G. & Jeanne, V. (2013). "Robust pulse rate from chrominance-based rPPG." *IEEE TBME*
- Bobbia, S. et al. (2019). "Unsupervised skin tissue segmentation for rPPG." *Pattern Recognition Letters*
- UBFC-Phys Dataset — Meziati Sabour et al. (2021)

---

## 📝 License

MIT License — see [LICENSE](LICENSE) file.

---

*Built with ❤️ for IPCV Project — all inference on-device, zero cloud dependencies.*
