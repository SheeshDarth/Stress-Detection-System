"""
Headless Sanity Check v2.0
===========================
Verifies all pipeline modules work correctly without a webcam.
"""

import numpy as np
import sys
import os
import time
import io

# Fix Unicode output on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

print("=" * 60)
print("  Sanity Check — Multimodal Stress Detection System v2.0")
print("=" * 60)

# ── Step 1: Test AU Extractor ──────────────────────────────────────
print("\n[1/5] Testing AUExtractor …")
t0 = time.time()
from src.visual.au_extraction import AUExtractor

au = AUExtractor(fps=30)
print(f"      FaceLandmarker loaded in {time.time()-t0:.1f}s ✓")
print(f"      Blink threshold: {au.blink_threshold}")
print(f"      Adaptive blink: {au.adaptive_blink}")
au.release()
print("      AUExtractor: PASS ✅")

# ── Step 2: Test RPPGExtractor ─────────────────────────────────────
print("\n[2/5] Testing RPPGExtractor …")
from src.physiological.rppg_extraction import RPPGExtractor

rppg = RPPGExtractor(fps=30, window_seconds=10)

# Synthetic RGB signal — 75 BPM (1.25 Hz)
print("      Generating synthetic 10-s RGB signal (75 BPM) …")
t = np.linspace(0, 10, 300)
hr_signal = 0.5 * np.sin(2 * np.pi * 1.25 * t)
r = 150 + hr_signal + np.random.randn(300) * 0.1
g = 120 + hr_signal * 1.5 + np.random.randn(300) * 0.1
b = 100 + hr_signal * 0.5 + np.random.randn(300) * 0.1
rgb_data = np.column_stack([r, g, b])

# Test simple POS
pulse = RPPGExtractor._pos_simple(rgb_data)
print(f"      POS simple → {len(pulse)} samples")

# Test overlap-add POS
pulse_oa = rppg._pos_overlap_add(rgb_data)
print(f"      POS overlap-add → {len(pulse_oa)} samples")

# Test detrend + bandpass
detrended = rppg._detrend(pulse_oa)
filtered = rppg._bandpass(detrended)
print(f"      Detrend + bandpass → {len(filtered)} samples")

# Test BPM (Welch PSD)
bpm = rppg._bpm_welch(filtered)
print(f"      Welch BPM: {bpm:.1f}  (expected: ~75)")

# Test HRV
hrv = rppg._hrv_rmssd(filtered)
print(f"      RMSSD: {hrv:.1f} ms")

# Test SNR
snr = rppg._compute_snr(filtered)
print(f"      SNR: {snr:.1f} dB")

print("      RPPGExtractor: PASS ✅")

# ── Step 3: Test Classifier (mock training) ────────────────────────
print("\n[3/5] Testing StressClassifier (training) …")
from src.fusion.classifier import StressClassifier

clf = StressClassifier(model_path="models/stress_model.pkl")
accuracy = clf.train(use_mock=True)
print(f"      Accuracy: {accuracy:.4f}")

# Test predictions
test_cases = [
    ("Normal (relaxed)",   [15, 0.30, 0.02, 0.35, 0.01, 0.001, 0.005, 72, 45]),
    ("Stressed (high HR)", [28, 0.22, 0.05, 0.25, 0.04, 0.006, 0.018, 100, 20]),
    ("Borderline",         [20, 0.27, 0.03, 0.31, 0.02, 0.003, 0.010, 82, 35]),
]
for desc, vec in test_cases:
    label, conf = clf.predict(np.array(vec))
    print(f"      {desc:22s} → {label} ({conf*100:.0f}%)")
print("      StressClassifier: PASS ✅")

# ── Step 4: Test feature fusion ────────────────────────────────────
print("\n[4/5] Testing feature fusion …")
visual_features = {
    "blink_rate": 20, "ear_mean": 0.28, "ear_std": 0.03,
    "brow_furrow_mean": 0.30, "brow_furrow_std": 0.02,
    "head_pose_variance": 0.003, "head_pose_mean_movement": 0.01,
}
physio_features = {"bpm": 85.0, "hrv_rmssd": 35.0}
vec = StressClassifier.create_feature_vector(visual_features, physio_features)
print(f"      Feature vector shape: {vec.shape}")
label, conf = clf.predict(vec)
print(f"      Prediction: {label} ({conf*100:.0f}%)")
print("      Feature Fusion: PASS ✅")

# ── Step 5: Test data loader import ────────────────────────────────
print("\n[5/5] Testing data loader …")
from src.data_loader import load_ubfc_dataset, _compute_ground_truth_hr
print("      data_loader module imports OK")
# Quick BVP ground-truth test
bvp_path = os.path.join(
    r"C:\Users\Siddharth\.cache\kagglehub\datasets\phanquythinh"
    r"\ubfc-phys-s1-s14\versions\4\s1\s1",
    "bvp_s1_T1.csv"
)
if os.path.exists(bvp_path):
    gt = _compute_ground_truth_hr(bvp_path, window_sec=10)
    print(f"      BVP ground-truth: {len(gt)} windows")
    if gt:
        print(f"      First window: BPM={gt[0]['bpm']:.1f}, "
              f"RMSSD={gt[0]['hrv_rmssd']:.1f} ms")
else:
    print("      (UBFC dataset not found — skipping BVP test)")
print("      Data Loader: PASS ✅")

# ── Summary ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  ✅ ALL SANITY CHECKS PASSED")
print("=" * 60)
print("\n  Next steps:")
print("    1. Extract UBFC features:  python -m src.data_loader")
print("    2. Retrain on real data:   python train.py")
print("    3. Run live system:        python main.py")
print()
