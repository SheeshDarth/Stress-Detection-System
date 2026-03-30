"""
UBFC-Phys Dataset Loader & Feature Extractor
=============================================
Processes the UBFC-Phys dataset (subjects s1–s14) to extract
ground-truth labelled feature vectors for classifier training.

Dataset structure (per subject)
-------------------------------
  sN/sN/
    vid_sN_T1.avi         — baseline/rest video   (label: 0 = Normal)
    vid_sN_T2.avi         — speech task video      (label: 1 = Stressed)
    vid_sN_T3.avi         — arithmetic task video  (label: 1 = Stressed)
    bvp_sN_T1.csv         — blood-volume-pulse @ 64 Hz
    bvp_sN_T2.csv         — blood-volume-pulse @ 64 Hz
    bvp_sN_T3.csv         — blood-volume-pulse @ 64 Hz
    eda_sN_T1.csv         — electrodermal activity
    selfReportedAnx_sN.csv— 3 rows (T1,T2,T3) × 2 anxiety columns

Labelling strategy
------------------
  • T1 → Normal  (0)
  • T2 → Stressed (1)  — speech stress task
  • T3 → Stressed (1)  — arithmetic stress task
  • Additionally validated with self-reported anxiety scores:
    if mean anxiety > threshold → Stressed, else Normal.
"""

import os
import csv
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Generator

from src.visual.au_extraction import AUExtractor
from src.physiological.rppg_extraction import RPPGExtractor


# ── Constants ─────────────────────────────────────────────────────────
DEFAULT_DATASET_PATH = (
    r"C:\Users\Siddharth\.cache\kagglehub\datasets"
    r"\phanquythinh\ubfc-phys-s1-s14\versions\4"
)
WINDOW_SECONDS = 10
BVP_SAMPLE_RATE = 64      # Hz (UBFC-Phys BVP ground truth)
ANXIETY_THRESHOLD = 2.5   # Mean anxiety score above this → stressed


def _load_anxiety_scores(subject_dir: str, subject_id: str) -> list[float]:
    """Load the mean anxiety score for each task (T1, T2, T3)."""
    path = os.path.join(subject_dir, f"selfReportedAnx_{subject_id}.csv")
    scores = []
    if os.path.exists(path):
        with open(path) as f:
            reader = csv.reader(f)
            for row in reader:
                mean_score = np.mean([float(x) for x in row if x.strip()])
                scores.append(mean_score)
    return scores


def _compute_ground_truth_hr(bvp_path: str, window_sec: int = 10) -> list[dict]:
    """
    Compute ground-truth BPM + HRV-RMSSD from the BVP signal
    in non-overlapping windows.
    """
    if not os.path.exists(bvp_path):
        return []

    bvp = pd.read_csv(bvp_path, header=None).values.flatten()
    window_samples = BVP_SAMPLE_RATE * window_sec
    metrics = []

    for start in range(0, len(bvp) - window_samples, window_samples):
        segment = bvp[start:start + window_samples]

        # BPM from FFT of BVP signal
        windowed = segment * np.hanning(len(segment))
        fft_mag = np.abs(np.fft.rfft(windowed))
        freqs = np.fft.rfftfreq(len(windowed), d=1.0 / BVP_SAMPLE_RATE)
        valid = (freqs >= 0.7) & (freqs <= 4.0)
        if valid.any():
            bpm = float(freqs[valid][np.argmax(fft_mag[valid])] * 60.0)
        else:
            bpm = 0.0

        # Inter-beat intervals from BVP peaks → RMSSD
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(segment, distance=int(BVP_SAMPLE_RATE * 0.3))
        if len(peaks) >= 3:
            ibi_ms = np.diff(peaks) / BVP_SAMPLE_RATE * 1000.0
            valid_ibi = ibi_ms[(ibi_ms > 250) & (ibi_ms < 1500)]
            if len(valid_ibi) >= 2:
                rmssd = float(np.sqrt(np.mean(np.diff(valid_ibi) ** 2)))
            else:
                rmssd = 0.0
        else:
            rmssd = 0.0

        metrics.append({"bpm": bpm, "hrv_rmssd": rmssd})

    return metrics


def extract_features_from_video(
    video_path: str,
    max_windows: int = 15,
) -> list[dict]:
    """
    Process a video file and extract windowed behavioural + physiological
    feature vectors.

    Creates fresh AU and rPPG extractors per video to guarantee
    monotonically increasing timestamps for FaceLandmarker.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"    ⚠ Cannot open: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 35.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    window_frames = int(fps * WINDOW_SECONDS)

    # Create FRESH extractors for each video (timestamp must be monotonic)
    au = AUExtractor(fps=int(fps))
    rppg = RPPGExtractor(fps=int(fps), window_seconds=WINDOW_SECONDS)

    features_list = []
    frame_idx = 0
    skip_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            au_result = au.process_frame(frame_rgb)

            if au_result is not None:
                rppg.process_frame(frame_rgb, au_result["landmarks"])
        except Exception:
            skip_count += 1
            if skip_count > 50:
                print(f"    ⚠ Too many errors, stopping early")
                break
            continue

        frame_idx += 1

        # Every window_frames, try to extract a feature vector
        if frame_idx > 0 and frame_idx % window_frames == 0:
            vis = au.get_window_features(WINDOW_SECONDS)
            if vis is not None:
                physio = rppg.get_current_metrics()
                features_list.append({
                    "visual": vis,
                    "physio": {
                        "bpm": physio["bpm"],
                        "hrv_rmssd": physio["hrv_rmssd"],
                    },
                })

            if len(features_list) >= max_windows:
                break

    cap.release()
    au.release()
    return features_list


def load_ubfc_dataset(
    dataset_path: str = DEFAULT_DATASET_PATH,
    max_windows_per_video: int = 12,
    use_ground_truth_hr: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the UBFC-Phys dataset and extract labelled feature vectors.

    Parameters
    ----------
    dataset_path : str
        Root directory containing s1/, s2/, … folders.
    max_windows_per_video : int
        Maximum 10-s windows to extract per video (limits processing time).
    use_ground_truth_hr : bool
        If True, use BVP-derived BPM/HRV instead of rPPG estimates
        for more accurate training labels.

    Returns
    -------
    X : np.ndarray  — shape (N, 9), feature vectors
    y : np.ndarray  — shape (N,),   binary labels (0=Normal, 1=Stressed)
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at: {dataset_path}")

    subjects = sorted(
        d for d in os.listdir(dataset_path)
        if os.path.isdir(os.path.join(dataset_path, d))
    )
    print(f"\n  Found {len(subjects)} subjects: {subjects}")

    all_X, all_y = [], []

    for subj in subjects:
        subj_dir = os.path.join(dataset_path, subj, subj)
        if not os.path.isdir(subj_dir):
            continue

        print(f"\n  ── Processing {subj} ──")

        # Load anxiety scores for labelling validation
        anxiety = _load_anxiety_scores(subj_dir, subj)

        for task_idx, task in enumerate(["T1", "T2", "T3"]):
            vid_path = os.path.join(subj_dir, f"vid_{subj}_{task}.avi")
            bvp_path = os.path.join(subj_dir, f"bvp_{subj}_{task}.csv")

            if not os.path.exists(vid_path):
                print(f"    Skipping {task} (no video)")
                continue

            # Label: T1 = Normal (0), T2/T3 = Stressed (1)
            label = 0 if task == "T1" else 1

            # Validate with self-reported anxiety
            if task_idx < len(anxiety):
                anx_score = anxiety[task_idx]
                anx_label = 1 if anx_score >= ANXIETY_THRESHOLD else 0
                if anx_label != label and task != "T1":
                    print(f"    Note: {task} anxiety={anx_score:.2f} "
                          f"(below threshold, but keeping stress label)")

            print(f"    {task} → label={'Stressed' if label else 'Normal'} "
                  f"| video: {os.path.basename(vid_path)}")

            # Extract visual features from video
            feats = extract_features_from_video(
                vid_path,
                max_windows=max_windows_per_video,
            )

            # Get ground-truth HR from BVP if requested
            gt_hr = []
            if use_ground_truth_hr:
                gt_hr = _compute_ground_truth_hr(bvp_path, WINDOW_SECONDS)

            for i, feat in enumerate(feats):
                vis = feat["visual"]

                # Use ground-truth BPM/HRV when available
                if use_ground_truth_hr and i < len(gt_hr) and gt_hr[i]["bpm"] > 0:
                    bpm = gt_hr[i]["bpm"]
                    hrv = gt_hr[i]["hrv_rmssd"]
                else:
                    bpm = feat["physio"]["bpm"]
                    hrv = feat["physio"]["hrv_rmssd"]

                vec = np.array([
                    vis["blink_rate"],
                    vis["ear_mean"],
                    vis["ear_std"],
                    vis["brow_furrow_mean"],
                    vis["brow_furrow_std"],
                    vis["head_pose_variance"],
                    vis["head_pose_mean_movement"],
                    bpm,
                    hrv,
                ])

                all_X.append(vec)
                all_y.append(label)

            print(f"    → Extracted {len(feats)} windows")

    X = np.array(all_X, dtype=np.float64)
    y = np.array(all_y, dtype=np.int32)

    print(f"\n  ═══════════════════════════════════════════")
    print(f"  Dataset Summary")
    print(f"  ═══════════════════════════════════════════")
    print(f"  Total samples : {len(y)}")
    print(f"  Normal  (0)   : {np.sum(y == 0)}")
    print(f"  Stressed (1)  : {np.sum(y == 1)}")
    print(f"  Feature shape : {X.shape}")
    print(f"  ═══════════════════════════════════════════\n")

    return X, y


if __name__ == "__main__":
    X, y = load_ubfc_dataset()
    np.save("data/X_ubfc.npy", X)
    np.save("data/y_ubfc.npy", y)
    print(f"Saved to data/X_ubfc.npy and data/y_ubfc.npy")
