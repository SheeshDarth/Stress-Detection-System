"""
Training Script — Train stress classifier on UBFC-Phys + synthetic data
=========================================================================
Usage:
  python train.py                  # Synthetic data only (fast)
  python train.py --extract        # Extract features from UBFC videos + train
  python train.py --tune           # + hyperparameter tuning (slow but optimal)
"""

import argparse
import os
import sys
import numpy as np

from src.fusion.classifier import StressClassifier


def main():
    parser = argparse.ArgumentParser(description="Train the stress classifier")
    parser.add_argument("--extract", action="store_true",
                        help="Extract features from UBFC-Phys videos first")
    parser.add_argument("--tune", action="store_true",
                        help="Run GridSearchCV hyperparameter tuning")
    parser.add_argument("--dataset", type=str,
                        default=r"C:\Users\Siddharth\.cache\kagglehub\datasets"
                                r"\phanquythinh\ubfc-phys-s1-s14\versions\4",
                        help="Path to UBFC-Phys dataset root")
    parser.add_argument("--max-windows", type=int, default=12,
                        help="Max 10-s windows per video (default: 12)")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("  Stress Classifier — Training Pipeline")
    print("=" * 60)

    # Step 1: Extract features from videos if requested
    if args.extract:
        print("\n  Step 1: Extracting features from UBFC-Phys videos …")
        print("  (This may take 10–30 minutes depending on your CPU)\n")

        from src.data_loader import load_ubfc_dataset
        X, y = load_ubfc_dataset(
            dataset_path=args.dataset,
            max_windows_per_video=args.max_windows,
            use_ground_truth_hr=True,
        )

        # Save extracted features
        os.makedirs("data", exist_ok=True)
        np.save("data/X_ubfc.npy", X)
        np.save("data/y_ubfc.npy", y)
        print(f"\n  Saved: data/X_ubfc.npy ({X.shape})")
        print(f"  Saved: data/y_ubfc.npy ({y.shape})")
    else:
        print("\n  Step 1: Skipping video extraction (use --extract to enable)")

    # Step 2: Train classifier
    print("\n  Step 2: Training classifier …\n")

    clf = StressClassifier(model_path="models/stress_model.pkl")
    accuracy = clf.train(
        use_mock=True,
        tune_hyperparams=args.tune,
    )

    print(f"\n  Training complete! Accuracy: {accuracy:.4f}")
    print(f"  Model saved to: models/stress_model.pkl")
    print(f"  Metrics saved to: models/stress_model_metrics.json")
    print(f"\n  Run `python main.py` to start the live system.\n")


if __name__ == "__main__":
    main()
