"""
Phase 1 Evaluation Script - Ablation + Baselines + Statistical Tests
=====================================================================
Generates the full results table for the IEEE paper.

Usage
-----
    # Run on synthetic data only (fast, ~2 min):
    python run_ablation.py

    # Include WESAD cross-dataset validation:
    python run_ablation.py --wesad /path/to/WESAD

    # Save results to CSV:
    python run_ablation.py --save results/ablation_results.csv

    # Verbose logging:
    python run_ablation.py --verbose
"""

import argparse
import logging
import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))


def main():
    parser = argparse.ArgumentParser(description="Phase 1 ablation + significance study")
    parser.add_argument("--wesad",   type=str, default=None,
                        help="Path to WESAD dataset root (optional)")
    parser.add_argument("--save",    type=str, default=None,
                        help="Path to save CSV results (e.g. results/ablation.csv)")
    parser.add_argument("--n-folds", type=int, default=5,
                        help="Number of CV folds (default: 5)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    # -- Imports ----------------------------------------------------------
    from src.fusion.classifier import StressClassifier
    from src.evaluation.ablation import AblationStudy
    from src.evaluation.stats    import SignificanceReport

    print("\n" + "=" * 70)
    print("  PHASE 1 EVALUATION - Ablation Study + Statistical Significance")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # -- Step 1: Build dataset ---------------------------------------------
    print("\n[1/4] Building dataset ...")
    X_mock, y_mock = StressClassifier.generate_mock_dataset(2000)
    logger.info("Synthetic dataset: %d samples", len(y_mock))

    # Load UBFC-Phys if available
    clf_temp = StressClassifier()
    ubfc = clf_temp.load_ubfc_features()
    if ubfc is not None:
        X_all = np.vstack([X_mock, ubfc[0]])
        y_all = np.concatenate([y_mock, ubfc[1]])
        logger.info("UBFC-Phys appended: total %d samples", len(y_all))
    else:
        X_all, y_all = X_mock, y_mock
        logger.info("UBFC-Phys not found - using synthetic only.")

    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"    Dataset: {len(y_all)} samples  "
          f"(Normal={int(np.sum(y_all==0))}, Stressed={int(np.sum(y_all==1))})")

    # -- Step 2: Ablation study ---------------------------------------------
    print(f"\n[2/4] Running ablation ({args.n_folds}-fold CV, 10 configurations) ...")
    print("    This may take 3-8 minutes ...\n")
    study  = AblationStudy(n_splits=args.n_folds)
    results = study.run(X_all, y_all)
    study.print_table(results)

    # -- Step 3: Temporal stability -----------------------------------------
    print("\n[3/4] Temporal stability ablation ...")
    # Build a synthetic sequential stress trace (Normal-Stressed-Normal blocks)
    rng = np.random.default_rng(99)
    n_per_block = 120  # 2-minute blocks at 1 Hz
    X_seq_parts, y_seq_parts = [], []
    for label in [0, 1, 0, 1, 0]:
        base_x = X_all[y_all == label]
        if len(base_x) == 0:
            continue
        idx    = rng.integers(0, len(base_x), size=n_per_block)
        block  = base_x[idx].copy()
        # Add temporal noise (smoothly correlated)
        noise  = rng.normal(0, 0.01, block.shape)
        block += noise
        X_seq_parts.append(block)
        y_seq_parts.append(np.full(n_per_block, label))
    X_seq = np.vstack(X_seq_parts)
    y_seq = np.concatenate(y_seq_parts)
    X_seq = np.nan_to_num(X_seq, nan=0.0, posinf=0.0, neginf=0.0)

    temporal = study.run_temporal_ablation(X_seq, y_seq)
    study.print_temporal_table(temporal)

    # -- Step 4: Statistical significance ----------------------------------
    print("\n[4/4] Statistical significance tests (paired t-test + Wilcoxon) ...")
    sig = SignificanceReport()
    sig_df = sig.run(results, baseline_key="A1  Full System (11D, RF+ET)", metric="f1")
    sig.print_table()

    print("\n  95% Confidence Intervals (F1-Score):")
    ci_df = sig.ci_table(results, metric="f1")
    print(ci_df.to_string(index=False))

    # -- WESAD cross-dataset (optional) ------------------------------------
    wesad_results = None
    if args.wesad:
        print(f"\n[+] WESAD cross-dataset validation ({args.wesad}) ...")
        from src.data.wesad_loader import WESADLoader
        loader = WESADLoader(dataset_root=args.wesad)
        try:
            # Try cache first
            cached = loader.load_cached()
            if cached is not None:
                X_w, y_w, ids_w = cached
            else:
                X_w, y_w, ids_w = loader.load_all()
                loader.save_cache(X_w, y_w, ids_w)

            print(f"    WESAD: {len(y_w)} windows from {len(set(ids_w))} subjects")
            print(f"    Note: 9 behavioral features = 0 (no video in WESAD)")
            print(f"    Evaluating physiological subset (BPM + HRV) only ...")

            # Evaluate using physiological features only (fair comparison)
            from src.evaluation.ablation import PHYSIO, _rf_only
            from sklearn.model_selection import StratifiedKFold, cross_validate
            from sklearn.metrics import make_scorer, f1_score, accuracy_score, roc_auc_score
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler

            X_w_physio = X_w[:, PHYSIO]
            pipe = Pipeline([("scaler", StandardScaler()), ("clf", _rf_only().named_steps["clf"])])
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            w_scores = cross_validate(
                pipe, X_w_physio, y_w, cv=cv,
                scoring={
                    "f1": make_scorer(f1_score, zero_division=0),
                    "accuracy": make_scorer(accuracy_score),
                },
                n_jobs=-1,
            )
            wesad_f1  = w_scores["test_f1"].mean()
            wesad_acc = w_scores["test_accuracy"].mean()
            wesad_results = {"f1": wesad_f1, "accuracy": wesad_acc}
            print(f"\n  WESAD Cross-Dataset (physiological only):")
            print(f"    Accuracy: {wesad_acc:.4f}  F1: {wesad_f1:.4f}")
            print(f"  (Compare against A4 Physiological-Only row above for cross-dataset gap)")
        except Exception as exc:
            logger.warning("WESAD evaluation failed: %s", exc)

    # -- Save results -------------------------------------------------------
    display_cols = ["Configuration", "Dim", "Accuracy", "Acc - std",
                    "F1-Score", "F1 - std", "ROC-AUC", "AUC - std"]
    out_data = {
        "timestamp": datetime.now().isoformat(),
        "n_samples":  int(len(y_all)),
        "n_folds":    args.n_folds,
        "ablation":   results[display_cols].to_dict(orient="records"),
        "significance": sig_df.to_dict(orient="records"),
        "temporal":   temporal,
    }
    if wesad_results:
        out_data["wesad_cross_dataset"] = wesad_results

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        # Save main table as CSV
        results[display_cols].to_csv(args.save, index=False)
        # Save significance table as sidecar
        sig_path = args.save.replace(".csv", "_significance.csv")
        sig_df.to_csv(sig_path, index=False)
        # Save JSON summary
        json_path = args.save.replace(".csv", "_summary.json")
        with open(json_path, "w") as f:
            json.dump(out_data, f, indent=2, default=str)
        print(f"\n  Saved: {args.save}")
        print(f"  Saved: {sig_path}")
        print(f"  Saved: {json_path}")

    print("\n" + "=" * 70)
    print("  Phase 1 evaluation complete.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
