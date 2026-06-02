"""
Ablation Study - Systematic component analysis for IEEE paper  (Phase 1)
=========================================================================
Tests 5 feature-ablation configs + 5 baseline models via 5-fold stratified
cross-validation, then compares every configuration against the full system
using paired t-tests and Wilcoxon signed-rank tests.

Feature index map (11-D schema)
--------------------------------
0  blink_rate              behavioral
1  ear_mean                behavioral
2  ear_std                 behavioral
3  brow_furrow_mean        behavioral
4  brow_furrow_std         behavioral
5  head_pose_variance      behavioral
6  head_pose_mean_movement behavioral
7  bpm                     physiological
8  hrv_rmssd               physiological
9  lip_depression_mean     behavioral (v3.1)
10 jaw_clenching_std       behavioral (v3.1)

Ablation Configurations
------------------------
A1  full_system           All 11 features  +  RF+ET ensemble
A2  no_lip_jaw            9-D (drop features 9-10)
A3  behavioral_only       7-D behavioral  (drop BPM, HRV)
A4  physiological_only    2-D (BPM + HRV only)
A5  rf_only               11-D  +  single RF  (no ExtraTrees)

Baseline Models (same 11-D features)
--------------------------------------
B1  svm_rbf               RBF-kernel SVM
B2  logistic_regression   L2-penalised logistic regression
B3  knn_5                 k-NN  (k=5, Euclidean)
B4  au_only_rf            9-D behavioral subset  +  RF
B5  bpm_threshold         Hard threshold on BPM alone (no ML)

Usage
-----
    from src.fusion.classifier import StressClassifier
    X, y = StressClassifier.generate_mock_dataset(2000)

    from src.evaluation.ablation import AblationStudy
    study = AblationStudy()
    results_df = study.run(X, y)
    study.print_table(results_df)
"""

import logging
import numpy as np
import pandas as pd

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    VotingClassifier,
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer, f1_score, roc_auc_score, accuracy_score

logger = logging.getLogger(__name__)

# -- Feature index constants -----------------------------------------------
ALL_11   = list(range(11))
NINE_D   = list(range(9))                    # drop lip(9) + jaw(10)
BEHAV    = [0, 1, 2, 3, 4, 5, 6, 9, 10]     # drop BPM(7) + HRV(8)
PHYSIO   = [7, 8]                             # BPM + HRV only
BEHAV_9D = list(range(7))                    # 7 original behavioral only


# -- Pipeline factories ----------------------------------------------------

def _rf_et():
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_split=5,
        min_samples_leaf=2, max_features="sqrt",
        random_state=42, n_jobs=-1, class_weight="balanced",
    )
    et = ExtraTreesClassifier(
        n_estimators=200, max_depth=12, min_samples_split=5,
        min_samples_leaf=2, max_features="sqrt",
        random_state=43, n_jobs=-1, class_weight="balanced",
    )
    ensemble = VotingClassifier(
        estimators=[("rf", rf), ("et", et)], voting="soft", n_jobs=-1
    )
    return Pipeline([("scaler", StandardScaler()), ("clf", ensemble)])


def _rf_only():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_split=5,
            min_samples_leaf=2, max_features="sqrt",
            random_state=42, n_jobs=-1, class_weight="balanced",
        )),
    ])


def _svm():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            kernel="rbf", C=10, gamma="scale",
            probability=True, class_weight="balanced", random_state=42,
        )),
    ])


def _lr():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=1.0, max_iter=1000, class_weight="balanced", random_state=42,
        )),
    ])


def _knn():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(n_neighbors=5, metric="euclidean", n_jobs=-1)),
    ])


class _BPMThreshold(BaseEstimator, ClassifierMixin):
    """Hard-threshold baseline: stressed if BPM > threshold learned from training fold."""

    def __init__(self, bpm_col: int = 7):
        self.bpm_col = bpm_col
        self.threshold_ = 85.0

    def fit(self, X, y):
        bpm = X[:, self.bpm_col]
        # Sweep thresholds and pick best F1 on training fold
        best_f1, best_t = 0.0, 85.0
        for t in np.linspace(bpm.min(), bpm.max(), 50):
            preds = (bpm > t).astype(int)
            score = f1_score(y, preds, zero_division=0)
            if score > best_f1:
                best_f1, best_t = score, t
        self.threshold_ = best_t
        return self

    def predict(self, X):
        return (X[:, self.bpm_col] > self.threshold_).astype(int)

    def predict_proba(self, X):
        bpm = X[:, self.bpm_col]
        prob = np.clip((bpm - self.threshold_) / 30.0 + 0.5, 0.0, 1.0)
        return np.column_stack([1 - prob, prob])


def _bpm_threshold():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", _BPMThreshold()),
    ])


# -- Configuration registry ------------------------------------------------

CONFIGS = {
    # -- Ablation of proposed system --
    "A1  Full System (11D, RF+ET)":      {"idx": ALL_11,   "pipe": _rf_et},
    "A2  No Lip/Jaw (9D, RF+ET)":        {"idx": NINE_D,   "pipe": _rf_et},
    "A3  Behavioral Only (9D, RF+ET)":   {"idx": BEHAV,    "pipe": _rf_et},
    "A4  Physiological Only (2D, RF+ET)":{"idx": PHYSIO,   "pipe": _rf_et},
    "A5  RF Only (11D, no ExtraTrees)":  {"idx": ALL_11,   "pipe": _rf_only},
    # -- Baselines --
    "B1  SVM-RBF (11D)":                 {"idx": ALL_11,   "pipe": _svm},
    "B2  Logistic Regression (11D)":     {"idx": ALL_11,   "pipe": _lr},
    "B3  k-NN k=5 (11D)":               {"idx": ALL_11,   "pipe": _knn},
    "B4  AU-Only RF (7D behav)":         {"idx": BEHAV_9D, "pipe": _rf_only},
    "B5  BPM Threshold (1D)":            {"idx": ALL_11,   "pipe": _bpm_threshold},
}


# -- Ablation runner -------------------------------------------------------

class AblationStudy:
    """
    Runs all configurations on dataset (X, y) using k-fold CV.
    Returns a DataFrame suitable for a LaTeX/Word results table.
    """

    def __init__(self, n_splits: int = 5, random_state: int = 42):
        self.n_splits    = n_splits
        self.random_state = random_state

    # -- core run ---------------------------------------------------------

    def run(self, X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        """
        Evaluate all configurations.
        Returns DataFrame with per-config mean/std metrics
        and the raw fold scores (needed for significance tests).
        """
        cv = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True,
            random_state=self.random_state,
        )
        # Use sklearn's built-in string scorer for roc_auc so it gracefully
        # falls back to decision_function / predict_proba as available.
        scorers = {
            "accuracy": make_scorer(accuracy_score),
            "f1":       make_scorer(f1_score, zero_division=0),
            "roc_auc":  "roc_auc",
        }

        rows = []
        for name, cfg in CONFIGS.items():
            logger.info("Running: %s ...", name)
            X_sub = X[:, cfg["idx"]]
            pipe  = cfg["pipe"]()
            cv_out = cross_validate(
                pipe, X_sub, y,
                cv=cv, scoring=scorers, n_jobs=-1,
            )
            rows.append({
                "Configuration":    name,
                "Dim":              len(cfg["idx"]),
                "Accuracy":         f'{cv_out["test_accuracy"].mean():.4f}',
                "Acc - std":        f'-{cv_out["test_accuracy"].std():.4f}',
                "F1-Score":         f'{cv_out["test_f1"].mean():.4f}',
                "F1 - std":         f'-{cv_out["test_f1"].std():.4f}',
                "ROC-AUC":          f'{cv_out["test_roc_auc"].mean():.4f}',
                "AUC - std":        f'-{cv_out["test_roc_auc"].std():.4f}',
                # raw folds for significance tests
                "_f1_folds":        cv_out["test_f1"],
                "_acc_folds":       cv_out["test_accuracy"],
                "_auc_folds":       cv_out["test_roc_auc"],
            })
            logger.info(
                "  Acc=%-10s F1=%-10s AUC=%s",
                rows[-1]["Accuracy"] + rows[-1]["Acc - std"],
                rows[-1]["F1-Score"] + rows[-1]["F1 - std"],
                rows[-1]["ROC-AUC"] + rows[-1]["AUC - std"],
            )

        return pd.DataFrame(rows)

    # -- temporal stability ablation ---------------------------------------

    def run_temporal_ablation(
        self,
        X_seq: np.ndarray,
        y_seq: np.ndarray,
        buffer_size: int = 40,
        majority_threshold: float = 0.75,
        override_score_threshold: float = 28.0,
    ) -> dict:
        """
        Evaluates temporal hysteresis and rule-based override on a
        sequential test set (y_seq has block structure: Normal-Stressed-Normal).

        Metrics reported:
          - flip_rate:      state changes per minute (lower = more stable)
          - transition_lag: frames until correct state after a ground-truth change
          - lock_in_rate:   fraction of time stuck in wrong state for > 10 frames
        """
        # Train a quick RF on 80% of data
        split = int(0.8 * len(X_seq))
        X_tr, X_te = X_seq[:split], X_seq[split:]
        y_tr, y_te = y_seq[:split], y_seq[split:]

        scaler = StandardScaler().fit(X_tr)
        X_tr_s = scaler.transform(X_tr)
        X_te_s = scaler.transform(X_te)

        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_tr_s, y_tr)
        raw_preds = rf.predict(X_te_s)
        raw_probs = rf.predict_proba(X_te_s)[:, 1]

        def _with_hysteresis(preds):
            buf, state = [], "Normal"
            out = []
            for p in preds:
                buf.append("Stressed" if p == 1 else "Normal")
                if len(buf) > buffer_size:
                    buf.pop(0)
                if len(buf) >= 15:
                    stressed_pct = buf.count("Stressed") / len(buf)
                    if stressed_pct > majority_threshold:
                        state = "Stressed"
                    elif stressed_pct < (1 - majority_threshold):
                        state = "Normal"
                out.append(state)
            return np.array([1 if s == "Stressed" else 0 for s in out])

        def _without_hysteresis(preds):
            return preds.copy()

        def _with_override(preds, probs):
            # Simulate: if "physiological calm" (low prob), snap to Normal
            out = preds.copy()
            for i, prob in enumerate(probs):
                if prob < (override_score_threshold / 100.0):
                    out[i] = 0
            return out

        def _stability_metrics(preds, y_true, fps=1):
            flips = np.sum(np.diff(preds) != 0)
            total_minutes = len(preds) / (fps * 60)
            flip_rate = flips / max(total_minutes, 1e-6)

            # Transition lag: frames from ground-truth change until pred catches up
            lags = []
            for i in range(1, len(y_true)):
                if y_true[i] != y_true[i - 1]:
                    for j in range(i, min(i + 60, len(preds))):
                        if preds[j] == y_true[i]:
                            lags.append(j - i)
                            break

            # Lock-in rate: consecutive wrong frames > 10
            wrong = (preds != y_true).astype(int)
            lock_count = 0
            run = 0
            for w in wrong:
                if w:
                    run += 1
                    if run > 10:
                        lock_count += 1
                else:
                    run = 0
            lock_rate = lock_count / max(len(preds), 1)

            return {
                "flip_rate":      round(flip_rate, 3),
                "transition_lag": round(np.mean(lags) if lags else 0, 1),
                "lock_in_rate":   round(lock_rate, 4),
                "accuracy":       round((preds == y_true).mean(), 4),
            }

        hysteresis_preds    = _with_hysteresis(raw_preds)
        no_hysteresis_preds = _without_hysteresis(raw_preds)
        with_override_preds = _with_override(hysteresis_preds, raw_probs)
        no_override_preds   = hysteresis_preds.copy()

        y_binary = y_te.astype(int)
        return {
            "with_hysteresis_and_override": _stability_metrics(with_override_preds, y_binary),
            "with_hysteresis_no_override":  _stability_metrics(no_override_preds, y_binary),
            "no_hysteresis_no_override":    _stability_metrics(no_hysteresis_preds, y_binary),
        }

    # -- display helpers ---------------------------------------------------

    @staticmethod
    def print_table(df: pd.DataFrame) -> None:
        """Pretty-print results table (console)."""
        display_cols = ["Configuration", "Dim", "Accuracy", "Acc - std",
                        "F1-Score", "F1 - std", "ROC-AUC", "AUC - std"]
        table = df[display_cols].copy()
        sep = "-" * 110
        print(f"\n{sep}")
        print(f"  ABLATION & BASELINE RESULTS  ({len(df)} configurations, 5-fold stratified CV)")
        print(sep)
        print(table.to_string(index=False))
        print(sep)

    @staticmethod
    def print_temporal_table(temporal: dict) -> None:
        """Pretty-print temporal stability table."""
        print("\n-- Temporal Stability Ablation --------------------------")
        print(f"{'Configuration':<40} {'Acc':>6} {'Flips/min':>10} {'Lag (s)':>9} {'Lock-in':>9}")
        print("-" * 80)
        for name, m in temporal.items():
            print(f"{name:<40} {m['accuracy']:>6.4f} {m['flip_rate']:>10.3f} "
                  f"{m['transition_lag']:>9.1f} {m['lock_in_rate']:>9.4f}")
        print("-" * 80)
