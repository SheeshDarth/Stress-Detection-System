"""
Statistical Significance Testing - Phase 1 (IEEE paper)
=========================================================
Compares the proposed full system against every other configuration
using paired tests across cross-validation folds.

Tests applied
-------------
  1. Paired t-test (parametric)    - assumes approximately normal fold scores
  2. Wilcoxon signed-rank test     - non-parametric alternative (robust, small N)

Significance levels
-------------------
  ***  p < 0.001
  **   p < 0.01
  *    p < 0.05
  ns   p >= 0.05

Usage
-----
    from src.evaluation.stats import SignificanceReport
    report = SignificanceReport()
    report.run(results_df, baseline_key="A1  Full System (11D, RF+ET)")
    report.print_table()
"""

import logging
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

logger = logging.getLogger(__name__)

_LEVELS = [(0.001, "***"), (0.01, "**"), (0.05, "*")]


def _stars(p: float) -> str:
    for threshold, label in _LEVELS:
        if p < threshold:
            return label
    return "ns"


class SignificanceReport:
    """Run paired statistical tests vs. the full-system baseline."""

    def __init__(self):
        self.results_: pd.DataFrame | None = None

    def run(
        self,
        ablation_df: pd.DataFrame,
        baseline_key: str = "A1  Full System (11D, RF+ET)",
        metric: str = "f1",
    ) -> pd.DataFrame:
        """
        Compare every config's CV fold scores against the baseline config.

        Parameters
        ----------
        ablation_df : DataFrame returned by AblationStudy.run()
        baseline_key : exact Configuration string of the proposed system row
        metric : 'f1' | 'accuracy' | 'roc_auc'

        Returns
        -------
        DataFrame with one row per non-baseline config showing
        mean difference, t-statistic, Wilcoxon statistic, p-values.
        """
        fold_col = f"_{metric}_folds"
        baseline_row = ablation_df[ablation_df["Configuration"] == baseline_key]
        if baseline_row.empty:
            raise ValueError(f"Baseline '{baseline_key}' not found in DataFrame.")

        base_folds = np.array(baseline_row.iloc[0][fold_col])

        rows = []
        for _, row in ablation_df.iterrows():
            if row["Configuration"] == baseline_key:
                continue

            other_folds = np.array(row[fold_col])
            diff = base_folds - other_folds
            mean_diff = float(diff.mean())

            # Paired t-test
            if np.std(diff) == 0:
                t_stat, p_t = 0.0, 1.0
            else:
                t_stat, p_t = ttest_rel(base_folds, other_folds)

            # Wilcoxon signed-rank (requires non-zero differences)
            try:
                w_stat, p_w = wilcoxon(base_folds, other_folds)
            except ValueError:
                w_stat, p_w = 0.0, 1.0

            rows.append({
                "Configuration":     row["Configuration"],
                f"- {metric.upper()} (vs full)": f"{mean_diff:+.4f}",
                "t-stat":            f"{t_stat:.3f}",
                "p (t-test)":        f"{p_t:.4f}",
                "sig (t)":           _stars(p_t),
                "W-stat":            f"{w_stat:.1f}",
                "p (Wilcoxon)":      f"{p_w:.4f}",
                "sig (W)":           _stars(p_w),
                "Conclusion":        (
                    f"Full system significantly better (p<0.05)"
                    if min(p_t, p_w) < 0.05 and mean_diff > 0
                    else (
                        f"No significant difference"
                        if min(p_t, p_w) >= 0.05
                        else f"Baseline significantly better (p<0.05)"
                    )
                ),
            })

        self.results_ = pd.DataFrame(rows)
        return self.results_

    def print_table(self) -> None:
        """Print the significance table to stdout."""
        if self.results_ is None:
            print("Run .run() first.")
            return
        sep = "-" * 120
        print(f"\n{sep}")
        print("  STATISTICAL SIGNIFICANCE  (paired t-test + Wilcoxon signed-rank vs full system)")
        print("  *** p<0.001   ** p<0.01   * p<0.05   ns not significant")
        print(sep)
        print(self.results_.to_string(index=False))
        print(sep)

    def ci_table(
        self,
        ablation_df: pd.DataFrame,
        metric: str = "f1",
        confidence: float = 0.95,
    ) -> pd.DataFrame:
        """
        Compute bootstrap confidence intervals for each configuration.

        Returns DataFrame with lower/upper bounds (useful for Figure error bars).
        """
        from scipy.stats import t as t_dist
        fold_col = f"_{metric}_folds"
        rows = []
        for _, row in ablation_df.iterrows():
            folds = np.array(row[fold_col])
            n = len(folds)
            mean = folds.mean()
            se = folds.std(ddof=1) / np.sqrt(n)
            alpha = 1 - confidence
            t_crit = t_dist.ppf(1 - alpha / 2, df=n - 1)
            lower = mean - t_crit * se
            upper = mean + t_crit * se
            rows.append({
                "Configuration": row["Configuration"],
                f"{metric.upper()} mean": f"{mean:.4f}",
                f"{int(confidence*100)}% CI lower": f"{lower:.4f}",
                f"{int(confidence*100)}% CI upper": f"{upper:.4f}",
                "CI width": f"{upper - lower:.4f}",
            })
        return pd.DataFrame(rows)
