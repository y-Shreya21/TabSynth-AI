"""Numerical marginal distribution fidelity metrics for tabular data."""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance


def compute_numerical_fidelity(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    continuous_cols: List[str],
) -> Dict[str, Any]:
    """Computes comprehensive numerical distribution fidelity metrics.
    
    Evaluates:
        - Kolmogorov-Smirnov test statistic and p-value.
        - Wasserstein distance (normalized by standard deviation of real feature).
        - Absolute and relative mean differences.
        - Absolute standard deviation differences.
        - Multi-quantile differences (10th, 25th, 50th, 75th, 90th).
        
    Args:
        real_df: Reference real DataFrame.
        synth_df: Evaluated synthetic DataFrame.
        continuous_cols: List of continuous column names.
        
    Returns:
        Dictionary containing aggregate summary metrics and per-column breakdowns.
    """
    ks_statistics: Dict[str, float] = {}
    ks_pvalues: Dict[str, float] = {}
    wasserstein_dists: Dict[str, float] = {}
    mean_differences: Dict[str, float] = {}
    std_differences: Dict[str, float] = {}
    quantile_errors: Dict[str, Dict[str, float]] = {}

    quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]

    for col in continuous_cols:
        if col not in real_df.columns or col not in synth_df.columns:
            continue

        r_vals = real_df[col].dropna().values.astype(float)
        s_vals = synth_df[col].dropna().values.astype(float)

        if len(r_vals) == 0 or len(s_vals) == 0:
            continue

        # 1. Kolmogorov-Smirnov Test
        ks_res = ks_2samp(r_vals, s_vals)
        ks_statistics[col] = float(ks_res.statistic)
        ks_pvalues[col] = float(ks_res.pvalue)

        # 2. Scale-Normalized Wasserstein Distance
        r_std = float(np.std(r_vals))
        scale = r_std if r_std > 1e-6 else 1.0
        w_dist = float(wasserstein_distance(r_vals, s_vals) / scale)
        wasserstein_dists[col] = w_dist

        # 3. Mean & Standard Deviation Differences
        mean_diff = float(abs(np.mean(r_vals) - np.mean(s_vals)))
        std_diff = float(abs(r_std - np.std(s_vals)))
        mean_differences[col] = mean_diff
        std_differences[col] = std_diff

        # 4. Quantile Deviations
        r_quants = np.quantile(r_vals, quantiles)
        s_quants = np.quantile(s_vals, quantiles)
        q_diffs = {f"q{int(q*100)}": float(abs(rq - sq)) for q, rq, sq in zip(quantiles, r_quants, s_quants)}
        quantile_errors[col] = q_diffs

    mean_ks = float(np.mean(list(ks_statistics.values()))) if ks_statistics else 0.0
    mean_wass = float(np.mean(list(wasserstein_dists.values()))) if wasserstein_dists else 0.0
    mean_mean_diff = float(np.mean(list(mean_differences.values()))) if mean_differences else 0.0
    mean_std_diff = float(np.mean(list(std_differences.values()))) if std_differences else 0.0

    return {
        "mean_ks": mean_ks,
        "mean_wasserstein": mean_wass,
        "mean_mean_difference": mean_mean_diff,
        "mean_std_difference": mean_std_diff,
        "ks_statistics": ks_statistics,
        "ks_pvalues": ks_pvalues,
        "wasserstein_distances": wasserstein_dists,
        "mean_differences": mean_differences,
        "std_differences": std_differences,
        "quantile_errors": quantile_errors,
    }
