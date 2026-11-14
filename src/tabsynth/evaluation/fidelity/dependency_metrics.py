"""Multi-variate dependency and correlation preservation metrics for mixed-type tabular data."""

from typing import Dict, List, Any, Optional, Tuple
import math
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def compute_numerical_correlation_matrix(
    df: pd.DataFrame,
    continuous_cols: List[str],
    method: str = "pearson",
) -> np.ndarray:
    """Computes correlation matrix among numerical features."""
    if len(continuous_cols) == 0:
        return np.empty((0, 0), dtype=np.float32)

    sub_df = df[continuous_cols].fillna(0.0)
    corr = sub_df.corr(method=method).fillna(0.0).values
    return corr.astype(np.float32)


def compute_cramers_v(s1: pd.Series, s2: pd.Series) -> float:
    """Computes Cramer's V association statistic between two categorical series in [0, 1]."""
    contingency = pd.crosstab(s1.astype(str), s2.astype(str))
    n = contingency.sum().sum()
    if n == 0 or contingency.shape[0] <= 1 or contingency.shape[1] <= 1:
        return 0.0

    chi2, _, _, _ = chi2_contingency(contingency, correction=False)
    phi2 = chi2 / n
    r, k = contingency.shape
    phi2_corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    r_corr = r - ((r - 1) ** 2) / (n - 1)
    k_corr = k - ((k - 1) ** 2) / (n - 1)
    denom = min(k_corr - 1, r_corr - 1)
    if denom <= 0:
        return 0.0
    return float(np.sqrt(phi2_corr / denom))


def compute_theils_u(s1: pd.Series, s2: pd.Series) -> float:
    """Computes Theil's U (Uncertainty Coefficient) U(s1 | s2) in [0, 1].
    
    Measures how much knowing s2 reduces uncertainty about s1.
    """
    contingency = pd.crosstab(s1.astype(str), s2.astype(str))
    n = contingency.sum().sum()
    if n == 0:
        return 0.0

    p_s1 = contingency.sum(axis=1) / n
    p_s2 = contingency.sum(axis=0) / n

    # Entropy H(s1)
    h_s1 = -sum(p * math.log(p) for p in p_s1 if p > 0)
    if h_s1 == 0:
        return 1.0

    # Conditional Entropy H(s1 | s2)
    h_s1_given_s2 = 0.0
    for col_val in contingency.columns:
        col_prob = p_s2[col_val]
        if col_prob <= 0:
            continue
        col_counts = contingency[col_val]
        col_total = col_counts.sum()
        sub_probs = col_counts / col_total
        h_sub = -sum(p * math.log(p) for p in sub_probs if p > 0)
        h_s1_given_s2 += col_prob * h_sub

    u = (h_s1 - h_s1_given_s2) / h_s1
    return float(np.clip(u, 0.0, 1.0))


def compute_categorical_association_matrix(
    df: pd.DataFrame,
    categorical_cols: List[str],
    method: str = "theils_u",
) -> np.ndarray:
    """Computes pairwise categorical association matrix."""
    n_cat = len(categorical_cols)
    if n_cat == 0:
        return np.empty((0, 0), dtype=np.float32)

    assoc_mat = np.eye(n_cat, dtype=np.float32)

    for i in range(n_cat):
        for j in range(i + 1, n_cat):
            col_i = df[categorical_cols[i]].fillna("<missing>")
            col_j = df[categorical_cols[j]].fillna("<missing>")
            if method == "theils_u":
                # Average asymmetric U(i|j) and U(j|i) for symmetric representation
                val = 0.5 * (compute_theils_u(col_i, col_j) + compute_theils_u(col_j, col_i))
            else:
                val = compute_cramers_v(col_i, col_j)

            assoc_mat[i, j] = val
            assoc_mat[j, i] = val

    return assoc_mat


def compute_dependency_fidelity(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    continuous_cols: List[str],
    categorical_cols: List[str],
) -> Dict[str, Any]:
    """Evaluates multi-variate continuous correlation and discrete association fidelity.
    
    Returns:
        Dictionary of Frobenius norm matrix distances and pairwise difference matrices.
    """
    results: Dict[str, Any] = {}

    # 1. Numerical Pearson Correlation Frobenius Difference
    if len(continuous_cols) > 1:
        r_pearson = compute_numerical_correlation_matrix(real_df, continuous_cols, method="pearson")
        s_pearson = compute_numerical_correlation_matrix(synth_df, continuous_cols, method="pearson")
        frob_denom = np.linalg.norm(r_pearson, ord="fro")
        frob_num_pearson = float(np.linalg.norm(r_pearson - s_pearson, ord="fro") / max(1e-6, frob_denom))

        r_spearman = compute_numerical_correlation_matrix(real_df, continuous_cols, method="spearman")
        s_spearman = compute_numerical_correlation_matrix(synth_df, continuous_cols, method="spearman")
        frob_denom_sp = np.linalg.norm(r_spearman, ord="fro")
        frob_num_spearman = float(np.linalg.norm(r_spearman - s_spearman, ord="fro") / max(1e-6, frob_denom_sp))
    else:
        frob_num_pearson = 0.0
        frob_num_spearman = 0.0

    results["pearson_correlation_frobenius_distance"] = frob_num_pearson
    results["spearman_correlation_frobenius_distance"] = frob_num_spearman

    # 2. Categorical Association Frobenius Difference
    if len(categorical_cols) > 1:
        r_assoc = compute_categorical_association_matrix(real_df, categorical_cols, method="theils_u")
        s_assoc = compute_categorical_association_matrix(synth_df, categorical_cols, method="theils_u")
        frob_denom_cat = np.linalg.norm(r_assoc, ord="fro")
        frob_cat_assoc = float(np.linalg.norm(r_assoc - s_assoc, ord="fro") / max(1e-6, frob_denom_cat))
    else:
        frob_cat_assoc = 0.0

    results["categorical_association_frobenius_distance"] = frob_cat_assoc

    # 3. Overall Combined Dependency Distance (weighted average)
    results["overall_dependency_distance"] = float(0.5 * frob_num_pearson + 0.5 * frob_cat_assoc)

    return results
