"""Distance-to-Closest-Record (DCR), NNDR, and Gower distance metrics."""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd


def compute_gower_distance_matrix(
    num_A: np.ndarray,
    cat_A: np.ndarray,
    num_B: np.ndarray,
    cat_B: np.ndarray,
    num_ranges: np.ndarray,
) -> np.ndarray:
    """Computes pairwise normalized Gower distance between two sets of mixed-type records.
    
    Args:
        num_A: (N_A, N_cont) float32 array.
        cat_A: (N_A, N_cat) string/object array.
        num_B: (N_B, N_cont) float32 array.
        cat_B: (N_B, N_cat) string/object array.
        num_ranges: Range (max - min) of continuous features for normalization.
        
    Returns:
        (N_A, N_B) distance matrix with values in [0, 1].
    """
    n_a = num_A.shape[0] if num_A.shape[1] > 0 else cat_A.shape[0]
    n_b = num_B.shape[0] if num_B.shape[1] > 0 else cat_B.shape[0]
    num_cont = num_A.shape[1]
    num_cat = cat_A.shape[1]
    total_features = max(1, num_cont + num_cat)

    diff_matrix = np.zeros((n_a, n_b), dtype=np.float32)

    # 1. Continuous Features: normalized absolute difference
    if num_cont > 0:
        for idx in range(num_cont):
            col_a = num_A[:, idx : idx + 1]  # (N_A, 1) float32
            col_b = num_B[:, idx : idx + 1].T  # (1, N_B) float32
            rng = max(1e-6, float(num_ranges[idx]))
            diff_matrix += (np.abs(col_a - col_b) / rng).astype(np.float32)

    # 2. Categorical Features: mismatch indicator (0 if match, 1 if mismatch)
    if num_cat > 0:
        for idx in range(num_cat):
            col_a = cat_A[:, idx : idx + 1]
            col_b = cat_B[:, idx : idx + 1].T
            diff_matrix += (col_a != col_b).astype(np.float32)

    return diff_matrix / float(total_features)


def prepare_arrays_for_distance(
    df: pd.DataFrame,
    continuous_cols: List[str],
    categorical_cols: List[str],
    reference_ranges: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extracts continuous and categorical arrays and computes or applies feature ranges."""
    num_part = (
        df[continuous_cols].fillna(0.0).values.astype(np.float32)
        if len(continuous_cols) > 0
        else np.empty((len(df), 0), dtype=np.float32)
    )
    cat_part = (
        df[categorical_cols].astype(str).values
        if len(categorical_cols) > 0
        else np.empty((len(df), 0), dtype=object)
    )

    if reference_ranges is not None:
        ranges = reference_ranges
    else:
        ranges = np.zeros(len(continuous_cols), dtype=np.float32)
        for idx in range(len(continuous_cols)):
            col_vals = num_part[:, idx]
            ranges[idx] = float(np.max(col_vals) - np.min(col_vals)) if len(col_vals) > 0 else 1.0

    return num_part, cat_part, ranges


def compute_dcr_and_nndr(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    df_synth: pd.DataFrame,
    continuous_cols: List[str],
    categorical_cols: List[str],
    subsample_size: int = 1000,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Computes comprehensive DCR distributions, NNDR, and exact match audits."""
    n_s = min(len(df_synth), subsample_size)
    n_tr = min(len(df_train), subsample_size)
    n_te = min(len(df_test), subsample_size)

    s_sample = df_synth.sample(n=n_s, random_state=random_state)
    tr_sample = df_train.sample(n=n_tr, random_state=random_state)
    te_sample = df_test.sample(n=n_te, random_state=random_state)

    # Compute normalization ranges strictly on the real training partition to avoid leakage
    num_tr, cat_tr, tr_ranges = prepare_arrays_for_distance(tr_sample, continuous_cols, categorical_cols)
    num_s, cat_s, _ = prepare_arrays_for_distance(s_sample, continuous_cols, categorical_cols, reference_ranges=tr_ranges)
    num_te, cat_te, _ = prepare_arrays_for_distance(te_sample, continuous_cols, categorical_cols, reference_ranges=tr_ranges)

    # 1. Pairwise Distances: Synth -> Train, Synth -> Test, Test -> Train
    dist_synth_train = compute_gower_distance_matrix(num_s, cat_s, num_tr, cat_tr, tr_ranges)
    dist_synth_test = compute_gower_distance_matrix(num_s, cat_s, num_te, cat_te, tr_ranges)
    dist_test_train = compute_gower_distance_matrix(num_te, cat_te, num_tr, cat_tr, tr_ranges)

    # 2. DCR Computations
    dcr_synth_train = np.min(dist_synth_train, axis=1)
    dcr_synth_test = np.min(dist_synth_test, axis=1)
    dcr_test_train = np.min(dist_test_train, axis=1)

    exact_matches = int((dcr_synth_train < 1e-5).sum())
    exact_match_rate = float(exact_matches / n_s)

    # 3. NNDR (Nearest Neighbor Distance Ratio): d_1 / d_2
    sorted_dists = np.sort(dist_synth_train, axis=1)
    d_1 = sorted_dists[:, 0]
    d_2 = sorted_dists[:, 1] if sorted_dists.shape[1] > 1 else d_1
    nndr = d_1 / np.clip(d_2, a_min=1e-6, a_max=None)

    # 4. Percentiles and Statistics
    def _get_stats(arr: np.ndarray) -> Dict[str, float]:
        return {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "p5": float(np.percentile(arr, 5)),
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "p95": float(np.percentile(arr, 95)),
        }

    stats_dcr_train = _get_stats(dcr_synth_train)
    stats_dcr_test = _get_stats(dcr_synth_test)
    stats_dcr_test_train = _get_stats(dcr_test_train)

    mean_dcr_tr = stats_dcr_train["mean"]
    mean_dcr_te = stats_dcr_test["mean"]
    dcr_ratio = float(mean_dcr_tr / (mean_dcr_te if mean_dcr_te > 1e-6 else 1.0))
    p5_ratio = float(stats_dcr_train["p5"] / (stats_dcr_test_train["p5"] if stats_dcr_test_train["p5"] > 1e-6 else 1.0))

    return {
        "n_synthetic_audited": n_s,
        "n_train_audited": n_tr,
        "n_test_audited": n_te,
        "dcr_synth_to_train": stats_dcr_train,
        "dcr_synth_to_test": stats_dcr_test,
        "dcr_test_to_train_baseline": stats_dcr_test_train,
        "mean_dcr_train": mean_dcr_tr,
        "mean_dcr_test": mean_dcr_te,
        "dcr_ratio": dcr_ratio,
        "p5_dcr_ratio_vs_test": p5_ratio,
        "dcr_5th_percentile_train": stats_dcr_train["p5"],
        "exact_record_matches": exact_matches,
        "exact_match_rate": exact_match_rate,
        "mean_nndr": float(np.mean(nndr)),
        "median_nndr": float(np.median(nndr)),
        "nndr_below_0_5_fraction": float((nndr < 0.5).mean()),
        "nndr_below_0_2_fraction": float((nndr < 0.2).mean()),
        "raw_distances": {
            "dist_synth_train": dist_synth_train,
            "dist_synth_test": dist_synth_test,
        },
    }
