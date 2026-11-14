"""Membership Inference Attack (MIA) evaluation on synthetic tabular data."""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from .distance_metrics import compute_gower_distance_matrix, prepare_arrays_for_distance


def evaluate_membership_inference(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    df_synth: pd.DataFrame,
    continuous_cols: List[str],
    categorical_cols: List[str],
    subsample_size: int = 1000,
    random_state: int = 42,
    precomputed_dists: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Any]:
    """Evaluates distance-based Membership Inference Attack (MIA) resistance.
    
    Adversary attempts to infer whether an arbitrary record was a member of the
    training set by measuring its distance to the closest synthetic record.
    """
    if precomputed_dists is not None and "dist_synth_train" in precomputed_dists:
        dist_s_tr = precomputed_dists["dist_synth_train"]
        dist_s_te = precomputed_dists["dist_synth_test"]
    else:
        n_s = min(len(df_synth), subsample_size)
        n_tr = min(len(df_train), subsample_size)
        n_te = min(len(df_test), subsample_size)

        s_sample = df_synth.sample(n=n_s, random_state=random_state)
        tr_sample = df_train.sample(n=n_tr, random_state=random_state)
        te_sample = df_test.sample(n=n_te, random_state=random_state)

        num_tr, cat_tr, tr_ranges = prepare_arrays_for_distance(tr_sample, continuous_cols, categorical_cols)
        num_s, cat_s, _ = prepare_arrays_for_distance(s_sample, continuous_cols, categorical_cols, reference_ranges=tr_ranges)
        num_te, cat_te, _ = prepare_arrays_for_distance(te_sample, continuous_cols, categorical_cols, reference_ranges=tr_ranges)

        dist_s_tr = compute_gower_distance_matrix(num_s, cat_s, num_tr, cat_tr, tr_ranges)
        dist_s_te = compute_gower_distance_matrix(num_s, cat_s, num_te, cat_te, tr_ranges)

    # Closest distance from each real record (train / test) to the synthetic set
    # dist_s_tr is (N_synth, N_train), so transpose to get (N_train, N_synth)
    d_tr_to_synth = np.min(dist_s_tr.T, axis=1)
    d_te_to_synth = np.min(dist_s_te.T, axis=1)

    n_tr = len(d_tr_to_synth)
    n_te = len(d_te_to_synth)

    # Balance evaluation partition if sizes differ
    min_size = min(n_tr, n_te)
    d_tr = d_tr_to_synth[:min_size]
    d_te = d_te_to_synth[:min_size]

    # Attacker score: lower distance => higher probability of being a member
    scores = np.concatenate([-d_tr, -d_te])
    labels = np.concatenate([np.ones(min_size), np.zeros(min_size)])

    try:
        auroc = float(roc_auc_score(labels, scores))
    except Exception:
        auroc = 0.50

    # Threshold-based predictions at the median score
    threshold = float(np.median(scores))
    preds = (scores >= threshold).astype(int)

    acc = float(accuracy_score(labels, preds))
    prec = float(precision_score(labels, preds, zero_division=0))
    rec = float(recall_score(labels, preds, zero_division=0))

    # Risk categorization
    if auroc < 0.55:
        risk = "LOW_RISK"
    elif auroc < 0.65:
        risk = "MODERATE_RISK"
    else:
        risk = "HIGH_RISK"

    return {
        "mia_roc_auc": auroc,
        "mia_accuracy": acc,
        "mia_precision": prec,
        "mia_recall": rec,
        "risk_level": risk,
        "num_members_audited": min_size,
        "num_non_members_audited": min_size,
        "mean_member_distance_to_synth": float(np.mean(d_tr)),
        "mean_non_member_distance_to_synth": float(np.mean(d_te)),
    }
