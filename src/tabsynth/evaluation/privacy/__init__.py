"""Empirical privacy evaluation package for TabSynth AI."""

from typing import Dict, List, Any, Optional
import pandas as pd
from .distance_metrics import (
    compute_gower_distance_matrix,
    prepare_arrays_for_distance,
    compute_dcr_and_nndr,
)
from .membership_inference import evaluate_membership_inference
from .attribute_inference import evaluate_attribute_inference
from .singling_out import evaluate_singling_out
from .privacy_report import PrivacyAuditor, FORMAL_PRIVACY_DISCLAIMER


def evaluate_empirical_privacy(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    df_synth: pd.DataFrame,
    continuous_cols: List[str],
    categorical_cols: List[str],
    subsample_size: int = 1000,
) -> Dict[str, Any]:
    """Backward-compatible entrypoint for empirical privacy evaluation."""
    auditor = PrivacyAuditor(
        continuous_cols=continuous_cols,
        categorical_cols=categorical_cols,
        subsample_size=subsample_size,
    )
    res = auditor.audit(df_train=df_train, df_test=df_test, df_synth=df_synth)
    dcr_m = res["dcr_and_nndr"]
    mia_m = res["membership_inference"]

    return {
        "mean_dcr_train": dcr_m["mean_dcr_train"],
        "mean_dcr_test": dcr_m["mean_dcr_test"],
        "dcr_ratio": dcr_m["dcr_ratio"],
        "dcr_5th_percentile_train": dcr_m["dcr_5th_percentile_train"],
        "exact_record_matches": dcr_m["exact_record_matches"],
        "exact_match_rate": dcr_m["exact_match_rate"],
        "mean_nndr": dcr_m["mean_nndr"],
        "nndr_below_0_5_fraction": dcr_m["nndr_below_0_5_fraction"],
        "membership_inference_auroc": mia_m["mia_roc_auc"],
        "privacy_risk_assessment": res["overall_privacy_risk"],
        "disclaimer": res["formal_disclaimer"],
    }


__all__ = [
    "compute_gower_distance_matrix",
    "prepare_arrays_for_distance",
    "compute_dcr_and_nndr",
    "evaluate_membership_inference",
    "evaluate_attribute_inference",
    "evaluate_singling_out",
    "PrivacyAuditor",
    "FORMAL_PRIVACY_DISCLAIMER",
    "evaluate_empirical_privacy",
]
