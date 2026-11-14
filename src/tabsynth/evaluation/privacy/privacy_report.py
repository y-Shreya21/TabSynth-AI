"""Comprehensive PrivacyAuditor assembling empirical privacy metrics and risk assessments."""

from typing import Dict, List, Any, Optional
import pandas as pd
from .distance_metrics import compute_dcr_and_nndr
from .membership_inference import evaluate_membership_inference
from .attribute_inference import evaluate_attribute_inference
from .singling_out import evaluate_singling_out


FORMAL_PRIVACY_DISCLAIMER = (
    "FORMAL PRIVACY NOTICE: Empirical privacy diagnostics (Distance-to-Closest-Record, "
    "Nearest Neighbor Distance Ratios, Membership Inference ROC-AUC, Attribute Inference Advantage, "
    "and Singling-Out Risk) evaluate empirical memorization and reconstruction vulnerability on specific "
    "finite sample partitions. These empirical scores DO NOT provide provable, worst-case mathematical "
    "differential privacy guarantees (epsilon, delta). Provable guarantees require formal DP mechanisms "
    "such as DP-SGD or PATE during model training."
)


class PrivacyAuditor:
    """Master evaluator for empirical privacy auditing of synthetic tabular datasets."""

    def __init__(
        self,
        continuous_cols: List[str],
        categorical_cols: List[str],
        sensitive_column: Optional[str] = None,
        quasi_identifiers: Optional[List[str]] = None,
        subsample_size: int = 1000,
        random_state: int = 42,
    ):
        self.continuous_cols = continuous_cols
        self.categorical_cols = categorical_cols
        self.sensitive_column = sensitive_column
        self.quasi_identifiers = quasi_identifiers
        self.subsample_size = subsample_size
        self.random_state = random_state

    def audit(
        self,
        df_train: pd.DataFrame,
        df_test: pd.DataFrame,
        df_synth: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Runs the complete privacy auditing battery."""
        # 1. DCR and NNDR Distributions
        dcr_nndr = compute_dcr_and_nndr(
            df_train=df_train,
            df_test=df_test,
            df_synth=df_synth,
            continuous_cols=self.continuous_cols,
            categorical_cols=self.categorical_cols,
            subsample_size=self.subsample_size,
            random_state=self.random_state,
        )

        # 2. Membership Inference Attack (MIA)
        raw_dists = dcr_nndr.pop("raw_distances", None)
        mia = evaluate_membership_inference(
            df_train=df_train,
            df_test=df_test,
            df_synth=df_synth,
            continuous_cols=self.continuous_cols,
            categorical_cols=self.categorical_cols,
            subsample_size=self.subsample_size,
            random_state=self.random_state,
            precomputed_dists=raw_dists,
        )

        # 3. Attribute Inference Attack (AIA)
        aia: Optional[Dict[str, Any]] = None
        if self.sensitive_column and self.sensitive_column in df_train.columns:
            aia = evaluate_attribute_inference(
                df_train=df_train,
                df_test=df_test,
                df_synth=df_synth,
                sensitive_column=self.sensitive_column,
                quasi_identifiers=self.quasi_identifiers,
                subsample_size=self.subsample_size,
                random_state=self.random_state,
            )

        # 4. Singling Out Risk
        singling_out = evaluate_singling_out(
            df_train=df_train,
            df_synth=df_synth,
            quasi_identifiers=self.quasi_identifiers,
            random_state=self.random_state,
        )

        # Aggregate risk level
        risk_votes = [mia["risk_level"], singling_out["risk_level"]]
        if aia is not None:
            risk_votes.append(aia["risk_level"])

        if "HIGH_RISK" in risk_votes:
            overall_risk = "HIGH_RISK"
        elif "MODERATE_RISK" in risk_votes:
            overall_risk = "MODERATE_RISK"
        else:
            overall_risk = "LOW_RISK"

        return {
            "overall_privacy_risk": overall_risk,
            "dcr_and_nndr": dcr_nndr,
            "membership_inference": mia,
            "attribute_inference": aia,
            "singling_out": singling_out,
            "formal_disclaimer": FORMAL_PRIVACY_DISCLAIMER,
        }
