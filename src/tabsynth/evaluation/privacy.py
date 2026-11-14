"""Empirical privacy risk evaluation facade maintaining backward compatibility."""

from typing import Dict, List, Any, Optional
import pandas as pd
from tabsynth.evaluation.privacy.distance_metrics import (
    compute_gower_distance_matrix,
    prepare_arrays_for_distance,
    compute_dcr_and_nndr,
)
from tabsynth.evaluation.privacy.membership_inference import evaluate_membership_inference
from tabsynth.evaluation.privacy.attribute_inference import evaluate_attribute_inference
from tabsynth.evaluation.privacy.singling_out import evaluate_singling_out
from tabsynth.evaluation.privacy.privacy_report import PrivacyAuditor, FORMAL_PRIVACY_DISCLAIMER
from tabsynth.evaluation.privacy import evaluate_empirical_privacy

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
