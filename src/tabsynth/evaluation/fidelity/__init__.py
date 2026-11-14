"""Fidelity Evaluation Package for TabSynth AI."""

from tabsynth.evaluation.fidelity.numerical_metrics import compute_numerical_fidelity
from tabsynth.evaluation.fidelity.categorical_metrics import compute_tvd, compute_categorical_fidelity
from tabsynth.evaluation.fidelity.dependency_metrics import (
    compute_cramers_v,
    compute_theils_u,
    compute_numerical_correlation_matrix,
    compute_categorical_association_matrix,
    compute_dependency_fidelity,
)
from tabsynth.evaluation.fidelity.c2st import evaluate_c2st
from tabsynth.evaluation.fidelity.sdmetrics_evaluator import evaluate_sdmetrics
from tabsynth.evaluation.fidelity.fidelity_report import FidelityEvaluator, evaluate_statistical_fidelity

__all__ = [
    "compute_numerical_fidelity",
    "compute_tvd",
    "compute_categorical_fidelity",
    "compute_cramers_v",
    "compute_theils_u",
    "compute_numerical_correlation_matrix",
    "compute_categorical_association_matrix",
    "compute_dependency_fidelity",
    "evaluate_c2st",
    "evaluate_sdmetrics",
    "FidelityEvaluator",
    "evaluate_statistical_fidelity",
]
