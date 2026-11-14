"""Statistical fidelity evaluation facade maintaining backward compatibility."""

from typing import Dict, List, Any, Optional
import pandas as pd
from tabsynth.evaluation.fidelity.numerical_metrics import compute_numerical_fidelity
from tabsynth.evaluation.fidelity.categorical_metrics import compute_tvd, compute_categorical_fidelity
from tabsynth.evaluation.fidelity.dependency_metrics import (
    compute_numerical_correlation_matrix as compute_correlation_matrix,
    compute_dependency_fidelity,
)
from tabsynth.evaluation.fidelity.c2st import evaluate_c2st
from tabsynth.evaluation.fidelity.fidelity_report import FidelityEvaluator


def evaluate_statistical_fidelity(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    continuous_cols: List[str],
    categorical_cols: List[str],
) -> Dict[str, Any]:
    """Evaluates 1D marginals, 2D correlations, and C2ST indistinguishability.
    
    Backward-compatible entrypoint.
    """
    evaluator = FidelityEvaluator(
        continuous_cols=continuous_cols,
        categorical_cols=categorical_cols,
        run_c2st=True,
        run_sdmetrics=False,
    )
    rep = evaluator.evaluate(real_df, synth_df)

    # Maintain exact dictionary structure expected by Phase 01/02 callers
    summary = rep["summary"]
    return {
        "mean_kolmogorov_smirnov": summary["mean_ks"],
        "mean_wasserstein_distance": summary["mean_wasserstein"],
        "mean_total_variation_distance": summary["mean_tvd"],
        "correlation_frobenius_distance": summary["pearson_frobenius_distance"],
        "c2st_accuracy": summary.get("c2st_accuracy", 0.50),
        "ks_per_feature": rep["numerical"]["ks_statistics"],
        "wasserstein_per_feature": rep["numerical"]["wasserstein_distances"],
        "tvd_per_feature": rep["categorical"]["tv_distances"],
    }
