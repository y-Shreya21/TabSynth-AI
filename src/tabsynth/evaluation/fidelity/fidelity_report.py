"""Aggregated statistical fidelity reporting suite for tabular synthesis."""

from typing import Dict, List, Any, Optional
import pandas as pd
from tabsynth.evaluation.fidelity.numerical_metrics import compute_numerical_fidelity
from tabsynth.evaluation.fidelity.categorical_metrics import compute_categorical_fidelity
from tabsynth.evaluation.fidelity.dependency_metrics import compute_dependency_fidelity
from tabsynth.evaluation.fidelity.c2st import evaluate_c2st
from tabsynth.evaluation.fidelity.sdmetrics_evaluator import evaluate_sdmetrics


class FidelityEvaluator:
    """Evaluates marginal, dependency, C2ST, and diagnostic fidelity metrics."""

    def __init__(
        self,
        continuous_cols: List[str],
        categorical_cols: List[str],
        run_c2st: bool = True,
        run_sdmetrics: bool = True,
    ):
        self.continuous_cols = continuous_cols
        self.categorical_cols = categorical_cols
        self.run_c2st = run_c2st
        self.run_sdmetrics = run_sdmetrics

    def evaluate(
        self,
        real_df: pd.DataFrame,
        synth_df: pd.DataFrame,
        c2st_classifier: str = "rf",
        c2st_subsample: int = 1000,
    ) -> Dict[str, Any]:
        """Runs full suite of fidelity metrics comparing real and synthetic data."""
        # 1. Numerical 1D Marginals
        num_metrics = compute_numerical_fidelity(real_df, synth_df, self.continuous_cols)

        # 2. Categorical 1D Marginals & Unseen categories
        cat_metrics = compute_categorical_fidelity(real_df, synth_df, self.categorical_cols)

        # 3. 2D Dependencies and Correlations
        dep_metrics = compute_dependency_fidelity(real_df, synth_df, self.continuous_cols, self.categorical_cols)

        # 4. Classifier Two-Sample Test (C2ST)
        if self.run_c2st:
            c2st_metrics = evaluate_c2st(
                real_df=real_df,
                synth_df=synth_df,
                continuous_cols=self.continuous_cols,
                categorical_cols=self.categorical_cols,
                classifier_type=c2st_classifier,
                subsample_size=c2st_subsample,
            )
        else:
            c2st_metrics = {}

        # 5. SDMetrics Quality & Diagnostic Breakdown
        if self.run_sdmetrics:
            sdm_metrics = evaluate_sdmetrics(
                real_df=real_df,
                synth_df=synth_df,
                continuous_cols=self.continuous_cols,
                categorical_cols=self.categorical_cols,
            )
        else:
            sdm_metrics = {}

        report = {
            "summary": {
                "mean_ks": num_metrics["mean_ks"],
                "mean_wasserstein": num_metrics["mean_wasserstein"],
                "mean_tvd": cat_metrics["mean_tvd"],
                "pearson_frobenius_distance": dep_metrics["pearson_correlation_frobenius_distance"],
                "spearman_frobenius_distance": dep_metrics["spearman_correlation_frobenius_distance"],
                "categorical_association_distance": dep_metrics["categorical_association_frobenius_distance"],
                "c2st_accuracy": c2st_metrics.get("c2st_accuracy"),
                "c2st_roc_auc": c2st_metrics.get("c2st_roc_auc"),
                "sdmetrics_quality_score": sdm_metrics.get("overall_quality_score"),
                "total_unseen_categories": cat_metrics["total_unseen_categories"],
            },
            "numerical": num_metrics,
            "categorical": cat_metrics,
            "dependency": dep_metrics,
            "c2st": c2st_metrics,
            "sdmetrics": sdm_metrics,
        }
        return report


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

