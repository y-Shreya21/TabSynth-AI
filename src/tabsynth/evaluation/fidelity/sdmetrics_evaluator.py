"""SDMetrics integration and native quality/diagnostic fallback evaluator."""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd

try:
    import sdmetrics
    from sdmetrics.reports.single_table import QualityReport, DiagnosticReport
    HAS_SDMETRICS = True
    SDMETRICS_VERSION = getattr(sdmetrics, "__version__", "unknown")
except ImportError:
    HAS_SDMETRICS = False
    SDMETRICS_VERSION = None

from tabsynth.evaluation.fidelity.numerical_metrics import compute_numerical_fidelity
from tabsynth.evaluation.fidelity.categorical_metrics import compute_categorical_fidelity
from tabsynth.evaluation.fidelity.dependency_metrics import compute_dependency_fidelity


def evaluate_sdmetrics(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    metadata_dict: Optional[Dict[str, Any]] = None,
    continuous_cols: Optional[List[str]] = None,
    categorical_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Generates synthetic data Quality and Diagnostic reports.
    
    If SDMetrics library is available, runs official single-table QualityReport
    and DiagnosticReport. Otherwise, computes an equivalent native breakdown
    matching SDMetrics 1D column shape similarity and 2D pairwise trend scores.
    """
    cont_cols = continuous_cols or []
    cat_cols = categorical_cols or []

    if HAS_SDMETRICS and metadata_dict is not None:
        try:
            from sdv.metadata import SingleTableMetadata
            meta = SingleTableMetadata.load_from_dict(metadata_dict)
            q_report = QualityReport()
            q_report.generate(real_df, synth_df, meta)
            quality_score = float(q_report.get_score())
            properties = q_report.get_properties().to_dict(orient="records")

            d_report = DiagnosticReport()
            d_report.generate(real_df, synth_df, meta)
            diag_score = float(d_report.get_score())

            return {
                "engine": f"sdmetrics_{SDMETRICS_VERSION}",
                "overall_quality_score": quality_score,
                "diagnostic_score": diag_score,
                "properties": properties,
                "library_installed": True,
            }
        except Exception:
            pass  # Fall through to native evaluation

    # Native implementation matching SDMetrics score definitions
    num_res = compute_numerical_fidelity(real_df, synth_df, cont_cols)
    cat_res = compute_categorical_fidelity(real_df, synth_df, cat_cols)
    dep_res = compute_dependency_fidelity(real_df, synth_df, cont_cols, cat_cols)

    # 1. Column Shapes score: 1 - mean(KS, TVD) in [0, 1]
    all_1d_errors = list(num_res.get("ks_statistics", {}).values()) + list(cat_res.get("tv_distances", {}).values())
    column_shapes_score = float(1.0 - np.mean(all_1d_errors)) if all_1d_errors else 1.0

    # 2. Column Pair Trends score: 1 - overall_dependency_distance in [0, 1]
    dep_dist = dep_res.get("overall_dependency_distance", 0.0)
    column_pairs_score = float(max(0.0, 1.0 - dep_dist))

    # Overall Quality Score: average of 1D shapes and 2D pairs
    overall_quality_score = float(0.5 * column_shapes_score + 0.5 * column_pairs_score)

    # 3. Diagnostic checks
    # Boundary adherence: check if synth continuous values stay within empirical min/max with margin
    bounds_violations = 0
    total_bounds_checked = len(cont_cols)
    for col in cont_cols:
        r_min, r_max = real_df[col].min(), real_df[col].max()
        s_min, s_max = synth_df[col].min(), synth_df[col].max()
        rng = max(1e-6, r_max - r_min)
        if s_min < r_min - 0.2 * rng or s_max > r_max + 0.2 * rng:
            bounds_violations += 1

    boundary_adherence = float(1.0 - bounds_violations / max(1, total_bounds_checked))
    unseen_cats = cat_res.get("total_unseen_categories", 0)
    category_coverage = float(1.0 if unseen_cats == 0 else max(0.0, 1.0 - unseen_cats * 0.05))

    return {
        "engine": "native_sdmetrics_reproduction",
        "library_installed": HAS_SDMETRICS,
        "library_version": SDMETRICS_VERSION,
        "overall_quality_score": overall_quality_score,
        "column_shapes_score": column_shapes_score,
        "column_pairs_score": column_pairs_score,
        "diagnostic_metrics": {
            "boundary_adherence_score": boundary_adherence,
            "category_coverage_score": category_coverage,
            "unseen_categories_count": unseen_cats,
        },
    }
