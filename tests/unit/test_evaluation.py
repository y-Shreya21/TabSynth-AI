"""Unit tests for statistical fidelity, downstream utility, and privacy audit modules."""

import pytest
import numpy as np
import pandas as pd
from tabsynth.evaluation.fidelity import evaluate_statistical_fidelity
from tabsynth.evaluation.utility import evaluate_downstream_utility
from tabsynth.evaluation.privacy import evaluate_empirical_privacy


@pytest.fixture
def mock_eval_datasets():
    """Generates synthetic real train, real test, and synthetic data for evaluation testing."""
    np.random.seed(42)
    n = 100

    def make_df():
        return pd.DataFrame({
            "age": np.random.normal(40, 10, size=n),
            "income": np.random.exponential(50000, size=n),
            "education": np.random.choice(["Bachelors", "Masters", "PhD"], size=n),
            "target": np.random.choice(["0", "1"], p=[0.8, 0.2], size=n),
        })

    return {
        "train": make_df(),
        "test": make_df(),
        "synth": make_df(),
        "continuous_cols": ["age", "income"],
        "categorical_cols": ["education"],
        "target_col": "target",
    }


def test_fidelity_metrics(mock_eval_datasets):
    """Verify statistical fidelity evaluation calculations."""
    data = mock_eval_datasets
    report = evaluate_statistical_fidelity(
        real_df=data["test"],
        synth_df=data["synth"],
        continuous_cols=data["continuous_cols"],
        categorical_cols=data["categorical_cols"],
    )

    assert "mean_kolmogorov_smirnov" in report
    assert "mean_wasserstein_distance" in report
    assert "mean_total_variation_distance" in report
    assert "correlation_frobenius_distance" in report
    assert "c2st_accuracy" in report

    assert 0.0 <= report["mean_kolmogorov_smirnov"] <= 1.0
    assert 0.0 <= report["mean_total_variation_distance"] <= 1.0
    assert 0.0 <= report["c2st_accuracy"] <= 1.0


def test_utility_evaluation(mock_eval_datasets):
    """Verify TSTR vs TRTR downstream utility benchmarking."""
    data = mock_eval_datasets
    report = evaluate_downstream_utility(
        df_train_real=data["train"],
        df_train_synth=data["synth"],
        df_test_real=data["test"],
        target_column=data["target_col"],
        classifiers=["logistic_regression", "random_forest"],
    )

    assert "logistic_regression" in report
    assert "random_forest" in report

    rf = report["random_forest"]
    assert "trtr" in rf and "tstr" in rf
    assert "macro_f1" in rf["trtr"]
    assert "minority_recall" in rf["tstr"]
    assert "relative_utility_parity" in rf


def test_privacy_evaluation(mock_eval_datasets):
    """Verify DCR, NNDR, and Membership Inference Attack auditing."""
    data = mock_eval_datasets
    report = evaluate_empirical_privacy(
        df_train=data["train"],
        df_test=data["test"],
        df_synth=data["synth"],
        continuous_cols=data["continuous_cols"],
        categorical_cols=data["categorical_cols"],
        subsample_size=50,
    )

    assert "mean_dcr_train" in report
    assert "mean_dcr_test" in report
    assert "dcr_ratio" in report
    assert "mean_nndr" in report
    assert "membership_inference_auroc" in report
    assert "privacy_risk_assessment" in report
    assert 0.0 <= report["membership_inference_auroc"] <= 1.0
