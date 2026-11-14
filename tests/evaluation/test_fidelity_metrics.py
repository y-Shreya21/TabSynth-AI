"""Tests for Phase 05 statistical fidelity metrics."""

import pytest
import numpy as np
import pandas as pd

from tabsynth.evaluation.fidelity.numerical_metrics import compute_numerical_fidelity
from tabsynth.evaluation.fidelity.categorical_metrics import compute_tvd, compute_categorical_fidelity
from tabsynth.evaluation.fidelity.dependency_metrics import compute_dependency_fidelity
from tabsynth.evaluation.fidelity.fidelity_report import FidelityEvaluator


@pytest.fixture
def sample_real_and_synth_dfs():
    np.random.seed(42)
    n = 100
    real = pd.DataFrame({
        "num1": np.random.normal(10, 2, n),
        "num2": np.random.exponential(5, n),
        "cat1": np.random.choice(["A", "B", "C"], n),
        "cat2": np.random.choice(["X", "Y"], n),
    })
    # Synth: slight perturbation
    synth = pd.DataFrame({
        "num1": np.random.normal(10.2, 2.1, n),
        "num2": np.random.exponential(5.1, n),
        "cat1": np.random.choice(["A", "B", "C"], n),
        "cat2": np.random.choice(["X", "Y"], n),
    })
    return real, synth


def test_numerical_fidelity_metrics(sample_real_and_synth_dfs):
    real, synth = sample_real_and_synth_dfs
    metrics = compute_numerical_fidelity(real, synth, ["num1", "num2"])

    assert "mean_ks" in metrics
    assert "mean_wasserstein" in metrics
    assert 0.0 <= metrics["mean_ks"] <= 1.0
    assert metrics["mean_wasserstein"] >= 0.0
    assert "num1" in metrics["ks_statistics"]
    assert "num2" in metrics["ks_statistics"]
    assert len(metrics["quantile_errors"]["num1"]) == 5


def test_categorical_fidelity_metrics(sample_real_and_synth_dfs):
    real, synth = sample_real_and_synth_dfs
    metrics = compute_categorical_fidelity(real, synth, ["cat1", "cat2"])

    assert "mean_tvd" in metrics
    assert 0.0 <= metrics["mean_tvd"] <= 1.0
    assert metrics["total_unseen_categories"] == 0
    assert "cat1" in metrics["tv_distances"]
    assert "cat2" in metrics["tv_distances"]


def test_dependency_fidelity_metrics(sample_real_and_synth_dfs):
    real, synth = sample_real_and_synth_dfs
    metrics = compute_dependency_fidelity(real, synth, ["num1", "num2"], ["cat1", "cat2"])

    assert "pearson_correlation_frobenius_distance" in metrics
    assert "spearman_correlation_frobenius_distance" in metrics
    assert metrics["pearson_correlation_frobenius_distance"] >= 0.0
    assert "categorical_association_frobenius_distance" in metrics


def test_fidelity_evaluator_aggregate(sample_real_and_synth_dfs):
    real, synth = sample_real_and_synth_dfs
    evaluator = FidelityEvaluator(
        continuous_cols=["num1", "num2"],
        categorical_cols=["cat1", "cat2"],
        run_c2st=True,
        run_sdmetrics=False,
    )
    rep = evaluator.evaluate(real, synth, c2st_subsample=100)

    assert "summary" in rep
    assert "mean_ks" in rep["summary"]
    assert "mean_tvd" in rep["summary"]
    assert "c2st_roc_auc" in rep["summary"]
