"""Tests for Classifier Two-Sample Test (C2ST)."""

import pytest
import numpy as np
import pandas as pd

from tabsynth.evaluation.fidelity.c2st import evaluate_c2st


def test_c2st_indistinguishable_data():
    """Identical distributions should yield C2ST ROC-AUC near 0.50."""
    np.random.seed(42)
    n = 100
    real = pd.DataFrame({
        "num": np.random.randn(n),
        "cat": np.random.choice(["A", "B"], size=n),
    })
    synth = pd.DataFrame({
        "num": np.random.randn(n),
        "cat": np.random.choice(["A", "B"], size=n),
    })

    res = evaluate_c2st(
        real_df=real,
        synth_df=synth,
        continuous_cols=["num"],
        categorical_cols=["cat"],
        classifier_type="lr",
        subsample_size=100,
    )

    assert "c2st_accuracy" in res
    assert "c2st_roc_auc" in res
    # Indistinguishable data should be within [0.35, 0.65]
    assert 0.35 <= res["c2st_roc_auc"] <= 0.65


def test_c2st_separable_data():
    """Distinct distributions should yield high C2ST ROC-AUC (> 0.85)."""
    np.random.seed(42)
    n = 100
    real = pd.DataFrame({
        "num": np.random.normal(0, 1, size=n),
        "cat": ["A"] * n,
    })
    synth = pd.DataFrame({
        "num": np.random.normal(10, 1, size=n),
        "cat": ["B"] * n,
    })

    res = evaluate_c2st(
        real_df=real,
        synth_df=synth,
        continuous_cols=["num"],
        categorical_cols=["cat"],
        classifier_type="rf",
        subsample_size=100,
    )

    assert res["c2st_roc_auc"] >= 0.85
