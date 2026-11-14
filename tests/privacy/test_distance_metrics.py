"""Tests for Phase 05 Gower distance, DCR, and NNDR privacy metrics."""

import pytest
import numpy as np
import pandas as pd

from tabsynth.evaluation.privacy.distance_metrics import (
    compute_gower_distance_matrix,
    prepare_arrays_for_distance,
    compute_dcr_and_nndr,
)


def test_gower_distance_matrix_properties():
    num_a = np.array([[1.0], [2.0]], dtype=np.float32)
    cat_a = np.array([["A"], ["B"]], dtype=object)

    num_b = np.array([[1.0], [3.0]], dtype=np.float32)
    cat_b = np.array([["A"], ["C"]], dtype=object)

    ranges = np.array([2.0], dtype=np.float32)

    dist = compute_gower_distance_matrix(num_a, cat_a, num_b, cat_b, ranges)

    assert dist.shape == (2, 2)
    # Identical record (1.0, "A") vs (1.0, "A") must be 0.0
    assert dist[0, 0] == 0.0
    # Values bounded in [0, 1]
    assert np.all((dist >= 0.0) & (dist <= 1.0))


def test_compute_dcr_and_nndr():
    df_train = pd.DataFrame({
        "age": [20.0, 30.0, 40.0],
        "workclass": ["Private", "Self-emp", "Gov"],
    })
    df_test = pd.DataFrame({
        "age": [25.0, 35.0, 45.0],
        "workclass": ["Private", "Gov", "Private"],
    })
    # Synth includes one exact match to train[0] and novel records
    df_synth = pd.DataFrame({
        "age": [20.0, 28.0],
        "workclass": ["Private", "Gov"],
    })

    res = compute_dcr_and_nndr(
        df_train=df_train,
        df_test=df_test,
        df_synth=df_synth,
        continuous_cols=["age"],
        categorical_cols=["workclass"],
        subsample_size=10,
    )

    assert "mean_dcr_train" in res
    assert "mean_dcr_test" in res
    assert "dcr_ratio" in res
    assert "exact_record_matches" in res
    assert res["exact_record_matches"] == 1
    assert "mean_nndr" in res
    assert 0.0 <= res["mean_nndr"] <= 1.0
