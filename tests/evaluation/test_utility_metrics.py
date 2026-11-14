"""Tests for Phase 05 downstream utility evaluation."""

import pytest
import numpy as np
import pandas as pd

from tabsynth.evaluation.utility.classification_metrics import compute_classification_metrics
from tabsynth.evaluation.utility.downstream import DownstreamDataPreprocessor, get_downstream_classifiers
from tabsynth.evaluation.utility.trtr import evaluate_trtr
from tabsynth.evaluation.utility.tstr import evaluate_tstr
from tabsynth.evaluation.utility.augmentation import evaluate_minority_augmentation
from tabsynth.evaluation.utility.utility_report import UtilityEvaluator


def test_classification_metrics_battery():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 1, 1, 0])
    y_proba = np.array([
        [0.9, 0.1], [0.8, 0.2], [0.7, 0.3], [0.4, 0.6],
        [0.2, 0.8], [0.3, 0.7], [0.1, 0.9], [0.6, 0.4],
    ])

    metrics = compute_classification_metrics(y_true, y_pred, y_proba, minority_idx=1)

    assert "accuracy" in metrics
    assert "macro_f1" in metrics
    assert "minority_recall" in metrics
    assert "auroc" in metrics
    assert metrics["accuracy"] == 0.75
    assert metrics["minority_recall"] == 0.75


def test_downstream_data_preprocessor():
    df_train = pd.DataFrame({
        "num": [1.0, 2.0, 3.0, 4.0],
        "cat": ["A", "B", "A", "C"],
        "target": ["No", "No", "Yes", "No"],
    })
    df_synth = pd.DataFrame({
        "num": [1.5, 2.5],
        "cat": ["A", "D"],  # "D" is unseen
        "target": ["No", "Yes"],
    })
    df_test = pd.DataFrame({
        "num": [2.0, 3.5],
        "cat": ["B", "A"],
        "target": ["No", "Yes"],
    })

    prep = DownstreamDataPreprocessor(target_column="target")
    (
        X_tr, y_tr,
        X_sy, y_sy,
        X_te, y_te,
        min_idx,
    ) = prep.fit_transform_splits(df_train, df_synth, df_test)

    assert X_tr.shape[0] == 4
    assert X_sy.shape[0] == 2
    assert X_te.shape[0] == 2
    assert X_tr.shape[1] == X_sy.shape[1] == X_te.shape[1]
    assert len(y_tr) == 4
    assert len(y_sy) == 2
    assert len(y_te) == 2


def test_trtr_and_tstr_evaluation():
    np.random.seed(42)
    n = 80
    X_tr = np.random.randn(n, 4).astype(np.float32)
    y_tr = (X_tr[:, 0] > 0).astype(int)

    X_sy = np.random.randn(n, 4).astype(np.float32)
    y_sy = (X_sy[:, 0] > 0).astype(int)

    X_te = np.random.randn(40, 4).astype(np.float32)
    y_te = (X_te[:, 0] > 0).astype(int)

    trtr_res = evaluate_trtr(X_tr, y_tr, X_te, y_te, classifiers=["logistic_regression"])
    assert "logistic_regression" in trtr_res
    assert "macro_f1" in trtr_res["logistic_regression"]

    tstr_res = evaluate_tstr(X_sy, y_sy, X_te, y_te, trtr_results=trtr_res, classifiers=["logistic_regression"])
    assert "logistic_regression" in tstr_res
    assert "relative_f1_parity" in tstr_res["logistic_regression"]
    assert "macro_f1_delta" in tstr_res["logistic_regression"]


def test_minority_augmentation_study():
    np.random.seed(42)
    n = 60
    X_tr = np.random.randn(n, 4).astype(np.float32)
    y_tr = np.random.choice([0, 1], p=[0.8, 0.2], size=n)

    X_sy = np.random.randn(n, 4).astype(np.float32)
    y_sy = np.random.choice([0, 1], p=[0.5, 0.5], size=n)

    X_te = np.random.randn(30, 4).astype(np.float32)
    y_te = np.random.choice([0, 1], p=[0.8, 0.2], size=30)

    aug_res = evaluate_minority_augmentation(
        X_train_real=X_tr,
        y_train_real=y_tr,
        X_train_synth=X_sy,
        y_train_synth=y_sy,
        X_test_real=X_te,
        y_test_real=y_te,
        augmentation_ratios=[0.5, 1.0],
        classifiers=["logistic_regression"],
    )

    assert "by_ratio" in aug_res
    assert "0.0" in aug_res["by_ratio"]
    assert "0.5" in aug_res["by_ratio"]
    assert "1.0" in aug_res["by_ratio"]
