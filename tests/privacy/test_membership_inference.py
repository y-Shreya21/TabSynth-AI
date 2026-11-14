"""Tests for Membership Inference, Attribute Inference, and Singling-Out attacks."""

import pytest
import numpy as np
import pandas as pd

from tabsynth.evaluation.privacy.membership_inference import evaluate_membership_inference
from tabsynth.evaluation.privacy.attribute_inference import evaluate_attribute_inference
from tabsynth.evaluation.privacy.singling_out import evaluate_singling_out
from tabsynth.evaluation.privacy.privacy_report import PrivacyAuditor


@pytest.fixture
def privacy_sample_splits():
    np.random.seed(42)
    n = 60
    df_tr = pd.DataFrame({
        "age": np.random.randint(20, 60, size=n).astype(float),
        "education": np.random.choice(["Bachelors", "HS-grad"], size=n),
        "income": np.random.choice(["<=50K", ">50K"], size=n),
    })
    df_te = pd.DataFrame({
        "age": np.random.randint(20, 60, size=n).astype(float),
        "education": np.random.choice(["Bachelors", "HS-grad"], size=n),
        "income": np.random.choice(["<=50K", ">50K"], size=n),
    })
    df_sy = pd.DataFrame({
        "age": np.random.randint(20, 60, size=n).astype(float),
        "education": np.random.choice(["Bachelors", "HS-grad"], size=n),
        "income": np.random.choice(["<=50K", ">50K"], size=n),
    })
    return df_tr, df_te, df_sy


def test_membership_inference_attack(privacy_sample_splits):
    df_tr, df_te, df_sy = privacy_sample_splits
    mia_res = evaluate_membership_inference(
        df_train=df_tr,
        df_test=df_te,
        df_synth=df_sy,
        continuous_cols=["age"],
        categorical_cols=["education", "income"],
        subsample_size=40,
    )

    assert "mia_roc_auc" in mia_res
    assert 0.0 <= mia_res["mia_roc_auc"] <= 1.0
    assert mia_res["risk_level"] in ("LOW_RISK", "MODERATE_RISK", "HIGH_RISK")


def test_attribute_inference_attack(privacy_sample_splits):
    df_tr, df_te, df_sy = privacy_sample_splits
    aia_res = evaluate_attribute_inference(
        df_train=df_tr,
        df_test=df_te,
        df_synth=df_sy,
        sensitive_column="income",
        quasi_identifiers=["age", "education"],
        subsample_size=40,
    )

    assert "attacker_accuracy_on_train" in aia_res
    assert "adversary_advantage_over_test" in aia_res
    assert aia_res["risk_level"] in ("LOW_RISK", "MODERATE_RISK", "HIGH_RISK")


def test_singling_out_risk(privacy_sample_splits):
    df_tr, df_te, df_sy = privacy_sample_splits
    so_res = evaluate_singling_out(
        df_train=df_tr,
        df_synth=df_sy,
        quasi_identifiers=["education", "income"],
    )

    assert "total_predicates_tested" in so_res
    assert "singling_out_risk_rate" in so_res
    assert 0.0 <= so_res["singling_out_risk_rate"] <= 1.0


def test_privacy_auditor_aggregate(privacy_sample_splits):
    df_tr, df_te, df_sy = privacy_sample_splits
    auditor = PrivacyAuditor(
        continuous_cols=["age"],
        categorical_cols=["education"],
        sensitive_column="income",
        subsample_size=30,
    )
    rep = auditor.audit(df_tr, df_te, df_sy)

    assert "overall_privacy_risk" in rep
    assert "dcr_and_nndr" in rep
    assert "membership_inference" in rep
    assert "attribute_inference" in rep
    assert "singling_out" in rep
    assert "formal_disclaimer" in rep
