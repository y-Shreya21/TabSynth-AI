"""Attribute Inference Attack (AIA) risk evaluation."""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder


def evaluate_attribute_inference(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    df_synth: pd.DataFrame,
    sensitive_column: str,
    quasi_identifiers: Optional[List[str]] = None,
    subsample_size: int = 2000,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Evaluates Attribute Inference Attack (AIA) risk.
    
    Adversary trains a model on synthetic data using quasi-identifiers to predict
    the sensitive attribute, then attempts reconstruction on real training records.
    """
    if sensitive_column not in df_train.columns:
        raise ValueError(f"Sensitive column '{sensitive_column}' not found in training data.")

    q_cols = quasi_identifiers or [col for col in df_train.columns if col != sensitive_column]

    n_s = min(len(df_synth), subsample_size)
    n_tr = min(len(df_train), subsample_size)
    n_te = min(len(df_test), subsample_size)

    s_sample = df_synth.sample(n=n_s, random_state=random_state)
    tr_sample = df_train.sample(n=n_tr, random_state=random_state)
    te_sample = df_test.sample(n=n_te, random_state=random_state)

    # Encode quasi-identifiers identically across splits
    X_s_df = s_sample[q_cols]
    X_tr_df = tr_sample[q_cols]
    X_te_df = te_sample[q_cols]

    X_combined = pd.concat([X_s_df, X_tr_df, X_te_df], axis=0).reset_index(drop=True)
    X_encoded = pd.get_dummies(X_combined, drop_first=True)
    medians = X_encoded.iloc[:n_s].median(numeric_only=True)
    X_imputed = X_encoded.fillna(medians).values.astype(np.float32)

    X_s = X_imputed[:n_s]
    X_tr = X_imputed[n_s : n_s + n_tr]
    X_te = X_imputed[n_s + n_tr :]

    # Determine if target is categorical or continuous
    is_categorical = (
        df_train[sensitive_column].dtype == object
        or str(df_train[sensitive_column].dtype) == "category"
        or len(df_train[sensitive_column].unique()) < 20
    )

    if is_categorical:
        le = LabelEncoder()
        y_s = le.fit_transform(s_sample[sensitive_column].astype(str).values)
        known = set(le.classes_)
        y_tr = np.array([le.transform([c])[0] if c in known else 0 for c in tr_sample[sensitive_column].astype(str).values])
        y_te = np.array([le.transform([c])[0] if c in known else 0 for c in te_sample[sensitive_column].astype(str).values])

        # Baseline: Majority class
        counts = np.bincount(y_s)
        maj_class = int(np.argmax(counts))
        baseline_acc = float(np.mean(y_tr == maj_class))

        # Train attacker on synthetic data
        attacker = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=random_state, n_jobs=-1)
        attacker.fit(X_s, y_s)

        pred_tr = attacker.predict(X_tr)
        pred_te = attacker.predict(X_te)

        acc_tr = float(np.mean(pred_tr == y_tr))
        acc_te = float(np.mean(pred_te == y_te))

        # Adversary advantage over test or majority baseline
        adv_over_baseline = float(acc_tr - baseline_acc)
        adv_over_test = float(acc_tr - acc_te)

        risk_level = "LOW_RISK" if adv_over_test <= 0.05 else ("MODERATE_RISK" if adv_over_test <= 0.15 else "HIGH_RISK")

        return {
            "sensitive_column": sensitive_column,
            "target_type": "categorical",
            "baseline_majority_accuracy": baseline_acc,
            "attacker_accuracy_on_train": acc_tr,
            "attacker_accuracy_on_test": acc_te,
            "adversary_advantage_over_baseline": adv_over_baseline,
            "adversary_advantage_over_test": adv_over_test,
            "risk_level": risk_level,
        }
    else:
        y_s = s_sample[sensitive_column].fillna(0.0).values.astype(np.float32)
        y_tr = tr_sample[sensitive_column].fillna(0.0).values.astype(np.float32)
        y_te = te_sample[sensitive_column].fillna(0.0).values.astype(np.float32)

        baseline_val = float(np.mean(y_s))
        baseline_mae = float(np.mean(np.abs(y_tr - baseline_val)))

        attacker = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=random_state, n_jobs=-1)
        attacker.fit(X_s, y_s)

        pred_tr = attacker.predict(X_tr)
        pred_te = attacker.predict(X_te)

        mae_tr = float(np.mean(np.abs(pred_tr - y_tr)))
        mae_te = float(np.mean(np.abs(pred_te - y_te)))

        mae_ratio = float(mae_tr / (mae_te if mae_te > 1e-6 else 1.0))
        risk_level = "LOW_RISK" if mae_ratio >= 0.90 else ("MODERATE_RISK" if mae_ratio >= 0.75 else "HIGH_RISK")

        return {
            "sensitive_column": sensitive_column,
            "target_type": "continuous",
            "baseline_mean_mae": baseline_mae,
            "attacker_mae_on_train": mae_tr,
            "attacker_mae_on_test": mae_te,
            "attacker_mae_ratio_train_to_test": mae_ratio,
            "risk_level": risk_level,
        }
