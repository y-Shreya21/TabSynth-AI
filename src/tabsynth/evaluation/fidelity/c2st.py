"""Classifier Two-Sample Test (C2ST) for distributional indistinguishability evaluation."""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    confusion_matrix,
)


def evaluate_c2st(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    continuous_cols: List[str],
    categorical_cols: List[str],
    n_splits: int = 5,
    classifier_type: str = "rf",
    subsample_size: int = 2000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Conducts a rigorous 5-fold Stratified Cross-Validated Classifier Two-Sample Test.
    
    Labels real records as y=1 and synthetic records as y=0. If synthetic data faithfully
    mirrors real distributions, a well-regularized discriminator cannot distinguish between
    them, yielding an accuracy near 0.50 and ROC-AUC near 0.50.
    
    Args:
        real_df: Real reference partition.
        synth_df: Synthetic partition.
        continuous_cols: Continuous column names.
        categorical_cols: Categorical column names.
        n_splits: Number of cross-validation folds.
        classifier_type: 'rf' (Random Forest), 'hist_gb' (HistGradientBoosting), or 'lr' (Logistic Regression).
        subsample_size: Maximum sample count per class to maintain efficient evaluation.
        seed: Random seed.
        
    Returns:
        Dictionary containing mean cross-validation accuracy, ROC-AUC, precision, recall,
        fold scores, and scientific interpretation.
    """
    all_cols = [c for c in continuous_cols + categorical_cols if c in real_df.columns and c in synth_df.columns]
    if len(all_cols) == 0:
        return {"c2st_accuracy": 0.50, "c2st_roc_auc": 0.50}

    # Balance class sizes to prevent trivial majority voting
    n_samples = min(len(real_df), len(synth_df), subsample_size)
    r_sub = real_df[all_cols].sample(n=n_samples, random_state=seed).copy()
    s_sub = synth_df[all_cols].sample(n=n_samples, random_state=seed).copy()

    r_sub["__c2st_target__"] = 1
    s_sub["__c2st_target__"] = 0
    merged = pd.concat([r_sub, s_sub], axis=0).reset_index(drop=True)

    X_df = merged.drop(columns=["__c2st_target__"])
    y = merged["__c2st_target__"].values

    # One-hot encode categoricals
    if len(categorical_cols) > 0:
        X_df = pd.get_dummies(X_df, columns=[c for c in categorical_cols if c in X_df.columns], drop_first=True)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    accuracies: List[float] = []
    roc_aucs: List[float] = []
    precisions: List[float] = []
    recalls: List[float] = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_df, y)):
        X_tr = X_df.iloc[train_idx].copy()
        y_tr = y[train_idx]
        X_val = X_df.iloc[val_idx].copy()
        y_val = y[val_idx]

        # Impute missing values strictly from training fold
        tr_medians = X_tr.median(numeric_only=True)
        X_tr = X_tr.fillna(tr_medians).values.astype(np.float32)
        X_val = X_val.fillna(tr_medians).values.astype(np.float32)

        # Scale features strictly from training fold
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_val = scaler.transform(X_val)

        # Instantiate classifier
        if classifier_type == "lr":
            clf = LogisticRegression(max_iter=500, random_state=seed + fold)
        elif classifier_type == "hist_gb":
            clf = HistGradientBoostingClassifier(max_iter=100, random_state=seed + fold)
        else:
            clf = RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=seed + fold)

        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_val)
        y_proba = clf.predict_proba(X_val)[:, 1] if hasattr(clf, "predict_proba") else y_pred

        accuracies.append(float(accuracy_score(y_val, y_pred)))
        precisions.append(float(precision_score(y_val, y_pred, zero_division=0)))
        recalls.append(float(recall_score(y_val, y_pred, zero_division=0)))
        try:
            roc_aucs.append(float(roc_auc_score(y_val, y_proba)))
        except Exception:
            roc_aucs.append(0.50)

    mean_acc = float(np.mean(accuracies))
    mean_auc = float(np.mean(roc_aucs))
    mean_prec = float(np.mean(precisions))
    mean_rec = float(np.mean(recalls))

    return {
        "c2st_accuracy": mean_acc,
        "c2st_roc_auc": mean_auc,
        "c2st_precision": mean_prec,
        "c2st_recall": mean_rec,
        "classifier_type": classifier_type,
        "num_folds": n_splits,
        "sample_size_per_class": n_samples,
        "fold_accuracies": accuracies,
        "fold_roc_aucs": roc_aucs,
        "interpretation": (
            "An accuracy or ROC-AUC near 0.50 indicates high indistinguishability. "
            "However, C2ST close to 0.50 alone does not establish complete distributional equivalence."
        ),
    }
