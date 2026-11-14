"""Comprehensive classification metrics suite for downstream evaluation."""

from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    minority_idx: Optional[int] = None,
) -> Dict[str, Any]:
    """Computes a complete battery of classification metrics.
    
    Includes overall metrics, threshold-independent AUROC/AUPRC, and dedicated
    minority-class recall, precision, and F1 to address extreme class imbalance.
    """
    classes = np.unique(y_true)
    num_classes = len(classes)

    # Determine minority class if not specified
    if minority_idx is None:
        counts = np.bincount(y_true)
        minority_idx = int(np.argmin(counts))

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))

    # Minority-class metrics
    min_prec = float(precision_score(y_true == minority_idx, y_pred == minority_idx, zero_division=0))
    min_rec = float(recall_score(y_true == minority_idx, y_pred == minority_idx, zero_division=0))
    min_f1 = float(f1_score(y_true == minority_idx, y_pred == minority_idx, zero_division=0))

    # Probabilistic curves (AUROC, AUPRC)
    auroc = 0.50
    auprc = 0.0
    if y_proba is not None:
        try:
            if num_classes == 2:
                prob_pos = y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba.ravel()
                auroc = float(roc_auc_score(y_true, prob_pos))
                auprc = float(average_precision_score(y_true == minority_idx, prob_pos))
            else:
                auroc = float(roc_auc_score(y_true, y_proba, multi_class="ovr"))
                auprc = float(average_precision_score(pd.get_dummies(y_true).values, y_proba, average="macro"))
        except Exception:
            auroc = 0.50
            auprc = 0.0

    cm = confusion_matrix(y_true, y_pred).tolist()

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "balanced_accuracy": bal_acc,
        "precision": prec,
        "recall": rec,
        "minority_precision": min_prec,
        "minority_recall": min_rec,
        "minority_f1": min_f1,
        "minority_class_index": minority_idx,
        "auroc": auroc,
        "auprc": auprc,
        "confusion_matrix": cm,
    }
