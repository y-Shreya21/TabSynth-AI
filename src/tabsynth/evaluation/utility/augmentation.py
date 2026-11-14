"""Synthetic minority-class augmentation evaluation."""

from typing import Dict, List, Any, Optional
import numpy as np
from .downstream import get_downstream_classifiers
from .classification_metrics import compute_classification_metrics


def evaluate_minority_augmentation(
    X_train_real: np.ndarray,
    y_train_real: np.ndarray,
    X_train_synth: np.ndarray,
    y_train_synth: np.ndarray,
    X_test_real: np.ndarray,
    y_test_real: np.ndarray,
    augmentation_ratios: Optional[List[float]] = None,
    classifiers: Optional[List[str]] = None,
    minority_idx: Optional[int] = None,
    random_state: int = 42,
) -> Dict[str, Any]:
    """Evaluates mixed real + synthetic minority-class data augmentation.
    
    Augments the real training partition with synthetic minority class samples
    at specified ratios (default: [0.25, 0.5, 1.0, 2.0]), then evaluates downstream
    classifiers on the untouched real test partition.
    """
    ratios = augmentation_ratios or [0.25, 0.5, 1.0, 2.0]
    all_clfs = get_downstream_classifiers(random_state=random_state)
    selected_clfs = classifiers or list(all_clfs.keys())

    if minority_idx is None:
        counts = np.bincount(y_train_real)
        minority_idx = int(np.argmin(counts))

    # Identify synthetic minority candidates
    synth_min_mask = (y_train_synth == minority_idx)
    X_synth_min = X_train_synth[synth_min_mask]
    y_synth_min = y_train_synth[synth_min_mask]

    n_real_min = int(np.sum(y_train_real == minority_idx))
    n_synth_available = len(X_synth_min)

    rng = np.random.default_rng(random_state)

    # 1. Baseline: Ratio 0.0 (Pure Real)
    baseline_results: Dict[str, Dict[str, Any]] = {}
    for name in selected_clfs:
        if name not in all_clfs:
            continue
        clf = get_downstream_classifiers(random_state=random_state)[name]
        clf.fit(X_train_real, y_train_real)
        y_pred = clf.predict(X_test_real)
        y_proba = clf.predict_proba(X_test_real) if hasattr(clf, "predict_proba") else None
        baseline_results[name] = compute_classification_metrics(
            y_true=y_test_real,
            y_pred=y_pred,
            y_proba=y_proba,
            minority_idx=minority_idx,
        )

    ratio_results: Dict[str, Dict[str, Any]] = {}
    ratio_results["0.0"] = {
        "n_synthetic_added": 0,
        "models": baseline_results,
    }

    # 2. Iterate across augmentation ratios
    for ratio in ratios:
        n_to_add = int(np.round(ratio * n_real_min))
        if n_to_add == 0:
            continue

        if n_synth_available == 0:
            # No synthetic minority samples available
            sampled_indices = np.array([], dtype=int)
        elif n_synth_available >= n_to_add:
            sampled_indices = rng.choice(n_synth_available, size=n_to_add, replace=False)
        else:
            # Oversample with replacement if needed
            sampled_indices = rng.choice(n_synth_available, size=n_to_add, replace=True)

        if len(sampled_indices) > 0:
            X_aug_sub = X_synth_min[sampled_indices]
            y_aug_sub = y_synth_min[sampled_indices]
            X_train_augmented = np.concatenate([X_train_real, X_aug_sub], axis=0)
            y_train_augmented = np.concatenate([y_train_real, y_aug_sub], axis=0)
        else:
            X_train_augmented = X_train_real.copy()
            y_train_augmented = y_train_real.copy()

        models_at_ratio: Dict[str, Any] = {}
        for name in selected_clfs:
            if name not in all_clfs:
                continue
            clf = get_downstream_classifiers(random_state=random_state)[name]
            clf.fit(X_train_augmented, y_train_augmented)
            y_pred = clf.predict(X_test_real)
            y_proba = clf.predict_proba(X_test_real) if hasattr(clf, "predict_proba") else None
            metrics = compute_classification_metrics(
                y_true=y_test_real,
                y_pred=y_pred,
                y_proba=y_proba,
                minority_idx=minority_idx,
            )

            # Compare to baseline
            base_m = baseline_results[name]
            delta = {
                "minority_recall_gain": float(metrics["minority_recall"] - base_m["minority_recall"]),
                "minority_f1_gain": float(metrics["minority_f1"] - base_m["minority_f1"]),
                "macro_f1_gain": float(metrics["macro_f1"] - base_m["macro_f1"]),
                "balanced_acc_gain": float(metrics["balanced_accuracy"] - base_m["balanced_accuracy"]),
            }

            models_at_ratio[name] = {
                "metrics": metrics,
                "deltas_vs_baseline": delta,
            }

        ratio_results[str(ratio)] = {
            "n_synthetic_added": int(len(sampled_indices)),
            "models": models_at_ratio,
        }

    return {
        "minority_class_index": minority_idx,
        "n_real_minority": n_real_min,
        "n_synth_minority_available": n_synth_available,
        "by_ratio": ratio_results,
    }
