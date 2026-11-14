"""Train on Synthetic, Test on Real (TSTR) downstream utility evaluation."""

from typing import Dict, List, Any, Optional
import numpy as np
from .downstream import get_downstream_classifiers
from .classification_metrics import compute_classification_metrics


def evaluate_tstr(
    X_train_synth: np.ndarray,
    y_train_synth: np.ndarray,
    X_test_real: np.ndarray,
    y_test_real: np.ndarray,
    trtr_results: Optional[Dict[str, Dict[str, Any]]] = None,
    classifiers: Optional[List[str]] = None,
    minority_idx: Optional[int] = None,
    random_state: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """Trains downstream classifiers on synthetic data and evaluates on real test data.
    
    Computes absolute utility metrics and compares against TRTR performance to calculate
    macro-F1 parity ratios, accuracy gaps, and minority recall deltas.
    """
    all_clfs = get_downstream_classifiers(random_state=random_state)
    selected_clfs = classifiers or list(all_clfs.keys())

    if minority_idx is None:
        counts = np.bincount(y_test_real)
        minority_idx = int(np.argmin(counts))

    results: Dict[str, Dict[str, Any]] = {}

    for name in selected_clfs:
        if name not in all_clfs:
            continue

        clf = all_clfs[name]
        clf.fit(X_train_synth, y_train_synth)

        y_pred = clf.predict(X_test_real)
        y_proba = clf.predict_proba(X_test_real) if hasattr(clf, "predict_proba") else None

        tstr_metrics = compute_classification_metrics(
            y_true=y_test_real,
            y_pred=y_pred,
            y_proba=y_proba,
            minority_idx=minority_idx,
        )

        model_result: Dict[str, Any] = {"tstr": tstr_metrics}

        if trtr_results is not None and name in trtr_results:
            trtr_metrics = trtr_results[name]
            model_result["trtr"] = trtr_metrics

            # Calculate Parity and Deltas
            ref_macro_f1 = max(1e-4, trtr_metrics["macro_f1"])
            ref_acc = max(1e-4, trtr_metrics["accuracy"])

            model_result["relative_f1_parity"] = float(tstr_metrics["macro_f1"] / ref_macro_f1)
            model_result["relative_accuracy_parity"] = float(tstr_metrics["accuracy"] / ref_acc)
            model_result["macro_f1_delta"] = float(tstr_metrics["macro_f1"] - trtr_metrics["macro_f1"])
            model_result["accuracy_delta"] = float(tstr_metrics["accuracy"] - trtr_metrics["accuracy"])
            model_result["minority_recall_delta"] = float(tstr_metrics["minority_recall"] - trtr_metrics["minority_recall"])
            model_result["minority_f1_delta"] = float(tstr_metrics["minority_f1"] - trtr_metrics["minority_f1"])
            model_result["auroc_delta"] = float(tstr_metrics["auroc"] - trtr_metrics["auroc"])

        results[name] = model_result

    return results
