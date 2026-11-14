"""Train on Real, Test on Real (TRTR) downstream utility evaluation."""

from typing import Dict, List, Any, Optional
import numpy as np
from .downstream import get_downstream_classifiers
from .classification_metrics import compute_classification_metrics


def evaluate_trtr(
    X_train_real: np.ndarray,
    y_train_real: np.ndarray,
    X_test_real: np.ndarray,
    y_test_real: np.ndarray,
    classifiers: Optional[List[str]] = None,
    minority_idx: Optional[int] = None,
    random_state: int = 42,
) -> Dict[str, Dict[str, Any]]:
    """Trains downstream classifiers on real training data and evaluates on real test data.
    
    Establishes the upper-bound reference performance for all downstream models.
    """
    all_clfs = get_downstream_classifiers(random_state=random_state)
    selected_clfs = classifiers or list(all_clfs.keys())

    if minority_idx is None:
        counts = np.bincount(y_train_real)
        minority_idx = int(np.argmin(counts))

    results: Dict[str, Dict[str, Any]] = {}

    for name in selected_clfs:
        if name not in all_clfs:
            continue

        clf = all_clfs[name]
        clf.fit(X_train_real, y_train_real)

        y_pred = clf.predict(X_test_real)
        y_proba = clf.predict_proba(X_test_real) if hasattr(clf, "predict_proba") else None

        metrics = compute_classification_metrics(
            y_true=y_test_real,
            y_pred=y_pred,
            y_proba=y_proba,
            minority_idx=minority_idx,
        )
        results[name] = metrics

    return results
