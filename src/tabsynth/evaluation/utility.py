"""Downstream machine learning utility evaluation: TSTR vs. TRTR benchmarking.

Backward-compatible facade delegating to the modular utility subpackage.
"""

from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np

from .utility.classification_metrics import compute_classification_metrics
from .utility.downstream import DownstreamDataPreprocessor, get_downstream_classifiers
from .utility.trtr import evaluate_trtr
from .utility.tstr import evaluate_tstr
from .utility.augmentation import evaluate_minority_augmentation
from .utility.utility_report import UtilityEvaluator


def prepare_tstr_trtr_data(
    df_train_real: pd.DataFrame,
    df_train_synth: pd.DataFrame,
    df_test_real: pd.DataFrame,
    target_column: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """Encodes features and targets identically and without leakage across TRTR and TSTR."""
    preprocessor = DownstreamDataPreprocessor(target_column=target_column)
    return preprocessor.fit_transform_splits(
        df_train_real=df_train_real,
        df_train_synth=df_train_synth,
        df_test_real=df_test_real,
    )


def evaluate_downstream_utility(
    df_train_real: pd.DataFrame,
    df_train_synth: pd.DataFrame,
    df_test_real: pd.DataFrame,
    target_column: str,
    classifiers: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Evaluates Train on Real, Test on Real (TRTR) vs Train on Synthetic, Test on Real (TSTR)."""
    evaluator = UtilityEvaluator(target_column=target_column, classifiers=classifiers)
    report = evaluator.evaluate(
        df_train_real=df_train_real,
        df_train_synth=df_train_synth,
        df_test_real=df_test_real,
    )
    # Return dictionary structured for backward compatibility with existing tests
    results: Dict[str, Any] = {}
    for name, res in report["tstr"].items():
        results[name] = {
            "trtr": res.get("trtr", {}),
            "tstr": res.get("tstr", {}),
            "relative_utility_parity": res.get("relative_f1_parity", 0.0),
            "minority_recall_delta": res.get("minority_recall_delta", 0.0),
        }
    return results


__all__ = [
    "prepare_tstr_trtr_data",
    "compute_classification_metrics",
    "evaluate_downstream_utility",
    "DownstreamDataPreprocessor",
    "get_downstream_classifiers",
    "evaluate_trtr",
    "evaluate_tstr",
    "evaluate_minority_augmentation",
    "UtilityEvaluator",
]
