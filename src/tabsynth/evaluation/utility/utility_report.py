"""Master UtilityEvaluator orchestrating TRTR, TSTR, and minority augmentation."""

from typing import Dict, List, Any, Optional
import pandas as pd
from .downstream import DownstreamDataPreprocessor, get_downstream_classifiers
from .trtr import evaluate_trtr
from .tstr import evaluate_tstr
from .augmentation import evaluate_minority_augmentation


class UtilityEvaluator:
    """End-to-end downstream utility evaluator."""

    def __init__(
        self,
        target_column: str,
        classifiers: Optional[List[str]] = None,
        augmentation_ratios: Optional[List[float]] = None,
        random_state: int = 42,
    ):
        self.target_column = target_column
        self.classifiers = classifiers or ["logistic_regression", "random_forest", "hist_gradient_boosting"]
        self.augmentation_ratios = augmentation_ratios or [0.25, 0.5, 1.0, 2.0]
        self.random_state = random_state
        self.preprocessor = DownstreamDataPreprocessor(target_column=target_column)

    def evaluate(
        self,
        df_train_real: pd.DataFrame,
        df_train_synth: pd.DataFrame,
        df_test_real: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Runs the complete downstream utility benchmark suite.
        
        Guarantees zero data leakage: feature encoders, scalers, and imputation
        are fitted strictly on df_train_real.
        """
        (
            X_tr_real,
            y_tr_real,
            X_tr_synth,
            y_tr_synth,
            X_te_real,
            y_te_real,
            minority_idx,
        ) = self.preprocessor.fit_transform_splits(
            df_train_real=df_train_real,
            df_train_synth=df_train_synth,
            df_test_real=df_test_real,
        )

        # 1. TRTR Evaluation (Upper Bound)
        trtr_results = evaluate_trtr(
            X_train_real=X_tr_real,
            y_train_real=y_tr_real,
            X_test_real=X_te_real,
            y_test_real=y_te_real,
            classifiers=self.classifiers,
            minority_idx=minority_idx,
            random_state=self.random_state,
        )

        # 2. TSTR Evaluation (Synthetic Utility vs Upper Bound)
        tstr_results = evaluate_tstr(
            X_train_synth=X_tr_synth,
            y_train_synth=y_tr_synth,
            X_test_real=X_te_real,
            y_test_real=y_te_real,
            trtr_results=trtr_results,
            classifiers=self.classifiers,
            minority_idx=minority_idx,
            random_state=self.random_state,
        )

        # 3. Minority Augmentation Evaluation
        augmentation_results = evaluate_minority_augmentation(
            X_train_real=X_tr_real,
            y_train_real=y_tr_real,
            X_train_synth=X_tr_synth,
            y_train_synth=y_tr_synth,
            X_test_real=X_te_real,
            y_test_real=y_te_real,
            augmentation_ratios=self.augmentation_ratios,
            classifiers=self.classifiers,
            minority_idx=minority_idx,
            random_state=self.random_state,
        )

        # Compute aggregate summary across classifiers
        f1_parities = [
            res["relative_f1_parity"]
            for res in tstr_results.values()
            if "relative_f1_parity" in res
        ]
        mean_f1_parity = float(sum(f1_parities) / len(f1_parities)) if f1_parities else 0.0

        return {
            "target_column": self.target_column,
            "minority_class_index": minority_idx,
            "mean_f1_parity": mean_f1_parity,
            "trtr": trtr_results,
            "tstr": tstr_results,
            "augmentation": augmentation_results,
        }
