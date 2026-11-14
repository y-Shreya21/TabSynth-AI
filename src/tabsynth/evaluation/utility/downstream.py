"""Downstream classifier definitions and leakage-free tabular data preparation."""

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler


def get_downstream_classifiers(random_state: int = 42) -> Dict[str, Any]:
    """Returns a standardized dictionary of downstream classifier instances."""
    return {
        "logistic_regression": LogisticRegression(max_iter=500, random_state=random_state),
        "random_forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=random_state, n_jobs=-1),
        "hist_gradient_boosting": HistGradientBoostingClassifier(max_iter=100, max_depth=6, random_state=random_state),
    }


class DownstreamDataPreprocessor:
    """Encodes tabular features and target labels without data leakage across splits.
    
    All scalers, encoders, and imputation statistics are fitted strictly on the
    training partition of the real dataset.
    """

    def __init__(self, target_column: str):
        self.target_column = target_column
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_columns: List[str] = []
        self.dummy_columns: List[str] = []
        self.train_medians: Optional[pd.Series] = None
        self.minority_class_idx: int = 0
        self.is_fitted: bool = False

    def fit(self, df_train_real: pd.DataFrame) -> "DownstreamDataPreprocessor":
        """Fits encoders and scalers on real training data."""
        y_raw = df_train_real[self.target_column].astype(str).values
        y_encoded = self.label_encoder.fit_transform(y_raw)
        
        counts = np.bincount(y_encoded)
        self.minority_class_idx = int(np.argmin(counts))

        X_df = df_train_real.drop(columns=[self.target_column])
        self.feature_columns = list(X_df.columns)

        # One-hot encode categoricals
        X_encoded = pd.get_dummies(X_df, drop_first=True)
        self.dummy_columns = list(X_encoded.columns)

        # Imputation medians
        self.train_medians = X_encoded.median(numeric_only=True)
        X_imputed = X_encoded.fillna(self.train_medians).values.astype(np.float32)

        # Scaler
        self.scaler.fit(X_imputed)
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Transforms a DataFrame using pre-fitted encoders and scalers.
        
        Aligns columns with the training feature schema and imputes unknown levels.
        """
        if not self.is_fitted:
            raise RuntimeError("DownstreamDataPreprocessor must be fitted before transform.")

        y_encoded: Optional[np.ndarray] = None
        if self.target_column in df.columns:
            known_classes = set(self.label_encoder.classes_)
            y_raw = df[self.target_column].astype(str).values
            y_encoded = np.array([
                self.label_encoder.transform([c])[0] if c in known_classes else 0
                for c in y_raw
            ], dtype=np.int64)

        X_df = df.drop(columns=[self.target_column]) if self.target_column in df.columns else df.copy()

        # One-hot encode and reindex to match train dummy columns exactly
        X_encoded = pd.get_dummies(X_df, drop_first=True)
        X_aligned = X_encoded.reindex(columns=self.dummy_columns, fill_value=0.0)

        # Impute
        X_imputed = X_aligned.fillna(self.train_medians).values.astype(np.float32)

        # Scale
        X_scaled = self.scaler.transform(X_imputed)

        return X_scaled, y_encoded

    def fit_transform_splits(
        self,
        df_train_real: pd.DataFrame,
        df_train_synth: pd.DataFrame,
        df_test_real: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
        """Convenience method to process train_real, train_synth, and test_real."""
        self.fit(df_train_real)
        X_tr_real, y_tr_real = self.transform(df_train_real)
        X_tr_synth, y_tr_synth = self.transform(df_train_synth)
        X_te_real, y_te_real = self.transform(df_test_real)

        assert y_tr_real is not None and y_tr_synth is not None and y_te_real is not None

        return (
            X_tr_real,
            y_tr_real,
            X_tr_synth,
            y_tr_synth,
            X_te_real,
            y_te_real,
            self.minority_class_idx,
        )
