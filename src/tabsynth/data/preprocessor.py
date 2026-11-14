"""Zero-data-leakage tabular preprocessor with invertible transformations."""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer, StandardScaler
from tabsynth.data.schema import TableSchema


class TabularPreprocessor:
    """Zero-leakage tabular preprocessing pipeline with verified invertibility.
    
    Fits transformations exclusively on the training set partition D_train,
    preventing any test set information from contaminating feature scalers
    or categorical vocabularies.
    """

    def __init__(
        self,
        schema: TableSchema,
        continuous_scaler: str = "quantile",
        missing_cat_token: str = "<missing>",
        unknown_cat_token: str = "<unk>",
        seed: int = 42,
    ):
        self.schema = schema
        self.continuous_scaler_type = continuous_scaler
        self.missing_cat_token = missing_cat_token
        self.unknown_cat_token = unknown_cat_token
        self.seed = seed

        self.num_scalers: Dict[str, Any] = {}
        self.num_medians: Dict[str, float] = {}
        self.cat_to_idx: Dict[str, Dict[str, int]] = {}
        self.idx_to_cat: Dict[str, Dict[int, str]] = {}
        self.target_to_idx: Dict[str, int] = {}
        self.idx_to_target: Dict[int, str] = {}
        self.is_fitted = False

    def fit(self, df_train: pd.DataFrame) -> "TabularPreprocessor":
        """Fits scalers and categorical mappings exclusively on training data."""
        # 1. Fit Continuous Feature Scalers
        for col in self.schema.continuous_columns:
            series = df_train[col].astype(float)
            median_val = float(series.median())
            self.num_medians[col] = median_val

            imputed_vals = series.fillna(median_val).values.reshape(-1, 1)

            if self.continuous_scaler_type == "quantile":
                scaler = QuantileTransformer(
                    output_distribution="normal",
                    n_quantiles=min(1000, len(imputed_vals)),
                    subsample=int(1e5),
                    random_state=self.seed,
                )
            else:
                scaler = StandardScaler()

            scaler.fit(imputed_vals)
            self.num_scalers[col] = scaler

        # 2. Fit Categorical Feature Encodings
        for col in self.schema.categorical_columns:
            series = df_train[col].fillna(self.missing_cat_token).astype(str)
            unique_cats = sorted(series.unique().tolist())

            # Always reserve missing and unknown tokens in vocabulary
            vocab = []
            for cat in unique_cats:
                if cat != self.missing_cat_token and cat != self.unknown_cat_token:
                    vocab.append(cat)

            # Assign indices: 0..N-1 for known categories, then missing, then unknown
            cat_map = {cat: idx for idx, cat in enumerate(vocab)}
            if self.missing_cat_token not in cat_map:
                cat_map[self.missing_cat_token] = len(cat_map)
            if self.unknown_cat_token not in cat_map:
                cat_map[self.unknown_cat_token] = len(cat_map)

            inv_map = {idx: cat for cat, idx in cat_map.items()}

            self.cat_to_idx[col] = cat_map
            self.idx_to_cat[col] = inv_map

        # Update cardinalities in schema to match fitted vocabularies
        self.schema.cardinalities = [
            len(self.cat_to_idx[col]) for col in self.schema.categorical_columns
        ]

        # 3. Fit Target Column Mapping
        if self.schema.target_column is not None and self.schema.target_column in df_train.columns:
            target_series = df_train[self.schema.target_column].astype(str)
            target_cats = sorted(target_series.unique().tolist())
            self.target_to_idx = {cat: idx for idx, cat in enumerate(target_cats)}
            self.idx_to_target = {idx: cat for cat, idx in self.target_to_idx.items()}

        self.is_fitted = True
        return self

    def transform(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, List[np.ndarray], Optional[np.ndarray]]:
        """Transforms raw DataFrame into standardized continuous and one-hot categorical arrays.
        
        Returns:
            x_num: Continuous numpy array of shape (N, num_continuous).
            x_cat_list: List of one-hot numpy arrays, each of shape (N, K_j).
            labels: Integer target labels of shape (N,) or None.
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted on training data before calling transform.")

        # 1. Continuous Features
        num_cols = self.schema.continuous_columns
        if len(num_cols) > 0:
            transformed_num = []
            for col in num_cols:
                series = df[col].astype(float) if col in df.columns else pd.Series(self.num_medians[col], index=df.index)
                imputed = series.fillna(self.num_medians[col]).values.reshape(-1, 1)
                scaled = self.num_scalers[col].transform(imputed)
                # Clip extreme standard normal tails to avoid numerical instability
                scaled = np.clip(scaled, -5.0, 5.0)
                transformed_num.append(scaled)
            x_num = np.hstack(transformed_num).astype(np.float32)
        else:
            x_num = np.empty((len(df), 0), dtype=np.float32)

        # 2. Categorical Features
        x_cat_list = []
        for col in self.schema.categorical_columns:
            cat_map = self.cat_to_idx[col]
            k = len(cat_map)
            unk_idx = cat_map[self.unknown_cat_token]

            if col in df.columns:
                series = df[col].fillna(self.missing_cat_token).astype(str)
                indices = series.map(lambda c: cat_map.get(c, unk_idx)).values
            else:
                indices = np.full(len(df), cat_map[self.missing_cat_token])

            # Convert to one-hot encoding
            one_hot = np.zeros((len(df), k), dtype=np.float32)
            one_hot[np.arange(len(df)), indices] = 1.0
            x_cat_list.append(one_hot)

        # 3. Target Column
        labels = None
        if self.schema.target_column is not None and self.schema.target_column in df.columns:
            target_series = df[self.schema.target_column].astype(str)
            labels = target_series.map(lambda c: self.target_to_idx.get(c, 0)).values.astype(np.int64)

        return x_num, x_cat_list, labels

    def inverse_transform(
        self,
        x_num: np.ndarray,
        x_cat_list: List[np.ndarray],
        labels: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """Inverts scaled continuous and discrete one-hot features back into original DataFrame."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted before inverse_transform.")

        reconstructed_data: Dict[str, Any] = {}

        # 1. Invert Continuous Features
        num_cols = self.schema.continuous_columns
        if len(num_cols) > 0 and x_num.shape[1] == len(num_cols):
            for idx, col in enumerate(num_cols):
                col_scaled = x_num[:, idx : idx + 1]
                inv_vals = self.num_scalers[col].inverse_transform(col_scaled).ravel()
                # Cast to integer if originally discrete numeric
                if "int" in self.schema.columns_metadata[col].dtype:
                    inv_vals = np.round(inv_vals).astype(int)
                reconstructed_data[col] = inv_vals

        # 2. Invert Categorical Features
        for idx, col in enumerate(self.schema.categorical_columns):
            one_hot_or_logits = x_cat_list[idx]
            best_cat_indices = np.argmax(one_hot_or_logits, axis=-1)
            inv_map = self.idx_to_cat[col]
            categories = [inv_map.get(int(i), self.unknown_cat_token) for i in best_cat_indices]
            reconstructed_data[col] = categories

        # 3. Invert Target Column
        if labels is not None and self.schema.target_column is not None:
            target_strings = [self.idx_to_target.get(int(lbl), "unknown") for lbl in labels]
            reconstructed_data[self.schema.target_column] = target_strings

        # Preserve original column order
        ordered_cols = [c for c in self.schema.columns_metadata.keys() if c in reconstructed_data]
        return pd.DataFrame(reconstructed_data)[ordered_cols]
