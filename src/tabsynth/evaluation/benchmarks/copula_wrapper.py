"""Gaussian Copula baseline wrapped under the BaseSynthesizer interface."""

import time
import pickle
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from tabsynth.data.schema import TableSchema, infer_table_schema
from tabsynth.baselines.copula import GaussianCopulaBaseline
from .base import BaseSynthesizer


class GaussianCopulaSynthesizer(BaseSynthesizer):
    """Gaussian Copula baseline synthesizer."""

    def __init__(self, schema: Optional[TableSchema] = None, seed: int = 42, **kwargs):
        super().__init__(name="gaussian_copula", seed=seed, **kwargs)
        self.schema = schema
        self.seed = seed
        self.model: Optional[GaussianCopulaBaseline] = None
        self.target_column: Optional[str] = None

    def fit(self, df_train: pd.DataFrame, target_column: Optional[str] = None) -> "GaussianCopulaSynthesizer":
        """Fits Gaussian Copula marginals and covariance matrix."""
        start_t = time.time()
        self.target_column = target_column or (self.schema.target_column if self.schema else None)
        if self.schema is None:
            self.schema = infer_table_schema(df_train, target_column=self.target_column)

        self.model = GaussianCopulaBaseline(schema=self.schema, seed=self.seed)
        self.model.fit(df_train)
        self.training_time_sec = float(time.time() - start_t)
        self.is_fitted = True

        # Covariance matrix size
        n_cont = len(self.schema.continuous_columns)
        self.parameter_count = n_cont + (n_cont * (n_cont + 1)) // 2
        return self

    def sample(self, num_samples: int) -> pd.DataFrame:
        """Samples synthetic records unconditionally."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("GaussianCopulaSynthesizer must be fitted before sampling.")
        return self.model.sample(num_samples=num_samples)

    def sample_conditional(self, class_condition: Any, num_samples: int) -> pd.DataFrame:
        """Generates class-conditioned records via rejection sampling or direct assignment."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("GaussianCopulaSynthesizer must be fitted before sampling.")
        if self.target_column is None:
            raise ValueError("Target column must be specified for conditional sampling.")

        # Rejection sampling with fallback
        collected = []
        batch_size = max(num_samples * 3, 500)
        max_iters = 20
        target_str = str(class_condition)

        for _ in range(max_iters):
            batch = self.model.sample(batch_size)
            matching = batch[batch[self.target_column].astype(str) == target_str]
            if len(matching) > 0:
                collected.append(matching)
                if sum(len(b) for b in collected) >= num_samples:
                    break

        if len(collected) > 0:
            df_res = pd.concat(collected, axis=0).reset_index(drop=True)
            if len(df_res) >= num_samples:
                return df_res.iloc[:num_samples]
            else:
                # Top up with unconditional batch and force target column
                needed = num_samples - len(df_res)
                extra = self.model.sample(needed)
                extra[self.target_column] = class_condition
                return pd.concat([df_res, extra], axis=0).reset_index(drop=True)
        else:
            # Fallback
            res = self.model.sample(num_samples)
            res[self.target_column] = class_condition
            return res

    def save(self, path: str) -> None:
        """Saves copula model to disk."""
        with open(path, "wb") as f:
            pickle.dump({
                "schema": self.schema,
                "model": self.model,
                "seed": self.seed,
                "target_column": self.target_column,
                "training_time_sec": self.training_time_sec,
                "parameter_count": self.parameter_count,
            }, f)

    def load(self, path: str) -> "GaussianCopulaSynthesizer":
        """Loads copula model from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.schema = data["schema"]
        self.model = data["model"]
        self.seed = data["seed"]
        self.target_column = data["target_column"]
        self.training_time_sec = data.get("training_time_sec", 0.0)
        self.parameter_count = data.get("parameter_count", 0)
        self.is_fitted = True
        return self
