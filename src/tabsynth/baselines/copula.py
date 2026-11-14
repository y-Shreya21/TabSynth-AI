"""Gaussian Copula baseline model for tabular data synthesis."""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy.stats import norm
from tabsynth.data.schema import TableSchema


class GaussianCopulaBaseline:
    """Parametric Gaussian Copula synthesizer for mixed tabular features.
    
    Transforms continuous marginals into standard normal variables using
    the probability integral transform, models linear correlations via a
    covariance matrix, and samples discrete categories from marginal frequencies.
    """

    def __init__(self, schema: TableSchema, seed: int = 42):
        self.schema = schema
        self.seed = seed
        self.mean_vector: Optional[np.ndarray] = None
        self.cov_matrix: Optional[np.ndarray] = None
        self.cat_marginals: Dict[str, Dict[str, float]] = {}
        self.num_quantiles: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self.is_fitted = False

    def fit(self, df: pd.DataFrame) -> "GaussianCopulaBaseline":
        """Fits empirical marginal CDFs and normal copula covariance."""
        np.random.seed(self.seed)

        # 1. Fit Continuous Features via Empirical CDF to Standard Normal
        norm_continuous = []
        for col in self.schema.continuous_columns:
            vals = df[col].dropna().values.astype(float)
            sorted_vals = np.sort(vals)
            n = len(sorted_vals)
            # Uniform empirical probabilities in (0, 1)
            probs = (np.arange(1, n + 1) - 0.5) / float(n)
            self.num_quantiles[col] = (sorted_vals, probs)

            # Map sample values to standard normal z-scores: Phi^{-1}(F(x))
            # Rank interpolate
            sample_probs = np.interp(df[col].fillna(np.median(vals)).values, sorted_vals, probs)
            sample_probs = np.clip(sample_probs, 1e-4, 1.0 - 1e-4)
            z_scores = norm.ppf(sample_probs)
            norm_continuous.append(z_scores.reshape(-1, 1))

        if len(norm_continuous) > 0:
            Z = np.hstack(norm_continuous)
            self.mean_vector = np.mean(Z, axis=0)
            self.cov_matrix = np.cov(Z, rowvar=False)
            # Regularize covariance matrix to guarantee positive semi-definiteness
            self.cov_matrix += np.eye(self.cov_matrix.shape[0]) * 1e-4

        # 2. Fit Discrete Categorical Marginals
        for col in self.schema.categorical_columns:
            counts = df[col].fillna("<missing>").astype(str).value_counts(normalize=True)
            self.cat_marginals[col] = {str(k): float(v) for k, v in counts.items()}

        # 3. Fit Target Column Marginals
        if self.schema.target_column is not None and self.schema.target_column in df.columns:
            t_counts = df[self.schema.target_column].astype(str).value_counts(normalize=True)
            self.cat_marginals[self.schema.target_column] = {str(k): float(v) for k, v in t_counts.items()}

        self.is_fitted = True
        return self

    def sample(self, num_samples: int = 1000) -> pd.DataFrame:
        """Samples synthetic records from the fitted Gaussian copula."""
        if not self.is_fitted:
            raise RuntimeError("GaussianCopulaBaseline must be fitted before sampling.")

        np.random.seed(self.seed)
        generated_data: Dict[str, Any] = {}

        # 1. Sample Continuous Latent Normal Vector & Invert CDF
        if len(self.schema.continuous_columns) > 0 and self.cov_matrix is not None:
            Z_sample = np.random.multivariate_normal(
                mean=self.mean_vector,
                cov=self.cov_matrix,
                size=num_samples,
            )
            # Map z back to empirical quantiles: F^{-1}(Phi(z))
            for idx, col in enumerate(self.schema.continuous_columns):
                z_col = Z_sample[:, idx]
                uniform_p = norm.cdf(z_col)
                sorted_vals, probs = self.num_quantiles[col]
                x_inv = np.interp(uniform_p, probs, sorted_vals)
                if "int" in self.schema.columns_metadata[col].dtype:
                    x_inv = np.round(x_inv).astype(int)
                generated_data[col] = x_inv

        # 2. Sample Categorical Features from Marginal Probabilities
        for col in self.schema.categorical_columns:
            dist = self.cat_marginals[col]
            cats = list(dist.keys())
            p_vals = list(dist.values())
            p_vals = np.array(p_vals) / sum(p_vals)
            generated_data[col] = np.random.choice(cats, size=num_samples, p=p_vals)

        # 3. Sample Target Column
        if self.schema.target_column is not None and self.schema.target_column in self.cat_marginals:
            dist = self.cat_marginals[self.schema.target_column]
            cats = list(dist.keys())
            p_vals = np.array(list(dist.values())) / sum(dist.values())
            generated_data[self.schema.target_column] = np.random.choice(cats, size=num_samples, p=p_vals)

        ordered_cols = [c for c in self.schema.columns_metadata.keys() if c in generated_data]
        return pd.DataFrame(generated_data)[ordered_cols]
