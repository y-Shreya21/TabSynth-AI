"""Native PyTorch Tabular Variational Autoencoder (TVAE) synthesizer baseline."""

import time
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from tabsynth.data.schema import TableSchema, infer_table_schema
from .base import BaseSynthesizer


class TVAEModule(nn.Module):
    """PyTorch VAE architecture for mixed-type tabular data."""

    def __init__(
        self,
        num_continuous: int,
        cardinalities: List[int],
        latent_dim: int = 32,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.num_continuous = num_continuous
        self.cardinalities = cardinalities
        self.latent_dim = latent_dim
        self.total_cat_dim = sum(cardinalities)
        self.input_dim = num_continuous + self.total_cat_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # Decoder
        self.decoder_hidden = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        if num_continuous > 0:
            self.decoder_num = nn.Linear(hidden_dim, num_continuous)
        else:
            self.decoder_num = None

        self.decoder_cat = nn.ModuleList([
            nn.Linear(hidden_dim, card) for card in cardinalities
        ])

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * torch.clamp(logvar, -10.0, 10.0))
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> Tuple[Optional[torch.Tensor], List[torch.Tensor]]:
        h = self.decoder_hidden(z)
        x_num = self.decoder_num(h) if self.decoder_num is not None else None
        x_cat_logits = [head(h) for head in self.decoder_cat]
        return x_num, x_cat_logits

    def forward(self, x: torch.Tensor) -> Tuple[Optional[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_num, x_cat_logits = self.decode(z)
        return x_num, x_cat_logits, mu, logvar


class TVAESynthesizer(BaseSynthesizer):
    """Native PyTorch Tabular VAE synthesizer baseline."""

    def __init__(
        self,
        schema: Optional[TableSchema] = None,
        latent_dim: int = 32,
        hidden_dim: int = 128,
        batch_size: int = 128,
        epochs: int = 40,
        lr: float = 1e-3,
        kl_weight: float = 0.5,
        device: Optional[str] = None,
        seed: int = 42,
        **kwargs,
    ):
        super().__init__(name="tvae", latent_dim=latent_dim, hidden_dim=hidden_dim, **kwargs)
        self.schema = schema
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.kl_weight = kl_weight
        self.seed = seed

        # Device selection: MPS > CPU
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

        self.model: Optional[TVAEModule] = None
        self.continuous_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.target_column: Optional[str] = None
        self.category_maps: Dict[str, List[str]] = {}
        self.num_means: np.ndarray = np.array([])
        self.num_stds: np.ndarray = np.array([])

    def _prepare_data(self, df: pd.DataFrame) -> torch.Tensor:
        """Converts DataFrame into numerical + one-hot concatenated tensor."""
        tensors = []

        # 1. Continuous
        if len(self.continuous_cols) > 0:
            num_vals = df[self.continuous_cols].fillna(pd.Series(self.num_means, index=self.continuous_cols)).values.astype(np.float32)
            num_scaled = (num_vals - self.num_means) / np.clip(self.num_stds, 1e-6, None)
            tensors.append(torch.tensor(num_scaled, dtype=torch.float32))

        # 2. Categoricals
        for col in self.categorical_cols:
            cats = self.category_maps[col]
            cat_to_idx = {c: i for i, c in enumerate(cats)}
            series_str = df[col].astype(str).values
            indices = np.array([cat_to_idx.get(val, 0) for val in series_str], dtype=np.int64)
            one_hot = F.one_hot(torch.tensor(indices, dtype=torch.long), num_classes=len(cats)).float()
            tensors.append(one_hot)

        return torch.cat(tensors, dim=-1)

    def fit(self, df_train: pd.DataFrame, target_column: Optional[str] = None) -> "TVAESynthesizer":
        """Fits the TVAE model using Adam optimizer."""
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        start_t = time.time()

        self.target_column = target_column or (self.schema.target_column if self.schema else None)
        if self.schema is None:
            self.schema = infer_table_schema(df_train, target_column=self.target_column)

        self.continuous_cols = list(self.schema.continuous_columns)
        self.categorical_cols = list(self.schema.categorical_columns)
        if self.target_column and self.target_column not in self.continuous_cols and self.target_column not in self.categorical_cols:
            self.categorical_cols.append(self.target_column)

        # Precompute normalization statistics
        if len(self.continuous_cols) > 0:
            self.num_means = df_train[self.continuous_cols].mean(numeric_only=True).values.astype(np.float32)
            self.num_stds = df_train[self.continuous_cols].std(numeric_only=True).values.astype(np.float32)
            self.num_stds = np.where(self.num_stds < 1e-6, 1.0, self.num_stds)

        # Precompute category maps
        cardinalities = []
        for col in self.categorical_cols:
            unique_cats = sorted(df_train[col].dropna().astype(str).unique().tolist())
            if "<missing>" not in unique_cats:
                unique_cats.append("<missing>")
            self.category_maps[col] = unique_cats
            cardinalities.append(len(unique_cats))

        # Prepare dataset
        X_train_tensor = self._prepare_data(df_train)
        dataset = TensorDataset(X_train_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Build model
        self.model = TVAEModule(
            num_continuous=len(self.continuous_cols),
            cardinalities=cardinalities,
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
        ).to(self.device)

        self.parameter_count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # Training loop
        self.model.train()
        for epoch in range(self.epochs):
            for batch in loader:
                x_batch = batch[0].to(self.device)
                optimizer.zero_grad()

                x_num_pred, x_cat_preds, mu, logvar = self.model(x_batch)

                # Reconstruction loss
                rec_loss = torch.tensor(0.0, device=self.device)
                cursor = 0

                # Continuous MSE
                if len(self.continuous_cols) > 0:
                    n_cont = len(self.continuous_cols)
                    target_num = x_batch[:, cursor : cursor + n_cont]
                    cursor += n_cont
                    rec_loss += F.mse_loss(x_num_pred, target_num, reduction="mean")

                # Categorical CrossEntropy
                for idx, (col, card) in enumerate(zip(self.categorical_cols, cardinalities)):
                    target_onehot = x_batch[:, cursor : cursor + card]
                    target_labels = torch.argmax(target_onehot, dim=-1)
                    cursor += card
                    rec_loss += F.cross_entropy(x_cat_preds[idx], target_labels, reduction="mean")

                # KL divergence
                kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
                total_loss = rec_loss + self.kl_weight * kl_div

                total_loss.backward()
                optimizer.step()

        self.training_time_sec = float(time.time() - start_t)
        self.is_fitted = True
        return self

    def sample(self, num_samples: int) -> pd.DataFrame:
        """Samples synthetic records by sampling from prior z ~ N(0, I)."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("TVAESynthesizer must be fitted before sampling.")

        self.model.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim, device=self.device)
            x_num_pred, x_cat_logits = self.model.decode(z)

            res_dict: Dict[str, Any] = {}

            # 1. Invert Continuous
            if len(self.continuous_cols) > 0 and x_num_pred is not None:
                x_num_np = x_num_pred.cpu().numpy()
                x_num_unscaled = x_num_np * self.num_stds + self.num_means
                for idx, col in enumerate(self.continuous_cols):
                    res_dict[col] = x_num_unscaled[:, idx]

            # 2. Decode Categoricals
            for idx, col in enumerate(self.categorical_cols):
                logits = x_cat_logits[idx]
                probs = F.softmax(logits, dim=-1).cpu().numpy()
                cats = self.category_maps[col]
                # Sample or argmax
                pred_indices = np.array([np.random.choice(len(cats), p=probs[i]) for i in range(num_samples)])
                res_dict[col] = [cats[k] for k in pred_indices]

        df_out = pd.DataFrame(res_dict)
        # Ensure column ordering matches schema
        ordered = [c for c in self.schema.columns_metadata.keys() if c in df_out.columns]
        return df_out[ordered]

    def sample_conditional(self, class_condition: Any, num_samples: int) -> pd.DataFrame:
        """Generates class-conditioned records via rejection sampling."""
        if not self.is_fitted or self.model is None:
            raise RuntimeError("TVAESynthesizer must be fitted before sampling.")
        if self.target_column is None:
            raise ValueError("Target column must be specified for conditional sampling.")

        collected = []
        batch_size = max(num_samples * 3, 500)
        target_str = str(class_condition)

        for _ in range(25):
            df_batch = self.sample(batch_size)
            matching = df_batch[df_batch[self.target_column].astype(str) == target_str]
            if len(matching) > 0:
                collected.append(matching)
                if sum(len(b) for b in collected) >= num_samples:
                    break

        if len(collected) > 0:
            df_res = pd.concat(collected, axis=0).reset_index(drop=True)
            if len(df_res) >= num_samples:
                return df_res.iloc[:num_samples]
            else:
                extra = self.sample(num_samples - len(df_res))
                extra[self.target_column] = class_condition
                return pd.concat([df_res, extra], axis=0).reset_index(drop=True)
        else:
            res = self.sample(num_samples)
            res[self.target_column] = class_condition
            return res

    def save(self, path: str) -> None:
        """Saves model weights and parameters."""
        if self.model is None:
            raise RuntimeError("Model is not initialized.")
        save_dict = {
            "model_state_dict": self.model.state_dict(),
            "schema": self.schema,
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "continuous_cols": self.continuous_cols,
            "categorical_cols": self.categorical_cols,
            "target_column": self.target_column,
            "category_maps": self.category_maps,
            "num_means": self.num_means,
            "num_stds": self.num_stds,
            "training_time_sec": self.training_time_sec,
            "parameter_count": self.parameter_count,
            "seed": self.seed,
        }
        torch.save(save_dict, path)

    def load(self, path: str) -> "TVAESynthesizer":
        """Loads model weights and parameters from disk."""
        data = torch.load(path, map_location=self.device, weights_only=False)
        self.schema = data["schema"]
        self.latent_dim = data["latent_dim"]
        self.hidden_dim = data["hidden_dim"]
        self.continuous_cols = data["continuous_cols"]
        self.categorical_cols = data["categorical_cols"]
        self.target_column = data["target_column"]
        self.category_maps = data["category_maps"]
        self.num_means = data["num_means"]
        self.num_stds = data["num_stds"]
        self.training_time_sec = data.get("training_time_sec", 0.0)
        self.parameter_count = data.get("parameter_count", 0)
        self.seed = data.get("seed", 42)

        cardinalities = [len(self.category_maps[col]) for col in self.categorical_cols]
        self.model = TVAEModule(
            num_continuous=len(self.continuous_cols),
            cardinalities=cardinalities,
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
        ).to(self.device)
        self.model.load_state_dict(data["model_state_dict"])
        self.is_fitted = True
        return self
