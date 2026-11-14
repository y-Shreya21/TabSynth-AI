"""Residual MLP Denoising Network modules preserving per-feature token structures."""

from typing import List, Tuple, Optional
import torch
import torch.nn as nn


class FeatureResidualMLPBlock(nn.Module):
    """Residual dense block operating on a sequence of feature tokens with conditioning."""

    def __init__(
        self,
        d_model: int,
        cond_dim: int,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        activation: str = "GELU",
    ):
        super().__init__()
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model)
        self.cond_proj = nn.Linear(cond_dim, d_model)

        act_layer = nn.GELU() if activation.upper() == "GELU" else nn.SiLU()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            act_layer,
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Args:
            x: Feature tokens (batch_size, num_features, d_model).
            cond: Conditioning embedding (batch_size, cond_dim).
            
        Returns:
            Residual updated tokens (batch_size, num_features, d_model).
        """
        residual = x
        h = self.norm(x)
        # Broadcast conditioning across all features: (B, 1, d_model)
        cond_bias = self.cond_proj(cond).unsqueeze(1)
        h = h + cond_bias
        h = self.ffn(h)
        return residual + h


class FeatureResidualMLPDenoiser(nn.Module):
    """Stack of feature-level residual MLP blocks with fused conditioning."""

    def __init__(
        self,
        d_model: int = 64,
        cond_dim: int = 64,
        dim_feedforward: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        activation: str = "GELU",
    ):
        super().__init__()
        self.d_model = d_model
        self.blocks = nn.ModuleList([
            FeatureResidualMLPBlock(
                d_model=d_model,
                cond_dim=cond_dim,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """Args:
            tokens: (batch_size, num_features, d_model).
            cond: (batch_size, cond_dim).
            
        Returns:
            contextualized_tokens: (batch_size, num_features, d_model).
        """
        h = tokens
        for block in self.blocks:
            h = block(h, cond)
        return self.final_norm(h)


# ==============================================================================
# Legacy Flat MLP Denoiser (for Phase 01 backward compatibility)
# ==============================================================================

class ResidualMLPBlock(nn.Module):
    """Residual dense block conditioned on fused time and class embeddings."""

    def __init__(self, hidden_dim: int, cond_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.cond_proj = nn.Linear(cond_dim, hidden_dim)
        self.dense1 = nn.Linear(hidden_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, hidden_dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.norm(x)
        h = h + self.cond_proj(cond)
        h = self.act(self.dense1(h))
        h = self.dropout(h)
        h = self.dense2(h)
        return residual + h


class ResidualMLPDenoiser(nn.Module):
    """Multi-layer residual MLP denoiser predicting Gaussian noise and categorical logits."""

    def __init__(
        self,
        in_dim: int,
        cond_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_continuous: int = 0,
        cardinalities: List[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_continuous = num_continuous
        self.cardinalities = cardinalities or []

        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.act = nn.SiLU()

        self.blocks = nn.ModuleList([
            ResidualMLPBlock(hidden_dim=hidden_dim, cond_dim=cond_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)

        if num_continuous > 0:
            self.head_num = nn.Linear(hidden_dim, num_continuous)
        else:
            self.head_num = None

        self.heads_cat = nn.ModuleList([
            nn.Linear(hidden_dim, K) for K in self.cardinalities
        ])

    def forward(
        self,
        h_flat: torch.Tensor,
        cond: torch.Tensor,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        h = self.act(self.input_proj(h_flat))
        for block in self.blocks:
            h = block(h, cond)
        h = self.final_norm(h)

        pred_eps = self.head_num(h) if self.head_num is not None else torch.empty(h.shape[0], 0, device=h.device)
        pred_logits = [head(h) for head in self.heads_cat]
        return pred_eps, pred_logits
