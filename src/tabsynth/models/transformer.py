"""Pre-LayerNorm Transformer Encoder for cross-feature interaction modeling."""

from typing import Optional
import torch
import torch.nn as nn


class TransformerEncoderBlock(nn.Module):
    """Pre-LN Transformer block with Multi-Head Self-Attention and GELU Feed-Forward Network."""

    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        activation: str = "GELU",
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        act_layer = nn.GELU() if activation.upper() == "GELU" else nn.SiLU()
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            act_layer,
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Args:
            x: Feature token tensor of shape (batch_size, num_features, d_model).
            attn_mask: Optional attention mask.
            
        Returns:
            out: Contextualized tokens of shape (batch_size, num_features, d_model).
        """
        # 1. Pre-LN Self-Attention
        norm_x = self.norm1(x)
        attn_out, _ = self.self_attn(norm_x, norm_x, norm_x, attn_mask=attn_mask)
        x = x + self.dropout1(attn_out)

        # 2. Pre-LN Feed-Forward
        norm_x2 = self.norm2(x)
        ffn_out = self.ffn(norm_x2)
        x = x + self.dropout2(ffn_out)

        return x


class TransformerFeatureEncoder(nn.Module):
    """Stack of Pre-LN Transformer encoder blocks for cross-column feature interaction."""

    def __init__(
        self,
        num_features: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        activation: str = "GELU",
        add_col_embeddings: bool = False,
    ):
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model
        self.add_col_embeddings = add_col_embeddings

        if add_col_embeddings:
            self.col_identity_embeddings = nn.Parameter(torch.empty(num_features, d_model))
            nn.init.normal_(self.col_identity_embeddings, std=0.02)
        else:
            self.col_identity_embeddings = None

        self.layers = nn.ModuleList([
            TransformerEncoderBlock(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
            )
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Args:
            tokens: Token tensor of shape (batch_size, num_features, d_model).
            
        Returns:
            out: Contextualized tokens (batch_size, num_features, d_model).
        """
        x = tokens
        if self.col_identity_embeddings is not None:
            x = x + self.col_identity_embeddings.unsqueeze(0)

        for layer in self.layers:
            x = layer(x)

        return self.final_norm(x)
