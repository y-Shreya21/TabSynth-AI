"""Timestep and class conditioning embedding modules for diffusion models."""

from typing import List, Optional, Tuple, Union
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# Re-export tokenizers for backward compatibility
from tabsynth.models.tokenizer import (
    NumericalFeatureTokenizer as NumericalTokenizer,
    CategoricalFeatureTokenizer as CategoricalTokenizer,
    FeatureTokenizer,
)


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional timestep embedding followed by a 2-layer trainable MLP."""

    def __init__(self, d_model: int, max_period: float = 10000.0):
        super().__init__()
        assert d_model >= 2, f"d_model must be >= 2, got {d_model}"
        self.d_model = d_model
        half_dim = d_model // 2
        emb_scale = math.log(max_period) / max(1, half_dim - 1)
        self.register_buffer("freqs", torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb_scale))

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Args:
            t: Timestep indices of shape (batch_size,).
            
        Returns:
            time_emb: (batch_size, d_model).
        """
        assert t.ndim == 1, f"Timestep tensor must be 1D (B,), got shape {t.shape}"
        args = t.float().unsqueeze(-1) * self.freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if emb.shape[-1] < self.d_model:
            emb = F.pad(emb, (0, self.d_model - emb.shape[-1]))
        return self.mlp(emb)


class MixedTimestepEmbedding(nn.Module):
    """Fuses independent continuous numerical and discrete categorical timesteps into a joint representation.
    
    If both t_num and t_cat are provided, embeds both sinusoidally and fuses them via a projection layer.
    If only one timestep is provided, routes through its dedicated embedding.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.time_embed_num = SinusoidalTimeEmbedding(d_model=d_model)
        self.time_embed_cat = SinusoidalTimeEmbedding(d_model=d_model)
        self.fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(
        self,
        t_num: Optional[torch.Tensor] = None,
        t_cat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Args:
            t_num: 1D continuous timesteps (B,) or None.
            t_cat: 1D discrete timesteps (B,) or None.
            
        Returns:
            fused_time_emb: (B, d_model).
        """
        assert t_num is not None or t_cat is not None, "At least one of t_num or t_cat must be provided"

        if t_num is not None and t_cat is not None:
            e_num = self.time_embed_num(t_num)
            e_cat = self.time_embed_cat(t_cat)
            fused = torch.cat([e_num, e_cat], dim=-1)
            return self.fusion(fused)
        elif t_num is not None:
            return self.time_embed_num(t_num)
        else:
            return self.time_embed_cat(t_cat)


class ClassConditioningEmbedding(nn.Module):
    """Class label embedding table supporting classifier-free guidance label dropout."""

    def __init__(self, num_classes: int, d_model: int, p_uncond: float = 0.15):
        super().__init__()
        assert num_classes >= 1, f"num_classes must be >= 1, got {num_classes}"
        assert 0.0 <= p_uncond <= 1.0, f"p_uncond must be in [0, 1], got {p_uncond}"
        self.num_classes = num_classes
        self.d_model = d_model
        self.p_uncond = p_uncond

        # Class index `num_classes` is reserved for the unconditional null token
        self.embedding = nn.Embedding(num_classes + 1, d_model)
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(
        self,
        c: Optional[torch.Tensor],
        batch_size: int,
        force_uncond: bool = False,
    ) -> torch.Tensor:
        """Args:
            c: Class index tensor of shape (batch_size,). If None, null token is used.
            batch_size: Fallback batch size if c is None.
            force_uncond: If True, forces all conditioning tokens to null token.
            
        Returns:
            class_emb: (batch_size, d_model).
        """
        device = self.embedding.weight.device
        null_token = self.num_classes

        if c is None or force_uncond:
            indices = torch.full((batch_size,), null_token, dtype=torch.long, device=device)
            return self.embedding(indices)

        c = c.to(device=device, dtype=torch.long)
        if self.training and self.p_uncond > 0.0:
            mask = torch.rand(batch_size, device=device) < self.p_uncond
            c = torch.where(mask, torch.full_like(c, null_token), c)

        return self.embedding(c)
