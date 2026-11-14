"""Timestep embedding modules for diffusion models."""

import math
import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for continuous or discrete diffusion timesteps.
    
    Maps scalar timesteps t in [0, T-1] to d-dimensional vectors using sine and cosine frequencies:
        PE(t, 2i)   = sin(t / 10000^(2i/d))
        PE(t, 2i+1) = cos(t / 10000^(2i/d))
    """

    def __init__(self, embed_dim: int, max_period: float = 10000.0):
        super().__init__()
        assert embed_dim % 2 == 0, f"embed_dim must be divisible by 2, got {embed_dim}"
        self.embed_dim = embed_dim
        self.max_period = max_period

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Args:
            timesteps: 1D tensor of timesteps of shape (B,).
            
        Returns:
            embedding: 2D tensor of shape (B, embed_dim).
        """
        assert timesteps.ndim == 1, f"timesteps must be 1D, got {timesteps.ndim}D"
        half_dim = self.embed_dim // 2
        device = timesteps.device

        freqs = torch.exp(
            -math.log(self.max_period)
            * torch.arange(start=0, end=half_dim, dtype=torch.float32, device=device)
            / half_dim
        )
        args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
        embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return embedding


class TimestepMLP(nn.Module):
    """Embeds timesteps sinusoidally and projects through a 2-layer MLP with non-linearity."""

    def __init__(self, in_features: int, out_features: int, activation: str = "silu"):
        super().__init__()
        self.sinusoidal = SinusoidalTimeEmbedding(embed_dim=in_features)
        act_layer = nn.SiLU() if activation.lower() == "silu" else nn.GELU()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, out_features),
            act_layer,
            nn.Linear(out_features, out_features),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Args:
            timesteps: 1D tensor of timesteps (B,).
            
        Returns:
            projected: 2D tensor of shape (B, out_features).
        """
        embed = self.sinusoidal(timesteps)
        return self.mlp(embed)
