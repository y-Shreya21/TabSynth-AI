"""Configurable noise variance schedules for continuous and discrete diffusion processes.

Explicitly 0-indexed: timesteps t in {0, 1, ..., T - 1}.
For t = 0:
    alpha_0 = 1 - beta_0
    alpha_bar_0 = alpha_0
    alpha_bar_{prev} = alpha_bar_{-1} = 1.0
"""

from typing import Tuple, Literal, Optional, Dict
import math
import torch


def linear_beta_schedule(
    timesteps: int,
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
) -> torch.Tensor:
    """Standard linear noise schedule as in Ho et al. (2020).
    
    Linearly interpolates beta from beta_start to beta_end over T steps.
    
    Args:
        timesteps: Total diffusion steps T >= 1.
        beta_start: Starting variance beta_0.
        beta_end: Ending variance beta_{T-1}.
        
    Returns:
        betas: 1D float32 tensor of length T with 0 < beta_t < 1.
    """
    assert timesteps >= 1, f"timesteps must be >= 1, got {timesteps}"
    assert 0.0 < beta_start < 1.0, f"beta_start must be in (0, 1), got {beta_start}"
    assert 0.0 < beta_end < 1.0, f"beta_end must be in (0, 1), got {beta_end}"
    assert beta_start <= beta_end, f"beta_start ({beta_start}) must be <= beta_end ({beta_end})"
    
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)


def cosine_beta_schedule(
    timesteps: int,
    s: float = 0.008,
    max_beta: float = 0.999,
) -> torch.Tensor:
    """Cosine noise schedule as proposed in Nichol & Dhariwal (2021).
    
    alpha_bar_t = f(t) / f(0) where f(t) = cos(((t/T + s) / (1 + s)) * (pi / 2))^2.
    beta_t = 1 - alpha_bar_t / alpha_bar_{t-1}, clipped to max_beta.
    
    Args:
        timesteps: Total diffusion steps T >= 1.
        s: Small offset to prevent beta_t from being too small near t = 0.
        max_beta: Upper clip bound for beta_t (prevents singularities).
        
    Returns:
        betas: 1D float32 tensor of length T with 0 < beta_t < 1.
    """
    assert timesteps >= 1, f"timesteps must be >= 1, got {timesteps}"
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float32)
    alphas_cumprod = torch.cos(((t / timesteps) + s) / (1.0 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, min=1e-5, max=max_beta)


def get_noise_schedule(
    schedule_type: Literal["cosine", "linear"] = "cosine",
    timesteps: int = 100,
    **kwargs,
) -> torch.Tensor:
    """Factory function for noise schedules.
    
    Args:
        schedule_type: "cosine" or "linear".
        timesteps: Total diffusion steps T.
        **kwargs: Optional schedule-specific arguments (e.g. beta_start, beta_end, s).
        
    Returns:
        betas: 1D float32 tensor of length T with 0 < beta_t < 1.
    """
    if schedule_type == "cosine":
        s = kwargs.get("s", kwargs.get("cosine_s", 0.008))
        max_beta = kwargs.get("max_beta", 0.999)
        return cosine_beta_schedule(timesteps=timesteps, s=s, max_beta=max_beta)
    elif schedule_type == "linear":
        beta_start = kwargs.get("beta_start", 0.0001)
        beta_end = kwargs.get("beta_end", 0.02)
        return linear_beta_schedule(timesteps=timesteps, beta_start=beta_start, beta_end=beta_end)
    else:
        raise ValueError(f"Unknown schedule_type: '{schedule_type}'. Choose 'cosine' or 'linear'.")


class PrecomputedDiffusionConstants:
    """Precomputes and stores all analytical diffusion constants for fast lookup.
    
    All constants are cached as float32 tensors on the specified device.
    Zero-indexed: t in {0, 1, ..., num_timesteps - 1}.
    
    Mathematical definitions:
        alpha_t = 1 - beta_t
        alpha_bar_t = prod_{s=0}^t alpha_s
        alpha_bar_{prev} = [1.0, alpha_bar_0, alpha_bar_1, ..., alpha_bar_{T-2}]
        posterior_variance = beta_t * (1 - alpha_bar_{prev}) / (1 - alpha_bar_t)
        posterior_mean_coef1 = beta_t * sqrt(alpha_bar_{prev}) / (1 - alpha_bar_t)
        posterior_mean_coef2 = (1 - alpha_bar_{prev}) * sqrt(alpha_t) / (1 - alpha_bar_t)
    """

    def __init__(self, betas: torch.Tensor, device: Optional[torch.device] = None):
        dev = device or betas.device
        self.device = dev
        self.betas = betas.to(device=dev, dtype=torch.float32)
        self.num_timesteps = int(len(self.betas))
        
        # Verify valid beta bounds: 0 < beta_t < 1
        assert (self.betas > 0.0).all(), "All betas must be strictly positive (> 0)."
        assert (self.betas < 1.0).all(), "All betas must be strictly less than 1 (< 1)."
        
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        
        # Verify monotonic decrease: alpha_bar_t < alpha_bar_{t-1}
        diffs = self.alphas_cumprod[1:] - self.alphas_cumprod[:-1]
        assert (diffs <= 0.0).all(), "alphas_cumprod must be monotonically non-increasing."

        # alpha_bar_{t-1} with alpha_bar_{-1} = 1.0
        self.alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], device=dev, dtype=torch.float32), self.alphas_cumprod[:-1]]
        )

        # Forward process coefficients: q(x_t | x_0) = sqrt(alpha_bar_t)*x_0 + sqrt(1 - alpha_bar_t)*eps
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(torch.clamp(1.0 - self.alphas_cumprod, min=1e-20))

        # Reverse posterior coefficients: q(x_{t-1} | x_t, x_0)
        # posterior_variance = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
        denom = torch.clamp(1.0 - self.alphas_cumprod, min=1e-20)
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / denom
        
        # Log variance clipped at t = 0 to avoid log(0)
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        
        # Posterior mean: coef1 * x_0 + coef2 * x_t
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / denom
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / denom

        # Ensure no NaNs or Infs in any precomputed buffers
        for name, buf in self.get_all_buffers().items():
            assert not torch.isnan(buf).any(), f"NaN detected in precomputed buffer: {name}"
            assert not torch.isinf(buf).any(), f"Inf detected in precomputed buffer: {name}"

    def get_all_buffers(self) -> Dict[str, torch.Tensor]:
        """Returns dictionary of all scalar schedule tensors."""
        return {
            "betas": self.betas,
            "alphas": self.alphas,
            "alphas_cumprod": self.alphas_cumprod,
            "alphas_cumprod_prev": self.alphas_cumprod_prev,
            "sqrt_alphas_cumprod": self.sqrt_alphas_cumprod,
            "sqrt_one_minus_alphas_cumprod": self.sqrt_one_minus_alphas_cumprod,
            "log_one_minus_alphas_cumprod": self.log_one_minus_alphas_cumprod,
            "posterior_variance": self.posterior_variance,
            "posterior_log_variance_clipped": self.posterior_log_variance_clipped,
            "posterior_mean_coef1": self.posterior_mean_coef1,
            "posterior_mean_coef2": self.posterior_mean_coef2,
        }

    def extract(self, tensor: torch.Tensor, t: torch.Tensor, broadcast_shape: Tuple[int, ...]) -> torch.Tensor:
        """Extract coefficients at timesteps t and reshape for broadcasting to broadcast_shape.
        
        Args:
            tensor: 1D buffer tensor of length T.
            t: 1D integer timestep tensor of shape (batch_size,).
            broadcast_shape: Target tensor shape, e.g. (batch_size, num_features).
            
        Returns:
            out: Broadcastable tensor of shape (batch_size, 1, 1, ...).
        """
        assert t.ndim == 1, f"t must be 1D, got ndim={t.ndim}"
        out = tensor.gather(0, t.to(self.device))
        reshape_dims = (out.shape[0],) + (1,) * (len(broadcast_shape) - 1)
        return out.reshape(reshape_dims)
