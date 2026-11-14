"""Continuous Gaussian Denoising Diffusion Probabilistic Model (DDPM) for numerical features."""

from typing import Optional, Tuple
import torch
import torch.nn as nn
from tabsynth.diffusion.schedules import PrecomputedDiffusionConstants, get_noise_schedule


class GaussianDiffusion(nn.Module):
    """Gaussian diffusion engine for continuous numerical tabular features.
    
    Implements standard DDPM forward noise addition q(x_t | x_0) and reverse
    posterior sampling p_theta(x_{t-1} | x_t) according to Ho et al. (2020) and
    Nichol & Dhariwal (2021).
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        schedule_type: str = "cosine",
        device: Optional[torch.device] = None,
        **schedule_kwargs,
    ):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.schedule_type = schedule_type
        self.device = device or torch.device("cpu")

        betas = get_noise_schedule(schedule_type=schedule_type, timesteps=num_timesteps, **schedule_kwargs)
        self.constants = PrecomputedDiffusionConstants(betas=betas, device=self.device)

    def to_device(self, device: torch.device) -> "GaussianDiffusion":
        """Move diffusion buffers to the specified compute device."""
        self.device = device
        self.constants = PrecomputedDiffusionConstants(betas=self.constants.betas, device=device)
        return self

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion process q(x_t | x_0) = sqrt(alpha_bar_t)*x_0 + sqrt(1 - alpha_bar_t)*eps.
        
        Args:
            x_0: Clean continuous feature tensor of shape (batch_size, num_continuous).
            t: Discrete timesteps of shape (batch_size,) with values in [0, num_timesteps - 1].
            noise: Optional pre-sampled Gaussian noise matching x_0.
            
        Returns:
            x_t: Noisy corrupted feature tensor.
            noise: The Gaussian noise perturbation added.
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha_bar = self.constants.extract(self.constants.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alpha_bar = self.constants.extract(
            self.constants.sqrt_one_minus_alphas_cumprod, t, x_0.shape
        )

        x_t = sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise
        return x_t, noise

    def predict_x0_from_eps(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstruct x_0 estimate given noisy state x_t and predicted noise eps."""
        sqrt_alpha_bar = self.constants.extract(self.constants.sqrt_alphas_cumprod, t, x_t.shape)
        sqrt_one_minus_alpha_bar = self.constants.extract(
            self.constants.sqrt_one_minus_alphas_cumprod, t, x_t.shape
        )
        return (x_t - sqrt_one_minus_alpha_bar * eps) / torch.clamp(sqrt_alpha_bar, min=1e-8)

    def q_posterior_mean_variance(
        self,
        x_0: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the true posterior distribution q(x_{t-1} | x_t, x_0)."""
        posterior_mean = (
            self.constants.extract(self.constants.posterior_mean_coef1, t, x_t.shape) * x_0
            + self.constants.extract(self.constants.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = self.constants.extract(self.constants.posterior_variance, t, x_t.shape)
        posterior_log_variance = self.constants.extract(
            self.constants.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance

    def p_mean_variance(
        self,
        predicted_eps: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        clip_denoised: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute predicted mean and variance of reverse distribution p_theta(x_{t-1} | x_t)."""
        pred_x_0 = self.predict_x0_from_eps(x_t=x_t, t=t, eps=predicted_eps)
        if clip_denoised:
            pred_x_0 = torch.clamp(pred_x_0, -5.0, 5.0)

        mean, variance, log_variance = self.q_posterior_mean_variance(x_0=pred_x_0, x_t=x_t, t=t)
        return mean, variance, log_variance

    def p_sample(
        self,
        predicted_eps: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor,
        clip_denoised: bool = False,
    ) -> torch.Tensor:
        """Sample x_{t-1} ~ p_theta(x_{t-1} | x_t) given model noise prediction."""
        mean, _, log_variance = self.p_mean_variance(
            predicted_eps=predicted_eps,
            x_t=x_t,
            t=t,
            clip_denoised=clip_denoised,
        )

        noise = torch.randn_like(x_t)
        # No noise added when t == 0
        nonzero_mask = (t != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        return mean + nonzero_mask * torch.exp(0.5 * log_variance) * noise
