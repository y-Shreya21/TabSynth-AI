"""Continuous Gaussian Denoising Diffusion Probabilistic Model (DDPM) for numerical features.

Mathematical formulation:
    Forward process:
        q(x_t | x_0) = Normal(sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
    
    Reverse posterior:
        q(x_{t-1} | x_t, x_0) = Normal(tilde_mu_t(x_t, x_0), tilde_beta_t * I)
        where:
            tilde_mu_t = [sqrt(alpha_bar_{t-1}) * beta_t / (1 - alpha_bar_t)] * x_0
                       + [sqrt(alpha_t) * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)] * x_t
            tilde_beta_t = beta_t * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)
"""

from typing import Optional, Tuple, Literal, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from tabsynth.diffusion.schedules import PrecomputedDiffusionConstants, get_noise_schedule
from tabsynth.diffusion.utils import select_device, extract_into_tensor


class GaussianDiffusion(nn.Module):
    """Gaussian DDPM engine for continuous numerical tabular features."""

    def __init__(
        self,
        num_timesteps: int = 100,
        schedule_type: Literal["cosine", "linear"] = "cosine",
        device: Optional[Union[str, torch.device]] = "auto",
        **schedule_kwargs,
    ):
        super().__init__()
        self.num_timesteps = num_timesteps
        self.schedule_type = schedule_type
        self.device = select_device(device)

        betas = get_noise_schedule(
            schedule_type=schedule_type,
            timesteps=num_timesteps,
            **schedule_kwargs,
        )
        self.constants = PrecomputedDiffusionConstants(betas=betas, device=self.device)

    def to_device(self, device: Union[str, torch.device]) -> "GaussianDiffusion":
        """Moves diffusion schedule buffers to the specified compute device."""
        self.device = select_device(device)
        self.constants = PrecomputedDiffusionConstants(
            betas=self.constants.betas, device=self.device
        )
        return self

    def q_sample(
        self,
        x_start: torch.Tensor,
        timesteps: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion process: adds Gaussian noise according to q(x_t | x_0).
        
        q(x_t | x_0) = sqrt(alpha_bar_t) * x_start + sqrt(1 - alpha_bar_t) * noise
        
        Args:
            x_start: Clean continuous tensor of shape (B, D).
            timesteps: 1D tensor of integer timesteps in [0, T-1] of shape (B,).
            noise: Optional Gaussian noise of shape (B, D). If None, sampled from N(0, I).
            
        Returns:
            x_t: Noisy state tensor of shape (B, D).
            noise: The noise perturbation tensor added of shape (B, D).
        """
        assert x_start.ndim == 2, f"x_start must be 2D (B, D), got {x_start.shape}"
        assert timesteps.ndim == 1, f"timesteps must be 1D (B,), got {timesteps.shape}"
        assert x_start.shape[0] == timesteps.shape[0], "Batch size of x_start and timesteps must match."
        assert (timesteps >= 0).all() and (timesteps < self.num_timesteps).all(), (
            f"Timesteps must be within [0, {self.num_timesteps - 1}]."
        )

        if noise is None:
            noise = torch.randn_like(x_start)
        else:
            assert noise.shape == x_start.shape, (
                f"Noise shape {noise.shape} must match x_start {x_start.shape}."
            )

        sqrt_alpha_bar = extract_into_tensor(
            self.constants.sqrt_alphas_cumprod, timesteps, x_start.shape
        )
        sqrt_one_minus_alpha_bar = extract_into_tensor(
            self.constants.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape
        )

        x_t = sqrt_alpha_bar * x_start + sqrt_one_minus_alpha_bar * noise
        return x_t, noise

    def training_target(
        self,
        x_start: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Returns the training target for epsilon-prediction parameterization.
        
        Args:
            x_start: Clean feature tensor (B, D).
            noise: Sampled Gaussian noise (B, D).
            
        Returns:
            target: The ground-truth noise epsilon to be predicted.
        """
        assert x_start.shape == noise.shape, "x_start and noise shapes must match."
        return noise

    def loss(
        self,
        predicted_noise: torch.Tensor,
        actual_noise: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ) -> torch.Tensor:
        """Calculates Mean Squared Error (MSE) loss between predicted and actual noise.
        
        Args:
            predicted_noise: Model output tensor of shape (B, D).
            actual_noise: Ground truth noise tensor of shape (B, D).
            weights: Optional per-feature or per-sample weighting tensor broadcastable to (B, D).
            reduction: "mean", "sum", or "none".
            
        Returns:
            Computed scalar loss (if mean/sum) or per-element tensor (if none).
        """
        assert predicted_noise.shape == actual_noise.shape, (
            f"predicted_noise {predicted_noise.shape} and actual_noise {actual_noise.shape} mismatch."
        )
        diff_sq = (predicted_noise - actual_noise) ** 2
        if weights is not None:
            diff_sq = diff_sq * weights.to(diff_sq.device)

        if reduction == "mean":
            return diff_sq.mean()
        elif reduction == "sum":
            return diff_sq.sum()
        elif reduction == "none":
            return diff_sq
        else:
            raise ValueError(f"Unknown reduction: {reduction}. Choose 'mean', 'sum', or 'none'.")

    def predict_x0_from_eps(
        self,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        eps: torch.Tensor,
    ) -> torch.Tensor:
        """Reconstructs estimated clean state x_0 given noisy state x_t and predicted noise eps:
        
        hat_x_0 = (x_t - sqrt(1 - alpha_bar_t) * eps) / sqrt(alpha_bar_t)
        """
        sqrt_alpha_bar = extract_into_tensor(
            self.constants.sqrt_alphas_cumprod, timesteps, x_t.shape
        )
        sqrt_one_minus_alpha_bar = extract_into_tensor(
            self.constants.sqrt_one_minus_alphas_cumprod, timesteps, x_t.shape
        )
        return (x_t - sqrt_one_minus_alpha_bar * eps) / torch.clamp(sqrt_alpha_bar, min=1e-8)

    def q_posterior_mean_variance(
        self,
        x_start: torch.Tensor,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculates exact analytical posterior distribution q(x_{t-1} | x_t, x_start).
        
        Returns:
            posterior_mean: Mean vector of shape (B, D).
            posterior_variance: Variance vector of shape (B, 1).
            posterior_log_variance: Log variance vector of shape (B, 1).
        """
        posterior_mean = (
            extract_into_tensor(self.constants.posterior_mean_coef1, timesteps, x_t.shape) * x_start
            + extract_into_tensor(self.constants.posterior_mean_coef2, timesteps, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(
            self.constants.posterior_variance, timesteps, x_t.shape
        )
        posterior_log_variance = extract_into_tensor(
            self.constants.posterior_log_variance_clipped, timesteps, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance

    def p_mean_variance(
        self,
        predicted_eps: torch.Tensor,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        clip_denoised: bool = False,
        clip_range: Tuple[float, float] = (-5.0, 5.0),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Calculates mean and variance of model reverse distribution p_theta(x_{t-1} | x_t)."""
        pred_x_start = self.predict_x0_from_eps(x_t=x_t, timesteps=timesteps, eps=predicted_eps)
        if clip_denoised:
            pred_x_start = torch.clamp(pred_x_start, clip_range[0], clip_range[1])

        mean, variance, log_variance = self.q_posterior_mean_variance(
            x_start=pred_x_start, x_t=x_t, timesteps=timesteps
        )
        return mean, variance, log_variance

    def p_sample_step(
        self,
        predicted_eps: torch.Tensor,
        x_t: torch.Tensor,
        timesteps: torch.Tensor,
        clip_denoised: bool = False,
        clip_range: Tuple[float, float] = (-5.0, 5.0),
    ) -> torch.Tensor:
        """Executes a single DDPM reverse sampling step: x_{t-1} ~ p_theta(x_{t-1} | x_t).
        
        Suppresses stochastic noise injection when t == 0.
        """
        mean, _, log_variance = self.p_mean_variance(
            predicted_eps=predicted_eps,
            x_t=x_t,
            timesteps=timesteps,
            clip_denoised=clip_denoised,
            clip_range=clip_range,
        )
        noise = torch.randn_like(x_t)
        # Suppress noise when timesteps == 0
        nonzero_mask = (timesteps != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        return mean + nonzero_mask * torch.exp(0.5 * log_variance) * noise
