"""Iterative reverse trajectory numerical sampler for continuous features."""

from typing import Callable, Optional, Tuple, List, Union
import torch
from tabsynth.diffusion.numerical.gaussian_diffusion import GaussianDiffusion
from tabsynth.diffusion.utils import select_device


class NumericalSampler:
    """Iterative ancestral reverse sampler for continuous numerical features.
    
    Decoupled from specific network architectures: accepts any callable
    `model_fn(x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor`
    predicting Gaussian noise epsilon.
    """

    def __init__(self, diffusion: GaussianDiffusion):
        self.diffusion = diffusion
        self.num_timesteps = diffusion.num_timesteps

    @torch.no_grad()
    def sample(
        self,
        model_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        shape: Tuple[int, int],
        device: Optional[Union[str, torch.device]] = None,
        clip_denoised: bool = False,
        clip_range: Tuple[float, float] = (-5.0, 5.0),
        return_all_steps: bool = False,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """Samples numerical features via iterative DDPM ancestral denoising.
        
        Args:
            model_fn: Callable taking (x_t, t) and returning predicted noise eps (B, D).
            shape: (batch_size, num_continuous_features).
            device: Target device (defaults to diffusion.device).
            clip_denoised: Whether to clamp predicted x_0 during reverse steps.
            clip_range: Bounds for clamping if clip_denoised is True.
            return_all_steps: If True, returns list of states across all timesteps.
            
        Returns:
            Generated continuous tensor of shape (batch_size, num_continuous_features),
            or list of intermediate states if return_all_steps is True.
        """
        dev = select_device(device) if device is not None else self.diffusion.device
        batch_size, num_features = shape

        # Initial pure Gaussian noise: x_{T-1} ~ N(0, I)
        img = torch.randn(shape, device=dev, dtype=torch.float32)
        all_steps = [img] if return_all_steps else None

        # Reverse loop: T-1 down to 0
        for t_idx in reversed(range(self.num_timesteps)):
            t_batch = torch.full((batch_size,), t_idx, device=dev, dtype=torch.long)
            predicted_eps = model_fn(img, t_batch)
            img = self.diffusion.p_sample_step(
                predicted_eps=predicted_eps,
                x_t=img,
                timesteps=t_batch,
                clip_denoised=clip_denoised,
                clip_range=clip_range,
            )
            if return_all_steps:
                all_steps.append(img)

        return all_steps if return_all_steps else img
