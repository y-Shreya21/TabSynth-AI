"""Unified Mixed-Type Tabular Diffusion Interface.

Coordinates independent Gaussian and Multinomial diffusion processes for heterogeneous tables
containing both continuous and discrete features.
"""

from typing import List, Tuple, Optional, Dict, Callable, Union, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F

from tabsynth.diffusion.numerical.gaussian_diffusion import GaussianDiffusion
from tabsynth.diffusion.categorical.categorical_diffusion import CategoricalDiffusion
from tabsynth.diffusion.utils import select_device, to_one_hot


class MixedTypeDiffusion(nn.Module):
    """Mixed-Type Tabular Diffusion Engine coordinating continuous and discrete features."""

    def __init__(
        self,
        num_numerical_features: int,
        categorical_cardinalities: Optional[List[int]] = None,
        numerical_timesteps: int = 100,
        categorical_timesteps: int = 100,
        numerical_schedule: Literal["cosine", "linear"] = "cosine",
        categorical_schedule: Literal["cosine", "linear"] = "cosine",
        lambda_num: float = 1.0,
        lambda_cat: float = 1.0,
        lambda_aux: float = 0.2,
        device: Optional[Union[str, torch.device]] = "auto",
        **schedule_kwargs,
    ):
        super().__init__()
        self.num_numerical_features = max(0, int(num_numerical_features))
        self.categorical_cardinalities = list(categorical_cardinalities or [])
        self.num_categorical_features = len(self.categorical_cardinalities)
        self.numerical_timesteps = numerical_timesteps
        self.categorical_timesteps = categorical_timesteps
        self.lambda_num = lambda_num
        self.lambda_cat = lambda_cat
        self.lambda_aux = lambda_aux
        self.device = select_device(device)

        # Initialize Continuous Gaussian Diffusion if numerical features exist
        if self.num_numerical_features > 0:
            self.numerical_diffusion = GaussianDiffusion(
                num_timesteps=numerical_timesteps,
                schedule_type=numerical_schedule,
                device=self.device,
                **schedule_kwargs,
            )
        else:
            self.numerical_diffusion = None

        # Initialize Discrete Multinomial Diffusion if categorical features exist
        if self.num_categorical_features > 0:
            self.categorical_diffusion = CategoricalDiffusion(
                cardinalities=self.categorical_cardinalities,
                num_timesteps=categorical_timesteps,
                schedule_type=categorical_schedule,
                device=self.device,
                **schedule_kwargs,
            )
        else:
            self.categorical_diffusion = None

    def to_device(self, device: Union[str, torch.device]) -> "MixedTypeDiffusion":
        """Transfers underlying diffusion modules to specified device."""
        self.device = select_device(device)
        if self.numerical_diffusion is not None:
            self.numerical_diffusion.to_device(self.device)
        if self.categorical_diffusion is not None:
            self.categorical_diffusion.to_device(self.device)
        return self

    def forward_numerical(
        self,
        x_num: torch.Tensor,
        timesteps: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Applies Gaussian continuous forward corruption to numerical features."""
        if self.numerical_diffusion is None or self.num_numerical_features == 0:
            empty = torch.empty((len(timesteps), 0), device=self.device, dtype=torch.float32)
            return empty, empty

        return self.numerical_diffusion.q_sample(x_start=x_num, timesteps=timesteps, noise=noise)

    def forward_categorical(
        self,
        x_cat_list: List[torch.Tensor],
        timesteps: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Applies D3PM discrete forward corruption to categorical features."""
        if self.categorical_diffusion is None or self.num_categorical_features == 0:
            return [], []

        return self.categorical_diffusion.q_sample_categorical(x_0_list=x_cat_list, timesteps=timesteps)

    def forward(
        self,
        x_num: Optional[torch.Tensor],
        x_cat_list: Optional[List[torch.Tensor]],
        t_num: Optional[torch.Tensor] = None,
        t_cat: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """Applies forward diffusion to both numerical and categorical inputs independently.
        
        Returns:
            (x_t_num, noise_num, x_t_cat_onehots, x_t_cat_indices)
        """
        # 1. Numerical forward process
        if self.num_numerical_features > 0 and x_num is not None:
            batch_size = x_num.shape[0]
            if t_num is None:
                t_num = torch.randint(0, self.numerical_timesteps, (batch_size,), device=self.device)
            x_t_num, noise_num = self.forward_numerical(x_num, t_num)
        else:
            batch_size = len(x_cat_list[0]) if x_cat_list and len(x_cat_list) > 0 else 0
            x_t_num = torch.empty((batch_size, 0), device=self.device, dtype=torch.float32)
            noise_num = torch.empty((batch_size, 0), device=self.device, dtype=torch.float32)

        # 2. Categorical forward process
        if self.num_categorical_features > 0 and x_cat_list is not None:
            if t_cat is None:
                t_cat = torch.randint(0, self.categorical_timesteps, (batch_size,), device=self.device)
            x_t_cat_onehots, x_t_cat_indices = self.forward_categorical(x_cat_list, t_cat)
        else:
            x_t_cat_onehots, x_t_cat_indices = [], []

        return x_t_num, noise_num, x_t_cat_onehots, x_t_cat_indices

    def compute_training_loss(
        self,
        pred_eps: Optional[torch.Tensor],
        target_noise: Optional[torch.Tensor],
        pred_logits_list: Optional[List[torch.Tensor]],
        clean_cat_list: Optional[List[torch.Tensor]],
        corrupted_cat_list: Optional[List[torch.Tensor]],
        t_cat: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Calculates joint mixed-type training losses.
        
        L_total = lambda_num * L_num + lambda_cat * L_cat + lambda_aux * L_aux
        """
        zero = torch.tensor(0.0, device=self.device)
        loss_num = zero
        loss_cat = zero
        loss_aux = zero

        # 1. Continuous Numerical Loss (MSE on predicted noise)
        if self.num_numerical_features > 0 and pred_eps is not None and target_noise is not None:
            loss_num = self.numerical_diffusion.loss(pred_eps, target_noise, reduction="mean")

        # 2. Discrete Categorical Loss (Variational KL + Auxiliary CE)
        if self.num_categorical_features > 0 and pred_logits_list is not None and clean_cat_list is not None:
            kl_losses = []
            aux_losses = []
            for j, K in enumerate(self.categorical_cardinalities):
                x_0 = clean_cat_list[j]
                x_0_oh = to_one_hot(x_0, K) if x_0.ndim == 1 else x_0
                x_t_oh = corrupted_cat_list[j] if corrupted_cat_list is not None else x_0_oh
                logits = pred_logits_list[j]

                kl_j = self.categorical_diffusion.compute_kl_divergence(
                    x_t_onehot=x_t_oh,
                    x_0_onehot=x_0_oh,
                    pred_x_0_logits=logits,
                    timesteps=t_cat,
                    K=K,
                )
                kl_losses.append(kl_j)

                if self.lambda_aux > 0:
                    aux_j = self.categorical_diffusion.auxiliary_cross_entropy_loss(
                        pred_x_0_logits=logits,
                        x_0=x_0_oh,
                    )
                    aux_losses.append(aux_j)

            loss_cat = torch.stack(kl_losses).mean() if kl_losses else zero
            loss_aux = torch.stack(aux_losses).mean() if aux_losses else zero

        loss_total = (
            self.lambda_num * loss_num
            + self.lambda_cat * loss_cat
            + self.lambda_aux * loss_aux
        )

        return {
            "loss_total": loss_total,
            "loss_num": loss_num,
            "loss_cat": loss_cat,
            "loss_aux": loss_aux,
        }

    @torch.no_grad()
    def sample(
        self,
        model_fn: Callable[
            [torch.Tensor, List[torch.Tensor], torch.Tensor, torch.Tensor],
            Tuple[torch.Tensor, List[torch.Tensor]]
        ],
        batch_size: int,
        device: Optional[Union[str, torch.device]] = None,
        clip_denoised: bool = True,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """Iterative joint reverse sampling for both numerical and categorical features.
        
        Args:
            model_fn: Callable taking (x_num, x_cat_list, t_num, t_cat) and returning (pred_eps, pred_logits_list).
            batch_size: Number of synthetic rows to generate.
            device: Target device.
            clip_denoised: Whether to clamp continuous predictions.
            
        Returns:
            (sampled_x_num, sampled_x_cat_onehots)
        """
        dev = select_device(device) if device is not None else self.device

        # Initialize Numerical state from standard Gaussian noise
        if self.num_numerical_features > 0:
            x_num = torch.randn((batch_size, self.num_numerical_features), device=dev, dtype=torch.float32)
        else:
            x_num = torch.empty((batch_size, 0), device=dev, dtype=torch.float32)

        # Initialize Categorical states from uniform discrete distribution
        x_cat_list: List[torch.Tensor] = []
        for K in self.categorical_cardinalities:
            uniform = torch.full((batch_size, K), 1.0 / float(K), device=dev, dtype=torch.float32)
            idx = torch.multinomial(uniform, num_samples=1).squeeze(-1)
            x_cat_list.append(to_one_hot(idx, num_classes=K))

        # Synchronized or maximum timestep loop
        max_steps = max(self.numerical_timesteps, self.categorical_timesteps)

        for step in reversed(range(max_steps)):
            # Map global step to continuous and discrete timestep indices
            t_num_val = min(step, self.numerical_timesteps - 1)
            t_cat_val = min(step, self.categorical_timesteps - 1)

            t_num = torch.full((batch_size,), t_num_val, device=dev, dtype=torch.long)
            t_cat = torch.full((batch_size,), t_cat_val, device=dev, dtype=torch.long)

            # Query the denoising network
            pred_eps, pred_logits_list = model_fn(x_num, x_cat_list, t_num, t_cat)

            # Continuous reverse step
            if self.num_numerical_features > 0 and step < self.numerical_timesteps:
                x_num = self.numerical_diffusion.p_sample_step(
                    predicted_eps=pred_eps,
                    x_t=x_num,
                    timesteps=t_num,
                    clip_denoised=clip_denoised,
                )

            # Discrete reverse step
            if self.num_categorical_features > 0 and step < self.categorical_timesteps:
                next_cat_list: List[torch.Tensor] = []
                for j, K in enumerate(self.categorical_cardinalities):
                    x_t_j = x_cat_list[j]
                    logits_j = pred_logits_list[j]

                    if t_cat_val == 0:
                        clean_idx = torch.argmax(logits_j, dim=-1)
                        x_prev_j = to_one_hot(clean_idx, num_classes=K)
                    else:
                        p_prev = self.categorical_diffusion.p_probs(
                            x_t_onehot=x_t_j,
                            pred_x_0_logits=logits_j,
                            timesteps=t_cat,
                            K=K,
                        )
                        samples = torch.multinomial(p_prev, num_samples=1).squeeze(-1)
                        x_prev_j = to_one_hot(samples, num_classes=K)

                    next_cat_list.append(x_prev_j)
                x_cat_list = next_cat_list

        return x_num, x_cat_list
