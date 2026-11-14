"""Validation evaluation loop for tabular diffusion models."""

from typing import Dict, Any, List, Optional, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tabsynth.diffusion.numerical.gaussian_diffusion import GaussianDiffusion
from tabsynth.diffusion.categorical.categorical_diffusion import CategoricalDiffusion
from tabsynth.diffusion.mixed.mixed_diffusion import MixedTypeDiffusion
from tabsynth.training.losses import TabSynthLoss
from tabsynth.training.metrics import MetricAccumulator


@torch.no_grad()
def evaluate_validation(
    model: nn.Module,
    val_loader: DataLoader,
    loss_fn: TabSynthLoss,
    gaussian_diffusion: Optional[GaussianDiffusion] = None,
    multinomial_diffusion: Optional[CategoricalDiffusion] = None,
    mixed_diffusion: Optional[MixedTypeDiffusion] = None,
    device: Optional[torch.device] = None,
    cardinalities: Optional[List[int]] = None,
    num_timesteps: int = 100,
) -> Dict[str, float]:
    """Runs a complete validation pass over held-out validation data.
    
    Args:
        model: Denoising neural network.
        val_loader: Validation DataLoader.
        loss_fn: TabSynthLoss instance.
        gaussian_diffusion: Optional Gaussian continuous diffusion.
        multinomial_diffusion: Optional Categorical discrete diffusion.
        mixed_diffusion: Optional MixedTypeDiffusion engine.
        device: Target compute device.
        cardinalities: Categorical cardinalities.
        num_timesteps: Number of diffusion timesteps.
        
    Returns:
        Dictionary of validation metrics with 'val_' prefix.
    """
    model.eval()
    target_device = device or next(model.parameters()).device
    accumulator = MetricAccumulator(prefix="val")

    steps = num_timesteps
    if gaussian_diffusion is not None:
        steps = gaussian_diffusion.num_timesteps
    elif mixed_diffusion is not None:
        steps = mixed_diffusion.numerical_timesteps
    elif multinomial_diffusion is not None:
        steps = multinomial_diffusion.num_timesteps

    for batch in val_loader:
        x_num_0, x_cat_0_list, labels = batch
        batch_size = x_num_0.shape[0] if x_num_0 is not None and x_num_0.ndim >= 2 and x_num_0.shape[0] > 0 else len(x_cat_0_list[0])

        x_num_0 = x_num_0.to(target_device)
        x_cat_0_list = [cat.to(target_device) for cat in x_cat_0_list]
        labels = labels.to(target_device)

        t = torch.randint(0, steps, (batch_size,), device=target_device)

        # Forward corruption
        if mixed_diffusion is not None:
            x_num_t, target_eps, x_cat_t_list, _ = mixed_diffusion.forward(
                x_num=x_num_0,
                x_cat_list=x_cat_0_list,
                t_num=t,
                t_cat=t,
            )
        else:
            if x_num_0.shape[1] > 0 and gaussian_diffusion is not None:
                if hasattr(gaussian_diffusion, "q_sample"):
                    x_num_t, target_eps = gaussian_diffusion.q_sample(x_start=x_num_0, timesteps=t) if "x_start" in gaussian_diffusion.q_sample.__code__.co_varnames else gaussian_diffusion.q_sample(x_0=x_num_0, t=t)
                else:
                    x_num_t = x_num_0
                    target_eps = torch.empty(batch_size, 0, device=target_device)
            else:
                x_num_t = x_num_0
                target_eps = torch.empty(batch_size, 0, device=target_device)

            x_cat_t_list = []
            if multinomial_diffusion is not None and cardinalities is not None:
                for j, K in enumerate(cardinalities):
                    x_0_j = x_cat_0_list[j]
                    if hasattr(multinomial_diffusion, "q_sample_single"):
                        x_t_j, _ = multinomial_diffusion.q_sample_single(x_0_onehot=x_0_j, timesteps=t, K=K)
                    elif hasattr(multinomial_diffusion, "q_sample"):
                        if "x_0_onehot" in multinomial_diffusion.q_sample.__code__.co_varnames:
                            x_t_j = multinomial_diffusion.q_sample(x_0_onehot=x_0_j, t=t, K=K)
                        else:
                            x_t_j = multinomial_diffusion.q_sample(x_0=x_0_j, timesteps=t)
                    else:
                        x_t_j = x_0_j
                    x_cat_t_list.append(x_t_j)
            else:
                x_cat_t_list = [cat for cat in x_cat_0_list]

        # Model forward
        output = model(
            x_num_t=x_num_t,
            x_cat_t=x_cat_t_list,
            timesteps_num=t,
            timesteps_cat=t,
            class_labels=labels,
            force_uncond=False,
        )

        if isinstance(output, tuple):
            pred_eps, pred_cat_logits = output
        elif isinstance(output, dict):
            pred_eps = output["predicted_noise"]
            pred_cat_logits = output["categorical_logits"]
        else:
            pred_eps = output.predicted_noise
            pred_cat_logits = output.categorical_logits

        _, metrics = loss_fn(
            pred_eps=pred_eps,
            target_eps=target_eps,
            pred_cat_logits=pred_cat_logits,
            x_cat_t_list=x_cat_t_list,
            x_cat_0_list=x_cat_0_list,
            t=t,
        )

        accumulator.update(metrics, batch_size=batch_size)

    accumulator.stop_timer()
    return accumulator.compute()
