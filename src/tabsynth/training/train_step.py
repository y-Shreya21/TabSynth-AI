"""Single training step execution for mixed-type tabular diffusion models."""

from typing import Dict, Any, List, Optional, Tuple, Union
import torch
import torch.nn as nn
from tabsynth.diffusion.numerical.gaussian_diffusion import GaussianDiffusion
from tabsynth.diffusion.categorical.categorical_diffusion import CategoricalDiffusion
from tabsynth.diffusion.mixed.mixed_diffusion import MixedTypeDiffusion
from tabsynth.training.losses import TabSynthLoss


def execute_train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch: Tuple[torch.Tensor, List[torch.Tensor], torch.Tensor],
    loss_fn: TabSynthLoss,
    gaussian_diffusion: Optional[GaussianDiffusion] = None,
    multinomial_diffusion: Optional[CategoricalDiffusion] = None,
    mixed_diffusion: Optional[MixedTypeDiffusion] = None,
    device: Optional[torch.device] = None,
    max_grad_norm: float = 1.0,
    cardinalities: Optional[List[int]] = None,
    num_timesteps: int = 100,
) -> Dict[str, float]:
    """Executes a single forward-backward-update training step on a mini-batch.
    
    Args:
        model: Denoising neural network.
        optimizer: Configured PyTorch optimizer.
        batch: Tuple of (x_num_0, x_cat_0_list, labels).
        loss_fn: TabSynthLoss instance.
        gaussian_diffusion: Optional continuous Gaussian diffusion module.
        multinomial_diffusion: Optional discrete Categorical diffusion module.
        mixed_diffusion: Optional unified mixed-type diffusion engine.
        device: Target compute device.
        max_grad_norm: Maximum gradient norm for clipping.
        cardinalities: Categorical cardinalities.
        num_timesteps: Number of diffusion timesteps.
        
    Returns:
        Dictionary of loss terms and step metrics (including grad_norm).
    """
    model.train()
    target_device = device or next(model.parameters()).device

    x_num_0, x_cat_0_list, labels = batch
    batch_size = x_num_0.shape[0] if x_num_0 is not None and x_num_0.ndim >= 2 and x_num_0.shape[0] > 0 else len(x_cat_0_list[0])

    x_num_0 = x_num_0.to(target_device)
    x_cat_0_list = [cat.to(target_device) for cat in x_cat_0_list]
    labels = labels.to(target_device)

    # 1. Sample Diffusion Timesteps uniformly in [0, T-1]
    steps = num_timesteps
    if gaussian_diffusion is not None:
        steps = gaussian_diffusion.num_timesteps
    elif mixed_diffusion is not None:
        steps = mixed_diffusion.numerical_timesteps
    elif multinomial_diffusion is not None:
        steps = multinomial_diffusion.num_timesteps

    t = torch.randint(0, steps, (batch_size,), device=target_device)

    # 2. Forward Diffusion Corruption
    if mixed_diffusion is not None:
        x_num_t, target_eps, x_cat_t_list, _ = mixed_diffusion.forward(
            x_num=x_num_0,
            x_cat_list=x_cat_0_list,
            t_num=t,
            t_cat=t,
        )
    else:
        # Continuous Gaussian forward process
        if x_num_0.shape[1] > 0 and gaussian_diffusion is not None:
            if hasattr(gaussian_diffusion, "q_sample"):
                x_num_t, target_eps = gaussian_diffusion.q_sample(x_start=x_num_0, timesteps=t) if "x_start" in gaussian_diffusion.q_sample.__code__.co_varnames else gaussian_diffusion.q_sample(x_0=x_num_0, t=t)
            else:
                x_num_t = x_num_0
                target_eps = torch.empty(batch_size, 0, device=target_device)
        else:
            x_num_t = x_num_0
            target_eps = torch.empty(batch_size, 0, device=target_device)

        # Discrete D3PM Categorical forward process
        x_cat_t_list = []
        if multinomial_diffusion is not None and cardinalities is not None:
            for j, K in enumerate(cardinalities):
                x_0_j = x_cat_0_list[j]
                if hasattr(multinomial_diffusion, "q_sample_single"):
                    x_t_j, _ = multinomial_diffusion.q_sample_single(x_0_onehot=x_0_j, timesteps=t, K=K)
                elif hasattr(multinomial_diffusion, "q_sample"):
                    # Check q_sample signature
                    if "x_0_onehot" in multinomial_diffusion.q_sample.__code__.co_varnames:
                        x_t_j = multinomial_diffusion.q_sample(x_0_onehot=x_0_j, t=t, K=K)
                    else:
                        x_t_j = multinomial_diffusion.q_sample(x_0=x_0_j, timesteps=t)
                else:
                    x_t_j = x_0_j
                x_cat_t_list.append(x_t_j)
        else:
            x_cat_t_list = [cat for cat in x_cat_0_list]

    # 3. Model Forward Denoising Pass
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

    # 4. Multi-Objective Loss Computation
    loss, metrics = loss_fn(
        pred_eps=pred_eps,
        target_eps=target_eps,
        pred_cat_logits=pred_cat_logits,
        x_cat_t_list=x_cat_t_list,
        x_cat_0_list=x_cat_0_list,
        t=t,
    )

    # 5. Backpropagation and Optimization Step
    optimizer.zero_grad(set_to_none=True)
    loss.backward()

    # Gradient clipping
    grad_norm = 0.0
    if max_grad_norm > 0:
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm).item())

    optimizer.step()

    metrics["grad_norm"] = grad_norm
    return metrics
