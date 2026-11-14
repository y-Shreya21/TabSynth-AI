"""Optimizer factory and parameter grouping for tabular diffusion architectures."""

from typing import Dict, Any, List, Tuple, Optional
import torch
import torch.nn as nn


def get_parameter_groups(
    model: nn.Module,
    weight_decay: float = 1e-4,
    no_decay_keywords: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """Separates parameters into decay and no-decay groups.
    
    1D tensors (e.g. biases, LayerNorm/BatchNorm scales and offsets, embedding weights)
    receive weight_decay=0.0 to prevent destructive shrinkage of normalization statistics
    and representation biases. All other parameters (>=2D projection/attention weights)
    receive the specified weight decay.
    """
    if no_decay_keywords is None:
        no_decay_keywords = ["bias", "norm", "ln", "layer_norm", "batch_norm", "embedding", "identity"]

    decay_params: List[nn.Parameter] = []
    no_decay_params: List[nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        # 1D tensors (bias, LayerNorm scale/shift, etc.) or matching keywords
        if param.ndim <= 1 or any(kw in name.lower() for kw in no_decay_keywords):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return groups


def build_optimizer(
    model: nn.Module,
    optimizer_type: str = "adamw",
    learning_rate: float = 2e-4,
    weight_decay: float = 1e-4,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    momentum: float = 0.9,
    use_param_groups: bool = True,
) -> torch.optim.Optimizer:
    """Builds and configures optimizer with optional parameter grouping."""
    opt_type = optimizer_type.lower()

    if use_param_groups and weight_decay > 0.0:
        param_groups = get_parameter_groups(model, weight_decay=weight_decay)
        # Pass base lr; weight_decay is specified per group
        if opt_type in ("adamw", "adam_w"):
            return torch.optim.AdamW(param_groups, lr=learning_rate, betas=betas, eps=eps)
        elif opt_type == "adam":
            return torch.optim.Adam(param_groups, lr=learning_rate, betas=betas, eps=eps)
        elif opt_type == "sgd":
            return torch.optim.SGD(param_groups, lr=learning_rate, momentum=momentum)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}. Choose 'adamw', 'adam', or 'sgd'.")
    else:
        # Standard flat parameter list
        params = [p for p in model.parameters() if p.requires_grad]
        if opt_type in ("adamw", "adam_w"):
            return torch.optim.AdamW(params, lr=learning_rate, weight_decay=weight_decay, betas=betas, eps=eps)
        elif opt_type == "adam":
            return torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay, betas=betas, eps=eps)
        elif opt_type == "sgd":
            return torch.optim.SGD(params, lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
        else:
            raise ValueError(f"Unsupported optimizer type: {optimizer_type}. Choose 'adamw', 'adam', or 'sgd'.")
