"""Model diagnostics, parameter profiling, and architectural inspection utilities."""

from typing import Dict, Any
import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Calculates total and trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable
    return {
        "total": total,
        "trainable": trainable,
        "non_trainable": non_trainable,
    }


def get_model_summary(model: nn.Module, model_name: str = "Model") -> Dict[str, Any]:
    """Inspects model dimensions, parameters, device, and dtype."""
    params = count_parameters(model)
    first_param = next(model.parameters(), None)
    device = str(first_param.device) if first_param is not None else "unknown"
    dtype = str(first_param.dtype) if first_param is not None else "unknown"

    submodule_counts = {}
    for name, child in model.named_children():
        submodule_counts[name] = sum(p.numel() for p in child.parameters())

    return {
        "model_name": model_name,
        "device": device,
        "dtype": dtype,
        "total_parameters": params["total"],
        "trainable_parameters": params["trainable"],
        "submodule_parameters": submodule_counts,
    }


def compare_models(hybrid_model: nn.Module, mlp_model: nn.Module) -> Dict[str, Any]:
    """Compares parameter counts and architectural components between Hybrid and MLP baseline."""
    hybrid_summary = get_model_summary(hybrid_model, "TabSynthHybridDenoiser")
    mlp_summary = get_model_summary(mlp_model, "TabSynthMLPDenoiser")

    diff_params = hybrid_summary["total_parameters"] - mlp_summary["total_parameters"]
    pct_diff = (diff_params / max(1, mlp_summary["total_parameters"])) * 100.0

    return {
        "hybrid": hybrid_summary,
        "mlp_baseline": mlp_summary,
        "parameter_delta": diff_params,
        "percentage_difference": pct_diff,
    }
