"""Learning rate schedulers and warmup policies for tabular diffusion training."""

from typing import Optional, Any
import math
import torch
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    StepLR,
    ConstantLR,
    SequentialLR,
    _LRScheduler,
)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "cosine",
    total_epochs: int = 100,
    warmup_epochs: int = 5,
    min_lr: float = 1e-6,
    step_size: int = 30,
    gamma: float = 0.5,
) -> Optional[_LRScheduler]:
    """Builds and returns a learning rate scheduler with optional warmup.
    
    Supported scheduler types:
        - 'cosine' or 'cosine_annealing': Cosine decay with optional initial linear warmup.
        - 'linear': Linear decay with optional warmup.
        - 'step': Step-based multiplicative decay.
        - 'constant': Constant learning rate.
        - 'none': Returns None.
    """
    stype = scheduler_type.lower() if scheduler_type else "none"

    if stype in ("none", "null", ""):
        return None

    if stype in ("cosine", "cosine_annealing", "cosine_with_warmup"):
        if warmup_epochs > 0 and warmup_epochs < total_epochs:
            # Linear warmup from start_factor=0.01 to 1.0 over warmup_epochs
            warmup = LinearLR(
                optimizer,
                start_factor=1e-2,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            cosine_epochs = max(1, total_epochs - warmup_epochs)
            cosine = CosineAnnealingLR(
                optimizer,
                T_max=cosine_epochs,
                eta_min=min_lr,
            )
            return SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_epochs],
            )
        else:
            return CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=min_lr)

    elif stype == "linear":
        if warmup_epochs > 0 and warmup_epochs < total_epochs:
            warmup = LinearLR(optimizer, start_factor=1e-2, end_factor=1.0, total_iters=warmup_epochs)
            decay = LinearLR(optimizer, start_factor=1.0, end_factor=min_lr / max(1e-9, optimizer.param_groups[0]["lr"]), total_iters=total_epochs - warmup_epochs)
            return SequentialLR(optimizer, schedulers=[warmup, decay], milestones=[warmup_epochs])
        else:
            return LinearLR(optimizer, start_factor=1.0, end_factor=1e-3, total_iters=total_epochs)

    elif stype == "step":
        return StepLR(optimizer, step_size=step_size, gamma=gamma)

    elif stype == "constant":
        return ConstantLR(optimizer, factor=1.0, total_iters=total_epochs)

    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}. Choose 'cosine', 'linear', 'step', or 'constant'.")


def get_current_lr(optimizer: torch.optim.Optimizer) -> float:
    """Returns the current learning rate from the first parameter group."""
    if not optimizer.param_groups:
        return 0.0
    return float(optimizer.param_groups[0].get("lr", 0.0))
