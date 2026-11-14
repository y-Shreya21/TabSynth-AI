"""Deterministic seed initialization across hardware and software backends."""

import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42, deterministic: bool = False) -> int:
    """Sets deterministic seed across Python random, NumPy, PyTorch, and MPS/CUDA.
    
    Args:
        seed: Random seed value.
        deterministic: Whether to enforce deterministic algorithm execution.
        
    Returns:
        The integer seed used.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        if hasattr(torch.mps, "manual_seed"):
            torch.mps.manual_seed(seed)

    if deterministic:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            # Some operations on MPS or CPU might not have deterministic implementations
            pass

    return seed
