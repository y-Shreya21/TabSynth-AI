"""Shared utility functions for numerical, categorical, and mixed diffusion."""

from typing import Tuple, Optional, Union
import torch
import torch.nn.functional as F


def select_device(device: Optional[Union[str, torch.device]] = "auto") -> torch.device:
    """Selects best available PyTorch compute device prioritizing Apple Silicon MPS.
    
    Order of preference for 'auto':
    1. Apple Silicon Metal Performance Shaders ('mps')
    2. NVIDIA CUDA ('cuda')
    3. CPU fallback ('cpu')
    
    Args:
        device: 'auto', 'mps', 'cuda', 'cpu', or torch.device instance.
        
    Returns:
        torch.device instance.
    """
    if isinstance(device, torch.device):
        return device

    dev_str = str(device or "auto").lower().strip()
    if dev_str == "auto":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    elif dev_str == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    elif dev_str == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    else:
        return torch.device("cpu")


def extract_into_tensor(
    tensor: torch.Tensor,
    timesteps: torch.Tensor,
    broadcast_shape: Tuple[int, ...],
) -> torch.Tensor:
    """Extracts values from 1D buffer at specified timesteps and reshapes for broadcasting.
    
    Args:
        tensor: 1D tensor of shape (T,).
        timesteps: 1D tensor of indices of shape (B,).
        broadcast_shape: Shape to broadcast against, e.g. (B, D).
        
    Returns:
        Tensor of shape (B, 1, 1, ...).
    """
    assert timesteps.ndim == 1, f"timesteps must be 1D, got ndim={timesteps.ndim}"
    res = tensor.gather(0, timesteps.to(tensor.device))
    reshape_dims = (res.shape[0],) + (1,) * (len(broadcast_shape) - 1)
    return res.reshape(reshape_dims)


def check_probability_simplex(
    probs: torch.Tensor,
    atol: float = 1e-4,
    dim: int = -1,
) -> bool:
    """Validates whether a tensor represents a valid probability distribution on the simplex.
    
    Checks:
    1. Non-negativity: probs >= -atol
    2. Sum along specified dimension equals 1.0 within tolerance.
    3. No NaNs or infinities.
    """
    if torch.isnan(probs).any() or torch.isinf(probs).any():
        return False
    if (probs < -atol).any():
        return False
    sums = probs.sum(dim=dim)
    if not torch.allclose(sums, torch.ones_like(sums), atol=atol):
        return False
    return True


def to_one_hot(indices: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Converts discrete integer class indices to float32 one-hot vectors."""
    assert indices.dtype in (torch.int64, torch.int32, torch.long), "indices must be integer type"
    return F.one_hot(indices.long(), num_classes=num_classes).to(dtype=torch.float32)


def from_one_hot(one_hot: torch.Tensor) -> torch.Tensor:
    """Extracts argmax class indices from one-hot or probability tensor."""
    return torch.argmax(one_hot, dim=-1)
