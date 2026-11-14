"""Transition matrices and cumulative Markov transitions for discrete categorical diffusion.

Adheres strictly to the row-vector probability convention:
    x_t = x_{t-1} @ Q_t
where:
    x_{t-1} is a 1 x K row vector of probabilities on the simplex.
    [Q_t]_{ij} = q(x_t = j | x_{t-1} = i) is the probability of transitioning from state i to state j.
    Every row of Q_t sums to 1.0: sum_{j=1}^K [Q_t]_{ij} = 1.0.

For a uniform transition kernel with transition parameter alpha_t = 1 - beta_t:
    Q_t = alpha_t * I + (1 - alpha_t) * U
where:
    I is the K x K identity matrix.
    U is the K x K uniform matrix with every entry U_{ij} = 1 / K.

Cumulative transition matrix:
    Q_bar_t = Q_1 @ Q_2 @ ... @ Q_t = alpha_bar_t * I + (1 - alpha_bar_t) * U
where:
    alpha_bar_t = prod_{s=1}^t alpha_s
"""

from typing import Union, Optional
import torch


def get_transition_matrix(
    num_categories: int,
    alpha_t: Union[float, torch.Tensor],
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Constructs the discrete uniform transition matrix Q_t for a single feature.
    
    Q_t = alpha_t * I + (1 - alpha_t) / K * 1 1^T
    
    Args:
        num_categories: K >= 2 discrete category states.
        alpha_t: Transition retention probability scalar in [0, 1].
        device: Target compute device.
        
    Returns:
        Q_t: (K, K) row-stochastic transition matrix of float32.
    """
    assert num_categories >= 2, f"num_categories must be >= 2, got {num_categories}"
    K = num_categories

    if isinstance(alpha_t, (int, float)):
        alpha_val = float(alpha_t)
        dev = device or torch.device("cpu")
        eye = torch.eye(K, device=dev, dtype=torch.float32)
        uniform = torch.full((K, K), 1.0 / float(K), device=dev, dtype=torch.float32)
        Q = alpha_val * eye + (1.0 - alpha_val) * uniform
    else:
        dev = device or alpha_t.device
        alpha_tensor = alpha_t.to(device=dev, dtype=torch.float32)
        eye = torch.eye(K, device=dev, dtype=torch.float32)
        uniform = torch.full((K, K), 1.0 / float(K), device=dev, dtype=torch.float32)
        Q = alpha_tensor * eye + (1.0 - alpha_tensor) * uniform

    return Q


def get_cumulative_transition_matrix(
    num_categories: int,
    alpha_bar_t: Union[float, torch.Tensor],
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Constructs the cumulative discrete transition matrix Q_bar_t.
    
    Under uniform corruption, the product of uniform transition matrices is itself uniform:
        Q_bar_t = Q_1 @ ... @ Q_t = alpha_bar_t * I + (1 - alpha_bar_t) / K * 1 1^T
        
    Args:
        num_categories: K >= 2.
        alpha_bar_t: Cumulative alpha product prod_{s=1}^t alpha_s.
        device: Target device.
        
    Returns:
        Q_bar_t: (K, K) cumulative row-stochastic transition matrix.
    """
    return get_transition_matrix(
        num_categories=num_categories,
        alpha_t=alpha_bar_t,
        device=device,
    )


def validate_transition_matrix(
    Q: torch.Tensor,
    atol: float = 1e-5,
) -> bool:
    """Validates row-stochasticity, dimensionality, and bounds of a transition matrix.
    
    Checks:
    1. 2D square matrix (K, K).
    2. Non-negative entries: Q_{ij} >= -atol.
    3. Row stochasticity: sum_j Q_{ij} = 1.0 within atol.
    4. No NaNs or infinities.
    """
    if Q.ndim != 2:
        return False
    if Q.shape[0] != Q.shape[1]:
        return False
    if torch.isnan(Q).any() or torch.isinf(Q).any():
        return False
    if (Q < -atol).any():
        return False
    row_sums = Q.sum(dim=-1)
    ones = torch.ones_like(row_sums)
    if not torch.allclose(row_sums, ones, atol=atol):
        return False
    return True
