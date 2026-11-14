"""Unit and mathematical reference tests for discrete transition matrices."""

import pytest
import torch
import numpy as np

from tabsynth.diffusion.categorical.transition import (
    get_transition_matrix,
    get_cumulative_transition_matrix,
    validate_transition_matrix,
)


def test_transition_matrix_properties():
    for K in [2, 3, 5, 10, 25]:
        alpha_t = 0.85
        Q = get_transition_matrix(num_categories=K, alpha_t=alpha_t)
        assert Q.shape == (K, K)
        assert validate_transition_matrix(Q)

        # Check diagonal entries: alpha_t + (1 - alpha_t) / K
        expected_diag = alpha_t + (1.0 - alpha_t) / float(K)
        assert torch.allclose(torch.diag(Q), torch.full((K,), expected_diag), atol=1e-6)

        # Check off-diagonal entries: (1 - alpha_t) / K
        expected_offdiag = (1.0 - alpha_t) / float(K)
        eye = torch.eye(K)
        offdiag = Q[eye == 0]
        assert torch.allclose(offdiag, torch.full_like(offdiag, expected_offdiag), atol=1e-6)


def test_transition_matrix_boundary_values():
    K = 4
    # When alpha = 1.0 (no corruption), Q should be the exact identity matrix
    Q_identity = get_transition_matrix(num_categories=K, alpha_t=1.0)
    assert torch.allclose(Q_identity, torch.eye(K))

    # When alpha = 0.0 (maximum corruption), Q should be pure uniform
    Q_uniform = get_transition_matrix(num_categories=K, alpha_t=0.0)
    assert torch.allclose(Q_uniform, torch.full((K, K), 1.0 / float(K)))


def test_cumulative_matrix_mathematical_reference_numpy():
    """Validates PyTorch cumulative transition against recursive NumPy matrix multiplication."""
    K = 3
    alphas = [0.95, 0.90, 0.85, 0.80]

    # 1. Compute cumulative product recursively in pure NumPy
    I_np = np.eye(K, dtype=np.float64)
    U_np = np.full((K, K), 1.0 / K, dtype=np.float64)

    Q_cum_np = I_np.copy()
    alpha_bar = 1.0
    for a in alphas:
        alpha_bar *= a
        Q_step_np = a * I_np + (1.0 - a) * U_np
        # Row-vector convention: x_1 = x_0 @ Q_1, x_2 = x_1 @ Q_2 = x_0 @ (Q_1 @ Q_2)
        Q_cum_np = Q_cum_np @ Q_step_np

    # 2. Compute via PyTorch cumulative transition matrix
    Q_cum_torch = get_cumulative_transition_matrix(num_categories=K, alpha_bar_t=alpha_bar)

    # 3. Mathematical equivalence check
    assert validate_transition_matrix(Q_cum_torch)
    np.testing.assert_allclose(
        Q_cum_torch.numpy(),
        Q_cum_np.astype(np.float32),
        rtol=1e-5,
        atol=1e-5,
        err_msg="PyTorch cumulative transition matrix differs from NumPy recursive matrix product!"
    )


def test_row_vector_multiplication_distribution():
    """Verify x_t = x_0 @ Q_bar_t preserves probability distribution simplex."""
    K = 4
    alpha_bar = 0.70
    Q_bar = get_cumulative_transition_matrix(num_categories=K, alpha_bar_t=alpha_bar)

    # Clean one-hot state: state 1 (index 0)
    x_0 = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    x_t_dist = x_0 @ Q_bar  # (1, K)

    # Expected: alpha_bar for state 0 + (1 - alpha_bar) / K, and (1 - alpha_bar) / K for others
    expected_prob_0 = alpha_bar + (1.0 - alpha_bar) / 4.0
    expected_prob_other = (1.0 - alpha_bar) / 4.0

    assert torch.isclose(x_t_dist[0, 0], torch.tensor(expected_prob_0))
    assert torch.isclose(x_t_dist[0, 1], torch.tensor(expected_prob_other))
    assert torch.isclose(x_t_dist.sum(), torch.tensor(1.0))
