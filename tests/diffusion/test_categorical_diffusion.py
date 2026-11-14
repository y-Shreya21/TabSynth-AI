"""Unit and mathematical reference tests for discrete categorical diffusion."""

import pytest
import torch
import torch.nn.functional as F
import numpy as np

from tabsynth.diffusion.categorical.categorical_diffusion import CategoricalDiffusion
from tabsynth.diffusion.utils import check_probability_simplex, to_one_hot


def test_q_sample_categorical_shapes_and_ranges():
    cardinalities = [3, 5, 8]
    B = 32
    T = 50
    diff = CategoricalDiffusion(cardinalities=cardinalities, num_timesteps=T, device="cpu")

    # Clean integer inputs
    x_0_list = [
        torch.randint(0, K, (B,)) for K in cardinalities
    ]
    t = torch.randint(0, T, (B,))

    onehots, indices = diff.q_sample_categorical(x_0_list=x_0_list, timesteps=t)

    assert len(onehots) == len(cardinalities)
    assert len(indices) == len(cardinalities)

    for j, K in enumerate(cardinalities):
        oh = onehots[j]
        idx = indices[j]
        assert oh.shape == (B, K)
        assert idx.shape == (B,)
        # Valid integer category range
        assert (idx >= 0).all() and (idx < K).all()
        # Sum of one-hot along last dimension must be exactly 1
        assert torch.allclose(oh.sum(dim=-1), torch.ones(B))


def test_exact_posterior_mathematical_reference_numpy():
    """Validates PyTorch exact posterior q(x_{t-1} | x_t, x_0) against independent NumPy analytical derivation."""
    K = 3
    diff = CategoricalDiffusion(cardinalities=[K], num_timesteps=20, schedule_type="linear", device="cpu")

    t_val = 5
    t = torch.tensor([t_val])
    alpha_t = float(diff.alphas[t_val].item())
    alpha_bar_prev = float(diff.alphas_cumprod_prev[t_val].item())

    # Case: x_0 = state 0 ([1, 0, 0]), x_t = state 1 ([0, 1, 0])
    x_0_torch = torch.tensor([[1.0, 0.0, 0.0]])
    x_t_torch = torch.tensor([[0.0, 1.0, 0.0]])

    # 1. PyTorch computation
    post_torch = diff.q_posterior_probs(
        x_t_onehot=x_t_torch,
        x_0_onehot=x_0_torch,
        timesteps=t,
        K=K,
    )
    assert check_probability_simplex(post_torch)

    # 2. Independent NumPy derivation
    # q(x_t = 1 | x_{t-1} = j) for j in {0, 1, 2}:
    # For j = 1: alpha_t + (1 - alpha_t) / 3
    # For j != 1: (1 - alpha_t) / 3
    q_xt_given_xtm1 = np.array([
        (1.0 - alpha_t) / 3.0,
        alpha_t + (1.0 - alpha_t) / 3.0,
        (1.0 - alpha_t) / 3.0,
    ], dtype=np.float64)

    # q(x_{t-1} = j | x_0 = 0) for j in {0, 1, 2}:
    # For j = 0: alpha_bar_prev + (1 - alpha_bar_prev) / 3
    # For j != 0: (1 - alpha_bar_prev) / 3
    q_xtm1_given_x0 = np.array([
        alpha_bar_prev + (1.0 - alpha_bar_prev) / 3.0,
        (1.0 - alpha_bar_prev) / 3.0,
        (1.0 - alpha_bar_prev) / 3.0,
    ], dtype=np.float64)

    # Unnormalized posterior product
    unnorm_np = q_xt_given_xtm1 * q_xtm1_given_x0
    post_np = unnorm_np / np.sum(unnorm_np)

    # 3. Numerical assertion
    np.testing.assert_allclose(
        post_torch.numpy().flatten(),
        post_np.astype(np.float32),
        rtol=1e-5,
        atol=1e-5,
        err_msg="PyTorch discrete posterior differs from analytical NumPy posterior!",
    )


def test_learned_reverse_posterior_probability_marginalization():
    """Verifies that learned reverse posterior correctly marginalizes over probabilities without argmax."""
    K = 3
    diff = CategoricalDiffusion(cardinalities=[K], num_timesteps=10, device="cpu")
    t = torch.tensor([3])
    x_t = torch.tensor([[0.0, 1.0, 0.0]])  # State 1

    # Soft non-degenerate logits
    logits = torch.tensor([[2.0, 1.0, 0.5]])
    probs = F.softmax(logits, dim=-1)

    p_prev = diff.p_probs(x_t_onehot=x_t, pred_x_0_logits=logits, timesteps=t, K=K)
    assert check_probability_simplex(p_prev)

    # Compare against hard argmax (which must NOT be equal to full marginalization)
    hard_argmax_logits = torch.tensor([[100.0, -100.0, -100.0]])  # 100% on state 0
    p_prev_hard = diff.p_probs(x_t_onehot=x_t, pred_x_0_logits=hard_argmax_logits, timesteps=t, K=K)

    # Since smooth distribution is a mixture of all states, it must differ from pure state 0
    assert not torch.allclose(p_prev, p_prev_hard, atol=1e-2)

    # Manual mixture assertion: p_prev = sum_k probs[k] * q_post(x_0 = e_k)
    q_k0 = diff.q_posterior_probs(x_t, torch.tensor([[1.0, 0.0, 0.0]]), t, K)
    q_k1 = diff.q_posterior_probs(x_t, torch.tensor([[0.0, 1.0, 0.0]]), t, K)
    q_k2 = diff.q_posterior_probs(x_t, torch.tensor([[0.0, 0.0, 1.0]]), t, K)
    expected_mixture = probs[0, 0] * q_k0 + probs[0, 1] * q_k1 + probs[0, 2] * q_k2

    assert torch.allclose(p_prev, expected_mixture, atol=1e-5)


def test_kl_divergence_and_boundary_at_zero():
    K = 4
    B = 16
    diff = CategoricalDiffusion(cardinalities=[K], num_timesteps=20, device="cpu")

    x_0 = to_one_hot(torch.randint(0, K, (B,)), K)
    x_t = to_one_hot(torch.randint(0, K, (B,)), K)
    logits = torch.randn(B, K)

    # 1. Non-zero timesteps: KL divergence is non-negative when p_pred == q_post
    t_nonzero = torch.full((B,), 5, dtype=torch.long)
    loss_nonzero = diff.compute_kl_divergence(x_t, x_0, logits, t_nonzero, K)
    assert not torch.isnan(loss_nonzero)
    assert not torch.isinf(loss_nonzero)

    # Perfect prediction (model predicting clean x_0) yields zero KL divergence
    perfect_x0_logits = torch.log(x_0 + 1e-12)
    min_kl = diff.compute_kl_divergence(x_t, x_0, perfect_x0_logits, t_nonzero, K)
    assert min_kl < 1e-4

    # 2. Timestep 0: Exact NLL cross-entropy
    t_zero = torch.zeros(B, dtype=torch.long)
    loss_zero = diff.compute_kl_divergence(x_t, x_0, logits, t_zero, K)
    expected_nll = F.cross_entropy(logits, torch.argmax(x_0, dim=-1))
    assert torch.isclose(loss_zero, expected_nll, atol=1e-4)
