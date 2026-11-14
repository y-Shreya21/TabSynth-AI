"""Unit tests for continuous Gaussian diffusion engine."""

import pytest
import torch
import numpy as np

from tabsynth.diffusion.numerical.gaussian_diffusion import GaussianDiffusion


def test_q_sample_shapes_and_noise():
    B, D = 32, 6
    T = 100
    diffusion = GaussianDiffusion(num_timesteps=T, schedule_type="cosine", device="cpu")

    x_0 = torch.randn(B, D)
    t = torch.randint(0, T, (B,))

    # 1. Unspecified noise (sampled internally)
    x_t, noise = diffusion.q_sample(x_start=x_0, timesteps=t)
    assert x_t.shape == (B, D)
    assert noise.shape == (B, D)
    assert x_t.dtype == torch.float32

    # 2. Supplied explicit noise
    fixed_noise = torch.ones_like(x_0) * 0.5
    x_t_fixed, noise_ret = diffusion.q_sample(x_start=x_0, timesteps=t, noise=fixed_noise)
    assert torch.allclose(noise_ret, fixed_noise)

    # 3. Mathematical exact check: x_t = sqrt(alpha_bar)*x_0 + sqrt(1 - alpha_bar)*noise
    alpha_bar = diffusion.constants.alphas_cumprod[t].unsqueeze(1)
    expected_x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1.0 - alpha_bar) * fixed_noise
    assert torch.allclose(x_t_fixed, expected_x_t, atol=1e-5)


def test_q_sample_boundary_behaviors():
    B, D = 100, 4
    T = 100
    diffusion = GaussianDiffusion(num_timesteps=T, schedule_type="cosine", device="cpu")
    x_0 = torch.randn(B, D)

    # At t = 0: alpha_bar_0 is close to 1.0 (noise variance is minimal)
    t_zero = torch.zeros(B, dtype=torch.long)
    x_t_zero, _ = diffusion.q_sample(x_start=x_0, timesteps=t_zero)
    # Distance between x_t_zero and x_0 should be very small
    assert torch.mean((x_t_zero - x_0) ** 2) < 0.05

    # At t = T-1: alpha_bar is small, x_t dominated by Gaussian noise
    t_last = torch.full((B,), T - 1, dtype=torch.long)
    fixed_noise = torch.randn(B, D)
    x_t_last, _ = diffusion.q_sample(x_start=x_0, timesteps=t_last, noise=fixed_noise)
    # Correlation between x_t_last and fixed_noise should be high
    corr = torch.cosine_similarity(x_t_last.flatten(), fixed_noise.flatten(), dim=0)
    assert corr > 0.85


def test_predict_x0_and_posterior_mean():
    B, D = 16, 4
    T = 50
    diffusion = GaussianDiffusion(num_timesteps=T, schedule_type="cosine", device="cpu")

    x_0 = torch.randn(B, D)
    t = torch.randint(0, T, (B,))
    noise = torch.randn(B, D)
    x_t, _ = diffusion.q_sample(x_start=x_0, timesteps=t, noise=noise)

    # Perfect noise prediction should reconstruct exact x_0
    rec_x_0 = diffusion.predict_x0_from_eps(x_t=x_t, timesteps=t, eps=noise)
    assert torch.allclose(rec_x_0, x_0, atol=1e-4)

    # Posterior mean and variance
    mean, var, log_var = diffusion.q_posterior_mean_variance(x_start=x_0, x_t=x_t, timesteps=t)
    assert mean.shape == (B, D)
    assert var.shape == (B, 1)
    assert (var >= 0.0).all()


def test_loss_reductions_and_weights():
    diffusion = GaussianDiffusion(num_timesteps=100, device="cpu")
    pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    target = torch.tensor([[1.5, 2.0], [2.0, 5.0]])
    # Squared errors: [[0.25, 0.0], [1.0, 1.0]] -> mean = 2.25 / 4 = 0.5625

    loss_mean = diffusion.loss(pred, target, reduction="mean")
    assert torch.isclose(loss_mean, torch.tensor(0.5625))

    loss_sum = diffusion.loss(pred, target, reduction="sum")
    assert torch.isclose(loss_sum, torch.tensor(2.25))

    weights = torch.tensor([[2.0, 1.0], [1.0, 2.0]])
    # Weighted: [[0.5, 0.0], [1.0, 2.0]] -> mean = 3.5 / 4 = 0.875
    loss_weighted = diffusion.loss(pred, target, weights=weights, reduction="mean")
    assert torch.isclose(loss_weighted, torch.tensor(0.875))


def test_p_sample_noise_suppression_at_zero():
    B, D = 8, 3
    diffusion = GaussianDiffusion(num_timesteps=10, device="cpu")
    x_t = torch.randn(B, D)
    predicted_eps = torch.zeros(B, D)

    # When t = 0, no noise should be added; consecutive runs should be deterministic given x_t
    t_zero = torch.zeros(B, dtype=torch.long)
    step1 = diffusion.p_sample_step(predicted_eps, x_t, t_zero)
    step2 = diffusion.p_sample_step(predicted_eps, x_t, t_zero)
    assert torch.allclose(step1, step2)

    # When t > 0, stochastic noise is injected; consecutive runs differ
    t_nonzero = torch.full((B,), 5, dtype=torch.long)
    step_stochastic1 = diffusion.p_sample_step(predicted_eps, x_t, t_nonzero)
    step_stochastic2 = diffusion.p_sample_step(predicted_eps, x_t, t_nonzero)
    assert not torch.allclose(step_stochastic1, step_stochastic2)
