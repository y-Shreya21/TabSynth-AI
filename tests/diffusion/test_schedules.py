"""Unit tests for diffusion noise schedules and precomputed constants."""

import pytest
import torch
import numpy as np

from tabsynth.diffusion.schedules import (
    linear_beta_schedule,
    cosine_beta_schedule,
    get_noise_schedule,
    PrecomputedDiffusionConstants,
)


def test_linear_beta_schedule():
    T = 100
    betas = linear_beta_schedule(timesteps=T, beta_start=1e-4, beta_end=0.02)
    assert betas.shape == (T,)
    assert betas.dtype == torch.float32
    assert (betas > 0.0).all() and (betas < 1.0).all()
    # Check monotonicity of linear schedule
    diffs = torch.diff(betas)
    assert (diffs >= 0.0).all()
    assert torch.isclose(betas[0], torch.tensor(1e-4))
    assert torch.isclose(betas[-1], torch.tensor(0.02))


def test_cosine_beta_schedule():
    T = 100
    betas = cosine_beta_schedule(timesteps=T, s=0.008, max_beta=0.999)
    assert betas.shape == (T,)
    assert betas.dtype == torch.float32
    assert (betas > 0.0).all()
    assert (betas <= 0.999).all()
    assert not torch.isnan(betas).any()
    assert not torch.isinf(betas).any()


def test_schedule_factory():
    betas_cos = get_noise_schedule("cosine", timesteps=50)
    assert len(betas_cos) == 50
    betas_lin = get_noise_schedule("linear", timesteps=50)
    assert len(betas_lin) == 50

    with pytest.raises(ValueError):
        get_noise_schedule("unknown_schedule", timesteps=50)


def test_precomputed_constants_monotonicity_and_bounds():
    for sched in ["cosine", "linear"]:
        betas = get_noise_schedule(sched, timesteps=100)
        constants = PrecomputedDiffusionConstants(betas, device=torch.device("cpu"))

        # 1. Alphas monotonicity: alpha_bar_t must be monotonically decreasing
        alphas_cumprod = constants.alphas_cumprod
        diffs = alphas_cumprod[1:] - alphas_cumprod[:-1]
        assert (diffs <= 0.0).all(), f"Cumulative alphas must decrease monotonically in {sched}"

        # 2. Cumulative alphas bounds: strictly between 0 and 1
        assert (alphas_cumprod > 0.0).all()
        assert (alphas_cumprod <= 1.0).all()

        # 3. Posterior variance non-negativity
        assert (constants.posterior_variance >= 0.0).all()

        # 4. Absence of NaN or Inf
        for name, tensor in constants.get_all_buffers().items():
            assert not torch.isnan(tensor).any(), f"Found NaN in {name} for {sched}"
            assert not torch.isinf(tensor).any(), f"Found Inf in {name} for {sched}"


def test_extract_coefficients_broadcasting():
    betas = get_noise_schedule("cosine", timesteps=100)
    constants = PrecomputedDiffusionConstants(betas, device=torch.device("cpu"))

    batch_size = 16
    t = torch.randint(0, 100, (batch_size,))
    target_shape = (batch_size, 8)

    extracted = constants.extract(constants.sqrt_alphas_cumprod, t, target_shape)
    assert extracted.shape == (batch_size, 1)

    # Test multiplication broadcast
    data = torch.randn(target_shape)
    scaled = extracted * data
    assert scaled.shape == target_shape
