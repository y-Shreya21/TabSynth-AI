"""Unit tests for MixedAncestralSampler."""

import pytest
import torch
from tabsynth.models.hybrid import TabSynthHybridDenoiser
from tabsynth.diffusion.numerical.gaussian_diffusion import GaussianDiffusion
from tabsynth.diffusion.categorical.categorical_diffusion import CategoricalDiffusion
from tabsynth.sampling.mixed_sampler import MixedAncestralSampler


def test_mixed_sampler_trajectory_and_bounds():
    """Verify reverse sampling trajectory produces valid shapes and simplex bounds."""
    device = torch.device("cpu")
    b = 4
    num_continuous = 3
    cardinalities = [4, 5]
    num_classes = 2

    model = TabSynthHybridDenoiser(
        num_continuous=num_continuous,
        cardinalities=cardinalities,
        num_classes=num_classes,
        d_model=32,
        transformer_layers=1,
        mlp_layers=1,
    )

    gaussian_diff = GaussianDiffusion(num_timesteps=5, schedule_type="cosine", device=device)
    multinomial_diff = CategoricalDiffusion(cardinalities=cardinalities, num_timesteps=5, device=device)

    sampler = MixedAncestralSampler(
        model=model,
        num_continuous=num_continuous,
        cardinalities=cardinalities,
        gaussian_diffusion=gaussian_diff,
        multinomial_diffusion=multinomial_diff,
        num_timesteps=5,
        device=device,
    )

    c = torch.tensor([0, 1, 0, 1], device=device)
    x_num, x_cat_list = sampler.sample_step(batch_size=b, c=c, cfg_scale=1.0)

    # Verify continuous features
    assert x_num.shape == (b, num_continuous)
    assert not torch.isnan(x_num).any()

    # Verify discrete one-hot features
    assert len(x_cat_list) == len(cardinalities)
    for j, K in enumerate(cardinalities):
        assert x_cat_list[j].shape == (b, K)
        # Check sum to 1.0 (one-hot simplex)
        sums = x_cat_list[j].sum(dim=-1)
        assert torch.allclose(sums, torch.ones(b, device=device))
