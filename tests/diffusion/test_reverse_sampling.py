"""Unit tests for numerical and categorical reverse samplers."""

import pytest
import torch
import torch.nn.functional as F

from tabsynth.diffusion.numerical.gaussian_diffusion import GaussianDiffusion
from tabsynth.diffusion.numerical.numerical_sampler import NumericalSampler
from tabsynth.diffusion.categorical.categorical_diffusion import CategoricalDiffusion
from tabsynth.diffusion.categorical.categorical_sampler import CategoricalSampler


def test_numerical_sampler_batch_sizes():
    T = 10
    diffusion = GaussianDiffusion(num_timesteps=T, device="cpu")
    sampler = NumericalSampler(diffusion=diffusion)

    # Simple dummy noise predictor model: predicts zero noise
    def dummy_noise_model(x_t, t):
        return torch.zeros_like(x_t)

    for B in [1, 16]:
        D = 4
        samples = sampler.sample(model_fn=dummy_noise_model, shape=(B, D), device="cpu")
        assert samples.shape == (B, D)
        assert not torch.isnan(samples).any()
        assert not torch.isinf(samples).any()


def test_numerical_sampler_all_steps():
    T = 5
    diffusion = GaussianDiffusion(num_timesteps=T, device="cpu")
    sampler = NumericalSampler(diffusion=diffusion)

    def dummy_noise_model(x_t, t):
        return torch.randn_like(x_t) * 0.1

    all_steps = sampler.sample(
        model_fn=dummy_noise_model, shape=(4, 2), device="cpu", return_all_steps=True
    )
    # T steps + initial noise state = T + 1 states
    assert len(all_steps) == T + 1
    for s in all_steps:
        assert s.shape == (4, 2)


def test_categorical_sampler_multi_feature_and_batches():
    cardinalities = [3, 6, 2]
    T = 8
    diff = CategoricalDiffusion(cardinalities=cardinalities, num_timesteps=T, device="cpu")
    sampler = CategoricalSampler(diffusion=diff)

    # Dummy model returning uniform random logits for each categorical column
    def dummy_cat_model(x_cat_list, t):
        return [torch.randn(x.shape[0], K) for x, K in zip(x_cat_list, cardinalities)]

    for B in [1, 8]:
        onehots, indices = sampler.sample(
            model_fn=dummy_cat_model,
            batch_size=B,
            device="cpu",
            return_indices=True,
        )

        assert len(onehots) == len(cardinalities)
        assert len(indices) == len(cardinalities)

        for j, K in enumerate(cardinalities):
            oh = onehots[j]
            idx = indices[j]
            assert oh.shape == (B, K)
            assert idx.shape == (B,)
            assert (idx >= 0).all() and (idx < K).all()
            # Ensure valid one-hot: max is 1, sum is 1
            assert torch.allclose(oh.sum(dim=-1), torch.ones(B))
            assert (oh >= 0.0).all()
