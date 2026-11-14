"""Unit tests for ConditionalSampler and condition generation strategies."""

import pytest
import torch
from tabsynth.models.hybrid import TabSynthHybridDenoiser
from tabsynth.diffusion.numerical.gaussian_diffusion import GaussianDiffusion
from tabsynth.diffusion.categorical.categorical_diffusion import CategoricalDiffusion
from tabsynth.sampling.mixed_sampler import MixedAncestralSampler
from tabsynth.sampling.conditional_sampler import ConditionalSampler


def test_conditional_sampler_strategies():
    """Verify balanced, natural, and class-specific conditioning policies."""
    device = torch.device("cpu")
    num_classes = 3
    class_dist = {"A": 0.8, "B": 0.1, "C": 0.1}
    idx_to_target = {0: "A", 1: "B", 2: "C"}

    model = TabSynthHybridDenoiser(
        num_continuous=2,
        cardinalities=[3],
        num_classes=num_classes,
        d_model=32,
        transformer_layers=1,
        mlp_layers=1,
    )
    g_diff = GaussianDiffusion(num_timesteps=5, device=device)
    c_diff = CategoricalDiffusion(cardinalities=[3], num_timesteps=5, device=device)

    mixed = MixedAncestralSampler(
        model=model,
        num_continuous=2,
        cardinalities=[3],
        gaussian_diffusion=g_diff,
        multinomial_diffusion=c_diff,
        num_timesteps=5,
        device=device,
    )

    sampler = ConditionalSampler(
        sampler=mixed,
        num_classes=num_classes,
        class_distribution=class_dist,
        idx_to_target=idx_to_target,
    )

    # 1. Balanced strategy
    c_bal = sampler.generate_conditions(batch_size=30, condition_strategy="balanced")
    assert c_bal.shape == (30,)
    assert (c_bal >= 0).all() and (c_bal < num_classes).all()

    # 2. Class specific
    c_spec = sampler.generate_conditions(batch_size=10, condition_strategy="class_specific", specific_class=2)
    assert (c_spec == 2).all()

    # 3. Sample batch
    x_num, x_cat_list, c_out = sampler.sample_batch(batch_size=4, condition_strategy="class_specific", specific_class=1)
    assert x_num.shape == (4, 2)
    assert len(x_cat_list) == 1
    assert (c_out == 1).all()
