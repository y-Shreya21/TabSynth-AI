"""Integration tests connecting denoising models with the Phase 02 diffusion engine."""

import pytest
import torch
import torch.nn.functional as F

from tabsynth.diffusion.mixed.mixed_diffusion import MixedTypeDiffusion
from tabsynth.models.hybrid import TabSynthHybridDenoiser
from tabsynth.models.mlp_baseline import TabSynthMLPDenoiser
from tabsynth.models.integration import ModelDiffusionIntegrator
from tabsynth.diffusion.utils import to_one_hot


@pytest.mark.parametrize("device_str", ["cpu", "mps"] if torch.backends.mps.is_available() else ["cpu"])
def test_hybrid_diffusion_integration(device_str):
    device = torch.device(device_str)
    B = 8
    D_num = 2
    cardinalities = [3, 4]
    num_classes = 2
    timesteps = 10

    # 1. Instantiate Phase 02 Mixed Diffusion Engine
    diffusion = MixedTypeDiffusion(
        num_numerical_features=D_num,
        categorical_cardinalities=cardinalities,
        numerical_timesteps=timesteps,
        categorical_timesteps=timesteps,
        device=device,
    )

    # 2. Instantiate Phase 03 Hybrid Model
    model = TabSynthHybridDenoiser(
        num_continuous=D_num,
        cardinalities=cardinalities,
        num_classes=num_classes,
        d_model=32,
        transformer_layers=1,
        attention_heads=2,
        mlp_layers=1,
    ).to(device)

    # 3. Instantiate Integrator
    integrator = ModelDiffusionIntegrator(model=model, diffusion=diffusion)

    # 4. Clean synthetic batch
    x_num = torch.randn(B, D_num, device=device)
    x_cat = [to_one_hot(torch.randint(0, K, (B,), device=device), num_classes=K) for K in cardinalities]
    labels = torch.randint(0, num_classes, (B,), device=device)

    # 5. Forward Loss Evaluation
    losses = integrator.forward_loss(
        x_num_0=x_num,
        x_cat_0_list=x_cat,
        class_labels=labels,
    )

    assert "loss_total" in losses
    assert "loss_num" in losses
    assert "loss_cat" in losses
    assert "loss_aux" in losses
    assert not torch.isnan(losses["loss_total"])

    # 6. Backpropagation
    losses["loss_total"].backward()
    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()

    # 7. Reverse Sampling
    sampled_num, sampled_cats = integrator.sample(batch_size=4, class_labels=torch.zeros(4, dtype=torch.long, device=device))
    assert sampled_num.shape == (4, D_num)
    assert len(sampled_cats) == len(cardinalities)
    for j, K in enumerate(cardinalities):
        assert sampled_cats[j].shape == (4, K)


def test_mlp_baseline_diffusion_integration():
    B = 8
    D_num = 2
    cardinalities = [2, 5]
    diffusion = MixedTypeDiffusion(
        num_numerical_features=D_num,
        categorical_cardinalities=cardinalities,
        numerical_timesteps=10,
        categorical_timesteps=10,
        device="cpu",
    )
    model = TabSynthMLPDenoiser(
        num_continuous=D_num,
        cardinalities=cardinalities,
        d_model=32,
        mlp_layers=2,
    )
    integrator = ModelDiffusionIntegrator(model=model, diffusion=diffusion)

    x_num = torch.randn(B, D_num)
    x_cat = [to_one_hot(torch.randint(0, K, (B,)), num_classes=K) for K in cardinalities]
    labels = torch.randint(0, 2, (B,))

    losses = integrator.forward_loss(x_num_0=x_num, x_cat_0_list=x_cat, class_labels=labels)
    assert losses["loss_total"] > 0.0
    losses["loss_total"].backward()
