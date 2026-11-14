"""Unit tests for train step execution and gradient clipping."""

import pytest
import torch
import torch.nn as nn
from tabsynth.models.hybrid import TabSynthHybridDenoiser
from tabsynth.diffusion.numerical.gaussian_diffusion import GaussianDiffusion
from tabsynth.diffusion.categorical.categorical_diffusion import CategoricalDiffusion
from tabsynth.training.losses import TabSynthLoss
from tabsynth.training.train_step import execute_train_step


def test_train_step_gradient_flow_and_clipping():
    """Verify train step updates model weights and clips gradients."""
    device = torch.device("cpu")
    batch_size = 8
    num_continuous = 2
    cardinalities = [3, 4]
    num_classes = 2

    model = TabSynthHybridDenoiser(
        num_continuous=num_continuous,
        cardinalities=cardinalities,
        num_classes=num_classes,
        d_model=32,
        transformer_layers=1,
        mlp_layers=1,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    gaussian_diff = GaussianDiffusion(num_timesteps=10, schedule_type="cosine", device=device)
    multinomial_diff = CategoricalDiffusion(cardinalities=cardinalities, num_timesteps=10, device=device)
    loss_fn = TabSynthLoss(multinomial_diffusion=multinomial_diff, cardinalities=cardinalities)

    # Dummy batch
    x_num = torch.randn(batch_size, num_continuous)
    x_cat = [
        torch.eye(3)[torch.randint(0, 3, (batch_size,))],
        torch.eye(4)[torch.randint(0, 4, (batch_size,))],
    ]
    labels = torch.randint(0, num_classes, (batch_size,))
    batch = (x_num, x_cat, labels)

    # Record parameter before step
    param_before = list(model.parameters())[0].clone()

    metrics = execute_train_step(
        model=model,
        optimizer=optimizer,
        batch=batch,
        loss_fn=loss_fn,
        gaussian_diffusion=gaussian_diff,
        multinomial_diffusion=multinomial_diff,
        device=device,
        max_grad_norm=1.0,
        cardinalities=cardinalities,
    )

    param_after = list(model.parameters())[0]

    assert "loss_total" in metrics
    assert "loss_num_mse" in metrics
    assert "loss_cat_kl" in metrics
    assert "grad_norm" in metrics
    assert metrics["grad_norm"] >= 0.0

    # Ensure weights were updated
    assert not torch.allclose(param_before, param_after)
