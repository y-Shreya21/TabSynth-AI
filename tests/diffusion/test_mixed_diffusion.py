"""Unit and integration tests for the unified MixedTypeDiffusion interface."""

import pytest
import torch
import torch.nn.functional as F

from tabsynth.diffusion.mixed.mixed_diffusion import MixedTypeDiffusion
from tabsynth.diffusion.utils import to_one_hot


@pytest.mark.parametrize("device_str", ["cpu", "mps"] if torch.backends.mps.is_available() else ["cpu"])
def test_mixed_diffusion_mixed_features(device_str):
    device = torch.device(device_str)
    num_continuous = 3
    cardinalities = [2, 5, 4]
    B = 16
    T_num = 15
    T_cat = 15

    mixed_diff = MixedTypeDiffusion(
        num_numerical_features=num_continuous,
        categorical_cardinalities=cardinalities,
        numerical_timesteps=T_num,
        categorical_timesteps=T_cat,
        device=device,
    )

    # Clean synthetic batch
    x_num = torch.randn(B, num_continuous, device=device)
    x_cat_list = [
        to_one_hot(torch.randint(0, K, (B,), device=device), num_classes=K)
        for K in cardinalities
    ]

    # 1. Forward diffusion
    x_t_num, noise_num, x_t_cat_oh, x_t_cat_idx = mixed_diff.forward(x_num, x_cat_list)
    assert x_t_num.shape == (B, num_continuous)
    assert noise_num.shape == (B, num_continuous)
    assert len(x_t_cat_oh) == len(cardinalities)
    for j, K in enumerate(cardinalities):
        assert x_t_cat_oh[j].shape == (B, K)
        assert x_t_cat_idx[j].shape == (B,)

    # 2. Loss computation
    pred_eps = torch.randn_like(x_t_num)
    pred_logits_list = [torch.randn(B, K, device=device) for K in cardinalities]
    t_cat = torch.randint(0, T_cat, (B,), device=device)

    losses = mixed_diff.compute_training_loss(
        pred_eps=pred_eps,
        target_noise=noise_num,
        pred_logits_list=pred_logits_list,
        clean_cat_list=x_cat_list,
        corrupted_cat_list=x_t_cat_oh,
        t_cat=t_cat,
    )

    assert "loss_total" in losses
    assert "loss_num" in losses
    assert "loss_cat" in losses
    assert "loss_aux" in losses
    assert losses["loss_total"] > 0.0
    assert not torch.isnan(losses["loss_total"])

    # 3. Sampling
    def dummy_model(x_n, x_c_list, t_n, t_c):
        eps = torch.zeros_like(x_n)
        logits = [torch.randn(x.shape[0], K, device=device) for x, K in zip(x_c_list, cardinalities)]
        return eps, logits

    samp_num, samp_cats = mixed_diff.sample(
        model_fn=dummy_model,
        batch_size=B,
        device=device,
    )
    assert samp_num.shape == (B, num_continuous)
    assert len(samp_cats) == len(cardinalities)
    for j, K in enumerate(cardinalities):
        assert samp_cats[j].shape == (B, K)
        assert torch.allclose(samp_cats[j].sum(dim=-1), torch.ones(B, device=device))


def test_mixed_diffusion_numerical_only():
    B = 8
    num_continuous = 4
    diff = MixedTypeDiffusion(
        num_numerical_features=num_continuous,
        categorical_cardinalities=[],
        device="cpu",
    )

    x_num = torch.randn(B, num_continuous)
    x_t_num, noise, x_t_cat, _ = diff.forward(x_num, None)
    assert x_t_num.shape == (B, num_continuous)
    assert len(x_t_cat) == 0

    def dummy_model(x_n, x_c_list, t_n, t_c):
        return torch.zeros_like(x_n), []

    samp_num, samp_cat = diff.sample(dummy_model, batch_size=B, device="cpu")
    assert samp_num.shape == (B, num_continuous)
    assert len(samp_cat) == 0


def test_mixed_diffusion_categorical_only():
    B = 8
    cardinalities = [3, 7]
    diff = MixedTypeDiffusion(
        num_numerical_features=0,
        categorical_cardinalities=cardinalities,
        device="cpu",
    )

    x_cat_list = [to_one_hot(torch.randint(0, K, (B,)), K) for K in cardinalities]
    x_t_num, _, x_t_cat, _ = diff.forward(None, x_cat_list)
    assert x_t_num.shape == (B, 0)
    assert len(x_t_cat) == 2

    def dummy_model(x_n, x_c_list, t_n, t_c):
        return torch.empty((B, 0)), [torch.randn(B, K) for K in cardinalities]

    samp_num, samp_cat = diff.sample(dummy_model, batch_size=B, device="cpu")
    assert samp_num.shape == (B, 0)
    assert len(samp_cat) == 2
    for j, K in enumerate(cardinalities):
        assert samp_cat[j].shape == (B, K)


def test_batch_size_one():
    diff = MixedTypeDiffusion(
        num_numerical_features=2,
        categorical_cardinalities=[3],
        device="cpu",
    )
    x_num = torch.randn(1, 2)
    x_cat = [to_one_hot(torch.tensor([1]), 3)]

    x_t_num, _, x_t_cat, _ = diff.forward(x_num, x_cat)
    assert x_t_num.shape == (1, 2)
    assert x_t_cat[0].shape == (1, 3)

    def dummy_model(x_n, x_c_list, t_n, t_c):
        return torch.zeros_like(x_n), [torch.randn(1, 3)]

    s_num, s_cat = diff.sample(dummy_model, batch_size=1, device="cpu")
    assert s_num.shape == (1, 2)
    assert s_cat[0].shape == (1, 3)
