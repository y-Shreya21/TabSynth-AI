"""Unit tests for TabSynthMLPDenoiser baseline and architectural comparison."""

import pytest
import torch
import torch.nn.functional as F

from tabsynth.models.mlp_baseline import TabSynthMLPDenoiser
from tabsynth.models.hybrid import TabSynthHybridDenoiser, ModelOutput
from tabsynth.models.diagnostics import compare_models


def test_mlp_denoiser_shapes_and_interface():
    B = 8
    D_num = 2
    cardinalities = [3, 6]
    num_classes = 2
    d_model = 32

    mlp_model = TabSynthMLPDenoiser(
        num_continuous=D_num,
        cardinalities=cardinalities,
        num_classes=num_classes,
        d_model=d_model,
        mlp_layers=3,
    )

    x_num = torch.randn(B, D_num)
    x_cat = [F.one_hot(torch.randint(0, K, (B,)), num_classes=K).float() for K in cardinalities]
    t = torch.randint(0, 100, (B,))
    labels = torch.randint(0, num_classes, (B,))

    output = mlp_model(
        x_num_t=x_num,
        x_cat_t=x_cat,
        timesteps_num=t,
        timesteps_cat=t,
        class_labels=labels,
    )

    assert isinstance(output, ModelOutput)
    assert output.predicted_noise.shape == (B, D_num)
    assert len(output.categorical_logits) == len(cardinalities)
    for j, K in enumerate(cardinalities):
        assert output.categorical_logits[j].shape == (B, K)


def test_mlp_denoiser_gradient_flow():
    B = 4
    D_num = 2
    cardinalities = [3, 4]
    model = TabSynthMLPDenoiser(
        num_continuous=D_num,
        cardinalities=cardinalities,
        d_model=16,
        mlp_layers=2,
    )

    x_num = torch.randn(B, D_num)
    x_cat = [F.one_hot(torch.randint(0, K, (B,)), num_classes=K).float() for K in cardinalities]
    t = torch.randint(0, 50, (B,))

    output = model(x_num_t=x_num, x_cat_t=x_cat, timesteps_num=t)
    loss = output.predicted_noise.sum() + sum(l.sum() for l in output.categorical_logits)
    loss.backward()

    assert model.tokenizer.num_tokenizer.weights.grad is not None
    assert model.denoiser.blocks[0].ffn[0].weight.grad is not None
    assert model.head_num.proj.weight.grad is not None


def test_compare_hybrid_and_mlp_baseline():
    D_num = 4
    cardinalities = [4, 8, 2]
    d_model = 64

    hybrid = TabSynthHybridDenoiser(
        num_continuous=D_num,
        cardinalities=cardinalities,
        d_model=d_model,
        transformer_layers=2,
        attention_heads=4,
        mlp_layers=2,
    )
    mlp = TabSynthMLPDenoiser(
        num_continuous=D_num,
        cardinalities=cardinalities,
        d_model=d_model,
        mlp_layers=4,
    )

    comparison = compare_models(hybrid, mlp)
    assert "hybrid" in comparison
    assert "mlp_baseline" in comparison
    assert comparison["hybrid"]["total_parameters"] > 0
    assert comparison["mlp_baseline"]["total_parameters"] > 0
