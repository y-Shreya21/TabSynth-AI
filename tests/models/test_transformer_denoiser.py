"""Unit tests for TransformerFeatureEncoder and TabSynthHybridDenoiser."""

import pytest
import torch
import torch.nn.functional as F

from tabsynth.models.transformer import TransformerEncoderBlock, TransformerFeatureEncoder
from tabsynth.models.hybrid import TabSynthHybridDenoiser, ModelOutput


def test_transformer_block_and_encoder_shapes():
    B, D, d_model = 8, 7, 64
    block = TransformerEncoderBlock(d_model=d_model, nhead=4, dim_feedforward=256, dropout=0.1)
    tokens = torch.randn(B, D, d_model)

    out_block = block(tokens)
    assert out_block.shape == (B, D, d_model)

    encoder = TransformerFeatureEncoder(
        num_features=D,
        d_model=d_model,
        nhead=4,
        num_layers=2,
        dim_feedforward=256,
    )
    out_encoder = encoder(tokens)
    assert out_encoder.shape == (B, D, d_model)
    assert not torch.isnan(out_encoder).any()


def test_hybrid_denoiser_forward_and_output_shapes():
    B = 16
    D_num = 3
    cardinalities = [2, 5, 4]
    num_classes = 2
    d_model = 32

    model = TabSynthHybridDenoiser(
        num_continuous=D_num,
        cardinalities=cardinalities,
        num_classes=num_classes,
        d_model=d_model,
        transformer_layers=2,
        attention_heads=4,
        mlp_layers=2,
    )

    x_num = torch.randn(B, D_num)
    x_cat = [F.one_hot(torch.randint(0, K, (B,)), num_classes=K).float() for K in cardinalities]
    t = torch.randint(0, 100, (B,))
    labels = torch.randint(0, num_classes, (B,))

    output = model(
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

    # Tuple unpacking test
    pred_eps, pred_cat = output
    assert pred_eps.shape == (B, D_num)
    assert len(pred_cat) == len(cardinalities)


def test_hybrid_denoiser_gradient_flow():
    B = 4
    D_num = 2
    cardinalities = [3, 4]
    model = TabSynthHybridDenoiser(
        num_continuous=D_num,
        cardinalities=cardinalities,
        d_model=16,
        transformer_layers=1,
        attention_heads=2,
        mlp_layers=1,
    )

    x_num = torch.randn(B, D_num)
    x_cat = [F.one_hot(torch.randint(0, K, (B,)), num_classes=K).float() for K in cardinalities]
    t = torch.randint(0, 50, (B,))
    labels = torch.randint(0, 2, (B,))

    output = model(x_num_t=x_num, x_cat_t=x_cat, timesteps_num=t, class_labels=labels)
    loss = output.predicted_noise.sum() + sum(l.sum() for l in output.categorical_logits)
    loss.backward()

    # Verify gradients exist across all major modules
    assert model.tokenizer.num_tokenizer.weights.grad is not None
    assert model.transformer.layers[0].self_attn.in_proj_weight.grad is not None
    assert model.denoiser.blocks[0].ffn[0].weight.grad is not None
    assert model.head_num.proj.weight.grad is not None
    assert model.heads_cat.heads[0].weight.grad is not None
