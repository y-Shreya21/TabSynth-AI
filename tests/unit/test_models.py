"""Unit tests for TabSynth Hybrid Model and TabDDPM MLP baseline."""

import pytest
import torch
import torch.nn.functional as F
from tabsynth.models.embeddings import (
    NumericalTokenizer,
    CategoricalTokenizer,
    SinusoidalTimeEmbedding,
    ClassConditioningEmbedding,
)
from tabsynth.models.transformer import TransformerFeatureEncoder
from tabsynth.models.hybrid import TabSynthHybridModel
from tabsynth.baselines.mlp_diffusion import TabDDPMMLPModel


def test_tokenizers_and_embeddings():
    """Verify tokenizer projection tensor shapes and conditioning modules."""
    batch_size = 8
    d_model = 64
    num_continuous = 4
    cardinalities = [3, 5, 2]

    # 1. Numerical Tokenizer
    num_tok = NumericalTokenizer(num_features=num_continuous, d_model=d_model)
    x_num = torch.randn(batch_size, num_continuous)
    z_num = num_tok(x_num)
    assert z_num.shape == (batch_size, num_continuous, d_model)

    # 2. Categorical Tokenizer
    cat_tok = CategoricalTokenizer(cardinalities=cardinalities, d_model=d_model)
    x_cat_list = [F.one_hot(torch.randint(0, K, (batch_size,)), num_classes=K).float() for K in cardinalities]
    z_cat = cat_tok(x_cat_list)
    assert z_cat.shape == (batch_size, len(cardinalities), d_model)

    # 3. Sinusoidal Time Embedding
    time_emb = SinusoidalTimeEmbedding(d_model=d_model)
    t = torch.randint(0, 1000, (batch_size,))
    e_time = time_emb(t)
    assert e_time.shape == (batch_size, d_model)

    # 4. Class Conditioning Embedding
    class_emb = ClassConditioningEmbedding(num_classes=2, d_model=d_model, p_uncond=0.0)
    c = torch.randint(0, 2, (batch_size,))
    e_class = class_emb(c, batch_size=batch_size)
    assert e_class.shape == (batch_size, d_model)


def test_transformer_feature_encoder():
    """Verify Transformer feature encoder attention layers."""
    batch_size = 8
    num_features = 7
    d_model = 64

    transformer = TransformerFeatureEncoder(
        num_features=num_features,
        d_model=d_model,
        nhead=4,
        num_layers=2,
    )
    tokens = torch.randn(batch_size, num_features, d_model)
    out = transformer(tokens)
    assert out.shape == (batch_size, num_features, d_model)
    assert not torch.isnan(out).any()


def test_hybrid_model_forward():
    """Verify end-to-end forward pass of TabSynth Hybrid Model."""
    batch_size = 4
    num_continuous = 6
    cardinalities = [8, 16, 7]
    num_classes = 2

    model = TabSynthHybridModel(
        num_continuous=num_continuous,
        cardinalities=cardinalities,
        num_classes=num_classes,
        d_model=32,
        transformer_layers=2,
        attention_heads=4,
        mlp_hidden_dim=128,
        mlp_layers=2,
    )

    x_num = torch.randn(batch_size, num_continuous)
    x_cat_list = [F.one_hot(torch.randint(0, K, (batch_size,)), num_classes=K).float() for K in cardinalities]
    t = torch.randint(0, 1000, (batch_size,))
    c = torch.randint(0, num_classes, (batch_size,))

    pred_eps, pred_logits = model(x_num_t=x_num, x_cat_t_list=x_cat_list, t=t, c=c)

    assert pred_eps.shape == (batch_size, num_continuous)
    assert len(pred_logits) == len(cardinalities)
    for logits, K in zip(pred_logits, cardinalities):
        assert logits.shape == (batch_size, K)
        assert not torch.isnan(logits).any()


def test_mlp_baseline_forward():
    """Verify end-to-end forward pass of TabDDPM MLP baseline model."""
    batch_size = 4
    num_continuous = 6
    cardinalities = [8, 16, 7]
    num_classes = 2

    model = TabDDPMMLPModel(
        num_continuous=num_continuous,
        cardinalities=cardinalities,
        num_classes=num_classes,
        d_model=32,
        mlp_hidden_dim=128,
        mlp_layers=2,
    )

    x_num = torch.randn(batch_size, num_continuous)
    x_cat_list = [F.one_hot(torch.randint(0, K, (batch_size,)), num_classes=K).float() for K in cardinalities]
    t = torch.randint(0, 1000, (batch_size,))
    c = torch.randint(0, num_classes, (batch_size,))

    pred_eps, pred_logits = model(x_num_t=x_num, x_cat_t_list=x_cat_list, t=t, c=c)

    assert pred_eps.shape == (batch_size, num_continuous)
    assert len(pred_logits) == len(cardinalities)
    for logits, K in zip(pred_logits, cardinalities):
        assert logits.shape == (batch_size, K)
