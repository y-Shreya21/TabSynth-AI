"""Unit tests for feature tokenizers, column identity embeddings, and feature type embeddings."""

import pytest
import torch
import torch.nn.functional as F

from tabsynth.models.tokenizer import (
    NumericalFeatureTokenizer,
    CategoricalFeatureTokenizer,
    FeatureTokenizer,
)


def test_numerical_tokenizer_shapes():
    B, D_num, d_model = 16, 5, 64
    tokenizer = NumericalFeatureTokenizer(num_features=D_num, d_model=d_model)

    x_num = torch.randn(B, D_num)
    tokens = tokenizer(x_num)

    assert tokens.shape == (B, D_num, d_model)
    assert not torch.isnan(tokens).any()

    # Verify column distinctness: weights are separate per feature
    assert tokenizer.weights.shape == (D_num, d_model)
    assert tokenizer.biases.shape == (D_num, d_model)


def test_categorical_tokenizer_one_hot_and_integer_indices():
    B, d_model = 8, 32
    cardinalities = [3, 5, 8]
    tokenizer = CategoricalFeatureTokenizer(cardinalities=cardinalities, d_model=d_model)

    # 1. One-hot input
    x_cat_onehot = [
        F.one_hot(torch.randint(0, K, (B,)), num_classes=K).float()
        for K in cardinalities
    ]
    tokens_oh = tokenizer(x_cat_onehot)
    assert tokens_oh.shape == (B, len(cardinalities), d_model)

    # 2. Integer index input
    x_cat_int = [torch.randint(0, K, (B,)) for K in cardinalities]
    tokens_int = tokenizer(x_cat_int)
    assert tokens_int.shape == (B, len(cardinalities), d_model)


def test_feature_tokenizer_mixed_data():
    B, d_model = 12, 64
    D_num = 3
    cardinalities = [2, 4, 6]
    total_features = D_num + len(cardinalities)

    tokenizer = FeatureTokenizer(
        num_continuous=D_num,
        cardinalities=cardinalities,
        d_model=d_model,
    )

    x_num = torch.randn(B, D_num)
    x_cat = [torch.randint(0, K, (B,)) for K in cardinalities]

    tokens = tokenizer(x_num=x_num, x_cat_list=x_cat)
    assert tokens.shape == (B, total_features, d_model)
    assert not torch.isnan(tokens).any()


def test_feature_tokenizer_numerical_only():
    B, d_model, D_num = 6, 32, 4
    tokenizer = FeatureTokenizer(num_continuous=D_num, cardinalities=[], d_model=d_model)

    x_num = torch.randn(B, D_num)
    tokens = tokenizer(x_num=x_num, x_cat_list=None)
    assert tokens.shape == (B, D_num, d_model)


def test_feature_tokenizer_categorical_only():
    B, d_model = 6, 32
    cardinalities = [3, 7]
    tokenizer = FeatureTokenizer(num_continuous=0, cardinalities=cardinalities, d_model=d_model)

    x_cat = [torch.randint(0, K, (B,)) for K in cardinalities]
    tokens = tokenizer(x_num=None, x_cat_list=x_cat)
    assert tokens.shape == (B, len(cardinalities), d_model)


def test_feature_tokenizer_batch_size_one():
    d_model = 16
    tokenizer = FeatureTokenizer(num_continuous=2, cardinalities=[3], d_model=d_model)

    x_num = torch.randn(1, 2)
    x_cat = [torch.tensor([1])]
    tokens = tokenizer(x_num=x_num, x_cat_list=x_cat)
    assert tokens.shape == (1, 3, d_model)
