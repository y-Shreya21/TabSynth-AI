"""Unit tests for NumericalOutputHead and CategoricalOutputHeads."""

import pytest
import torch

from tabsynth.models.heads import NumericalOutputHead, CategoricalOutputHeads


def test_numerical_output_head():
    B, D_num, d_model = 10, 4, 32
    head = NumericalOutputHead(num_continuous=D_num, d_model=d_model)

    tokens = torch.randn(B, D_num, d_model)
    pred_eps = head(tokens)

    assert pred_eps.shape == (B, D_num)
    assert not torch.isnan(pred_eps).any()

    # Gradient flow
    loss = pred_eps.sum()
    loss.backward()
    assert head.proj.weight.grad is not None


def test_categorical_output_heads_arbitrary_cardinalities():
    B, d_model = 8, 64
    cardinalities = [2, 5, 12, 3]
    heads = CategoricalOutputHeads(cardinalities=cardinalities, d_model=d_model)

    cat_tokens = torch.randn(B, len(cardinalities), d_model)
    logits_list = heads(cat_tokens)

    assert len(logits_list) == len(cardinalities)
    for j, K in enumerate(cardinalities):
        assert logits_list[j].shape == (B, K)
        assert not torch.isnan(logits_list[j]).any()

    # Gradient flow
    loss = sum(l.sum() for l in logits_list)
    loss.backward()
    for head in heads.heads:
        assert head.weight.grad is not None
