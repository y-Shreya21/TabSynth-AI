"""Unit tests for optimizer factory and parameter grouping."""

import pytest
import torch
import torch.nn as nn
from tabsynth.models.hybrid import TabSynthHybridDenoiser
from tabsynth.training.optimizer import build_optimizer, get_parameter_groups


def test_parameter_grouping_decay_separation():
    """Verify 2D+ weights have weight_decay while 1D biases/norms have weight_decay=0."""
    model = TabSynthHybridDenoiser(
        num_continuous=3,
        cardinalities=[4, 5],
        num_classes=2,
        d_model=32,
        transformer_layers=1,
        mlp_layers=1,
    )

    groups = get_parameter_groups(model, weight_decay=1e-4)
    assert len(groups) == 2
    assert groups[0]["weight_decay"] == 1e-4
    assert groups[1]["weight_decay"] == 0.0

    # Ensure no param is in both groups and every trainable param is accounted for
    decay_set = set(id(p) for p in groups[0]["params"])
    no_decay_set = set(id(p) for p in groups[1]["params"])
    assert len(decay_set.intersection(no_decay_set)) == 0

    all_params = set(id(p) for p in model.parameters() if p.requires_grad)
    assert decay_set.union(no_decay_set) == all_params

    # Verify no 1D param is in decay group
    for p in groups[0]["params"]:
        assert p.ndim >= 2


def test_build_optimizer_types():
    """Verify optimizer factory instantiates AdamW, Adam, and SGD."""
    model = nn.Linear(10, 2)

    opt_adamw = build_optimizer(model, optimizer_type="adamw", learning_rate=1e-3, weight_decay=1e-4)
    assert isinstance(opt_adamw, torch.optim.AdamW)

    opt_adam = build_optimizer(model, optimizer_type="adam", learning_rate=1e-3, weight_decay=1e-4)
    assert isinstance(opt_adam, torch.optim.Adam)

    opt_sgd = build_optimizer(model, optimizer_type="sgd", learning_rate=1e-2, weight_decay=1e-4)
    assert isinstance(opt_sgd, torch.optim.SGD)

    with pytest.raises(ValueError):
        build_optimizer(model, optimizer_type="unsupported_optimizer")
