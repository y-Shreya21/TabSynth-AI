"""Unit tests for early stopping monitor and best model restoration."""

import pytest
import torch
import torch.nn as nn
from tabsynth.training.early_stopping import EarlyStopping


def test_early_stopping_patience_and_termination():
    """Verify early stopping triggers when validation loss fails to improve within patience."""
    es = EarlyStopping(patience=3, min_delta=0.01, mode="min")

    assert not es(1.0, epoch=1)  # Best loss = 1.0
    assert not es(0.95, epoch=2)  # Improvement: best loss = 0.95
    assert not es(0.96, epoch=3)  # No improvement: count=1
    assert not es(0.97, epoch=4)  # No improvement: count=2
    should_stop = es(0.96, epoch=5)  # No improvement: count=3 == patience
    assert should_stop
    assert es.should_stop
    assert es.best_epoch == 2
    assert pytest.approx(abs(es.best_score), rel=1e-5) == 0.95


def test_early_stopping_restores_best_weights():
    """Verify model weights are restored to the best epoch when early stopping triggers."""
    model = nn.Linear(4, 2)
    # Record initial weights
    w_initial = model.weight.clone()

    es = EarlyStopping(patience=2, min_delta=0.01, restore_best_weights=True)

    # Epoch 1: best loss
    es(1.0, epoch=1, model=model)
    w_best = model.weight.clone()

    # Modify weights as if trained
    with torch.no_grad():
        model.weight.add_(torch.ones_like(model.weight) * 5.0)

    # Epoch 2 & 3: worsening loss
    es(1.2, epoch=2, model=model)
    es(1.3, epoch=3, model=model)

    assert es.should_stop
    # Verify weights were restored to w_best
    assert torch.allclose(model.weight, w_best)
