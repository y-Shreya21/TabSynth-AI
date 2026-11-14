"""Unit tests for checkpoint manager and top-k retention."""

import os
import shutil
import pytest
import torch
import torch.nn as nn
from tabsynth.training.checkpoint import CheckpointManager


@pytest.fixture
def tmp_ckpt_dir(tmp_path):
    d = str(tmp_path / "ckpts")
    os.makedirs(d, exist_ok=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_checkpoint_top_k_pruning(tmp_ckpt_dir):
    """Verify top-k checkpoint retention prunes worse checkpoints."""
    mgr = CheckpointManager(save_dir=tmp_ckpt_dir, keep_top_k=2)
    model = nn.Linear(5, 2)

    # Save epoch 1: loss 1.5
    p1 = mgr.save(model=model, preprocessor=None, epoch=1, val_loss=1.5, prefix="test")
    assert os.path.exists(p1)
    assert len(mgr.best_checkpoints) == 1

    # Save epoch 2: loss 1.2 (better)
    p2 = mgr.save(model=model, preprocessor=None, epoch=2, val_loss=1.2, prefix="test")
    assert os.path.exists(p2)
    assert len(mgr.best_checkpoints) == 2

    # Save epoch 3: loss 1.0 (best) -> should prune p1 (loss 1.5)
    p3 = mgr.save(model=model, preprocessor=None, epoch=3, val_loss=1.0, prefix="test")
    assert os.path.exists(p3)
    assert len(mgr.best_checkpoints) == 2

    # p1 should have been deleted because keep_top_k is 2
    assert not os.path.exists(p1)
    assert os.path.exists(p2)
    assert os.path.exists(p3)

    # Best checkpoint helper
    best_path = mgr.get_best_checkpoint(prefix="test")
    assert best_path is not None
    assert os.path.exists(best_path)


def test_checkpoint_save_and_load(tmp_ckpt_dir):
    """Verify state dict serialization and restoration."""
    mgr = CheckpointManager(save_dir=tmp_ckpt_dir, keep_top_k=2)
    model = nn.Linear(5, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    p = mgr.save(
        model=model,
        preprocessor="dummy_preprocessor",
        epoch=5,
        val_loss=0.42,
        optimizer=optimizer,
        config={"dim": 64},
        prefix="save_load",
    )

    # Load back into fresh model
    new_model = nn.Linear(5, 2)
    ckpt = CheckpointManager.load(checkpoint_path=p, model=new_model, device="cpu")

    assert ckpt["epoch"] == 5
    assert pytest.approx(ckpt["val_loss"], rel=1e-4) == 0.42
    assert ckpt["preprocessor"] == "dummy_preprocessor"
    assert ckpt["config"] == {"dim": 64}

    # Verify weights are identical
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.allclose(p1, p2)
