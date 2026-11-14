"""Unit tests for learning rate schedulers and warmup policies."""

import pytest
import torch
import torch.nn as nn
from tabsynth.training.scheduler import build_scheduler, get_current_lr


def test_scheduler_warmup_and_cosine_decay():
    """Verify linear warmup followed by cosine annealing decay."""
    model = nn.Linear(10, 2)
    base_lr = 1e-3
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)

    scheduler = build_scheduler(
        optimizer=optimizer,
        scheduler_type="cosine",
        total_epochs=10,
        warmup_epochs=2,
        min_lr=1e-5,
    )
    assert scheduler is not None

    # Initial lr during warmup
    initial_lr = get_current_lr(optimizer)
    assert initial_lr < base_lr

    # Step through warmup
    optimizer.step()
    scheduler.step()  # Epoch 1
    optimizer.step()
    scheduler.step()  # Epoch 2 (end of warmup)
    warmup_end_lr = get_current_lr(optimizer)
    assert warmup_end_lr > initial_lr

    # Step through cosine decay
    for _ in range(8):
        optimizer.step()
        scheduler.step()

    final_lr = get_current_lr(optimizer)
    assert final_lr >= 1e-5
    assert final_lr < base_lr


def test_scheduler_none_and_constant():
    """Verify None and constant scheduler types."""
    model = nn.Linear(10, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    sched_none = build_scheduler(optimizer, scheduler_type="none")
    assert sched_none is None

    sched_const = build_scheduler(optimizer, scheduler_type="constant")
    assert sched_const is not None
    optimizer.step()
    sched_const.step()
    assert pytest.approx(get_current_lr(optimizer), rel=1e-5) == 1e-3
