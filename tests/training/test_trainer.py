"""Unit tests for TabSynthTrainer end-to-end multi-epoch lifecycle."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tabsynth.models.hybrid import TabSynthHybridDenoiser
from tabsynth.diffusion.numerical.gaussian_diffusion import GaussianDiffusion
from tabsynth.diffusion.categorical.categorical_diffusion import CategoricalDiffusion
from tabsynth.training.losses import TabSynthLoss
from tabsynth.training.trainer import TabSynthTrainer


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, n=32):
        self.n = n
        self.x_num = torch.randn(n, 2)
        self.x_cat0 = torch.eye(3)[torch.randint(0, 3, (n,))]
        self.x_cat1 = torch.eye(4)[torch.randint(0, 4, (n,))]
        self.y = torch.randint(0, 2, (n,))

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return (self.x_num[idx], [self.x_cat0[idx], self.x_cat1[idx]], self.y[idx])


def collate_fn(batch):
    nums, cats, ys = zip(*batch)
    return (
        torch.stack(nums),
        [torch.stack([c[0] for c in cats]), torch.stack([c[1] for c in cats])],
        torch.stack(ys),
    )


def test_trainer_fit_convergence(tmp_path):
    """Verify trainer executes multi-epoch fit and records metrics."""
    train_ds = DummyDataset(n=32)
    val_ds = DummyDataset(n=16)

    train_loader = DataLoader(train_ds, batch_size=16, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=16, collate_fn=collate_fn)

    cardinalities = [3, 4]
    model = TabSynthHybridDenoiser(
        num_continuous=2,
        cardinalities=cardinalities,
        num_classes=2,
        d_model=32,
        transformer_layers=1,
        mlp_layers=1,
    )

    gaussian_diff = GaussianDiffusion(num_timesteps=10, schedule_type="cosine")
    multinomial_diff = CategoricalDiffusion(cardinalities=cardinalities, num_timesteps=10)
    loss_fn = TabSynthLoss(multinomial_diffusion=multinomial_diff, cardinalities=cardinalities)

    trainer = TabSynthTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        gaussian_diffusion=gaussian_diff,
        multinomial_diffusion=multinomial_diff,
        loss_fn=loss_fn,
        learning_rate=1e-3,
        checkpoint_dir=str(tmp_path / "ckpts"),
    )

    history = trainer.fit(epochs=2, verbose_interval=1)
    assert len(history) == 2
    assert "loss_total" in history[0]
    assert "val_loss_total" in history[0]
    assert "learning_rate" in history[0]
