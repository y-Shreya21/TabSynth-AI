"""End-to-end integration test for training loop, checkpointing, and reverse sampling."""

import os
import shutil
import pytest
import numpy as np
import pandas as pd
import torch
from tabsynth.data.schema import infer_table_schema
from tabsynth.data.preprocessor import TabularPreprocessor
from tabsynth.data.dataset import create_stratified_split, build_dataloaders
from tabsynth.diffusion.gaussian import GaussianDiffusion
from tabsynth.diffusion.multinomial import MultinomialDiffusion
from tabsynth.models.hybrid import TabSynthHybridModel
from tabsynth.training.losses import TabSynthLoss
from tabsynth.training.trainer import TabSynthTrainer, select_device
from tabsynth.sampling.sampler import TabSynthSampler


@pytest.fixture
def temp_checkpoint_dir(tmp_path):
    """Temporary checkpoint directory for integration tests."""
    ckpt_dir = str(tmp_path / "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    yield ckpt_dir
    shutil.rmtree(ckpt_dir, ignore_errors=True)


def test_end_to_end_training_and_sampling(temp_checkpoint_dir):
    """Verify full workflow: preprocessor -> model -> trainer -> checkpoint -> sampler."""
    np.random.seed(42)
    n = 120
    df = pd.DataFrame({
        "age": np.random.randint(20, 60, size=n).astype(float),
        "income": np.random.exponential(scale=30000, size=n),
        "education": np.random.choice(["Bachelors", "Masters", "PhD"], size=n),
        "target": np.random.choice(["0", "1"], size=n),
    })

    # 1. Stratified split and schema
    df_train, df_val, df_test = create_stratified_split(df, target_column="target", seed=42)
    schema = infer_table_schema(df_train, target_column="target")

    # 2. Preprocessor & DataLoaders
    preprocessor = TabularPreprocessor(schema=schema, seed=42).fit(df_train)
    train_loader, val_loader, test_loader = build_dataloaders(
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        preprocessor=preprocessor,
        batch_size=16,
    )

    # 3. Model & Diffusion Engine
    device = select_device("auto")
    num_timesteps = 20

    gaussian_diff = GaussianDiffusion(num_timesteps=num_timesteps, schedule_type="cosine", device=device)
    multinomial_diff = MultinomialDiffusion(
        cardinalities=schema.cardinalities,
        num_timesteps=num_timesteps,
        device=device,
    )

    model = TabSynthHybridModel(
        num_continuous=len(schema.continuous_columns),
        cardinalities=schema.cardinalities,
        num_classes=2,
        d_model=32,
        transformer_layers=1,
        attention_heads=2,
        mlp_hidden_dim=64,
        mlp_layers=2,
    )

    loss_fn = TabSynthLoss(
        multinomial_diffusion=multinomial_diff,
        cardinalities=schema.cardinalities,
    )

    # 4. Train model for 2 epochs
    trainer = TabSynthTrainer(
        model=model,
        gaussian_diffusion=gaussian_diff,
        multinomial_diffusion=multinomial_diff,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        preprocessor=preprocessor,
        learning_rate=1e-3,
        device=str(device),
        checkpoint_dir=temp_checkpoint_dir,
    )

    history = trainer.fit(epochs=2)
    assert len(history) == 2
    assert "loss_total" in history[0]
    assert "val_loss_total" in history[0]

    # Verify checkpoint was written
    saved_ckpts = [f for f in os.listdir(temp_checkpoint_dir) if f.endswith(".pt")]
    assert len(saved_ckpts) > 0

    # 5. Reverse Sampling
    sampler = TabSynthSampler(
        model=model,
        gaussian_diffusion=gaussian_diff,
        multinomial_diffusion=multinomial_diff,
        preprocessor=preprocessor,
        device=device,
    )

    synthetic_df = sampler.sample(
        num_samples=20,
        batch_size=10,
        condition_strategy="balanced",
        cfg_scale=1.0,
    )

    assert len(synthetic_df) == 20
    assert list(synthetic_df.columns) == list(df.columns)
    assert not synthetic_df["age"].isna().any()
    assert set(synthetic_df["education"].unique()).issubset({"Bachelors", "Masters", "PhD", "<missing>", "<unk>"})
