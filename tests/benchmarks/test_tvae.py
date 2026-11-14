"""Tests for native PyTorch TVAESynthesizer."""

import os
import pytest
import numpy as np
import pandas as pd

from tabsynth.evaluation.benchmarks.tvae import TVAESynthesizer


def test_tvae_fit_sample_conditional_save_load(tmp_path):
    np.random.seed(42)
    n = 60
    df = pd.DataFrame({
        "num": np.random.normal(5, 1, size=n),
        "cat": np.random.choice(["Red", "Blue"], size=n),
        "target": np.random.choice(["ClassA", "ClassB"], size=n),
    })

    tvae = TVAESynthesizer(
        epochs=3,
        batch_size=32,
        latent_dim=8,
        hidden_dim=32,
        device="cpu",
        seed=42,
    )
    tvae.fit(df, target_column="target")
    assert tvae.is_fitted
    assert tvae.parameter_count > 0

    # Test unconditional sampling
    synth_df = tvae.sample(num_samples=20)
    assert len(synth_df) == 20
    assert set(synth_df.columns) == set(df.columns)

    # Test conditional sampling
    cond_df = tvae.sample_conditional("ClassA", num_samples=10)
    assert len(cond_df) == 10
    assert (cond_df["target"] == "ClassA").all()

    # Test save and load
    save_path = os.path.join(tmp_path, "tvae_test.pt")
    tvae.save(save_path)
    assert os.path.exists(save_path)

    loaded_tvae = TVAESynthesizer(device="cpu").load(save_path)
    assert loaded_tvae.is_fitted
    re_synth = loaded_tvae.sample(num_samples=10)
    assert len(re_synth) == 10
