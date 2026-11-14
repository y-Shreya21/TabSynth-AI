"""Unit tests for GenerationPipeline DataFrame synthesis."""

import pytest
import numpy as np
import pandas as pd
import torch
from tabsynth.data.schema import infer_table_schema
from tabsynth.data.preprocessor import TabularPreprocessor
from tabsynth.models.hybrid import TabSynthHybridDenoiser
from tabsynth.diffusion.numerical.gaussian_diffusion import GaussianDiffusion
from tabsynth.diffusion.categorical.categorical_diffusion import CategoricalDiffusion
from tabsynth.sampling.generation import GenerationPipeline


def test_generation_pipeline_dataframe_reconstruction():
    """Verify synthetic DataFrame matches original schema, column names, and non-null values."""
    np.random.seed(42)
    n = 60
    df = pd.DataFrame({
        "age": np.random.randint(20, 60, size=n).astype(float),
        "salary": np.random.exponential(scale=50000, size=n),
        "department": np.random.choice(["Engineering", "Sales", "HR"], size=n),
        "promoted": np.random.choice(["Yes", "No"], size=n),
    })

    schema = infer_table_schema(df, target_column="promoted")
    preprocessor = TabularPreprocessor(schema=schema, seed=42).fit(df)

    cardinalities = schema.cardinalities
    num_classes = len(schema.class_distribution)

    model = TabSynthHybridDenoiser(
        num_continuous=len(schema.continuous_columns),
        cardinalities=cardinalities,
        num_classes=num_classes,
        d_model=32,
        transformer_layers=1,
        mlp_layers=1,
    )

    g_diff = GaussianDiffusion(num_timesteps=5)
    c_diff = CategoricalDiffusion(cardinalities=cardinalities, num_timesteps=5)

    pipeline = GenerationPipeline(
        model=model,
        preprocessor=preprocessor,
        gaussian_diffusion=g_diff,
        multinomial_diffusion=c_diff,
        num_timesteps=5,
        device="cpu",
    )

    synth_df = pipeline.generate(num_samples=25, batch_size=10, condition_strategy="balanced")

    assert len(synth_df) == 25
    assert set(synth_df.columns) == set(df.columns)
    assert not synth_df.isnull().any().any()
    assert set(synth_df["department"].unique()).issubset({"Engineering", "Sales", "HR", "<missing>", "<unk>"})
    assert set(synth_df["promoted"].unique()).issubset({"Yes", "No", "unknown"})
