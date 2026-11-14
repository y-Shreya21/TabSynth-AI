"""Unit tests for schema detection, zero-leakage preprocessor, and DataLoader pipeline."""

import pytest
import numpy as np
import pandas as pd
from tabsynth.data.schema import infer_table_schema
from tabsynth.data.preprocessor import TabularPreprocessor
from tabsynth.data.dataset import create_stratified_split, build_dataloaders


@pytest.fixture
def mock_raw_dataframe():
    """Generates synthetic heterogeneous tabular DataFrame."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "age": np.random.randint(18, 70, size=n).astype(float),
        "income_num": np.random.exponential(scale=50000, size=n),
        "education": np.random.choice(["Bachelors", "Masters", "PhD", "HS"], size=n),
        "occupation": np.random.choice(["Tech", "Finance", "Healthcare", "Admin", None], size=n),
        "target": np.random.choice(["<=50K", ">50K"], p=[0.75, 0.25], size=n),
    })


def test_schema_inference(mock_raw_dataframe):
    """Verify automatic classification of continuous and categorical columns."""
    schema = infer_table_schema(
        df=mock_raw_dataframe,
        dataset_name="mock_test",
        target_column="target",
    )

    assert schema.num_rows == 200
    assert "age" in schema.continuous_columns
    assert "income_num" in schema.continuous_columns
    assert "education" in schema.categorical_columns
    assert "occupation" in schema.categorical_columns
    assert schema.target_column == "target"
    assert schema.class_distribution is not None


def test_preprocessor_zero_leakage_and_invertibility(mock_raw_dataframe):
    """Verify preprocessor fit/transform isolation and accurate inversion."""
    df_train, df_val, df_test = create_stratified_split(
        df=mock_raw_dataframe,
        target_column="target",
        split_ratios=(0.70, 0.15, 0.15),
        seed=42,
    )

    schema = infer_table_schema(df_train, target_column="target")
    preproc = TabularPreprocessor(schema=schema, continuous_scaler="quantile", seed=42)

    # 1. Fit ONLY on training data
    preproc.fit(df_train)
    assert preproc.is_fitted

    # 2. Transform train and test partitions
    x_num_tr, x_cat_tr, y_tr = preproc.transform(df_train)
    x_num_te, x_cat_te, y_te = preproc.transform(df_test)

    assert x_num_tr.shape == (len(df_train), 2)
    assert len(x_cat_tr) == 2  # education and occupation
    assert len(y_tr) == len(df_train)

    # 3. Test Invertibility on training set
    df_reconstructed = preproc.inverse_transform(x_num_tr, x_cat_tr, y_tr)
    assert len(df_reconstructed) == len(df_train)

    # Check numerical reconstruction error
    np.testing.assert_allclose(
        df_reconstructed["age"].values,
        df_train["age"].values,
        atol=1.0,  # Quantile mapping with integer rounding
    )

    # Check categorical reconstruction (where not missing)
    for col in ["education"]:
        matches = (df_reconstructed[col] == df_train[col]).mean()
        assert matches > 0.99


def test_dataloaders_batching(mock_raw_dataframe):
    """Verify PyTorch DataLoader batch generation and shapes."""
    df_train, df_val, df_test = create_stratified_split(mock_raw_dataframe, target_column="target")
    schema = infer_table_schema(df_train, target_column="target")
    preproc = TabularPreprocessor(schema=schema).fit(df_train)

    train_loader, val_loader, test_loader = build_dataloaders(
        df_train=df_train,
        df_val=df_val,
        df_test=df_test,
        preprocessor=preproc,
        batch_size=32,
    )

    batch = next(iter(train_loader))
    x_num_b, x_cat_b_list, y_b = batch

    assert x_num_b.shape == (32, 2)
    assert len(x_cat_b_list) == 2
    assert x_cat_b_list[0].shape[0] == 32
    assert y_b.shape == (32,)
