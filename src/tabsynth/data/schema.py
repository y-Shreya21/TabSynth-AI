"""Schema detection, column classification, and metadata validation."""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class ColumnMetadata(BaseModel):
    name: str
    dtype: str
    is_continuous: bool
    is_target: bool = False
    cardinality: Optional[int] = None
    categories: Optional[List[str]] = None
    missing_count: int = 0
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    median_val: Optional[float] = None


class TableSchema(BaseModel):
    dataset_name: str
    num_rows: int
    num_columns: int
    target_column: Optional[str] = None
    continuous_columns: List[str] = Field(default_factory=list)
    categorical_columns: List[str] = Field(default_factory=list)
    cardinalities: List[int] = Field(default_factory=list)
    columns_metadata: Dict[str, ColumnMetadata] = Field(default_factory=dict)
    class_distribution: Optional[Dict[str, float]] = None


def infer_table_schema(
    df: pd.DataFrame,
    dataset_name: str = "tabular_dataset",
    target_column: Optional[str] = None,
    categorical_threshold: int = 20,
    forced_categorical: Optional[List[str]] = None,
    forced_continuous: Optional[List[str]] = None,
) -> TableSchema:
    """Automatically detects feature types, categories, and numerical distributions."""
    forced_cat = set(forced_categorical or [])
    forced_cont = set(forced_continuous or [])

    continuous_cols: List[str] = []
    categorical_cols: List[str] = []
    cardinalities: List[int] = []
    cols_meta: Dict[str, ColumnMetadata] = {}

    for col in df.columns:
        if col == target_column:
            continue

        series = df[col]
        missing_cnt = int(series.isna().sum())
        num_uniques = series.nunique(dropna=True)

        is_numeric = pd.api.types.is_numeric_dtype(series)

        if col in forced_cat:
            is_cont = False
        elif col in forced_cont:
            is_cont = True
        elif not is_numeric:
            is_cont = False
        elif num_uniques <= categorical_threshold:
            is_cont = False
        else:
            is_cont = True

        if is_cont:
            continuous_cols.append(col)
            valid_vals = series.dropna()
            cols_meta[col] = ColumnMetadata(
                name=col,
                dtype=str(series.dtype),
                is_continuous=True,
                missing_count=missing_cnt,
                min_val=float(valid_vals.min()) if len(valid_vals) > 0 else 0.0,
                max_val=float(valid_vals.max()) if len(valid_vals) > 0 else 0.0,
                median_val=float(valid_vals.median()) if len(valid_vals) > 0 else 0.0,
            )
        else:
            categorical_cols.append(col)
            unique_cats = [str(x) for x in series.dropna().unique().tolist()]
            # +1 for <missing> category if missing values exist
            card = len(unique_cats) + (1 if missing_cnt > 0 else 0)
            cardinalities.append(card)
            cols_meta[col] = ColumnMetadata(
                name=col,
                dtype="categorical",
                is_continuous=False,
                cardinality=card,
                categories=sorted(unique_cats),
                missing_count=missing_cnt,
            )

    # Process target column if provided
    class_dist = None
    if target_column is not None and target_column in df.columns:
        target_series = df[target_column].astype(str)
        counts = target_series.value_counts(normalize=True)
        class_dist = {str(k): float(v) for k, v in counts.items()}
        cols_meta[target_column] = ColumnMetadata(
            name=target_column,
            dtype="target",
            is_continuous=False,
            is_target=True,
            cardinality=len(counts),
            categories=sorted(counts.index.tolist()),
            missing_count=int(df[target_column].isna().sum()),
        )

    return TableSchema(
        dataset_name=dataset_name,
        num_rows=len(df),
        num_columns=len(df.columns),
        target_column=target_column,
        continuous_columns=continuous_cols,
        categorical_columns=categorical_cols,
        cardinalities=cardinalities,
        columns_metadata=cols_meta,
        class_distribution=class_dist,
    )
