"""Endpoints for dataset ingestion, profiling, and metadata retrieval."""

import os
import pandas as pd
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from api.schemas.datasets import (
    DatasetUploadResponse,
    TableSchemaResponse,
    DatasetDetailResponse,
    DatasetListResponse,
)
from api.config import settings
from api.services.dataset_service import DatasetService
from api.utils.security import validate_safe_path, validate_csv_upload
from tabsynth.data.schema import infer_table_schema

router = APIRouter(prefix="/datasets", tags=["Datasets"])


@router.post("/upload", response_model=DatasetUploadResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    name: Optional[str] = Form(None),
    target_column: Optional[str] = Form(None),
):
    """Upload a tabular CSV dataset with schema validation and profiling."""
    filename = file.filename or "dataset.csv"
    content = await file.read()
    validate_csv_upload(filename, content, max_size_bytes=settings.max_upload_size_bytes)
    record = DatasetService.upload_dataset(
        file_bytes=content,
        filename=filename,
        name=name,
        target_column=target_column,
    )
    return DatasetUploadResponse(
        dataset_id=record.id,
        name=record.name,
        num_rows=record.num_rows,
        num_columns=record.num_columns,
        target_column=record.target_column,
        file_path=record.file_path,
    )


@router.get("", response_model=DatasetListResponse)
async def list_datasets():
    """List all registered datasets."""
    records = DatasetService.list_datasets()
    return DatasetListResponse(
        datasets=[
            DatasetDetailResponse(
                id=r.id,
                name=r.name,
                file_path=r.file_path,
                num_rows=r.num_rows,
                num_columns=r.num_columns,
                continuous_columns=r.continuous_columns,
                categorical_columns=r.categorical_columns,
                cardinalities=r.cardinalities,
                target_column=r.target_column,
                class_distribution=r.class_distribution,
                missing_values=r.missing_values,
                created_at=r.created_at,
            )
            for r in records
        ]
    )


@router.get("/schema", response_model=TableSchemaResponse)
async def get_dataset_schema_compat(
    data_path: str = "data/raw/adult.csv",
    target_column: Optional[str] = "income",
):
    """Retrieve schema directly from filesystem path (backward compatible)."""
    safe_data_path = validate_safe_path(
        data_path,
        allowed_roots=settings.get_allowed_storage_roots(),
        error_message=f"Access denied: Path '{data_path}' is outside authorized data storage directories.",
    )
    if not os.path.exists(safe_data_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset not found at {data_path}",
        )
    df = pd.read_csv(safe_data_path)
    schema = infer_table_schema(df, target_column=target_column)
    return TableSchemaResponse(
        dataset_name=schema.dataset_name,
        num_rows=schema.num_rows,
        num_columns=schema.num_columns,
        continuous_columns=schema.continuous_columns,
        categorical_columns=schema.categorical_columns,
        cardinalities=schema.cardinalities,
        class_distribution=schema.class_distribution,
    )


@router.get("/{dataset_id}", response_model=DatasetDetailResponse)
async def get_dataset(dataset_id: str):
    """Retrieve detailed dataset profile and sample preview records."""
    record = DatasetService.get_dataset(dataset_id)
    preview = DatasetService.get_dataset_preview(dataset_id, n_rows=10)
    return DatasetDetailResponse(
        id=record.id,
        name=record.name,
        file_path=record.file_path,
        num_rows=record.num_rows,
        num_columns=record.num_columns,
        continuous_columns=record.continuous_columns,
        categorical_columns=record.categorical_columns,
        cardinalities=record.cardinalities,
        target_column=record.target_column,
        class_distribution=record.class_distribution,
        missing_values=record.missing_values,
        preview_records=preview,
        created_at=record.created_at,
    )


@router.get("/{dataset_id}/schema", response_model=TableSchemaResponse)
async def get_dataset_schema(dataset_id: str):
    """Retrieve inferred tabular schema for a registered dataset."""
    record = DatasetService.get_dataset(dataset_id)
    return TableSchemaResponse(
        dataset_name=record.name,
        num_rows=record.num_rows,
        num_columns=record.num_columns,
        continuous_columns=record.continuous_columns,
        categorical_columns=record.categorical_columns,
        cardinalities=record.cardinalities,
        class_distribution=record.class_distribution,
    )
