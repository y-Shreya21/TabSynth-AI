"""Service handling dataset upload, schema inference, validation, and metadata persistence."""

import os
import io
import re
import uuid
import pandas as pd
from typing import List, Optional, Dict, Any
from fastapi import HTTPException, status
from api.config import settings
from api.db.models import DatasetRecord
from api.db.repositories import DatasetRepository
from tabsynth.data.schema import infer_table_schema


class DatasetService:
    """Encapsulates dataset ingestion, validation rules, and schema profiling."""

    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        """Sanitize filename to prevent directory traversal and special character injection."""
        base = os.path.basename(filename.replace("\\", "/"))
        sanitized = re.sub(r"[^a-zA-Z0-9_\-\.]", "_", base)
        return sanitized or "dataset.csv"

    @classmethod
    def upload_dataset(
        cls,
        file_bytes: bytes,
        filename: str,
        name: Optional[str] = None,
        target_column: Optional[str] = None,
    ) -> DatasetRecord:
        """Validate and ingest an uploaded tabular CSV dataset."""
        # 1. Size validation
        if len(file_bytes) > settings.max_upload_size_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Uploaded file exceeds maximum limit of {settings.max_upload_size_bytes / (1024*1024):.1f}MB.",
            )

        # 2. Non-empty validation
        if len(file_bytes) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file is empty (0 bytes).",
            )

        # 3. Duplicate columns validation in raw header
        lines = [line for line in file_bytes.splitlines() if line.strip()]
        if not lines:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file contains no data rows.",
            )
        header_line = lines[0].decode("utf-8", errors="ignore")
        headers = [h.strip() for h in header_line.split(",")]
        if len(headers) != len(set(headers)):
            duplicates = [col for col in headers if headers.count(col) > 1]
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Dataset contains duplicate column names: {list(set(duplicates))}",
            )

        # 4. CSV parsing validation
        try:
            df = pd.read_csv(io.BytesIO(file_bytes))
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to parse CSV file: {str(exc)}",
            )

        if df.empty or len(df.columns) < 2:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Dataset must contain at least 1 row and 2 columns.",
            )

        # 4. Duplicate columns validation
        if len(df.columns) != len(set(df.columns)):
            duplicates = [col for col in df.columns if list(df.columns).count(col) > 1]
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Dataset contains duplicate column names: {list(set(duplicates))}",
            )

        # 5. Target column validation
        if target_column:
            if target_column not in df.columns:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Target column '{target_column}' does not exist in the uploaded dataset.",
                )

        # 6. Profile schema and statistics
        dataset_id = f"ds_{uuid.uuid4().hex[:10]}"
        safe_filename = cls._sanitize_filename(filename)
        save_path = os.path.join(settings.raw_data_dir, f"{dataset_id}_{safe_filename}")

        # Save to filesystem
        with open(save_path, "wb") as buffer:
            buffer.write(file_bytes)

        schema = infer_table_schema(
            df=df,
            dataset_name=name or safe_filename,
            target_column=target_column,
        )

        missing_dict = df.isnull().sum().to_dict()

        record = DatasetRecord(
            id=dataset_id,
            name=name or safe_filename.replace(".csv", ""),
            file_path=save_path,
            num_rows=len(df),
            num_columns=len(df.columns),
            continuous_columns=schema.continuous_columns,
            categorical_columns=schema.categorical_columns,
            cardinalities=schema.cardinalities,
            target_column=schema.target_column,
            class_distribution=schema.class_distribution,
            missing_values={str(k): int(v) for k, v in missing_dict.items()},
        )

        DatasetRepository.create(record)
        return record

    @staticmethod
    def get_dataset(dataset_id: str) -> DatasetRecord:
        record = DatasetRepository.get_by_id(dataset_id)
        if not record:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset '{dataset_id}' not found.",
            )
        return record

    @staticmethod
    def list_datasets() -> List[DatasetRecord]:
        return DatasetRepository.list_all()

    @staticmethod
    def get_dataset_preview(dataset_id: str, n_rows: int = 10) -> List[Dict[str, Any]]:
        record = DatasetService.get_dataset(dataset_id)
        if not os.path.exists(record.file_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Underlying dataset file '{record.file_path}' was not found.",
            )
        df = pd.read_csv(record.file_path, nrows=n_rows)
        return df.to_dict(orient="records")
