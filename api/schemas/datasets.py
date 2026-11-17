"""Schemas for dataset management and profiling."""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class DatasetUploadResponse(BaseModel):
    dataset_id: str
    name: str
    num_rows: int
    num_columns: int
    target_column: Optional[str] = None
    file_path: str


class TableSchemaResponse(BaseModel):
    dataset_name: str
    num_rows: int
    num_columns: int
    continuous_columns: List[str]
    categorical_columns: List[str]
    cardinalities: List[int]
    class_distribution: Optional[Dict[str, float]] = None


class DatasetDetailResponse(BaseModel):
    id: str
    name: str
    file_path: str
    num_rows: int
    num_columns: int
    continuous_columns: List[str] = Field(default_factory=list)
    categorical_columns: List[str] = Field(default_factory=list)
    cardinalities: List[int] = Field(default_factory=list)
    target_column: Optional[str] = None
    class_distribution: Optional[Dict[str, float]] = None
    missing_values: Optional[Dict[str, int]] = None
    duplicate_rows: Optional[int] = 0
    preview_records: Optional[List[Dict[str, Any]]] = None
    created_at: Optional[str] = None


class DatasetListResponse(BaseModel):
    datasets: List[DatasetDetailResponse]
