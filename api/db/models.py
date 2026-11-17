"""Data classes and Pydantic models for database entities."""

import json
from datetime import datetime
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class DatasetRecord(BaseModel):
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
    created_at: Optional[str] = None


class ModelRecord(BaseModel):
    id: str
    name: str
    model_type: str
    dataset_id: Optional[str] = None
    target_column: Optional[str] = None
    checkpoint_path: str
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    metrics: Optional[Dict[str, Any]] = None
    created_at: Optional[str] = None


class JobRecord(BaseModel):
    id: str
    job_type: str  # "training", "generation", "evaluation"
    status: str    # "queued", "running", "completed", "failed", "cancelled"
    progress: float = 0.0
    current_step: int = 0
    total_steps: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class ExperimentRecord(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    dataset_id: Optional[str] = None
    model_id: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[str] = None


class ArtifactRecord(BaseModel):
    id: str
    job_id: Optional[str] = None
    artifact_type: str  # "synthetic_csv", "checkpoint", "report"
    file_path: str
    created_at: Optional[str] = None
