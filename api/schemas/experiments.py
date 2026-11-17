"""Schemas for experiment tracking and historical report retrieval."""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class ExperimentDetailResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    dataset_id: Optional[str] = None
    model_id: Optional[str] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[str] = None


class ExperimentListResponse(BaseModel):
    experiments: List[Any]
