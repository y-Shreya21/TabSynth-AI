"""Schemas for synthetic data generation requests and responses."""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    checkpoint_path: Optional[str] = None
    model_id: Optional[str] = None
    num_samples: int = Field(default=1000, gt=0)
    condition_strategy: str = "balanced"  # "balanced", "natural", "class_specific", "unconditional"
    specific_class: Optional[int] = None
    cfg_scale: float = 1.0
    steps: int = 100
    output_path: Optional[str] = "data/synthetic/generated_records.csv"


class ConditionalGenerateRequest(BaseModel):
    checkpoint_path: Optional[str] = None
    model_id: Optional[str] = None
    target_class: int = 1
    num_samples: int = Field(default=500, gt=0)
    cfg_scale: float = 1.5
    steps: int = 100
    output_path: Optional[str] = None


class GenerateResponse(BaseModel):
    status: str
    num_samples: int
    output_path: str
    preview_records: List[Dict[str, Any]]
    job_id: Optional[str] = None
