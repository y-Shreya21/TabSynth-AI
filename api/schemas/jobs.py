"""Schemas for background jobs and asynchronous task tracking."""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class JobStatusResponse(BaseModel):
    job_id: str
    job_type: str
    status: str
    progress: float
    current_step: int
    total_steps: int
    metadata: Dict[str, Any] = Field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class JobListResponse(BaseModel):
    jobs: List[JobStatusResponse]


class JobCancelResponse(BaseModel):
    job_id: str
    status: str
    message: str
