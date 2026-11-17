"""Common API schemas."""

from typing import Optional
from pydantic import BaseModel


class ErrorResponse(BaseModel):
    error: str
    detail: str
    request_id: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    mps_available: bool
    cuda_available: bool
    version: str


class ReadyResponse(BaseModel):
    status: str
    database: str
    storage_writable: bool


class VersionResponse(BaseModel):
    version: str
    service: str
    status: str
