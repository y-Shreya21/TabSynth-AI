"""Health, readiness, and service discovery endpoints."""

import os
import torch
from fastapi import APIRouter
from api.config import settings
from api.schemas.common import HealthResponse, ReadyResponse, VersionResponse
from api.db.session import get_db_connection

router = APIRouter(tags=["Health & Status"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Verify application liveness and accelerator availability."""
    return HealthResponse(
        status="healthy",
        mps_available=bool(torch.backends.mps.is_available()),
        cuda_available=bool(torch.cuda.is_available()),
        version=settings.app_version,
    )


@router.get("/ready", response_model=ReadyResponse)
async def readiness_check():
    """Verify readiness: SQLite connectivity and disk write capabilities."""
    db_ok = "connected"
    try:
        with get_db_connection() as conn:
            conn.execute("SELECT 1").fetchone()
    except Exception as exc:
        db_ok = f"error: {str(exc)}"

    storage_ok = os.access(settings.data_dir, os.W_OK)
    return ReadyResponse(
        status="ready" if db_ok == "connected" and storage_ok else "not_ready",
        database=db_ok,
        storage_writable=storage_ok,
    )


@router.get("/version", response_model=VersionResponse)
async def version_info():
    """Return API version and metadata."""
    return VersionResponse(
        version=settings.app_version,
        service=settings.app_name,
        status="online",
    )
