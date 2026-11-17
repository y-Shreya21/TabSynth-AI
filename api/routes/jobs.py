"""Endpoints for background job tracking, progress monitoring, and cancellation."""

from fastapi import APIRouter, HTTPException, status
from api.schemas.jobs import JobStatusResponse, JobListResponse, JobCancelResponse
from api.services.job_service import job_manager

router = APIRouter(prefix="/jobs", tags=["Jobs"])


@router.get("", response_model=JobListResponse)
async def list_jobs(limit: int = 50):
    """List recent background task executions."""
    records = job_manager.list_jobs(limit=limit)
    return JobListResponse(
        jobs=[
            JobStatusResponse(
                job_id=r.id,
                job_type=r.job_type,
                status=r.status,
                progress=r.progress,
                current_step=r.current_step,
                total_steps=r.total_steps,
                metadata=r.metadata,
                result=r.result,
                error=r.error,
                created_at=r.created_at,
                updated_at=r.updated_at,
            )
            for r in records
        ]
    )


@router.get("/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Retrieve execution status, current progress, and outputs for a specific job."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found.",
        )
    return JobStatusResponse(
        job_id=job.id,
        job_type=job.job_type,
        status=job.status,
        progress=job.progress,
        current_step=job.current_step,
        total_steps=job.total_steps,
        metadata=job.metadata,
        result=job.result,
        error=job.error,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


@router.post("/{job_id}/cancel", response_model=JobCancelResponse)
async def cancel_job(job_id: str):
    """Request safe cancellation of an active or queued job."""
    success = job_manager.cancel_job(job_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Job '{job_id}' could not be cancelled or has already completed.",
        )
    return JobCancelResponse(
        job_id=job_id,
        status="cancelled",
        message="Job cancellation requested successfully.",
    )
