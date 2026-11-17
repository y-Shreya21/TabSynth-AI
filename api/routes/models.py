"""Endpoints for model training initiation, registry inspection, and job status polling."""

from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, status
from api.schemas.models import (
    TrainRequest,
    TrainStatusResponse,
    ModelDetailResponse,
    ModelListResponse,
)
from api.services.model_service import ModelService
from api.services.job_service import job_manager

router = APIRouter(prefix="/models", tags=["Models"])


@router.post("/train", response_model=Dict[str, str])
async def train_model(request: TrainRequest):
    """Enqueue an asynchronous model training job."""
    job_id = ModelService.start_training_job(request)
    return {"job_id": job_id, "status": "queued"}


@router.get("", response_model=ModelListResponse)
async def list_models():
    """List all trained model checkpoints and configurations."""
    records = ModelService.list_models()
    return ModelListResponse(
        models=[
            ModelDetailResponse(
                id=r.id,
                name=r.name,
                model_type=r.model_type,
                dataset_id=r.dataset_id,
                target_column=r.target_column,
                checkpoint_path=r.checkpoint_path,
                hyperparameters=r.hyperparameters,
                metrics=r.metrics,
                created_at=r.created_at,
            )
            for r in records
        ]
    )


@router.get("/status", response_model=TrainStatusResponse)
async def get_training_status_compat(job_id: str):
    """Retrieve training job status by query parameter (backward compatible)."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job ID {job_id} not found",
        )
    extra = job.result or {}
    return TrainStatusResponse(
        job_id=job.id,
        status=job.status,
        current_epoch=job.current_step,
        total_epochs=job.total_steps,
        latest_train_loss=extra.get("train_loss"),
        latest_val_loss=extra.get("val_loss"),
        best_checkpoint_path=extra.get("checkpoint_path"),
    )


@router.get("/{model_id}", response_model=ModelDetailResponse)
async def get_model(model_id: str):
    """Retrieve specific model checkpoint metadata and hyperparameters."""
    r = ModelService.get_model(model_id)
    return ModelDetailResponse(
        id=r.id,
        name=r.name,
        model_type=r.model_type,
        dataset_id=r.dataset_id,
        target_column=r.target_column,
        checkpoint_path=r.checkpoint_path,
        hyperparameters=r.hyperparameters,
        metrics=r.metrics,
        created_at=r.created_at,
    )


@router.get("/{model_id}/status", response_model=TrainStatusResponse)
async def get_model_status(model_id: str):
    """Retrieve status of training job associated with model."""
    job = job_manager.get_job(model_id)
    if not job:
        # Try checking if model exists
        model = ModelService.get_model(model_id)
        return TrainStatusResponse(
            job_id=model.id,
            status="completed",
            current_epoch=model.hyperparameters.get("epochs", 30),
            total_epochs=model.hyperparameters.get("epochs", 30),
            latest_train_loss=None,
            latest_val_loss=model.metrics.get("best_val_loss") if model.metrics else None,
            best_checkpoint_path=model.checkpoint_path,
        )

    extra = job.result or {}
    return TrainStatusResponse(
        job_id=job.id,
        status=job.status,
        current_epoch=job.current_step,
        total_epochs=job.total_steps,
        latest_train_loss=extra.get("train_loss"),
        latest_val_loss=extra.get("val_loss"),
        best_checkpoint_path=extra.get("checkpoint_path"),
    )
