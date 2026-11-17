"""Endpoints for experiment log browsing, benchmark reports, and comparisons."""

from typing import Dict, Any, List
from fastapi import APIRouter
from api.services.experiment_service import ExperimentService

router = APIRouter(prefix="/experiments", tags=["Experiments"])


@router.get("", response_model=Dict[str, List[Any]])
async def list_experiments():
    """List historical experiment evaluations and serialized reports."""
    exps = ExperimentService.list_experiments()
    # Format backward compatible {"experiments": [...]}
    return {"experiments": exps}


@router.get("/{experiment_id}", response_model=Dict[str, Any])
async def get_experiment(experiment_id: str):
    """Retrieve detailed metrics and configuration for an experiment."""
    return ExperimentService.get_experiment(experiment_id)
