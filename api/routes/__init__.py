"""Aggregated routes and top-level router for TabSynth Analytics API."""

from fastapi import APIRouter
from api.routes.health import router as health_router
from api.routes.datasets import router as datasets_router
from api.routes.models import router as models_router
from api.routes.generation import router as generation_router
from api.routes.evaluation import router as evaluation_router
from api.routes.experiments import router as experiments_router
from api.routes.jobs import router as jobs_router

api_v1_router = APIRouter()
api_v1_router.include_router(health_router)
api_v1_router.include_router(datasets_router)
api_v1_router.include_router(models_router)
api_v1_router.include_router(generation_router)
api_v1_router.include_router(evaluation_router)
api_v1_router.include_router(experiments_router)
api_v1_router.include_router(jobs_router)

__all__ = [
    "api_v1_router",
    "health_router",
    "datasets_router",
    "models_router",
    "generation_router",
    "evaluation_router",
    "experiments_router",
    "jobs_router",
]
