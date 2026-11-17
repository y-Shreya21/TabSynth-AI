"""Endpoints for scientific evaluation benchmarks across fidelity, utility, and privacy."""

from typing import Dict, Any
from fastapi import APIRouter
from api.schemas.evaluation import (
    FidelityEvalRequest,
    UtilityEvalRequest,
    PrivacyEvalRequest,
)
from api.services.evaluation_service import EvaluationService

router = APIRouter(prefix="/evaluate", tags=["Evaluation"])


@router.post("/fidelity", response_model=Dict[str, Any])
async def evaluate_fidelity(request: FidelityEvalRequest):
    """Compute 1D marginal distances, correlation errors, and C2ST indistinguishability."""
    return EvaluationService.evaluate_fidelity(
        real_data_path=request.real_data_path,
        synthetic_data_path=request.synthetic_data_path,
        target_column=request.target_column,
    )


@router.post("/utility", response_model=Dict[str, Any])
async def evaluate_utility(request: UtilityEvalRequest):
    """Benchmark downstream machine learning utility (TSTR vs TRTR)."""
    return EvaluationService.evaluate_utility(
        real_train_path=request.real_train_path,
        synthetic_train_path=request.synthetic_train_path,
        real_test_path=request.real_test_path,
        target_column=request.target_column,
    )


@router.post("/privacy", response_model=Dict[str, Any])
async def evaluate_privacy(request: PrivacyEvalRequest):
    """Perform Distance-to-Closest-Record and shadow Membership Inference privacy audit."""
    return EvaluationService.evaluate_privacy(
        real_train_path=request.real_train_path,
        real_test_path=request.real_test_path,
        synthetic_path=request.synthetic_path,
        subsample_size=request.subsample_size,
    )
