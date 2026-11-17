"""Endpoints for synthetic data sampling and class-guided generation."""

from fastapi import APIRouter
from api.schemas.generation import (
    GenerateRequest,
    ConditionalGenerateRequest,
    GenerateResponse,
)
from api.services.generation_service import GenerationService

router = APIRouter(tags=["Generation"])


@router.post("/generate", response_model=GenerateResponse)
async def generate_samples(request: GenerateRequest):
    """Synthesize tabular records using reverse diffusion sampling and Classifier-Free Guidance."""
    return GenerationService.generate(request)


@router.post("/generate/conditional", response_model=GenerateResponse)
async def generate_conditional_samples(request: ConditionalGenerateRequest):
    """Synthesize minority or specific class tabular records with enhanced guidance."""
    return GenerationService.generate_conditional(request)
