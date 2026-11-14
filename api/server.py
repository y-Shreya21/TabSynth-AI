"""FastAPI entry point and server initialization for TabSynth Analytics."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import router as api_router

app = FastAPI(
    title="TabSynth Analytics API",
    description="High-Fidelity, Class-Conditional & Privacy-Aware Tabular Diffusion Service",
    version="1.0.0",
)

# Enable Cross-Origin Resource Sharing for frontend dashboard integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")


@app.get("/")
async def root():
    return {
        "service": "TabSynth Analytics API",
        "version": "1.0.0",
        "status": "online",
        "docs_url": "/docs",
    }


@app.get("/health")
async def health_check():
    import torch
    return {
        "status": "healthy",
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
    }
