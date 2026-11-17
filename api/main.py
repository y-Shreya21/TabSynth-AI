"""FastAPI application factory for TabSynth Analytics API."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException
from fastapi.exceptions import RequestValidationError

from api.config import settings
from api.routes import api_v1_router
from api.routes.health import router as root_health_router
from api.middleware import (
    RequestLoggingMiddleware,
    http_exception_handler,
    validation_exception_handler,
    general_exception_handler,
)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    app = FastAPI(
        title=settings.app_name,
        description=settings.app_description,
        version=settings.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # 1. Custom Logging & Request ID Middleware
    app.add_middleware(RequestLoggingMiddleware)

    # 2. CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 3. Centralized Exception Handlers
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
    app.add_exception_handler(Exception, general_exception_handler)

    # 4. Root Service Discovery
    @app.get("/", tags=["Discovery"])
    async def root():
        return {
            "service": settings.app_name,
            "version": settings.app_version,
            "status": "online",
            "docs_url": "/docs",
            "redoc_url": "/redoc",
            "api_prefix": "/api/v1",
        }

    # 5. Mount root health endpoints (/health, /ready, /version)
    app.include_router(root_health_router)

    # 6. Mount API v1 router
    app.include_router(api_v1_router, prefix="/api/v1")

    return app


app = create_app()
