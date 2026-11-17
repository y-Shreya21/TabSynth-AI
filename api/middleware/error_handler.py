"""Centralized exception handling preventing internal stack traces in client responses."""

import logging
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger("tabsynth.api")


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "unknown")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTPException",
            "detail": exc.detail,
            "request_id": request_id,
        },
        headers={"X-Request-ID": request_id},
    )


async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "unknown")
    logger.warning(
        f"Validation error for {request.url.path}: {exc.errors()}",
        extra={"request_id": request_id},
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "ValidationError",
            "detail": exc.errors(),
            "request_id": request_id,
        },
        headers={"X-Request-ID": request_id},
    )


async def general_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    request_id = getattr(request.state, "request_id", "unknown")
    logger.exception(
        f"Unhandled server error on {request.url.path}: {exc}",
        extra={"request_id": request_id},
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "detail": "An internal error occurred. Please contact the administrator with Request ID.",
            "request_id": request_id,
        },
        headers={"X-Request-ID": request_id},
    )
