"""Structured request logging and correlation ID middleware."""

import time
import uuid
import logging
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("tabsynth.api")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [ReqID: %(request_id)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware attaching request IDs and logging structured execution metrics."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Extract or generate correlation Request ID
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        request.state.request_id = request_id

        start_time = time.perf_counter()
        
        extra = {"request_id": request_id}
        logger.info(
            f"Incoming {request.method} {request.url.path}",
            extra=extra,
        )

        try:
            response = await call_next(request)
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            response.headers["X-Request-ID"] = request_id
            
            logger.info(
                f"Completed {request.method} {request.url.path} - Status {response.status_code} ({duration_ms:.2f}ms)",
                extra=extra,
            )
            return response
        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error(
                f"Failed {request.method} {request.url.path} after {duration_ms:.2f}ms: {exc}",
                extra=extra,
            )
            raise
