"""
Request Context Middleware

Adds request-scoped context (request_id, timing) to all logs.
Follows Python Backend Best Practices (PBP) Section 5.
"""

from __future__ import annotations

import time
import uuid
from typing import Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from proteinscore.logging import bind_context, clear_context, get_logger

logger = get_logger(__name__)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """
    Middleware that adds request context to all logs.

    Binds the following context variables:
    - request_id: Unique identifier for the request
    - path: Request path
    - method: HTTP method

    Also logs request completion with timing information.

    Example:
        >>> from fastapi import FastAPI
        >>> from proteinscore.middleware import RequestContextMiddleware
        >>>
        >>> app = FastAPI()
        >>> app.add_middleware(RequestContextMiddleware)
    """

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Response]
    ) -> Response:
        """Process the request with context binding."""
        # Generate or extract request ID
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())

        # Bind context for all logs in this request
        bind_context(
            request_id=request_id,
            path=request.url.path,
            method=request.method,
        )

        # Track timing
        start_time = time.perf_counter()

        try:
            response = await call_next(request)

            # Calculate duration
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Log completion
            logger.info(
                "request_completed",
                status_code=response.status_code,
                duration_ms=round(duration_ms, 2),
            )

            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id

            return response

        except Exception as exc:
            duration_ms = (time.perf_counter() - start_time) * 1000
            logger.error(
                "request_failed",
                error=str(exc),
                error_type=type(exc).__name__,
                duration_ms=round(duration_ms, 2),
            )
            raise

        finally:
            # Clear context after request completes
            clear_context()


__all__ = ["RequestContextMiddleware"]
