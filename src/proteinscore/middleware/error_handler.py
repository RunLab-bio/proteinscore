"""
Global Exception Handlers for FastAPI

Centralized error handling with structured logging and consistent API responses.
Follows Python Backend Best Practices (PBP) Section 4.
"""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError as PydanticValidationError

from proteinscore.exceptions import ProteinScoreError
from proteinscore.logging import get_logger

logger = get_logger(__name__)


def register_exception_handlers(app: FastAPI) -> None:
    """
    Register all exception handlers on the FastAPI app.

    Args:
        app: FastAPI application instance

    Example:
        >>> from fastapi import FastAPI
        >>> from proteinscore.middleware import register_exception_handlers
        >>>
        >>> app = FastAPI()
        >>> register_exception_handlers(app)
    """

    @app.exception_handler(ProteinScoreError)
    async def proteinscore_error_handler(
        request: Request, exc: ProteinScoreError
    ) -> JSONResponse:
        """Handle ProteinScore-specific errors."""
        log_level = "warning" if exc.is_operational else "error"
        getattr(logger, log_level)(
            "proteinscore_error",
            error_code=exc.error_code,
            message=exc.message,
            path=request.url.path,
            method=request.method,
            details=exc.details,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content=exc.to_dict(),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """Handle Pydantic request validation errors."""
        errors = [
            {
                "field": ".".join(str(loc) for loc in err["loc"]),
                "message": err["msg"],
            }
            for err in exc.errors()
        ]
        logger.warning(
            "validation_error",
            path=request.url.path,
            method=request.method,
            errors=errors,
        )
        return JSONResponse(
            status_code=400,
            content={
                "message": "Validation error",
                "code": "VALIDATION_ERROR",
                "details": {"errors": errors},
            },
        )

    @app.exception_handler(PydanticValidationError)
    async def pydantic_error_handler(
        request: Request, exc: PydanticValidationError
    ) -> JSONResponse:
        """Handle Pydantic model validation errors."""
        errors = [
            {
                "field": ".".join(str(loc) for loc in err["loc"]),
                "message": err["msg"],
            }
            for err in exc.errors()
        ]
        logger.warning(
            "pydantic_validation_error",
            path=request.url.path,
            errors=errors,
        )
        return JSONResponse(
            status_code=400,
            content={
                "message": "Validation error",
                "code": "VALIDATION_ERROR",
                "details": {"errors": errors},
            },
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Handle unexpected exceptions."""
        logger.exception(
            "unhandled_exception",
            path=request.url.path,
            method=request.method,
            error_type=type(exc).__name__,
            error=str(exc),
        )
        return JSONResponse(
            status_code=500,
            content={
                "message": "An unexpected error occurred",
                "code": "INTERNAL_ERROR",
            },
        )


__all__ = ["register_exception_handlers"]
