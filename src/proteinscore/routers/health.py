"""
Health Check Router

Provides health check endpoints for Kubernetes probes and monitoring.
Follows Python Backend Best Practices (PBP) Section 8.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field

from proteinscore import __version__


router = APIRouter(tags=["health"])


class HealthStatus(BaseModel):
    """Health check response model."""

    status: Literal["healthy", "degraded", "unhealthy"] = Field(
        description="Overall health status"
    )
    version: str = Field(description="Application version")
    timestamp: datetime = Field(description="Current server time")


class ReadinessStatus(BaseModel):
    """Readiness check response model."""

    ready: bool = Field(description="Whether the service is ready to accept traffic")
    checks: dict[str, bool] = Field(description="Individual component checks")


@router.get(
    "/health",
    response_model=HealthStatus,
    summary="Health Check",
    description="Basic health check for liveness probes",
)
async def health_check() -> HealthStatus:
    """
    Health check endpoint for Kubernetes liveness probes.

    Returns basic health status - if this responds, the service is alive.
    """
    return HealthStatus(
        status="healthy",
        version=__version__,
        timestamp=datetime.now(timezone.utc),
    )


@router.get(
    "/health/ready",
    response_model=ReadinessStatus,
    summary="Readiness Check",
    description="Readiness check for Kubernetes readiness probes",
)
async def readiness_check() -> ReadinessStatus:
    """
    Readiness check endpoint for Kubernetes readiness probes.

    Checks all dependencies and returns overall readiness.
    """
    checks: dict[str, bool] = {}

    # Check predictors are importable
    try:
        import proteinscore.predictors  # noqa: F401
        checks["predictors"] = True
    except ImportError:
        checks["predictors"] = False

    # Check configuration
    try:
        from proteinscore.config import get_default_config
        _ = get_default_config()
        checks["config"] = True
    except Exception:
        checks["config"] = False

    # Overall readiness
    ready = all(checks.values())

    return ReadinessStatus(ready=ready, checks=checks)


@router.get(
    "/health/live",
    summary="Liveness Check",
    description="Simple liveness check",
)
async def liveness_check() -> dict[str, str]:
    """
    Simple liveness check - just returns ok.

    Use this for basic "is the service running" checks.
    """
    return {"status": "ok"}


__all__ = ["router"]
