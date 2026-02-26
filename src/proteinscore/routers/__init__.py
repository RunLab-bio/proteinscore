"""
ProteinScore API Routers

FastAPI router definitions for the ProteinScore API.
"""

from proteinscore.routers.health import router as health_router

__all__ = ["health_router"]
