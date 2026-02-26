"""
ProteinScore Middleware

FastAPI middleware components for request handling, error management, and observability.
Follows Python Backend Best Practices (PBP) Sections 4-6.
"""

from proteinscore.middleware.error_handler import register_exception_handlers
from proteinscore.middleware.request_context import RequestContextMiddleware

__all__ = [
    "register_exception_handlers",
    "RequestContextMiddleware",
]
