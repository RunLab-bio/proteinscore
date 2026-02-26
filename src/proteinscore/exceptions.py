"""
ProteinScore Exception Hierarchy

Structured exception classes for clear error handling and debugging.
Follows Python Backend Best Practices (PBP) Section 3.

Each exception includes:
- HTTP status code for API responses
- Error code for programmatic handling
- Detailed context for debugging
"""

from __future__ import annotations

from datetime import datetime
from typing import Any


class ProteinScoreError(Exception):
    """
    Base exception for all ProteinScore errors.

    Attributes:
        status_code: HTTP status code for API responses
        error_code: Machine-readable error code
        is_operational: Whether this is an expected operational error
    """

    status_code: int = 500
    error_code: str = "INTERNAL_ERROR"
    is_operational: bool = True

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        if error_code is not None:
            self.error_code = error_code
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert to API response format."""
        response: dict[str, Any] = {
            "message": self.message,
            "code": self.error_code,
        }
        if self.details:
            response["details"] = self.details
        return response

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class ValidationError(ProteinScoreError):
    """Validation failed for input data (400)."""

    status_code = 400
    error_code = "VALIDATION_ERROR"

    def __init__(
        self,
        message: str = "Validation error",
        errors: list[dict[str, str]] | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        combined_details = details or {}
        if errors:
            combined_details["errors"] = errors
        super().__init__(message, details=combined_details)


class InvalidSequenceError(ValidationError):
    """Invalid protein sequence provided (400)."""

    error_code = "INVALID_SEQUENCE"
    VALID_AA = set("ACDEFGHIKLMNPQRSTVWY")

    def __init__(
        self,
        message: str,
        sequence: str | None = None,
        invalid_chars: list[str] | None = None,
    ) -> None:
        details: dict[str, Any] = {}
        if sequence:
            details["sequence_length"] = len(sequence)
            details["sequence_preview"] = sequence[:50] + "..." if len(sequence) > 50 else sequence
        if invalid_chars:
            details["invalid_characters"] = invalid_chars
            details["valid_characters"] = "".join(sorted(self.VALID_AA))
        super().__init__(message, details=details)
        self.sequence = sequence
        self.invalid_chars = invalid_chars


class InvalidAlleleError(ValidationError):
    """Invalid HLA allele name provided (400)."""

    error_code = "INVALID_ALLELE"

    def __init__(
        self,
        message: str,
        allele: str | None = None,
        suggestions: list[str] | None = None,
    ) -> None:
        details: dict[str, Any] = {}
        if allele:
            details["allele"] = allele
        if suggestions:
            details["suggestions"] = suggestions[:5]
        super().__init__(message, details=details)
        self.allele = allele
        self.suggestions = suggestions


class APIError(ProteinScoreError):
    """Error communicating with the RunLab API (502/503)."""

    status_code = 502
    error_code = "API_ERROR"
    is_operational = False

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response_body: str | None = None,
        request_id: str | None = None,
    ) -> None:
        details: dict[str, Any] = {}
        if status_code:
            details["upstream_status_code"] = status_code
        if response_body:
            details["response_body"] = response_body[:500]
        if request_id:
            details["request_id"] = request_id
        super().__init__(message, status_code=502, details=details)
        self.upstream_status_code = status_code
        self.response_body = response_body
        self.request_id = request_id


class AuthenticationError(ProteinScoreError):
    """Authentication failed with the API (401)."""

    status_code = 401
    error_code = "UNAUTHORIZED"

    def __init__(self, message: str = "Invalid or missing API key") -> None:
        super().__init__(message)


class RateLimitError(ProteinScoreError):
    """Rate limit exceeded for the API (429)."""

    status_code = 429
    error_code = "RATE_LIMIT_EXCEEDED"

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: datetime | int | None = None,
        limit: int | None = None,
        remaining: int | None = None,
    ) -> None:
        details: dict[str, Any] = {}
        if retry_after:
            if isinstance(retry_after, datetime):
                details["retry_after"] = retry_after.isoformat()
            else:
                details["retry_after"] = retry_after
        if limit:
            details["limit"] = limit
        if remaining is not None:
            details["remaining"] = remaining
        super().__init__(message, details=details)
        self.retry_after = retry_after
        self.limit = limit
        self.remaining = remaining


class PredictorError(ProteinScoreError):
    """Error in a local predictor module (500)."""

    status_code = 500
    error_code = "PREDICTOR_ERROR"

    def __init__(
        self,
        message: str,
        predictor: str,
        original_error: Exception | None = None,
    ) -> None:
        details: dict[str, Any] = {"predictor": predictor}
        if original_error:
            details["original_error"] = str(original_error)
            details["error_type"] = type(original_error).__name__
        super().__init__(message, details=details)
        self.predictor = predictor
        self.original_error = original_error


class ConfigurationError(ProteinScoreError):
    """Invalid configuration provided (500)."""

    status_code = 500
    error_code = "CONFIGURATION_ERROR"


class CacheError(ProteinScoreError):
    """Error with caching operations (500)."""

    status_code = 500
    error_code = "CACHE_ERROR"


class NotFoundError(ProteinScoreError):
    """Resource not found (404)."""

    status_code = 404
    error_code = "NOT_FOUND"

    def __init__(self, resource: str = "Resource", identifier: str | None = None) -> None:
        message = f"{resource} not found"
        if identifier:
            message = f"{resource} '{identifier}' not found"
        super().__init__(message)


class ConflictError(ProteinScoreError):
    """Resource conflict (409)."""

    status_code = 409
    error_code = "CONFLICT"

    def __init__(self, message: str) -> None:
        super().__init__(message)


class ServiceUnavailableError(ProteinScoreError):
    """External service unavailable (503)."""

    status_code = 503
    error_code = "SERVICE_UNAVAILABLE"
    is_operational = False

    def __init__(self, service: str, message: str | None = None) -> None:
        msg = message or f"Service '{service}' is temporarily unavailable"
        super().__init__(msg, details={"service": service})
