"""
ProteinScore Exception Hierarchy

Structured exception classes for clear error handling and debugging.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any


class ProteinScoreError(Exception):
    """Base exception for all ProteinScore errors."""

    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        if self.details:
            return f"{self.message} | Details: {self.details}"
        return self.message


class ValidationError(ProteinScoreError):
    """Validation failed for input data."""

    pass


class InvalidSequenceError(ValidationError):
    """Invalid protein sequence provided."""

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
        super().__init__(message, details)
        self.sequence = sequence
        self.invalid_chars = invalid_chars


class InvalidAlleleError(ValidationError):
    """Invalid HLA allele name provided."""

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
        super().__init__(message, details)
        self.allele = allele
        self.suggestions = suggestions


class APIError(ProteinScoreError):
    """Error communicating with the RunLab API."""

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        response_body: str | None = None,
        request_id: str | None = None,
    ) -> None:
        details: dict[str, Any] = {}
        if status_code:
            details["status_code"] = status_code
        if response_body:
            details["response_body"] = response_body[:500]
        if request_id:
            details["request_id"] = request_id
        super().__init__(message, details)
        self.status_code = status_code
        self.response_body = response_body
        self.request_id = request_id


class AuthenticationError(APIError):
    """Authentication failed with the API."""

    def __init__(self, message: str = "Invalid or missing API key") -> None:
        super().__init__(message, status_code=401)


class RateLimitError(APIError):
    """Rate limit exceeded for the API."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: datetime | None = None,
        limit: int | None = None,
        remaining: int | None = None,
    ) -> None:
        details: dict[str, Any] = {}
        if retry_after:
            details["retry_after"] = retry_after.isoformat()
        if limit:
            details["limit"] = limit
        if remaining is not None:
            details["remaining"] = remaining
        super().__init__(message, status_code=429)
        self.details.update(details)
        self.retry_after = retry_after
        self.limit = limit
        self.remaining = remaining


class PredictorError(ProteinScoreError):
    """Error in a local predictor module."""

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
        super().__init__(message, details)
        self.predictor = predictor
        self.original_error = original_error


class ConfigurationError(ProteinScoreError):
    """Invalid configuration provided."""

    pass


class CacheError(ProteinScoreError):
    """Error with caching operations."""

    pass
