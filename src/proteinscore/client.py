"""
ProteinScore API Client

HTTP client for direct interaction with the RunLab RIP API.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from proteinscore.exceptions import APIError, AuthenticationError, RateLimitError

logger = logging.getLogger(__name__)


class RIPClient:
    """
    HTTP client for the RunLab RIP (Residue Immunogenicity Predictor) API.

    Provides direct access to all RIP API endpoints with automatic
    retry, rate limit handling, and error parsing.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://api.runlab.bio",
        timeout: float = 30.0,
    ) -> None:
        """
        Initialize the RIP API client.

        Args:
            api_key: RunLab API key. Uses anonymous tier if not provided.
            base_url: Base URL for the API.
            timeout: Request timeout in seconds.
        """
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "ProteinScore/0.1.0",
        }
        if api_key:
            headers["X-API-Key"] = api_key

        self._client = httpx.Client(
            base_url=self._base_url,
            timeout=timeout,
            headers=headers,
        )

        # Track rate limit state
        self._rate_limit: int | None = None
        self._rate_remaining: int | None = None
        self._rate_reset: datetime | None = None

    @property
    def is_authenticated(self) -> bool:
        """Check if client has an API key configured."""
        return self._api_key is not None

    @property
    def rate_limit_remaining(self) -> int | None:
        """Get remaining requests in current period."""
        return self._rate_remaining

    def health(self) -> dict[str, Any]:
        """
        Check API health status.

        Returns:
            Health status with version and model info.
        """
        response = self._request("GET", "/rip/health")
        return response

    def list_alleles(
        self,
        prefix: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        """
        List supported HLA alleles.

        Args:
            prefix: Filter by prefix (e.g., "HLA-A*02").
            limit: Maximum results (default: 100, max: 1000).
            offset: Pagination offset.

        Returns:
            Dictionary with allele count and list.
        """
        params = {"limit": limit, "offset": offset}
        if prefix:
            params["prefix"] = prefix

        response = self._request("GET", "/rip/alleles", params=params)
        return response

    def predict(
        self,
        peptide: str,
        allele: str,
    ) -> dict[str, Any]:
        """
        Make a single peptide prediction.

        Args:
            peptide: Peptide sequence (8-15 amino acids).
            allele: HLA allele name.

        Returns:
            Prediction results including binding affinity and rank.
        """
        response = self._request(
            "POST",
            "/rip/predict",
            json={"peptide": peptide, "allele": allele},
        )
        return response

    def predict_batch(
        self,
        peptides: list[str],
        alleles: list[str],
    ) -> dict[str, Any]:
        """
        Make batch peptide predictions.

        Args:
            peptides: List of peptide sequences.
            alleles: List of HLA alleles.
                    If len(alleles) == 1, applies to all peptides.
                    If len(alleles) == len(peptides), 1:1 mapping.
                    Otherwise, Cartesian product.

        Returns:
            Batch prediction results.
        """
        response = self._request(
            "POST",
            "/rip/predict/batch",
            json={"peptides": peptides, "alleles": alleles},
        )
        return response

    def scan(
        self,
        sequence: str,
        alleles: list[str] | None = None,
        peptide_lengths: list[int] | None = None,
        threshold_percentile: float = 2.0,
    ) -> dict[str, Any]:
        """
        Scan a full protein sequence for epitopes.

        Args:
            sequence: Full protein sequence.
            alleles: HLA alleles to scan (default: top 20 by frequency).
            peptide_lengths: Peptide lengths to generate (default: [9]).
            threshold_percentile: Max percentile rank to report (default: 2.0).

        Returns:
            Scan results with epitopes and per-residue risk.
        """
        payload: dict[str, Any] = {
            "sequence": sequence,
            "threshold_percentile": threshold_percentile,
        }
        if alleles:
            payload["alleles"] = alleles
        if peptide_lengths:
            payload["peptide_lengths"] = peptide_lengths

        response = self._request("POST", "/rip/scan", json=payload)
        return response

    def coverage(
        self,
        alleles: list[str],
        population: str = "global",
    ) -> dict[str, Any]:
        """
        Calculate population coverage for a set of alleles.

        Args:
            alleles: List of HLA alleles.
            population: Population identifier
                       (global, european, african, asian, hispanic).

        Returns:
            Coverage percentage for the population.
        """
        response = self._request(
            "POST",
            "/rip/coverage",
            json={"alleles": alleles, "population": population},
        )
        return response

    @retry(
        retry=retry_if_exception_type(httpx.TransportError),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
    )
    def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make an HTTP request with error handling."""
        try:
            response = self._client.request(
                method=method,
                url=path,
                params=params,
                json=json,
            )

            # Update rate limit tracking
            self._update_rate_limits(response)

            # Handle errors
            if response.status_code == 401:
                raise AuthenticationError("Invalid or missing API key")

            if response.status_code == 429:
                raise RateLimitError(
                    "Rate limit exceeded",
                    retry_after=self._rate_reset,
                    limit=self._rate_limit,
                    remaining=self._rate_remaining,
                )

            if response.status_code >= 400:
                error_data = response.json() if response.content else {}
                raise APIError(
                    error_data.get("message", f"API error: {response.status_code}"),
                    status_code=response.status_code,
                    response_body=response.text,
                    request_id=response.headers.get("X-Request-ID"),
                )

            return response.json()

        except httpx.HTTPError as e:
            logger.error("HTTP error: %s", e)
            raise APIError(f"HTTP error: {e}") from e

    def _update_rate_limits(self, response: httpx.Response) -> None:
        """Update rate limit tracking from response headers."""
        if "X-RateLimit-Limit" in response.headers:
            self._rate_limit = int(response.headers["X-RateLimit-Limit"])
        if "X-RateLimit-Remaining" in response.headers:
            self._rate_remaining = int(response.headers["X-RateLimit-Remaining"])
        if "X-RateLimit-Reset" in response.headers:
            try:
                self._rate_reset = datetime.fromisoformat(
                    response.headers["X-RateLimit-Reset"].replace("Z", "+00:00")
                )
            except ValueError:
                pass

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()

    def __enter__(self) -> RIPClient:
        """Context manager entry."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self.close()

    def __repr__(self) -> str:
        return f"RIPClient(authenticated={self.is_authenticated})"
