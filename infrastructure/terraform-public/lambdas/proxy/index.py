"""
DevScore Public API - Lambda Proxy

Forwards validated requests to the internal RIP API.
Adds rate limit headers and handles upgrade redirects.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urljoin

import requests

# Environment variables
ENVIRONMENT = os.environ.get("ENVIRONMENT", "production")
INTERNAL_API_URL = os.environ.get("INTERNAL_API_URL", "https://api.runlab.bio")

# Timeout for internal API calls
REQUEST_TIMEOUT = 25  # seconds


def get_rate_limit_headers(context: dict[str, Any]) -> dict[str, str]:
    """Build rate limit headers from authorizer context."""
    headers = {}

    rate_limit = context.get("rateLimit", "")
    remaining = context.get("rateLimitRemaining", "")
    tier = context.get("tier", "anonymous")

    if rate_limit and rate_limit != "-1":
        headers["X-RateLimit-Limit"] = str(rate_limit)
        headers["X-RateLimit-Remaining"] = str(remaining)
        # Reset at midnight UTC
        tomorrow = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        headers["X-RateLimit-Reset"] = tomorrow.isoformat() + "Z"

    headers["X-RateLimit-Tier"] = tier

    return headers


def build_error_response(
    status_code: int,
    error: str,
    message: str,
    details: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build standardized error response."""
    body = {
        "error": error,
        "message": message,
    }
    if details:
        body["details"] = details

    response_headers = {
        "Content-Type": "application/json",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Headers": "Content-Type,X-API-Key",
        "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
    }
    if headers:
        response_headers.update(headers)

    return {
        "statusCode": status_code,
        "headers": response_headers,
        "body": json.dumps(body),
    }


def build_upgrade_response(context: dict[str, Any]) -> dict[str, Any]:
    """Build response for rate limit exceeded with upgrade URL."""
    upgrade_url = context.get("upgradeUrl", "https://runlab.bio/upgrade")
    tier = context.get("tier", "anonymous")
    limit = context.get("rateLimit", "100")

    # Reset at midnight UTC
    tomorrow = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )

    return build_error_response(
        status_code=429,
        error="rate_limit_exceeded",
        message=f"Daily rate limit of {limit} requests exceeded for {tier} tier",
        details={
            "tier": tier,
            "limit": int(limit),
            "upgrade_url": upgrade_url,
            "retry_after": tomorrow.isoformat() + "Z",
            "runlab_full_stack": "https://runlab.bio",
            "benefits": [
                "Unlimited API access with Enterprise tier",
                "Full DevScore analysis suite",
                "Custom HLA allele support",
                "Batch processing up to 10,000 peptides",
                "Priority support and SLAs",
            ],
        },
        headers={
            "X-RateLimit-Limit": str(limit),
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Reset": tomorrow.isoformat() + "Z",
            "X-Upgrade-URL": upgrade_url,
            "Retry-After": "3600",  # Suggest retry in 1 hour
        },
    )


def proxy_request(
    method: str,
    path: str,
    headers: dict[str, str],
    body: str | None,
    query_params: dict[str, str] | None,
) -> requests.Response:
    """Proxy request to internal API."""
    url = urljoin(INTERNAL_API_URL, path)

    # Filter headers to forward
    forward_headers = {
        "Content-Type": headers.get("Content-Type", "application/json"),
        "Accept": headers.get("Accept", "application/json"),
    }

    # Add internal auth if needed
    # forward_headers["X-Internal-Auth"] = get_internal_auth_token()

    kwargs: dict[str, Any] = {
        "headers": forward_headers,
        "timeout": REQUEST_TIMEOUT,
    }

    if query_params:
        kwargs["params"] = query_params

    if body and method in ("POST", "PUT", "PATCH"):
        kwargs["data"] = body

    return requests.request(method, url, **kwargs)


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """
    Lambda proxy handler.

    Forwards requests to internal API and adds rate limit headers.
    Supports both HTTP API v2 (payload format 2.0) and REST API formats.
    """
    # Get request context
    request_context = event.get("requestContext", {})

    # Handle both HTTP API v2 and REST API formats
    # HTTP API v2 uses requestContext.http.method, REST uses httpMethod
    http_context = request_context.get("http", {})
    http_method = http_context.get("method") or event.get("httpMethod", "GET")

    # Handle CORS preflight
    if http_method == "OPTIONS":
        return {
            "statusCode": 200,
            "headers": {
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type,X-API-Key",
                "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
                "Access-Control-Max-Age": "86400",
            },
            "body": "",
        }

    # Get authorizer context - for HTTP API v2, authorizer data is in requestContext.authorizer.lambda
    authorizer_context = request_context.get("authorizer", {})
    if "lambda" in authorizer_context:
        authorizer_context = authorizer_context.get("lambda", {})

    # Check if rate limited
    if authorizer_context.get("error") == "rate_limit_exceeded":
        return build_upgrade_response(authorizer_context)

    # Build rate limit headers
    rate_limit_headers = get_rate_limit_headers(authorizer_context)

    # Get request details - HTTP API v2 uses rawPath, REST uses path
    path = event.get("rawPath") or event.get("path", "/")
    headers = event.get("headers", {}) or {}
    body = event.get("body")
    query_params = event.get("queryStringParameters")

    # Validate batch size for batch endpoints
    if "/batch" in path and body:
        try:
            request_body = json.loads(body)
            batch_limit = int(authorizer_context.get("batchLimit", 10))

            peptides = request_body.get("peptides", [])
            alleles = request_body.get("alleles", [])

            if len(peptides) > batch_limit:
                return build_error_response(
                    status_code=400,
                    error="batch_size_exceeded",
                    message=f"Batch size {len(peptides)} exceeds limit {batch_limit} for your tier",
                    details={
                        "batch_limit": batch_limit,
                        "requested": len(peptides),
                        "tier": authorizer_context.get("tier", "anonymous"),
                        "upgrade_url": "https://runlab.bio/upgrade?reason=batch_size",
                    },
                    headers=rate_limit_headers,
                )
        except json.JSONDecodeError:
            return build_error_response(
                status_code=400,
                error="invalid_request",
                message="Invalid JSON in request body",
                headers=rate_limit_headers,
            )

    # Proxy to internal API
    try:
        response = proxy_request(
            method=http_method,
            path=path,
            headers=headers,
            body=body,
            query_params=query_params,
        )

        # Build response headers
        response_headers = {
            "Content-Type": response.headers.get("Content-Type", "application/json"),
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type,X-API-Key",
            "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
        }
        response_headers.update(rate_limit_headers)

        return {
            "statusCode": response.status_code,
            "headers": response_headers,
            "body": response.text,
        }

    except requests.Timeout:
        return build_error_response(
            status_code=504,
            error="gateway_timeout",
            message="Internal API request timed out",
            headers=rate_limit_headers,
        )

    except requests.RequestException as e:
        return build_error_response(
            status_code=502,
            error="bad_gateway",
            message=f"Failed to reach internal API: {str(e)}",
            headers=rate_limit_headers,
        )

    except Exception as e:
        return build_error_response(
            status_code=500,
            error="internal_error",
            message="An unexpected error occurred",
            details={"error": str(e)} if ENVIRONMENT != "production" else None,
            headers=rate_limit_headers,
        )
