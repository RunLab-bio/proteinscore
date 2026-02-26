"""
DevScore Public API - Lambda Authorizer

Validates API keys and enforces rate limits.
When rate limit is exceeded, returns upgrade URL to runlab.bio.
"""

from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any

import boto3
from botocore.exceptions import ClientError

# Environment variables
ENVIRONMENT = os.environ.get("ENVIRONMENT", "production")
API_KEYS_TABLE = os.environ["API_KEYS_TABLE"]
RATE_LIMITS_TABLE = os.environ["RATE_LIMITS_TABLE"]
RUNLAB_UPGRADE_URL = os.environ.get(
    "RUNLAB_UPGRADE_URL",
    "https://runlab.bio/upgrade?source=api-public&reason=rate_limit"
)

# Rate limits by tier
RATE_LIMITS = {
    "anonymous": int(os.environ.get("RATE_LIMIT_ANONYMOUS", "100")),
    "free": int(os.environ.get("RATE_LIMIT_FREE", "1000")),
    "pro": int(os.environ.get("RATE_LIMIT_PRO", "50000")),
    "enterprise": -1,  # unlimited
}

BATCH_LIMITS = {
    "anonymous": int(os.environ.get("BATCH_LIMIT_ANONYMOUS", "10")),
    "free": int(os.environ.get("BATCH_LIMIT_FREE", "100")),
    "pro": int(os.environ.get("BATCH_LIMIT_PRO", "1000")),
    "enterprise": 10000,
}

# DynamoDB client
dynamodb = boto3.resource("dynamodb")
api_keys_table = dynamodb.Table(API_KEYS_TABLE)
rate_limits_table = dynamodb.Table(RATE_LIMITS_TABLE)


def hash_api_key(api_key: str) -> str:
    """Hash API key for secure storage lookup."""
    return hashlib.sha256(api_key.encode()).hexdigest()[:32]


def get_client_ip(event: dict[str, Any]) -> str:
    """Extract client IP from request context."""
    request_context = event.get("requestContext", {})
    identity = request_context.get("identity", {})
    return identity.get("sourceIp", "unknown")


def get_api_key(event: dict[str, Any]) -> str | None:
    """Extract API key from headers."""
    headers = event.get("headers", {}) or {}
    # Check both cases for API Gateway
    return headers.get("X-API-Key") or headers.get("x-api-key")


def lookup_api_key(api_key: str) -> dict[str, Any] | None:
    """Look up API key in DynamoDB."""
    key_hash = hash_api_key(api_key)
    try:
        response = api_keys_table.get_item(Key={"pk": key_hash})
        item = response.get("Item")
        if item and item.get("enabled", True):
            return item
        return None
    except ClientError:
        return None


def check_rate_limit(
    identifier: str,
    tier: str,
) -> tuple[bool, int, int, str | None]:
    """
    Check and update rate limit for identifier.

    Returns:
        tuple: (allowed, remaining, limit, upgrade_url or None)
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    limit = RATE_LIMITS.get(tier, RATE_LIMITS["anonymous"])

    # Enterprise has unlimited access
    if limit == -1:
        return True, -1, -1, None

    try:
        # Atomic increment with conditional check
        response = rate_limits_table.update_item(
            Key={"pk": identifier, "sk": today},
            UpdateExpression=(
                "SET request_count = if_not_exists(request_count, :zero) + :inc, "
                "last_request = :now, "
                "tier = :tier, "
                "#limit = :limit, "
                "ttl = :ttl"
            ),
            ExpressionAttributeNames={"#limit": "limit"},
            ExpressionAttributeValues={
                ":zero": 0,
                ":inc": 1,
                ":now": datetime.now(timezone.utc).isoformat(),
                ":tier": tier,
                ":limit": limit,
                ":ttl": int(datetime.now(timezone.utc).timestamp()) + (7 * 24 * 60 * 60),
            },
            ReturnValues="ALL_NEW",
        )

        current_count = int(response["Attributes"]["request_count"])
        remaining = max(0, limit - current_count)

        if current_count > limit:
            # Rate limit exceeded - return upgrade URL
            upgrade_url = f"{RUNLAB_UPGRADE_URL}&tier={tier}&usage={current_count}"
            return False, 0, limit, upgrade_url

        return True, remaining, limit, None

    except ClientError as e:
        # On error, allow request but log
        print(f"Rate limit check error: {e}")
        return True, limit, limit, None


def generate_policy(
    principal_id: str,
    effect: str,
    resource: str,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate IAM policy for API Gateway."""
    policy = {
        "principalId": principal_id,
        "policyDocument": {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Action": "execute-api:Invoke",
                    "Effect": effect,
                    "Resource": resource,
                }
            ],
        },
    }

    if context:
        # Convert all values to strings for API Gateway context
        policy["context"] = {k: str(v) if not isinstance(v, str) else v for k, v in context.items()}

    return policy


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """
    Lambda authorizer handler.

    Validates API key and rate limits.
    Returns IAM policy with usage context.
    """
    method_arn = event.get("methodArn", "*")

    # Extract API key
    api_key = get_api_key(event)
    client_ip = get_client_ip(event)

    # Determine tier and identifier
    if api_key and api_key.startswith("rl_"):
        # Authenticated request
        key_info = lookup_api_key(api_key)
        if not key_info:
            # Invalid API key
            return generate_policy("unauthorized", "Deny", method_arn, {
                "error": "invalid_api_key",
                "message": "Invalid or disabled API key",
            })

        tier = key_info.get("tier", "free")
        identifier = hash_api_key(api_key)
        principal_id = key_info.get("owner_email", identifier)
    else:
        # Anonymous request
        tier = "anonymous"
        identifier = f"anonymous:{client_ip}"
        principal_id = f"anonymous-{client_ip}"

    # Check rate limit
    allowed, remaining, limit, upgrade_url = check_rate_limit(identifier, tier)

    # Build context for downstream Lambda
    auth_context = {
        "tier": tier,
        "identifier": identifier,
        "rateLimit": limit,
        "rateLimitRemaining": remaining,
        "batchLimit": BATCH_LIMITS.get(tier, BATCH_LIMITS["anonymous"]),
    }

    if not allowed:
        # Rate limit exceeded
        auth_context["error"] = "rate_limit_exceeded"
        auth_context["upgradeUrl"] = upgrade_url
        auth_context["message"] = (
            f"Daily rate limit of {limit} requests exceeded. "
            f"Upgrade at {upgrade_url}"
        )
        return generate_policy(principal_id, "Deny", method_arn, auth_context)

    # Request allowed
    return generate_policy(principal_id, "Allow", method_arn, auth_context)
