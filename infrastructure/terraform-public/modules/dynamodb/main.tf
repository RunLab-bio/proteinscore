# =============================================================================
# DynamoDB Tables for API Keys and Rate Limiting
# =============================================================================

variable "name_prefix" {
  type = string
}

variable "environment" {
  type = string
}

variable "tags" {
  type = map(string)
}

# =============================================================================
# API Keys Table
# =============================================================================
# Stores API keys and their associated metadata
# Schema:
#   pk (S): API key hash (e.g., "rl_abc123...")
#   tier (S): "anonymous" | "free" | "pro" | "enterprise"
#   owner_email (S): Owner's email
#   organization (S): Organization name
#   created_at (S): ISO timestamp
#   enabled (BOOL): Whether key is active
#   metadata (M): Additional metadata
# =============================================================================

resource "aws_dynamodb_table" "api_keys" {
  name         = "${var.name_prefix}-api-keys"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "pk"

  attribute {
    name = "pk"
    type = "S"
  }

  attribute {
    name = "owner_email"
    type = "S"
  }

  attribute {
    name = "tier"
    type = "S"
  }

  global_secondary_index {
    name            = "owner-email-index"
    hash_key        = "owner_email"
    projection_type = "ALL"
  }

  global_secondary_index {
    name            = "tier-index"
    hash_key        = "tier"
    projection_type = "KEYS_ONLY"
  }

  point_in_time_recovery {
    enabled = true
  }

  server_side_encryption {
    enabled = true
  }

  ttl {
    attribute_name = "expires_at"
    enabled        = true
  }

  tags = merge(var.tags, {
    Name = "${var.name_prefix}-api-keys"
  })
}

# =============================================================================
# Rate Limits Table
# =============================================================================
# Tracks daily API usage per key
# Schema:
#   pk (S): API key hash or "anonymous:{ip}"
#   sk (S): Date in YYYY-MM-DD format
#   request_count (N): Number of requests made
#   last_request (S): ISO timestamp of last request
#   tier (S): Cached tier for quick lookup
#   limit (N): Daily limit for this tier
#   upgrade_shown (BOOL): Whether upgrade prompt was shown
# =============================================================================

resource "aws_dynamodb_table" "rate_limits" {
  name         = "${var.name_prefix}-rate-limits"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "pk"
  range_key    = "sk"

  attribute {
    name = "pk"
    type = "S"
  }

  attribute {
    name = "sk"
    type = "S"
  }

  # TTL to auto-cleanup old records after 7 days
  ttl {
    attribute_name = "ttl"
    enabled        = true
  }

  point_in_time_recovery {
    enabled = true
  }

  server_side_encryption {
    enabled = true
  }

  tags = merge(var.tags, {
    Name = "${var.name_prefix}-rate-limits"
  })
}

# =============================================================================
# Usage Analytics Table
# =============================================================================
# Aggregated usage statistics for analytics
# Schema:
#   pk (S): "daily:{date}" or "monthly:{year-month}"
#   sk (S): API key or "total"
#   total_requests (N): Total request count
#   unique_peptides (N): Unique peptides analyzed
#   alleles_used (SS): Set of alleles used
#   endpoints (M): Map of endpoint -> count
# =============================================================================

resource "aws_dynamodb_table" "usage_analytics" {
  name         = "${var.name_prefix}-usage-analytics"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "pk"
  range_key    = "sk"

  attribute {
    name = "pk"
    type = "S"
  }

  attribute {
    name = "sk"
    type = "S"
  }

  ttl {
    attribute_name = "ttl"
    enabled        = true
  }

  server_side_encryption {
    enabled = true
  }

  tags = merge(var.tags, {
    Name = "${var.name_prefix}-usage-analytics"
  })
}

# =============================================================================
# Outputs
# =============================================================================

output "api_keys_table_name" {
  value = aws_dynamodb_table.api_keys.name
}

output "api_keys_table_arn" {
  value = aws_dynamodb_table.api_keys.arn
}

output "rate_limits_table_name" {
  value = aws_dynamodb_table.rate_limits.name
}

output "rate_limits_table_arn" {
  value = aws_dynamodb_table.rate_limits.arn
}

output "usage_analytics_table_name" {
  value = aws_dynamodb_table.usage_analytics.name
}

output "usage_analytics_table_arn" {
  value = aws_dynamodb_table.usage_analytics.arn
}
