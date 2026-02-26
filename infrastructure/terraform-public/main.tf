# =============================================================================
# DevScore Public API Infrastructure
# =============================================================================
# This infrastructure provides a public-facing API gateway that:
# 1. Proxies requests to the internal RIP API
# 2. Implements rate limiting per API key tier
# 3. Tracks usage in DynamoDB
# 4. Redirects to runlab.bio when rate limits are exceeded
# =============================================================================

locals {
  name_prefix = "devscore-public"

  rate_limits = {
    anonymous  = var.rate_limit_anonymous
    free       = var.rate_limit_free
    pro        = var.rate_limit_pro
    enterprise = -1 # unlimited
  }

  batch_limits = {
    anonymous  = var.batch_size_anonymous
    free       = var.batch_size_free
    pro        = var.batch_size_pro
    enterprise = 10000
  }

  common_tags = merge(var.tags, {
    Project     = "devscore-public"
    Environment = var.environment
    ManagedBy   = "terraform"
  })
}

# =============================================================================
# Data Sources
# =============================================================================

data "aws_caller_identity" "current" {}

data "aws_route53_zone" "main" {
  count        = var.enable_custom_domain ? 1 : 0
  name         = "runlab.bio"
  private_zone = false
}

# =============================================================================
# DynamoDB Tables
# =============================================================================

module "dynamodb" {
  source = "./modules/dynamodb"

  name_prefix        = local.name_prefix
  environment        = var.environment
  tags               = local.common_tags
}

# =============================================================================
# Lambda Functions
# =============================================================================

module "lambda_authorizer" {
  source = "./modules/lambda-authorizer"

  name_prefix         = local.name_prefix
  environment         = var.environment
  memory_size         = var.lambda_memory_size
  timeout             = 10
  log_retention_days  = var.log_retention_days

  api_keys_table_name    = module.dynamodb.api_keys_table_name
  rate_limits_table_name = module.dynamodb.rate_limits_table_name

  rate_limits = local.rate_limits
  batch_limits = local.batch_limits

  # Business logic: redirect URL when rate limited
  runlab_upgrade_url = "https://runlab.bio/upgrade?source=api-public&reason=rate_limit"

  tags = local.common_tags
}

module "lambda_proxy" {
  source = "./modules/lambda-proxy"

  name_prefix        = local.name_prefix
  environment        = var.environment
  memory_size        = var.lambda_memory_size
  timeout            = var.lambda_timeout
  log_retention_days = var.log_retention_days

  internal_api_url       = var.internal_api_url
  rate_limits_table_name = module.dynamodb.rate_limits_table_name

  tags = local.common_tags
}

# =============================================================================
# API Gateway
# =============================================================================

module "api_gateway" {
  source = "./modules/api-gateway"

  name_prefix  = local.name_prefix
  environment  = var.environment
  domain_name  = var.domain_name

  lambda_authorizer_arn        = module.lambda_authorizer.function_arn
  lambda_authorizer_invoke_arn = module.lambda_authorizer.invoke_arn
  lambda_proxy_arn             = module.lambda_proxy.function_arn
  lambda_proxy_invoke_arn      = module.lambda_proxy.invoke_arn

  enable_custom_domain = var.enable_custom_domain
  route53_zone_id      = var.enable_custom_domain ? data.aws_route53_zone.main[0].zone_id : ""

  tags = local.common_tags
}

# =============================================================================
# CloudFront + WAF (Optional)
# =============================================================================

module "cloudfront" {
  source = "./modules/cloudfront"
  count  = var.enable_waf && var.enable_custom_domain ? 1 : 0

  name_prefix  = local.name_prefix
  environment  = var.environment
  domain_name  = var.domain_name

  api_gateway_domain = module.api_gateway.api_domain
  route53_zone_id    = data.aws_route53_zone.main[0].zone_id

  tags = local.common_tags

  providers = {
    aws           = aws
    aws.us_east_1 = aws.us_east_1
  }
}

# =============================================================================
# Lambda Permissions for API Gateway
# =============================================================================

resource "aws_lambda_permission" "authorizer" {
  statement_id  = "AllowAPIGatewayInvokeAuthorizer"
  action        = "lambda:InvokeFunction"
  function_name = module.lambda_authorizer.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${module.api_gateway.execution_arn}/*"
}

resource "aws_lambda_permission" "proxy" {
  statement_id  = "AllowAPIGatewayInvokeProxy"
  action        = "lambda:InvokeFunction"
  function_name = module.lambda_proxy.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${module.api_gateway.execution_arn}/*"
}

# =============================================================================
# Outputs
# =============================================================================

output "api_endpoint" {
  description = "Public API endpoint"
  value       = var.enable_custom_domain ? "https://${var.domain_name}" : module.api_gateway.api_endpoint
}

output "api_gateway_id" {
  description = "API Gateway ID"
  value       = module.api_gateway.api_id
}

output "api_keys_table" {
  description = "DynamoDB API keys table name"
  value       = module.dynamodb.api_keys_table_name
}

output "rate_limits_table" {
  description = "DynamoDB rate limits table name"
  value       = module.dynamodb.rate_limits_table_name
}

output "upgrade_redirect_url" {
  description = "URL users are redirected to when rate limited"
  value       = "https://runlab.bio/upgrade?source=api-public&reason=rate_limit"
}
