# =============================================================================
# DevScore Public API - Outputs
# =============================================================================

output "api_url" {
  description = "Public API URL"
  value       = "https://${var.domain_name}"
}

output "api_gateway_endpoint" {
  description = "API Gateway endpoint (before CloudFront)"
  value       = module.api_gateway.api_endpoint
}

output "dynamodb_tables" {
  description = "DynamoDB table names"
  value = {
    api_keys       = module.dynamodb.api_keys_table_name
    rate_limits    = module.dynamodb.rate_limits_table_name
    usage_analytics = module.dynamodb.usage_analytics_table_name
  }
}

output "lambda_functions" {
  description = "Lambda function names"
  value = {
    authorizer = module.lambda_authorizer.function_name
    proxy      = module.lambda_proxy.function_name
  }
}

output "cloudfront_distribution_id" {
  description = "CloudFront distribution ID (if WAF enabled)"
  value       = var.enable_waf ? module.cloudfront[0].distribution_id : null
}

output "upgrade_url" {
  description = "URL users are redirected to when rate limited"
  value       = "https://runlab.bio/upgrade?source=api-public&reason=rate_limit"
}

output "rate_limits" {
  description = "Rate limits by tier"
  value = {
    anonymous  = "${var.rate_limit_anonymous}/day, batch: ${var.batch_size_anonymous}"
    free       = "${var.rate_limit_free}/day, batch: ${var.batch_size_free}"
    pro        = "${var.rate_limit_pro}/day, batch: ${var.batch_size_pro}"
    enterprise = "unlimited, batch: 10000"
  }
}
