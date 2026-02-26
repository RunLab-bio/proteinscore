# =============================================================================
# API Gateway for DevScore Public API
# =============================================================================
# HTTP API with Lambda authorizer and proxy integration
# Supports all RIP endpoints: health, alleles, predict, batch, scan, coverage
# =============================================================================

variable "name_prefix" {
  type = string
}

variable "environment" {
  type = string
}

variable "domain_name" {
  type = string
}

variable "lambda_authorizer_arn" {
  type = string
}

variable "lambda_authorizer_invoke_arn" {
  type = string
}

variable "lambda_proxy_arn" {
  type = string
}

variable "lambda_proxy_invoke_arn" {
  type = string
}

variable "enable_custom_domain" {
  type    = bool
  default = false
}

variable "route53_zone_id" {
  type    = string
  default = ""
}

variable "tags" {
  type = map(string)
}

# =============================================================================
# HTTP API
# =============================================================================

resource "aws_apigatewayv2_api" "main" {
  name          = "${var.name_prefix}-api"
  protocol_type = "HTTP"
  description   = "DevScore Public API - RIP Immunogenicity Predictions"

  cors_configuration {
    allow_origins     = ["*"]
    allow_methods     = ["GET", "POST", "OPTIONS"]
    allow_headers     = ["Content-Type", "X-API-Key", "Authorization"]
    expose_headers    = ["X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset", "X-RateLimit-Tier"]
    max_age           = 86400
    allow_credentials = false
  }

  tags = var.tags
}

# =============================================================================
# Lambda Authorizer
# =============================================================================

resource "aws_apigatewayv2_authorizer" "lambda" {
  api_id                            = aws_apigatewayv2_api.main.id
  authorizer_type                   = "REQUEST"
  authorizer_uri                    = var.lambda_authorizer_invoke_arn
  authorizer_payload_format_version = "2.0"
  name                              = "${var.name_prefix}-authorizer"
  enable_simple_responses           = false
  identity_sources                  = ["$request.header.X-API-Key"]

  # Cache authorizer results for 5 minutes
  authorizer_result_ttl_in_seconds = 300
}

# =============================================================================
# Lambda Integration (Proxy)
# =============================================================================

resource "aws_apigatewayv2_integration" "proxy" {
  api_id                 = aws_apigatewayv2_api.main.id
  integration_type       = "AWS_PROXY"
  integration_uri        = var.lambda_proxy_invoke_arn
  integration_method     = "POST"
  payload_format_version = "2.0"
  timeout_milliseconds   = 30000
}

# =============================================================================
# Routes - RIP API Endpoints
# =============================================================================

# Health Check (no auth required)
resource "aws_apigatewayv2_route" "health" {
  api_id    = aws_apigatewayv2_api.main.id
  route_key = "GET /rip/health"
  target    = "integrations/${aws_apigatewayv2_integration.proxy.id}"
  # No authorizer - public endpoint
}

# List Alleles (no auth required)
resource "aws_apigatewayv2_route" "alleles" {
  api_id    = aws_apigatewayv2_api.main.id
  route_key = "GET /rip/alleles"
  target    = "integrations/${aws_apigatewayv2_integration.proxy.id}"
  # No authorizer - public endpoint
}

# Single Prediction (auth + rate limit)
resource "aws_apigatewayv2_route" "predict" {
  api_id             = aws_apigatewayv2_api.main.id
  route_key          = "POST /rip/predict"
  target             = "integrations/${aws_apigatewayv2_integration.proxy.id}"
  authorizer_id      = aws_apigatewayv2_authorizer.lambda.id
  authorization_type = "CUSTOM"
}

# Batch Prediction (auth + rate limit)
resource "aws_apigatewayv2_route" "predict_batch" {
  api_id             = aws_apigatewayv2_api.main.id
  route_key          = "POST /rip/predict/batch"
  target             = "integrations/${aws_apigatewayv2_integration.proxy.id}"
  authorizer_id      = aws_apigatewayv2_authorizer.lambda.id
  authorization_type = "CUSTOM"
}

# Protein Scan (auth + rate limit)
resource "aws_apigatewayv2_route" "scan" {
  api_id             = aws_apigatewayv2_api.main.id
  route_key          = "POST /rip/scan"
  target             = "integrations/${aws_apigatewayv2_integration.proxy.id}"
  authorizer_id      = aws_apigatewayv2_authorizer.lambda.id
  authorization_type = "CUSTOM"
}

# Population Coverage (auth + rate limit)
resource "aws_apigatewayv2_route" "coverage" {
  api_id             = aws_apigatewayv2_api.main.id
  route_key          = "POST /rip/coverage"
  target             = "integrations/${aws_apigatewayv2_integration.proxy.id}"
  authorizer_id      = aws_apigatewayv2_authorizer.lambda.id
  authorization_type = "CUSTOM"
}

# =============================================================================
# Stage
# =============================================================================

resource "aws_apigatewayv2_stage" "main" {
  api_id      = aws_apigatewayv2_api.main.id
  name        = "$default"
  auto_deploy = true

  access_log_settings {
    destination_arn = aws_cloudwatch_log_group.api.arn
    format = jsonencode({
      requestId         = "$context.requestId"
      ip                = "$context.identity.sourceIp"
      requestTime       = "$context.requestTime"
      httpMethod        = "$context.httpMethod"
      routeKey          = "$context.routeKey"
      status            = "$context.status"
      responseLength    = "$context.responseLength"
      integrationError  = "$context.integrationErrorMessage"
      authorizerError   = "$context.authorizer.error"
      tier              = "$context.authorizer.tier"
      rateLimitRemaining = "$context.authorizer.rateLimitRemaining"
    })
  }

  default_route_settings {
    throttling_burst_limit = 1000
    throttling_rate_limit  = 500
  }

  tags = var.tags
}

resource "aws_cloudwatch_log_group" "api" {
  name              = "/aws/apigateway/${var.name_prefix}"
  retention_in_days = 30
  tags              = var.tags
}

# =============================================================================
# Custom Domain (Optional)
# =============================================================================

resource "aws_acm_certificate" "api" {
  count             = var.enable_custom_domain ? 1 : 0
  domain_name       = var.domain_name
  validation_method = "DNS"

  tags = var.tags

  lifecycle {
    create_before_destroy = true
  }
}

resource "aws_route53_record" "cert_validation" {
  for_each = var.enable_custom_domain ? {
    for dvo in aws_acm_certificate.api[0].domain_validation_options : dvo.domain_name => {
      name   = dvo.resource_record_name
      record = dvo.resource_record_value
      type   = dvo.resource_record_type
    }
  } : {}

  allow_overwrite = true
  name            = each.value.name
  records         = [each.value.record]
  ttl             = 60
  type            = each.value.type
  zone_id         = var.route53_zone_id
}

resource "aws_acm_certificate_validation" "api" {
  count                   = var.enable_custom_domain ? 1 : 0
  certificate_arn         = aws_acm_certificate.api[0].arn
  validation_record_fqdns = [for record in aws_route53_record.cert_validation : record.fqdn]
}

resource "aws_apigatewayv2_domain_name" "api" {
  count       = var.enable_custom_domain ? 1 : 0
  domain_name = var.domain_name

  domain_name_configuration {
    certificate_arn = aws_acm_certificate_validation.api[0].certificate_arn
    endpoint_type   = "REGIONAL"
    security_policy = "TLS_1_2"
  }

  tags = var.tags
}

resource "aws_apigatewayv2_api_mapping" "api" {
  count       = var.enable_custom_domain ? 1 : 0
  api_id      = aws_apigatewayv2_api.main.id
  domain_name = aws_apigatewayv2_domain_name.api[0].id
  stage       = aws_apigatewayv2_stage.main.id
}

resource "aws_route53_record" "api" {
  count   = var.enable_custom_domain ? 1 : 0
  name    = var.domain_name
  type    = "A"
  zone_id = var.route53_zone_id

  alias {
    name                   = aws_apigatewayv2_domain_name.api[0].domain_name_configuration[0].target_domain_name
    zone_id                = aws_apigatewayv2_domain_name.api[0].domain_name_configuration[0].hosted_zone_id
    evaluate_target_health = false
  }
}

# =============================================================================
# Outputs
# =============================================================================

output "api_id" {
  value = aws_apigatewayv2_api.main.id
}

output "api_endpoint" {
  value = aws_apigatewayv2_api.main.api_endpoint
}

output "api_domain" {
  value = var.enable_custom_domain ? aws_apigatewayv2_domain_name.api[0].domain_name_configuration[0].target_domain_name : aws_apigatewayv2_api.main.api_endpoint
}

output "execution_arn" {
  value = aws_apigatewayv2_api.main.execution_arn
}

output "stage_id" {
  value = aws_apigatewayv2_stage.main.id
}
