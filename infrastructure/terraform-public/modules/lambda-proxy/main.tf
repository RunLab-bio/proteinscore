# =============================================================================
# Lambda Proxy for Internal RIP API
# =============================================================================
# Forwards validated requests to the internal RIP API
# Adds rate limit headers to responses
# =============================================================================

variable "name_prefix" {
  type = string
}

variable "environment" {
  type = string
}

variable "memory_size" {
  type = number
}

variable "timeout" {
  type = number
}

variable "log_retention_days" {
  type = number
}

variable "internal_api_url" {
  type = string
}

variable "rate_limits_table_name" {
  type = string
}

variable "tags" {
  type = map(string)
}

# =============================================================================
# IAM Role
# =============================================================================

resource "aws_iam_role" "proxy" {
  name = "${var.name_prefix}-proxy-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })

  tags = var.tags
}

resource "aws_iam_role_policy" "proxy" {
  name = "${var.name_prefix}-proxy-policy"
  role = aws_iam_role.proxy.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:*"
      },
      {
        Effect = "Allow"
        Action = [
          "dynamodb:UpdateItem"
        ]
        Resource = "arn:aws:dynamodb:*:*:table/${var.rate_limits_table_name}"
      },
      {
        Effect = "Allow"
        Action = [
          "secretsmanager:GetSecretValue"
        ]
        Resource = "arn:aws:secretsmanager:*:*:secret:devscore/*"
      }
    ]
  })
}

# =============================================================================
# Lambda Layer for Dependencies
# =============================================================================

resource "aws_lambda_layer_version" "requests" {
  filename            = "${path.module}/../../layers/requests.zip"
  layer_name          = "${var.name_prefix}-requests-layer"
  compatible_runtimes = ["python3.11"]
  description         = "Python requests library"

  lifecycle {
    create_before_destroy = true
  }
}

# =============================================================================
# Lambda Function
# =============================================================================

data "archive_file" "proxy" {
  type        = "zip"
  source_dir  = "${path.module}/../../lambdas/proxy"
  output_path = "${path.module}/../../.build/proxy.zip"
}

resource "aws_lambda_function" "proxy" {
  filename         = data.archive_file.proxy.output_path
  function_name    = "${var.name_prefix}-proxy"
  role             = aws_iam_role.proxy.arn
  handler          = "index.handler"
  source_code_hash = data.archive_file.proxy.output_base64sha256
  runtime          = "python3.11"
  memory_size      = var.memory_size
  timeout          = var.timeout

  layers = [aws_lambda_layer_version.requests.arn]

  environment {
    variables = {
      ENVIRONMENT       = var.environment
      INTERNAL_API_URL  = var.internal_api_url
      RATE_LIMITS_TABLE = var.rate_limits_table_name
    }
  }

  tags = var.tags
}

resource "aws_cloudwatch_log_group" "proxy" {
  name              = "/aws/lambda/${aws_lambda_function.proxy.function_name}"
  retention_in_days = var.log_retention_days
  tags              = var.tags
}

# =============================================================================
# Outputs
# =============================================================================

output "function_name" {
  value = aws_lambda_function.proxy.function_name
}

output "function_arn" {
  value = aws_lambda_function.proxy.arn
}

output "invoke_arn" {
  value = aws_lambda_function.proxy.invoke_arn
}

output "role_arn" {
  value = aws_iam_role.proxy.arn
}
