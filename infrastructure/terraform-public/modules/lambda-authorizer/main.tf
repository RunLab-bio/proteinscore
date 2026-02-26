# =============================================================================
# Lambda Authorizer for API Gateway
# =============================================================================
# Validates API keys and enforces rate limits
# Returns policy with rate limit info in context
# Redirects to runlab.bio when limits exceeded
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

variable "api_keys_table_name" {
  type = string
}

variable "rate_limits_table_name" {
  type = string
}

variable "rate_limits" {
  type = map(number)
}

variable "batch_limits" {
  type = map(number)
}

variable "runlab_upgrade_url" {
  type = string
}

variable "tags" {
  type = map(string)
}

# =============================================================================
# IAM Role
# =============================================================================

resource "aws_iam_role" "authorizer" {
  name = "${var.name_prefix}-authorizer-role"

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

resource "aws_iam_role_policy" "authorizer" {
  name = "${var.name_prefix}-authorizer-policy"
  role = aws_iam_role.authorizer.id

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
          "dynamodb:GetItem",
          "dynamodb:Query"
        ]
        Resource = [
          "arn:aws:dynamodb:*:*:table/${var.api_keys_table_name}",
          "arn:aws:dynamodb:*:*:table/${var.api_keys_table_name}/index/*"
        ]
      },
      {
        Effect = "Allow"
        Action = [
          "dynamodb:GetItem",
          "dynamodb:PutItem",
          "dynamodb:UpdateItem"
        ]
        Resource = "arn:aws:dynamodb:*:*:table/${var.rate_limits_table_name}"
      }
    ]
  })
}

# =============================================================================
# Lambda Function
# =============================================================================

data "archive_file" "authorizer" {
  type        = "zip"
  source_dir  = "${path.module}/../../lambdas/authorizer"
  output_path = "${path.module}/../../.build/authorizer.zip"
}

resource "aws_lambda_function" "authorizer" {
  filename         = data.archive_file.authorizer.output_path
  function_name    = "${var.name_prefix}-authorizer"
  role             = aws_iam_role.authorizer.arn
  handler          = "index.handler"
  source_code_hash = data.archive_file.authorizer.output_base64sha256
  runtime          = "python3.11"
  memory_size      = var.memory_size
  timeout          = var.timeout

  environment {
    variables = {
      ENVIRONMENT            = var.environment
      API_KEYS_TABLE         = var.api_keys_table_name
      RATE_LIMITS_TABLE      = var.rate_limits_table_name
      RATE_LIMIT_ANONYMOUS   = var.rate_limits["anonymous"]
      RATE_LIMIT_FREE        = var.rate_limits["free"]
      RATE_LIMIT_PRO         = var.rate_limits["pro"]
      RATE_LIMIT_ENTERPRISE  = var.rate_limits["enterprise"]
      BATCH_LIMIT_ANONYMOUS  = var.batch_limits["anonymous"]
      BATCH_LIMIT_FREE       = var.batch_limits["free"]
      BATCH_LIMIT_PRO        = var.batch_limits["pro"]
      BATCH_LIMIT_ENTERPRISE = var.batch_limits["enterprise"]
      RUNLAB_UPGRADE_URL     = var.runlab_upgrade_url
    }
  }

  tags = var.tags
}

resource "aws_cloudwatch_log_group" "authorizer" {
  name              = "/aws/lambda/${aws_lambda_function.authorizer.function_name}"
  retention_in_days = var.log_retention_days
  tags              = var.tags
}

# =============================================================================
# Outputs
# =============================================================================

output "function_name" {
  value = aws_lambda_function.authorizer.function_name
}

output "function_arn" {
  value = aws_lambda_function.authorizer.arn
}

output "invoke_arn" {
  value = aws_lambda_function.authorizer.invoke_arn
}

output "role_arn" {
  value = aws_iam_role.authorizer.arn
}
