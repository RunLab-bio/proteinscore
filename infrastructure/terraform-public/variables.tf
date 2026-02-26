variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "us-east-1"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "domain_name" {
  description = "Domain name for the public API"
  type        = string
  default     = "api-public.runlab.bio"
}

variable "internal_api_url" {
  description = "Internal RIP API URL to proxy to"
  type        = string
  default     = "https://api.runlab.bio"
}

variable "rate_limit_anonymous" {
  description = "Daily rate limit for anonymous users"
  type        = number
  default     = 100
}

variable "rate_limit_free" {
  description = "Daily rate limit for free tier"
  type        = number
  default     = 1000
}

variable "rate_limit_pro" {
  description = "Daily rate limit for pro tier"
  type        = number
  default     = 50000
}

variable "batch_size_anonymous" {
  description = "Max batch size for anonymous users"
  type        = number
  default     = 10
}

variable "batch_size_free" {
  description = "Max batch size for free tier"
  type        = number
  default     = 100
}

variable "batch_size_pro" {
  description = "Max batch size for pro tier"
  type        = number
  default     = 1000
}

variable "lambda_memory_size" {
  description = "Memory size for Lambda functions"
  type        = number
  default     = 256
}

variable "lambda_timeout" {
  description = "Timeout for Lambda functions in seconds"
  type        = number
  default     = 30
}

variable "log_retention_days" {
  description = "CloudWatch log retention in days"
  type        = number
  default     = 30
}

variable "enable_waf" {
  description = "Enable WAF for CloudFront"
  type        = bool
  default     = false
}

variable "enable_custom_domain" {
  description = "Enable custom domain (requires Route53 hosted zone)"
  type        = bool
  default     = false
}

variable "tags" {
  description = "Additional tags for resources"
  type        = map(string)
  default     = {}
}
