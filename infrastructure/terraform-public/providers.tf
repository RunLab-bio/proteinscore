terraform {
  required_version = ">= 1.5.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.5"
    }
    archive = {
      source  = "hashicorp/archive"
      version = "~> 2.4"
    }
  }

  backend "s3" {
    bucket         = "devscore-public-terraform-state"
    key            = "devscore-api/terraform.tfstate"
    region         = "us-east-1"
    encrypt        = true
    dynamodb_table = "devscore-terraform-state-lock"
    profile        = "runlab-public"
  }
}

provider "aws" {
  region  = var.aws_region
  profile = "runlab-public"

  default_tags {
    tags = {
      Project     = "devscore"
      Environment = var.environment
      ManagedBy   = "terraform"
      Owner       = "runlab"
    }
  }
}

# Provider for CloudFront (must be us-east-1 for ACM certs)
provider "aws" {
  alias   = "us_east_1"
  region  = "us-east-1"
  profile = "runlab-public"

  default_tags {
    tags = {
      Project     = "devscore"
      Environment = var.environment
      ManagedBy   = "terraform"
      Owner       = "runlab"
    }
  }
}
