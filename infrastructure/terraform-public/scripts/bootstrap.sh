#!/bin/bash
# =============================================================================
# Bootstrap Terraform Backend
# =============================================================================
# Creates S3 bucket and DynamoDB table for Terraform state
# Run this once before first terraform init
# =============================================================================

set -e

AWS_REGION="us-east-1"
BUCKET_NAME="runlab-public-terraform-state"
DYNAMODB_TABLE="terraform-state-lock"
AWS_CLI="/usr/local/bin/aws"

echo "🔧 Bootstrapping Terraform backend..."

# Check AWS credentials
echo "📋 Checking AWS credentials..."
$AWS_CLI sts get-caller-identity

# Create S3 bucket for state
echo "🪣 Creating S3 bucket: $BUCKET_NAME..."
$AWS_CLI s3api create-bucket \
    --bucket "$BUCKET_NAME" \
    --region "$AWS_REGION" \
    2>/dev/null || echo "Bucket may already exist"

# Enable versioning
echo "📚 Enabling versioning..."
$AWS_CLI s3api put-bucket-versioning \
    --bucket "$BUCKET_NAME" \
    --versioning-configuration Status=Enabled

# Enable encryption
echo "🔒 Enabling encryption..."
$AWS_CLI s3api put-bucket-encryption \
    --bucket "$BUCKET_NAME" \
    --server-side-encryption-configuration '{
        "Rules": [{
            "ApplyServerSideEncryptionByDefault": {
                "SSEAlgorithm": "aws:kms"
            },
            "BucketKeyEnabled": true
        }]
    }'

# Block public access
echo "🚫 Blocking public access..."
$AWS_CLI s3api put-public-access-block \
    --bucket "$BUCKET_NAME" \
    --public-access-block-configuration '{
        "BlockPublicAcls": true,
        "IgnorePublicAcls": true,
        "BlockPublicPolicy": true,
        "RestrictPublicBuckets": true
    }'

# Create DynamoDB table for state locking
echo "🔐 Creating DynamoDB table: $DYNAMODB_TABLE..."
$AWS_CLI dynamodb create-table \
    --table-name "$DYNAMODB_TABLE" \
    --attribute-definitions AttributeName=LockID,AttributeType=S \
    --key-schema AttributeName=LockID,KeyType=HASH \
    --billing-mode PAY_PER_REQUEST \
    --region "$AWS_REGION" \
    2>/dev/null || echo "Table may already exist"

echo "✅ Bootstrap complete!"
echo ""
echo "Backend configuration:"
echo "  Bucket: $BUCKET_NAME"
echo "  Region: $AWS_REGION"
echo "  DynamoDB Table: $DYNAMODB_TABLE"
echo ""
echo "You can now run: terraform init"
