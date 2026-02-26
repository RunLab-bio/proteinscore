# ProteinScore Public API Infrastructure

Terraform infrastructure for the ProteinScore Public API at `https://api-public.runlab.bio`.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Internet                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CloudFront + WAF (Optional)                          │
│  • DDoS Protection                                                          │
│  • IP Reputation List                                                       │
│  • Rate Limiting (10k/5min)                                                │
│  • Caching (health, alleles)                                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         API Gateway (HTTP API)                              │
│  • Custom Domain: api-public.runlab.bio                                     │
│  • TLS 1.2+                                                                 │
│  • Access Logging                                                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
┌─────────────────────────────┐   ┌─────────────────────────────────────────┐
│   Lambda Authorizer         │   │   Lambda Proxy                          │
│  • API Key Validation       │──▶│  • Forward to Internal RIP API          │
│  • Rate Limit Checking      │   │  • Add Rate Limit Headers               │
│  • Tier Detection           │   │  • Handle Errors                        │
│  • Upgrade URL on Limit     │   │  • Batch Size Validation                │
└─────────────────────────────┘   └─────────────────────────────────────────┘
            │                                       │
            ▼                                       ▼
┌─────────────────────────────┐   ┌─────────────────────────────────────────┐
│   DynamoDB                  │   │   Internal RIP API                      │
│  • api_keys table           │   │   https://api.runlab.bio                │
│  • rate_limits table        │   │   (ideanova account)                    │
│  • usage_analytics table    │   └─────────────────────────────────────────┘
└─────────────────────────────┘
```

## Rate Limiting & Upgrade Flow

When a user exceeds their rate limit, they receive a 429 response with:

```json
{
  "error": "rate_limit_exceeded",
  "message": "Daily rate limit of 100 requests exceeded for anonymous tier",
  "details": {
    "tier": "anonymous",
    "limit": 100,
    "upgrade_url": "https://runlab.bio/upgrade?source=api-public&reason=rate_limit&tier=anonymous",
    "runlab_full_stack": "https://runlab.bio",
    "benefits": [
      "Unlimited API access with Enterprise tier",
      "Full ProteinScore analysis suite",
      "Custom HLA allele support",
      "Batch processing up to 10,000 peptides",
      "Priority support and SLAs"
    ]
  }
}
```

## Tiers

| Tier       | Requests/Day | Batch Size | Notes                    |
|------------|--------------|------------|--------------------------|
| Anonymous  | 100          | 10         | No API key required      |
| Free       | 1,000        | 100        | Requires registration    |
| Pro        | 50,000       | 1,000      | Paid tier                |
| Enterprise | Unlimited    | 10,000     | Custom contracts         |

## Endpoints

| Method | Path               | Auth Required | Description              |
|--------|--------------------|--------------:|--------------------------|
| GET    | /rip/health        | No            | Health check             |
| GET    | /rip/alleles       | No            | List supported alleles   |
| POST   | /rip/predict       | Yes*          | Single prediction        |
| POST   | /rip/predict/batch | Yes*          | Batch predictions        |
| POST   | /rip/scan          | Yes*          | Protein epitope scan     |
| POST   | /rip/coverage      | Yes*          | Population coverage      |

*Anonymous access allowed with reduced rate limits

## Deployment

### Prerequisites

1. AWS CLI configured with runlab-public account credentials
2. Terraform >= 1.5.0
3. Python 3.11+ (for building layers)
4. Route53 hosted zone for `runlab.bio`

### Build Lambda Layers

```bash
chmod +x scripts/build-layers.sh
./scripts/build-layers.sh
```

### Initialize Terraform

```bash
terraform init
```

### Plan Deployment

```bash
terraform plan -var="environment=production"
```

### Apply

```bash
terraform apply -var="environment=production"
```

### Destroy

```bash
terraform destroy -var="environment=production"
```

## Configuration

### Variables

| Variable              | Default              | Description                        |
|-----------------------|----------------------|------------------------------------|
| aws_region            | us-east-1            | AWS region                         |
| environment           | production           | Environment name                   |
| domain_name           | api-public.runlab.bio| Public API domain                  |
| internal_api_url      | https://api.runlab.bio| Internal RIP API                  |
| rate_limit_anonymous  | 100                  | Anonymous daily limit              |
| rate_limit_free       | 1000                 | Free tier daily limit              |
| rate_limit_pro        | 50000                | Pro tier daily limit               |
| enable_waf            | true                 | Enable CloudFront + WAF            |

### Environment-specific Configuration

Create `terraform.tfvars`:

```hcl
environment = "production"
domain_name = "api-public.runlab.bio"
enable_waf  = true

tags = {
  CostCenter = "research-api"
  Team       = "devops"
}
```

## Adding API Keys

Insert a new API key into DynamoDB:

```bash
aws dynamodb put-item \
  --table-name proteinscore-public-api-keys \
  --item '{
    "pk": {"S": "<sha256_hash_of_key>"},
    "tier": {"S": "free"},
    "owner_email": {"S": "user@example.com"},
    "organization": {"S": "Research Lab"},
    "created_at": {"S": "2026-02-25T00:00:00Z"},
    "enabled": {"BOOL": true}
  }'
```

## Monitoring

### CloudWatch Dashboards

- API Gateway access logs: `/aws/apigateway/proteinscore-public`
- Lambda Authorizer: `/aws/lambda/proteinscore-public-authorizer`
- Lambda Proxy: `/aws/lambda/proteinscore-public-proxy`

### Alarms to Consider

- Rate limit exceeded events (anomaly detection)
- Lambda error rates
- API Gateway 5xx errors
- WAF blocked requests

## Security

- TLS 1.2+ enforced
- API keys hashed with SHA-256
- DynamoDB encryption at rest
- WAF with AWS managed rules
- No credentials in code (uses IAM roles)
- CloudFront secret header for origin verification

## Cost Estimation

| Resource                | Estimated Monthly Cost |
|-------------------------|------------------------|
| API Gateway (1M req)    | ~$3.50                 |
| Lambda (1M invocations) | ~$2.00                 |
| DynamoDB (on-demand)    | ~$5.00                 |
| CloudFront (1TB)        | ~$85.00                |
| WAF (1M requests)       | ~$6.00                 |
| Route53                 | ~$0.50                 |
| **Total**               | **~$102/month**        |

*Costs vary based on usage patterns*
