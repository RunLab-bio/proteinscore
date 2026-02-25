# ProteinScore API Reference

> **RunLab RIP Public API**
>
> Base URL: `https://api.runlab.bio`

---

## Authentication

### API Key

Include your API key in the `X-API-Key` header:

```bash
curl -X POST https://api.runlab.bio/rip/predict \
  -H "X-API-Key: rl_your_api_key_here" \
  -H "Content-Type: application/json" \
  -d '{"peptide": "YLQPRTFLL", "allele": "HLA-A*02:01"}'
```

### Anonymous Access

Requests without an API key are allowed with reduced rate limits (100/day).

---

## Rate Limits

| Tier | Requests/Day | Batch Size |
|------|--------------|------------|
| Anonymous | 100 | 10 |
| Free | 1,000 | 100 |
| Pro | 50,000 | 1,000 |
| Enterprise | Unlimited | 10,000 |

### Rate Limit Headers

```http
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 847
X-RateLimit-Reset: 2026-02-26T00:00:00Z
```

### Rate Limit Exceeded

```json
{
  "error": "rate_limit_exceeded",
  "message": "Daily rate limit of 1000 requests exceeded",
  "retry_after": "2026-02-26T00:00:00Z"
}
```

---

## Endpoints

### Health Check

```http
GET /rip/health
```

**Response:**

```json
{
  "status": "healthy",
  "version": "2.1.0",
  "model": "rip_v2_sota",
  "available_alleles": 11360
}
```

---

### List Alleles

```http
GET /rip/alleles
```

**Response:**

```json
{
  "count": 11360,
  "alleles": [
    "HLA-A*01:01",
    "HLA-A*02:01",
    "HLA-A*02:02",
    "HLA-A*02:03",
    "..."
  ]
}
```

**Query Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `prefix` | string | Filter by prefix (e.g., `HLA-A*02`) |
| `limit` | integer | Max results (default: 100, max: 1000) |
| `offset` | integer | Pagination offset |

---

### Single Prediction

```http
POST /rip/predict
```

**Request Body:**

```json
{
  "peptide": "YLQPRTFLL",
  "allele": "HLA-A*02:01"
}
```

**Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `peptide` | string | Yes | Peptide sequence (8-15 amino acids) |
| `allele` | string | Yes | HLA allele name |

**Response:**

```json
{
  "peptide": "YLQPRTFLL",
  "allele": "HLA-A*02:01",
  "binding_affinity_nM": 12.5,
  "log_ic50": 1.097,
  "percentile_rank": 0.15,
  "is_strong_binder": true,
  "is_weak_binder": true,
  "confidence": 0.92
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `binding_affinity_nM` | float | Predicted IC50 in nanomolar |
| `log_ic50` | float | Log10 of IC50 |
| `percentile_rank` | float | Percentile rank (0-100, lower = stronger) |
| `is_strong_binder` | boolean | True if percentile_rank <= 0.5 |
| `is_weak_binder` | boolean | True if percentile_rank <= 2.0 |
| `confidence` | float | Model confidence (0-1) |

---

### Batch Prediction

```http
POST /rip/predict/batch
```

**Request Body:**

```json
{
  "peptides": [
    "YLQPRTFLL",
    "GILGFVFTL",
    "NLVPMVATV"
  ],
  "alleles": ["HLA-A*02:01"]
}
```

**Parameters:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `peptides` | array | Yes | List of peptide sequences |
| `alleles` | array | Yes | List of HLA alleles |

**Expansion Rules:**

- If `len(alleles) == 1`: Apply single allele to all peptides
- If `len(alleles) == len(peptides)`: 1:1 mapping
- Otherwise: Cartesian product (peptides x alleles)

**Response:**

```json
{
  "predictions": [
    {
      "peptide": "YLQPRTFLL",
      "allele": "HLA-A*02:01",
      "binding_affinity_nM": 12.5,
      "percentile_rank": 0.15,
      "is_strong_binder": true,
      "is_weak_binder": true,
      "confidence": 0.92
    },
    {
      "peptide": "GILGFVFTL",
      "allele": "HLA-A*02:01",
      "binding_affinity_nM": 8.3,
      "percentile_rank": 0.08,
      "is_strong_binder": true,
      "is_weak_binder": true,
      "confidence": 0.95
    },
    {
      "peptide": "NLVPMVATV",
      "allele": "HLA-A*02:01",
      "binding_affinity_nM": 45.2,
      "percentile_rank": 0.42,
      "is_strong_binder": true,
      "is_weak_binder": true,
      "confidence": 0.88
    }
  ],
  "count": 3,
  "processing_time_ms": 125
}
```

---

### Protein Scan

```http
POST /rip/scan
```

**Request Body:**

```json
{
  "sequence": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQQIAAALEHHHHHH",
  "alleles": ["HLA-A*02:01", "HLA-A*24:02", "HLA-B*07:02"],
  "peptide_lengths": [9],
  "threshold_percentile": 2.0
}
```

**Parameters:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `sequence` | string | Yes | - | Full protein sequence |
| `alleles` | array | No | Top 20 | HLA alleles to scan |
| `peptide_lengths` | array | No | [9] | Peptide lengths to generate |
| `threshold_percentile` | float | No | 2.0 | Max percentile rank to report |

**Response:**

```json
{
  "protein_length": 93,
  "alleles_scanned": 3,
  "peptides_scanned": 85,
  "strong_binders": 12,
  "weak_binders": 28,
  "epitopes": [
    {
      "peptide": "YLQPRTFLL",
      "position": [15, 24],
      "allele": "HLA-A*02:01",
      "binding_affinity_nM": 12.5,
      "percentile_rank": 0.15,
      "is_strong_binder": true
    }
  ],
  "per_residue_risk": [0.12, 0.15, 0.18, 0.45, 0.67, "..."],
  "processing_time_ms": 1250
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `protein_length` | integer | Length of input sequence |
| `alleles_scanned` | integer | Number of alleles analyzed |
| `peptides_scanned` | integer | Number of peptides generated |
| `strong_binders` | integer | Count of strong binders (<=0.5%) |
| `weak_binders` | integer | Count of weak binders (<=2.0%) |
| `epitopes` | array | List of predicted epitopes |
| `per_residue_risk` | array | Immunogenicity risk per position (0-1) |

---

### Population Coverage

```http
POST /rip/coverage
```

**Request Body:**

```json
{
  "alleles": ["HLA-A*02:01", "HLA-A*24:02", "HLA-B*07:02"],
  "population": "global"
}
```

**Parameters:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `alleles` | array | Yes | - | HLA alleles |
| `population` | string | No | "global" | Population identifier |

**Available Populations:**

- `global` - World population
- `european` - European ancestry
- `african` - African ancestry
- `asian` - East Asian ancestry
- `hispanic` - Hispanic/Latino ancestry

**Response:**

```json
{
  "alleles": ["HLA-A*02:01", "HLA-A*24:02", "HLA-B*07:02"],
  "population": "global",
  "coverage": 0.8234,
  "coverage_percent": "82.3%"
}
```

---

## Error Responses

### Standard Error Format

```json
{
  "error": "error_code",
  "message": "Human-readable error message",
  "details": {
    "field": "additional context"
  }
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `invalid_request` | 400 | Malformed request body |
| `invalid_peptide` | 400 | Invalid peptide sequence |
| `invalid_allele` | 400 | Unknown HLA allele |
| `unauthorized` | 401 | Missing or invalid API key |
| `rate_limit_exceeded` | 429 | Daily limit exceeded |
| `internal_error` | 500 | Server error |
| `service_unavailable` | 503 | Service temporarily unavailable |

### Validation Errors

```json
{
  "error": "invalid_peptide",
  "message": "Peptide sequence contains invalid characters",
  "details": {
    "peptide": "YLQPRTFXL",
    "invalid_characters": ["X"],
    "valid_characters": "ACDEFGHIKLMNPQRSTVWY"
  }
}
```

---

## Code Examples

### Python

```python
import requests

API_KEY = "rl_your_api_key"
BASE_URL = "https://api.runlab.bio"

# Single prediction
response = requests.post(
    f"{BASE_URL}/rip/predict",
    headers={"X-API-Key": API_KEY},
    json={
        "peptide": "YLQPRTFLL",
        "allele": "HLA-A*02:01"
    }
)

result = response.json()
print(f"Binding affinity: {result['binding_affinity_nM']} nM")
print(f"Strong binder: {result['is_strong_binder']}")
```

### JavaScript

```javascript
const API_KEY = "rl_your_api_key";
const BASE_URL = "https://api.runlab.bio";

const response = await fetch(`${BASE_URL}/rip/predict`, {
  method: "POST",
  headers: {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json"
  },
  body: JSON.stringify({
    peptide: "YLQPRTFLL",
    allele: "HLA-A*02:01"
  })
});

const result = await response.json();
console.log(`Binding affinity: ${result.binding_affinity_nM} nM`);
```

### cURL

```bash
curl -X POST https://api.runlab.bio/rip/predict \
  -H "X-API-Key: rl_your_api_key" \
  -H "Content-Type: application/json" \
  -d '{
    "peptide": "YLQPRTFLL",
    "allele": "HLA-A*02:01"
  }'
```

---

## SDKs

### Official SDKs

| Language | Package | Installation |
|----------|---------|--------------|
| Python | `proteinscore` | `pip install proteinscore` |
| JavaScript | `@runlab/proteinscore` | `npm install @runlab/proteinscore` |
| R | `proteinscore` | `devtools::install_github("RunLab-bio/proteinscore-r")` |

### Community SDKs

Contributions welcome! See [GitHub](https://github.com/RunLab-bio/proteinscore).

---

## Changelog

### v2.1.0 (2026-02-25)

- Initial public API release
- RIP model: Spearman 0.693, +23.8% vs NetMHCpan 4.1
- 11,360 supported HLA alleles
- Batch predictions up to 1,000 peptides

---

*API Reference v1.0 — 2026-02-25*
