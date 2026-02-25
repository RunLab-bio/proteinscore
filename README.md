# ProteinScore

> **Unified Protein Developability Scorer**
>
> Open-source library for scoring therapeutic protein developability

---

## Overview

ProteinScore is the **first open-source library** that unifies all protein developability metrics into a single actionable score. By integrating with RunLab's state-of-the-art immunogenicity predictor (RIP), it provides comprehensive assessment of therapeutic protein candidates.

### Why ProteinScore?

| Problem | Current State | ProteinScore Solution |
|---------|---------------|----------------------|
| **Fragmented Tools** | 6-8 separate tools for developability | Single unified interface |
| **No Integration** | Manual data transfer between tools | Automated pipeline |
| **Missing Immunogenicity** | Expensive commercial tools only | Free RIP API integration |
| **No Standard Score** | Different scales, formats | 0-100 ProteinScore |

---

## Quick Start

### Installation

```bash
pip install proteinscore
```

### Basic Usage

```python
from proteinscore import ProteinScore

# Initialize with free tier (100 requests/day)
scorer = ProteinScore()

# Score a protein sequence
result = scorer.score("MKTAYIAKQRQISFVKSHFSRQLE...")

print(f"ProteinScore: {result.total_score}/100")
print(f"Stability: {result.stability.score}")
print(f"Solubility: {result.solubility.score}")
print(f"Aggregation: {result.aggregation.score}")
print(f"Immunogenicity: {result.immunogenicity.score}")
```

### With API Key (Higher Limits)

```python
from proteinscore import ProteinScore

# Free tier: 1,000 requests/day
scorer = ProteinScore(api_key="rl_abc123...")

# Or set environment variable
# export RUNLAB_API_KEY=rl_abc123...
scorer = ProteinScore()
```

---

## Scoring Components

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  ProteinScore (MIT License)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌───────────┐ │
│  │ Stability   │ │ Solubility  │ │ Aggregation │ │ Immuno-   │ │
│  │             │ │             │ │             │ │ genicity  │ │
│  │ LOCAL       │ │ LOCAL       │ │ LOCAL       │ │ API CALL  │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └───────────┘ │
│         │               │               │               │       │
│         └───────────────┴───────────────┴───────────────┘       │
│                                 │                               │
│                    ┌────────────▼────────────┐                  │
│                    │   Score Aggregator      │                  │
│                    │   Weighted combination  │                  │
│                    │   0-100 final score     │                  │
│                    └─────────────────────────┘                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   RunLab RIP API        │
                    │   (Immunogenicity)      │
                    └─────────────────────────┘
```

### 1. Stability

Predicts thermodynamic stability using energy calculations.

| Score Range | Interpretation |
|-------------|----------------|
| 80-100 | Excellent stability |
| 60-79 | Good stability |
| 40-59 | Moderate stability |
| 0-39 | Poor stability, redesign needed |

### 2. Solubility

Predicts aqueous solubility from sequence features.

| Score Range | Interpretation |
|-------------|----------------|
| 80-100 | Highly soluble |
| 60-79 | Soluble |
| 40-59 | Moderate solubility |
| 0-39 | Aggregation-prone |

### 3. Aggregation

Identifies aggregation-prone regions (APRs).

| Score Range | Interpretation |
|-------------|----------------|
| 80-100 | Low aggregation risk |
| 60-79 | Acceptable |
| 40-59 | Some APRs detected |
| 0-39 | High aggregation risk |

### 4. Immunogenicity (RIP API)

State-of-the-art MHC-I binding prediction.

| Score Range | Interpretation |
|-------------|----------------|
| 80-100 | Low immunogenic risk |
| 60-79 | Acceptable for most applications |
| 40-59 | Consider deimmunization |
| 0-39 | High immunogenic risk |

---

## API Tiers

| Tier | Rate Limit | Features | Price |
|------|------------|----------|-------|
| **Anonymous** | 100/day | Basic scoring | Free |
| **Free Account** | 1,000/day | Full scoring + history | Free |
| **Pro** | 50,000/day | Batch + priority | $99/month |
| **Enterprise** | Unlimited | SLA + support | Custom |

---

## Advanced Usage

### Custom Weights

```python
from proteinscore import ProteinScore, ScoringWeights

# Emphasize stability for enzyme design
weights = ScoringWeights(
    stability=0.4,
    solubility=0.2,
    aggregation=0.2,
    immunogenicity=0.2
)

scorer = ProteinScore(weights=weights)
```

### Preset Weights

```python
from proteinscore import ScoringWeights

# For enzyme design (emphasizes stability)
weights = ScoringWeights.for_enzyme()

# For antibody therapeutics (balanced)
weights = ScoringWeights.for_antibody()

# For vaccine antigens (emphasizes immunogenicity)
weights = ScoringWeights.for_vaccine()
```

### Batch Processing

```python
sequences = ["MKTAY...", "MSKGE...", "MVLSG..."]
names = ["Protein A", "Protein B", "Protein C"]

results = scorer.score_batch(sequences, names=names)

# Compare results
comparison = scorer.compare(results)
print(comparison)
```

### Local-Only Mode

```python
# Skip API calls, use local estimation for immunogenicity
scorer = ProteinScore(local_only=True)
```

---

## Documentation

For full documentation, visit the [GitHub Wiki](https://github.com/RunLab-bio/proteinscore/wiki).

---

## License

ProteinScore is released under the [MIT License](LICENSE).

The RunLab RIP API (used for immunogenicity predictions) is a proprietary service with free tier access.

---

## Trademarks

ProteinScore™ and RunLab™ are trademarks of RunLab.

---

## Citation

If you use ProteinScore in your research, please cite:

```bibtex
@software{proteinscore2026,
  title = {ProteinScore: Unified Protein Developability Scorer},
  author = {RunLab Team},
  year = {2026},
  url = {https://github.com/RunLab-bio/proteinscore}
}
```

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## Support

- **Issues**: [GitHub Issues](https://github.com/RunLab-bio/proteinscore/issues)
- **Discussions**: [GitHub Discussions](https://github.com/RunLab-bio/proteinscore/discussions)

---

*ProteinScore v0.1.0*
