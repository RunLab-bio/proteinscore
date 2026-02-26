# ProteinScore

> **Unified Protein Developability Platform**
>
> Open-source library for assessing therapeutic protein developability

---

## Overview

ProteinScore is the **first open-source platform** that unifies protein developability assessment into domain-specific modules. Each module provides comprehensive metrics for its protein class, with a unified scoring system (0-100).

### Why ProteinScore?

| Problem | Current State | ProteinScore Solution |
|---------|---------------|----------------------|
| **Fragmented Tools** | 6-8 separate tools for developability | Single unified interface |
| **No Integration** | Manual data transfer between tools | Automated pipeline |
| **Missing Immunogenicity** | Expensive commercial tools only | Free RIP API integration |
| **No Standard Score** | Different scales, formats | 0-100 unified score |
| **Domain-Specific Needs** | Generic tools for all proteins | Specialized modules per protein class |

---

## Platform Architecture

```
proteinscore/
│
├── predictors/                  # Shared base metrics
│   ├── stability.py             # Thermostability (all modules)
│   ├── solubility.py            # CamSol-like (all modules)
│   ├── aggregation.py           # TANGO-like (all modules)
│   └── immunogenicity.py        # RIP API (all modules)
│
├── antibody/                    # ✅ Available (v0.1)
│   ├── scorer.py                # AntibodyScorer main class
│   ├── tap_metrics.py           # TAP: PSH, PPC, PNC, SFvCSP
│   ├── cdr.py                   # CDR detection & analysis
│   ├── hydrophobicity.py        # HIC, AC-SINS prediction
│   └── liabilities.py           # PTM, deamidation, oxidation
│
├── enzyme/                      # 🔜 Planned (v0.2)
│   ├── scorer.py                # EnzymeScorer main class
│   ├── thermostability.py       # Tm for process/storage
│   └── expression.py            # Host expression prediction
│
└── peptide/                     # 🔜 Planned (v0.3)
    ├── scorer.py                # PeptideScorer main class
    ├── stability.py             # Proteolytic degradation
    └── chemical_liability.py    # Oxidation, deamidation
```

### Design Principles

- **Developability Focus**: All metrics assess manufacturability, not function/activity
- **CPU-Only**: No GPU required, runs on any machine
- **Modular**: Use only what you need
- **Unified Scoring**: 0-100 scale across all modules

---

## Quick Start

### Installation

```bash
pip install proteinscore
```

### Antibody Analysis (Available Now)

```python
from proteinscore.antibody import AntibodyScorer

scorer = AntibodyScorer()

# Analyze antibody developability
result = scorer.score(
    vh_sequence="QVQLVQSGAEVKKPGAS...",
    vl_sequence="DIQMTQSPSSLSASVGD..."
)

print(f"Developability Score: {result.total_score}/100")
print(f"Liabilities Found: {len(result.liabilities)}")
```

### Using Individual Components

```python
from proteinscore.antibody import (
    CDRDetector,
    TAPMetrics,
    LiabilityScanner,
    predict_self_association,
)

# CDR Detection
cdr_detector = CDRDetector()
cdr_regions = cdr_detector.detect_cdrs(vh_sequence, chain_type="heavy")

# TAP Metrics
tap = TAPMetrics()
tap_result = tap.calculate(vh_sequence, vl_sequence)

# Self-Association Prediction
sa_result = predict_self_association(vh_sequence, vl_sequence)
print(f"Self-Association Risk: {sa_result.risk_level}")
```

### Base Predictors (Any Protein)

```python
from proteinscore.predictors import (
    StabilityPredictor,
    SolubilityPredictor,
    AggregationPredictor,
)

# Works with any protein sequence
stability = StabilityPredictor().predict(sequence)
solubility = SolubilityPredictor().predict(sequence)
aggregation = AggregationPredictor().predict(sequence)
```

### With RIP API (Immunogenicity)

```python
from proteinscore.predictors import ImmunogenicityPredictor

# Free tier: 1,000 requests/day
predictor = ImmunogenicityPredictor(api_key="rl_abc123...")
result = predictor.predict(sequence)
print(f"Immunogenicity Risk: {result.risk_level}")
```

---

## Module Details

### Antibody Module (v0.1) ✅

Comprehensive antibody developability assessment based on TAP (Therapeutic Antibody Profiler) and literature-validated metrics.

| Metric | Description | Reference |
|--------|-------------|-----------|
| **CDR Analysis** | Chothia/Kabat/IMGT numbering | Al-Lazikani 1997 |
| **PSH** | Patches of Surface Hydrophobicity | Developability Index |
| **PPC/PNC** | Positive/Negative Patches in CDRs | TAP metrics |
| **SFvCSP** | CDR Structural Features | TAP metrics |
| **HIC Prediction** | Hydrophobic Interaction Chromatography | Jain 2017 |
| **AC-SINS** | Self-association prediction | Jain 2017 |
| **Liabilities** | PTM sites, aggregation motifs | Literature |

### Enzyme Module (v0.2) 🔜

CPU-only enzyme developability assessment.

| Metric | Description | Expected Accuracy |
|--------|-------------|-------------------|
| **Thermostability** | Tm prediction for process/storage | r = 0.70-0.80 |
| **Expression** | E. coli/CHO expression prediction | 60-74% accuracy |
| **Aggregation** | Aggregation during production | Well-established |
| **Solubility** | Formulation solubility | r² ≈ 0.5-0.6 |

### Peptide Module (v0.3) 🔜

Peptide therapeutic developability.

| Metric | Description |
|--------|-------------|
| **Stability** | Proteolytic degradation susceptibility |
| **Solubility** | Formulation compatibility |
| **Aggregation** | Self-association propensity |
| **Chemical Liability** | Oxidation, deamidation sites |

### Base Predictors (Available Now) ✅

Universal metrics applicable to any protein via `predictors/` module.

| Metric | Description | Method |
|--------|-------------|--------|
| **Stability** | Thermostability estimation | Sequence features |
| **Solubility** | Intrinsic solubility | CamSol-like |
| **Aggregation** | APR detection | TANGO/Zyggregator-like |
| **Immunogenicity** | MHC binding prediction | RIP API |

---

## Scoring System

All modules use a unified 0-100 scoring system:

| Score Range | Interpretation | Action |
|-------------|----------------|--------|
| 80-100 | Excellent | Proceed to development |
| 60-79 | Good | Minor optimization may help |
| 40-59 | Moderate | Consider redesign |
| 0-39 | Poor | Significant issues, redesign needed |

---

## API Tiers (for Immunogenicity via RIP)

| Tier | Rate Limit | Features | Price |
|------|------------|----------|-------|
| **Anonymous** | 100/day | Basic scoring | Free |
| **Free Account** | 1,000/day | Full scoring + history | Free |
| **Pro** | 50,000/day | Batch + priority | $99/month |
| **Enterprise** | Unlimited | SLA + support | Custom |

---

## Advanced Usage

### Batch Processing

```python
from proteinscore.antibody import AntibodyScorer

scorer = AntibodyScorer()

# Analyze multiple candidates
candidates = [
    {"vh": "QVQLVQ...", "vl": "DIQMTQ...", "name": "Ab-001"},
    {"vh": "EVQLVE...", "vl": "EIVLTQ...", "name": "Ab-002"},
]

results = scorer.score_batch(candidates)

# Rank by developability
ranked = sorted(results, key=lambda r: r.total_score, reverse=True)
```

---

## What is Developability?

ProteinScore focuses exclusively on **developability** - the ability to manufacture and formulate a protein therapeutic successfully.

| Developability (Our Focus) | NOT Our Focus |
|----------------------------|---------------|
| Will it express well? | What is its binding affinity? |
| Will it aggregate? | What is the kcat? |
| Is it stable in storage? | What substrate does it accept? |
| Is it immunogenic? | What is the mechanism? |
| Will it survive the process? | How efficient is it? |

For function/activity prediction, see specialized tools for your domain.

---

## Roadmap

| Version | Module | Status | Target |
|---------|--------|--------|--------|
| v0.1 | antibody/ | ✅ Available | - |
| v0.2 | general/, enzyme/ | 🔜 Development | Q2 2026 |
| v0.3 | peptide/ | 📋 Planned | Q3 2026 |
| v1.0 | Full platform | 📋 Planned | Q4 2026 |

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
  title = {ProteinScore: Unified Protein Developability Platform},
  author = {RunLab Team},
  year = {2026},
  url = {https://github.com/RunLab-bio/proteinscore}
}
```

---

## References

### Antibody Module
- Raybould et al. (2019). TAP: Therapeutic Antibody Profiler. *Bioinformatics*
- Jain et al. (2017). Developability of antibodies. *PNAS*
- Chothia & Lesk (1987). Canonical structures for CDRs. *JMB*

### General Methods
- Sormanni et al. (2015). CamSol: Solubility prediction. *J Mol Biol*
- Fernandez-Escamilla et al. (2004). TANGO: Aggregation prediction. *Nat Biotechnol*

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## Support

- **Issues**: [GitHub Issues](https://github.com/RunLab-bio/proteinscore/issues)
- **Discussions**: [GitHub Discussions](https://github.com/RunLab-bio/proteinscore/discussions)

---

*ProteinScore v0.1.0 - Antibody Module Available*
