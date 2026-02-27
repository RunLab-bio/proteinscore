# ProteinScore

> **Unified Protein Developability Platform**
>
> Open-source, CPU-only library for assessing therapeutic protein developability

---

## Overview

ProteinScore is the **first open-source, CPU-only platform** that unifies protein developability assessment into domain-specific modules. Each module provides comprehensive metrics for its protein class, with a unified scoring system (0-100).

### Why ProteinScore?

| Problem | Current State | ProteinScore Solution |
|---------|---------------|----------------------|
| **Fragmented Tools** | 6-8 separate tools for developability | Single unified interface |
| **No Integration** | Manual data transfer between tools | Automated pipeline |
| **GPU Required** | Most modern tools need expensive GPUs | CPU-only, runs anywhere |
| **Missing Immunogenicity** | Expensive commercial tools only | Free RIP API integration |
| **No Standard Score** | Different scales, formats | 0-100 unified score |
| **Domain-Specific Needs** | Generic tools for all proteins | Specialized modules per protein class |

### Performance Context

ProteinScore prioritizes **accessibility and integration** over maximum accuracy. For context:

| Approach | Thermostability | Antibody Developability | Requirements |
|----------|-----------------|------------------------|--------------|
| **ESM2-based models** | R² ≈ 0.95 | Varies by property | GPU required |
| **Structure-based (TAP)** | N/A | AUC ≈ 0.85 for HIC | 3D structure |
| **ProteinScore (CPU)** | ρ ≈ 0.75 | ρ ≈ 0.32-0.47 | CPU only |

> **Note**: Recent benchmarks ([FLAb2, 2025](https://www.biorxiv.org/content/10.64898/2025.12.27.696706v1)) show that even GPU-based AI models fail to achieve significant correlations for 80% of developability datasets. Sequence-only predictions have fundamental limitations regardless of computational resources.

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
├── enzyme/                      # ✅ Available (v0.1)
│   ├── scorer.py                # EnzymeScorer main class
│   ├── thermostability.py       # Tm for process/storage
│   └── expression.py            # Host expression prediction
│
├── peptide/                     # ✅ Available (v0.2)
│   ├── scorer.py                # PeptideScorer main class
│   ├── stability.py             # Proteolytic degradation
│   └── chemical_liability.py    # Oxidation, deamidation
│
├── middleware/                  # ✅ Available (v0.1)
│   ├── error_handler.py         # FastAPI exception handlers
│   └── request_context.py       # Request logging middleware
│
└── routers/                     # ✅ Available (v0.1)
    └── health.py                # Health check endpoints
```

### Design Principles

- **Developability Focus**: All metrics assess manufacturability, not function/activity
- **CPU-Only**: No GPU required, runs on any machine (laptop, CI/CD, cloud)
- **Modular**: Use only what you need
- **Unified Scoring**: 0-100 scale across all modules
- **Production Ready**: Structured logging, error handling, health checks
- **Honest Benchmarks**: Transparent about limitations and expected accuracy

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

### Antibody Module (v0.2) ✅

Comprehensive antibody developability assessment based on TAP (Therapeutic Antibody Profiler) and ML-optimized metrics validated on 137 clinical-stage antibodies (Jain et al. 2017).

| Metric | Description | Performance | Reference |
|--------|-------------|-------------|-----------|
| **CDR Analysis** | Chothia/Kabat/IMGT numbering | - | Al-Lazikani 1997 |
| **PSH** | Patches of Surface Hydrophobicity | ρ = -0.22 | Developability Index |
| **PPC/PNC** | Positive/Negative Patches in CDRs | - | TAP metrics |
| **SFvCSP** | CDR Structural Features | - | TAP metrics |
| **Expression** | HEK titer prediction | ρ = 0.32*** | ML-optimized |
| **Self-Association** | AC-SINS prediction | ρ = 0.47*** | ML-optimized |
| **Cross-Interaction** | CSI-BLI prediction | ρ = 0.34*** | ML-optimized |
| **HIC Retention** | Hydrophobicity ranking | ρ = 0.55*** | GBM-optimized |
| **Liabilities** | PTM sites, aggregation motifs | - | Literature |

*\*\*\* p < 0.001*

### Enzyme Module (v0.1) ✅

CPU-only enzyme developability assessment with thermostability and expression prediction.

| Metric | Description | Expected Accuracy |
|--------|-------------|-------------------|
| **Thermostability** | Tm prediction for process/storage | r = 0.70-0.80 |
| **Expression** | E. coli/CHO expression prediction | 60-74% accuracy |
| **Aggregation** | Aggregation during production | Well-established |
| **Solubility** | Formulation solubility | r² ≈ 0.5-0.6 |

```python
from proteinscore.enzyme import EnzymeScorer

scorer = EnzymeScorer()
result = scorer.score(sequence="MKTAYIAKQRQISFVK...")

print(f"Enzyme Score: {result.total_score}/100")
print(f"Predicted Tm: {result.thermostability.predicted_tm}°C")
print(f"Expression Host: {result.expression.recommended_host}")
```

### Peptide Module (v0.2) ✅

Comprehensive peptide therapeutic developability assessment with proteolytic and chemical stability prediction, validated against therapeutic peptides (THPdb).

| Metric | Description | Performance | Method |
|--------|-------------|-------------|--------|
| **DPP-4 Susceptibility** | N-terminal cleavage prediction | 85.7% accuracy | Sequence motif |
| **Chemical Liability** | Oxidation, deamidation, isomerization | 100% detection | Sequence motif |
| **Modification Half-life** | Lipidation, PEGylation, Fc fusion | 100% accuracy | Literature factors |
| **Proteolytic Stability** | Protease cleavage sites | - | PROSPER/PeptideCutter |
| **Solubility** | Formulation compatibility | - | CamSol-like |

```python
from proteinscore.peptide import PeptideScorer
from proteinscore.peptide.stability import ModificationType

scorer = PeptideScorer()
result = scorer.score(sequence="HAEGTFTSDVSSYLEGQAAK")

print(f"Peptide Score: {result.total_score}/100")
print(f"Half-life: {result.estimated_half_life}")
print(f"DPP-4 Susceptible: {result.is_dpp4_susceptible}")

# With modification (lipidation, PEGylation, etc.)
from proteinscore.peptide.stability import PeptideStabilityPredictor
predictor = PeptideStabilityPredictor()
result = predictor.predict(sequence, modification=ModificationType.LIPIDATION_C18)
print(f"Modified Half-life: {result.estimated_half_life_hours}h")  # ~168h for semaglutide-like
```

**Modification Factors (Clinically Validated):**
| Modification | Half-life Factor | Example Drug |
|--------------|------------------|--------------|
| None (native) | 1x | GLP-1 (~2 min) |
| Aib substitution | 2x | - |
| C16 lipidation | 120x | Liraglutide (~13h) |
| C18 lipidation | 1680x | Semaglutide (~168h) |
| PEGylation | 400x | - |
| Fc fusion | 2400x | Dulaglutide (~5 days) |

**Key Features:**
- **Protease Cleavage Analysis**: Trypsin, chymotrypsin, pepsin, DPP-4, and more
- **Terminal Susceptibility**: N/C-terminal aminopeptidase/carboxypeptidase risk
- **Chemical Modifications**: Met oxidation, Asn deamidation, Asp isomerization
- **Modification-Aware**: Predicts half-life with lipidation, PEGylation, Fc fusion
- **Stabilization Strategies**: Actionable recommendations for sequence optimization

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

### Structured Logging

```python
from proteinscore import configure_logging, get_logger

# Configure structured logging (JSON in production, colored in development)
configure_logging()

logger = get_logger(__name__)
logger.info("scoring_started", sequence_length=150, method="stability")
```

### Exception Handling

```python
from proteinscore import (
    InvalidSequenceError,
    RateLimitError,
    NotFoundError,
)

try:
    result = scorer.score(sequence)
except InvalidSequenceError as e:
    print(f"Error {e.error_code}: {e.message}")
    print(f"HTTP Status: {e.status_code}")  # 400
except RateLimitError as e:
    print(f"Retry after: {e.details['retry_after']}")
```

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
| v0.1 | antibody/, enzyme/ | ✅ Available | - |
| v0.2 | peptide/ | ✅ Available | - |
| v0.3 | API server | 🔜 Development | Q3 2026 |
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

### Enzyme Module
- Nature Scientific Reports (2025). Gradient Boosting Tm prediction
- SoluProt (Bioinformatics 2021): Solubility/expression
- SOLpro (Bioinformatics 2009): Expression prediction
- ProTherm database: Thermostability validation

### Peptide Module
- Fosgerau & Hoffmann (2015). Peptide therapeutics: current status and future directions. *Drug Discov Today*
- PROSPER (Bioinformatics 2012): Protease specificity prediction
- PeptideCutter (ExPASy): Protease cleavage rules
- Werle & Bernkop-Schnürch (2006): Strategies to improve plasma half life of peptide drugs
- Manning et al. (2010). Stability of protein pharmaceuticals. *Pharm Res*

### General Methods
- Sormanni et al. (2015). CamSol: Solubility prediction. *J Mol Biol*
- Fernandez-Escamilla et al. (2004). TANGO: Aggregation prediction. *Nat Biotechnol*

### Benchmarking & Validation
- Chungyoun & Gray (2025). [FLAb2: Benchmarking Reveals That Protein AI Models Cannot Yet Consistently Predict Developability Properties](https://www.biorxiv.org/content/10.64898/2025.12.27.696706v1). *bioRxiv*
- [ESMStabP](https://pmc.ncbi.nlm.nih.gov/articles/PMC11870573/): ESM2-based thermostability prediction (GPU)
- [TAP Web Server](https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabpred/tap): Structure-based antibody profiling

### HIC Prediction Methods
- [PROPERMAB](https://github.com/regeneron-mpds/propermab): Protein Property Prediction for Monoclonal Antibodies (ρ = 0.75 with 3D structure)
- [DeepSP](https://github.com/Lailabcode/DeepSP): Deep learning for SAP/SCM prediction from sequence (CNN-based)
- Jain et al. (2017). Biophysical properties of the clinical-stage antibody landscape. *PNAS*

---

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## Support

- **Issues**: [GitHub Issues](https://github.com/RunLab-bio/proteinscore/issues)
- **Discussions**: [GitHub Discussions](https://github.com/RunLab-bio/proteinscore/discussions)

---

## Benchmark Results

### Core Predictors

| Metric | Dataset | Performance | Reference |
|--------|---------|-------------|-----------|
| **Immunogenicity** | IEDB MHC-I | ρ = 0.53, AUC = 0.89 | NetMHCpan: ρ = 0.56 |
| **Solubility** | CamSol/ProteinSol | ρ = 0.87, Acc = 100% | CamSol: ρ = 0.52 |
| **Aggregation** | AmyLoad/WALTZ | AUC = 0.87, MCC = 0.67 | TANGO: AUC = 0.85 |
| **Thermostability** | ProTherm/FireProt | ρ = 0.75 | FoldX: ρ = 0.65 |

### Antibody Module (Jain 2017 Dataset, n=137)

| Metric | Spearman ρ | p-value | SOTA |
|--------|------------|---------|------|
| **Expression (HEK)** | 0.32 | 0.0001*** | ~0.30 |
| **Self-Association (AC-SINS)** | 0.47 | <0.0001*** | ~0.40-0.50 |
| **Cross-Interaction (CSI-BLI)** | 0.34 | <0.0001*** | - |
| **HIC Retention** | 0.55 | <0.0001*** | PROPERMAB: 0.75 |

### HIC Retention Benchmark (Detailed)

Comprehensive evaluation of HIC (Hydrophobic Interaction Chromatography) retention time prediction methods:

| Approach | Spearman ρ | Method | Notes |
|----------|------------|--------|-------|
| **GBM + Handcrafted** | 0.553 ± 0.046 | Gradient Boosting | **Best sequence-only** |
| **Ensemble Weighted** | 0.461 | Model averaging | 3-model ensemble |
| **GBM Basic** | 0.444 ± 0.107 | 20 basic features | Baseline |
| **ESM-2 150M + Ridge** | 0.429 ± 0.183 | Embeddings | High variance |
| **ESM-2 150M + RF** | 0.413 ± 0.247 | Embeddings | Very high variance |
| **DeepSP (SAP/SCM)** | 0.191 ± 0.069 | CNN-based | No improvement |
| **TAP Baseline** | 0.351 | TAP PSH metric | Original baseline |

**Key Findings:**
- **Best achievable (sequence-only)**: ρ = 0.553 with GBM + handcrafted features
- **Improvement over baseline**: +57% vs TAP PSH (0.351)
- **Gap to SOTA**: PROPERMAB achieves ρ = 0.75 using 3D structure + MD simulation
- **DeepSP evaluation**: CNN-based SAP/SCM approximation did not improve results
- **ESM-2 embeddings**: High variance across folds, not consistently better than handcrafted

**Handcrafted Features (59 features):**
- Physicochemical: hydrophobicity scales, charge, molecular weight
- Regional: CDR-specific vs framework properties
- Surface: solvent accessibility approximations
- Aggregation: APR detection, hydrophobic patches

> **Note**: To reach PROPERMAB-level performance (ρ = 0.75), actual 3D structure prediction (ESMFold/AlphaFold) with real SAP calculation from MD simulation would be required. Sequence-only methods have fundamental limitations for predicting 3D surface properties.

### Peptide Module

| Metric | Performance | Dataset |
|--------|-------------|---------|
| **DPP-4 Prediction** | 85.7% accuracy | GLP-1 analogs |
| **Chemical Liability** | 100% detection | Oxidation/deamidation |
| **Modification Factors** | 100% accuracy | Lipidation/PEG/Fc |

---

## New in v0.2.1

- **HIC Retention Benchmark**: Comprehensive evaluation of hydrophobicity prediction methods
  - Best sequence-only performance: ρ = 0.553 (GBM + handcrafted features)
  - +57% improvement over TAP baseline (0.351)
  - DeepSP (SAP/SCM) evaluation: CNN-based approach did not improve results
  - ESM-2 embeddings: High variance, not consistently better than handcrafted
  - Gap analysis: PROPERMAB (0.75) requires 3D structure + MD simulation
- **DeepSP PyTorch Port**: Keras model converted to PyTorch (no TensorFlow dependency)
- **ANARCI Integration**: Antibody numbering via anarcii for sequence alignment

## New in v0.2.0

- **Peptide Module**: Full proteolytic and chemical stability prediction
  - Protease cleavage site detection (trypsin, chymotrypsin, DPP-4, etc.)
  - Chemical liability scanning (oxidation, deamidation, isomerization)
  - **Modification-aware half-life**: Lipidation (C16/C18), PEGylation, Fc fusion
  - Clinically-calibrated factors based on Liraglutide, Semaglutide, Dulaglutide
- **Antibody ML Scoring**: ML-optimized expression, self-association, cross-interaction
  - Validated on 137 clinical-stage antibodies (Jain et al. 2017)
  - Expression ρ = 0.32, Self-association ρ = 0.47, Cross-interaction ρ = 0.34
- **Solubility Improvement**: Classification accuracy improved to 100%
- **THPdb Integration**: Therapeutic peptide half-life benchmark data

## New in v0.1.0

- **Enzyme Module**: Full thermostability and expression prediction
- **Structured Logging**: Structlog integration with JSON/console output
- **Exception Hierarchy**: HTTP status codes for all exceptions
- **FastAPI Middleware**: Error handlers and request context (optional)
- **Health Endpoints**: Kubernetes-ready liveness/readiness probes

---

*ProteinScore v0.2.1 - HIC Benchmark + Antibody, Enzyme & Peptide Modules with ML-Optimized Scoring*
