# DevScore Benchmark Suite

Validation suite for DevScore protein developability predictions against industry-standard datasets.

## Quick Start

```bash
# Install DevScore
cd /path/to/DevScore
pip install -e .

# Download benchmark data
python benchmarks/download_flab.py

# Run all benchmarks
python benchmarks/benchmark_public.py      # Public datasets (IEDB, CamSol, etc.)
python benchmarks/benchmark_jain2017.py    # Jain 2017 antibody benchmark
python benchmarks/benchmark_flab.py        # FLAb2 antibody benchmark
```

## Available Benchmarks

### 1. Public Benchmark (`benchmark_public.py`)
General protein developability validation using public datasets:
- **IEDB** - Immunogenicity prediction (MHC binding)
- **CamSol/ProteinSol** - Solubility prediction
- **AmyLoad/WALTZ** - Aggregation propensity
- **ProTherm/FireProtDB** - Thermostability

### 2. Jain 2017 Benchmark (`benchmark_jain2017.py`)
**Gold Standard** for antibody developability.

| Property | Details |
|----------|---------|
| **Dataset** | 137 clinical-stage antibodies |
| **Status** | Phase I-III, FDA-approved |
| **Metrics** | Expression, Aggregation (AC-SINS), Cross-interaction (CSI-BLI) |
| **Reference** | Jain T et al. PNAS 114:944-949 (2017) |

### 3. FLAb2 Benchmark (`benchmark_flab.py`)
Largest public therapeutic antibody benchmark.

| Property | Details |
|----------|---------|
| **Dataset** | >4 million antibodies across 32 studies |
| **Categories** | Thermostability, Expression, Aggregation, Binding, PK, Polyreactivity |
| **Reference** | Chungyoun & Gray, bioRxiv 2025 |
| **Access** | https://github.com/Graylab/FLAb |

## Data Download

### FLAb2 Data
```bash
python benchmarks/download_flab.py

# Downloads to benchmarks/data/flab/
# - jain2017_expression.csv
# - jain2017_acsins.csv
# - jain2017_csibli.csv
# - garbinski2023_tm1.csv
# - garbinski2023_exp.csv
```

## Benchmark Results

### Public Benchmark (General Proteins)

| Category | Metric | Value | Reference |
|----------|--------|-------|-----------|
| **Immunogenicity** | Spearman ρ | 0.328 | Local |
| **Immunogenicity** | AUC-ROC | 0.821 | Local |
| **Immunogenicity** | Spearman ρ | 0.693 | RIP API |
| **Solubility** | Spearman ρ | 0.300 | Local |
| **Aggregation** | AUC-ROC | 0.650 | Local |
| **Thermostability** | Spearman ρ | 0.286 | Local |

### Antibody Benchmarks (Jain 2017 / FLAb2)

⚠️ **Note**: DevScore v0.1 is optimized for general proteins. Antibody-specific predictions require specialized models that account for CDR regions, framework stability, and VH/VL pairing.

| Dataset | Category | Spearman ρ | Status |
|---------|----------|------------|--------|
| garbinski2023_exp | Expression | 0.275 | ✓ Positive |
| jain2017_expression | Expression | -0.129 | Needs improvement |
| jain2017_acsins | Aggregation | -0.246 | Needs improvement |
| garbinski2023_tm1 | Thermostability | -0.256 | Needs improvement |

### Interpretation

DevScore performs well on:
- **General proteins** - Positive correlations with experimental data
- **Immunogenicity** - Strong AUC-ROC (0.821)
- **Therapeutic proteins** - Correct risk stratification (Trastuzumab, Adalimumab)

DevScore needs improvement for:
- **Antibody-specific properties** - Requires CDR analysis, pairing effects
- **Aggregation hotspots** - Need sequence-specific APR detection

## Therapeutic Protein Validation

| Protein | DevScore | Risk | Status |
|---------|----------|------|--------|
| Trastuzumab VH | 66.9 | Medium | FDA-approved |
| Adalimumab VH | 59.2 | Medium | FDA-approved |
| Insulin B-chain | 65.0 | Medium | Known aggregation issues |
| GLP-1 | 70.6 | Low | Requires modifications |

## DevScore Interpretation Guide

| Score | Risk Level | Interpretation |
|-------|------------|----------------|
| 80-100 | Low | Excellent developability |
| 60-79 | Medium | Good candidate |
| 40-59 | High | Engineering recommended |
| 0-39 | Critical | Major challenges |

## Component Scores

- **Stability (S)**: Thermodynamic stability prediction
- **Solubility (So)**: Expression/purification solubility
- **Aggregation (A)**: Aggregation resistance (higher = better)
- **Immunogenicity (I)**: Low immunogenic potential (higher = better)

## Output Files

Results are saved to `benchmarks/results/`:
- `public_benchmark_*.json` - Public benchmark results
- `jain2017_benchmark_*.json` - Jain 2017 results
- `flab_benchmark_*.json` - FLAb2 results

## References

### Benchmark Datasets
1. **Jain 2017**: Jain T et al. "Biophysical properties of the clinical-stage antibody landscape." PNAS 114:944-949 (2017)
2. **FLAb2**: Chungyoun M & Gray J. "FLAb2: Benchmarking Reveals That Protein AI Models Cannot Yet Consistently Predict Developability Properties." bioRxiv (2025)
3. **IEDB**: Vita R et al. "The Immune Epitope Database." Nucleic Acids Res. (2019)
4. **CamSol**: Sormanni P et al. J Mol Biol. (2015)
5. **WALTZ-DB**: Louros N et al. NAR 48(D1):D389-D393 (2020)

### Reference Methods
- NetMHCpan 4.1: Reynisson B et al. NAR (2020)
- MHCflurry 2.0: O'Donnell TJ et al. Cell Syst. (2020)
- FoldX: Schymkowitz J et al. NAR (2005)
- TANGO: Fernandez-Escamilla AM et al. Nat Biotechnol. (2004)

## Citation

```bibtex
@software{devscore2024,
  title={DevScore: Integrated Protein Developability Prediction},
  author={RunLab},
  year={2024},
  url={https://runlab.bio}
}
```
