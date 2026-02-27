# ProteinScore Benchmark Datasets

This document describes all datasets used for validating ProteinScore predictions.

## Dataset Overview

| Dataset | Module | Size | Format | Download |
|---------|--------|------|--------|----------|
| **IEDB** | Immunogenicity | ~50K peptides | CSV | API/Manual |
| **WALTZ-DB 2.0** | Aggregation | 1,416 entries | CSV | Web export |
| **AmyLoad** | Aggregation | ~1,400 entries | CSV | Web export |
| **ProThermDB** | Stability | 32,000+ entries | CSV | Web export |
| **FireProtDB** | Stability | Curated subset | CSV/JSON | Download page |
| **CamSol** | Solubility | 56 mutations | Paper | Embedded |
| **PSI:Biology** | Solubility/Expression | 11,226 sequences | CSV | Via papers |
| **FLAb2** | Antibody | 3M+ data points | CSV | GitHub |
| **Jain 2017** | Antibody | 137 mAbs | CSV | Via FLAb |
| **BIOPEP-UWM** | Peptide | Large | Web interface | Manual |
| **GLP-1 Analogs** | Peptide | ~10 peptides | Embedded | Literature |

---

## 1. Immunogenicity Datasets

### IEDB (Immune Epitope Database)
- **URL**: https://www.iedb.org/
- **Description**: Gold standard for MHC-peptide binding validation
- **Size**: >850,000 peptides, 48+ MHC alleles
- **Format**: CSV via REST API or web export
- **Used by**: NetMHCpan, MHCflurry, PRIME
- **Reference**: Vita et al., Nucleic Acids Res. 2019

**Download options:**
1. REST API: `https://www.iedb.org/downloader.php`
2. Web interface: Filter and export
3. Full database export (large, ~GB)

**Our usage**: Curated subset of HLA-A*02:01 binders/non-binders embedded in `benchmark_public.py`

---

## 2. Aggregation Datasets

### WALTZ-DB 2.0
- **URL**: http://waltzdb.switchlab.org/
- **Description**: Experimentally validated amyloidogenic hexapeptides
- **Size**: 1,416 hexapeptides
- **Format**: CSV export from web interface
- **Validation**: TEM, FTIR, ThT binding assays
- **Reference**: Louros et al., Nucleic Acids Res. 2020

**Dataset composition:**
- 720 experimentally verified fiber-forming peptides
- 229 in-house validated hexapeptides
- 98 literature-curated peptides

### AmyLoad
- **URL**: http://comprec-lin.iiar.pwr.edu.pl/amyload/
- **Description**: Consolidated amyloidogenic sequences
- **Size**: ~1,400 unique entries
- **Sources**: WALTZ-DB, AmylHex, AmylFrag, TANGO/AGGRESCAN validation sets
- **Reference**: Familia et al., Database 2015

**Our usage**: Core amyloid/non-amyloid sequences embedded in `benchmark_public.py`

---

## 3. Thermostability Datasets

### ProThermDB
- **URL**: https://web.iitm.ac.in/bioinfo2/prothermdb/
- **Description**: Largest thermodynamic database for proteins
- **Size**: 32,000+ proteins, ~120,000 data points
- **Parameters**: Tm, ΔG, ΔH, ΔCp
- **Format**: CSV export via web interface
- **Reference**: Nikam et al., Nucleic Acids Res. 2021

**Data includes:**
- 38% wild-type sequences
- 51% single point mutations
- Multiple organisms and experimental conditions

### FireProtDB
- **URL**: https://loschmidt.chemi.muni.cz/fireprotdb/
- **Description**: Manually curated protein stability data
- **Size**: Curated subset of ProTherm + additional sources
- **Format**: CSV, TSV, JSON, SQL dump
- **Reference**: Stourac et al., Nucleic Acids Res. 2021

**Export options:**
1. CSV - Spreadsheet compatible
2. TSV - Tab-separated
3. JSON Lines - For programmatic processing
4. SQL Dump - Full database recreation

**Our usage**: Representative proteins embedded in `benchmark_public.py`, with option to download full dataset.

---

## 4. Solubility Datasets

### CamSol Dataset
- **URL**: https://www-vendruscolo.ch.cam.ac.uk/camsolmethod.html
- **Description**: Experimentally validated solubility mutations
- **Size**: 56 mutations
- **Reference**: Sormanni et al., J Mol Biol 2015

### PSI:Biology / TargetTrack
- **URL**: https://targettrack.sbkb.org/ (archived)
- **Description**: Large-scale protein expression trials
- **Size**: 300,000+ proteins, 11,226 benchmark sequences
- **Format**: CSV (requires parsing)
- **Reference**: Multiple PSI consortium papers

**Benchmark datasets derived from PSI:Biology:**
- SolPro dataset
- PROSO II dataset
- NetSolP benchmark (11,226 sequences)

### SoluProtMutDB
- **URL**: https://loschmidt.chemi.muni.cz/soluprotmutdb/
- **Description**: Protein solubility changes upon mutations
- **Format**: CSV download
- **Reference**: Hon et al., Nucleic Acids Res. 2021

**Our usage**: Representative high/medium/low solubility proteins embedded in `benchmark_public.py`

---

## 5. Antibody Datasets

### FLAb2 (Fitness Landscape for Antibodies 2)
- **URL**: https://github.com/Graylab/FLAb
- **Description**: Largest public therapeutic antibody benchmark
- **Size**: >3 million data points, 32 studies
- **Format**: CSV files organized by category
- **Reference**: Chungyoun & Gray, bioRxiv 2025

**Categories:**
- `aggregation/` - AC-SINS, SEC, etc.
- `binding/` - Affinity measurements
- `expression/` - HEK, CHO titers
- `immunogenicity/` - ADA responses
- `pharmacokinetics/` - Half-life, clearance
- `polyreactivity/` - Non-specific binding
- `thermostability/` - Tm measurements

**Download**: `python benchmarks/download_flab.py`

### Jain 2017
- **URL**: Via FLAb repository
- **Description**: Gold standard clinical-stage antibodies
- **Size**: 137 mAbs (Phase I-III, FDA-approved)
- **Metrics**: Expression, AC-SINS, CSI-BLI, HIC
- **Reference**: Jain et al., PNAS 2017

**Our usage**: Primary antibody benchmark via `benchmark_jain2017.py`

---

## 6. Peptide Datasets

### BIOPEP-UWM
- **URL**: https://biochemia.uwm.edu.pl/biopep-uwm/
- **Description**: Bioactive peptides with proteolysis prediction
- **Features**: Protease cleavage site prediction, bioactivity annotation
- **Format**: Web interface (no bulk download)
- **Reference**: Minkiewicz et al., Int J Mol Sci 2019

**Our usage**: Protease cleavage rules derived from BIOPEP/PROSPER

### GLP-1 Analog Literature Data
- **Description**: Well-characterized therapeutic peptides
- **Source**: FDA labels, clinical papers
- **Reference**: Lau et al., Front Endocrinol 2019

**Embedded validation set:**

| Peptide | Half-life | DPP-4 Status | Position 2 | Source |
|---------|-----------|--------------|------------|--------|
| GLP-1 (native) | <2 min | Susceptible | Ala | Literature |
| Exenatide | 2.4 h | Resistant | Gly | FDA |
| Liraglutide | 13 h | Resistant | Ala + lipid | FDA |
| Semaglutide | 7 days | Resistant | Aib + lipid | FDA |
| Dulaglutide | 5 days | Resistant | Gly + Fc | FDA |

### Peptide Stability Study
- **Reference**: Jenssen & Bhargava, Peptides 2017
- **Description**: Differential stability in blood/plasma/serum
- **Data**: Multiple therapeutic peptides with measured half-lives

**Our usage**: Validation of proteolytic stability predictions

---

## 7. Expression Datasets

### RPOLP Dataset
- **Description**: E. coli recombinant protein expression
- **Size**: ~500 proteins
- **Accuracy**: 80% prediction accuracy reported
- **Reference**: Habibi et al., J Biotechnol 2015

### Nature 2024 Enzyme Benchmark
- **Description**: Neural network enzyme expression/activity
- **Size**: 500+ enzyme sequences
- **Reference**: Computational scoring of neural network enzymes, Nature Biotech 2024

---

## Download Scripts

### Available Scripts

| Script | Datasets | Size |
|--------|----------|------|
| `download_flab.py` | FLAb2, Jain 2017 | ~50 MB |
| `download_datasets.py` | All other datasets | ~10 MB |

### Usage

```bash
# Download FLAb2 antibody data
python benchmarks/download_flab.py

# Download all other benchmark datasets
python benchmarks/download_datasets.py

# Download specific category
python benchmarks/download_datasets.py --category aggregation
python benchmarks/download_datasets.py --category stability
python benchmarks/download_datasets.py --category peptide
```

### Data Directory Structure

```
benchmarks/data/
├── flab/                    # FLAb2 antibody data
│   ├── jain2017_expression.csv
│   ├── jain2017_acsins.csv
│   ├── jain2017_csibli.csv
│   └── garbinski2023_*.csv
├── aggregation/             # Aggregation datasets
│   ├── waltz_db.csv
│   └── amyload.csv
├── stability/               # Thermostability datasets
│   ├── protherm_subset.csv
│   └── fireprotdb.csv
├── solubility/              # Solubility datasets
│   └── psi_biology.csv
├── immunogenicity/          # Immunogenicity datasets
│   └── iedb_mhc1.csv
└── peptide/                 # Peptide datasets
    ├── glp1_analogs.csv
    └── dpp4_substrates.csv
```

---

## Embedded vs Downloaded Data

### Embedded in Code (No Download Required)
These datasets are small and curated directly in benchmark scripts:

- `benchmark_public.py`: IEDB subset, CamSol proteins, WALTZ core sequences, ProTherm representatives
- `benchmark_peptide.py`: GLP-1 analogs, DPP-4 substrates

### Requires Download
These datasets are larger and downloaded separately:

- FLAb2 / Jain 2017: `download_flab.py`
- Full ProThermDB: `download_datasets.py --category stability`
- Full WALTZ-DB: `download_datasets.py --category aggregation`

---

## Dataset Sizes Summary

| Category | Embedded | Downloaded | Total |
|----------|----------|------------|-------|
| Immunogenicity | 17 peptides | Optional IEDB | Small |
| Aggregation | 18 sequences | 1,416 (WALTZ) | ~100 KB |
| Stability | 7 proteins | 32,000+ (ProTherm) | ~5 MB |
| Solubility | 9 proteins | 11,226 (PSI) | ~2 MB |
| Antibody | - | 3M+ (FLAb) | ~50 MB |
| Peptide | 10 peptides | BIOPEP (web) | Small |

**Total download size**: ~60 MB (mostly FLAb2)

---

## References

### Primary Databases
1. **IEDB**: Vita R et al. (2019). Nucleic Acids Res. https://www.iedb.org/
2. **WALTZ-DB 2.0**: Louros N et al. (2020). Nucleic Acids Res. 48(D1):D389-D393
3. **AmyLoad**: Familia C et al. (2015). Database. bav021
4. **ProThermDB**: Nikam R et al. (2021). Nucleic Acids Res. 49(D1):D420-D424
5. **FireProtDB**: Stourac J et al. (2021). Nucleic Acids Res. 49(D1):D319-D324
6. **FLAb2**: Chungyoun M & Gray J (2025). bioRxiv

### Benchmark Papers
7. **Jain 2017**: Jain T et al. (2017). PNAS 114:944-949
8. **CamSol**: Sormanni P et al. (2015). J Mol Biol 427:478-490
9. **BIOPEP-UWM**: Minkiewicz P et al. (2019). Int J Mol Sci 20:5978
10. **GLP-1 Discovery**: Lau J et al. (2019). Front Endocrinol 10:155

### Prediction Method Papers
11. **TANGO**: Fernandez-Escamilla AM et al. (2004). Nat Biotechnol 22:1302-1306
12. **NetMHCpan**: Reynisson B et al. (2020). Nucleic Acids Res 48:W449-W454
13. **NetSolP**: Thumuluri V et al. (2022). Bioinformatics 38:941-946

---

## License and Usage

All datasets are publicly available for academic and research use. Commercial use may require separate licensing agreements with original data providers.

ProteinScore benchmark scripts are released under MIT License.
