# ProteinScore Python Library

> **Unified Protein Developability Scoring**
>
> `pip install proteinscore`

---

## Installation

### Basic Installation

```bash
pip install proteinscore
```

### With Optional Dependencies

```bash
# For local stability calculations (FoldX wrapper)
pip install proteinscore[stability]

# For local solubility calculations
pip install proteinscore[solubility]

# For local aggregation calculations
pip install proteinscore[aggregation]

# All local predictors
pip install proteinscore[all]
```

---

## Quick Start

```python
from proteinscore import ProteinScore

# Initialize (uses anonymous tier by default)
scorer = ProteinScore()

# Score a protein sequence
result = scorer.score("MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQQIAAALEHHHHHH")

# Print overall score
print(f"ProteinScore: {result.total_score}/100")

# Print component scores
print(f"Stability: {result.stability.score}/100")
print(f"Solubility: {result.solubility.score}/100")
print(f"Aggregation: {result.aggregation.score}/100")
print(f"Immunogenicity: {result.immunogenicity.score}/100")
```

---

## Configuration

### API Key

```python
from proteinscore import ProteinScore

# Option 1: Pass directly
scorer = ProteinScore(api_key="rl_your_api_key")

# Option 2: Environment variable
# export RUNLAB_API_KEY=rl_your_api_key
scorer = ProteinScore()

# Option 3: Config file (~/.proteinscore/config.yaml)
# api_key: rl_your_api_key
scorer = ProteinScore()
```

### Custom Weights

```python
from proteinscore import ProteinScore, ScoringWeights

# Emphasize stability for enzyme design
weights = ScoringWeights(
    stability=0.4,      # 40%
    solubility=0.2,     # 20%
    aggregation=0.2,    # 20%
    immunogenicity=0.2  # 20%
)

scorer = ProteinScore(weights=weights)
```

### HLA Alleles for Immunogenicity

```python
from proteinscore import ProteinScore

# Use specific alleles (e.g., for specific patient)
scorer = ProteinScore(
    hla_alleles=["HLA-A*02:01", "HLA-A*24:02", "HLA-B*07:02"]
)

# Use population-representative set
scorer = ProteinScore(hla_population="european")
```

---

## API Reference

### ProteinScore Class

```python
class ProteinScore:
    def __init__(
        self,
        api_key: Optional[str] = None,
        weights: Optional[ScoringWeights] = None,
        hla_alleles: Optional[List[str]] = None,
        hla_population: str = "global",
        local_only: bool = False,
        cache_results: bool = True,
    ):
        """
        Initialize ProteinScore scorer.

        Args:
            api_key: RunLab API key (optional, uses anonymous tier if not provided)
            weights: Custom scoring weights (default: equal weights)
            hla_alleles: Specific HLA alleles for immunogenicity (default: population-based)
            hla_population: Population for HLA selection (global, european, asian, etc.)
            local_only: Skip immunogenicity API calls (use local estimation)
            cache_results: Cache API responses locally
        """
```

### score() Method

```python
def score(
    self,
    sequence: str,
    structure: Optional[str] = None,
    name: Optional[str] = None,
) -> ProteinScoreResult:
    """
    Calculate developability score for a protein.

    Args:
        sequence: Amino acid sequence (one-letter code)
        structure: Optional PDB structure (path or string)
        name: Optional name for the protein

    Returns:
        ProteinScoreResult with component scores and overall score
    """
```

### score_batch() Method

```python
def score_batch(
    self,
    sequences: List[str],
    structures: Optional[List[str]] = None,
    names: Optional[List[str]] = None,
    parallel: bool = True,
) -> List[ProteinScoreResult]:
    """
    Calculate developability scores for multiple proteins.

    Args:
        sequences: List of amino acid sequences
        structures: Optional list of PDB structures
        names: Optional list of protein names
        parallel: Use parallel processing (default: True)

    Returns:
        List of ProteinScoreResult objects
    """
```

### compare() Method

```python
def compare(
    self,
    results: List[ProteinScoreResult],
    output_format: str = "table",
) -> Union[str, pd.DataFrame]:
    """
    Compare multiple scoring results.

    Args:
        results: List of ProteinScoreResult objects
        output_format: "table", "dataframe", or "json"

    Returns:
        Comparison table or DataFrame
    """
```

---

## Result Objects

### ProteinScoreResult

```python
@dataclass
class ProteinScoreResult:
    """Complete developability score result."""

    # Overall score (0-100)
    total_score: float

    # Component scores
    stability: StabilityResult
    solubility: SolubilityResult
    aggregation: AggregationResult
    immunogenicity: ImmunogenicityResult

    # Metadata
    sequence: str
    name: Optional[str]
    structure: Optional[str]
    timestamp: datetime
    weights_used: ScoringWeights

    # Recommendations
    recommendations: List[Recommendation]
    risk_level: str  # "low", "medium", "high"
```

### StabilityResult

```python
@dataclass
class StabilityResult:
    """Stability analysis result."""

    score: float  # 0-100
    ddg: float  # kcal/mol (if structure provided)
    melting_temp_estimate: float  # Celsius
    unstable_regions: List[Region]
    method: str  # "foldx", "rosetta", "sequence_based"
```

### SolubilityResult

```python
@dataclass
class SolubilityResult:
    """Solubility analysis result."""

    score: float  # 0-100
    solubility_class: str  # "high", "medium", "low"
    insoluble_regions: List[Region]
    hydrophobicity_profile: List[float]
    method: str  # "camsol", "protein_sol", "sequence_based"
```

### AggregationResult

```python
@dataclass
class AggregationResult:
    """Aggregation propensity result."""

    score: float  # 0-100
    aggregation_prone_regions: List[Region]
    amyloid_propensity: float
    method: str  # "aggrescan3d", "tango", "sequence_based"
```

### ImmunogenicityResult

```python
@dataclass
class ImmunogenicityResult:
    """Immunogenicity analysis result."""

    score: float  # 0-100
    epitope_count: int
    strong_binders: int
    weak_binders: int
    epitopes: List[Epitope]
    per_residue_risk: List[float]
    population_coverage: float
    method: str  # "rip_api", "local_estimate"
```

---

## Use Cases

### 1. Basic Protein Assessment

```python
from proteinscore import ProteinScore

scorer = ProteinScore()

# Score a therapeutic antibody
sequence = """
EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYA
DSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDRLSITIRPRYYGLDVWGQGTLVTVSS
"""

result = scorer.score(sequence.replace("\n", ""))

print(f"Total ProteinScore: {result.total_score:.1f}/100")
print(f"Risk Level: {result.risk_level}")
print(f"\nRecommendations:")
for rec in result.recommendations:
    print(f"  - {rec.message}")
```

### 2. Compare Multiple Variants

```python
from proteinscore import ProteinScore

scorer = ProteinScore()

variants = {
    "Wild-type": "MKTAYIAKQRQISFVK...",
    "S45A": "MKTAYIAKQRQIAFVK...",
    "K32R": "MKTAYIARQRQISFVK...",
}

results = []
for name, seq in variants.items():
    result = scorer.score(seq, name=name)
    results.append(result)

# Compare
comparison = scorer.compare(results, output_format="table")
print(comparison)
```

Output:
```
+------------+-------+----------+-----------+-------------+---------------+
| Variant    | Total | Stability| Solubility| Aggregation | Immunogenicity|
+------------+-------+----------+-----------+-------------+---------------+
| Wild-type  | 72.5  | 75.0     | 68.0      | 80.0        | 67.0          |
| S45A       | 78.2  | 82.0     | 70.0      | 82.0        | 79.0          |
| K32R       | 69.8  | 70.0     | 65.0      | 78.0        | 66.0          |
+------------+-------+----------+-----------+-------------+---------------+
```

### 3. Batch Processing

```python
from proteinscore import ProteinScore
import pandas as pd

scorer = ProteinScore(api_key="rl_your_key")

# Load sequences from file
df = pd.read_csv("variants.csv")

# Score all variants
results = scorer.score_batch(
    sequences=df["sequence"].tolist(),
    names=df["name"].tolist(),
    parallel=True
)

# Add scores to dataframe
df["proteinscore"] = [r.total_score for r in results]
df["risk_level"] = [r.risk_level for r in results]

# Filter promising candidates
promising = df[df["proteinscore"] >= 70]
print(f"Found {len(promising)} promising variants out of {len(df)}")
```

### 4. Focus on Immunogenicity

```python
from proteinscore import ProteinScore

# Patient-specific HLA alleles
patient_hla = [
    "HLA-A*02:01",
    "HLA-A*24:02",
    "HLA-B*07:02",
    "HLA-B*44:02",
]

scorer = ProteinScore(hla_alleles=patient_hla)

result = scorer.score(therapeutic_sequence)

print(f"Immunogenicity Score: {result.immunogenicity.score}/100")
print(f"Strong Binders: {result.immunogenicity.strong_binders}")
print(f"Population Coverage: {result.immunogenicity.population_coverage:.1%}")

# Identify problematic epitopes
for epitope in result.immunogenicity.epitopes[:5]:
    print(f"  {epitope.peptide} @ {epitope.position} - {epitope.allele}")
```

### 5. Integration with Structure Prediction

```python
from proteinscore import ProteinScore
import esmfold  # or alphafold

scorer = ProteinScore()

# Predict structure
structure = esmfold.predict(sequence)

# Score with structure (enables better stability/aggregation)
result = scorer.score(sequence, structure=structure)

print(f"Stability (structure-based): {result.stability.score}/100")
print(f"Method used: {result.stability.method}")
```

### 6. Export Results

```python
from proteinscore import ProteinScore

scorer = ProteinScore()
result = scorer.score(sequence)

# Export to JSON
result.to_json("result.json")

# Export to CSV (flat format)
result.to_csv("result.csv")

# Export detailed report
result.to_report("report.html", format="html")
result.to_report("report.pdf", format="pdf")
```

---

## Advanced Configuration

### Caching

```python
from proteinscore import ProteinScore

# Enable persistent caching
scorer = ProteinScore(
    cache_results=True,
    cache_dir="~/.proteinscore/cache",
    cache_ttl=86400  # 24 hours
)

# Clear cache
scorer.clear_cache()
```

### Logging

```python
import logging
from proteinscore import ProteinScore

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

scorer = ProteinScore()
```

### Custom Predictors

```python
from proteinscore import ProteinScore, BasePreditor

class MyStabilityPredictor(BasePredictor):
    def predict(self, sequence: str, structure: Optional[str]) -> float:
        # Your custom stability prediction
        return stability_score

scorer = ProteinScore(
    stability_predictor=MyStabilityPredictor()
)
```

---

## Error Handling

```python
from proteinscore import ProteinScore
from proteinscore.exceptions import (
    RateLimitError,
    InvalidSequenceError,
    APIError,
)

scorer = ProteinScore()

try:
    result = scorer.score(sequence)
except RateLimitError as e:
    print(f"Rate limit exceeded. Retry after: {e.retry_after}")
except InvalidSequenceError as e:
    print(f"Invalid sequence: {e.message}")
    print(f"Invalid characters: {e.invalid_chars}")
except APIError as e:
    print(f"API error: {e.message}")
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `RUNLAB_API_KEY` | RunLab API key | None (anonymous) |
| `PROTEINSCORE_CACHE_DIR` | Cache directory | `~/.proteinscore/cache` |
| `PROTEINSCORE_LOG_LEVEL` | Logging level | `INFO` |
| `PROTEINSCORE_TIMEOUT` | API timeout (seconds) | `30` |

---

## Requirements

- Python 3.9+
- requests
- numpy
- pandas (optional, for DataFrame output)
- biopython (optional, for structure parsing)

---

## License

ProteinScore is released under the **MIT License**.

The RunLab RIP API is a proprietary service with free tier access.

---

*Python Library Documentation v1.0 — 2026-02-25*
