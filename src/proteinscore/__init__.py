"""
ProteinScore: Unified Protein Developability Scorer

SOTA scoring combining stability, solubility, aggregation, and immunogenicity
into a single 0-100 developability score.

ProteinScore™ is a trademark of RunLab.

Example:
    >>> from proteinscore import ProteinScore
    >>> scorer = ProteinScore()
    >>> result = scorer.score("MKTAYIAKQRQISFVK...")
    >>> print(f"ProteinScore: {result.total_score}/100")
"""

from proteinscore.config import Config, ScoringWeights
from proteinscore.exceptions import (
    APIError,
    AuthenticationError,
    InvalidAlleleError,
    InvalidSequenceError,
    ProteinScoreError,
    RateLimitError,
    ValidationError,
)
from proteinscore.models import (
    AggregationResult,
    Epitope,
    ImmunogenicityResult,
    ProteinScoreResult,
    Recommendation,
    Region,
    RiskLevel,
    SolubilityClass,
    SolubilityResult,
    StabilityResult,
)
from proteinscore.scorer import ProteinScore

__version__ = "0.1.0"
__all__ = [
    # Main class
    "ProteinScore",
    # Configuration
    "Config",
    "ScoringWeights",
    # Result models
    "ProteinScoreResult",
    "StabilityResult",
    "SolubilityResult",
    "AggregationResult",
    "ImmunogenicityResult",
    "Epitope",
    "Region",
    "Recommendation",
    "RiskLevel",
    "SolubilityClass",
    # Exceptions
    "ProteinScoreError",
    "ValidationError",
    "InvalidSequenceError",
    "InvalidAlleleError",
    "APIError",
    "AuthenticationError",
    "RateLimitError",
]
