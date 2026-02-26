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

from proteinscore.config import Config, ScoringWeights, get_default_config
from proteinscore.exceptions import (
    APIError,
    AuthenticationError,
    CacheError,
    ConfigurationError,
    ConflictError,
    InvalidAlleleError,
    InvalidSequenceError,
    NotFoundError,
    PredictorError,
    ProteinScoreError,
    RateLimitError,
    ServiceUnavailableError,
    ValidationError,
)
from proteinscore.logging import bind_context, clear_context, configure_logging, get_logger
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

__version__ = "0.2.0"
__all__ = [
    # Main class
    "ProteinScore",
    # Configuration
    "Config",
    "ScoringWeights",
    "get_default_config",
    # Logging
    "configure_logging",
    "get_logger",
    "bind_context",
    "clear_context",
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
    "PredictorError",
    "ConfigurationError",
    "CacheError",
    "NotFoundError",
    "ConflictError",
    "ServiceUnavailableError",
]
