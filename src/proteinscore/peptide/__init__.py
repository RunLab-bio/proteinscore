"""
Peptide Module

Comprehensive developability assessment for therapeutic peptides.

This module provides:
- Proteolytic stability prediction (protease cleavage sites, half-life)
- Chemical liability detection (oxidation, deamidation, isomerization)
- Unified scoring combining all developability factors

Example:
    >>> from proteinscore.peptide import PeptideScorer
    >>> scorer = PeptideScorer()
    >>> result = scorer.score("HAEGTFTSDVSSYLEGQAAK")
    >>> print(f"Developability Score: {result.total_score}/100")
    >>> print(f"Half-life: {result.estimated_half_life}")

Note:
    This module focuses on DEVELOPABILITY only.
    Activity/binding prediction is explicitly out of scope.
"""

from proteinscore.peptide.chemical_liability import (
    ChemicalLiability,
    ChemicalLiabilityResult,
    ChemicalLiabilityScanner,
    ChemicalLiabilityType,
    LiabilityProfile,
    LiabilitySeverity,
)
from proteinscore.peptide.scorer import (
    PeptideResult,
    PeptideScorer,
)
from proteinscore.peptide.stability import (
    CleavageAnalysis,
    CleavageSeverity,
    CleavageSite,
    PeptideStabilityPredictor,
    PeptideStabilityResult,
    ProteaseType,
)

__all__ = [
    # Main scorer
    "PeptideScorer",
    "PeptideResult",
    # Stability
    "PeptideStabilityPredictor",
    "PeptideStabilityResult",
    "CleavageSite",
    "CleavageAnalysis",
    "ProteaseType",
    "CleavageSeverity",
    # Chemical liability
    "ChemicalLiabilityScanner",
    "ChemicalLiabilityResult",
    "ChemicalLiability",
    "LiabilityProfile",
    "ChemicalLiabilityType",
    "LiabilitySeverity",
]
