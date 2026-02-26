"""
Enzyme-Specific Developability Analysis

Comprehensive enzyme sequence analysis for industrial/therapeutic development:

Features:
- Thermostability prediction (Tm estimation, r = 0.70-0.80)
- Expression prediction (E. coli/CHO, 60-74% accuracy)
- Aggregation and solubility from base predictors
- Unified scoring (0-100 scale)

Note: This module focuses on DEVELOPABILITY metrics only.
Activity/function prediction is explicitly out of scope.

References:
    - Nature Scientific Reports 2025: Gradient Boosting Tm prediction
    - SoluProt (Bioinformatics 2021): Solubility/expression
    - SOLpro (Bioinformatics 2009): Expression prediction
    - ProTherm database: Thermostability validation
"""

from proteinscore.enzyme.thermostability import (
    EnzymeThermostabilityPredictor,
    ThermostabilityResult,
)
from proteinscore.enzyme.expression import (
    ExpressionPredictor,
    ExpressionResult,
    ExpressionHost,
)
from proteinscore.enzyme.scorer import EnzymeScorer, EnzymeResult

__all__ = [
    # Thermostability
    "EnzymeThermostabilityPredictor",
    "ThermostabilityResult",
    # Expression
    "ExpressionPredictor",
    "ExpressionResult",
    "ExpressionHost",
    # Full Scorer
    "EnzymeScorer",
    "EnzymeResult",
]
