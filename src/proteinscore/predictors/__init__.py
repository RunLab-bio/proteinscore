"""
ProteinScore Predictors

SOTA prediction modules for protein developability assessment.
"""

from proteinscore.predictors.base import BasePredictor
from proteinscore.predictors.stability import StabilityPredictor
from proteinscore.predictors.solubility import SolubilityPredictor
from proteinscore.predictors.aggregation import AggregationPredictor
from proteinscore.predictors.immunogenicity import ImmunogenicityPredictor

__all__ = [
    "BasePredictor",
    "StabilityPredictor",
    "SolubilityPredictor",
    "AggregationPredictor",
    "ImmunogenicityPredictor",
]
