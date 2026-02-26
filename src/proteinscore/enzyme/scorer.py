"""
Enzyme Developability Scorer

Unified scorer for enzyme developability assessment.
Combines thermostability, expression, aggregation, and solubility predictions.

Note: This module focuses on DEVELOPABILITY only.
Activity/function prediction is explicitly out of scope.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, computed_field

from proteinscore.enzyme.expression import (
    ExpressionHost,
    ExpressionPredictor,
    ExpressionResult,
)
from proteinscore.enzyme.thermostability import (
    EnzymeThermostabilityPredictor,
    ThermostabilityResult,
)
from proteinscore.models import (
    AggregationResult,
    ImmunogenicityResult,
    RiskLevel,
    SolubilityResult,
)
from proteinscore.predictors.aggregation import AggregationPredictor
from proteinscore.predictors.immunogenicity import ImmunogenicityPredictor
from proteinscore.predictors.solubility import SolubilityPredictor


class EnzymeResult(BaseModel):
    """Complete enzyme developability assessment result."""

    model_config = ConfigDict(frozen=True)

    # Overall score
    total_score: float = Field(ge=0, le=100, description="Overall developability score")

    # Component results
    thermostability: ThermostabilityResult
    expression: ExpressionResult
    solubility: SolubilityResult
    aggregation: AggregationResult
    immunogenicity: ImmunogenicityResult | None = Field(
        default=None, description="Immunogenicity (if API key provided)"
    )

    # Metadata
    sequence: str = Field(description="Input enzyme sequence")
    sequence_length: int = Field(ge=1, description="Sequence length")
    name: str | None = Field(default=None, description="Optional enzyme name")
    expression_host: ExpressionHost = Field(description="Target expression host")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    # Recommendations
    recommendations: list[str] = Field(
        default_factory=list, description="Actionable recommendations"
    )

    @computed_field
    @property
    def risk_level(self) -> RiskLevel:
        """Overall risk classification."""
        return RiskLevel.from_score(self.total_score)

    @computed_field
    @property
    def component_scores(self) -> dict[str, float]:
        """Dictionary of all component scores."""
        scores = {
            "thermostability": self.thermostability.score,
            "expression": self.expression.score,
            "solubility": self.solubility.score,
            "aggregation": self.aggregation.score,
        }
        if self.immunogenicity:
            scores["immunogenicity"] = self.immunogenicity.score
        return scores

    @computed_field
    @property
    def weakest_component(self) -> str:
        """The component with the lowest score."""
        scores = self.component_scores
        return min(scores, key=lambda k: scores[k])

    @computed_field
    @property
    def tm_estimate(self) -> float:
        """Estimated melting temperature."""
        return self.thermostability.tm_estimate

    @computed_field
    @property
    def expression_level(self) -> str:
        """Predicted expression level."""
        return self.expression.expression_level.value

    def summary(self) -> str:
        """Generate a text summary of the result."""
        lines = [
            f"Enzyme Developability Assessment: {self.name or 'Unnamed'}",
            f"{'=' * 50}",
            f"Total Score: {self.total_score:.1f}/100 ({self.risk_level.value.upper()} risk)",
            "",
            f"Melting Temperature: {self.tm_estimate:.1f}°C ({self.thermostability.thermostability_class.value})",
            f"Expression Host: {self.expression_host.value}",
            "",
            "Component Scores:",
            f"  Thermostability: {self.thermostability.score:5.1f}/100",
            f"  Expression:      {self.expression.score:5.1f}/100 ({self.expression_level})",
            f"  Solubility:      {self.solubility.score:5.1f}/100",
            f"  Aggregation:     {self.aggregation.score:5.1f}/100",
        ]

        if self.immunogenicity:
            lines.append(f"  Immunogenicity:  {self.immunogenicity.score:5.1f}/100")

        lines.extend(["", f"Weakest Component: {self.weakest_component}"])

        if self.recommendations:
            lines.extend(["", "Recommendations:"])
            for rec in self.recommendations[:5]:
                lines.append(f"  - {rec}")

        return "\n".join(lines)


class EnzymeScorer:
    """
    Unified enzyme developability scorer.

    Combines multiple predictors to provide comprehensive
    developability assessment for enzymes.

    Example:
        >>> scorer = EnzymeScorer(host=ExpressionHost.E_COLI)
        >>> result = scorer.score(enzyme_sequence)
        >>> print(f"Score: {result.total_score}/100")
        >>> print(f"Tm: {result.tm_estimate}°C")
    """

    def __init__(
        self,
        host: ExpressionHost = ExpressionHost.E_COLI,
        include_immunogenicity: bool = False,
        immunogenicity_api_key: str | None = None,
        weights: dict[str, float] | None = None,
    ) -> None:
        """
        Initialize the enzyme scorer.

        Args:
            host: Target expression host system
            include_immunogenicity: Whether to include immunogenicity prediction
            immunogenicity_api_key: API key for RIP immunogenicity prediction
            weights: Custom weights for score aggregation
        """
        self._host = host
        self._include_immunogenicity = include_immunogenicity
        self._immunogenicity_api_key = immunogenicity_api_key

        # Default weights optimized for enzyme developability
        self._weights = weights or {
            "thermostability": 0.30,
            "expression": 0.30,
            "solubility": 0.20,
            "aggregation": 0.15,
            "immunogenicity": 0.05,
        }

        # Initialize predictors
        self._thermo_predictor = EnzymeThermostabilityPredictor()
        self._expression_predictor = ExpressionPredictor(host=host)
        self._solubility_predictor = SolubilityPredictor()
        self._aggregation_predictor = AggregationPredictor()

        if include_immunogenicity:
            self._immunogenicity_predictor = ImmunogenicityPredictor(
                api_key=immunogenicity_api_key
            )
        else:
            self._immunogenicity_predictor = None

    def score(
        self,
        sequence: str,
        name: str | None = None,
    ) -> EnzymeResult:
        """
        Score an enzyme sequence for developability.

        Args:
            sequence: Enzyme amino acid sequence
            name: Optional name for the enzyme

        Returns:
            EnzymeResult with comprehensive developability assessment
        """
        # Validate and normalize sequence
        sequence = self._validate_sequence(sequence)

        # Run all predictions
        thermo_result = self._thermo_predictor.predict(sequence)
        expression_result = self._expression_predictor.predict(sequence)
        solubility_result = self._solubility_predictor.predict(sequence)
        aggregation_result = self._aggregation_predictor.predict(sequence)

        immuno_result = None
        if self._immunogenicity_predictor:
            immuno_result = self._immunogenicity_predictor.predict(sequence)

        # Calculate total score
        total_score = self._calculate_total_score(
            thermo_result.score,
            expression_result.score,
            solubility_result.score,
            aggregation_result.score,
            immuno_result.score if immuno_result else None,
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            thermo_result,
            expression_result,
            solubility_result,
            aggregation_result,
            immuno_result,
        )

        return EnzymeResult(
            total_score=round(total_score, 1),
            thermostability=thermo_result,
            expression=expression_result,
            solubility=solubility_result,
            aggregation=aggregation_result,
            immunogenicity=immuno_result,
            sequence=sequence,
            sequence_length=len(sequence),
            name=name,
            expression_host=self._host,
            recommendations=recommendations,
        )

    def score_batch(
        self,
        sequences: list[dict],
    ) -> list[EnzymeResult]:
        """
        Score multiple enzyme sequences.

        Args:
            sequences: List of dicts with 'sequence' and optional 'name' keys

        Returns:
            List of EnzymeResult objects
        """
        results = []
        for item in sequences:
            seq = item.get("sequence", item) if isinstance(item, dict) else item
            name = item.get("name") if isinstance(item, dict) else None
            results.append(self.score(seq, name=name))
        return results

    def _validate_sequence(self, sequence: str) -> str:
        """Validate and normalize enzyme sequence."""
        # Remove whitespace and convert to uppercase
        sequence = "".join(sequence.split()).upper()

        if not sequence:
            raise ValueError("Sequence cannot be empty")

        # Check for valid amino acids
        valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
        invalid = [aa for aa in sequence if aa not in valid_aa]
        if invalid:
            raise ValueError(f"Invalid amino acids: {set(invalid)}")

        return sequence

    def _calculate_total_score(
        self,
        thermo_score: float,
        expression_score: float,
        solubility_score: float,
        aggregation_score: float,
        immuno_score: float | None,
    ) -> float:
        """Calculate weighted total score."""
        # Get weights
        w = self._weights.copy()

        # If no immunogenicity, redistribute its weight
        if immuno_score is None:
            immuno_weight = w.pop("immunogenicity", 0)
            # Redistribute to other components proportionally
            remaining_total = sum(w.values())
            for key in w:
                w[key] += immuno_weight * (w[key] / remaining_total)

        # Calculate weighted score
        total = (
            w["thermostability"] * thermo_score +
            w["expression"] * expression_score +
            w["solubility"] * solubility_score +
            w["aggregation"] * aggregation_score
        )

        if immuno_score is not None:
            total += w.get("immunogenicity", 0.05) * immuno_score

        return total

    def _generate_recommendations(
        self,
        thermo: ThermostabilityResult,
        expression: ExpressionResult,
        solubility: SolubilityResult,
        aggregation: AggregationResult,
        immunogenicity: ImmunogenicityResult | None,
    ) -> list[str]:
        """Generate actionable recommendations based on scores."""
        recommendations = []

        # Thermostability recommendations
        if thermo.score < 50:
            recommendations.append(
                f"Low thermostability (Tm ~{thermo.tm_estimate:.0f}°C) - "
                "consider directed evolution or rational design for stability"
            )
        elif thermo.process_stability_score < 50:
            recommendations.append(
                "Limited process stability - consider process temperature optimization"
            )

        # Expression recommendations
        if expression.score < 50:
            recommendations.extend(expression.recommendations[:2])

        # Solubility recommendations
        if solubility.score < 50:
            recommendations.append(
                "Low predicted solubility - consider solubility-enhancing mutations "
                "or fusion tags"
            )

        # Aggregation recommendations
        if aggregation.score < 50:
            recommendations.append(
                f"High aggregation risk ({aggregation.apr_count} APRs detected) - "
                "consider gatekeeper mutations"
            )

        # Immunogenicity recommendations
        if immunogenicity and immunogenicity.score < 50:
            recommendations.append(
                "High immunogenicity risk - consider deimmunization if therapeutic use intended"
            )

        # Add general recommendations based on overall weakest component
        scores = {
            "thermostability": thermo.score,
            "expression": expression.score,
            "solubility": solubility.score,
            "aggregation": aggregation.score,
        }
        weakest = min(scores, key=lambda k: scores[k])
        if scores[weakest] < 40:
            recommendations.append(
                f"Priority: Address {weakest} issues before other optimizations"
            )

        return recommendations[:7]
