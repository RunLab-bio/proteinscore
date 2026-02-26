"""
Peptide Developability Scorer

Unified scorer for peptide therapeutic developability assessment.
Combines proteolytic stability, chemical liability, solubility, and aggregation predictions.

Note: This module focuses on DEVELOPABILITY only.
Activity/binding prediction is explicitly out of scope.

References:
    - Fosgerau & Hoffmann (2015) Drug Discovery Today - Peptide therapeutics
    - Henninot et al. (2018) J Med Chem - Current and future challenges
    - Di (2015) AAPS J - Strategies for peptide drug delivery
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, computed_field

from proteinscore.models import (
    AggregationResult,
    ImmunogenicityResult,
    RiskLevel,
    SolubilityResult,
)
from proteinscore.peptide.chemical_liability import (
    ChemicalLiabilityResult,
    ChemicalLiabilityScanner,
)
from proteinscore.peptide.stability import (
    PeptideStabilityPredictor,
    PeptideStabilityResult,
)
from proteinscore.predictors.aggregation import AggregationPredictor
from proteinscore.predictors.immunogenicity import ImmunogenicityPredictor
from proteinscore.predictors.solubility import SolubilityPredictor


class PeptideResult(BaseModel):
    """Complete peptide developability assessment result."""

    model_config = ConfigDict(frozen=True)

    # Overall score
    total_score: float = Field(ge=0, le=100, description="Overall developability score")

    # Component results
    stability: PeptideStabilityResult
    chemical_liability: ChemicalLiabilityResult
    solubility: SolubilityResult
    aggregation: AggregationResult
    immunogenicity: ImmunogenicityResult | None = Field(
        default=None, description="Immunogenicity (if API key provided)"
    )

    # Metadata
    sequence: str = Field(description="Input peptide sequence")
    sequence_length: int = Field(ge=1, description="Sequence length")
    name: str | None = Field(default=None, description="Optional peptide name")
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
            "stability": self.stability.score,
            "chemical_liability": self.chemical_liability.score,
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
    def estimated_half_life(self) -> str:
        """Estimated plasma half-life category."""
        return self.stability.estimated_half_life_category

    @computed_field
    @property
    def is_dpp4_susceptible(self) -> bool:
        """Whether peptide is susceptible to DPP-4 cleavage."""
        return self.stability.dpp4_susceptible

    @computed_field
    @property
    def high_risk_liabilities(self) -> int:
        """Total high-risk chemical and proteolytic sites."""
        return (
            self.stability.high_risk_sites +
            self.chemical_liability.high_risk_sites
        )

    def summary(self) -> str:
        """Generate a text summary of the result."""
        lines = [
            f"Peptide Developability Assessment: {self.name or 'Unnamed'}",
            f"{'=' * 50}",
            f"Total Score: {self.total_score:.1f}/100 ({self.risk_level.value.upper()} risk)",
            "",
            f"Sequence Length: {self.sequence_length} amino acids",
            f"Estimated Half-life: {self.estimated_half_life}",
            "",
            "Component Scores:",
            f"  Proteolytic Stability:  {self.stability.score:5.1f}/100",
            f"  Chemical Stability:     {self.chemical_liability.score:5.1f}/100",
            f"  Solubility:             {self.solubility.score:5.1f}/100",
            f"  Aggregation:            {self.aggregation.score:5.1f}/100",
        ]

        if self.immunogenicity:
            lines.append(f"  Immunogenicity:         {self.immunogenicity.score:5.1f}/100")

        # Key risk indicators
        lines.extend([
            "",
            "Risk Indicators:",
            f"  DPP-4 Susceptible: {'Yes' if self.is_dpp4_susceptible else 'No'}",
            f"  High-Risk Sites: {self.high_risk_liabilities}",
            f"  Cleavage Sites: {self.stability.total_cleavage_sites}",
            f"  Chemical Liabilities: {self.chemical_liability.total_liabilities}",
        ])

        lines.extend(["", f"Weakest Component: {self.weakest_component}"])

        if self.recommendations:
            lines.extend(["", "Recommendations:"])
            for rec in self.recommendations[:5]:
                lines.append(f"  - {rec}")

        return "\n".join(lines)


class PeptideScorer:
    """
    Unified peptide developability scorer.

    Combines multiple predictors to provide comprehensive
    developability assessment for therapeutic peptides.

    Optimized for peptides (typically 5-50 amino acids) with focus on:
    - Proteolytic stability (protease cleavage, half-life)
    - Chemical stability (oxidation, deamidation, isomerization)
    - Solubility and aggregation propensity
    - Immunogenicity (optional, requires API key)

    Example:
        >>> scorer = PeptideScorer()
        >>> result = scorer.score("HAEGTFTSDVSSYLEGQAAK")
        >>> print(f"Score: {result.total_score}/100")
        >>> print(f"Half-life: {result.estimated_half_life}")
    """

    # Typical peptide length range
    MIN_LENGTH = 3
    MAX_LENGTH = 100

    def __init__(
        self,
        include_immunogenicity: bool = False,
        immunogenicity_api_key: str | None = None,
        weights: dict[str, float] | None = None,
        plasma_context: bool = True,
    ) -> None:
        """
        Initialize the peptide scorer.

        Args:
            include_immunogenicity: Whether to include immunogenicity prediction
            immunogenicity_api_key: API key for RIP immunogenicity prediction
            weights: Custom weights for score aggregation
            plasma_context: Include plasma-relevant proteases (DPP4, NEP, ACE)
        """
        self._include_immunogenicity = include_immunogenicity
        self._immunogenicity_api_key = immunogenicity_api_key

        # Default weights optimized for peptide developability
        # Stability is critical for peptides due to rapid degradation
        self._weights = weights or {
            "stability": 0.35,
            "chemical_liability": 0.25,
            "solubility": 0.20,
            "aggregation": 0.15,
            "immunogenicity": 0.05,
        }

        # Initialize predictors
        self._stability_predictor = PeptideStabilityPredictor(
            plasma_context=plasma_context
        )
        self._liability_scanner = ChemicalLiabilityScanner()
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
    ) -> PeptideResult:
        """
        Score a peptide sequence for developability.

        Args:
            sequence: Peptide amino acid sequence
            name: Optional name for the peptide

        Returns:
            PeptideResult with comprehensive developability assessment
        """
        # Validate and normalize sequence
        sequence = self._validate_sequence(sequence)

        # Run all predictions
        stability_result = self._stability_predictor.predict(sequence)
        liability_result = self._liability_scanner.scan(sequence)
        solubility_result = self._solubility_predictor.predict(sequence)
        aggregation_result = self._aggregation_predictor.predict(sequence)

        immuno_result = None
        if self._immunogenicity_predictor:
            immuno_result = self._immunogenicity_predictor.predict(sequence)

        # Calculate total score
        total_score = self._calculate_total_score(
            stability_result.score,
            liability_result.score,
            solubility_result.score,
            aggregation_result.score,
            immuno_result.score if immuno_result else None,
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            stability_result,
            liability_result,
            solubility_result,
            aggregation_result,
            immuno_result,
        )

        return PeptideResult(
            total_score=round(total_score, 1),
            stability=stability_result,
            chemical_liability=liability_result,
            solubility=solubility_result,
            aggregation=aggregation_result,
            immunogenicity=immuno_result,
            sequence=sequence,
            sequence_length=len(sequence),
            name=name,
            recommendations=recommendations,
        )

    def score_batch(
        self,
        sequences: list[dict],
    ) -> list[PeptideResult]:
        """
        Score multiple peptide sequences.

        Args:
            sequences: List of dicts with 'sequence' and optional 'name' keys

        Returns:
            List of PeptideResult objects
        """
        results = []
        for item in sequences:
            seq = item.get("sequence", item) if isinstance(item, dict) else item
            name = item.get("name") if isinstance(item, dict) else None
            results.append(self.score(seq, name=name))
        return results

    def _validate_sequence(self, sequence: str) -> str:
        """Validate and normalize peptide sequence."""
        # Remove whitespace and convert to uppercase
        sequence = "".join(sequence.split()).upper()

        if not sequence:
            raise ValueError("Sequence cannot be empty")

        if len(sequence) < self.MIN_LENGTH:
            raise ValueError(
                f"Sequence too short ({len(sequence)} aa). "
                f"Minimum length: {self.MIN_LENGTH}"
            )

        if len(sequence) > self.MAX_LENGTH:
            raise ValueError(
                f"Sequence too long ({len(sequence)} aa). "
                f"Maximum length: {self.MAX_LENGTH}. "
                "For longer proteins, use EnzymeScorer or AntibodyScorer."
            )

        # Check for valid amino acids
        valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
        invalid = [aa for aa in sequence if aa not in valid_aa]
        if invalid:
            raise ValueError(f"Invalid amino acids: {set(invalid)}")

        return sequence

    def _calculate_total_score(
        self,
        stability_score: float,
        liability_score: float,
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
            w["stability"] * stability_score +
            w["chemical_liability"] * liability_score +
            w["solubility"] * solubility_score +
            w["aggregation"] * aggregation_score
        )

        if immuno_score is not None:
            total += w.get("immunogenicity", 0.05) * immuno_score

        return total

    def _generate_recommendations(
        self,
        stability: PeptideStabilityResult,
        liability: ChemicalLiabilityResult,
        solubility: SolubilityResult,
        aggregation: AggregationResult,
        immunogenicity: ImmunogenicityResult | None,
    ) -> list[str]:
        """Generate actionable recommendations based on scores."""
        recommendations = []

        # Proteolytic stability recommendations (highest priority for peptides)
        if stability.score < 50:
            if stability.dpp4_susceptible:
                recommendations.append(
                    "Critical: DPP-4 susceptible - modify position 2 to prevent rapid degradation"
                )
            recommendations.extend(stability.stabilization_strategies[:2])

        # Chemical liability recommendations
        if liability.score < 50:
            if liability.has_asn_gly:
                recommendations.append(
                    "High-priority: Replace Asn-Gly motif to prevent rapid deamidation"
                )
            if liability.has_met_oxidation:
                recommendations.append(
                    "Consider Met → Leu/Nle substitution to prevent oxidation"
                )
            recommendations.extend(liability.engineering_recommendations[:1])

        # Solubility recommendations
        if solubility.score < 50:
            recommendations.append(
                "Low solubility - consider adding charged residues or PEGylation"
            )

        # Aggregation recommendations
        if aggregation.score < 50:
            recommendations.append(
                f"Aggregation risk ({aggregation.apr_count} APRs) - "
                "consider sequence redesign or formulation additives"
            )

        # Immunogenicity recommendations
        if immunogenicity and immunogenicity.score < 50:
            recommendations.append(
                "High immunogenicity risk - consider PEGylation or sequence modification"
            )

        # General stability strategies based on weakest component
        scores = {
            "stability": stability.score,
            "chemical_liability": liability.score,
            "solubility": solubility.score,
            "aggregation": aggregation.score,
        }
        weakest = min(scores, key=lambda k: scores[k])
        if scores[weakest] < 40:
            recommendations.append(
                f"Priority: Address {weakest} issues before proceeding to development"
            )

        # Peptide-specific general recommendations
        if stability.estimated_half_life_category.startswith("very short"):
            recommendations.append(
                "Consider lipidation or extended half-life technologies (PEG, Fc fusion)"
            )

        return recommendations[:7]


__all__ = [
    "PeptideScorer",
    "PeptideResult",
]
