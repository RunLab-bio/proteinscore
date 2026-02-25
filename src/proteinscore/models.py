"""
ProteinScore Data Models

Pydantic models for type-safe, validated data structures.
All scores are normalized to 0-100 range.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, computed_field

if TYPE_CHECKING:
    import pandas as pd


class RiskLevel(str, Enum):
    """Risk classification based on total score."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

    @classmethod
    def from_score(cls, score: float) -> RiskLevel:
        """Classify risk level from a 0-100 score."""
        if score >= 70:
            return cls.LOW
        if score >= 50:
            return cls.MEDIUM
        return cls.HIGH


class SolubilityClass(str, Enum):
    """Solubility classification categories."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    AGGREGATION_PRONE = "aggregation_prone"

    @classmethod
    def from_score(cls, score: float) -> SolubilityClass:
        """Classify solubility from a 0-100 score."""
        if score >= 80:
            return cls.HIGH
        if score >= 60:
            return cls.MEDIUM
        if score >= 40:
            return cls.LOW
        return cls.AGGREGATION_PRONE


class Region(BaseModel):
    """A region of interest within a protein sequence."""

    model_config = ConfigDict(frozen=True)

    start: int = Field(ge=0, description="Start position (0-indexed)")
    end: int = Field(ge=0, description="End position (exclusive)")
    score: float = Field(ge=0, le=100, description="Region-specific score")
    sequence: str | None = Field(default=None, description="Sequence of the region")
    annotation: str | None = Field(default=None, description="Additional annotation")

    @computed_field
    @property
    def length(self) -> int:
        """Length of the region."""
        return self.end - self.start


class Epitope(BaseModel):
    """Predicted immunogenic epitope."""

    model_config = ConfigDict(frozen=True)

    peptide: str = Field(min_length=8, max_length=15, description="Epitope peptide sequence")
    position: tuple[int, int] = Field(description="Start and end position in parent sequence")
    allele: str = Field(description="HLA allele for this prediction")
    binding_affinity_nM: float = Field(ge=0, description="Predicted IC50 in nM")
    percentile_rank: float = Field(ge=0, le=100, description="Percentile rank (lower = stronger)")
    is_strong_binder: bool = Field(description="True if percentile_rank <= 0.5")
    is_weak_binder: bool = Field(description="True if percentile_rank <= 2.0")
    confidence: float = Field(ge=0, le=1, description="Model confidence score")


class Recommendation(BaseModel):
    """Actionable recommendation for improving developability."""

    model_config = ConfigDict(frozen=True)

    category: str = Field(description="Category: stability, solubility, aggregation, immunogenicity")
    severity: str = Field(description="Severity: critical, warning, info")
    message: str = Field(description="Human-readable recommendation")
    region: Region | None = Field(default=None, description="Affected region if applicable")
    suggested_mutations: list[str] | None = Field(
        default=None, description="Suggested mutations (e.g., 'S45A')"
    )


class StabilityResult(BaseModel):
    """Stability analysis result."""

    model_config = ConfigDict(frozen=True)

    score: float = Field(ge=0, le=100, description="Stability score (0-100)")
    ddg: float | None = Field(default=None, description="Delta-delta-G in kcal/mol")
    melting_temp_estimate: float | None = Field(default=None, description="Estimated Tm in Celsius")
    unstable_regions: list[Region] = Field(default_factory=list, description="Unstable regions")
    method: str = Field(description="Method used: foldx, rosetta, sequence_based")
    confidence: float = Field(ge=0, le=1, description="Prediction confidence")

    @computed_field
    @property
    def interpretation(self) -> str:
        """Human-readable interpretation of the score."""
        if self.score >= 80:
            return "Excellent stability - suitable for development"
        if self.score >= 60:
            return "Good stability - minor optimization may help"
        if self.score >= 40:
            return "Moderate stability - consider engineering"
        return "Poor stability - significant redesign recommended"


class SolubilityResult(BaseModel):
    """Solubility analysis result."""

    model_config = ConfigDict(frozen=True)

    score: float = Field(ge=0, le=100, description="Solubility score (0-100)")
    solubility_class: SolubilityClass = Field(description="Solubility classification")
    insoluble_regions: list[Region] = Field(default_factory=list, description="Insoluble regions")
    hydrophobicity_profile: list[float] = Field(
        default_factory=list, description="Per-residue hydrophobicity"
    )
    method: str = Field(description="Method used: camsol, protein_sol, sequence_based")
    confidence: float = Field(ge=0, le=1, description="Prediction confidence")

    @computed_field
    @property
    def interpretation(self) -> str:
        """Human-readable interpretation of the score."""
        if self.score >= 80:
            return "High solubility - excellent for production"
        if self.score >= 60:
            return "Medium solubility - may require optimization"
        if self.score >= 40:
            return "Low solubility - aggregation risk"
        return "Aggregation-prone - significant redesign needed"


class AggregationResult(BaseModel):
    """Aggregation propensity analysis result."""

    model_config = ConfigDict(frozen=True)

    score: float = Field(ge=0, le=100, description="Aggregation score (0-100, higher = better)")
    aggregation_prone_regions: list[Region] = Field(
        default_factory=list, description="Aggregation-prone regions (APRs)"
    )
    amyloid_propensity: float = Field(ge=0, le=1, description="Overall amyloid propensity")
    method: str = Field(description="Method used: aggrescan3d, tango, sequence_based")
    confidence: float = Field(ge=0, le=1, description="Prediction confidence")

    @computed_field
    @property
    def apr_count(self) -> int:
        """Number of aggregation-prone regions."""
        return len(self.aggregation_prone_regions)

    @computed_field
    @property
    def interpretation(self) -> str:
        """Human-readable interpretation of the score."""
        if self.score >= 80:
            return "Low aggregation risk - suitable for development"
        if self.score >= 60:
            return "Moderate aggregation risk - monitor during development"
        if self.score >= 40:
            return "High aggregation risk - engineering recommended"
        return "Very high aggregation risk - significant redesign needed"


class ImmunogenicityResult(BaseModel):
    """Immunogenicity analysis result from RIP API."""

    model_config = ConfigDict(frozen=True)

    score: float = Field(ge=0, le=100, description="Immunogenicity score (0-100, higher = better)")
    epitope_count: int = Field(ge=0, description="Total predicted epitopes")
    strong_binders: int = Field(ge=0, description="Count of strong binders (<=0.5%)")
    weak_binders: int = Field(ge=0, description="Count of weak binders (<=2.0%)")
    epitopes: list[Epitope] = Field(default_factory=list, description="Predicted epitopes")
    per_residue_risk: list[float] = Field(
        default_factory=list, description="Per-residue immunogenicity risk (0-1)"
    )
    population_coverage: float = Field(ge=0, le=1, description="HLA population coverage")
    method: str = Field(description="Method used: rip_api, local_estimate")
    confidence: float = Field(ge=0, le=1, description="Prediction confidence")

    @computed_field
    @property
    def high_risk_positions(self) -> list[int]:
        """Positions with risk > 0.5."""
        return [i for i, risk in enumerate(self.per_residue_risk) if risk > 0.5]

    @computed_field
    @property
    def interpretation(self) -> str:
        """Human-readable interpretation of the score."""
        if self.score >= 80:
            return "Low immunogenicity risk - favorable for therapeutics"
        if self.score >= 60:
            return "Moderate immunogenicity - consider deimmunization"
        if self.score >= 40:
            return "High immunogenicity - deimmunization recommended"
        return "Very high immunogenicity - significant engineering needed"


class ScoringWeights(BaseModel):
    """Weights for combining component scores."""

    model_config = ConfigDict(frozen=True)

    stability: float = Field(default=0.25, ge=0, le=1)
    solubility: float = Field(default=0.25, ge=0, le=1)
    aggregation: float = Field(default=0.25, ge=0, le=1)
    immunogenicity: float = Field(default=0.25, ge=0, le=1)

    def model_post_init(self, __context: Any) -> None:
        """Validate weights sum to 1.0."""
        total = self.stability + self.solubility + self.aggregation + self.immunogenicity
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Weights must sum to 1.0, got {total}")

    @classmethod
    def for_enzyme(cls) -> ScoringWeights:
        """Preset weights optimized for enzyme design."""
        return cls(stability=0.4, solubility=0.2, aggregation=0.2, immunogenicity=0.2)

    @classmethod
    def for_antibody(cls) -> ScoringWeights:
        """Preset weights optimized for antibody therapeutics."""
        return cls(stability=0.25, solubility=0.25, aggregation=0.25, immunogenicity=0.25)

    @classmethod
    def for_vaccine(cls) -> ScoringWeights:
        """Preset weights optimized for vaccine antigens (maximize immunogenicity)."""
        return cls(stability=0.3, solubility=0.2, aggregation=0.2, immunogenicity=0.3)


class ProteinScoreResult(BaseModel):
    """Complete developability score result."""

    model_config = ConfigDict(frozen=True)

    # Core scores
    total_score: float = Field(ge=0, le=100, description="Overall ProteinScore (0-100)")
    stability: StabilityResult
    solubility: SolubilityResult
    aggregation: AggregationResult
    immunogenicity: ImmunogenicityResult

    # Metadata
    sequence: str = Field(min_length=1, description="Input protein sequence")
    sequence_length: int = Field(ge=1, description="Length of sequence")
    name: str | None = Field(default=None, description="Optional protein name")
    structure_provided: bool = Field(default=False, description="Whether structure was provided")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    weights_used: ScoringWeights = Field(default_factory=ScoringWeights)

    # Recommendations
    recommendations: list[Recommendation] = Field(
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
        return {
            "stability": self.stability.score,
            "solubility": self.solubility.score,
            "aggregation": self.aggregation.score,
            "immunogenicity": self.immunogenicity.score,
        }

    @computed_field
    @property
    def weakest_component(self) -> str:
        """The component with the lowest score."""
        scores = self.component_scores
        return min(scores, key=lambda k: scores[k])

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump(mode="json")

    def to_json(self, path: str) -> None:
        """Export result to JSON file."""
        import json
        from pathlib import Path

        Path(path).write_text(json.dumps(self.to_dict(), indent=2, default=str))

    def to_csv(self, path: str) -> None:
        """Export result to CSV file (flat format)."""
        import csv
        from pathlib import Path

        flat = {
            "name": self.name,
            "sequence_length": self.sequence_length,
            "total_score": self.total_score,
            "risk_level": self.risk_level.value,
            "stability_score": self.stability.score,
            "solubility_score": self.solubility.score,
            "aggregation_score": self.aggregation.score,
            "immunogenicity_score": self.immunogenicity.score,
            "weakest_component": self.weakest_component,
            "timestamp": self.timestamp.isoformat(),
        }

        with Path(path).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=flat.keys())
            writer.writeheader()
            writer.writerow(flat)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        import pandas as pd

        return pd.DataFrame([self.to_dict()])

    def summary(self) -> str:
        """Generate a text summary of the result."""
        lines = [
            f"ProteinScore Result: {self.name or 'Unnamed'}",
            f"{'=' * 40}",
            f"Total Score: {self.total_score:.1f}/100 ({self.risk_level.value.upper()} risk)",
            "",
            "Component Scores:",
            f"  Stability:      {self.stability.score:5.1f}/100 - {self.stability.interpretation}",
            f"  Solubility:     {self.solubility.score:5.1f}/100 - {self.solubility.interpretation}",
            f"  Aggregation:    {self.aggregation.score:5.1f}/100 - {self.aggregation.interpretation}",
            f"  Immunogenicity: {self.immunogenicity.score:5.1f}/100 - {self.immunogenicity.interpretation}",
            "",
            f"Weakest Component: {self.weakest_component}",
        ]

        if self.recommendations:
            lines.extend(["", "Recommendations:"])
            for rec in self.recommendations[:5]:
                lines.append(f"  [{rec.severity.upper()}] {rec.message}")

        return "\n".join(lines)
