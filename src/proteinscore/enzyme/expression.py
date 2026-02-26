"""
Enzyme Expression Predictor

Predicts expression success in common host systems (E. coli, CHO, Pichia).
Based on SoluProt/SOLpro algorithms with host-specific adjustments.

Target accuracy: 60-74% (binary classification)

References:
    - SoluProt (Bioinformatics 2021): 58.5% accuracy, AUC 0.62
    - SOLpro (Bioinformatics 2009): 74.2% accuracy
    - TargetTrack database: Expression data
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from proteinscore.predictors.base import BasePredictor

if TYPE_CHECKING:
    from proteinscore.config import PredictorConfig


class ExpressionHost(str, Enum):
    """Expression host system."""

    E_COLI = "e_coli"
    CHO = "cho"
    PICHIA = "pichia"
    INSECT = "insect"
    YEAST = "yeast"


class ExpressionLevel(str, Enum):
    """Predicted expression level."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INSOLUBLE = "insoluble"

    @classmethod
    def from_score(cls, score: float) -> ExpressionLevel:
        """Classify expression level from score."""
        if score >= 75:
            return cls.HIGH
        if score >= 50:
            return cls.MEDIUM
        if score >= 25:
            return cls.LOW
        return cls.INSOLUBLE


class ExpressionResult(BaseModel):
    """Result of expression prediction."""

    model_config = ConfigDict(frozen=True)

    score: float = Field(ge=0, le=100, description="Expression score (0-100)")
    expression_level: ExpressionLevel = Field(description="Predicted expression level")
    host: ExpressionHost = Field(description="Target host system")
    solubility_score: float = Field(
        ge=0, le=100, description="Predicted solubility in host"
    )
    codon_adaptation_index: float | None = Field(
        default=None, description="CAI if codon sequence provided"
    )
    problem_regions: list[dict] = Field(
        default_factory=list, description="Regions that may cause expression issues"
    )
    recommendations: list[str] = Field(
        default_factory=list, description="Suggestions for improving expression"
    )
    method: str = Field(description="Method used for prediction")
    confidence: float = Field(ge=0, le=1, description="Prediction confidence")

    @property
    def interpretation(self) -> str:
        """Human-readable interpretation."""
        host_name = self.host.value.replace("_", " ").upper()
        if self.score >= 75:
            return f"High expression expected in {host_name}"
        if self.score >= 50:
            return f"Moderate expression expected in {host_name}"
        if self.score >= 25:
            return f"Low expression likely in {host_name} - optimization recommended"
        return f"Expression failure likely in {host_name} - consider alternative host"


# SoluProt-derived amino acid scores for E. coli expression
SOLUPROT_ECOLI = {
    "A": 0.04, "R": 0.08, "N": -0.02, "D": 0.06, "C": -0.12,
    "Q": 0.01, "E": 0.09, "G": -0.01, "H": 0.02, "I": -0.05,
    "L": -0.03, "K": 0.10, "M": -0.04, "F": -0.08, "P": 0.03,
    "S": 0.02, "T": 0.01, "W": -0.10, "Y": -0.06, "V": -0.02,
}

# CHO-specific adjustments (mammalian folding, glycosylation)
CHO_ADJUSTMENTS = {
    "N": 0.03,  # N-glycosylation beneficial
    "S": 0.02,  # O-glycosylation
    "T": 0.02,  # O-glycosylation
    "C": 0.05,  # Disulfides form properly in CHO
    "W": -0.02,  # Less problematic than E. coli
    "F": -0.02,
}

# Pichia-specific adjustments
PICHIA_ADJUSTMENTS = {
    "N": 0.02,  # Glycosylation
    "K": -0.02,  # Slightly lower than E. coli
    "R": -0.02,
    "C": 0.03,  # Disulfides form
}

# Rare codon positions (for E. coli)
ECOLI_RARE_CODONS = {"R": ["AGA", "AGG", "CGA"], "I": ["AUA"], "L": ["CUA"]}

# N-terminal secretion signal effects
SECRETION_SIGNALS = {
    "MKFL": 0.05,  # OmpA-like
    "MKKL": 0.04,
    "MKKT": 0.04,
    "ATGK": 0.03,
}

# Aggregation-prone regions threshold
HYDROPHOBIC_STRETCH_THRESHOLD = 7


class ExpressionPredictor(BasePredictor[ExpressionResult]):
    """
    Expression success predictor for enzymes.

    Predicts likelihood of successful expression in common host systems
    based on sequence features and host-specific factors.

    Target accuracy: 60-74%
    """

    def __init__(
        self,
        config: PredictorConfig | None = None,
        host: ExpressionHost = ExpressionHost.E_COLI,
    ) -> None:
        super().__init__(config)
        self._host = host

    @property
    def name(self) -> str:
        return f"Expression Predictor ({self._host.value})"

    @property
    def method(self) -> str:
        return "soluprot_derived"

    def predict(
        self,
        sequence: str,
        structure: str | None = None,
    ) -> ExpressionResult:
        """
        Predict expression success for an enzyme sequence.

        Args:
            sequence: Enzyme sequence (validated)
            structure: Optional PDB structure (not used)

        Returns:
            ExpressionResult with expression score and recommendations
        """
        self.validate_inputs(sequence, structure)

        # Calculate base expression score (SoluProt-like)
        base_score = self._calculate_base_score(sequence)

        # Apply host-specific adjustments
        host_adjusted = self._apply_host_adjustments(sequence, base_score)

        # Calculate solubility component
        solubility_score = self._calculate_solubility_score(sequence)

        # Identify problem regions
        problem_regions = self._identify_problem_regions(sequence)

        # Calculate penalties
        region_penalty = len(problem_regions) * 3
        length_penalty = self._calculate_length_penalty(len(sequence))

        # Final score
        final_score = host_adjusted + solubility_score * 0.3 - region_penalty - length_penalty
        final_score = max(0, min(100, final_score))

        # Generate recommendations
        recommendations = self._generate_recommendations(
            sequence, problem_regions, final_score
        )

        # Determine expression level
        expression_level = ExpressionLevel.from_score(final_score)

        # Calculate confidence
        confidence = self._calculate_confidence(len(sequence), len(problem_regions))

        return ExpressionResult(
            score=round(final_score, 1),
            expression_level=expression_level,
            host=self._host,
            solubility_score=round(solubility_score, 1),
            codon_adaptation_index=None,  # Requires codon sequence
            problem_regions=problem_regions,
            recommendations=recommendations,
            method=self.method,
            confidence=round(confidence, 2),
        )

    def _calculate_base_score(self, sequence: str) -> float:
        """Calculate base expression score using SoluProt features."""
        length = len(sequence)

        # Calculate amino acid composition score
        aa_score = sum(SOLUPROT_ECOLI.get(aa, 0.0) for aa in sequence)
        aa_score = aa_score / length * 100  # Normalize

        # Map to 0-100 scale (typical range: -0.05 to +0.05)
        base = 50 + aa_score * 500

        # Adjust for sequence length (optimal: 200-400 AA)
        if 200 <= length <= 400:
            base += 5
        elif length < 100:
            base += 3  # Small proteins often express well
        elif length > 600:
            base -= 5  # Large proteins harder

        return max(0, min(100, base))

    def _apply_host_adjustments(self, sequence: str, base_score: float) -> float:
        """Apply host-specific adjustments."""
        length = len(sequence)

        if self._host == ExpressionHost.E_COLI:
            # E. coli specific: check for rare codons (AA proxies)
            rare_penalty = 0
            arginine_fraction = sequence.count("R") / length
            if arginine_fraction > 0.08:
                rare_penalty += 5

            isoleucine_fraction = sequence.count("I") / length
            if isoleucine_fraction > 0.06:
                rare_penalty += 3

            # Check for membrane-spanning regions (many hydrophobic)
            hydrophobic = sum(1 for aa in sequence if aa in "VILMFYW")
            if hydrophobic / length > 0.45:
                rare_penalty += 5

            return base_score - rare_penalty

        elif self._host == ExpressionHost.CHO:
            # CHO: mammalian, handles complexity better
            adjustment = sum(
                CHO_ADJUSTMENTS.get(aa, 0.0) for aa in sequence
            ) / length * 50

            # Glycosylation sites (N-X-S/T motif)
            glyco_sites = 0
            for i in range(len(sequence) - 2):
                if sequence[i] == "N" and sequence[i + 1] != "P":
                    if sequence[i + 2] in "ST":
                        glyco_sites += 1

            # Some glycosylation is good, too much is problematic
            if 1 <= glyco_sites <= 5:
                adjustment += 5
            elif glyco_sites > 10:
                adjustment -= 5

            return base_score + adjustment

        elif self._host == ExpressionHost.PICHIA:
            adjustment = sum(
                PICHIA_ADJUSTMENTS.get(aa, 0.0) for aa in sequence
            ) / length * 50

            return base_score + adjustment

        return base_score

    def _calculate_solubility_score(self, sequence: str) -> float:
        """Calculate solubility component of expression score."""
        length = len(sequence)

        # Charged residues (solubilizing)
        charged = sum(1 for aa in sequence if aa in "DEKR")
        charged_fraction = charged / length

        # Hydrophobic residues (aggregation-prone)
        hydrophobic = sum(1 for aa in sequence if aa in "VILMFYW")
        hydrophobic_fraction = hydrophobic / length

        # Calculate base solubility
        solubility = 50 + charged_fraction * 100 - hydrophobic_fraction * 80

        # Penalize long hydrophobic stretches
        max_stretch = self._find_max_hydrophobic_stretch(sequence)
        if max_stretch >= HYDROPHOBIC_STRETCH_THRESHOLD:
            solubility -= (max_stretch - HYDROPHOBIC_STRETCH_THRESHOLD + 1) * 5

        return max(0, min(100, solubility))

    def _find_max_hydrophobic_stretch(self, sequence: str) -> int:
        """Find the longest stretch of hydrophobic residues."""
        hydrophobic = set("VILMFYW")
        max_stretch = 0
        current = 0

        for aa in sequence:
            if aa in hydrophobic:
                current += 1
                max_stretch = max(max_stretch, current)
            else:
                current = 0

        return max_stretch

    def _identify_problem_regions(self, sequence: str) -> list[dict]:
        """Identify regions that may cause expression problems."""
        problems = []
        length = len(sequence)

        # Check for long hydrophobic stretches
        hydrophobic = set("VILMFYW")
        stretch_start = None
        stretch_length = 0

        for i, aa in enumerate(sequence):
            if aa in hydrophobic:
                if stretch_start is None:
                    stretch_start = i
                stretch_length += 1
            else:
                if stretch_length >= HYDROPHOBIC_STRETCH_THRESHOLD:
                    problems.append({
                        "type": "hydrophobic_stretch",
                        "start": stretch_start,
                        "end": stretch_start + stretch_length,
                        "sequence": sequence[stretch_start:stretch_start + stretch_length],
                        "severity": "high" if stretch_length >= 10 else "medium",
                    })
                stretch_start = None
                stretch_length = 0

        # Check final stretch
        if stretch_length >= HYDROPHOBIC_STRETCH_THRESHOLD:
            problems.append({
                "type": "hydrophobic_stretch",
                "start": stretch_start,
                "end": stretch_start + stretch_length,
                "sequence": sequence[stretch_start:stretch_start + stretch_length],
                "severity": "high" if stretch_length >= 10 else "medium",
            })

        # Check for cysteine pairs (potential disulfide issues in E. coli)
        if self._host == ExpressionHost.E_COLI:
            cysteine_count = sequence.count("C")
            if cysteine_count >= 4:
                problems.append({
                    "type": "multiple_cysteines",
                    "count": cysteine_count,
                    "severity": "medium",
                    "note": "Multiple cysteines may form incorrect disulfides in E. coli cytoplasm",
                })

        # Check for low complexity regions
        for i in range(0, length - 10, 5):
            window = sequence[i:i + 10]
            unique_aa = len(set(window))
            if unique_aa <= 3:
                problems.append({
                    "type": "low_complexity",
                    "start": i,
                    "end": i + 10,
                    "sequence": window,
                    "severity": "low",
                })

        return problems[:10]  # Limit to top 10

    def _calculate_length_penalty(self, length: int) -> float:
        """Calculate penalty based on sequence length."""
        if length < 50:
            return 5  # Very short, may be unstable
        if length <= 100:
            return 0
        if length <= 400:
            return 0
        if length <= 600:
            return 3
        if length <= 800:
            return 7
        return 12  # Very large proteins

    def _generate_recommendations(
        self,
        sequence: str,
        problems: list[dict],
        score: float,
    ) -> list[str]:
        """Generate recommendations for improving expression."""
        recommendations = []

        if score < 50:
            recommendations.append(
                "Consider using a solubility-enhancing fusion tag (MBP, SUMO, GST)"
            )

        # Host-specific recommendations
        if self._host == ExpressionHost.E_COLI:
            cysteine_count = sequence.count("C")
            if cysteine_count >= 2:
                recommendations.append(
                    "Consider Shuffle or Origami strains for correct disulfide formation"
                )

            if len(sequence) > 500:
                recommendations.append(
                    "Large protein - consider codon optimization and low temperature expression"
                )

        elif self._host == ExpressionHost.CHO:
            if score < 60:
                recommendations.append(
                    "Consider signal peptide optimization for secretion"
                )

        # Problem-specific recommendations
        for problem in problems:
            if problem["type"] == "hydrophobic_stretch":
                recommendations.append(
                    f"Hydrophobic region at {problem['start']}-{problem['end']} may cause aggregation"
                )
            elif problem["type"] == "multiple_cysteines":
                recommendations.append(
                    "Multiple cysteines present - verify disulfide pattern"
                )

        return recommendations[:5]

    def _calculate_confidence(
        self, length: int, problem_count: int
    ) -> float:
        """Calculate prediction confidence."""
        # Base confidence by length
        if length < 100:
            confidence = 0.60
        elif length <= 500:
            confidence = 0.70
        else:
            confidence = 0.60

        # Reduce confidence for many problems
        if problem_count > 3:
            confidence -= 0.05
        if problem_count > 6:
            confidence -= 0.05

        return max(0.5, min(0.80, confidence))
