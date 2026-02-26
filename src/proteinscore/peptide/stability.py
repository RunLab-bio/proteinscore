"""
Peptide Stability Prediction

Proteolytic degradation susceptibility analysis for therapeutic peptides.

Predicts:
- Protease cleavage sites (trypsin, chymotrypsin, pepsin, etc.)
- Half-life estimation in plasma/serum
- Stability score (0-100, higher = more stable)

References:
    - PROSPER (Bioinformatics 2012): Protease specificity prediction
    - PeptideCutter (ExPASy): Protease cleavage rules
    - Werle & Bernkop-Schnürch (2006): Strategies for peptide stability
    - Di (2015): Strategic approaches for peptide drug delivery
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

from pydantic import BaseModel, ConfigDict, Field, computed_field


class ProteaseType(Enum):
    """Types of proteases that can cleave peptides."""

    TRYPSIN = "trypsin"
    CHYMOTRYPSIN = "chymotrypsin"
    PEPSIN = "pepsin"
    ELASTASE = "elastase"
    THERMOLYSIN = "thermolysin"
    AMINOPEPTIDASE = "aminopeptidase"
    CARBOXYPEPTIDASE = "carboxypeptidase"
    DPP4 = "dpp4"  # Dipeptidyl peptidase-4
    NEP = "nep"    # Neprilysin
    ACE = "ace"    # Angiotensin-converting enzyme


class CleavageSeverity(Enum):
    """Severity of cleavage site impact."""

    HIGH = "high"      # Primary site, rapid cleavage
    MEDIUM = "medium"  # Secondary site, moderate cleavage
    LOW = "low"        # Minor site, slow cleavage


# Protease cleavage rules based on literature
# Format: (P1 residues, P1' residues, specificity)
PROTEASE_RULES: dict[ProteaseType, dict] = {
    ProteaseType.TRYPSIN: {
        "p1_residues": {"K", "R"},
        "p1_prime_exclude": {"P"},
        "specificity": "high",
        "severity": CleavageSeverity.HIGH,
        "description": "Cleaves after K/R (not before P)",
    },
    ProteaseType.CHYMOTRYPSIN: {
        "p1_residues": {"F", "Y", "W", "L"},
        "p1_prime_exclude": {"P"},
        "specificity": "high",
        "severity": CleavageSeverity.HIGH,
        "description": "Cleaves after aromatic/large hydrophobic residues",
    },
    ProteaseType.PEPSIN: {
        "p1_residues": {"F", "L", "W", "Y"},
        "p1_prime_residues": {"F", "L", "W", "Y", "A", "E"},
        "specificity": "medium",
        "severity": CleavageSeverity.MEDIUM,
        "description": "Cleaves between hydrophobic residues (acidic pH)",
    },
    ProteaseType.ELASTASE: {
        "p1_residues": {"A", "V", "S", "G", "L", "I"},
        "p1_prime_exclude": {"P"},
        "specificity": "medium",
        "severity": CleavageSeverity.MEDIUM,
        "description": "Cleaves after small aliphatic residues",
    },
    ProteaseType.DPP4: {
        "n_terminal_pattern": True,
        "p2_residues": {"X"},  # Any
        "p1_residues": {"P", "A"},
        "specificity": "high",
        "severity": CleavageSeverity.HIGH,
        "description": "Removes N-terminal dipeptides (X-P or X-A)",
    },
    ProteaseType.AMINOPEPTIDASE: {
        "n_terminal": True,
        "susceptible_residues": {"A", "L", "M", "F", "Y", "K", "R"},
        "specificity": "high",
        "severity": CleavageSeverity.MEDIUM,
        "description": "Removes N-terminal residues sequentially",
    },
    ProteaseType.CARBOXYPEPTIDASE: {
        "c_terminal": True,
        "susceptible_residues": {"A", "L", "M", "F", "Y", "K", "R"},
        "cpa_targets": {"A", "L", "I", "V", "F", "Y", "W"},  # CPA
        "cpb_targets": {"K", "R"},  # CPB
        "specificity": "high",
        "severity": CleavageSeverity.MEDIUM,
        "description": "Removes C-terminal residues",
    },
}


class CleavageSite(NamedTuple):
    """A predicted protease cleavage site."""

    position: int
    protease: ProteaseType
    p1_residue: str
    p1_prime_residue: str | None
    severity: CleavageSeverity
    probability: float


@dataclass
class CleavageAnalysis:
    """Analysis of all cleavage sites in a peptide."""

    total_sites: int
    high_severity_sites: int
    medium_severity_sites: int
    low_severity_sites: int
    sites_by_protease: dict[ProteaseType, int]
    cleavage_sites: list[CleavageSite]

    @property
    def weighted_site_count(self) -> float:
        """Weighted count giving more weight to high severity sites."""
        return (
            self.high_severity_sites * 3.0 +
            self.medium_severity_sites * 1.5 +
            self.low_severity_sites * 0.5
        )


class PeptideStabilityResult(BaseModel):
    """Result of peptide stability prediction."""

    model_config = ConfigDict(frozen=True)

    score: float = Field(ge=0, le=100, description="Stability score (0-100, higher = more stable)")

    # Cleavage analysis
    total_cleavage_sites: int = Field(ge=0, description="Total predicted cleavage sites")
    high_risk_sites: int = Field(ge=0, description="High-severity cleavage sites")
    cleavage_sites: list[dict] = Field(default_factory=list, description="Detailed cleavage sites")

    # Half-life estimation
    estimated_half_life_category: str = Field(description="Estimated plasma half-life category")
    half_life_minutes: float | None = Field(default=None, description="Estimated half-life in minutes")

    # Terminal susceptibility
    n_terminal_susceptible: bool = Field(description="N-terminus susceptible to aminopeptidases")
    c_terminal_susceptible: bool = Field(description="C-terminus susceptible to carboxypeptidases")
    dpp4_susceptible: bool = Field(description="Susceptible to DPP-4 cleavage")

    # Recommendations
    stabilization_strategies: list[str] = Field(
        default_factory=list, description="Recommended stabilization strategies"
    )

    method: str = Field(default="sequence_based", description="Prediction method")
    confidence: float = Field(ge=0, le=1, description="Prediction confidence")

    @computed_field
    @property
    def interpretation(self) -> str:
        """Human-readable interpretation."""
        if self.score >= 80:
            return "High stability - minimal proteolytic susceptibility"
        if self.score >= 60:
            return "Moderate stability - some proteolytic sites present"
        if self.score >= 40:
            return "Low stability - significant proteolytic susceptibility"
        return "Very low stability - highly susceptible to degradation"


class PeptideStabilityPredictor:
    """
    Predicts peptide stability based on proteolytic susceptibility.

    Analyzes:
    - Protease cleavage sites (trypsin, chymotrypsin, etc.)
    - N/C-terminal susceptibility
    - DPP-4 susceptibility (important for GLP-1 analogs)

    Example:
        >>> predictor = PeptideStabilityPredictor()
        >>> result = predictor.predict("HAEGTFTSDVS")
        >>> print(f"Stability: {result.score}/100")
    """

    def __init__(
        self,
        include_proteases: list[ProteaseType] | None = None,
        plasma_context: bool = True,
    ) -> None:
        """
        Initialize predictor.

        Args:
            include_proteases: Specific proteases to analyze (default: all)
            plasma_context: Include plasma-relevant proteases (DPP4, NEP)
        """
        if include_proteases:
            self.proteases = include_proteases
        else:
            self.proteases = list(ProteaseType)
            if not plasma_context:
                self.proteases = [
                    p for p in self.proteases
                    if p not in {ProteaseType.DPP4, ProteaseType.NEP, ProteaseType.ACE}
                ]

    def predict(self, sequence: str) -> PeptideStabilityResult:
        """
        Predict peptide stability.

        Args:
            sequence: Peptide sequence (typically 5-50 amino acids)

        Returns:
            PeptideStabilityResult with stability score and analysis
        """
        sequence = sequence.upper().strip()

        # Analyze cleavage sites
        cleavage_analysis = self._analyze_cleavage_sites(sequence)

        # Analyze terminal susceptibility
        n_terminal_susc = self._check_n_terminal_susceptibility(sequence)
        c_terminal_susc = self._check_c_terminal_susceptibility(sequence)
        dpp4_susc = self._check_dpp4_susceptibility(sequence)

        # Calculate stability score
        score = self._calculate_stability_score(
            sequence,
            cleavage_analysis,
            n_terminal_susc,
            c_terminal_susc,
            dpp4_susc,
        )

        # Estimate half-life
        half_life_cat, half_life_min = self._estimate_half_life(
            cleavage_analysis, n_terminal_susc, c_terminal_susc, dpp4_susc
        )

        # Generate stabilization strategies
        strategies = self._generate_strategies(
            sequence, cleavage_analysis, n_terminal_susc, c_terminal_susc, dpp4_susc
        )

        # Convert cleavage sites to dicts for serialization
        sites_data = [
            {
                "position": site.position,
                "protease": site.protease.value,
                "p1_residue": site.p1_residue,
                "p1_prime_residue": site.p1_prime_residue,
                "severity": site.severity.value,
                "probability": site.probability,
            }
            for site in cleavage_analysis.cleavage_sites
        ]

        # Calculate confidence based on sequence length
        confidence = min(0.85, 0.5 + len(sequence) * 0.01)

        return PeptideStabilityResult(
            score=round(score, 1),
            total_cleavage_sites=cleavage_analysis.total_sites,
            high_risk_sites=cleavage_analysis.high_severity_sites,
            cleavage_sites=sites_data,
            estimated_half_life_category=half_life_cat,
            half_life_minutes=half_life_min,
            n_terminal_susceptible=n_terminal_susc,
            c_terminal_susceptible=c_terminal_susc,
            dpp4_susceptible=dpp4_susc,
            stabilization_strategies=strategies,
            method="sequence_based",
            confidence=confidence,
        )

    def _analyze_cleavage_sites(self, sequence: str) -> CleavageAnalysis:
        """Analyze all protease cleavage sites in the sequence."""
        sites: list[CleavageSite] = []
        sites_by_protease: dict[ProteaseType, int] = {p: 0 for p in self.proteases}

        for protease in self.proteases:
            rules = PROTEASE_RULES.get(protease)
            if not rules:
                continue

            # Skip terminal-specific proteases in main loop
            if rules.get("n_terminal") or rules.get("c_terminal"):
                continue
            if rules.get("n_terminal_pattern"):
                # DPP4 specific handling
                if len(sequence) >= 2:
                    if sequence[1] in rules.get("p1_residues", set()):
                        sites.append(CleavageSite(
                            position=1,
                            protease=protease,
                            p1_residue=sequence[1],
                            p1_prime_residue=sequence[2] if len(sequence) > 2 else None,
                            severity=rules["severity"],
                            probability=0.9,
                        ))
                        sites_by_protease[protease] += 1
                continue

            # Standard cleavage site analysis
            p1_residues = rules.get("p1_residues", set())
            p1_prime_exclude = rules.get("p1_prime_exclude", set())
            p1_prime_residues = rules.get("p1_prime_residues")

            for i, residue in enumerate(sequence[:-1]):  # Don't check last position
                if residue in p1_residues:
                    p1_prime = sequence[i + 1]

                    # Check exclusions
                    if p1_prime in p1_prime_exclude:
                        continue

                    # Check required P1' residues if specified
                    if p1_prime_residues and p1_prime not in p1_prime_residues:
                        continue

                    # Calculate probability based on context
                    prob = self._calculate_cleavage_probability(
                        sequence, i, protease, rules
                    )

                    sites.append(CleavageSite(
                        position=i,
                        protease=protease,
                        p1_residue=residue,
                        p1_prime_residue=p1_prime,
                        severity=rules["severity"],
                        probability=prob,
                    ))
                    sites_by_protease[protease] += 1

        # Count by severity
        high = sum(1 for s in sites if s.severity == CleavageSeverity.HIGH)
        medium = sum(1 for s in sites if s.severity == CleavageSeverity.MEDIUM)
        low = sum(1 for s in sites if s.severity == CleavageSeverity.LOW)

        return CleavageAnalysis(
            total_sites=len(sites),
            high_severity_sites=high,
            medium_severity_sites=medium,
            low_severity_sites=low,
            sites_by_protease=sites_by_protease,
            cleavage_sites=sites,
        )

    def _calculate_cleavage_probability(
        self,
        sequence: str,
        position: int,
        protease: ProteaseType,
        rules: dict,
    ) -> float:
        """Calculate probability of cleavage at a specific site."""
        base_prob = 0.7 if rules["specificity"] == "high" else 0.5

        # Context modifiers
        # Proline at P2' position often reduces cleavage
        if position + 2 < len(sequence) and sequence[position + 2] == "P":
            base_prob *= 0.5

        # Charged residues near site can affect accessibility
        window = sequence[max(0, position - 2):min(len(sequence), position + 3)]
        charged = sum(1 for r in window if r in "DEKRH")
        if charged >= 3:
            base_prob *= 0.8  # Reduced accessibility

        return min(base_prob, 1.0)

    def _check_n_terminal_susceptibility(self, sequence: str) -> bool:
        """Check if N-terminus is susceptible to aminopeptidases."""
        if not sequence:
            return False

        susceptible_residues = {"A", "L", "M", "F", "Y", "K", "R", "S", "T"}
        return sequence[0] in susceptible_residues

    def _check_c_terminal_susceptibility(self, sequence: str) -> bool:
        """Check if C-terminus is susceptible to carboxypeptidases."""
        if not sequence:
            return False

        # CPA targets aromatic/aliphatic, CPB targets basic
        susceptible_residues = {"A", "L", "I", "V", "F", "Y", "W", "K", "R"}
        return sequence[-1] in susceptible_residues

    def _check_dpp4_susceptibility(self, sequence: str) -> bool:
        """Check if peptide is susceptible to DPP-4 cleavage."""
        if len(sequence) < 2:
            return False

        # DPP-4 cleaves X-Pro or X-Ala dipeptides from N-terminus
        return sequence[1] in {"P", "A"}

    def _calculate_stability_score(
        self,
        sequence: str,
        cleavage: CleavageAnalysis,
        n_term_susc: bool,
        c_term_susc: bool,
        dpp4_susc: bool,
    ) -> float:
        """Calculate overall stability score (0-100)."""
        # Start with perfect score
        score = 100.0

        # Penalize cleavage sites (weighted)
        # More sites = lower score, high severity = more penalty
        site_penalty = cleavage.weighted_site_count * 5
        score -= min(site_penalty, 40)  # Cap at 40 points

        # Normalize by sequence length (longer peptides naturally have more sites)
        length_factor = len(sequence) / 10  # Normalize to ~10 residues
        if length_factor > 1:
            score += min(site_penalty * (1 - 1/length_factor) * 0.5, 15)

        # Terminal susceptibility penalties
        if n_term_susc:
            score -= 10
        if c_term_susc:
            score -= 8
        if dpp4_susc:
            score -= 15  # DPP-4 is clinically significant

        # Bonus for protective features
        # N-terminal pyroglutamate formation (E/Q at N-term)
        if sequence and sequence[0] in {"E", "Q"}:
            score += 5  # Can form protective pyroglutamate

        # C-terminal amidation potential
        if sequence and sequence[-1] == "G":
            score += 3  # Glycine can be amidated

        # D-amino acids would provide stability but we can't detect them
        # Cyclic peptides would be stable but we can't detect cyclization

        return max(0, min(100, score))

    def _estimate_half_life(
        self,
        cleavage: CleavageAnalysis,
        n_term_susc: bool,
        c_term_susc: bool,
        dpp4_susc: bool,
    ) -> tuple[str, float | None]:
        """Estimate plasma half-life category and approximate minutes."""
        # Risk factors
        risk_score = 0
        risk_score += cleavage.high_severity_sites * 3
        risk_score += cleavage.medium_severity_sites * 1.5
        risk_score += cleavage.low_severity_sites * 0.5
        if n_term_susc:
            risk_score += 2
        if c_term_susc:
            risk_score += 1.5
        if dpp4_susc:
            risk_score += 4  # DPP-4 is very active in plasma

        # Categorize
        if risk_score <= 2:
            return "long (>60 min)", 120.0
        elif risk_score <= 5:
            return "moderate (15-60 min)", 30.0
        elif risk_score <= 10:
            return "short (5-15 min)", 10.0
        else:
            return "very short (<5 min)", 3.0

    def _generate_strategies(
        self,
        sequence: str,
        cleavage: CleavageAnalysis,
        n_term_susc: bool,
        c_term_susc: bool,
        dpp4_susc: bool,
    ) -> list[str]:
        """Generate stabilization strategies based on analysis."""
        strategies: list[str] = []

        # Terminal modifications
        if n_term_susc:
            strategies.append("N-terminal acetylation or PEGylation to protect from aminopeptidases")
        if c_term_susc:
            strategies.append("C-terminal amidation to protect from carboxypeptidases")
        if dpp4_susc:
            strategies.append("Modify position 2 (P/A → other) to prevent DPP-4 cleavage")
            if sequence[1] == "A":
                strategies.append("Consider Aib (α-aminoisobutyric acid) at position 2")

        # Cleavage site strategies
        if cleavage.sites_by_protease.get(ProteaseType.TRYPSIN, 0) > 0:
            strategies.append("Replace K/R with non-basic residues or use D-amino acids at trypsin sites")
        if cleavage.sites_by_protease.get(ProteaseType.CHYMOTRYPSIN, 0) > 0:
            strategies.append("Replace F/Y/W with non-aromatic residues at chymotrypsin sites")

        # General strategies
        if cleavage.total_sites > 3:
            strategies.append("Consider backbone modifications (N-methylation, reduced peptide bonds)")
        if cleavage.high_severity_sites > 1:
            strategies.append("Consider retro-inverso design or cyclization for improved stability")

        # Length-based
        if len(sequence) < 10:
            strategies.append("Short peptides benefit from lipidation for extended half-life")

        return strategies[:5]  # Limit to top 5


__all__ = [
    "PeptideStabilityPredictor",
    "PeptideStabilityResult",
    "CleavageSite",
    "CleavageAnalysis",
    "ProteaseType",
    "CleavageSeverity",
]
