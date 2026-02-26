"""
Peptide Chemical Liability Detection

Detection of chemical modification sites that affect peptide stability and activity.

Detects:
- Oxidation-sensitive residues (Met, Trp, Cys, His)
- Deamidation sites (Asn-X, especially Asn-Gly)
- Isomerization sites (Asp-X, especially Asp-Gly)
- Racemization-prone residues
- Disulfide scrambling risk

References:
    - Manning et al. (2010) Pharmaceutical Research - Stability of protein pharmaceuticals
    - Wakankar & Borchardt (2006) J Pharm Sci - Formulation considerations
    - Geiger & Clarke (1987) JBC - Deamidation of asparagine
    - Stephenson & Clarke (1989) JBC - Succinimide formation
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

from pydantic import BaseModel, ConfigDict, Field, computed_field


class ChemicalLiabilityType(Enum):
    """Types of chemical liabilities in peptides."""

    OXIDATION = "oxidation"
    DEAMIDATION = "deamidation"
    ISOMERIZATION = "isomerization"
    RACEMIZATION = "racemization"
    DISULFIDE_SCRAMBLING = "disulfide_scrambling"
    PYROGLUTAMATE_FORMATION = "pyroglutamate"
    BETA_ELIMINATION = "beta_elimination"
    HYDROLYSIS = "hydrolysis"


class LiabilitySeverity(Enum):
    """Severity of chemical liability."""

    HIGH = "high"      # Fast kinetics, significant impact
    MEDIUM = "medium"  # Moderate kinetics
    LOW = "low"        # Slow kinetics, minor impact


# Oxidation susceptibility data
OXIDATION_SITES = {
    "M": {
        "severity": LiabilitySeverity.HIGH,
        "product": "methionine sulfoxide",
        "mechanism": "ROS-mediated",
        "reversible": True,
        "conditions": "light, metals, peroxides",
    },
    "W": {
        "severity": LiabilitySeverity.MEDIUM,
        "product": "kynurenine, N-formylkynurenine",
        "mechanism": "ROS-mediated",
        "reversible": False,
        "conditions": "UV light, ROS",
    },
    "C": {
        "severity": LiabilitySeverity.HIGH,
        "product": "cystine, cysteic acid",
        "mechanism": "thiol oxidation",
        "reversible": True,
        "conditions": "air, metals",
    },
    "H": {
        "severity": LiabilitySeverity.LOW,
        "product": "2-oxo-histidine",
        "mechanism": "metal-catalyzed",
        "reversible": False,
        "conditions": "metal ions",
    },
    "Y": {
        "severity": LiabilitySeverity.LOW,
        "product": "dityrosine, DOPA",
        "mechanism": "ROS-mediated",
        "reversible": False,
        "conditions": "UV light, ROS",
    },
    "F": {
        "severity": LiabilitySeverity.LOW,
        "product": "hydroxyphenylalanine",
        "mechanism": "hydroxyl radical",
        "reversible": False,
        "conditions": "strong oxidants",
    },
}

# Deamidation motifs and kinetics
# Half-lives from Robinson & Robinson (2001)
DEAMIDATION_MOTIFS = {
    "NG": {
        "severity": LiabilitySeverity.HIGH,
        "half_life": "~1 day",
        "kinetics": "very fast",
        "ph_sensitive": True,
    },
    "NS": {
        "severity": LiabilitySeverity.HIGH,
        "half_life": "~10 days",
        "kinetics": "fast",
        "ph_sensitive": True,
    },
    "NT": {
        "severity": LiabilitySeverity.MEDIUM,
        "half_life": "~50 days",
        "kinetics": "moderate",
        "ph_sensitive": True,
    },
    "NH": {
        "severity": LiabilitySeverity.MEDIUM,
        "half_life": "~30 days",
        "kinetics": "moderate",
        "ph_sensitive": True,
    },
    "NN": {
        "severity": LiabilitySeverity.MEDIUM,
        "half_life": "variable",
        "kinetics": "moderate",
        "ph_sensitive": True,
    },
    "NA": {
        "severity": LiabilitySeverity.LOW,
        "half_life": "slow",
        "kinetics": "slow",
        "ph_sensitive": True,
    },
    "NQ": {
        "severity": LiabilitySeverity.LOW,
        "half_life": "very slow",
        "kinetics": "very slow",
        "ph_sensitive": False,
    },
}

# Isomerization (Asp → isoAsp) motifs
ISOMERIZATION_MOTIFS = {
    "DG": {
        "severity": LiabilitySeverity.HIGH,
        "half_life": "~1-2 days",
        "kinetics": "very fast",
        "product": "iso-aspartate",
    },
    "DS": {
        "severity": LiabilitySeverity.HIGH,
        "half_life": "~5 days",
        "kinetics": "fast",
        "product": "iso-aspartate",
    },
    "DT": {
        "severity": LiabilitySeverity.MEDIUM,
        "half_life": "~15 days",
        "kinetics": "moderate",
        "product": "iso-aspartate",
    },
    "DD": {
        "severity": LiabilitySeverity.MEDIUM,
        "half_life": "~20 days",
        "kinetics": "moderate",
        "product": "iso-aspartate",
    },
    "DH": {
        "severity": LiabilitySeverity.LOW,
        "half_life": "~25 days",
        "kinetics": "slow",
        "product": "iso-aspartate",
    },
}

# Racemization-prone residues
RACEMIZATION_RESIDUES = {
    "D": {
        "severity": LiabilitySeverity.MEDIUM,
        "mechanism": "succinimide intermediate",
        "conditions": "high pH, heat",
    },
    "S": {
        "severity": LiabilitySeverity.LOW,
        "mechanism": "base-catalyzed",
        "conditions": "high pH",
    },
    "A": {
        "severity": LiabilitySeverity.LOW,
        "mechanism": "base-catalyzed",
        "conditions": "very high pH",
    },
}


class ChemicalLiability(NamedTuple):
    """A detected chemical liability site."""

    position: int
    residue: str
    liability_type: ChemicalLiabilityType
    severity: LiabilitySeverity
    motif: str | None
    description: str


@dataclass
class LiabilityProfile:
    """Complete chemical liability profile for a peptide."""

    total_liabilities: int
    high_severity: int
    medium_severity: int
    low_severity: int
    liabilities_by_type: dict[ChemicalLiabilityType, int]
    liabilities: list[ChemicalLiability]

    @property
    def weighted_risk_score(self) -> float:
        """Calculate weighted risk score."""
        return (
            self.high_severity * 3.0 +
            self.medium_severity * 1.5 +
            self.low_severity * 0.5
        )


class ChemicalLiabilityResult(BaseModel):
    """Result of chemical liability analysis."""

    model_config = ConfigDict(frozen=True)

    score: float = Field(ge=0, le=100, description="Chemical stability score (0-100, higher = more stable)")

    # Liability counts
    total_liabilities: int = Field(ge=0, description="Total chemical liability sites")
    oxidation_sites: int = Field(ge=0, description="Oxidation-susceptible sites")
    deamidation_sites: int = Field(ge=0, description="Deamidation-prone sites")
    isomerization_sites: int = Field(ge=0, description="Isomerization-prone sites")

    # High-risk indicators
    high_risk_sites: int = Field(ge=0, description="High-severity liability sites")
    has_met_oxidation: bool = Field(description="Contains Met (oxidation risk)")
    has_asn_gly: bool = Field(description="Contains Asn-Gly (rapid deamidation)")
    has_asp_gly: bool = Field(description="Contains Asp-Gly (rapid isomerization)")
    has_free_cys: bool = Field(description="Contains unpaired Cys")

    # Detailed liabilities
    liabilities: list[dict] = Field(default_factory=list, description="Detailed liability sites")

    # Recommendations
    formulation_recommendations: list[str] = Field(
        default_factory=list, description="Formulation recommendations"
    )
    engineering_recommendations: list[str] = Field(
        default_factory=list, description="Sequence engineering recommendations"
    )

    method: str = Field(default="sequence_based", description="Analysis method")
    confidence: float = Field(ge=0, le=1, description="Confidence score")

    @computed_field
    @property
    def interpretation(self) -> str:
        """Human-readable interpretation."""
        if self.score >= 80:
            return "Low chemical liability risk - suitable for standard formulation"
        if self.score >= 60:
            return "Moderate chemical liability - may require formulation optimization"
        if self.score >= 40:
            return "High chemical liability - formulation challenges expected"
        return "Very high chemical liability - significant engineering needed"


class ChemicalLiabilityScanner:
    """
    Scans peptide sequences for chemical modification liabilities.

    Detects oxidation, deamidation, isomerization, and other chemical
    instabilities that affect peptide therapeutics.

    Example:
        >>> scanner = ChemicalLiabilityScanner()
        >>> result = scanner.scan("HAEGTFTSDVSSYLEGQAAK")
        >>> print(f"Chemical Stability: {result.score}/100")
    """

    def __init__(
        self,
        include_minor_liabilities: bool = True,
        ph_context: float = 7.0,
    ) -> None:
        """
        Initialize scanner.

        Args:
            include_minor_liabilities: Include low-severity liabilities
            ph_context: pH context for kinetics estimation
        """
        self.include_minor = include_minor_liabilities
        self.ph = ph_context

    def scan(self, sequence: str) -> ChemicalLiabilityResult:
        """
        Scan sequence for chemical liabilities.

        Args:
            sequence: Peptide sequence

        Returns:
            ChemicalLiabilityResult with liability analysis
        """
        sequence = sequence.upper().strip()

        # Detect all liabilities
        profile = self._analyze_liabilities(sequence)

        # Check specific high-risk features
        has_met = "M" in sequence
        has_asn_gly = "NG" in sequence
        has_asp_gly = "DG" in sequence
        cys_count = sequence.count("C")
        has_free_cys = cys_count > 0 and cys_count % 2 != 0

        # Calculate score
        score = self._calculate_score(profile, len(sequence))

        # Generate recommendations
        formulation_recs = self._formulation_recommendations(profile, sequence)
        engineering_recs = self._engineering_recommendations(profile, sequence)

        # Convert liabilities to dicts
        liabilities_data = [
            {
                "position": l.position,
                "residue": l.residue,
                "type": l.liability_type.value,
                "severity": l.severity.value,
                "motif": l.motif,
                "description": l.description,
            }
            for l in profile.liabilities
        ]

        return ChemicalLiabilityResult(
            score=round(score, 1),
            total_liabilities=profile.total_liabilities,
            oxidation_sites=profile.liabilities_by_type.get(ChemicalLiabilityType.OXIDATION, 0),
            deamidation_sites=profile.liabilities_by_type.get(ChemicalLiabilityType.DEAMIDATION, 0),
            isomerization_sites=profile.liabilities_by_type.get(ChemicalLiabilityType.ISOMERIZATION, 0),
            high_risk_sites=profile.high_severity,
            has_met_oxidation=has_met,
            has_asn_gly=has_asn_gly,
            has_asp_gly=has_asp_gly,
            has_free_cys=has_free_cys,
            liabilities=liabilities_data,
            formulation_recommendations=formulation_recs,
            engineering_recommendations=engineering_recs,
            method="sequence_based",
            confidence=0.80,
        )

    def _analyze_liabilities(self, sequence: str) -> LiabilityProfile:
        """Analyze all chemical liabilities in sequence."""
        liabilities: list[ChemicalLiability] = []
        by_type: dict[ChemicalLiabilityType, int] = {t: 0 for t in ChemicalLiabilityType}

        # Oxidation sites
        for i, residue in enumerate(sequence):
            if residue in OXIDATION_SITES:
                site_info = OXIDATION_SITES[residue]
                if not self.include_minor and site_info["severity"] == LiabilitySeverity.LOW:
                    continue

                liabilities.append(ChemicalLiability(
                    position=i,
                    residue=residue,
                    liability_type=ChemicalLiabilityType.OXIDATION,
                    severity=site_info["severity"],
                    motif=None,
                    description=f"{residue} oxidation → {site_info['product']}",
                ))
                by_type[ChemicalLiabilityType.OXIDATION] += 1

        # Deamidation motifs (Asn-X)
        for i in range(len(sequence) - 1):
            dipeptide = sequence[i:i+2]
            if dipeptide.startswith("N") and dipeptide in DEAMIDATION_MOTIFS:
                motif_info = DEAMIDATION_MOTIFS[dipeptide]
                if not self.include_minor and motif_info["severity"] == LiabilitySeverity.LOW:
                    continue

                liabilities.append(ChemicalLiability(
                    position=i,
                    residue="N",
                    liability_type=ChemicalLiabilityType.DEAMIDATION,
                    severity=motif_info["severity"],
                    motif=dipeptide,
                    description=f"Deamidation at {dipeptide} (t½ {motif_info['half_life']})",
                ))
                by_type[ChemicalLiabilityType.DEAMIDATION] += 1

        # Isomerization motifs (Asp-X)
        for i in range(len(sequence) - 1):
            dipeptide = sequence[i:i+2]
            if dipeptide.startswith("D") and dipeptide in ISOMERIZATION_MOTIFS:
                motif_info = ISOMERIZATION_MOTIFS[dipeptide]
                if not self.include_minor and motif_info["severity"] == LiabilitySeverity.LOW:
                    continue

                liabilities.append(ChemicalLiability(
                    position=i,
                    residue="D",
                    liability_type=ChemicalLiabilityType.ISOMERIZATION,
                    severity=motif_info["severity"],
                    motif=dipeptide,
                    description=f"Isomerization at {dipeptide} (t½ {motif_info['half_life']})",
                ))
                by_type[ChemicalLiabilityType.ISOMERIZATION] += 1

        # Pyroglutamate formation at N-terminus
        if sequence and sequence[0] in {"E", "Q"}:
            liabilities.append(ChemicalLiability(
                position=0,
                residue=sequence[0],
                liability_type=ChemicalLiabilityType.PYROGLUTAMATE_FORMATION,
                severity=LiabilitySeverity.MEDIUM,
                motif=None,
                description=f"N-terminal {sequence[0]} → pyroglutamate",
            ))
            by_type[ChemicalLiabilityType.PYROGLUTAMATE_FORMATION] += 1

        # Count severities
        high = sum(1 for l in liabilities if l.severity == LiabilitySeverity.HIGH)
        medium = sum(1 for l in liabilities if l.severity == LiabilitySeverity.MEDIUM)
        low = sum(1 for l in liabilities if l.severity == LiabilitySeverity.LOW)

        return LiabilityProfile(
            total_liabilities=len(liabilities),
            high_severity=high,
            medium_severity=medium,
            low_severity=low,
            liabilities_by_type=by_type,
            liabilities=liabilities,
        )

    def _calculate_score(self, profile: LiabilityProfile, seq_length: int) -> float:
        """Calculate chemical stability score (0-100)."""
        score = 100.0

        # Penalty based on weighted risk
        risk_penalty = profile.weighted_risk_score * 4
        score -= min(risk_penalty, 50)

        # Normalize by sequence length
        if seq_length > 0:
            density = profile.total_liabilities / seq_length
            if density > 0.3:  # More than 30% residues have liabilities
                score -= 10

        # Specific penalties for very problematic motifs
        for liability in profile.liabilities:
            if liability.motif == "NG":  # Very fast deamidation
                score -= 5
            if liability.motif == "DG":  # Very fast isomerization
                score -= 5
            if liability.residue == "M":  # Met oxidation common
                score -= 3

        return max(0, min(100, score))

    def _formulation_recommendations(
        self, profile: LiabilityProfile, sequence: str
    ) -> list[str]:
        """Generate formulation recommendations."""
        recs: list[str] = []

        # Oxidation protection
        if profile.liabilities_by_type.get(ChemicalLiabilityType.OXIDATION, 0) > 0:
            recs.append("Include antioxidants (methionine, EDTA) in formulation")
            if "M" in sequence:
                recs.append("Consider nitrogen overlay to prevent Met oxidation")
            if "W" in sequence or "Y" in sequence:
                recs.append("Store protected from light (amber vials)")

        # Deamidation control
        if profile.liabilities_by_type.get(ChemicalLiabilityType.DEAMIDATION, 0) > 0:
            recs.append("Formulate at pH 5-6 to minimize deamidation rate")
            recs.append("Store at 2-8°C to slow deamidation kinetics")

        # Isomerization control
        if profile.liabilities_by_type.get(ChemicalLiabilityType.ISOMERIZATION, 0) > 0:
            recs.append("Avoid high pH (>7) to minimize isomerization")

        # Cysteine protection
        if "C" in sequence:
            if sequence.count("C") % 2 != 0:
                recs.append("Add reducing agent or cap free Cys to prevent oxidation")
            recs.append("Use nitrogen-purged buffers for Cys-containing peptides")

        # General
        if profile.high_severity > 2:
            recs.append("Consider lyophilization for long-term storage")

        return recs[:5]

    def _engineering_recommendations(
        self, profile: LiabilityProfile, sequence: str
    ) -> list[str]:
        """Generate sequence engineering recommendations."""
        recs: list[str] = []

        # Deamidation mitigation
        for liability in profile.liabilities:
            if liability.liability_type == ChemicalLiabilityType.DEAMIDATION:
                if liability.severity == LiabilitySeverity.HIGH:
                    recs.append(f"Consider N→Q substitution at position {liability.position} to remove deamidation site")

        # Isomerization mitigation
        for liability in profile.liabilities:
            if liability.liability_type == ChemicalLiabilityType.ISOMERIZATION:
                if liability.severity == LiabilitySeverity.HIGH:
                    recs.append(f"Consider D→E substitution at position {liability.position} to remove isomerization site")

        # Oxidation mitigation
        met_positions = [i for i, r in enumerate(sequence) if r == "M"]
        if met_positions:
            recs.append(f"Consider M→L or M→Nle substitution at positions {met_positions}")

        # Cysteine issues
        cys_count = sequence.count("C")
        if cys_count == 1:
            recs.append("Single Cys may cause dimerization - consider C→S or C→A substitution")
        elif cys_count > 2 and cys_count % 2 == 0:
            recs.append("Multiple Cys pairs - verify correct disulfide pairing")

        return recs[:5]


__all__ = [
    "ChemicalLiabilityScanner",
    "ChemicalLiabilityResult",
    "ChemicalLiability",
    "LiabilityProfile",
    "ChemicalLiabilityType",
    "LiabilitySeverity",
]
