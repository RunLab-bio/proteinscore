"""
Antibody Sequence Liability Detection

Comprehensive detection of developability liabilities in antibody sequences
based on LAP (Liability Antibody Profiler) and clinical antibody studies.

Detects:
- Deamidation sites (NG, NS, NA, NT, NH, NN, SN, TN, KN)
- Isomerization sites (DG, DS, DT, DD, DH, DN)
- Oxidation-sensitive sites (M, W in CDRs)
- N-linked glycosylation sites (N-X-S/T)
- Fragmentation/Clipping sites (DP, TS)
- Hydrolysis sites (NP)
- Unpaired cysteines
- Missing conserved cysteines
- Integrin binding motifs (polyreactivity risk)
- Aggregation-prone regions (APRs)

References:
    - Khetan et al. (2024) LAP: Liability Antibody Profiler. PLOS Comp Bio
    - Lu et al. (2019) Deamidation and isomerization analysis of 131 antibodies
    - Chennamsetty et al. (2009) PNAS - SAP scores
    - Geneious Biologics - Antibody Sequence Liabilities
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import NamedTuple

from proteinscore.antibody.cdr import CDRDetector, CDRRegions


# =============================================================================
# Liability Types
# =============================================================================

class LiabilityType(Enum):
    """Types of sequence liabilities."""
    DEAMIDATION = "deamidation"
    ISOMERIZATION = "isomerization"
    OXIDATION = "oxidation"
    GLYCOSYLATION = "n_glycosylation"
    FRAGMENTATION = "fragmentation"
    HYDROLYSIS = "hydrolysis"
    UNPAIRED_CYSTEINE = "unpaired_cysteine"
    MISSING_CYSTEINE = "missing_cysteine"
    INTEGRIN_BINDING = "integrin_binding"
    AGGREGATION = "aggregation_prone"


class LiabilitySeverity(Enum):
    """Severity levels for liabilities."""
    HIGH = "high"      # Known to cause issues, prioritize removal
    MEDIUM = "medium"  # May cause issues depending on context
    LOW = "low"        # Minor concern, often tolerated


# =============================================================================
# Liability Motifs (based on LAP 2024 and literature)
# =============================================================================

# Deamidation motifs (N → D/isoD conversion)
# Severity based on half-life: NG (~1 day) > NS (~10 days) > NT (~50 days)
DEAMIDATION_MOTIFS = {
    # High risk - fast kinetics
    "NG": {"severity": LiabilitySeverity.HIGH, "half_life": "~1 day", "risk": "very_high"},
    "NS": {"severity": LiabilitySeverity.HIGH, "half_life": "~10 days", "risk": "high"},
    # Medium risk
    "NT": {"severity": LiabilitySeverity.MEDIUM, "half_life": "~50 days", "risk": "medium"},
    "NH": {"severity": LiabilitySeverity.MEDIUM, "half_life": "~30 days", "risk": "medium"},
    "NN": {"severity": LiabilitySeverity.MEDIUM, "half_life": "variable", "risk": "medium"},
    "NA": {"severity": LiabilitySeverity.MEDIUM, "half_life": "slow", "risk": "medium"},
    # Low risk - slow kinetics or context-dependent
    "NQ": {"severity": LiabilitySeverity.LOW, "half_life": "slow", "risk": "low"},
    "NR": {"severity": LiabilitySeverity.LOW, "half_life": "very slow", "risk": "low"},
    "NK": {"severity": LiabilitySeverity.LOW, "half_life": "very slow", "risk": "low"},
    # Reverse motifs (less common but documented)
    "SN": {"severity": LiabilitySeverity.LOW, "half_life": "context-dependent", "risk": "low"},
    "TN": {"severity": LiabilitySeverity.LOW, "half_life": "context-dependent", "risk": "low"},
    "KN": {"severity": LiabilitySeverity.LOW, "half_life": "context-dependent", "risk": "low"},
}

# Isomerization motifs (D → isoD conversion)
# DG is fastest, then DS, DT
ISOMERIZATION_MOTIFS = {
    # High risk - fast kinetics
    "DG": {"severity": LiabilitySeverity.HIGH, "half_life": "~1-2 days", "risk": "very_high"},
    "DS": {"severity": LiabilitySeverity.HIGH, "half_life": "~5 days", "risk": "high"},
    # Medium risk
    "DT": {"severity": LiabilitySeverity.MEDIUM, "half_life": "~15 days", "risk": "medium"},
    "DD": {"severity": LiabilitySeverity.MEDIUM, "half_life": "~20 days", "risk": "medium"},
    "DH": {"severity": LiabilitySeverity.MEDIUM, "half_life": "~25 days", "risk": "medium"},
    # Low risk
    "DN": {"severity": LiabilitySeverity.LOW, "half_life": "slow", "risk": "low"},
    "DA": {"severity": LiabilitySeverity.LOW, "half_life": "slow", "risk": "low"},
}

# Oxidation-sensitive residues (context-dependent)
OXIDATION_RESIDUES = {
    "M": {"severity": LiabilitySeverity.HIGH, "product": "Met-sulfoxide", "reversible": True},
    "W": {"severity": LiabilitySeverity.MEDIUM, "product": "various", "reversible": False},
    "Y": {"severity": LiabilitySeverity.LOW, "product": "dityrosine", "reversible": False},
    "F": {"severity": LiabilitySeverity.LOW, "product": "hydroxyphenylalanine", "reversible": False},
    "H": {"severity": LiabilitySeverity.LOW, "product": "2-oxo-histidine", "reversible": False},
}

# Fragmentation/Clipping motifs
FRAGMENTATION_MOTIFS = {
    "DP": {"severity": LiabilitySeverity.HIGH, "mechanism": "Asp-Pro bond cleavage"},
    "TS": {"severity": LiabilitySeverity.MEDIUM, "mechanism": "Thr-Ser hydrolysis"},
    "DK": {"severity": LiabilitySeverity.MEDIUM, "mechanism": "Asp-Lys cleavage"},
    "DS": {"severity": LiabilitySeverity.LOW, "mechanism": "Asp-Ser hydrolysis (rare)"},
}

# Hydrolysis motifs
HYDROLYSIS_MOTIFS = {
    "NP": {"severity": LiabilitySeverity.MEDIUM, "mechanism": "Asn-Pro peptide hydrolysis"},
}

# Integrin binding motifs (polyreactivity/cross-reactivity risk)
INTEGRIN_MOTIFS = {
    "RGD": {"severity": LiabilitySeverity.HIGH, "receptor": "αvβ3, αIIbβ3, α5β1"},
    "LDV": {"severity": LiabilitySeverity.MEDIUM, "receptor": "α4β1"},
    "KGD": {"severity": LiabilitySeverity.MEDIUM, "receptor": "αIIbβ3"},
    "NGR": {"severity": LiabilitySeverity.MEDIUM, "receptor": "aminopeptidase N"},
    "RYD": {"severity": LiabilitySeverity.LOW, "receptor": "fibronectin receptor"},
    "DGE": {"severity": LiabilitySeverity.LOW, "receptor": "α5β1"},
    "GPR": {"severity": LiabilitySeverity.LOW, "receptor": "collagen receptor"},
}

# N-linked glycosylation pattern
GLYCOSYLATION_PATTERN = re.compile(r"N[^P][ST]")

# Conserved cysteine positions (variable due to CDR length variations)
# VH: typically C22-23 and C92-96 | VL: C22-23 and C87-92
# Using wider ranges to accommodate sequence length variations
CONSERVED_CYS_POSITIONS = {
    "heavy": [(18, 28), (88, 100)],  # C23 and C96 (±5 for variability)
    "light": [(18, 28), (82, 95)],   # C23 and C88 (±5 for variability)
}


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class Liability:
    """A detected sequence liability."""
    type: LiabilityType
    severity: LiabilitySeverity
    position: int           # 0-indexed position in sequence
    motif: str              # The problematic motif/residue
    region: str             # CDR or FR region name
    sequence_context: str   # Surrounding sequence (±3 residues)
    recommendation: str     # Suggested fix
    mechanism: str = ""     # Chemical mechanism
    half_life: str = ""     # Estimated half-life if known
    in_cdr: bool = False    # Is this in a CDR?

    @property
    def position_1indexed(self) -> int:
        """Return 1-indexed position for display."""
        return self.position + 1


class LiabilitySummary(NamedTuple):
    """Summary of liabilities by type."""
    deamidation: int
    isomerization: int
    oxidation: int
    glycosylation: int
    fragmentation: int
    hydrolysis: int
    unpaired_cysteine: int
    missing_cysteine: int
    integrin_binding: int
    aggregation: int
    total: int
    high_severity: int
    medium_severity: int
    low_severity: int
    cdr_liabilities: int  # Liabilities specifically in CDRs


@dataclass
class LiabilityScanResult:
    """Complete liability scan result."""
    liabilities: list[Liability]
    summary: LiabilitySummary
    chain_type: str  # "heavy", "light", or "paired"
    sequence_length: int
    cdr_coverage: float = 0.0  # Fraction of sequence in CDRs

    @property
    def has_critical(self) -> bool:
        """Check if any high-severity liabilities detected."""
        return self.summary.high_severity > 0

    @property
    def has_cdr_liabilities(self) -> bool:
        """Check if any liabilities are in CDRs."""
        return self.summary.cdr_liabilities > 0

    @property
    def liability_score(self) -> float:
        """
        Calculate liability score (0-100, higher = fewer/less severe liabilities).

        Calibrated so clinical antibodies score ~50-80 on average.

        Scoring (calibrated for clinical antibodies):
        - High severity in CDR: -3 points
        - High severity in FR: -2 points
        - Medium severity in CDR: -1.5 points
        - Medium severity in FR: -1 point
        - Low severity: -0.5 points
        """
        penalty = 0.0
        for lib in self.liabilities:
            if lib.severity == LiabilitySeverity.HIGH:
                penalty += 3.0 if lib.in_cdr else 2.0
            elif lib.severity == LiabilitySeverity.MEDIUM:
                penalty += 1.5 if lib.in_cdr else 1.0
            else:
                penalty += 0.5

        return max(0.0, 100.0 - penalty)

    @property
    def risk_level(self) -> str:
        """
        Overall risk assessment.

        Calibrated against clinical antibody data - most approved antibodies
        have some liabilities that are tolerated in practice.

        Thresholds based on Jain 2017 clinical antibody distribution:
        - High risk: significantly above average clinical antibody
        - Medium risk: average for clinical antibody
        - Low risk: better than average clinical antibody
        """
        # Clinical antibodies average ~9 high-severity, ~17 CDR liabilities
        if self.summary.high_severity >= 15 or self.summary.cdr_liabilities >= 25:
            return "high"
        elif self.summary.high_severity >= 10 or self.summary.cdr_liabilities >= 18:
            return "medium"
        else:
            return "low"

    def get_actionable_liabilities(self) -> list[Liability]:
        """Return liabilities prioritized for engineering (high severity in CDRs first)."""
        return sorted(
            self.liabilities,
            key=lambda x: (
                -int(x.severity == LiabilitySeverity.HIGH),
                -int(x.in_cdr),
                -int(x.severity == LiabilitySeverity.MEDIUM),
                x.position
            )
        )


# =============================================================================
# Liability Scanner
# =============================================================================

class LiabilityScanner:
    """
    Comprehensive antibody sequence liability scanner.

    Detects chemical and structural liabilities that can affect
    manufacturability, stability, immunogenicity, and efficacy.

    Based on LAP (Liability Antibody Profiler) methodology and
    analysis of 131+ clinical-stage antibodies.
    """

    def __init__(
        self,
        cdr_detector: CDRDetector | None = None,
        check_aggregation: bool = True,
        check_integrin: bool = True,
        strict_mode: bool = False,
    ):
        """
        Initialize liability scanner.

        Args:
            cdr_detector: CDR detector for region annotation
            check_aggregation: Scan for aggregation-prone regions
            check_integrin: Scan for integrin binding motifs
            strict_mode: If True, flag all potential liabilities including low-risk
        """
        self._cdr_detector = cdr_detector or CDRDetector()
        self._check_aggregation = check_aggregation
        self._check_integrin = check_integrin
        self._strict_mode = strict_mode

    def scan(
        self,
        sequence: str,
        chain_type: str | None = None,
    ) -> LiabilityScanResult:
        """
        Scan a single antibody chain for liabilities.

        Args:
            sequence: VH or VL sequence
            chain_type: "heavy" or "light" (auto-detected if None)

        Returns:
            LiabilityScanResult with all detected liabilities
        """
        sequence = sequence.upper().strip()

        # Detect CDRs for region annotation
        cdrs = self._cdr_detector.detect_cdrs(sequence, chain_type)
        chain_type = cdrs.chain_type

        liabilities = []

        # Chemical modifications
        liabilities.extend(self._scan_deamidation(sequence, cdrs))
        liabilities.extend(self._scan_isomerization(sequence, cdrs))
        liabilities.extend(self._scan_oxidation(sequence, cdrs))
        liabilities.extend(self._scan_glycosylation(sequence, cdrs))
        liabilities.extend(self._scan_fragmentation(sequence, cdrs))
        liabilities.extend(self._scan_hydrolysis(sequence, cdrs))

        # Structural issues
        liabilities.extend(self._scan_cysteine_issues(sequence, cdrs, chain_type))

        # Functional concerns
        if self._check_integrin:
            liabilities.extend(self._scan_integrin_motifs(sequence, cdrs))

        if self._check_aggregation:
            liabilities.extend(self._scan_aggregation_prone(sequence, cdrs))

        # Calculate summary
        summary = self._calculate_summary(liabilities)

        # Calculate CDR coverage
        cdr_length = sum(len(cdr.sequence) for cdr in cdrs.cdrs)
        cdr_coverage = cdr_length / len(sequence) if sequence else 0

        return LiabilityScanResult(
            liabilities=liabilities,
            summary=summary,
            chain_type=chain_type,
            sequence_length=len(sequence),
            cdr_coverage=cdr_coverage,
        )

    def scan_paired(
        self,
        vh_sequence: str,
        vl_sequence: str,
    ) -> LiabilityScanResult:
        """
        Scan a paired VH/VL for liabilities.

        Args:
            vh_sequence: Heavy chain variable region
            vl_sequence: Light chain variable region

        Returns:
            Combined liability scan result
        """
        vh_result = self.scan(vh_sequence, "heavy")
        vl_result = self.scan(vl_sequence, "light")

        # Combine liabilities with chain annotation
        all_liabilities = []

        for lib in vh_result.liabilities:
            lib.region = f"VH:{lib.region}"
            all_liabilities.append(lib)

        for lib in vl_result.liabilities:
            # Adjust position for combined sequence
            lib.region = f"VL:{lib.region}"
            all_liabilities.append(lib)

        # Combined summary
        summary = self._calculate_summary(all_liabilities)

        total_len = len(vh_sequence) + len(vl_sequence)
        cdr_coverage = (vh_result.cdr_coverage * len(vh_sequence) +
                       vl_result.cdr_coverage * len(vl_sequence)) / total_len if total_len else 0

        return LiabilityScanResult(
            liabilities=all_liabilities,
            summary=summary,
            chain_type="paired",
            sequence_length=total_len,
            cdr_coverage=cdr_coverage,
        )

    def _is_in_cdr(self, position: int, cdrs: CDRRegions) -> bool:
        """Check if position is within a CDR."""
        for cdr in cdrs.cdrs:
            if cdr.start <= position < cdr.end:
                return True
        return False

    def _get_region(self, position: int, cdrs: CDRRegions) -> str:
        """Get region name (CDR or FR) for a position."""
        for cdr in cdrs.cdrs:
            if cdr.start <= position < cdr.end:
                return cdr.name
        for fr in cdrs.framework_regions:
            if fr.start <= position < fr.end:
                return fr.name
        return "unknown"

    def _get_context(self, sequence: str, pos: int, motif_len: int = 2) -> str:
        """Get sequence context around a position."""
        start = max(0, pos - 3)
        end = min(len(sequence), pos + motif_len + 3)
        return sequence[start:end]

    def _scan_deamidation(
        self,
        sequence: str,
        cdrs: CDRRegions
    ) -> list[Liability]:
        """Scan for deamidation-prone motifs."""
        liabilities = []

        for motif, info in DEAMIDATION_MOTIFS.items():
            for match in re.finditer(motif, sequence):
                pos = match.start()
                region = self._get_region(pos, cdrs)
                in_cdr = self._is_in_cdr(pos, cdrs)

                # Elevate severity if in CDR
                severity = info["severity"]
                if in_cdr and severity == LiabilitySeverity.MEDIUM:
                    severity = LiabilitySeverity.HIGH
                elif in_cdr and severity == LiabilitySeverity.LOW:
                    severity = LiabilitySeverity.MEDIUM

                # Skip low severity in non-strict mode
                if not self._strict_mode and severity == LiabilitySeverity.LOW and not in_cdr:
                    continue

                # Generate specific recommendation
                if motif.startswith("N"):
                    rec = f"Consider N{pos+1}→Q mutation to prevent deamidation"
                else:
                    rec = f"Monitor position {pos+1} for deamidation"

                liabilities.append(Liability(
                    type=LiabilityType.DEAMIDATION,
                    severity=severity,
                    position=pos,
                    motif=motif,
                    region=region,
                    sequence_context=self._get_context(sequence, pos),
                    recommendation=rec,
                    mechanism="Asparagine deamidation to aspartate/isoaspartate",
                    half_life=info.get("half_life", ""),
                    in_cdr=in_cdr,
                ))

        return liabilities

    def _scan_isomerization(
        self,
        sequence: str,
        cdrs: CDRRegions
    ) -> list[Liability]:
        """Scan for isomerization-prone motifs."""
        liabilities = []

        for motif, info in ISOMERIZATION_MOTIFS.items():
            for match in re.finditer(motif, sequence):
                pos = match.start()
                region = self._get_region(pos, cdrs)
                in_cdr = self._is_in_cdr(pos, cdrs)

                severity = info["severity"]
                if in_cdr and severity == LiabilitySeverity.MEDIUM:
                    severity = LiabilitySeverity.HIGH
                elif in_cdr and severity == LiabilitySeverity.LOW:
                    severity = LiabilitySeverity.MEDIUM

                if not self._strict_mode and severity == LiabilitySeverity.LOW and not in_cdr:
                    continue

                liabilities.append(Liability(
                    type=LiabilityType.ISOMERIZATION,
                    severity=severity,
                    position=pos,
                    motif=motif,
                    region=region,
                    sequence_context=self._get_context(sequence, pos),
                    recommendation=f"Consider D{pos+1}→E mutation to prevent isomerization",
                    mechanism="Aspartate isomerization to isoaspartate via succinimide",
                    half_life=info.get("half_life", ""),
                    in_cdr=in_cdr,
                ))

        return liabilities

    def _scan_oxidation(
        self,
        sequence: str,
        cdrs: CDRRegions
    ) -> list[Liability]:
        """Scan for oxidation-sensitive residues."""
        liabilities = []

        for i, aa in enumerate(sequence):
            if aa not in OXIDATION_RESIDUES:
                continue

            info = OXIDATION_RESIDUES[aa]
            region = self._get_region(i, cdrs)
            in_cdr = self._is_in_cdr(i, cdrs)

            # Only flag M/W in CDRs by default, or all in strict mode
            if not in_cdr and aa in "YFH" and not self._strict_mode:
                continue
            if not in_cdr and aa == "W" and not self._strict_mode:
                continue

            severity = info["severity"]
            # Elevate severity if in CDR
            if in_cdr and severity == LiabilitySeverity.LOW:
                severity = LiabilitySeverity.MEDIUM
            elif in_cdr and severity == LiabilitySeverity.MEDIUM:
                severity = LiabilitySeverity.HIGH

            # Skip low-risk oxidation in framework
            if not self._strict_mode and severity == LiabilitySeverity.LOW:
                continue

            # Specific recommendations
            if aa == "M":
                rec = f"Consider M{i+1}→L mutation to prevent methionine oxidation"
            elif aa == "W":
                rec = f"Consider W{i+1}→F mutation if tryptophan oxidation is observed"
            else:
                rec = f"Monitor {aa}{i+1} for potential oxidation"

            liabilities.append(Liability(
                type=LiabilityType.OXIDATION,
                severity=severity,
                position=i,
                motif=aa,
                region=region,
                sequence_context=self._get_context(sequence, i, 1),
                recommendation=rec,
                mechanism=f"{aa} oxidation to {info['product']}",
                in_cdr=in_cdr,
            ))

        return liabilities

    def _scan_glycosylation(
        self,
        sequence: str,
        cdrs: CDRRegions
    ) -> list[Liability]:
        """Scan for N-linked glycosylation sites."""
        liabilities = []

        for match in GLYCOSYLATION_PATTERN.finditer(sequence):
            pos = match.start()
            motif = match.group()
            region = self._get_region(pos, cdrs)
            in_cdr = self._is_in_cdr(pos, cdrs)

            # Glycosylation in CDRs is high severity (affects binding)
            severity = LiabilitySeverity.HIGH if in_cdr else LiabilitySeverity.MEDIUM

            liabilities.append(Liability(
                type=LiabilityType.GLYCOSYLATION,
                severity=severity,
                position=pos,
                motif=motif,
                region=region,
                sequence_context=self._get_context(sequence, pos, 3),
                recommendation=f"N-glycosylation site at N{pos+1}. Consider N→Q or S/T→A to remove.",
                mechanism="N-linked glycosylation at N-X-S/T sequon",
                in_cdr=in_cdr,
            ))

        return liabilities

    def _scan_fragmentation(
        self,
        sequence: str,
        cdrs: CDRRegions
    ) -> list[Liability]:
        """Scan for fragmentation/clipping sites."""
        liabilities = []

        for motif, info in FRAGMENTATION_MOTIFS.items():
            for match in re.finditer(motif, sequence):
                pos = match.start()
                region = self._get_region(pos, cdrs)
                in_cdr = self._is_in_cdr(pos, cdrs)

                severity = info["severity"]
                if in_cdr:
                    severity = LiabilitySeverity.HIGH

                if not self._strict_mode and severity == LiabilitySeverity.LOW:
                    continue

                liabilities.append(Liability(
                    type=LiabilityType.FRAGMENTATION,
                    severity=severity,
                    position=pos,
                    motif=motif,
                    region=region,
                    sequence_context=self._get_context(sequence, pos),
                    recommendation=f"Potential clipping site at {motif} ({pos+1}-{pos+2}). Consider D→E or P→A if fragmentation observed.",
                    mechanism=info["mechanism"],
                    in_cdr=in_cdr,
                ))

        return liabilities

    def _scan_hydrolysis(
        self,
        sequence: str,
        cdrs: CDRRegions
    ) -> list[Liability]:
        """Scan for hydrolysis-prone sites."""
        liabilities = []

        for motif, info in HYDROLYSIS_MOTIFS.items():
            for match in re.finditer(motif, sequence):
                pos = match.start()
                region = self._get_region(pos, cdrs)
                in_cdr = self._is_in_cdr(pos, cdrs)

                severity = info["severity"]
                if in_cdr:
                    severity = LiabilitySeverity.HIGH

                liabilities.append(Liability(
                    type=LiabilityType.HYDROLYSIS,
                    severity=severity,
                    position=pos,
                    motif=motif,
                    region=region,
                    sequence_context=self._get_context(sequence, pos),
                    recommendation=f"Hydrolysis-prone site at {motif} ({pos+1}-{pos+2}). Monitor for cleavage.",
                    mechanism=info["mechanism"],
                    in_cdr=in_cdr,
                ))

        return liabilities

    def _scan_cysteine_issues(
        self,
        sequence: str,
        cdrs: CDRRegions,
        chain_type: str
    ) -> list[Liability]:
        """Check for cysteine-related issues."""
        liabilities = []

        cys_positions = [i for i, aa in enumerate(sequence) if aa == "C"]
        conserved_ranges = CONSERVED_CYS_POSITIONS.get(chain_type, [])

        # Check for missing conserved cysteines
        for cys_range in conserved_ranges:
            has_cys = any(cys_range[0] <= pos <= cys_range[1] for pos in cys_positions)
            if not has_cys:
                liabilities.append(Liability(
                    type=LiabilityType.MISSING_CYSTEINE,
                    severity=LiabilitySeverity.HIGH,
                    position=cys_range[0],
                    motif="missing C",
                    region="conserved",
                    sequence_context=sequence[cys_range[0]:cys_range[1]+1],
                    recommendation=f"Missing conserved cysteine in range {cys_range[0]+1}-{cys_range[1]+1}. Critical for disulfide bond.",
                    mechanism="Missing intradomain disulfide bond",
                    in_cdr=False,
                ))

        # Check for unpaired/extra cysteines
        conserved_cys = []
        extra_cys = []
        for pos in cys_positions:
            is_conserved = any(r[0] <= pos <= r[1] for r in conserved_ranges)
            if is_conserved:
                conserved_cys.append(pos)
            else:
                extra_cys.append(pos)

        # Extra cysteines are problematic
        for pos in extra_cys:
            region = self._get_region(pos, cdrs)
            in_cdr = self._is_in_cdr(pos, cdrs)

            liabilities.append(Liability(
                type=LiabilityType.UNPAIRED_CYSTEINE,
                severity=LiabilitySeverity.HIGH,
                position=pos,
                motif="C",
                region=region,
                sequence_context=self._get_context(sequence, pos, 1),
                recommendation=f"Extra cysteine at C{pos+1}. May cause aggregation via disulfide scrambling. Consider C→S mutation.",
                mechanism="Potential intermolecular disulfide formation",
                in_cdr=in_cdr,
            ))

        return liabilities

    def _scan_integrin_motifs(
        self,
        sequence: str,
        cdrs: CDRRegions
    ) -> list[Liability]:
        """Scan for integrin binding motifs (polyreactivity risk)."""
        liabilities = []

        for motif, info in INTEGRIN_MOTIFS.items():
            for match in re.finditer(motif, sequence):
                pos = match.start()
                region = self._get_region(pos, cdrs)
                in_cdr = self._is_in_cdr(pos, cdrs)

                # Only concerning in CDRs (surface-exposed)
                if not in_cdr and not self._strict_mode:
                    continue

                severity = info["severity"] if in_cdr else LiabilitySeverity.LOW

                liabilities.append(Liability(
                    type=LiabilityType.INTEGRIN_BINDING,
                    severity=severity,
                    position=pos,
                    motif=motif,
                    region=region,
                    sequence_context=self._get_context(sequence, pos, len(motif)),
                    recommendation=f"Integrin binding motif {motif} at {pos+1}-{pos+len(motif)}. May cause polyreactivity with {info['receptor']}.",
                    mechanism=f"Potential binding to integrin {info['receptor']}",
                    in_cdr=in_cdr,
                ))

        return liabilities

    def _scan_aggregation_prone(
        self,
        sequence: str,
        cdrs: CDRRegions
    ) -> list[Liability]:
        """
        Scan for aggregation-prone regions (APRs).

        Uses hydrophobicity-based detection similar to TANGO algorithm.
        """
        liabilities = []

        window_size = 7
        threshold = 3.5  # Slightly higher threshold to reduce false positives

        # Hydrophobicity scale
        hydro = {
            "I": 1.0, "V": 0.93, "L": 0.84, "F": 0.62, "C": 0.55,
            "M": 0.42, "A": 0.40, "G": -0.09, "T": -0.16, "S": -0.18,
            "W": -0.20, "Y": -0.29, "P": -0.35, "H": -0.71, "E": -0.78,
            "Q": -0.78, "D": -0.78, "N": -0.78, "K": -0.87, "R": -1.0,
        }

        i = 0
        while i <= len(sequence) - window_size:
            window = sequence[i:i + window_size]
            score = sum(hydro.get(aa, 0) for aa in window)

            if score > threshold:
                region = self._get_region(i, cdrs)
                in_cdr = self._is_in_cdr(i, cdrs)

                severity = LiabilitySeverity.HIGH if in_cdr else LiabilitySeverity.MEDIUM

                liabilities.append(Liability(
                    type=LiabilityType.AGGREGATION,
                    severity=severity,
                    position=i,
                    motif=window,
                    region=region,
                    sequence_context=sequence[max(0, i-2):min(len(sequence), i+window_size+2)],
                    recommendation=f"Hydrophobic stretch at {i+1}-{i+window_size}. Consider inserting charged residue (K, R, E, D).",
                    mechanism="Beta-sheet mediated aggregation",
                    in_cdr=in_cdr,
                ))

                # Skip overlapping windows
                i += window_size
            else:
                i += 1

        return liabilities

    def _calculate_summary(self, liabilities: list[Liability]) -> LiabilitySummary:
        """Calculate summary statistics."""
        by_type = {t: 0 for t in LiabilityType}
        high_sev = medium_sev = low_sev = cdr_count = 0

        for lib in liabilities:
            by_type[lib.type] += 1

            if lib.severity == LiabilitySeverity.HIGH:
                high_sev += 1
            elif lib.severity == LiabilitySeverity.MEDIUM:
                medium_sev += 1
            else:
                low_sev += 1

            if lib.in_cdr:
                cdr_count += 1

        return LiabilitySummary(
            deamidation=by_type[LiabilityType.DEAMIDATION],
            isomerization=by_type[LiabilityType.ISOMERIZATION],
            oxidation=by_type[LiabilityType.OXIDATION],
            glycosylation=by_type[LiabilityType.GLYCOSYLATION],
            fragmentation=by_type[LiabilityType.FRAGMENTATION],
            hydrolysis=by_type[LiabilityType.HYDROLYSIS],
            unpaired_cysteine=by_type[LiabilityType.UNPAIRED_CYSTEINE],
            missing_cysteine=by_type[LiabilityType.MISSING_CYSTEINE],
            integrin_binding=by_type[LiabilityType.INTEGRIN_BINDING],
            aggregation=by_type[LiabilityType.AGGREGATION],
            total=len(liabilities),
            high_severity=high_sev,
            medium_severity=medium_sev,
            low_severity=low_sev,
            cdr_liabilities=cdr_count,
        )


# =============================================================================
# Convenience Functions
# =============================================================================

def scan_antibody(
    vh_sequence: str,
    vl_sequence: str,
    strict: bool = False
) -> LiabilityScanResult:
    """
    Quick scan of an antibody pair for liabilities.

    Args:
        vh_sequence: Heavy chain variable region
        vl_sequence: Light chain variable region
        strict: Include low-severity liabilities

    Returns:
        LiabilityScanResult with all detected liabilities
    """
    scanner = LiabilityScanner(strict_mode=strict)
    return scanner.scan_paired(vh_sequence, vl_sequence)


def get_engineering_recommendations(result: LiabilityScanResult, top_n: int = 5) -> list[str]:
    """
    Get top engineering recommendations from a liability scan.

    Args:
        result: LiabilityScanResult from scanning
        top_n: Number of top recommendations to return

    Returns:
        List of recommendation strings
    """
    actionable = result.get_actionable_liabilities()[:top_n]
    return [
        f"{lib.position_1indexed}: {lib.recommendation}"
        for lib in actionable
    ]
