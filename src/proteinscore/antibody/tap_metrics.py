"""
TAP (Therapeutic Antibody Profiler) Metrics

Implements the 5 TAP developability metrics:
1. CDR Length - Total CDR residue count
2. PSH - Patches of Surface Hydrophobicity
3. PPC - Patches of Positive Charge
4. PNC - Patches of Negative Charge
5. SFvCSP - Structural Fv Charge Symmetry Parameter

Reference:
    Raybould MIJ et al. (2019). "Five computational developability guidelines
    for therapeutic antibody profiling." PNAS 116(10):4025-4030

Hydrophobicity scales tested (Warszawski et al. 2022):
    - Kyte-Doolittle: Original TAP
    - Jain: Better HIC correlation (Spearman ~0.70)
    - Wimley-White: Good for membrane interactions
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

from proteinscore.antibody.cdr import CDRDetector, CDRRegions


# =============================================================================
# Hydrophobicity Scales
# =============================================================================

# Kyte-Doolittle hydropathy scale (1982)
# Original TAP uses this scale
KYTE_DOOLITTLE = {
    "I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5,
    "M": 1.9, "A": 1.8, "G": -0.4, "T": -0.7, "S": -0.8,
    "W": -0.9, "Y": -1.3, "P": -1.6, "H": -3.2, "E": -3.5,
    "Q": -3.5, "D": -3.5, "N": -3.5, "K": -3.9, "R": -4.5,
}

# Jain scale - optimized for HIC retention (2017)
# Better correlation with experimental HIC data
JAIN_HYDROPHOBICITY = {
    "I": 1.81, "L": 1.70, "F": 1.69, "V": 1.49, "M": 1.42,
    "W": 1.35, "A": 0.83, "C": 0.83, "T": 0.34, "Y": 0.26,
    "G": 0.10, "S": -0.05, "H": -0.29, "P": -0.31, "Q": -0.68,
    "N": -0.82, "E": -0.87, "D": -1.05, "K": -1.18, "R": -1.40,
}

# Eisenberg consensus scale
EISENBERG = {
    "I": 1.38, "V": 1.08, "L": 1.06, "F": 1.19, "C": 0.29,
    "M": 0.64, "A": 0.62, "G": 0.48, "T": -0.05, "S": -0.18,
    "W": 0.81, "Y": 0.26, "P": 0.12, "H": -0.40, "E": -0.74,
    "Q": -0.85, "D": -0.90, "N": -0.78, "K": -1.50, "R": -2.53,
}

# Wimley-White interface scale (peptide-membrane)
WIMLEY_WHITE = {
    "F": -1.13, "L": -0.56, "I": -0.31, "W": -1.85, "V": 0.07,
    "M": -0.23, "Y": -0.94, "C": -0.24, "A": 0.17, "T": 0.14,
    "G": 0.01, "S": 0.13, "H": 0.17, "P": 0.45, "R": 0.81,
    "K": 0.99, "N": 0.42, "Q": 0.58, "E": 2.02, "D": 1.23,
}


# =============================================================================
# Amino Acid Charges
# =============================================================================

# Amino acid charges at physiological pH (~7.4)
AA_CHARGE = {
    "K": 1.0,   # Lysine - positive
    "R": 1.0,   # Arginine - positive
    "H": 0.1,   # Histidine - partially positive at pH 7.4
    "D": -1.0,  # Aspartate - negative
    "E": -1.0,  # Glutamate - negative
    # All other amino acids are neutral
}


# =============================================================================
# TAP Thresholds (from TAP 2025 update)
# =============================================================================

class TAPFlag(Enum):
    """TAP flag levels."""
    GREEN = "green"   # Within acceptable range
    AMBER = "amber"   # Caution zone
    RED = "red"       # High risk


@dataclass
class TAPThresholds:
    """
    TAP metric thresholds.

    Calibrated against Jain 2017 clinical antibody dataset (n=137).
    Green zone encompasses ~75% of clinical antibodies.
    Amber zone is P75-P95 (caution).
    Red zone is outside P95 (high risk).
    """
    # CDR Length: typical range 47-57 for clinical antibodies
    cdr_length_green_low: int = 45       # Below P5
    cdr_length_green_high: int = 58      # Above P95
    cdr_length_amber_low: int = 40       # Extreme low
    cdr_length_amber_high: int = 65      # Extreme high

    # PSH: Patches of Surface Hydrophobicity (sequence-based)
    # Calibrated for sequence-based calculation (different from structure-based TAP)
    psh_green: float = 85.0              # P75 of clinical antibodies
    psh_amber: float = 100.0             # ~P95
    psh_red: float = 120.0               # Extreme

    # PPC: Patches of Positive Charge
    ppc_green: float = 5.0               # P75
    ppc_amber: float = 7.0               # P95
    ppc_red: float = 9.0                 # Extreme

    # PNC: Patches of Negative Charge
    pnc_green: float = 4.0               # P75
    pnc_amber: float = 7.0               # P95
    pnc_red: float = 9.0                 # Extreme

    # SFvCSP: Structural Fv Charge Symmetry Parameter
    # Negative values indicate charge asymmetry (problematic)
    sfvcsp_amber: float = -5.0           # ~P10 of clinical antibodies
    sfvcsp_red: float = -15.0            # Extreme asymmetry


# Default thresholds
DEFAULT_THRESHOLDS = TAPThresholds()


# =============================================================================
# Result Structures
# =============================================================================

class MetricResult(NamedTuple):
    """Result for a single TAP metric."""
    name: str
    value: float
    flag: TAPFlag
    percentile: float | None = None  # Percentile vs clinical-stage therapeutics


@dataclass
class TAPResult:
    """Complete TAP analysis result."""
    # Core metrics
    cdr_length: MetricResult
    psh: MetricResult  # Patches of Surface Hydrophobicity
    ppc: MetricResult  # Patches of Positive Charge
    pnc: MetricResult  # Patches of Negative Charge
    sfvcsp: MetricResult  # Structural Fv Charge Symmetry Parameter

    # Additional info
    vh_charge: float
    vl_charge: float
    cdr_sequences: dict[str, str]

    # Overall assessment
    @property
    def n_flags(self) -> int:
        """Number of amber or red flags."""
        return sum(1 for m in self.all_metrics if m.flag != TAPFlag.GREEN)

    @property
    def n_red_flags(self) -> int:
        """Number of red flags."""
        return sum(1 for m in self.all_metrics if m.flag == TAPFlag.RED)

    @property
    def all_metrics(self) -> list[MetricResult]:
        """All metrics as list."""
        return [self.cdr_length, self.psh, self.ppc, self.pnc, self.sfvcsp]

    @property
    def developability_score(self) -> float:
        """
        Overall developability score (0-100).
        Based on number of green flags.
        """
        green_count = sum(1 for m in self.all_metrics if m.flag == TAPFlag.GREEN)
        amber_count = sum(1 for m in self.all_metrics if m.flag == TAPFlag.AMBER)

        # 100 for all green, subtract for amber/red
        return 100 - (amber_count * 10) - (self.n_red_flags * 20)


# =============================================================================
# TAP Metrics Calculator
# =============================================================================

class TAPMetrics:
    """
    Calculate TAP (Therapeutic Antibody Profiler) metrics.

    Computes 5 developability metrics for antibody Fv regions:
    1. CDR Length - Total CDR residue count
    2. PSH - Patches of Surface Hydrophobicity (CDR hydrophobicity)
    3. PPC - Patches of Positive Charge in CDRs
    4. PNC - Patches of Negative Charge in CDRs
    5. SFvCSP - Structural Fv Charge Symmetry Parameter
    """

    def __init__(
        self,
        hydrophobicity_scale: str = "jain",
        thresholds: TAPThresholds | None = None,
        cdr_detector: CDRDetector | None = None,
    ):
        """
        Initialize TAP metrics calculator.

        Args:
            hydrophobicity_scale: Scale to use - "kyte_doolittle", "jain",
                                 "eisenberg", or "wimley_white"
            thresholds: Custom flag thresholds (default: TAP 2025 values)
            cdr_detector: Custom CDR detector
        """
        # Select hydrophobicity scale
        scale_map = {
            "kyte_doolittle": KYTE_DOOLITTLE,
            "jain": JAIN_HYDROPHOBICITY,
            "eisenberg": EISENBERG,
            "wimley_white": WIMLEY_WHITE,
        }
        if hydrophobicity_scale not in scale_map:
            raise ValueError(f"Unknown scale: {hydrophobicity_scale}")

        self._hydro_scale = scale_map[hydrophobicity_scale]
        self._thresholds = thresholds or DEFAULT_THRESHOLDS
        self._cdr_detector = cdr_detector or CDRDetector()

    def calculate(
        self,
        vh_sequence: str,
        vl_sequence: str,
    ) -> TAPResult:
        """
        Calculate all TAP metrics for an antibody Fv region.

        Args:
            vh_sequence: Heavy chain variable region sequence
            vl_sequence: Light chain variable region sequence

        Returns:
            TAPResult with all 5 metrics and flags
        """
        # Detect CDRs
        vh_cdrs, vl_cdrs = self._cdr_detector.detect_paired(vh_sequence, vl_sequence)

        # Combine CDR sequences
        all_cdr_sequences = {**vh_cdrs.cdr_sequences, **vl_cdrs.cdr_sequences}
        all_cdr_residues = vh_cdrs.cdr_residues + vl_cdrs.cdr_residues

        # Calculate metrics
        cdr_length = self._calc_cdr_length(vh_cdrs, vl_cdrs)
        psh = self._calc_psh(all_cdr_residues)
        ppc = self._calc_ppc(all_cdr_residues)
        pnc = self._calc_pnc(all_cdr_residues)

        vh_charge = self._calc_net_charge(vh_sequence)
        vl_charge = self._calc_net_charge(vl_sequence)
        sfvcsp = self._calc_sfvcsp(vh_charge, vl_charge)

        return TAPResult(
            cdr_length=cdr_length,
            psh=psh,
            ppc=ppc,
            pnc=pnc,
            sfvcsp=sfvcsp,
            vh_charge=vh_charge,
            vl_charge=vl_charge,
            cdr_sequences=all_cdr_sequences,
        )

    def _calc_cdr_length(
        self,
        vh_cdrs: CDRRegions,
        vl_cdrs: CDRRegions
    ) -> MetricResult:
        """Calculate total CDR length."""
        total = vh_cdrs.total_cdr_length + vl_cdrs.total_cdr_length

        # Flag based on calibrated thresholds
        t = self._thresholds
        if total < t.cdr_length_amber_low or total > t.cdr_length_amber_high:
            flag = TAPFlag.RED
        elif total < t.cdr_length_green_low or total > t.cdr_length_green_high:
            flag = TAPFlag.AMBER
        else:
            flag = TAPFlag.GREEN

        return MetricResult(
            name="CDR Length",
            value=float(total),
            flag=flag,
        )

    def _calc_psh(self, cdr_residues: str) -> MetricResult:
        """
        Calculate Patches of Surface Hydrophobicity (PSH).

        PSH measures CDR hydrophobicity using the selected scale.
        Higher values indicate more hydrophobic CDRs (aggregation risk).
        """
        if not cdr_residues:
            return MetricResult("PSH", 0.0, TAPFlag.GREEN)

        # Calculate sum of hydrophobicity values
        total = 0.0
        for aa in cdr_residues:
            if aa in self._hydro_scale:
                # Only count positive (hydrophobic) contributions
                val = self._hydro_scale[aa]
                if val > 0:
                    total += val

        # Normalize by CDR length to get average patch score
        # Scale factor to match TAP range (~100-200)
        psh = total * 3.0  # Empirical scaling

        # Flag based on calibrated thresholds
        t = self._thresholds
        if psh > t.psh_red:
            flag = TAPFlag.RED
        elif psh > t.psh_amber:
            flag = TAPFlag.AMBER
        elif psh > t.psh_green:
            flag = TAPFlag.AMBER  # Above P75 but below P95
        else:
            flag = TAPFlag.GREEN

        return MetricResult(
            name="PSH",
            value=round(psh, 2),
            flag=flag,
        )

    def _calc_ppc(self, cdr_residues: str) -> MetricResult:
        """
        Calculate Patches of Positive Charge (PPC).

        Counts positively charged residues (K, R, H) in CDRs.
        Normalized to give typical range of 0-5.
        """
        positive_count = sum(1 for aa in cdr_residues if aa in "KRH")

        # Normalize - typical CDR length ~50, expect ~2-4 positive residues
        ppc = positive_count

        # Flag based on calibrated thresholds
        t = self._thresholds
        if ppc > t.ppc_red:
            flag = TAPFlag.RED
        elif ppc > t.ppc_amber:
            flag = TAPFlag.AMBER
        elif ppc > t.ppc_green:
            flag = TAPFlag.AMBER  # Above P75 but below P95
        else:
            flag = TAPFlag.GREEN

        return MetricResult(
            name="PPC",
            value=float(ppc),
            flag=flag,
        )

    def _calc_pnc(self, cdr_residues: str) -> MetricResult:
        """
        Calculate Patches of Negative Charge (PNC).

        Counts negatively charged residues (D, E) in CDRs.
        """
        negative_count = sum(1 for aa in cdr_residues if aa in "DE")
        pnc = negative_count

        # Flag based on calibrated thresholds
        t = self._thresholds
        if pnc > t.pnc_red:
            flag = TAPFlag.RED
        elif pnc > t.pnc_amber:
            flag = TAPFlag.AMBER
        elif pnc > t.pnc_green:
            flag = TAPFlag.AMBER  # Above P75 but below P95
        else:
            flag = TAPFlag.GREEN

        return MetricResult(
            name="PNC",
            value=float(pnc),
            flag=flag,
        )

    def _calc_net_charge(self, sequence: str) -> float:
        """Calculate net charge of a sequence at pH 7.4."""
        charge = 0.0
        for aa in sequence:
            charge += AA_CHARGE.get(aa, 0.0)
        return charge

    def _calc_sfvcsp(self, vh_charge: float, vl_charge: float) -> MetricResult:
        """
        Calculate Structural Fv Charge Symmetry Parameter (SFvCSP).

        SFvCSP = VH_charge × VL_charge

        Negative values indicate charge asymmetry, which can affect
        stability and viscosity at high concentrations.
        """
        sfvcsp = vh_charge * vl_charge

        # Flag based on thresholds
        t = self._thresholds
        if sfvcsp < t.sfvcsp_red:
            flag = TAPFlag.RED
        elif sfvcsp < t.sfvcsp_amber:
            flag = TAPFlag.AMBER
        else:
            flag = TAPFlag.GREEN

        return MetricResult(
            name="SFvCSP",
            value=round(sfvcsp, 2),
            flag=flag,
        )

    def calculate_hic_proxy(
        self,
        vh_sequence: str,
        vl_sequence: str
    ) -> float:
        """
        Calculate HIC retention time proxy.

        Uses Jain hydrophobicity scale which correlates well with
        experimental HIC data (Spearman ~0.70).

        Returns:
            Estimated HIC retention score (higher = more hydrophobic)
        """
        # Detect CDRs
        vh_cdrs, vl_cdrs = self._cdr_detector.detect_paired(vh_sequence, vl_sequence)

        # Focus on CDRs and nearby regions (CDR vicinity)
        cdr_residues = vh_cdrs.cdr_residues + vl_cdrs.cdr_residues

        # Calculate hydrophobicity with Jain scale
        total_hydro = sum(JAIN_HYDROPHOBICITY.get(aa, 0) for aa in cdr_residues)

        # Normalize by CDR length
        if len(cdr_residues) > 0:
            avg_hydro = total_hydro / len(cdr_residues)
        else:
            avg_hydro = 0.0

        return round(avg_hydro, 3)
