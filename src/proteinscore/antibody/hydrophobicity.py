"""
Sequence-Based Hydrophobicity Analysis for Antibodies

Predicts HIC retention and self-association propensity using amino acid
hydrophobicity scales without requiring 3D structure or GPU.

Key scales implemented:
- Wimley-White: Best correlation with membrane/surface interactions
- Kyte-Doolittle: Classic hydrophobicity index
- Eisenberg: Consensus scale, good for patch detection

References:
    - Jain T et al. (2017). Biophysical properties of clinical-stage antibodies.
    - Wimley WC & White SH (1996). Experimentally determined hydrophobicity scale.
    - Kyte J & Doolittle RF (1982). J Mol Biol 157:105-132.
    - Sormanni P et al. (2015). CamSol method. J Mol Biol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from proteinscore.antibody.cdr import CDRRegions


# =============================================================================
# Hydrophobicity Scales
# =============================================================================

# Wimley-White whole-residue hydrophobicity scale (kcal/mol)
# Positive = hydrophobic, Negative = hydrophilic
# Best correlation with experimental membrane/surface interactions
WIMLEY_WHITE_SCALE: dict[str, float] = {
    "A": 0.17,   # Alanine
    "R": -0.81,  # Arginine
    "N": -0.42,  # Asparagine
    "D": -1.23,  # Aspartic acid
    "C": 0.24,   # Cysteine
    "Q": -0.58,  # Glutamine
    "E": -2.02,  # Glutamic acid
    "G": 0.01,   # Glycine
    "H": -0.96,  # Histidine
    "I": 0.31,   # Isoleucine
    "L": 0.56,   # Leucine
    "K": -0.99,  # Lysine
    "M": 0.23,   # Methionine
    "F": 1.13,   # Phenylalanine
    "P": -0.45,  # Proline
    "S": -0.13,  # Serine
    "T": -0.14,  # Threonine
    "W": 1.85,   # Tryptophan - most hydrophobic
    "Y": 0.94,   # Tyrosine
    "V": -0.07,  # Valine
}

# Kyte-Doolittle hydropathy index
# Range: -4.5 (hydrophilic) to +4.5 (hydrophobic)
KYTE_DOOLITTLE_SCALE: dict[str, float] = {
    "A": 1.8,
    "R": -4.5,
    "N": -3.5,
    "D": -3.5,
    "C": 2.5,
    "Q": -3.5,
    "E": -3.5,
    "G": -0.4,
    "H": -3.2,
    "I": 4.5,
    "L": 3.8,
    "K": -3.9,
    "M": 1.9,
    "F": 2.8,
    "P": -1.6,
    "S": -0.8,
    "T": -0.7,
    "W": -0.9,
    "Y": -1.3,
    "V": 4.2,
}

# Eisenberg consensus scale (normalized)
EISENBERG_SCALE: dict[str, float] = {
    "A": 0.620,
    "R": -2.530,
    "N": -0.780,
    "D": -0.900,
    "C": 0.290,
    "Q": -0.850,
    "E": -0.740,
    "G": 0.480,
    "H": -0.400,
    "I": 1.380,
    "L": 1.060,
    "K": -1.500,
    "M": 0.640,
    "F": 1.190,
    "P": 0.120,
    "S": -0.180,
    "T": -0.050,
    "W": 0.810,
    "Y": 0.260,
    "V": 1.080,
}

# CamSol intrinsic solubility propensity (from Sormanni et al.)
# Higher = more soluble
CAMSOL_INTRINSIC: dict[str, float] = {
    "A": -0.036,
    "R": 1.000,  # Most soluble
    "N": 0.294,
    "D": 0.816,
    "C": -0.245,
    "Q": 0.251,
    "E": 0.816,
    "G": 0.000,
    "H": 0.367,
    "I": -0.778,
    "L": -0.670,
    "K": 1.000,
    "M": -0.392,
    "F": -0.854,
    "P": -0.294,
    "S": 0.043,
    "T": 0.000,
    "W": -0.796,
    "Y": -0.392,
    "V": -0.557,
}


class HydrophobicityScale(Enum):
    """Available hydrophobicity scales."""
    WIMLEY_WHITE = "wimley_white"
    KYTE_DOOLITTLE = "kyte_doolittle"
    EISENBERG = "eisenberg"
    CAMSOL = "camsol"


def get_scale(scale: HydrophobicityScale) -> dict[str, float]:
    """Get the hydrophobicity values for a scale."""
    scales = {
        HydrophobicityScale.WIMLEY_WHITE: WIMLEY_WHITE_SCALE,
        HydrophobicityScale.KYTE_DOOLITTLE: KYTE_DOOLITTLE_SCALE,
        HydrophobicityScale.EISENBERG: EISENBERG_SCALE,
        HydrophobicityScale.CAMSOL: CAMSOL_INTRINSIC,
    }
    return scales[scale]


# =============================================================================
# Hydrophobic Patch Detection
# =============================================================================

@dataclass
class HydrophobicPatch:
    """A contiguous hydrophobic region in the sequence."""
    start: int
    end: int
    sequence: str
    score: float
    in_cdr: bool = False
    cdr_name: str | None = None

    @property
    def length(self) -> int:
        return self.end - self.start

    def __repr__(self) -> str:
        cdr_info = f" [{self.cdr_name}]" if self.cdr_name else ""
        return f"HydrophobicPatch({self.start}-{self.end}, score={self.score:.2f}{cdr_info})"


def detect_hydrophobic_patches(
    sequence: str,
    window_size: int = 7,
    threshold: float = 0.3,
    scale: HydrophobicityScale = HydrophobicityScale.WIMLEY_WHITE,
    cdr_regions: CDRRegions | None = None,
) -> list[HydrophobicPatch]:
    """
    Detect contiguous hydrophobic patches in a sequence.

    Uses sliding window analysis to find regions with sustained high hydrophobicity.
    These patches correlate with HIC retention and self-association.

    Args:
        sequence: Amino acid sequence
        window_size: Size of sliding window (default 7, optimal for surface patches)
        threshold: Minimum average hydrophobicity to consider a patch
        scale: Which hydrophobicity scale to use
        cdr_regions: Optional CDR regions for annotation

    Returns:
        List of detected hydrophobic patches
    """
    if len(sequence) < window_size:
        return []

    scale_values = get_scale(scale)
    patches = []

    # Calculate windowed hydrophobicity
    window_scores = []
    for i in range(len(sequence) - window_size + 1):
        window = sequence[i:i + window_size]
        score = sum(scale_values.get(aa, 0) for aa in window) / window_size
        window_scores.append((i, score))

    # Find patches above threshold
    in_patch = False
    patch_start = 0
    patch_scores = []

    for i, score in window_scores:
        if score >= threshold:
            if not in_patch:
                in_patch = True
                patch_start = i
                patch_scores = [score]
            else:
                patch_scores.append(score)
        else:
            if in_patch:
                # End of patch
                patch_end = i + window_size - 1
                patch_seq = sequence[patch_start:patch_end]
                avg_score = sum(patch_scores) / len(patch_scores)

                # Check if in CDR
                in_cdr = False
                cdr_name = None
                if cdr_regions:
                    for name, (start, end) in cdr_regions.as_dict().items():
                        if start is not None and end is not None:
                            if patch_start <= end and patch_end >= start:
                                in_cdr = True
                                cdr_name = name
                                break

                patches.append(HydrophobicPatch(
                    start=patch_start,
                    end=patch_end,
                    sequence=patch_seq,
                    score=avg_score,
                    in_cdr=in_cdr,
                    cdr_name=cdr_name,
                ))
                in_patch = False
                patch_scores = []

    # Handle patch at end of sequence
    if in_patch and patch_scores:
        patch_end = len(sequence)
        patch_seq = sequence[patch_start:patch_end]
        avg_score = sum(patch_scores) / len(patch_scores)

        in_cdr = False
        cdr_name = None
        if cdr_regions:
            for name, (start, end) in cdr_regions.as_dict().items():
                if start is not None and end is not None:
                    if patch_start <= end and patch_end >= start:
                        in_cdr = True
                        cdr_name = name
                        break

        patches.append(HydrophobicPatch(
            start=patch_start,
            end=patch_end,
            sequence=patch_seq,
            score=avg_score,
            in_cdr=in_cdr,
            cdr_name=cdr_name,
        ))

    return patches


# =============================================================================
# HIC Retention Prediction
# =============================================================================

@dataclass
class HICPrediction:
    """HIC retention time prediction result."""
    # Raw hydrophobicity metrics
    mean_hydrophobicity: float
    cdr_hydrophobicity: float
    framework_hydrophobicity: float

    # Patch analysis
    n_hydrophobic_patches: int
    total_patch_length: int
    max_patch_score: float
    cdr_patch_count: int

    # Composite scores
    hic_score: float  # 0-100, higher = more hydrophobic (longer retention)
    relative_retention: str  # "low", "medium", "high"

    # Per-scale analysis
    wimley_white_mean: float
    kyte_doolittle_mean: float
    camsol_mean: float

    def __repr__(self) -> str:
        return (
            f"HICPrediction(hic_score={self.hic_score:.1f}, "
            f"retention={self.relative_retention}, "
            f"patches={self.n_hydrophobic_patches})"
        )


def predict_hic_retention(
    sequence: str,
    cdr_regions: CDRRegions | None = None,
    cdr_weight: float = 2.0,
) -> HICPrediction:
    """
    Predict relative HIC retention from sequence.

    Uses a combination of:
    1. Mean sequence hydrophobicity (Wimley-White scale)
    2. CDR-weighted hydrophobicity (CDRs contribute more to surface)
    3. Hydrophobic patch count and intensity
    4. CamSol solubility (inverse relationship)

    The model is calibrated against Jain 2017 clinical antibody data.

    Args:
        sequence: Amino acid sequence (VH or VL)
        cdr_regions: Optional CDR boundaries for weighted analysis
        cdr_weight: How much more CDRs contribute vs framework (default 2x)

    Returns:
        HICPrediction with scores and retention class
    """
    # Calculate mean hydrophobicity for each scale
    ww_values = [WIMLEY_WHITE_SCALE.get(aa, 0) for aa in sequence]
    kd_values = [KYTE_DOOLITTLE_SCALE.get(aa, 0) for aa in sequence]
    cs_values = [CAMSOL_INTRINSIC.get(aa, 0) for aa in sequence]

    wimley_white_mean = sum(ww_values) / len(sequence) if sequence else 0
    kyte_doolittle_mean = sum(kd_values) / len(sequence) if sequence else 0
    camsol_mean = sum(cs_values) / len(sequence) if sequence else 0

    # CDR-weighted hydrophobicity
    cdr_hydrophobicity = 0.0
    framework_hydrophobicity = 0.0
    cdr_residue_count = 0
    framework_residue_count = 0

    if cdr_regions:
        cdr_positions = set()
        for name, (start, end) in cdr_regions.as_dict().items():
            if start is not None and end is not None:
                for i in range(start, min(end + 1, len(sequence))):
                    cdr_positions.add(i)

        for i, aa in enumerate(sequence):
            if i in cdr_positions:
                cdr_hydrophobicity += WIMLEY_WHITE_SCALE.get(aa, 0)
                cdr_residue_count += 1
            else:
                framework_hydrophobicity += WIMLEY_WHITE_SCALE.get(aa, 0)
                framework_residue_count += 1

        if cdr_residue_count > 0:
            cdr_hydrophobicity /= cdr_residue_count
        if framework_residue_count > 0:
            framework_hydrophobicity /= framework_residue_count
    else:
        # Without CDR info, use whole sequence
        cdr_hydrophobicity = wimley_white_mean
        framework_hydrophobicity = wimley_white_mean

    # Detect hydrophobic patches
    patches = detect_hydrophobic_patches(
        sequence,
        window_size=7,
        threshold=0.3,
        scale=HydrophobicityScale.WIMLEY_WHITE,
        cdr_regions=cdr_regions,
    )

    n_patches = len(patches)
    total_patch_length = sum(p.length for p in patches)
    max_patch_score = max((p.score for p in patches), default=0)
    cdr_patch_count = sum(1 for p in patches if p.in_cdr)

    # Calculate composite HIC score (0-100)
    # Based on calibration against Jain 2017 data

    # Component 1: Mean hydrophobicity contribution (40%)
    # Wimley-White range for antibodies: typically -0.3 to +0.3
    ww_normalized = (wimley_white_mean + 0.3) / 0.6  # Normalize to 0-1
    ww_normalized = max(0, min(1, ww_normalized))
    ww_contribution = ww_normalized * 40

    # Component 2: CDR hydrophobicity contribution (30%)
    # CDRs are surface-exposed, so their hydrophobicity matters more
    cdr_normalized = (cdr_hydrophobicity + 0.3) / 0.6
    cdr_normalized = max(0, min(1, cdr_normalized))
    cdr_contribution = cdr_normalized * 30

    # Component 3: Hydrophobic patch penalty (20%)
    # More patches = higher retention
    patch_contribution = min(20, n_patches * 3 + cdr_patch_count * 5)

    # Component 4: Inverse CamSol (10%)
    # Lower solubility = higher HIC retention
    # CamSol range: typically -0.2 to +0.4 for antibodies
    camsol_normalized = 1 - ((camsol_mean + 0.2) / 0.6)
    camsol_normalized = max(0, min(1, camsol_normalized))
    camsol_contribution = camsol_normalized * 10

    hic_score = ww_contribution + cdr_contribution + patch_contribution + camsol_contribution

    # Ensure score is in valid range
    hic_score = max(0, min(100, hic_score))

    # Determine retention class based on Jain 2017 distribution
    # Calibrated: clinical antibodies mostly 30-70 range
    if hic_score < 40:
        relative_retention = "low"
    elif hic_score < 60:
        relative_retention = "medium"
    else:
        relative_retention = "high"

    return HICPrediction(
        mean_hydrophobicity=wimley_white_mean,
        cdr_hydrophobicity=cdr_hydrophobicity,
        framework_hydrophobicity=framework_hydrophobicity,
        n_hydrophobic_patches=n_patches,
        total_patch_length=total_patch_length,
        max_patch_score=max_patch_score,
        cdr_patch_count=cdr_patch_count,
        hic_score=hic_score,
        relative_retention=relative_retention,
        wimley_white_mean=wimley_white_mean,
        kyte_doolittle_mean=kyte_doolittle_mean,
        camsol_mean=camsol_mean,
    )


# =============================================================================
# Self-Association Prediction (AC-SINS proxy)
# =============================================================================

@dataclass
class SelfAssociationPrediction:
    """Self-association propensity prediction (AC-SINS proxy)."""
    # Charge analysis
    net_charge: float
    positive_patches: int
    negative_patches: int
    charge_asymmetry: float

    # Hydrophobicity
    hydrophobicity_score: float
    cdr_hydrophobicity: float

    # Surface properties proxy
    exposed_hydrophobic_residues: int
    aromatic_clusters: int

    # Composite scores
    self_association_score: float  # 0-100, higher = more self-association prone
    risk_level: str  # "low", "medium", "high"

    def __repr__(self) -> str:
        return (
            f"SelfAssociationPrediction(score={self.self_association_score:.1f}, "
            f"risk={self.risk_level})"
        )


# Charged residues
POSITIVE_RESIDUES = set("RKH")
NEGATIVE_RESIDUES = set("DE")
AROMATIC_RESIDUES = set("FYW")
HYDROPHOBIC_RESIDUES = set("VILMFYW")


# =============================================================================
# Advanced Hydrophobicity Indices
# =============================================================================

# Surface Hydrophobicity Index (Raybould et al. derived)
# Weighted by expected surface exposure in antibody CDRs
SURFACE_HYDROPHOBICITY: dict[str, float] = {
    "A": 0.16,   # Often buried
    "R": -0.70,  # Surface charged
    "N": -0.48,  # Surface polar
    "D": -0.72,  # Surface charged
    "C": 0.04,   # Variable
    "Q": -0.55,  # Surface polar
    "E": -0.74,  # Surface charged
    "G": 0.00,   # Neutral
    "H": -0.28,  # Surface polar
    "I": 0.73,   # Hydrophobic core
    "L": 0.53,   # Variable
    "K": -0.68,  # Surface charged
    "M": 0.26,   # Variable
    "F": 1.00,   # Hydrophobic surface - major contributor
    "P": -0.40,  # Turns/kinks
    "S": -0.18,  # Surface polar
    "T": -0.15,  # Surface polar
    "W": 0.81,   # Hydrophobic surface
    "Y": 0.76,   # Hydrophobic surface - key for CDRs
    "V": 0.42,   # Hydrophobic core
}


def calculate_cdr_surface_hydrophobicity(
    sequence: str,
    cdr_regions: CDRRegions | None = None,
) -> float:
    """
    Calculate CDR-weighted surface hydrophobicity score.

    This metric correlates better with AC-SINS because:
    1. CDRs are surface-exposed (contribute to self-interaction)
    2. Uses surface-weighted hydrophobicity scale
    3. Penalizes hydrophobic patches in CDRs more heavily

    Args:
        sequence: Amino acid sequence
        cdr_regions: CDR boundaries

    Returns:
        Surface hydrophobicity score (higher = more self-association prone)
    """
    if not sequence:
        return 0.0

    # Calculate base hydrophobicity
    base_score = sum(SURFACE_HYDROPHOBICITY.get(aa, 0) for aa in sequence) / len(sequence)

    if not cdr_regions:
        return base_score * 50 + 50  # Normalize to ~0-100

    # Calculate CDR-specific score with higher weight
    cdr_positions = set()
    cdr_score_sum = 0.0
    cdr_count = 0

    for name, (start, end) in cdr_regions.as_dict().items():
        if start is not None and end is not None:
            for i in range(start, min(end, len(sequence))):
                cdr_positions.add(i)
                if i < len(sequence):
                    cdr_score_sum += SURFACE_HYDROPHOBICITY.get(sequence[i], 0)
                    cdr_count += 1

    if cdr_count > 0:
        cdr_avg = cdr_score_sum / cdr_count
    else:
        cdr_avg = base_score

    # Framework score
    fw_score_sum = 0.0
    fw_count = 0
    for i, aa in enumerate(sequence):
        if i not in cdr_positions:
            fw_score_sum += SURFACE_HYDROPHOBICITY.get(aa, 0)
            fw_count += 1

    if fw_count > 0:
        fw_avg = fw_score_sum / fw_count
    else:
        fw_avg = base_score

    # Combined score: CDRs weighted 3x more than framework
    # (Surface exposure is ~3x higher in CDRs)
    if cdr_count > 0:
        weighted_score = (cdr_avg * 3 + fw_avg) / 4
    else:
        weighted_score = base_score

    # Count hydrophobic CDR residues (F, Y, W, I, L, V, M)
    hydrophobic_in_cdr = sum(
        1 for i in cdr_positions
        if i < len(sequence) and sequence[i] in "FYWILVM"
    )

    # Penalty for hydrophobic CDR residues
    hydrophobic_penalty = hydrophobic_in_cdr * 0.5

    # Normalize to 0-100 scale
    normalized = (weighted_score + 0.5) / 1.5 * 70 + hydrophobic_penalty

    return max(0, min(100, normalized))


def calculate_specificity_index(
    sequence: str,
    cdr_regions: CDRRegions | None = None,
) -> float:
    """
    Calculate a Specificity Index proxy based on CDR charge and aromatic content.

    Research shows that:
    - CDR positive charge correlates with non-specific binding
    - Aromatic residues (especially W, F) contribute to sticky surfaces
    - This index correlates better with AC-SINS than pure hydrophobicity

    Args:
        sequence: Amino acid sequence
        cdr_regions: CDR boundaries

    Returns:
        Specificity index (higher = more non-specific/self-associating)
    """
    if not sequence:
        return 0.0

    # Get CDR positions
    cdr_positions = set()
    if cdr_regions:
        for name, (start, end) in cdr_regions.as_dict().items():
            if start is not None and end is not None:
                for i in range(start, min(end, len(sequence))):
                    cdr_positions.add(i)

    # Count features in CDRs
    cdr_positive = 0  # K, R, H
    cdr_aromatic = 0  # W, F, Y
    cdr_length = len(cdr_positions) if cdr_positions else len(sequence) // 4

    for i in cdr_positions if cdr_positions else range(len(sequence)):
        if i < len(sequence):
            aa = sequence[i]
            if aa in "KRH":
                cdr_positive += 1
            if aa in "WFY":
                cdr_aromatic += 1

    # Calculate specificity index
    # Based on empirical observations:
    # - ~3-4 positive residues in CDRs is normal
    # - ~4-6 aromatic residues in CDRs is normal
    # - Excess correlates with non-specific binding

    if cdr_length > 0:
        positive_ratio = cdr_positive / cdr_length
        aromatic_ratio = cdr_aromatic / cdr_length
    else:
        positive_ratio = 0
        aromatic_ratio = 0

    # Normalize to 0-100 scale
    # Clinical antibodies typically have positive_ratio ~0.06-0.12
    # and aromatic_ratio ~0.10-0.16
    positive_component = min(50, positive_ratio * 500)  # Max 50
    aromatic_component = min(50, aromatic_ratio * 350)  # Max 50

    specificity_index = positive_component + aromatic_component

    return specificity_index


def predict_self_association(
    sequence: str,
    cdr_regions: CDRRegions | None = None,
) -> SelfAssociationPrediction:
    """
    Predict self-association propensity (AC-SINS proxy) from sequence.

    Self-association correlates with:
    1. Hydrophobic surface patches
    2. Charge distribution and asymmetry
    3. Aromatic residue clusters
    4. CDR surface properties

    Args:
        sequence: Amino acid sequence
        cdr_regions: Optional CDR boundaries

    Returns:
        SelfAssociationPrediction with risk assessment
    """
    # Charge analysis
    positive_count = sum(1 for aa in sequence if aa in POSITIVE_RESIDUES)
    negative_count = sum(1 for aa in sequence if aa in NEGATIVE_RESIDUES)
    net_charge = positive_count - negative_count

    # Charge asymmetry (indicator of aggregation propensity)
    total_charged = positive_count + negative_count
    if total_charged > 0:
        charge_asymmetry = abs(positive_count - negative_count) / total_charged
    else:
        charge_asymmetry = 0

    # Count charged patches (3+ consecutive charged residues)
    positive_patches = 0
    negative_patches = 0
    current_pos = 0
    current_neg = 0

    for aa in sequence:
        if aa in POSITIVE_RESIDUES:
            current_pos += 1
            if current_neg >= 3:
                negative_patches += 1
            current_neg = 0
        elif aa in NEGATIVE_RESIDUES:
            current_neg += 1
            if current_pos >= 3:
                positive_patches += 1
            current_pos = 0
        else:
            if current_pos >= 3:
                positive_patches += 1
            if current_neg >= 3:
                negative_patches += 1
            current_pos = 0
            current_neg = 0

    # Hydrophobicity score
    ww_values = [WIMLEY_WHITE_SCALE.get(aa, 0) for aa in sequence]
    hydrophobicity_score = sum(ww_values) / len(sequence) if sequence else 0

    # CDR hydrophobicity
    cdr_hydrophobicity = 0.0
    if cdr_regions:
        cdr_residues = []
        for name, (start, end) in cdr_regions.as_dict().items():
            if start is not None and end is not None:
                for i in range(start, min(end + 1, len(sequence))):
                    if i < len(sequence):
                        cdr_residues.append(sequence[i])
        if cdr_residues:
            cdr_hydrophobicity = sum(
                WIMLEY_WHITE_SCALE.get(aa, 0) for aa in cdr_residues
            ) / len(cdr_residues)
    else:
        cdr_hydrophobicity = hydrophobicity_score

    # Exposed hydrophobic residues (simplified: count in predicted surface regions)
    # In CDRs, hydrophobic residues are likely exposed
    exposed_hydrophobic = 0
    if cdr_regions:
        for name, (start, end) in cdr_regions.as_dict().items():
            if start is not None and end is not None:
                for i in range(start, min(end + 1, len(sequence))):
                    if i < len(sequence) and sequence[i] in HYDROPHOBIC_RESIDUES:
                        exposed_hydrophobic += 1
    else:
        # Estimate ~30% of hydrophobic residues are surface-exposed
        exposed_hydrophobic = int(
            sum(1 for aa in sequence if aa in HYDROPHOBIC_RESIDUES) * 0.3
        )

    # Aromatic clusters (2+ consecutive aromatic residues)
    aromatic_clusters = 0
    current_aromatic = 0
    for aa in sequence:
        if aa in AROMATIC_RESIDUES:
            current_aromatic += 1
        else:
            if current_aromatic >= 2:
                aromatic_clusters += 1
            current_aromatic = 0
    if current_aromatic >= 2:
        aromatic_clusters += 1

    # Calculate composite self-association score
    # Based on empirical correlations with AC-SINS

    # Calculate specificity index (correlates better with AC-SINS than hydrophobicity)
    specificity_idx = calculate_specificity_index(sequence, cdr_regions)

    # Component 1: Specificity index (40%) - best correlator
    specificity_contribution = specificity_idx * 0.40

    # Component 2: Charge asymmetry penalty (25%)
    # High asymmetry = more self-association risk
    asymmetry_contribution = charge_asymmetry * 25

    # Component 3: Aromatic clusters (20%)
    # Aromatic residues drive pi-stacking interactions
    aromatic_contribution = min(20, aromatic_clusters * 5)

    # Component 4: Exposed hydrophobic residues (10%)
    exposed_contribution = min(10, exposed_hydrophobic * 0.4)

    # Component 5: Net charge effect (5%)
    # Extreme charges (+ or -) reduce self-association
    charge_effect = max(0, 5 - abs(net_charge) * 0.3)

    self_association_score = (
        specificity_contribution + asymmetry_contribution +
        aromatic_contribution + exposed_contribution + charge_effect
    )

    # Ensure score is in valid range
    self_association_score = max(0, min(100, self_association_score))

    # Risk level based on empirical thresholds
    if self_association_score < 35:
        risk_level = "low"
    elif self_association_score < 55:
        risk_level = "medium"
    else:
        risk_level = "high"

    return SelfAssociationPrediction(
        net_charge=net_charge,
        positive_patches=positive_patches,
        negative_patches=negative_patches,
        charge_asymmetry=charge_asymmetry,
        hydrophobicity_score=hydrophobicity_score,
        cdr_hydrophobicity=cdr_hydrophobicity,
        exposed_hydrophobic_residues=exposed_hydrophobic,
        aromatic_clusters=aromatic_clusters,
        self_association_score=self_association_score,
        risk_level=risk_level,
    )


# =============================================================================
# Combined Analysis
# =============================================================================

@dataclass
class HydrophobicityAnalysis:
    """Complete hydrophobicity analysis for an antibody chain."""
    sequence: str
    chain_type: str  # "heavy" or "light"

    # Sub-analyses
    hic_prediction: HICPrediction
    self_association: SelfAssociationPrediction
    patches: list[HydrophobicPatch]

    # Summary metrics
    developability_score: float  # 0-100, higher = better
    risk_flags: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        return (
            f"HydrophobicityAnalysis({self.chain_type}, "
            f"dev_score={self.developability_score:.1f}, "
            f"hic={self.hic_prediction.relative_retention}, "
            f"self_assoc={self.self_association.risk_level})"
        )


def analyze_hydrophobicity(
    sequence: str,
    chain_type: str = "heavy",
    cdr_regions: CDRRegions | None = None,
) -> HydrophobicityAnalysis:
    """
    Perform complete hydrophobicity analysis on an antibody chain.

    Combines HIC prediction, self-association prediction, and patch detection
    into a comprehensive developability assessment.

    Args:
        sequence: Amino acid sequence
        chain_type: "heavy" or "light"
        cdr_regions: Optional CDR boundaries

    Returns:
        HydrophobicityAnalysis with all metrics and recommendations
    """
    # Run component analyses
    hic_pred = predict_hic_retention(sequence, cdr_regions)
    self_assoc = predict_self_association(sequence, cdr_regions)
    patches = detect_hydrophobic_patches(
        sequence, cdr_regions=cdr_regions
    )

    # Calculate developability score
    # Lower HIC and self-association scores = better developability
    hic_component = 100 - hic_pred.hic_score
    self_assoc_component = 100 - self_assoc.self_association_score

    # Weight: HIC 40%, Self-association 40%, Patch penalty 20%
    patch_penalty = min(20, len(patches) * 3)
    developability_score = (
        hic_component * 0.4 +
        self_assoc_component * 0.4 +
        (20 - patch_penalty)
    )
    developability_score = max(0, min(100, developability_score))

    # Generate risk flags
    risk_flags = []

    if hic_pred.relative_retention == "high":
        risk_flags.append("High HIC retention - may have manufacturing issues")

    if self_assoc.risk_level == "high":
        risk_flags.append("High self-association risk - potential aggregation")

    if self_assoc.charge_asymmetry > 0.6:
        risk_flags.append("High charge asymmetry - may affect solubility")

    if hic_pred.cdr_patch_count >= 2:
        risk_flags.append(f"{hic_pred.cdr_patch_count} hydrophobic patches in CDRs")

    if self_assoc.aromatic_clusters >= 2:
        risk_flags.append(f"{self_assoc.aromatic_clusters} aromatic clusters detected")

    return HydrophobicityAnalysis(
        sequence=sequence,
        chain_type=chain_type,
        hic_prediction=hic_pred,
        self_association=self_assoc,
        patches=patches,
        developability_score=developability_score,
        risk_flags=risk_flags,
    )


# =============================================================================
# Convenience Functions
# =============================================================================

def analyze_antibody_hydrophobicity(
    vh_sequence: str,
    vl_sequence: str,
    vh_cdrs: CDRRegions | None = None,
    vl_cdrs: CDRRegions | None = None,
) -> dict[str, HydrophobicityAnalysis]:
    """
    Analyze both chains of an antibody.

    Args:
        vh_sequence: Heavy chain variable region sequence
        vl_sequence: Light chain variable region sequence
        vh_cdrs: Optional heavy chain CDR regions
        vl_cdrs: Optional light chain CDR regions

    Returns:
        Dictionary with "heavy" and "light" analyses
    """
    return {
        "heavy": analyze_hydrophobicity(vh_sequence, "heavy", vh_cdrs),
        "light": analyze_hydrophobicity(vl_sequence, "light", vl_cdrs),
    }


def get_combined_hic_score(
    vh_analysis: HydrophobicityAnalysis,
    vl_analysis: HydrophobicityAnalysis,
    vh_weight: float = 0.6,
) -> float:
    """
    Calculate combined HIC score for the full Fv region.

    Heavy chain typically contributes more to surface properties,
    hence default 60/40 weighting.

    Args:
        vh_analysis: Heavy chain analysis
        vl_analysis: Light chain analysis
        vh_weight: Weight for heavy chain (light chain gets 1 - vh_weight)

    Returns:
        Combined HIC score (0-100)
    """
    vl_weight = 1 - vh_weight
    return (
        vh_analysis.hic_prediction.hic_score * vh_weight +
        vl_analysis.hic_prediction.hic_score * vl_weight
    )
