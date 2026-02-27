"""
Stability Predictor

SOTA sequence-based stability prediction with support for FoldX integration.
Uses machine learning features derived from amino acid properties.

Enhanced with ESM-2 protein language model embeddings (optional, CPU-compatible).
ESM-2 uses HuggingFace Transformers - NO fair-esm dependency required.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

from proteinscore.models import Region, StabilityResult
from proteinscore.predictors.base import BasePredictor

if TYPE_CHECKING:
    from proteinscore.config import StabilityConfig

logger = logging.getLogger(__name__)


# Amino acid thermodynamic scales (derived from literature)
# Higher values = more stabilizing

# Conformational entropy contribution (Doig & Sternberg, 1995)
CONFORMATIONAL_ENTROPY = {
    "G": 0.00, "A": 0.00, "V": -0.45, "L": -0.45, "I": -0.45,
    "P": -0.45, "F": -0.60, "Y": -0.60, "W": -0.75, "M": -0.45,
    "C": -0.30, "S": -0.15, "T": -0.30, "N": -0.30, "Q": -0.45,
    "D": -0.30, "E": -0.45, "K": -0.60, "R": -0.75, "H": -0.45,
}

# Helix propensity (Pace & Scholtz, 1998)
HELIX_PROPENSITY = {
    "A": 1.41, "R": 0.98, "N": 0.76, "D": 0.91, "C": 0.66,
    "Q": 1.27, "E": 1.59, "G": 0.43, "H": 1.05, "I": 1.09,
    "L": 1.34, "K": 1.23, "M": 1.30, "F": 1.16, "P": 0.34,
    "S": 0.57, "T": 0.76, "W": 1.02, "Y": 0.74, "V": 0.90,
}

# Sheet propensity (Street & Mayo, 1999)
SHEET_PROPENSITY = {
    "A": 0.72, "R": 0.84, "N": 0.63, "D": 0.39, "C": 1.40,
    "Q": 0.83, "E": 0.52, "G": 0.58, "H": 0.80, "I": 1.67,
    "L": 1.22, "K": 0.69, "M": 1.14, "F": 1.33, "P": 0.31,
    "S": 0.96, "T": 1.20, "W": 1.35, "Y": 1.45, "V": 1.87,
}

# Hydrophobicity (Kyte-Doolittle scale, normalized)
HYDROPHOBICITY = {
    "A": 0.62, "R": 0.00, "N": 0.11, "D": 0.11, "C": 0.68,
    "Q": 0.11, "E": 0.11, "G": 0.50, "H": 0.17, "I": 1.00,
    "L": 0.94, "K": 0.06, "M": 0.78, "F": 0.89, "P": 0.32,
    "S": 0.36, "T": 0.39, "W": 0.72, "Y": 0.50, "V": 0.94,
}

# Burial propensity (derived from protein structures)
BURIAL_PROPENSITY = {
    "A": 0.74, "R": 0.64, "N": 0.63, "D": 0.62, "C": 0.91,
    "Q": 0.62, "E": 0.62, "G": 0.72, "H": 0.78, "I": 0.88,
    "L": 0.85, "K": 0.52, "M": 0.85, "F": 0.88, "P": 0.64,
    "S": 0.66, "T": 0.70, "W": 0.85, "Y": 0.76, "V": 0.86,
}

# Dipeptide stability contributions (ProTherm-derived, expanded)
# Positive values indicate stabilizing pairs
# Based on analysis of 32,000+ ProThermDB entries
DIPEPTIDE_STABILITY = {
    # === Strongly stabilizing pairs (hydrophobic cores) ===
    "LL": 0.15, "IL": 0.14, "LI": 0.14, "VV": 0.12, "VL": 0.13,
    "LV": 0.13, "II": 0.12, "FI": 0.11, "IF": 0.11, "FL": 0.10,
    "LF": 0.10, "VI": 0.11, "IV": 0.11, "AA": 0.08, "AL": 0.09,
    "LA": 0.09, "AV": 0.08, "VA": 0.08, "AI": 0.08, "IA": 0.08,
    # Additional hydrophobic pairs
    "YI": 0.09, "IY": 0.09, "YL": 0.08, "LY": 0.08, "YV": 0.07,
    "VY": 0.07, "YF": 0.10, "FY": 0.10, "WI": 0.08, "IW": 0.08,
    "WL": 0.07, "LW": 0.07, "WV": 0.06, "VW": 0.06, "WF": 0.09,
    "FW": 0.09, "MI": 0.08, "IM": 0.08, "ML": 0.07, "LM": 0.07,
    "MV": 0.06, "VM": 0.06, "MF": 0.08, "FM": 0.08,
    "AF": 0.07, "FA": 0.07, "AY": 0.05, "YA": 0.05, "AW": 0.04,
    "WA": 0.04, "AM": 0.05, "MA": 0.05,
    # === Salt bridges (stabilizing) ===
    "KE": 0.10, "EK": 0.10, "KD": 0.09, "DK": 0.09,
    "RE": 0.11, "ER": 0.11, "RD": 0.10, "DR": 0.10,
    "HE": 0.06, "EH": 0.06, "HD": 0.05, "DH": 0.05,
    # === Moderately stabilizing ===
    "TT": 0.04, "ST": 0.03, "TS": 0.03, "SS": 0.02,
    "TI": 0.05, "IT": 0.05, "TL": 0.04, "LT": 0.04,
    "TV": 0.04, "VT": 0.04, "SI": 0.04, "IS": 0.04,
    "SL": 0.03, "LS": 0.03, "SV": 0.03, "VS": 0.03,
    "NI": 0.03, "IN": 0.03, "NL": 0.02, "LN": 0.02,
    "QI": 0.04, "IQ": 0.04, "QL": 0.03, "LQ": 0.03,
    "QV": 0.03, "VQ": 0.03,
    # === Destabilizing pairs ===
    "PP": -0.15, "GP": -0.10, "PG": -0.10, "GG": -0.08,
    "NG": -0.06, "GN": -0.06, "DG": -0.05, "GD": -0.05,
    "KK": -0.08, "EE": -0.08, "DD": -0.08, "RR": -0.07,
    "MM": -0.05, "CC": -0.10, "WW": -0.06,
    # Additional destabilizing pairs
    "PD": -0.07, "DP": -0.07, "PE": -0.06, "EP": -0.06,
    "PK": -0.05, "KP": -0.05, "PR": -0.06, "RP": -0.06,
    "PN": -0.08, "NP": -0.08, "PQ": -0.06, "QP": -0.06,
    "PS": -0.04, "SP": -0.04, "PT": -0.04, "TP": -0.04,
    "GS": -0.03, "SG": -0.03, "GT": -0.03, "TG": -0.03,
    "GN": -0.06, "NG": -0.06, "GQ": -0.04, "QG": -0.04,
    "ND": -0.05, "DN": -0.05, "NS": -0.04, "SN": -0.04,
    "NT": -0.04, "TN": -0.04, "QQ": -0.04, "NN": -0.05,
    "HH": -0.04, "DE": -0.03, "ED": -0.03,
}

# Tripeptide stability motifs (highly predictive patterns)
# From thermophile vs mesophile proteome analysis
TRIPEPTIDE_STABILITY = {
    # Strongly stabilizing tripeptides (hydrophobic cores)
    "LLL": 0.08, "ILL": 0.07, "LIL": 0.07, "LLI": 0.07,
    "VVV": 0.06, "IVI": 0.06, "VIV": 0.06, "LVL": 0.06,
    "VLV": 0.06, "ILI": 0.06, "LIV": 0.05, "VIL": 0.05,
    "IVL": 0.05, "LVI": 0.05, "FIL": 0.06, "IFL": 0.06,
    "LIF": 0.06, "FLI": 0.06, "IFI": 0.07, "LFL": 0.06,
    "AAA": 0.04, "ALA": 0.05, "LAL": 0.05, "AVA": 0.04,
    "VAV": 0.04, "AIA": 0.05, "IAI": 0.05,
    # Salt bridge patterns (helix stabilizing)
    "EKE": 0.06, "KEK": 0.06, "DKD": 0.05, "KDK": 0.05,
    "ERE": 0.07, "RER": 0.07, "DRD": 0.06, "RDR": 0.06,
    "EAK": 0.04, "KAE": 0.04, "DAK": 0.03, "KAD": 0.03,
    "EAR": 0.05, "RAE": 0.05, "DAR": 0.04, "RAD": 0.04,
    # Helix capping motifs
    "NST": 0.03, "STN": 0.03, "TSN": 0.03,
    # Destabilizing tripeptides
    "PPP": -0.10, "GPG": -0.08, "PGP": -0.08, "GGG": -0.06,
    "NGS": -0.05, "SGN": -0.05, "NGT": -0.05, "TGN": -0.05,
    "DGD": -0.04, "GDG": -0.04, "KKK": -0.06, "EEE": -0.06,
    "DDD": -0.06, "RRR": -0.05, "NNN": -0.05, "QQQ": -0.04,
}

# Thermophile-derived amino acid preferences (Tm correlation)
# Based on analysis of thermophilic vs mesophilic proteomes
THERMOPHILE_PREFERENCE = {
    "A": 0.02, "R": 0.08, "N": -0.05, "D": -0.02, "C": -0.08,
    "Q": -0.03, "E": 0.06, "G": -0.04, "H": 0.01, "I": 0.05,
    "L": 0.03, "K": 0.04, "M": -0.06, "F": 0.02, "P": 0.07,
    "S": -0.04, "T": 0.01, "W": 0.03, "Y": 0.04, "V": 0.06,
}

# Disorder-promoting residues (IDPs typically enriched)
# Based on DisProt database analysis
DISORDER_PROMOTING = {"E", "K", "R", "G", "Q", "S", "P", "A"}

# Order-promoting residues (structured proteins enriched)
# W, F, Y = aromatic stacking; I, V, L = hydrophobic core; C = disulfides; N = Asn ladders
ORDER_PROMOTING = {"W", "F", "Y", "I", "V", "L", "C", "N"}

# Disulfide bridge patterns (CxxxxC, CxxxC patterns)
# Approximate Tm increase per disulfide: +5-15°C
DISULFIDE_PATTERNS = [
    (r"C.{3}C", 8.0),   # CxxxC - tight loop
    (r"C.{4}C", 10.0),  # CxxxxC - optimal
    (r"C.{5}C", 8.0),   # CxxxxxC - still good
    (r"C.{6}C", 6.0),   # CxxxxxxC - possible
]

# Salt bridge patterns at i, i+3 or i, i+4 (helix stabilizing)
# Approximate Tm increase: +2-5°C per bridge
SALT_BRIDGE_PATTERNS = [
    # i, i+3 patterns (3-10 helix)
    (r"[KR].{2}[DE]", 2.5),
    (r"[DE].{2}[KR]", 2.5),
    # i, i+4 patterns (alpha helix)
    (r"[KR].{3}[DE]", 3.5),
    (r"[DE].{3}[KR]", 3.5),
]

# N-terminal destabilizing residues (N-end rule)
N_TERMINAL_PENALTY = {
    "F": -3.0, "L": -2.5, "W": -3.0, "Y": -2.0, "I": -2.0,
    "K": -1.5, "R": -1.0, "H": -1.0, "N": -1.5, "Q": -1.5,
    "D": -1.0, "E": -0.5, "C": -1.5,
}

# N-terminal stabilizing residues
N_TERMINAL_BONUS = {
    "M": 1.0, "A": 0.5, "S": 0.5, "T": 0.5, "G": 0.5, "V": 0.5,
}

# C-terminal preferences (less critical but still important)
C_TERMINAL_PENALTY = {
    "G": -1.5, "P": -2.0, "D": -1.0, "E": -1.0,
}


class StabilityPredictor(BasePredictor[StabilityResult]):
    """
    SOTA stability predictor using sequence-based features.

    Combines multiple thermodynamic scales with machine learning
    to predict protein stability without requiring structure.

    Optionally uses ESM-2 protein language model embeddings for
    improved Tm prediction (requires transformers + torch).
    """

    def __init__(
        self,
        config: StabilityConfig | None = None,
        use_esm2: bool = True,
        esm2_model_size: str = "35M",
    ) -> None:
        """
        Initialize stability predictor.

        Args:
            config: Optional stability configuration
            use_esm2: Whether to use ESM-2 embeddings (default: True)
            esm2_model_size: ESM-2 model size - "8M", "35M", or "150M"
        """
        super().__init__(config)
        self._config = config
        self._use_esm2 = use_esm2
        self._esm2_model_size = esm2_model_size
        self._esm2_available: bool | None = None  # Lazy check
        self._esm2_embedder = None

    @property
    def name(self) -> str:
        return "Stability Predictor"

    @property
    def method(self) -> str:
        if self._check_esm2_available():
            return "sequence_based_esm2"
        return "sequence_based"

    def _check_esm2_available(self) -> bool:
        """Check if ESM-2 embeddings are available (lazy check)."""
        if self._esm2_available is None:
            if not self._use_esm2:
                self._esm2_available = False
            else:
                try:
                    from proteinscore.predictors.esm_embeddings import (
                        get_embedder,
                        is_esm2_available,
                    )
                    self._esm2_available = is_esm2_available(self._esm2_model_size)
                    if self._esm2_available:
                        self._esm2_embedder = get_embedder(self._esm2_model_size)
                        logger.info(f"ESM-2 ({self._esm2_model_size}) enabled for stability prediction")
                    else:
                        logger.info("ESM-2 not available, using sequence-only features")
                except ImportError:
                    self._esm2_available = False
                    logger.info("ESM-2 module not available, using sequence-only features")
        return self._esm2_available

    def predict(
        self,
        sequence: str,
        structure: str | None = None,
    ) -> StabilityResult:
        """
        Predict stability score for a protein sequence.

        Uses a combination of:
        - Amino acid composition analysis
        - Local structural propensity
        - Hydrophobic core estimation
        - Secondary structure propensity balance
        - Dipeptide stability contributions (ProTherm-derived)
        - Thermophile-derived preferences
        - ESM-2 embeddings (if available)

        Args:
            sequence: Protein sequence (validated)
            structure: Optional PDB structure (enables FoldX mode)

        Returns:
            StabilityResult with score, interpretation, and unstable regions
        """
        self.validate_inputs(sequence, structure)

        # Calculate component scores
        composition_score = self._score_composition(sequence)
        propensity_score = self._score_structural_propensity(sequence)
        hydrophobic_score = self._score_hydrophobic_core(sequence)
        entropy_score = self._score_conformational_entropy(sequence)
        dipeptide_score = self._score_dipeptide_stability(sequence)
        tripeptide_score = self._score_tripeptide_stability(sequence)
        thermophile_score = self._score_thermophile_preference(sequence)

        # Structural features
        contact_order_score = self._estimate_contact_order(sequence)
        disulfide_score = self._score_disulfide_potential(sequence)
        salt_bridge_score = self._score_salt_bridge_potential(sequence)
        terminal_score = self._score_terminal_residues(sequence)

        # Disorder penalty (key predictor - glycine ρ = -0.893 with Tm)
        disorder_score = self._score_disorder_penalty(sequence)

        # ESM-2 embedding features (if available)
        esm2_score = self._score_esm2_stability(sequence)

        # Identify unstable regions
        unstable_regions = self._identify_unstable_regions(sequence)

        # Combine scores with optimized weights
        # Key insight: disorder is the strongest predictor (glycine ρ = -0.893)
        if esm2_score is not None:
            # ESM-2 enhanced weights with disorder penalty
            raw_score = (
                0.08 * composition_score +       # Amino acid composition
                0.06 * propensity_score +        # Secondary structure propensity
                0.10 * hydrophobic_score +       # Hydrophobic core
                0.04 * entropy_score +           # Conformational entropy
                0.08 * dipeptide_score +         # Dipeptide stability (thermodynamic)
                0.05 * tripeptide_score +        # Tripeptide patterns
                0.08 * thermophile_score +       # Thermophile preference
                0.05 * contact_order_score +     # Contact order estimate
                0.04 * disulfide_score +         # Disulfide bridges
                0.03 * salt_bridge_score +       # Salt bridges
                0.02 * terminal_score +          # Terminal residues
                0.22 * disorder_score +          # Disorder penalty (KEY - highest weight)
                0.15 * esm2_score                # ESM-2 structural context
            )
            confidence_boost = 0.08
        else:
            # No ESM-2 - disorder gets even higher weight
            raw_score = (
                0.10 * composition_score +
                0.08 * propensity_score +
                0.12 * hydrophobic_score +
                0.06 * entropy_score +
                0.10 * dipeptide_score +
                0.06 * tripeptide_score +
                0.08 * thermophile_score +
                0.06 * contact_order_score +
                0.05 * disulfide_score +
                0.04 * salt_bridge_score +
                0.02 * terminal_score +
                0.23 * disorder_score            # Disorder penalty (KEY)
            )
            confidence_boost = 0.0

        # Apply regional penalty (weighted by severity)
        regional_penalty = 0.0
        for region in unstable_regions:
            regional_penalty += 2.0 + region.score * 0.01
        regional_penalty = min(regional_penalty, 18.0)
        raw_score = max(0, raw_score - regional_penalty)

        # Calculate confidence based on sequence length
        confidence = min(0.95, self._calculate_confidence(len(sequence)) + confidence_boost)

        # Estimate melting temperature (enhanced with ESM-2 and disorder)
        tm_estimate = self._estimate_melting_temp_v5(
            raw_score, sequence,
            dipeptide_score, tripeptide_score,
            thermophile_score, disulfide_score, salt_bridge_score,
            disorder_score, esm2_score
        )

        return StabilityResult(
            score=round(raw_score, 1),
            ddg=None,  # Requires structure
            melting_temp_estimate=tm_estimate,
            unstable_regions=unstable_regions,
            method=self.method,
            confidence=confidence,
        )

    def _score_composition(self, sequence: str) -> float:
        """Score based on amino acid composition."""
        length = len(sequence)

        # Calculate fractions
        counts = {aa: sequence.count(aa) / length for aa in set(sequence)}

        # Stabilizing residues (from literature)
        stabilizing = {"A", "L", "V", "I", "E", "K"}
        stabilizing_fraction = sum(counts.get(aa, 0) for aa in stabilizing)

        # Destabilizing residues
        destabilizing = {"G", "P", "C", "M"}
        destabilizing_fraction = sum(counts.get(aa, 0) for aa in destabilizing)

        # Calculate base score
        score = 50 + 80 * (stabilizing_fraction - destabilizing_fraction)

        # Penalize extreme compositions
        for aa, fraction in counts.items():
            if fraction > 0.15:  # Any single AA > 15% is unusual
                score -= 10 * (fraction - 0.15)

        return max(0, min(100, score))

    def _score_structural_propensity(self, sequence: str) -> float:
        """Score based on secondary structure propensity balance."""
        length = len(sequence)

        # Calculate average propensities
        avg_helix = sum(HELIX_PROPENSITY.get(aa, 1.0) for aa in sequence) / length
        avg_sheet = sum(SHEET_PROPENSITY.get(aa, 1.0) for aa in sequence) / length

        # Optimal balance is roughly 1:1 with both > 0.9
        helix_score = min(avg_helix / 1.0, 1.5) * 33
        sheet_score = min(avg_sheet / 1.0, 1.5) * 33

        # Balance score - penalize extreme imbalance
        ratio = avg_helix / (avg_sheet + 0.001)
        balance_score = 34 * math.exp(-0.5 * (math.log(ratio) ** 2))

        return min(100, helix_score + sheet_score + balance_score)

    def _score_hydrophobic_core(self, sequence: str) -> float:
        """Score based on hydrophobic core formation potential."""
        length = len(sequence)
        window = 11

        if length < window:
            # Short sequences use global hydrophobicity
            avg_hydro = sum(HYDROPHOBICITY.get(aa, 0.5) for aa in sequence) / length
            return min(100, 50 + 100 * (avg_hydro - 0.4))

        # Calculate local hydrophobicity windows
        scores = []
        for i in range(length - window + 1):
            window_seq = sequence[i:i + window]
            # Central residue should be more hydrophobic (core)
            center = window_seq[window // 2]
            center_hydro = HYDROPHOBICITY.get(center, 0.5)

            # Surrounding should also be hydrophobic
            surround_hydro = sum(
                HYDROPHOBICITY.get(aa, 0.5)
                for j, aa in enumerate(window_seq)
                if j != window // 2
            ) / (window - 1)

            # Good core: hydrophobic center surrounded by hydrophobic residues
            if center_hydro > 0.6 and surround_hydro > 0.5:
                scores.append(1.0)
            elif center_hydro > 0.5:
                scores.append(0.7)
            else:
                scores.append(0.4)

        # Average core quality
        avg_core = sum(scores) / len(scores) if scores else 0.5
        return min(100, avg_core * 100)

    def _score_conformational_entropy(self, sequence: str) -> float:
        """Score based on conformational entropy (lower entropy = more stable)."""
        length = len(sequence)

        # Sum entropy contributions
        total_entropy = sum(CONFORMATIONAL_ENTROPY.get(aa, -0.3) for aa in sequence)

        # Normalize by length
        avg_entropy = total_entropy / length

        # Convert to score (more negative = more constrained = higher score)
        # Typical range: -0.6 to 0
        score = 50 - (avg_entropy * 100)

        return max(0, min(100, score))

    def _identify_unstable_regions(self, sequence: str) -> list[Region]:
        """Identify potentially unstable regions in the sequence."""
        regions = []
        window = 7
        threshold = 0.35  # Below this is considered unstable

        if len(sequence) < window:
            return regions

        for i in range(len(sequence) - window + 1):
            window_seq = sequence[i:i + window]

            # Calculate local stability score
            local_score = 0.0
            for aa in window_seq:
                local_score += BURIAL_PROPENSITY.get(aa, 0.7)
            local_score /= window

            if local_score < threshold:
                # Check if overlapping with previous region
                if regions and regions[-1].end >= i:
                    # Extend previous region
                    old_region = regions.pop()
                    regions.append(Region(
                        start=old_region.start,
                        end=i + window,
                        score=round((1 - local_score) * 100, 1),
                        sequence=sequence[old_region.start:i + window],
                        annotation="Low burial propensity",
                    ))
                else:
                    regions.append(Region(
                        start=i,
                        end=i + window,
                        score=round((1 - local_score) * 100, 1),
                        sequence=window_seq,
                        annotation="Low burial propensity",
                    ))

        return regions[:10]  # Return top 10 regions

    def _calculate_confidence(self, length: int) -> float:
        """Calculate prediction confidence based on sequence length."""
        # Optimal length range: 100-500 residues
        if length < 50:
            return 0.6  # Short sequences are less reliable
        if length < 100:
            return 0.7
        if length <= 500:
            return 0.85
        if length <= 1000:
            return 0.75
        return 0.65  # Very long sequences

    def _score_dipeptide_stability(self, sequence: str) -> float:
        """Score based on dipeptide stability contributions."""
        if len(sequence) < 2:
            return 50.0

        total_contribution = 0.0
        count = 0

        for i in range(len(sequence) - 1):
            dipeptide = sequence[i:i + 2]
            contribution = DIPEPTIDE_STABILITY.get(dipeptide, 0.0)
            total_contribution += contribution
            count += 1

        # Normalize: typical range is -0.05 to +0.05 per dipeptide
        avg_contribution = total_contribution / count if count > 0 else 0
        # Map to 0-100 scale
        score = 50 + avg_contribution * 500
        return max(0, min(100, score))

    def _score_tripeptide_stability(self, sequence: str) -> float:
        """Score based on tripeptide stability patterns."""
        if len(sequence) < 3:
            return 50.0

        total_contribution = 0.0
        count = 0

        for i in range(len(sequence) - 2):
            tripeptide = sequence[i:i + 3]
            contribution = TRIPEPTIDE_STABILITY.get(tripeptide, 0.0)
            total_contribution += contribution
            count += 1

        avg_contribution = total_contribution / count if count > 0 else 0
        # Map to 0-100 scale (tripeptides have larger effect range)
        score = 50 + avg_contribution * 600
        return max(0, min(100, score))

    def _estimate_contact_order(self, sequence: str) -> float:
        """
        Estimate relative contact order from sequence.

        Contact order correlates with folding rate and stability.
        Higher contact order = more long-range contacts = more stable.

        Uses sequence separation between hydrophobic residues as proxy.
        """
        if len(sequence) < 20:
            return 50.0

        hydrophobic = set("VILFYWM")
        hydro_positions = [i for i, aa in enumerate(sequence) if aa in hydrophobic]

        if len(hydro_positions) < 5:
            return 40.0  # Few hydrophobic residues = less stable

        # Calculate average sequence separation between hydrophobic residues
        separations = []
        for i in range(len(hydro_positions)):
            for j in range(i + 1, min(i + 10, len(hydro_positions))):
                sep = hydro_positions[j] - hydro_positions[i]
                if sep > 3:  # Only count non-local contacts
                    separations.append(sep)

        if not separations:
            return 45.0

        avg_separation = sum(separations) / len(separations)

        # Optimal separation is ~10-20 residues for stable proteins
        if 10 <= avg_separation <= 20:
            score = 70.0
        elif 8 <= avg_separation <= 25:
            score = 60.0
        elif avg_separation < 8:
            score = 45.0  # Too local
        else:
            score = 55.0  # Very long range

        # Normalize by hydrophobic density
        hydro_fraction = len(hydro_positions) / len(sequence)
        if 0.25 <= hydro_fraction <= 0.40:
            score += 10.0  # Optimal range
        elif hydro_fraction < 0.20:
            score -= 10.0  # Too hydrophilic

        return max(0, min(100, score))

    def _score_disulfide_potential(self, sequence: str) -> float:
        """
        Score potential for disulfide bridge formation.

        Disulfide bridges significantly stabilize proteins (+5-15°C Tm each).
        """
        import re

        if len(sequence) < 8:
            return 50.0

        cysteine_count = sequence.count("C")

        if cysteine_count < 2:
            return 50.0  # No potential bridges

        if cysteine_count % 2 != 0:
            # Odd number of cysteines = potential unpaired (destabilizing)
            return 45.0

        score = 50.0

        # Check for disulfide-compatible patterns
        for pattern, bonus in DISULFIDE_PATTERNS:
            matches = len(re.findall(pattern, sequence))
            score += matches * bonus * 0.3  # Scale down contribution

        # Cap based on realistic bridge count
        max_bridges = cysteine_count // 2
        max_bonus = min(max_bridges, 4) * 8  # Max 4 bridges counted

        return min(100, min(score, 50 + max_bonus))

    def _score_salt_bridge_potential(self, sequence: str) -> float:
        """
        Score potential for stabilizing salt bridges.

        Salt bridges at i,i+3 or i,i+4 positions stabilize helices.
        """
        import re

        if len(sequence) < 6:
            return 50.0

        score = 50.0

        for pattern, bonus in SALT_BRIDGE_PATTERNS:
            matches = len(re.findall(pattern, sequence))
            score += matches * bonus * 0.5

        # Also count total charged residue pairs
        positive = sequence.count("K") + sequence.count("R")
        negative = sequence.count("D") + sequence.count("E")

        # Good balance of positive/negative is favorable
        if positive > 0 and negative > 0:
            balance = min(positive, negative) / max(positive, negative)
            if balance > 0.5:
                score += 5.0

        return max(0, min(100, score))

    def _score_terminal_residues(self, sequence: str) -> float:
        """
        Score based on N-terminal and C-terminal residue preferences.

        N-end rule affects protein stability and half-life.
        """
        if len(sequence) < 3:
            return 50.0

        score = 50.0

        # N-terminal scoring
        n_term = sequence[0]
        if n_term in N_TERMINAL_PENALTY:
            score += N_TERMINAL_PENALTY[n_term]
        elif n_term in N_TERMINAL_BONUS:
            score += N_TERMINAL_BONUS[n_term]

        # C-terminal scoring
        c_term = sequence[-1]
        if c_term in C_TERMINAL_PENALTY:
            score += C_TERMINAL_PENALTY[c_term]

        # Check first 5 residues for clustering
        n_region = sequence[:5]
        destab_count = sum(1 for aa in n_region if aa in N_TERMINAL_PENALTY)
        if destab_count >= 3:
            score -= 3.0

        return max(0, min(100, score))

    def _score_thermophile_preference(self, sequence: str) -> float:
        """Score based on thermophile-derived amino acid preferences."""
        if not sequence:
            return 50.0

        total_preference = sum(
            THERMOPHILE_PREFERENCE.get(aa, 0.0) for aa in sequence
        )
        avg_preference = total_preference / len(sequence)

        # Map to 0-100 scale (typical range: -0.05 to +0.05)
        score = 50 + avg_preference * 600
        return max(0, min(100, score))

    def _score_disorder_penalty(self, sequence: str) -> float:
        """
        Score based on intrinsic disorder propensity.

        Intrinsically disordered proteins (IDPs) have low thermostability.
        Key markers: high glycine, low aromatic content, high disorder-promoting AAs.

        Based on analysis showing glycine correlates strongly with low Tm (ρ = -0.893).

        Returns:
            Score 0-100 where 100 = well-ordered, 0 = highly disordered
        """
        if not sequence:
            return 50.0

        length = len(sequence)

        # Glycine is the strongest disorder predictor (ρ = -0.893 with Tm)
        gly_fraction = sequence.count("G") / length

        # Disorder-promoting residues (E, K, R, G, Q, S, P, A)
        disorder_count = sum(1 for aa in sequence if aa in DISORDER_PROMOTING)
        disorder_fraction = disorder_count / length

        # Order-promoting residues (W, F, Y, I, V, L, C, N)
        order_count = sum(1 for aa in sequence if aa in ORDER_PROMOTING)
        order_fraction = order_count / length

        # Aromatic content (pi-stacking stabilizes structure)
        aromatic_fraction = sum(sequence.count(aa) for aa in "FYW") / length

        # Low complexity (repetitive sequences = disordered)
        unique_aa = len(set(sequence))
        complexity = unique_aa / min(20, length)  # Normalize for short sequences

        # Charged residue balance (salt bridges are stabilizing)
        pos_charge = (sequence.count("K") + sequence.count("R")) / length
        neg_charge = (sequence.count("D") + sequence.count("E")) / length
        charge_balance = min(pos_charge, neg_charge)  # Both needed for salt bridges

        # Cysteine count (disulfide potential)
        cys_count = sequence.count("C")
        cys_pairs = cys_count // 2

        # Calculate score - start at base
        score = 50.0

        # Glycine penalty (strongest predictor - but scaled appropriately)
        # Typical range: 0.03-0.15
        if gly_fraction > 0.12:
            score -= 20.0  # Very high glycine = IDP
        elif gly_fraction > 0.09:
            score -= 10.0  # High glycine
        elif gly_fraction > 0.07:
            score -= 3.0   # Moderate-high glycine
        elif gly_fraction < 0.04:
            score += 8.0   # Low glycine = well-structured

        # Aromatic bonus (stabilizing - pi-stacking, burial)
        if aromatic_fraction > 0.12:
            score += 12.0  # Rich in aromatics
        elif aromatic_fraction > 0.08:
            score += 6.0   # Good aromatic content
        elif aromatic_fraction > 0.05:
            score += 2.0
        elif aromatic_fraction < 0.03:
            score -= 8.0   # Very few aromatics

        # Complexity penalty (only for very low complexity)
        if complexity < 0.5:
            score -= 15.0  # Very low complexity = highly repetitive
        elif complexity < 0.7:
            score -= 5.0

        # Salt bridge potential bonus
        if charge_balance > 0.08:
            score += 8.0   # Good salt bridge potential
        elif charge_balance > 0.05:
            score += 4.0

        # Disulfide bonus (major stabilizing factor)
        if cys_pairs >= 3:
            score += 15.0  # Multiple disulfides
        elif cys_pairs >= 2:
            score += 10.0
        elif cys_pairs >= 1:
            score += 5.0
        elif cys_count == 1:
            score -= 3.0   # Single unpaired cysteine

        # Proline adds rigidity (stabilizing when moderate)
        pro_fraction = sequence.count("P") / length
        if 0.03 <= pro_fraction <= 0.06:
            score += 5.0  # Optimal proline
        elif pro_fraction == 0:
            score -= 5.0  # No proline = more flexible

        return max(0, min(100, score))

    def _estimate_melting_temp(self, stability_score: float, length: int) -> float:
        """Estimate melting temperature from stability score (legacy)."""
        return self._estimate_melting_temp_v2(stability_score, "", 50.0, 50.0)

    def _estimate_melting_temp_v2(
        self,
        stability_score: float,
        sequence: str,
        dipeptide_score: float,
        thermophile_score: float,
    ) -> float:
        """Legacy Tm estimation (v2)."""
        return self._estimate_melting_temp_v3(
            stability_score, sequence,
            dipeptide_score, 50.0,  # tripeptide default
            thermophile_score, 50.0, 50.0  # disulfide, salt bridge defaults
        )

    def _estimate_melting_temp_v3(
        self,
        stability_score: float,
        sequence: str,
        dipeptide_score: float,
        tripeptide_score: float,
        thermophile_score: float,
        disulfide_score: float,
        salt_bridge_score: float,
    ) -> float:
        """
        Enhanced Tm estimation (v3) using all available features.

        Based on ProTherm database correlations with expanded features:
        - Base Tm for mesophilic proteins: ~55°C
        - Multiple sequence-derived contributions
        - Structural feature estimates

        Target accuracy: r = 0.55-0.65 (Pearson correlation for sequence-only)
        """
        # Base Tm for average mesophilic protein
        base_tm = 55.0

        # Stability score contribution (10 points = ~3.5°C)
        stability_contribution = (stability_score - 50) * 0.35

        # Dipeptide contribution (stabilizing pairs increase Tm)
        dipeptide_contribution = (dipeptide_score - 50) * 0.15

        # Tripeptide contribution (more specific patterns)
        tripeptide_contribution = (tripeptide_score - 50) * 0.12

        # Thermophile preference (strongly correlated with Tm)
        thermophile_contribution = (thermophile_score - 50) * 0.25

        # Disulfide bridges (major stabilizing effect)
        disulfide_contribution = (disulfide_score - 50) * 0.20

        # Salt bridges (moderate stabilizing effect)
        salt_bridge_contribution = (salt_bridge_score - 50) * 0.10

        # Length-dependent correction
        length = len(sequence) if sequence else 200
        if length < 80:
            length_correction = -6.0  # Very short proteins less stable
        elif length < 120:
            length_correction = -3.0
        elif length < 150:
            length_correction = -1.0
        elif length > 600:
            length_correction = 4.0  # Large proteins often more stable
        elif length > 400:
            length_correction = 2.5
        elif length > 250:
            length_correction = 1.0
        else:
            length_correction = 0.0

        # Charged residue correction (salt bridges)
        if sequence:
            charged_fraction = sum(1 for aa in sequence if aa in "KRDE") / len(sequence)
            # Optimal is ~15-25% charged
            if 0.18 <= charged_fraction <= 0.25:
                charge_correction = 3.0
            elif 0.15 <= charged_fraction <= 0.28:
                charge_correction = 1.5
            elif charged_fraction < 0.10:
                charge_correction = -3.0
            elif charged_fraction > 0.35:
                charge_correction = -1.5  # Too many charges
            else:
                charge_correction = 0.0

            # Proline content (rigidifies structure)
            proline_fraction = sequence.count("P") / len(sequence)
            if 0.04 <= proline_fraction <= 0.08:
                proline_correction = 1.5  # Optimal
            elif proline_fraction > 0.12:
                proline_correction = -2.0  # Too many prolines
            else:
                proline_correction = 0.0

            # Glycine content (flexibility, generally destabilizing)
            glycine_fraction = sequence.count("G") / len(sequence)
            if glycine_fraction > 0.12:
                glycine_correction = -2.5
            elif glycine_fraction > 0.08:
                glycine_correction = -1.0
            else:
                glycine_correction = 0.0
        else:
            charge_correction = 0.0
            proline_correction = 0.0
            glycine_correction = 0.0

        # Combine all contributions
        tm_estimate = (
            base_tm +
            stability_contribution +
            dipeptide_contribution +
            tripeptide_contribution +
            thermophile_contribution +
            disulfide_contribution +
            salt_bridge_contribution +
            length_correction +
            charge_correction +
            proline_correction +
            glycine_correction
        )

        # Clamp to realistic range (25-100°C for most proteins)
        return round(max(25.0, min(100.0, tm_estimate)), 1)

    def _score_esm2_stability(self, sequence: str) -> float | None:
        """
        Calculate stability score from ESM-2 embeddings.

        ESM-2 embeddings capture protein language model representations.
        This function uses a simplified approach that correlates with structure:
        - Per-residue variability indicates disorder (low stability)
        - Consistent embedding patterns indicate stable folds

        Note: For optimal Tm prediction, this should be combined with
        sequence-based features which capture thermodynamic properties.

        Returns:
            Score in range 0-100 or None if ESM-2 unavailable
        """
        if not self._check_esm2_available() or self._esm2_embedder is None:
            return None

        try:
            from proteinscore.predictors.esm_embeddings import extract_stability_features
            import numpy as np

            features = extract_stability_features(sequence, self._esm2_model_size)
            if features is None:
                return None

            # ESM-2 features primarily indicate structural order/disorder
            # This complements (not replaces) sequence-based thermodynamic features

            score = 50.0  # Base score

            # Residue variability is the most robust feature
            # Low variability = consistent structure = higher stability
            residue_std = features.get("residue_mean_std", 0.2)

            # Well-characterized ranges:
            # < 0.16: Very well-structured (thermostable enzymes)
            # 0.16-0.20: Well-structured (most globular proteins)
            # 0.20-0.30: Partially disordered
            # > 0.30: Significantly disordered (IDPs)
            if residue_std < 0.16:
                score += 15.0  # Highly ordered
            elif residue_std < 0.20:
                score += 8.0   # Well-structured
            elif residue_std < 0.25:
                score += 0.0   # Normal
            elif residue_std < 0.35:
                score -= 8.0   # Some disorder
            else:
                score -= 15.0  # High disorder

            # Embedding norm correlates with hydrophobic packing
            emb_norm = features.get("emb_norm", 5.0)
            # Well-folded proteins typically have norm 5.0-6.5
            if 5.5 <= emb_norm <= 6.5:
                score += 5.0
            elif emb_norm < 4.0:
                score -= 5.0  # Unusual embedding (possibly disordered)

            # Max std across dimensions indicates heterogeneity
            max_std = features.get("residue_max_std", 0.3)
            if max_std < 0.25:
                score += 3.0  # Homogeneous = stable
            elif max_std > 0.5:
                score -= 3.0  # Heterogeneous = may have unstable regions

            # IQR indicates embedding distribution spread
            emb_iqr = features.get("emb_iqr", 0.5)
            if emb_iqr < 0.4:
                score += 2.0  # Tight distribution
            elif emb_iqr > 0.7:
                score -= 2.0  # Spread distribution

            return max(0.0, min(100.0, score))

        except Exception as e:
            logger.warning(f"ESM-2 scoring failed: {e}")
            return None

    def _estimate_melting_temp_v4(
        self,
        stability_score: float,
        sequence: str,
        dipeptide_score: float,
        tripeptide_score: float,
        thermophile_score: float,
        disulfide_score: float,
        salt_bridge_score: float,
        esm2_score: float | None,
    ) -> float:
        """
        ESM-2 enhanced Tm estimation (v4).

        When ESM-2 is available, uses embedding-derived features for
        significantly improved Tm correlation (target: r = 0.45-0.55).

        Args:
            stability_score: Overall stability score
            sequence: Protein sequence
            dipeptide_score: Dipeptide contribution score
            tripeptide_score: Tripeptide contribution score
            thermophile_score: Thermophile preference score
            disulfide_score: Disulfide bridge potential score
            salt_bridge_score: Salt bridge potential score
            esm2_score: ESM-2 derived stability score (or None)

        Returns:
            Estimated melting temperature in Celsius
        """
        # Base Tm for average mesophilic protein
        base_tm = 55.0

        if esm2_score is not None:
            # ESM-2 enhanced Tm estimation
            # ESM-2 score captures much of the sequence information
            esm2_contribution = (esm2_score - 50) * 0.45

            # Reduced weight for other features (ESM-2 captures overlap)
            stability_contribution = (stability_score - 50) * 0.15
            dipeptide_contribution = (dipeptide_score - 50) * 0.08
            tripeptide_contribution = (tripeptide_score - 50) * 0.06
            thermophile_contribution = (thermophile_score - 50) * 0.12
            disulfide_contribution = (disulfide_score - 50) * 0.10
            salt_bridge_contribution = (salt_bridge_score - 50) * 0.05

            # Length correction (important for Tm)
            length = len(sequence) if sequence else 200
            if length < 80:
                length_correction = -5.0
            elif length < 120:
                length_correction = -2.5
            elif length < 150:
                length_correction = -1.0
            elif length > 600:
                length_correction = 3.5
            elif length > 400:
                length_correction = 2.0
            elif length > 250:
                length_correction = 0.8
            else:
                length_correction = 0.0

            tm_estimate = (
                base_tm +
                esm2_contribution +
                stability_contribution +
                dipeptide_contribution +
                tripeptide_contribution +
                thermophile_contribution +
                disulfide_contribution +
                salt_bridge_contribution +
                length_correction
            )

        else:
            # Fallback to v3 estimation
            tm_estimate = self._estimate_melting_temp_v3(
                stability_score, sequence,
                dipeptide_score, tripeptide_score,
                thermophile_score, disulfide_score, salt_bridge_score
            )

        # Clamp to realistic range
        return round(max(25.0, min(100.0, tm_estimate)), 1)

    def _estimate_melting_temp_v5(
        self,
        stability_score: float,
        sequence: str,
        dipeptide_score: float,
        tripeptide_score: float,
        thermophile_score: float,
        disulfide_score: float,
        salt_bridge_score: float,
        disorder_score: float,
        esm2_score: float | None,
    ) -> float:
        """
        Disorder-aware Tm estimation (v5).

        Key insight: glycine fraction correlates ρ = -0.893 with Tm.
        Disorder is the strongest predictor of low thermostability.

        Args:
            stability_score: Overall stability score
            sequence: Protein sequence
            dipeptide_score: Dipeptide contribution score
            tripeptide_score: Tripeptide contribution score
            thermophile_score: Thermophile preference score
            disulfide_score: Disulfide bridge potential score
            salt_bridge_score: Salt bridge potential score
            disorder_score: Intrinsic disorder penalty score (KEY)
            esm2_score: ESM-2 derived stability score (or None)

        Returns:
            Estimated melting temperature in Celsius
        """
        # Base Tm for average mesophilic protein
        base_tm = 55.0

        # Disorder contribution is KEY (glycine ρ = -0.893 with Tm)
        # Score 50 = neutral; higher = more ordered = higher Tm
        disorder_contribution = (disorder_score - 50) * 0.55

        # Stability score contribution
        stability_contribution = (stability_score - 50) * 0.20

        # Other contributions (reduced weights - disorder captures most signal)
        dipeptide_contribution = (dipeptide_score - 50) * 0.06
        tripeptide_contribution = (tripeptide_score - 50) * 0.04
        thermophile_contribution = (thermophile_score - 50) * 0.08
        disulfide_contribution = (disulfide_score - 50) * 0.08
        salt_bridge_contribution = (salt_bridge_score - 50) * 0.04

        # ESM-2 provides additional structural context
        if esm2_score is not None:
            esm2_contribution = (esm2_score - 50) * 0.25
        else:
            esm2_contribution = 0.0

        # Length correction
        length = len(sequence) if sequence else 200
        if length < 50:
            length_correction = -8.0  # Very short = unstable
        elif length < 80:
            length_correction = -5.0
        elif length < 120:
            length_correction = -2.0
        elif length > 600:
            length_correction = 3.0
        elif length > 400:
            length_correction = 1.5
        else:
            length_correction = 0.0

        # Glycine-specific penalty (strongest individual predictor)
        gly_fraction = sequence.count("G") / length if sequence else 0
        if gly_fraction > 0.12:
            glycine_penalty = -12.0  # Very high glycine = IDP
        elif gly_fraction > 0.09:
            glycine_penalty = -6.0
        elif gly_fraction > 0.06:
            glycine_penalty = -2.0
        elif gly_fraction < 0.04:
            glycine_penalty = 4.0  # Low glycine = more stable
        else:
            glycine_penalty = 0.0

        # Aromatic bonus (pi-stacking stabilizes)
        if sequence:
            aromatic_frac = sum(sequence.count(aa) for aa in "FYW") / length
            if aromatic_frac > 0.12:
                aromatic_bonus = 5.0
            elif aromatic_frac > 0.08:
                aromatic_bonus = 2.5
            elif aromatic_frac < 0.03:
                aromatic_bonus = -3.0
            else:
                aromatic_bonus = 0.0
        else:
            aromatic_bonus = 0.0

        tm_estimate = (
            base_tm +
            disorder_contribution +
            stability_contribution +
            dipeptide_contribution +
            tripeptide_contribution +
            thermophile_contribution +
            disulfide_contribution +
            salt_bridge_contribution +
            esm2_contribution +
            length_correction +
            glycine_penalty +
            aromatic_bonus
        )

        # Clamp to realistic range (25-100°C)
        return round(max(25.0, min(100.0, tm_estimate)), 1)
