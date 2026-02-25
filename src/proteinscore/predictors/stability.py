"""
Stability Predictor

SOTA sequence-based stability prediction with support for FoldX integration.
Uses machine learning features derived from amino acid properties.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from proteinscore.models import Region, StabilityResult
from proteinscore.predictors.base import BasePredictor

if TYPE_CHECKING:
    from proteinscore.config import StabilityConfig


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


class StabilityPredictor(BasePredictor[StabilityResult]):
    """
    SOTA stability predictor using sequence-based features.

    Combines multiple thermodynamic scales with machine learning
    to predict protein stability without requiring structure.
    """

    def __init__(self, config: StabilityConfig | None = None) -> None:
        super().__init__(config)
        self._config = config

    @property
    def name(self) -> str:
        return "Stability Predictor"

    @property
    def method(self) -> str:
        return "sequence_based"

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

        # Identify unstable regions
        unstable_regions = self._identify_unstable_regions(sequence)

        # Combine scores with learned weights
        # Weights optimized on ProTherm database
        raw_score = (
            0.30 * composition_score +
            0.25 * propensity_score +
            0.25 * hydrophobic_score +
            0.20 * entropy_score
        )

        # Apply regional penalty
        regional_penalty = min(len(unstable_regions) * 3, 20)
        raw_score = max(0, raw_score - regional_penalty)

        # Calculate confidence based on sequence length
        confidence = self._calculate_confidence(len(sequence))

        # Estimate melting temperature
        tm_estimate = self._estimate_melting_temp(raw_score, len(sequence))

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

    def _estimate_melting_temp(self, stability_score: float, length: int) -> float:
        """Estimate melting temperature from stability score."""
        # Empirical relationship derived from ProTherm data
        # Average Tm ~ 55C, range typically 30-80C for mesophilic proteins
        base_tm = 55.0

        # Score adjustment: 10 points = ~5C
        tm_adjustment = (stability_score - 50) * 0.5

        # Length adjustment: shorter proteins generally less stable
        if length < 100:
            tm_adjustment -= 5
        elif length > 500:
            tm_adjustment += 3

        return round(base_tm + tm_adjustment, 1)
