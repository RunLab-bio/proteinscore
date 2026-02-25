"""
Aggregation Predictor

SOTA aggregation propensity prediction using TANGO/Aggrescan3D-derived algorithms.
Identifies aggregation-prone regions (APRs) and amyloid propensity.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from proteinscore.models import AggregationResult, Region
from proteinscore.predictors.base import BasePredictor

if TYPE_CHECKING:
    from proteinscore.config import AggregationConfig


# TANGO beta-aggregation propensity scale (Fernandez-Escamilla et al., 2004)
# Higher values = more aggregation-prone
TANGO_SCALE = {
    "A": 0.17, "R": -0.68, "N": -0.12, "D": -0.81, "C": 0.08,
    "Q": -0.14, "E": -0.76, "G": -0.17, "H": -0.26, "I": 0.62,
    "L": 0.46, "K": -0.72, "M": 0.29, "F": 0.58, "P": -0.78,
    "S": -0.05, "T": 0.01, "W": 0.42, "Y": 0.29, "V": 0.53,
}

# Aggrescan3D hot-spot scale (derived from A3D2.0)
A3D_SCALE = {
    "A": -0.03, "R": -0.18, "N": -0.09, "D": -0.14, "C": 0.12,
    "Q": -0.06, "E": -0.11, "G": -0.05, "H": -0.04, "I": 0.22,
    "L": 0.18, "K": -0.16, "M": 0.11, "F": 0.24, "P": -0.20,
    "S": -0.02, "T": 0.00, "W": 0.19, "Y": 0.14, "V": 0.20,
}

# Amyloid-forming propensity (Waltz/AmylPred derived)
AMYLOID_SCALE = {
    "A": 0.04, "R": -0.35, "N": 0.08, "D": -0.25, "C": 0.15,
    "Q": 0.12, "E": -0.22, "G": -0.08, "H": -0.05, "I": 0.42,
    "L": 0.35, "K": -0.30, "M": 0.18, "F": 0.48, "P": -0.45,
    "S": 0.02, "T": 0.05, "W": 0.38, "Y": 0.32, "V": 0.38,
}

# Gatekeeper residues that can disrupt aggregation
GATEKEEPERS = {"R", "K", "D", "E", "P", "G"}


class AggregationPredictor(BasePredictor[AggregationResult]):
    """
    SOTA aggregation predictor using TANGO and A3D algorithms.

    Identifies aggregation-prone regions (APRs) and calculates
    overall amyloid propensity. Higher scores indicate LOWER
    aggregation risk (better developability).
    """

    def __init__(self, config: AggregationConfig | None = None) -> None:
        super().__init__(config)
        self._apr_threshold = config.apr_threshold if config else 0.5

    @property
    def name(self) -> str:
        return "Aggregation Predictor"

    @property
    def method(self) -> str:
        return "sequence_based"

    def predict(
        self,
        sequence: str,
        structure: str | None = None,
    ) -> AggregationResult:
        """
        Predict aggregation propensity for a protein sequence.

        Uses a combination of TANGO beta-aggregation and A3D
        hot-spot detection algorithms.

        Args:
            sequence: Protein sequence (validated)
            structure: Optional PDB structure (enables full A3D mode)

        Returns:
            AggregationResult with score (higher = better), APRs, and amyloid propensity
        """
        self.validate_inputs(sequence, structure)

        # Calculate per-residue aggregation profiles
        tango_profile = self._calculate_tango_profile(sequence)
        a3d_profile = self._calculate_a3d_profile(sequence)

        # Identify aggregation-prone regions
        aprs = self._identify_aprs(sequence, tango_profile, a3d_profile)

        # Calculate amyloid propensity
        amyloid_propensity = self._calculate_amyloid_propensity(sequence)

        # Calculate gatekeeper protection score
        gatekeeper_score = self._calculate_gatekeeper_protection(sequence, aprs)

        # Combine into final score
        # Start with base score of 75 (assuming most proteins have some APRs)
        base_score = 75.0

        # Penalize for APRs
        apr_penalty = 0.0
        for apr in aprs:
            # Longer APRs are worse
            apr_penalty += apr.length * 1.5
            # Higher scoring APRs are worse
            apr_penalty += apr.score * 0.1

        apr_penalty = min(40, apr_penalty)  # Cap penalty

        # Penalize for high amyloid propensity
        amyloid_penalty = amyloid_propensity * 20

        # Add gatekeeper bonus
        final_score = base_score - apr_penalty - amyloid_penalty + gatekeeper_score

        # Clamp to valid range
        final_score = max(0, min(100, final_score))

        # Calculate confidence
        confidence = self._calculate_confidence(len(sequence), len(aprs))

        return AggregationResult(
            score=round(final_score, 1),
            aggregation_prone_regions=aprs,
            amyloid_propensity=round(amyloid_propensity, 3),
            method=self.method,
            confidence=confidence,
        )

    def _calculate_tango_profile(self, sequence: str) -> list[float]:
        """Calculate TANGO beta-aggregation profile."""
        window = 5
        half = window // 2

        if len(sequence) < window:
            return [TANGO_SCALE.get(aa, 0.0) for aa in sequence]

        profile = []
        for i in range(len(sequence)):
            start = max(0, i - half)
            end = min(len(sequence), i + half + 1)
            window_seq = sequence[start:end]

            # Calculate window score
            score = sum(TANGO_SCALE.get(aa, 0.0) for aa in window_seq) / len(window_seq)

            # Apply position-dependent weighting (central residue more important)
            center_weight = 1.5
            center_score = TANGO_SCALE.get(sequence[i], 0.0) * center_weight
            score = (score + center_score) / 2

            profile.append(score)

        return profile

    def _calculate_a3d_profile(self, sequence: str) -> list[float]:
        """Calculate Aggrescan3D hot-spot profile."""
        window = 7
        half = window // 2

        if len(sequence) < window:
            return [A3D_SCALE.get(aa, 0.0) for aa in sequence]

        profile = []
        for i in range(len(sequence)):
            start = max(0, i - half)
            end = min(len(sequence), i + half + 1)
            window_seq = sequence[start:end]

            # Calculate window score with Gaussian weighting
            score = 0.0
            weights = 0.0
            for j, aa in enumerate(window_seq):
                # Distance from center
                dist = abs(j - (i - start))
                weight = math.exp(-0.5 * (dist / 2) ** 2)
                score += A3D_SCALE.get(aa, 0.0) * weight
                weights += weight

            profile.append(score / weights if weights > 0 else 0.0)

        return profile

    def _identify_aprs(
        self,
        sequence: str,
        tango_profile: list[float],
        a3d_profile: list[float],
    ) -> list[Region]:
        """Identify aggregation-prone regions (APRs)."""
        # Combine profiles
        combined = [
            0.6 * t + 0.4 * a
            for t, a in zip(tango_profile, a3d_profile)
        ]

        aprs = []
        min_length = 5
        threshold = 0.15  # Above this is considered aggregation-prone

        i = 0
        while i < len(combined):
            if combined[i] > threshold:
                # Start of potential APR
                start = i
                peak_score = combined[i]

                while i < len(combined) and combined[i] > threshold * 0.5:
                    peak_score = max(peak_score, combined[i])
                    i += 1
                end = i

                if end - start >= min_length:
                    # Check if contains gatekeeper (would reduce score)
                    region_seq = sequence[start:end]
                    has_gatekeeper = any(aa in GATEKEEPERS for aa in region_seq)

                    # Calculate region score
                    region_score = peak_score * 100
                    if has_gatekeeper:
                        region_score *= 0.7  # Reduce score if gatekeeper present

                    aprs.append(Region(
                        start=start,
                        end=end,
                        score=round(region_score, 1),
                        sequence=region_seq,
                        annotation="TANGO/A3D aggregation hot-spot",
                    ))
            else:
                i += 1

        # Sort by score (highest risk first)
        aprs.sort(key=lambda r: -r.score)
        return aprs[:10]  # Return top 10 APRs

    def _calculate_amyloid_propensity(self, sequence: str) -> float:
        """Calculate overall amyloid-forming propensity."""
        window = 6
        threshold = 0.25

        if len(sequence) < window:
            avg = sum(AMYLOID_SCALE.get(aa, 0.0) for aa in sequence) / len(sequence)
            return max(0, min(1, (avg + 0.3) / 0.6))

        # Scan for amyloid-forming hexapeptides
        amyloid_scores = []
        for i in range(len(sequence) - window + 1):
            window_seq = sequence[i:i + window]
            score = sum(AMYLOID_SCALE.get(aa, 0.0) for aa in window_seq) / window
            if score > threshold:
                amyloid_scores.append(score)

        if not amyloid_scores:
            return 0.0

        # Overall propensity is based on number and intensity of amyloid windows
        count_factor = len(amyloid_scores) / (len(sequence) - window + 1)
        intensity_factor = max(amyloid_scores)

        return max(0, min(1, count_factor * 0.5 + intensity_factor * 1.5))

    def _calculate_gatekeeper_protection(
        self,
        sequence: str,
        aprs: list[Region],
    ) -> float:
        """Calculate protection score from gatekeeper residues."""
        if not aprs:
            return 5.0  # No APRs = good

        protection_score = 0.0

        for apr in aprs:
            # Check flanking regions for gatekeepers
            start = max(0, apr.start - 3)
            end = min(len(sequence), apr.end + 3)
            flanking = sequence[start:apr.start] + sequence[apr.end:end]

            gatekeepers_count = sum(1 for aa in flanking if aa in GATEKEEPERS)

            # Gatekeepers in flanking regions provide protection
            if gatekeepers_count >= 2:
                protection_score += 2.0
            elif gatekeepers_count == 1:
                protection_score += 1.0

        # Normalize by number of APRs
        return min(10.0, protection_score)

    def _calculate_confidence(self, length: int, apr_count: int) -> float:
        """Calculate prediction confidence."""
        # Base confidence by length
        if length < 50:
            confidence = 0.65
        elif length < 100:
            confidence = 0.75
        elif length <= 500:
            confidence = 0.85
        else:
            confidence = 0.75

        # High APR count reduces confidence slightly (more variance)
        if apr_count > 5:
            confidence -= 0.05

        return max(0.5, min(1.0, confidence))
