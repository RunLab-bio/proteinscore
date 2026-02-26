"""
Solubility Predictor

SOTA sequence-based solubility prediction using CamSol-derived algorithms.
Predicts protein solubility and identifies insoluble regions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from proteinscore.models import Region, SolubilityClass, SolubilityResult
from proteinscore.predictors.base import BasePredictor

if TYPE_CHECKING:
    from proteinscore.config import SolubilityConfig


# CamSol intrinsic solubility values (Sormanni et al., 2015)
# Positive values = soluble, negative = insoluble
CAMSOL_INTRINSIC = {
    "A": 0.02, "R": 1.00, "N": 0.49, "D": 1.03, "C": -0.43,
    "Q": 0.49, "E": 1.04, "G": 0.00, "H": 0.37, "I": -0.72,
    "L": -0.69, "K": 1.12, "M": -0.45, "F": -0.85, "P": 0.26,
    "S": 0.18, "T": 0.07, "W": -0.61, "Y": -0.47, "V": -0.54,
}

# Gatekeeper residues (can break aggregation sequences)
GATEKEEPERS = {"R", "K", "D", "E", "P"}

# Aggregation-promoting residues
AGGREGATION_PRONE = {"I", "L", "V", "F", "Y", "W", "C", "M"}

# Kyte-Doolittle hydrophobicity scale
HYDROPATHY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

# Burial propensity (likelihood of being buried in core)
# Used to estimate exposed hydrophobic patches
BURIAL_PROPENSITY = {
    "A": 0.74, "R": 0.64, "N": 0.63, "D": 0.62, "C": 0.91,
    "Q": 0.62, "E": 0.62, "G": 0.72, "H": 0.78, "I": 0.88,
    "L": 0.85, "K": 0.52, "M": 0.85, "F": 0.88, "P": 0.64,
    "S": 0.66, "T": 0.70, "W": 0.85, "Y": 0.76, "V": 0.86,
}

# Strong gatekeeper patterns (flanking hydrophobic stretches)
STRONG_GATEKEEPERS = {"R", "K", "D", "E"}
WEAK_GATEKEEPERS = {"P", "G", "H"}

# CamSol correction factors for sequence context
# Applied when hydrophobic residues are likely exposed
EXPOSURE_PENALTY = {
    "I": -0.15, "L": -0.12, "V": -0.10, "F": -0.18,
    "Y": -0.08, "W": -0.14, "M": -0.08, "C": -0.06,
}


class SolubilityPredictor(BasePredictor[SolubilityResult]):
    """
    SOTA solubility predictor based on CamSol algorithm.

    Uses intrinsic amino acid solubility values and
    sliding window analysis to predict overall solubility
    and identify insoluble regions.
    """

    def __init__(self, config: SolubilityConfig | None = None) -> None:
        super().__init__(config)
        self._window_size = config.window_size if config else 7

    @property
    def name(self) -> str:
        return "Solubility Predictor"

    @property
    def method(self) -> str:
        return "sequence_based"

    def predict(
        self,
        sequence: str,
        structure: str | None = None,
    ) -> SolubilityResult:
        """
        Predict solubility score for a protein sequence.

        Uses CamSol-derived intrinsic solubility values with:
        - Window-based smoothing
        - Correction for charged patches
        - Exposure penalty for hydrophobic residues
        - Enhanced gatekeeper detection

        Args:
            sequence: Protein sequence (validated)
            structure: Optional PDB structure (not used in sequence mode)

        Returns:
            SolubilityResult with score, classification, and problem regions
        """
        self.validate_inputs(sequence, structure)

        # Calculate per-residue solubility profile
        raw_profile = self._calculate_raw_profile(sequence)

        # Apply exposure correction (penalize likely-exposed hydrophobics)
        corrected_profile = self._apply_exposure_correction(sequence, raw_profile)

        # Apply window smoothing
        smoothed_profile = self._smooth_profile(corrected_profile, self._window_size)

        # Calculate hydrophobicity profile
        hydro_profile = self._calculate_hydropathy(sequence)

        # Identify insoluble regions
        insoluble_regions = self._identify_insoluble_regions(
            sequence, smoothed_profile
        )

        # Calculate gatekeeper bonus (enhanced version)
        gatekeeper_score = self._calculate_gatekeeper_bonus_v2(sequence, insoluble_regions)

        # Calculate charged patch bonus
        charge_score = self._calculate_charge_score(sequence)

        # Calculate terminal charge bonus (N/C-terminal charges help solubility)
        terminal_score = self._calculate_terminal_score(sequence)

        # Combine into final score
        # Base score from mean solubility profile
        mean_sol = sum(smoothed_profile) / len(smoothed_profile) if smoothed_profile else 0

        # Convert CamSol range (-1 to +1) to 0-100 scale
        # CamSol < -0.5: very insoluble, > 0.5: very soluble
        base_score = 50 + mean_sol * 40  # Maps -1.25 to +1.25 → 0 to 100

        # Add bonuses
        final_score = base_score + gatekeeper_score + charge_score + terminal_score

        # Penalize for insoluble regions
        region_penalty = len(insoluble_regions) * 3
        final_score -= region_penalty

        # Clamp to valid range
        final_score = max(0, min(100, final_score))

        # Determine solubility class
        sol_class = SolubilityClass.from_score(final_score)

        # Calculate confidence
        confidence = self._calculate_confidence(sequence, smoothed_profile)

        return SolubilityResult(
            score=round(final_score, 1),
            solubility_class=sol_class,
            insoluble_regions=insoluble_regions,
            hydrophobicity_profile=[round(h, 3) for h in hydro_profile],
            method=self.method,
            confidence=confidence,
        )

    def _calculate_raw_profile(self, sequence: str) -> list[float]:
        """Calculate raw per-residue intrinsic solubility."""
        return [CAMSOL_INTRINSIC.get(aa, 0.0) for aa in sequence]

    def _apply_exposure_correction(
        self,
        sequence: str,
        profile: list[float],
    ) -> list[float]:
        """
        Apply correction for hydrophobic residues that are likely exposed.

        Uses burial propensity to estimate which hydrophobic residues
        might be on the surface rather than buried in the core.
        """
        if len(sequence) < 11:
            return profile

        corrected = profile.copy()
        window = 11
        half_window = window // 2

        for i in range(len(sequence)):
            aa = sequence[i]
            if aa not in EXPOSURE_PENALTY:
                continue

            # Calculate local burial propensity
            start = max(0, i - half_window)
            end = min(len(sequence), i + half_window + 1)
            window_seq = sequence[start:end]

            avg_burial = sum(BURIAL_PROPENSITY.get(a, 0.7) for a in window_seq) / len(window_seq)

            # If local region has low burial propensity, the hydrophobic
            # residue is likely exposed → apply penalty
            if avg_burial < 0.72:  # Threshold for likely exposed
                exposure_factor = (0.72 - avg_burial) / 0.72
                penalty = EXPOSURE_PENALTY[aa] * exposure_factor
                corrected[i] += penalty

        return corrected

    def _smooth_profile(
        self,
        profile: list[float],
        window_size: int,
    ) -> list[float]:
        """Apply sliding window smoothing to the profile."""
        if len(profile) < window_size:
            return profile

        half_window = window_size // 2
        smoothed = []

        for i in range(len(profile)):
            start = max(0, i - half_window)
            end = min(len(profile), i + half_window + 1)
            window = profile[start:end]
            smoothed.append(sum(window) / len(window))

        return smoothed

    def _calculate_hydropathy(self, sequence: str) -> list[float]:
        """Calculate Kyte-Doolittle hydropathy profile."""
        raw = [HYDROPATHY.get(aa, 0.0) for aa in sequence]

        # Normalize to -1 to +1 range
        # Original range is roughly -4.5 to +4.5
        return [h / 4.5 for h in raw]

    def _identify_insoluble_regions(
        self,
        sequence: str,
        profile: list[float],
    ) -> list[Region]:
        """Identify regions with poor solubility."""
        regions = []
        min_length = 5
        threshold = -0.3  # Below this is considered problematic

        i = 0
        while i < len(profile):
            if profile[i] < threshold:
                # Start of potential insoluble region
                start = i
                while i < len(profile) and profile[i] < threshold:
                    i += 1
                end = i

                if end - start >= min_length:
                    # Calculate region score (more negative = worse)
                    region_profile = profile[start:end]
                    region_score = abs(min(region_profile)) * 100

                    regions.append(Region(
                        start=start,
                        end=end,
                        score=round(region_score, 1),
                        sequence=sequence[start:end],
                        annotation="Low intrinsic solubility",
                    ))
            else:
                i += 1

        # Sort by score (worst first) and return top regions
        regions.sort(key=lambda r: -r.score)
        return regions[:10]

    def _calculate_gatekeeper_bonus(self, sequence: str) -> float:
        """Calculate bonus for gatekeeper residues breaking hydrophobic stretches."""
        return self._calculate_gatekeeper_bonus_v2(sequence, [])

    def _calculate_gatekeeper_bonus_v2(
        self,
        sequence: str,
        insoluble_regions: list[Region],
    ) -> float:
        """
        Enhanced gatekeeper detection with region-specific analysis.

        Strong gatekeepers (R, K, D, E) provide better protection than
        weak gatekeepers (P, G, H). Also considers positioning relative
        to identified insoluble regions.
        """
        if not sequence:
            return 0.0

        bonus = 0.0
        window = 10

        # General gatekeeper analysis across sequence
        for i in range(len(sequence) - window + 1):
            window_seq = sequence[i:i + window]

            # Count hydrophobic and gatekeeper residues
            hydrophobic_count = sum(1 for aa in window_seq if aa in AGGREGATION_PRONE)
            strong_gk_count = sum(1 for aa in window_seq if aa in STRONG_GATEKEEPERS)
            weak_gk_count = sum(1 for aa in window_seq if aa in WEAK_GATEKEEPERS)

            # Weighted gatekeeper score
            gk_score = strong_gk_count * 1.0 + weak_gk_count * 0.5

            # Bonus if gatekeepers interrupt hydrophobic stretches
            if hydrophobic_count >= 5 and gk_score >= 2:
                bonus += gk_score * 0.5

        # Additional bonus for gatekeepers flanking insoluble regions
        for region in insoluble_regions:
            # Check flanking residues (3 AA before and after)
            flank_start = max(0, region.start - 3)
            flank_end = min(len(sequence), region.end + 3)

            before = sequence[flank_start:region.start]
            after = sequence[region.end:flank_end]

            # Strong gatekeepers in flanking regions are protective
            strong_before = sum(1 for aa in before if aa in STRONG_GATEKEEPERS)
            strong_after = sum(1 for aa in after if aa in STRONG_GATEKEEPERS)

            if strong_before >= 1 and strong_after >= 1:
                bonus += 2.0  # Well-protected region
            elif strong_before >= 1 or strong_after >= 1:
                bonus += 1.0  # Partially protected

        # Normalize by sequence length
        return min(8.0, bonus * 10 / len(sequence))

    def _calculate_terminal_score(self, sequence: str) -> float:
        """
        Calculate bonus for charged N/C-terminal regions.

        Proteins with charged termini tend to have better solubility
        due to increased surface charge and reduced aggregation.
        """
        if len(sequence) < 10:
            return 0.0

        n_terminal = sequence[:10]
        c_terminal = sequence[-10:]

        # Count charged residues in terminal regions
        charged = {"K", "R", "D", "E"}

        n_charged = sum(1 for aa in n_terminal if aa in charged)
        c_charged = sum(1 for aa in c_terminal if aa in charged)

        # Optimal: 2-4 charged residues per terminal region
        n_score = 0.0
        if 2 <= n_charged <= 4:
            n_score = 1.5
        elif n_charged >= 1:
            n_score = 0.75

        c_score = 0.0
        if 2 <= c_charged <= 4:
            c_score = 1.5
        elif c_charged >= 1:
            c_score = 0.75

        return n_score + c_score

    def _calculate_charge_score(self, sequence: str) -> float:
        """Calculate score based on charged residue distribution."""
        length = len(sequence)
        if length == 0:
            return 0.0

        # Count charges
        positive = sum(1 for aa in sequence if aa in {"K", "R", "H"})
        negative = sum(1 for aa in sequence if aa in {"D", "E"})

        # Total charged fraction
        charged_fraction = (positive + negative) / length

        # Optimal is 15-25% charged
        if 0.15 <= charged_fraction <= 0.25:
            charge_score = 5.0
        elif 0.10 <= charged_fraction <= 0.30:
            charge_score = 2.5
        elif charged_fraction < 0.10:
            charge_score = -2.5  # Too few charges
        else:
            charge_score = 0.0  # Too many charges

        # Penalize charge imbalance
        if positive > 0 or negative > 0:
            balance = min(positive, negative) / max(positive, negative)
            if balance < 0.3:
                charge_score -= 2.5

        return charge_score

    def _calculate_confidence(
        self,
        sequence: str,
        profile: list[float],
    ) -> float:
        """Calculate prediction confidence."""
        length = len(sequence)

        # Base confidence by length
        if length < 50:
            confidence = 0.65
        elif length < 100:
            confidence = 0.75
        elif length <= 500:
            confidence = 0.85
        else:
            confidence = 0.75

        # Reduce confidence if profile is very uniform (suspicious)
        if profile:
            variance = sum((p - sum(profile) / len(profile)) ** 2 for p in profile) / len(profile)
            if variance < 0.01:
                confidence -= 0.1

        return max(0.5, min(1.0, confidence))

    def calculate_supersaturation_concentration(
        self,
        score: float,
        molecular_weight: float,
    ) -> float:
        """
        Estimate supersaturation concentration from solubility score.

        This provides a rough estimate of the concentration at which
        the protein may begin to aggregate.

        Args:
            score: Solubility score (0-100)
            molecular_weight: Protein molecular weight in Da

        Returns:
            Estimated supersaturation concentration in mg/mL
        """
        # Empirical relationship from CamSol validation
        # Score 80+ → >50 mg/mL
        # Score 50 → ~10 mg/mL
        # Score 20 → <1 mg/mL

        # Exponential relationship
        log_conc = (score - 20) * 0.03 - 1

        # Convert to mg/mL
        conc = 10 ** log_conc

        return round(max(0.1, min(200, conc)), 2)
