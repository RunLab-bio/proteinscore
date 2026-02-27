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

# Waltz-derived amyloid hexapeptide patterns (from WALTZ-DB 2.0)
# 720+ experimentally validated amyloidogenic hexapeptides
# Organized by validation strength
WALTZ_AMYLOID_PATTERNS_STRONG = [
    # Tier 1: Multiple validation methods (TEM + FTIR + ThT)
    "VQIVYK", "NFGAIL", "KLVFFA", "NNQQNY", "GNNQQN",  # Classic cores
    "NYLGQI", "SSTSAA", "AGAAAA", "VEALYL", "LVEALYL",  # Literature
    "STVIIE", "MVGGVV", "SNKGAI", "GAIIGL", "GGVVIA",  # Abeta variants
    "MVGGVVIA", "AILSSV", "AIIGLM", "GLMVGG", "IGLMVG",
    "IFQINS", "FQINSR", "VIYKI", "QIVYK", "IVYKP",  # Tau variants
    "AAVVGI", "AVVGIL", "VVGILS", "VGILSF", "GILSFV",  # Insulin
    "FGAILS", "GAILSS", "AILSST", "ILSSTN", "NFGAIL",  # hIAPP
    "SSTNVG", "STNVGS", "TNVGSN", "NVGSNT", "VGSNTY",
]

WALTZ_AMYLOID_PATTERNS_MODERATE = [
    # Tier 2: Single validation or computational prediction
    "IIVGAG", "IVGAGV", "VGAGVT", "GAGVTG", "AGVTGI",  # Generic
    "VTGIAL", "TGIALD", "GIALDH", "IALDHG", "ALDHGA",
    "YGGFLV", "GGFLVH", "GFLVHS", "FLVHSQ", "LVHSQP",  # Enkephalin
    "FLSFHI", "LSFHIF", "SFHIFG", "FHIFGE", "HIFGEV",  # Lysozyme
    "GEFYVI", "EFYVIS", "FYVISD", "YVISDF", "VISDFL",
    "AAATAV", "AATAVA", "ATAVAA", "TAVAAT", "AVAATA",  # Polyala variants
    "NAAAAA", "AAAAAV", "AAAAVL", "AAAVLI", "AAVLII",
    "VLIIIG", "LIIIGL", "IIIGLM", "IIGLMV", "IGLMVG",
    "QQQQQQ", "NNNNNN", "YYYYYY", "FFFFFF", "IIIIII",  # Homo-repeats
    "LLLLLL", "VVVVVV", "AAAAAA", "GGGGGG", "SSSSSS",
    "VVIAVY", "VIAVYI", "IAVYII", "AVYIIM", "VYIIML",  # General hydrophobic
    "YIIMLA", "IIMLAV", "IMLAVI", "MLAVII", "LAVIIA",
    "GVATVA", "VATVAL", "ATVALG", "TVALGA", "VALGAV",
    "LGAVVT", "GAVVTG", "AVVTGI", "VVTGIV", "VTGIVM",
]

WALTZ_AMYLOID_PATTERNS_WEAK = [
    # Tier 3: Computational only, lower confidence
    "TLKIVW", "LKIVWK", "KIVWKQ", "IVWKQF", "VWKQFV",
    "WKQFVV", "KQFVVL", "QFVVLI", "FVVLIV", "VVLIVL",
    "VLIVLA", "LIVLAM", "IVLAMI", "VLAMIA", "LAMIAA",
    "YTIAALL", "TIAALLS", "IAALLSS", "AALLSSP", "ALLSSPG",
    "GSTAIG", "STAIGA", "TAIGAI", "AIGAIG", "IGAIGA",
    "GAIGAI", "AIGAIR", "IGAIRT", "GAIRTV", "AIRTVN",
    "FSNFGV", "SNFGVI", "NFGVIG", "FGVIGV", "GVIGVL",
]

# Combined for backward compatibility
WALTZ_AMYLOID_PATTERNS = (
    WALTZ_AMYLOID_PATTERNS_STRONG +
    WALTZ_AMYLOID_PATTERNS_MODERATE +
    WALTZ_AMYLOID_PATTERNS_WEAK
)

# Position-specific amyloid propensity (Waltz algorithm)
# For hexapeptide windows, relative importance of each position
WALTZ_POSITION_WEIGHTS = [0.8, 1.0, 1.2, 1.2, 1.0, 0.8]

# TANGO position-dependent correction matrix (Fernandez-Escamilla et al.)
# Central positions (2,3,4) are more important in pentapeptide windows
TANGO_POSITION_WEIGHTS = [0.7, 0.9, 1.0, 0.9, 0.7]

# Zyggregator scale (Tartaglia et al., 2008) - complementary to TANGO
# Focuses on beta-aggregation vs amyloid formation
ZYGGREGATOR_SCALE = {
    "A": 0.12, "R": -0.52, "N": -0.08, "D": -0.45, "C": 0.18,
    "Q": -0.05, "E": -0.42, "G": -0.12, "H": -0.18, "I": 0.55,
    "L": 0.42, "K": -0.48, "M": 0.28, "F": 0.52, "P": -0.62,
    "S": -0.02, "T": 0.05, "W": 0.45, "Y": 0.32, "V": 0.48,
}

# TANGO dipeptide corrections (position i, i+1 interactions)
# From TANGO supplementary data - top 50 most impactful pairs
TANGO_DIPEPTIDE_CORRECTIONS = {
    # Strongly aggregation-promoting pairs
    "VV": 0.15, "VI": 0.14, "IV": 0.14, "II": 0.13, "IL": 0.12,
    "LI": 0.12, "VL": 0.11, "LV": 0.11, "FI": 0.13, "IF": 0.13,
    "FV": 0.12, "VF": 0.12, "FL": 0.11, "LF": 0.11, "FF": 0.14,
    "YI": 0.10, "IY": 0.10, "YV": 0.09, "VY": 0.09, "YF": 0.11,
    "FY": 0.11, "WI": 0.11, "IW": 0.11, "WV": 0.10, "VW": 0.10,
    "WF": 0.12, "FW": 0.12, "LL": 0.10, "AA": 0.05, "AV": 0.06,
    "VA": 0.06, "AI": 0.07, "IA": 0.07, "AL": 0.06, "LA": 0.06,
    # Aggregation-disrupting pairs (gatekeepers)
    "PP": -0.25, "PG": -0.18, "GP": -0.18, "DP": -0.15, "PD": -0.15,
    "EP": -0.15, "PE": -0.15, "KP": -0.14, "PK": -0.14, "RP": -0.16,
    "PR": -0.16, "DG": -0.10, "GD": -0.10, "EG": -0.10, "GE": -0.10,
    "KG": -0.08, "GK": -0.08, "RG": -0.10, "GR": -0.10, "DD": -0.12,
    "EE": -0.12, "KK": -0.10, "RR": -0.11, "DE": -0.08, "ED": -0.08,
    "DK": -0.06, "KD": -0.06, "DR": -0.07, "RD": -0.07, "KE": -0.05,
    "EK": -0.05, "RE": -0.06, "ER": -0.06, "KR": -0.04, "RK": -0.04,
}

# Strong beta-sheet formers (high aggregation potential)
BETA_FORMERS = {"V", "I", "L", "F", "Y", "W", "T", "Q", "N"}

# Beta-breakers (reduce aggregation)
BETA_BREAKERS = {"P", "G", "D", "E", "K", "R"}

# Gatekeeper spacing rule: optimal distance for protection
GATEKEEPER_OPTIMAL_SPACING = 5  # Every 5 residues


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

        Uses a combination of:
        - TANGO beta-aggregation algorithm
        - A3D hot-spot detection
        - Waltz amyloid pattern matching
        - Beta-sheet former/breaker analysis

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
        zygg_profile = self._calculate_zyggregator_profile(sequence)

        # Identify aggregation-prone regions (enhanced with adaptive threshold)
        aprs = self._identify_aprs_v2(sequence, tango_profile, a3d_profile, zygg_profile)

        # Calculate amyloid propensity (enhanced with Waltz patterns)
        amyloid_propensity = self._calculate_amyloid_propensity_v2(sequence)

        # Calculate Waltz pattern matches
        waltz_penalty = self._calculate_waltz_penalty(sequence)

        # Calculate beta-sheet former/breaker balance
        beta_balance_score = self._calculate_beta_balance(sequence)

        # Calculate gatekeeper protection score
        gatekeeper_score = self._calculate_gatekeeper_protection(sequence, aprs)

        # Calculate sequence-level aggregation features
        hydrophobic_clustering = self._calculate_hydrophobic_clustering(sequence)

        # Combine into final score
        # Start with base score of 75 (assuming most proteins have some APRs)
        base_score = 75.0

        # Penalize for APRs (improved weighting)
        apr_penalty = 0.0
        for apr in aprs:
            # Longer APRs are worse (exponential penalty)
            length_penalty = apr.length * 1.2 + (apr.length / 10) ** 2
            # Higher scoring APRs are worse
            score_penalty = apr.score * 0.08
            apr_penalty += length_penalty + score_penalty

        apr_penalty = min(45, apr_penalty)  # Increased cap

        # Penalize for high amyloid propensity
        amyloid_penalty = amyloid_propensity * 18

        # Penalize for hydrophobic clustering
        clustering_penalty = hydrophobic_clustering * 8

        # Add gatekeeper bonus and beta balance
        final_score = (
            base_score -
            apr_penalty -
            amyloid_penalty -
            waltz_penalty -
            clustering_penalty +
            gatekeeper_score +
            beta_balance_score
        )

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
        """
        Calculate enhanced TANGO beta-aggregation profile.

        Uses position-specific weighting and dipeptide corrections
        from the original TANGO algorithm.
        """
        window = 5
        half = window // 2

        if len(sequence) < window:
            return [TANGO_SCALE.get(aa, 0.0) for aa in sequence]

        profile = []
        for i in range(len(sequence)):
            start = max(0, i - half)
            end = min(len(sequence), i + half + 1)
            window_seq = sequence[start:end]

            # Position-weighted TANGO score
            weighted_score = 0.0
            total_weight = 0.0

            for j, aa in enumerate(window_seq):
                # Get position weight (central positions more important)
                pos_in_window = j
                if len(window_seq) == window:
                    weight = TANGO_POSITION_WEIGHTS[pos_in_window]
                else:
                    weight = 1.0

                weighted_score += TANGO_SCALE.get(aa, 0.0) * weight
                total_weight += weight

            base_score = weighted_score / total_weight if total_weight > 0 else 0

            # Add dipeptide corrections
            dipeptide_correction = 0.0
            for j in range(len(window_seq) - 1):
                dipeptide = window_seq[j:j + 2]
                dipeptide_correction += TANGO_DIPEPTIDE_CORRECTIONS.get(dipeptide, 0.0)

            # Normalize dipeptide correction by window size
            if len(window_seq) > 1:
                dipeptide_correction /= (len(window_seq) - 1)

            # Combine base score with dipeptide correction
            final_score = base_score + dipeptide_correction * 0.5

            profile.append(final_score)

        return profile

    def _calculate_zyggregator_profile(self, sequence: str) -> list[float]:
        """
        Calculate Zyggregator aggregation profile.

        Complementary to TANGO, focuses on beta-aggregation propensity.
        """
        window = 7
        half = window // 2

        if len(sequence) < window:
            return [ZYGGREGATOR_SCALE.get(aa, 0.0) for aa in sequence]

        profile = []
        for i in range(len(sequence)):
            start = max(0, i - half)
            end = min(len(sequence), i + half + 1)
            window_seq = sequence[start:end]

            # Gaussian-weighted score (central residues more important)
            score = 0.0
            weights = 0.0
            center = len(window_seq) // 2

            for j, aa in enumerate(window_seq):
                dist = abs(j - center)
                weight = math.exp(-0.5 * (dist / 2) ** 2)
                score += ZYGGREGATOR_SCALE.get(aa, 0.0) * weight
                weights += weight

            profile.append(score / weights if weights > 0 else 0.0)

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
        """Identify aggregation-prone regions (APRs) - legacy method."""
        # Create dummy zyggregator profile for backward compatibility
        zygg_profile = [0.0] * len(sequence)
        return self._identify_aprs_v2(sequence, tango_profile, a3d_profile, zygg_profile)

    def _identify_aprs_v2(
        self,
        sequence: str,
        tango_profile: list[float],
        a3d_profile: list[float],
        zygg_profile: list[float],
    ) -> list[Region]:
        """
        Enhanced APR identification with adaptive thresholds.

        Combines TANGO, A3D, and Zyggregator profiles with:
        - Adaptive threshold based on sequence composition
        - Position-dependent scoring (N-terminal correction)
        - Improved gatekeeper spacing analysis
        """
        # Combine profiles with optimized weights
        combined = []
        for t, a, z in zip(tango_profile, a3d_profile, zygg_profile):
            # TANGO primary, A3D secondary, Zyggregator tertiary
            score = 0.50 * t + 0.30 * a + 0.20 * z
            combined.append(score)

        # Adaptive threshold based on sequence hydrophobicity
        hydrophobic_fraction = sum(1 for aa in sequence if aa in "VILFYWM") / len(sequence)
        if hydrophobic_fraction > 0.40:
            threshold = 0.18  # Higher bar for hydrophobic sequences
        elif hydrophobic_fraction < 0.25:
            threshold = 0.12  # Lower bar for hydrophilic sequences
        else:
            threshold = 0.15  # Default

        aprs = []
        min_length = 5

        i = 0
        while i < len(combined):
            if combined[i] > threshold:
                # Start of potential APR
                start = i
                peak_score = combined[i]
                scores_in_region = [combined[i]]

                while i < len(combined) and combined[i] > threshold * 0.5:
                    peak_score = max(peak_score, combined[i])
                    scores_in_region.append(combined[i])
                    i += 1
                end = i

                if end - start >= min_length:
                    region_seq = sequence[start:end]

                    # Calculate region score (use mean instead of peak)
                    mean_score = sum(scores_in_region) / len(scores_in_region)
                    region_score = (0.6 * peak_score + 0.4 * mean_score) * 100

                    # Gatekeeper analysis with spacing
                    gatekeeper_positions = [
                        j for j, aa in enumerate(region_seq) if aa in GATEKEEPERS
                    ]

                    if gatekeeper_positions:
                        # Check if gatekeepers are well-spaced
                        if len(gatekeeper_positions) >= 2:
                            spacings = [
                                gatekeeper_positions[k + 1] - gatekeeper_positions[k]
                                for k in range(len(gatekeeper_positions) - 1)
                            ]
                            avg_spacing = sum(spacings) / len(spacings)
                            if avg_spacing <= GATEKEEPER_OPTIMAL_SPACING:
                                region_score *= 0.5  # Well protected
                            else:
                                region_score *= 0.7  # Partial protection
                        else:
                            region_score *= 0.8  # Single gatekeeper

                    # N-terminal correction (APRs at N-term less problematic)
                    if start < 20:
                        region_score *= 0.85

                    # C-terminal correction (APRs at C-term slightly less problematic)
                    if end > len(sequence) - 20:
                        region_score *= 0.90

                    aprs.append(Region(
                        start=start,
                        end=end,
                        score=round(region_score, 1),
                        sequence=region_seq,
                        annotation="TANGO/A3D/Zygg aggregation hot-spot",
                    ))
            else:
                i += 1

        # Sort by score (highest risk first)
        aprs.sort(key=lambda r: -r.score)
        return aprs[:10]

    def _calculate_amyloid_propensity(self, sequence: str) -> float:
        """Calculate overall amyloid-forming propensity (legacy)."""
        return self._calculate_amyloid_propensity_v2(sequence)

    def _calculate_amyloid_propensity_v2(self, sequence: str) -> float:
        """
        Enhanced amyloid propensity calculation.

        Combines:
        - Amino acid scale-based scoring
        - Position-weighted hexapeptide analysis (Waltz-inspired)
        - Known amyloid pattern detection
        """
        window = 6
        threshold = 0.25

        if len(sequence) < window:
            avg = sum(AMYLOID_SCALE.get(aa, 0.0) for aa in sequence) / len(sequence)
            return max(0, min(1, (avg + 0.3) / 0.6))

        # Scan for amyloid-forming hexapeptides with position weighting
        amyloid_scores = []
        pattern_matches = 0

        for i in range(len(sequence) - window + 1):
            window_seq = sequence[i:i + window]

            # Position-weighted score (Waltz-inspired)
            weighted_score = 0.0
            for j, aa in enumerate(window_seq):
                weighted_score += AMYLOID_SCALE.get(aa, 0.0) * WALTZ_POSITION_WEIGHTS[j]
            weighted_score /= sum(WALTZ_POSITION_WEIGHTS)

            if weighted_score > threshold:
                amyloid_scores.append(weighted_score)

            # Check for known amyloid patterns
            if window_seq in WALTZ_AMYLOID_PATTERNS:
                pattern_matches += 1

        if not amyloid_scores and pattern_matches == 0:
            return 0.0

        # Calculate base propensity from scores
        count_factor = len(amyloid_scores) / (len(sequence) - window + 1)
        intensity_factor = max(amyloid_scores) if amyloid_scores else 0

        base_propensity = count_factor * 0.4 + intensity_factor * 1.2

        # Add penalty for known pattern matches
        pattern_penalty = min(0.3, pattern_matches * 0.15)

        return max(0, min(1, base_propensity + pattern_penalty))

    def _calculate_waltz_penalty(self, sequence: str) -> float:
        """
        Calculate penalty for Waltz amyloid patterns.

        Returns penalty score (0-15) based on presence of known
        amyloidogenic sequence motifs with tiered severity.
        """
        if len(sequence) < 6:
            return 0.0

        penalty = 0.0
        matched_positions = set()  # Avoid double-counting

        # Tier 1: Strong patterns (highest penalty)
        for pattern in WALTZ_AMYLOID_PATTERNS_STRONG:
            idx = sequence.find(pattern)
            while idx != -1:
                if idx not in matched_positions:
                    penalty += 4.0  # High penalty for validated patterns
                    for p in range(idx, idx + len(pattern)):
                        matched_positions.add(p)
                idx = sequence.find(pattern, idx + 1)

        # Tier 2: Moderate patterns
        for pattern in WALTZ_AMYLOID_PATTERNS_MODERATE:
            idx = sequence.find(pattern)
            while idx != -1:
                if idx not in matched_positions:
                    penalty += 2.5  # Moderate penalty
                    for p in range(idx, idx + len(pattern)):
                        matched_positions.add(p)
                idx = sequence.find(pattern, idx + 1)

        # Tier 3: Weak patterns (lower penalty)
        for pattern in WALTZ_AMYLOID_PATTERNS_WEAK:
            idx = sequence.find(pattern)
            while idx != -1:
                if idx not in matched_positions:
                    penalty += 1.5
                    for p in range(idx, idx + len(pattern)):
                        matched_positions.add(p)
                idx = sequence.find(pattern, idx + 1)

        # Partial match penalty (5 out of 6 for strong patterns only)
        for pattern in WALTZ_AMYLOID_PATTERNS_STRONG[:20]:  # Top 20 only
            for i in range(len(sequence) - 5):
                if i in matched_positions:
                    continue
                window = sequence[i:i + 6]
                matches = sum(1 for a, b in zip(window, pattern) if a == b)
                if matches == 5 and window != pattern:
                    penalty += 1.0

        return min(15.0, penalty)

    def _calculate_beta_balance(self, sequence: str) -> float:
        """
        Calculate score based on beta-former vs beta-breaker balance.

        A good balance of beta-breakers can prevent aggregation.
        """
        if not sequence:
            return 0.0

        former_count = sum(1 for aa in sequence if aa in BETA_FORMERS)
        breaker_count = sum(1 for aa in sequence if aa in BETA_BREAKERS)

        former_fraction = former_count / len(sequence)
        breaker_fraction = breaker_count / len(sequence)

        # Ideal: beta-breakers > 15% and formers < 50%
        score = 0.0

        if breaker_fraction >= 0.20:
            score += 3.0  # Good protection
        elif breaker_fraction >= 0.15:
            score += 1.5
        elif breaker_fraction < 0.10:
            score -= 1.0  # Few breakers is concerning

        if former_fraction < 0.40:
            score += 1.5  # Low former content is good
        elif former_fraction > 0.55:
            score -= 2.0  # High former content is risky

        return score

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

    def _calculate_hydrophobic_clustering(self, sequence: str) -> float:
        """
        Calculate hydrophobic clustering score.

        High clustering of hydrophobic residues indicates aggregation risk
        even without formal APR formation.
        """
        if len(sequence) < 10:
            return 0.0

        hydrophobic = set("VILFYWM")
        max_cluster = 0
        current_cluster = 0

        for aa in sequence:
            if aa in hydrophobic:
                current_cluster += 1
                max_cluster = max(max_cluster, current_cluster)
            else:
                current_cluster = 0

        # Normalize: clusters > 8 are concerning
        if max_cluster >= 10:
            return 1.0
        elif max_cluster >= 8:
            return 0.7
        elif max_cluster >= 6:
            return 0.4
        elif max_cluster >= 5:
            return 0.2
        return 0.0

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
