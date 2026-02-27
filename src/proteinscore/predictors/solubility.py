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

# CamSol dipeptide corrections (Sormanni et al., 2015)
# Context-dependent adjustments to intrinsic solubility
CAMSOL_DIPEPTIDE_CORRECTIONS = {
    # Strongly solubilizing pairs (charged interactions)
    "KE": 0.12, "EK": 0.12, "KD": 0.10, "DK": 0.10,
    "RE": 0.14, "ER": 0.14, "RD": 0.12, "DR": 0.12,
    "RK": 0.08, "KR": 0.08, "DE": 0.06, "ED": 0.06,
    # Proline-containing (structure breakers, help solubility)
    "PP": 0.10, "PK": 0.08, "KP": 0.08, "PR": 0.09, "RP": 0.09,
    "PE": 0.07, "EP": 0.07, "PD": 0.06, "DP": 0.06,
    "PG": 0.05, "GP": 0.05, "PS": 0.04, "SP": 0.04,
    # Strongly insolubilizing pairs (hydrophobic clusters)
    "II": -0.18, "IL": -0.16, "LI": -0.16, "LL": -0.14,
    "IV": -0.14, "VI": -0.14, "VV": -0.12, "VL": -0.13,
    "LV": -0.13, "IF": -0.17, "FI": -0.17, "FF": -0.20,
    "LF": -0.15, "FL": -0.15, "VF": -0.14, "FV": -0.14,
    "IW": -0.16, "WI": -0.16, "LW": -0.14, "WL": -0.14,
    "FW": -0.18, "WF": -0.18, "WW": -0.22, "YY": -0.12,
    "YF": -0.14, "FY": -0.14, "YW": -0.15, "WY": -0.15,
    # Cysteine pairs (aggregation-prone)
    "CC": -0.15, "CI": -0.10, "IC": -0.10, "CL": -0.08,
    "LC": -0.08, "CF": -0.12, "FC": -0.12,
    # Methionine oxidation-prone
    "MM": -0.10, "MW": -0.08, "WM": -0.08,
    # Asparagine deamidation-prone contexts
    "NG": -0.08, "NS": -0.06, "NT": -0.05,
    # Moderately solubilizing
    "KK": 0.04, "RR": 0.05, "EE": 0.03, "DD": 0.02,
    "KS": 0.03, "SK": 0.03, "RS": 0.04, "SR": 0.04,
    "ES": 0.02, "SE": 0.02, "DS": 0.02, "SD": 0.02,
    # Neutral/mild effects
    "AA": 0.01, "SS": 0.02, "TT": 0.01, "GG": -0.02,
    "AG": 0.00, "GA": 0.00, "AS": 0.01, "SA": 0.01,
    "AT": 0.01, "TA": 0.01, "ST": 0.01, "TS": 0.01,
}

# Aromatic clustering patterns (strongly insolubilizing)
AROMATIC_RESIDUES = {"F", "Y", "W"}

# Charge clustering threshold (consecutive charges)
MAX_CHARGE_CLUSTER = 4  # More than this is problematic

# N-glycosylation motif (improves solubility when glycosylated)
NGLYC_PATTERN = "N[^P][ST]"  # NxS or NxT where x != P

# Prion/aggregation-prone polar residues (Q, N enrichment is problematic)
PRION_RESIDUES = {"Q", "N"}
QN_HIGH_THRESHOLD = 0.25  # >25% Q+N is highly aggregation-prone
QN_MODERATE_THRESHOLD = 0.15  # >15% Q+N is moderately problematic

# Net charge thresholds (positive net charge helps solubility)
NET_CHARGE_BENEFICIAL = 0.02  # >2% net positive charge per residue
NET_CHARGE_NEUTRAL = -0.02  # Between -2% and +2%

# Hydrophobicity thresholds (calibrated for globular proteins)
# Note: Many soluble proteins have ~40% hydrophobic content but buried in core
HYDRO_HIGH_THRESHOLD = 0.45  # >45% hydrophobic is problematic
HYDRO_MODERATE_THRESHOLD = 0.40  # >40% is moderately problematic

# Length-based complexity thresholds (from ML analysis)
# Larger proteins have more folding complexity → more aggregation risk
LENGTH_PENALTY_THRESHOLD = 100  # Proteins >100 aa get length penalty
LENGTH_SEVERE_THRESHOLD = 200   # Proteins >200 aa get stronger penalty

# Amyloidogenic motif patterns (from AmyLoad database)
AMYLOID_MOTIFS = {
    "KLVFF",   # Abeta core (strongest amyloid motif)
    "LVFFA",   # Abeta extension
    "VFFAE",   # Abeta C-term core
    "GAIIG",   # Abeta aggregation hotspot
    "MVGGVV",  # Abeta C-terminal (very strong)
    "NFGAIL",  # hIAPP core
    "GVATVA",  # Alpha-synuclein
    "VEALYL",  # Insulin B
    "STVIIE",  # Tau PHF6
    "VQIVYK",  # Tau PHF6*
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

    def _detect_signal_peptide(self, sequence: str) -> int:
        """
        Detect potential signal peptide and return its length.

        Signal peptides are N-terminal sequences that direct protein
        secretion and are cleaved during maturation. They typically:
        - Start with Met
        - Have a positively charged N-region (1-5 residues)
        - Have a hydrophobic H-region (7-15 residues)
        - End with a cleavage site (small residues like A, G, S)

        Args:
            sequence: Protein sequence

        Returns:
            Estimated signal peptide length (0 if none detected)
        """
        if len(sequence) < 20:
            return 0

        # Check N-terminal hydrophobicity (first 25 residues)
        hydrophobic = {"I", "L", "V", "F", "W", "M", "A"}
        n_term = sequence[:25]
        hydro_count = sum(1 for aa in n_term if aa in hydrophobic)
        hydro_frac = hydro_count / 25

        # Signal peptides typically have >50% hydrophobic in N-term
        if hydro_frac < 0.50:
            return 0

        # Look for cleavage site (small residues after hydrophobic stretch)
        cleavage_residues = {"A", "G", "S", "T", "C"}

        # Search for pattern: hydrophobic stretch followed by small residue
        for i in range(15, min(30, len(sequence) - 1)):
            if sequence[i] in cleavage_residues:
                # Check if preceding region is hydrophobic
                window = sequence[i - 10:i]
                window_hydro = sum(1 for aa in window if aa in hydrophobic)
                if window_hydro >= 6:
                    return i + 1

        # Default: if very hydrophobic N-term, assume ~20 residue signal
        if hydro_frac > 0.60:
            return 20

        return 0

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

        # Detect and optionally skip signal peptide
        # Signal peptides are hydrophobic but don't affect mature protein solubility
        signal_length = self._detect_signal_peptide(sequence)
        if signal_length > 0:
            # Use mature sequence for analysis
            mature_sequence = sequence[signal_length:]
            if len(mature_sequence) < 20:
                # Signal too long, use original
                mature_sequence = sequence
                signal_length = 0
        else:
            mature_sequence = sequence

        # Calculate per-residue solubility profile (on mature sequence)
        raw_profile = self._calculate_raw_profile(mature_sequence)

        # Apply dipeptide context corrections (CamSol enhancement)
        dipeptide_profile = self._apply_dipeptide_corrections(mature_sequence, raw_profile)

        # Apply exposure correction (penalize likely-exposed hydrophobics)
        corrected_profile = self._apply_exposure_correction(mature_sequence, dipeptide_profile)

        # Apply window smoothing (adaptive window size)
        adaptive_window = self._get_adaptive_window_size(len(mature_sequence))
        smoothed_profile = self._smooth_profile(corrected_profile, adaptive_window)

        # Calculate hydrophobicity profile (on original sequence for reporting)
        hydro_profile = self._calculate_hydropathy(sequence)

        # Identify insoluble regions
        insoluble_regions = self._identify_insoluble_regions(
            mature_sequence, smoothed_profile
        )

        # Calculate gatekeeper bonus (enhanced version)
        gatekeeper_score = self._calculate_gatekeeper_bonus_v2(mature_sequence, insoluble_regions)

        # Calculate charged patch bonus
        charge_score = self._calculate_charge_score(mature_sequence)

        # Calculate terminal charge bonus (N/C-terminal charges help solubility)
        terminal_score = self._calculate_terminal_score(mature_sequence)

        # Calculate aromatic clustering penalty
        aromatic_penalty = self._calculate_aromatic_clustering_penalty(mature_sequence)

        # Calculate charge clustering penalty
        charge_cluster_penalty = self._calculate_charge_clustering_penalty(mature_sequence)

        # Calculate proline-rich region bonus
        proline_bonus = self._calculate_proline_bonus(mature_sequence)

        # Calculate potential N-glycosylation bonus
        nglyc_bonus = self._calculate_nglyc_bonus(mature_sequence)

        # Calculate Q+N enrichment penalty (prion-like sequences)
        qn_penalty = self._calculate_qn_penalty(mature_sequence)

        # Calculate net charge bonus/penalty
        net_charge_score = self._calculate_net_charge_score(mature_sequence)

        # Calculate overall hydrophobicity penalty
        hydro_penalty = self._calculate_hydrophobicity_penalty(mature_sequence)

        # Calculate length/complexity penalty (large proteins aggregate more)
        length_penalty = self._calculate_length_penalty(mature_sequence)

        # Calculate amyloid motif penalty (known aggregation hotspots)
        amyloid_penalty = self._calculate_amyloid_motif_penalty(mature_sequence)

        # Combine into final score
        # Base score from mean solubility profile
        mean_sol = sum(smoothed_profile) / len(smoothed_profile) if smoothed_profile else 0

        # Convert CamSol range (-1 to +1) to 0-100 scale
        # CamSol < -0.5: very insoluble, > 0.5: very soluble
        # Increased range for better discrimination
        base_score = 55 + mean_sol * 50

        # Add bonuses and penalties
        final_score = (
            base_score +
            gatekeeper_score +
            charge_score +
            terminal_score +
            proline_bonus +
            nglyc_bonus +
            net_charge_score -      # Can be positive or negative
            aromatic_penalty -
            charge_cluster_penalty -
            qn_penalty -            # Prion-like penalty
            hydro_penalty -         # Overall hydrophobicity penalty
            length_penalty -        # NEW: Large protein complexity penalty
            amyloid_penalty         # NEW: Known amyloid motif penalty
        )

        # Penalize for insoluble regions (weighted by severity)
        region_penalty = 0.0
        for region in insoluble_regions:
            region_penalty += 2.0 + region.score * 0.02
        region_penalty = min(region_penalty, 15.0)  # Cap penalty
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
        """Calculate raw per-residue intrinsic solubility.

        Applies context-dependent corrections for Q and N residues,
        which can be problematic in aggregation-prone contexts despite
        having positive intrinsic solubility values.
        """
        profile = []
        length = len(sequence)

        for i, aa in enumerate(sequence):
            base_value = CAMSOL_INTRINSIC.get(aa, 0.0)

            # Context correction for Q and N
            # In isolation Q/N are soluble, but in clusters they promote aggregation
            if aa in PRION_RESIDUES and length >= 10:
                # Check local Q+N density (5 residue window)
                start = max(0, i - 2)
                end = min(length, i + 3)
                local_window = sequence[start:end]
                local_qn = sum(1 for x in local_window if x in PRION_RESIDUES) / len(local_window)

                # If in Q+N-rich context, reduce solubility value
                if local_qn >= 0.6:
                    base_value = -0.3  # Strongly negative in prion-like context
                elif local_qn >= 0.4:
                    base_value = 0.0   # Neutral
                # Otherwise keep original positive value

            profile.append(base_value)

        return profile

    def _apply_dipeptide_corrections(
        self,
        sequence: str,
        profile: list[float],
    ) -> list[float]:
        """
        Apply CamSol dipeptide context corrections.

        The solubility of a residue depends on its neighbors.
        This captures local sequence context effects.
        """
        if len(sequence) < 2:
            return profile

        corrected = profile.copy()

        for i in range(len(sequence) - 1):
            dipeptide = sequence[i:i + 2]
            correction = CAMSOL_DIPEPTIDE_CORRECTIONS.get(dipeptide, 0.0)

            # Apply correction to both residues in the pair
            corrected[i] += correction * 0.6  # First residue gets 60%
            corrected[i + 1] += correction * 0.4  # Second residue gets 40%

        return corrected

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

    def _get_adaptive_window_size(self, length: int) -> int:
        """
        Get adaptive window size based on sequence length.

        Shorter sequences need smaller windows for local accuracy.
        Longer sequences can use larger windows for smoothing.
        """
        if length < 30:
            return 5
        elif length < 100:
            return 7
        elif length < 300:
            return 9
        else:
            return 11

    def _calculate_hydropathy(self, sequence: str) -> list[float]:
        """Calculate Kyte-Doolittle hydropathy profile."""
        raw = [HYDROPATHY.get(aa, 0.0) for aa in sequence]

        # Normalize to -1 to +1 range
        # Original range is roughly -4.5 to +4.5
        return [h / 4.5 for h in raw]

    def _calculate_aromatic_clustering_penalty(self, sequence: str) -> float:
        """
        Calculate penalty for aromatic residue clustering.

        Clusters of aromatic residues (F, Y, W) can promote aggregation
        through pi-pi stacking interactions. However, isolated aromatics
        or aromatics near charged residues are less problematic.
        """
        if len(sequence) < 4:
            return 0.0

        penalty = 0.0

        # Check for adjacent aromatics (strongest aggregation signal)
        adjacent_aromatics = 0
        for i in range(len(sequence) - 1):
            if sequence[i] in AROMATIC_RESIDUES and sequence[i + 1] in AROMATIC_RESIDUES:
                # Check if there's a charged residue nearby (protective)
                start = max(0, i - 2)
                end = min(len(sequence), i + 4)
                nearby = sequence[start:end]
                has_charge = any(aa in {"K", "R", "D", "E"} for aa in nearby)

                if not has_charge:
                    adjacent_aromatics += 1
                    # WW or FF is particularly bad
                    if sequence[i:i + 2] in ("WW", "FF", "WF", "FW"):
                        penalty += 1.5

        # Penalize only when multiple unprotected adjacent aromatics
        if adjacent_aromatics >= 3:
            penalty += 3.0
        elif adjacent_aromatics >= 2:
            penalty += 1.5
        elif adjacent_aromatics >= 1:
            penalty += 0.5

        return min(6.0, penalty)

    def _calculate_charge_clustering_penalty(self, sequence: str) -> float:
        """
        Calculate penalty for excessive charge clustering.

        While charges generally help solubility, clusters of same-charge
        residues can cause aggregation through electrostatic repulsion
        leading to unfolding.
        """
        if len(sequence) < 5:
            return 0.0

        penalty = 0.0
        positive = {"K", "R", "H"}
        negative = {"D", "E"}

        # Check for positive charge clusters
        pos_cluster = 0
        for aa in sequence:
            if aa in positive:
                pos_cluster += 1
                if pos_cluster > MAX_CHARGE_CLUSTER:
                    penalty += 0.5
            else:
                pos_cluster = 0

        # Check for negative charge clusters
        neg_cluster = 0
        for aa in sequence:
            if aa in negative:
                neg_cluster += 1
                if neg_cluster > MAX_CHARGE_CLUSTER:
                    penalty += 0.5
            else:
                neg_cluster = 0

        return min(5.0, penalty)

    def _calculate_proline_bonus(self, sequence: str) -> float:
        """
        Calculate bonus for proline and glycine-rich regions.

        Proline and glycine break secondary structure and prevent aggregation.
        G/P-rich sequences are common in highly soluble proteins like nanobodies.
        PxxP and GxxG motifs are particularly protective.
        """
        if len(sequence) < 10:
            return 0.0

        bonus = 0.0

        # Check for PxxP motifs
        for i in range(len(sequence) - 3):
            if sequence[i] == "P" and sequence[i + 3] == "P":
                bonus += 1.0
            # GxxG motifs also provide flexibility
            if sequence[i] == "G" and sequence[i + 3] == "G":
                bonus += 0.5

        # G+P content bonus (structure breakers)
        gp_fraction = (sequence.count("G") + sequence.count("P")) / len(sequence)

        if gp_fraction >= 0.20:
            bonus += 10.0  # Very high G+P - nanobody-like flexibility
        elif gp_fraction >= 0.15:
            bonus += 6.0   # High G+P content
        elif gp_fraction >= 0.10:
            bonus += 3.0   # Good G+P content
        elif gp_fraction >= 0.08:
            bonus += 1.5   # Moderate G+P

        # Additional bonus for G+S rich sequences (nanobody/VHH pattern)
        # These are highly soluble due to serine hydrogen bonding
        gs_fraction = (sequence.count("G") + sequence.count("S")) / len(sequence)
        if gs_fraction >= 0.30:
            bonus += 5.0  # Very high G+S - highly soluble
        elif gs_fraction >= 0.20:
            bonus += 3.0  # High G+S

        return min(15.0, bonus)

    def _calculate_nglyc_bonus(self, sequence: str) -> float:
        """
        Calculate bonus for potential N-glycosylation sites.

        N-glycosylation (NxS/T where x != P) improves solubility
        when the protein is expressed in mammalian cells.
        """
        import re

        if len(sequence) < 4:
            return 0.0

        # Find NxS/T motifs (potential N-glycosylation sites)
        pattern = r"N[^P][ST]"
        matches = re.findall(pattern, sequence)

        # Each site provides small bonus (max 3 sites counted)
        site_count = min(len(matches), 3)
        return site_count * 1.0

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
        to identified insoluble regions and terminal hydrophobic tails.
        """
        if not sequence:
            return 0.0

        bonus = 0.0
        penalty = 0.0
        window = 10

        # Check for unprotected terminal hydrophobic stretches
        # C-terminal hydrophobic tails (like in Abeta42) are particularly bad
        c_term = sequence[-15:] if len(sequence) >= 15 else sequence
        hydrophobic = {"I", "L", "V", "F", "W", "A", "M"}
        c_term_hydro = sum(1 for aa in c_term if aa in hydrophobic)
        c_term_gk = sum(1 for aa in c_term if aa in STRONG_GATEKEEPERS)

        if c_term_hydro >= 10 and c_term_gk < 2:
            # Unprotected C-terminal hydrophobic tail - major problem
            penalty += 4.0

        # N-terminal check
        n_term = sequence[:15] if len(sequence) >= 15 else sequence
        n_term_hydro = sum(1 for aa in n_term if aa in hydrophobic)
        n_term_gk = sum(1 for aa in n_term if aa in STRONG_GATEKEEPERS)

        if n_term_hydro >= 10 and n_term_gk < 2:
            # Unprotected N-terminal hydrophobic stretch
            penalty += 2.0

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
                bonus += gk_score * 0.4

        # Additional bonus for gatekeepers flanking insoluble regions
        for region in insoluble_regions:
            flank_start = max(0, region.start - 3)
            flank_end = min(len(sequence), region.end + 3)

            before = sequence[flank_start:region.start]
            after = sequence[region.end:flank_end]

            strong_before = sum(1 for aa in before if aa in STRONG_GATEKEEPERS)
            strong_after = sum(1 for aa in after if aa in STRONG_GATEKEEPERS)

            if strong_before >= 1 and strong_after >= 1:
                bonus += 1.5
            elif strong_before >= 1 or strong_after >= 1:
                bonus += 0.75

        # Normalize by sequence length
        normalized_bonus = bonus * 10 / len(sequence)

        return max(0.0, min(8.0, normalized_bonus - penalty))

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
        if 2 <= n_charged <= 5:
            n_score = 2.5
        elif n_charged >= 1:
            n_score = 1.0

        c_score = 0.0
        if 2 <= c_charged <= 5:
            c_score = 2.5
        elif c_charged >= 1:
            c_score = 1.0

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

        # Optimal is 15-30% charged
        if 0.15 <= charged_fraction <= 0.30:
            charge_score = 6.0
        elif 0.12 <= charged_fraction <= 0.35:
            charge_score = 3.0
        elif charged_fraction < 0.10:
            charge_score = -2.5  # Very few charges
        elif charged_fraction < 0.12:
            charge_score = -1.0  # Few charges (may be OK if G/P-rich)
        else:
            charge_score = 1.0  # High but still helpful

        # Penalize charge imbalance
        if positive > 0 or negative > 0:
            balance = min(positive, negative) / max(positive, negative)
            if balance < 0.3:
                charge_score -= 2.5

        # Size-dependent adjustment: larger proteins with high charge
        # may still have solubility issues due to folding complexity
        if length > 100 and charged_fraction > 0.24:
            # Reduce bonus for highly charged larger proteins
            # These often have aggregation/expression issues
            charge_score = charge_score * 0.5

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

    def _calculate_qn_penalty(self, sequence: str) -> float:
        """
        Calculate penalty for Q+N enrichment (prion-like sequences).

        High glutamine (Q) and asparagine (N) content is associated with
        prion-like aggregation and amyloid formation. Examples:
        - Sup35 (yeast prion): ~34% Q+N
        - Huntingtin polyQ: expanded Q repeats
        - Abeta42: high N content

        Args:
            sequence: Protein sequence

        Returns:
            Penalty score (0-20, higher = more problematic)
        """
        if len(sequence) < 10:
            return 0.0

        # Calculate Q+N fraction
        qn_count = sum(1 for aa in sequence if aa in PRION_RESIDUES)
        qn_fraction = qn_count / len(sequence)

        penalty = 0.0

        # High Q+N content (>25%) - severe penalty
        if qn_fraction >= QN_HIGH_THRESHOLD:
            penalty = 15.0 + (qn_fraction - QN_HIGH_THRESHOLD) * 50

        # Moderate Q+N content (15-25%) - moderate penalty
        elif qn_fraction >= QN_MODERATE_THRESHOLD:
            penalty = 6.0 + (qn_fraction - QN_MODERATE_THRESHOLD) * 90

        # Check for Q/N runs (consecutive Q or N residues)
        max_run = 0
        current_run = 0
        for aa in sequence:
            if aa in PRION_RESIDUES:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0

        # Penalize long Q/N runs (polyQ/polyN)
        if max_run >= 8:
            penalty += 10.0  # Severe - like polyQ diseases
        elif max_run >= 5:
            penalty += 5.0   # Moderate - still problematic
        elif max_run >= 3:
            penalty += 2.0   # Mild concern

        # Check for Q+N-rich windows (local enrichment)
        # Even if global fraction is low, local clusters are problematic
        window_size = 30
        max_local_qn = 0.0
        for i in range(len(sequence) - window_size + 1):
            window = sequence[i:i + window_size]
            local_qn = sum(1 for aa in window if aa in PRION_RESIDUES) / window_size
            max_local_qn = max(max_local_qn, local_qn)

        # Penalize local Q+N enrichment
        if max_local_qn >= 0.50:
            penalty += 10.0  # Very high local Q+N
        elif max_local_qn >= 0.40:
            penalty += 6.0   # High local Q+N
        elif max_local_qn >= 0.30:
            penalty += 3.0   # Moderate local Q+N

        # Extra penalty for multiple Q+N runs (prion-like spreading)
        qn_run_count = 0
        current = 0
        for aa in sequence:
            if aa in PRION_RESIDUES:
                current += 1
            else:
                if current >= 3:
                    qn_run_count += 1
                current = 0
        if current >= 3:
            qn_run_count += 1

        if qn_run_count >= 10:
            penalty += 8.0  # Many Q+N runs - classic prion
        elif qn_run_count >= 5:
            penalty += 4.0  # Multiple Q+N runs
        elif qn_run_count >= 3:
            penalty += 2.0  # Several Q+N runs

        return min(25.0, penalty)

    def _calculate_net_charge_score(self, sequence: str) -> float:
        """
        Calculate score based on net charge per residue.

        Positive net charge generally improves solubility by:
        - Increasing electrostatic repulsion between molecules
        - Reducing aggregation propensity

        However, many soluble proteins have slightly negative net charge,
        so we only strongly penalize very negative proteins.

        Args:
            sequence: Protein sequence

        Returns:
            Score (-3 to +3, positive = good for solubility)
        """
        if len(sequence) < 5:
            return 0.0

        # Count charges (at pH 7)
        # K, R: +1 (always positive at physiological pH)
        # H: +0.3 (partially protonated at pH 7, pKa ~6)
        # D, E: -1 (always negative at physiological pH)
        positive_charge = sequence.count("K") + sequence.count("R") + sequence.count("H") * 0.3
        negative_charge = sequence.count("D") + sequence.count("E")

        net_charge = positive_charge - negative_charge
        net_charge_per_residue = net_charge / len(sequence)

        # Score based on net charge per residue
        # Net charge is a strong predictor of solubility (ρ=0.61)
        if net_charge_per_residue >= 0.04:
            # Strong positive net charge - significant bonus
            return min(6.0, 3.0 + net_charge_per_residue * 75)

        elif net_charge_per_residue >= 0.02:
            # Moderate positive - good bonus
            return 2.0 + net_charge_per_residue * 50

        elif net_charge_per_residue >= 0.0:
            # Slightly positive or neutral - small bonus
            return 1.0

        elif net_charge_per_residue >= -0.02:
            # Slightly negative - mild penalty
            return -1.0

        else:
            # Strongly negative - significant penalty
            return max(-6.0, net_charge_per_residue * 100)

    def _calculate_hydrophobicity_penalty(self, sequence: str) -> float:
        """
        Calculate penalty for overall high hydrophobicity.

        Proteins with high hydrophobic content are prone to:
        - Aggregation through hydrophobic interactions
        - Poor expression yields
        - Inclusion body formation

        Note: Most globular proteins have ~35-45% hydrophobic residues
        which are buried in the core. We only penalize extreme cases
        or sequences with consecutive hydrophobic stretches.

        Args:
            sequence: Protein sequence

        Returns:
            Penalty score (0-8, higher = more problematic)
        """
        if len(sequence) < 10:
            return 0.0

        # Define hydrophobic residues
        # A and C are only mildly hydrophobic and often surface-exposed
        strongly_hydrophobic = {"I", "L", "V", "F", "W"}

        # Calculate strongly hydrophobic fraction
        strong_hydro_count = sum(1 for aa in sequence if aa in strongly_hydrophobic)
        strong_hydro_fraction = strong_hydro_count / len(sequence)

        penalty = 0.0

        # Only penalize very high strongly hydrophobic content
        if strong_hydro_fraction >= 0.30:
            penalty = 4.0 + (strong_hydro_fraction - 0.30) * 40
        elif strong_hydro_fraction >= 0.25:
            penalty = 1.5 + (strong_hydro_fraction - 0.25) * 50

        # Check for long hydrophobic stretches (more indicative of aggregation)
        max_hydro_run = 0
        current_run = 0
        for aa in sequence:
            if aa in strongly_hydrophobic:
                current_run += 1
                max_hydro_run = max(max_hydro_run, current_run)
            else:
                current_run = 0

        # Long hydrophobic runs are problematic
        if max_hydro_run >= 8:
            penalty += 8.0
        elif max_hydro_run >= 6:
            penalty += 5.0
        elif max_hydro_run >= 5:
            penalty += 3.5
        elif max_hydro_run >= 4:
            penalty += 2.0

        # Check for multiple hydrophobic runs
        run_count = 0
        current = 0
        for aa in sequence:
            if aa in strongly_hydrophobic:
                current += 1
            else:
                if current >= 3:
                    run_count += 1
                current = 0
        if current >= 3:
            run_count += 1

        # Multiple hydrophobic stretches compound the problem
        if run_count >= 4:
            penalty += 3.0
        elif run_count >= 3:
            penalty += 1.5

        return min(12.0, penalty)

    def _calculate_length_penalty(self, sequence: str) -> float:
        """
        Calculate penalty for large proteins based on folding complexity.

        Larger proteins have:
        - More complex folding landscapes → higher aggregation risk
        - More potential aggregation-prone regions
        - Higher expression challenges in E. coli
        - More hydrophobic surface area if misfolded

        This penalty was derived from ML analysis showing that protein length
        negatively correlates with experimental solubility (ρ = -0.41 in benchmark).

        The relationship is non-linear: proteins >150 aa face exponentially
        more folding challenges, especially when hydrophobic content is high.

        Args:
            sequence: Protein sequence

        Returns:
            Penalty score (0-18, higher = more problematic)
        """
        length = len(sequence)

        if length <= LENGTH_PENALTY_THRESHOLD:
            # Small proteins (<100 aa) - minimal complexity penalty
            return 0.0

        penalty = 0.0

        # Base penalty for length >100
        if length > LENGTH_PENALTY_THRESHOLD:
            excess = length - LENGTH_PENALTY_THRESHOLD
            # ~0.07 penalty per residue over 100 (increased from 0.05)
            penalty = excess * 0.07

        # Additional penalty for very large proteins (>200 aa)
        if length > LENGTH_SEVERE_THRESHOLD:
            severe_excess = length - LENGTH_SEVERE_THRESHOLD
            # Additional ~0.04 penalty per residue over 200
            penalty += severe_excess * 0.04

        # Check for hydrophobic content in large proteins
        # Large hydrophobic proteins are particularly problematic
        hydrophobic = {"I", "L", "V", "F", "W"}
        hydro_fraction = sum(1 for aa in sequence if aa in hydrophobic) / length

        if length > LENGTH_PENALTY_THRESHOLD and hydro_fraction > 0.22:
            # Large hydrophobic proteins get extra penalty (lower threshold)
            # Proteins >100 aa with >22% strongly hydrophobic are at risk
            penalty += (hydro_fraction - 0.22) * 50

        # Extra penalty for medium-large proteins (120-180 aa) with high hydrophobicity
        # These often have expression/solubility issues in E. coli
        if 120 <= length <= 180 and hydro_fraction > 0.25:
            penalty += 3.0

        # Cap the penalty
        return min(18.0, penalty)

    def _calculate_amyloid_motif_penalty(self, sequence: str) -> float:
        """
        Calculate penalty for known amyloidogenic sequence motifs.

        These motifs are derived from the AmyLoad database and represent
        experimentally validated aggregation-prone sequences. Their presence
        strongly indicates aggregation risk.

        Key motifs:
        - KLVFF: Abeta core (Alzheimer's)
        - NFGAIL: hIAPP core (Type 2 diabetes)
        - GVATVA: Alpha-synuclein (Parkinson's)
        - VQIVYK: Tau PHF6 (Alzheimer's)

        Args:
            sequence: Protein sequence

        Returns:
            Penalty score (0-25, higher = more problematic)
        """
        if len(sequence) < 5:
            return 0.0

        penalty = 0.0
        motifs_found = []

        # Check for exact amyloid motifs
        for motif in AMYLOID_MOTIFS:
            if motif in sequence:
                motifs_found.append(motif)
                # Strong penalty for each motif
                if motif in ("KLVFF", "MVGGVV", "NFGAIL"):
                    penalty += 12.0  # Very strong amyloid formers
                else:
                    penalty += 8.0   # Strong amyloid formers

        # Check for C-terminal hydrophobic stretch (like Abeta42)
        # This is a major aggregation driver
        if len(sequence) >= 10:
            c_term = sequence[-10:]
            hydrophobic = {"I", "L", "V", "F", "W", "A", "M"}
            c_term_hydro = sum(1 for aa in c_term if aa in hydrophobic)

            if c_term_hydro >= 7:
                # Highly hydrophobic C-terminus (like Abeta42: ...GLMVGGVVIA)
                penalty += 8.0
            elif c_term_hydro >= 5:
                # Moderately hydrophobic C-terminus
                penalty += 4.0

        # Check for beta-sheet promoting sequences (VLVL, IVIV patterns)
        beta_patterns = ["VLVL", "LVLV", "IVIV", "VIVI", "LILI", "ILIL"]
        for pattern in beta_patterns:
            if pattern in sequence:
                penalty += 5.0
                break  # Only count once

        # Multiple motifs compound the problem
        if len(motifs_found) >= 2:
            penalty += 5.0  # Synergistic effect

        return min(25.0, penalty)
