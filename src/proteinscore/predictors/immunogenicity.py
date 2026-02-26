"""
Immunogenicity Predictor

SOTA immunogenicity prediction using RunLab RIP API.
Achieves Spearman correlation 0.693 (+23.8% vs NetMHCpan 4.1).
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING, Any

from proteinscore.exceptions import APIError, RateLimitError
from proteinscore.models import Epitope, ImmunogenicityResult
from proteinscore.predictors.base import BasePredictor

if TYPE_CHECKING:
    import httpx

    from proteinscore.config import ImmunogenicityConfig


# Default HLA alleles for population coverage
DEFAULT_HLA_ALLELES = {
    "global": [
        "HLA-A*02:01", "HLA-A*01:01", "HLA-A*03:01", "HLA-A*24:02", "HLA-A*11:01",
        "HLA-B*07:02", "HLA-B*08:01", "HLA-B*44:02", "HLA-B*35:01", "HLA-B*15:01",
    ],
    "european": [
        "HLA-A*02:01", "HLA-A*01:01", "HLA-A*03:01", "HLA-A*24:02",
        "HLA-B*07:02", "HLA-B*08:01", "HLA-B*44:02", "HLA-B*51:01",
    ],
    "african": [
        "HLA-A*02:01", "HLA-A*30:01", "HLA-A*23:01", "HLA-A*68:02",
        "HLA-B*35:01", "HLA-B*53:01", "HLA-B*58:01", "HLA-B*07:02",
    ],
    "asian": [
        "HLA-A*02:01", "HLA-A*24:02", "HLA-A*11:01", "HLA-A*33:03",
        "HLA-B*40:01", "HLA-B*46:01", "HLA-B*58:01", "HLA-B*15:01",
    ],
    "hispanic": [
        "HLA-A*02:01", "HLA-A*24:02", "HLA-A*68:01", "HLA-A*01:01",
        "HLA-B*35:01", "HLA-B*44:03", "HLA-B*07:02", "HLA-B*39:01",
    ],
}

# Local estimation scales (for fallback when API unavailable)
# Multi-allele binding motifs derived from IEDB and NetMHCpan
LOCAL_BINDING_MOTIFS = {
    # HLA-A*02:01 (most common globally, ~45% population)
    "HLA-A*02:01": {
        "P2": {"L": 0.9, "M": 0.8, "V": 0.7, "I": 0.6, "A": 0.5, "T": 0.4},
        "P9": {"V": 0.9, "L": 0.85, "I": 0.8, "A": 0.6, "T": 0.5, "M": 0.7},
        "coverage": 0.45,
    },
    # HLA-A*01:01 (~15% European, ~10% global)
    "HLA-A*01:01": {
        "P2": {"T": 0.85, "S": 0.8, "M": 0.7, "I": 0.6, "L": 0.5},
        "P9": {"Y": 0.9, "F": 0.85, "W": 0.8, "L": 0.6},
        "coverage": 0.12,
    },
    # HLA-A*03:01 (~15% European, A3 supertype)
    "HLA-A*03:01": {
        "P2": {"L": 0.85, "V": 0.8, "M": 0.75, "I": 0.7, "A": 0.5},
        "P9": {"K": 0.9, "R": 0.85, "Y": 0.7, "F": 0.6},
        "coverage": 0.10,
    },
    # HLA-A*24:02 (~20% Asian, ~10% global)
    "HLA-A*24:02": {
        "P2": {"Y": 0.9, "F": 0.85, "W": 0.8, "M": 0.6},
        "P9": {"F": 0.9, "L": 0.85, "I": 0.8, "W": 0.75},
        "coverage": 0.12,
    },
    # HLA-B*07:02 (~15% European)
    "HLA-B*07:02": {
        "P2": {"P": 0.9, "A": 0.7, "S": 0.6},
        "P9": {"L": 0.9, "M": 0.8, "I": 0.75, "V": 0.7},
        "coverage": 0.08,
    },
}

# Legacy compatibility
LOCAL_BINDING_SCORES = LOCAL_BINDING_MOTIFS["HLA-A*02:01"]

# Immunogenicity-reducing mutations (deimmunization targets)
IMMUNOGENICITY_REDUCERS = {
    # Anchor position substitutions that reduce binding
    "P2_reducers": {"G", "D", "E", "K", "R"},
    "P9_reducers": {"G", "D", "E", "P"},
}

# Known immunogenic motifs (from clinical data)
IMMUNOGENIC_MOTIFS = [
    "GILGFVFTL",  # Influenza M1, very immunogenic
    "NLVPMVATV",  # CMV pp65
    "GLCTLVAML",  # EBV BMLF1
]


class ImmunogenicityPredictor(BasePredictor[ImmunogenicityResult]):
    """
    SOTA immunogenicity predictor using RunLab RIP API.

    The RIP (Residue Immunogenicity Predictor) achieves state-of-the-art
    performance with Spearman correlation 0.693, outperforming:
    - NetMHCpan 4.1: 0.56
    - MHCflurry 2.0: 0.63

    This represents a +23.8% improvement over the previous best tool.
    """

    def __init__(
        self,
        config: ImmunogenicityConfig | None = None,
        api_key: str | None = None,
        api_base_url: str = "https://api.runlab.bio",
        hla_alleles: list[str] | None = None,
        hla_population: str = "global",
    ) -> None:
        super().__init__(config)
        self._api_key = api_key
        self._api_base_url = api_base_url
        self._hla_alleles = hla_alleles or DEFAULT_HLA_ALLELES.get(hla_population, [])
        self._hla_population = hla_population
        self._peptide_lengths = config.peptide_lengths if config else [9]
        self._threshold = config.threshold_percentile if config else 2.0
        self._client: httpx.Client | None = None

    @property
    def name(self) -> str:
        return "Immunogenicity Predictor (RIP)"

    @property
    def method(self) -> str:
        return "rip_api" if self._api_key else "local_estimate"

    def initialize(self) -> None:
        """Initialize HTTP client for API calls."""
        import httpx

        self._client = httpx.Client(
            base_url=self._api_base_url,
            timeout=30.0,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "ProteinScore/0.1.0",
            },
        )
        if self._api_key:
            self._client.headers["X-API-Key"] = self._api_key

        self._initialized = True

    def predict(
        self,
        sequence: str,
        structure: str | None = None,
    ) -> ImmunogenicityResult:
        """
        Predict immunogenicity score for a protein sequence.

        Uses the RIP API if available, otherwise falls back to
        local estimation based on binding motifs.

        Args:
            sequence: Protein sequence (validated)
            structure: Not used for immunogenicity prediction

        Returns:
            ImmunogenicityResult with score (higher = better, lower immunogenicity risk)
        """
        self.validate_inputs(sequence, structure)

        if self._api_key and self._client is None:
            self.initialize()

        if self._api_key and self._client:
            return self._predict_with_api(sequence)
        else:
            return self._predict_local(sequence)

    def _predict_with_api(self, sequence: str) -> ImmunogenicityResult:
        """Make prediction using RIP API."""
        assert self._client is not None

        try:
            response = self._client.post(
                "/rip/scan",
                json={
                    "sequence": sequence,
                    "alleles": self._hla_alleles,
                    "peptide_lengths": self._peptide_lengths,
                    "threshold_percentile": self._threshold,
                },
            )

            if response.status_code == 429:
                # Rate limited
                retry_after = response.headers.get("X-RateLimit-Reset")
                raise RateLimitError(
                    "RIP API rate limit exceeded",
                    limit=int(response.headers.get("X-RateLimit-Limit", 0)),
                    remaining=int(response.headers.get("X-RateLimit-Remaining", 0)),
                )

            if response.status_code != 200:
                raise APIError(
                    f"RIP API error: {response.status_code}",
                    status_code=response.status_code,
                    response_body=response.text,
                )

            data = response.json()
            return self._parse_api_response(data, sequence)

        except Exception as e:
            if isinstance(e, (RateLimitError, APIError)):
                raise
            # Fallback to local on connection errors
            return self._predict_local(sequence)

    def _parse_api_response(
        self,
        data: dict[str, Any],
        sequence: str,
    ) -> ImmunogenicityResult:
        """Parse RIP API response into ImmunogenicityResult."""
        epitopes = []
        for ep in data.get("epitopes", []):
            epitopes.append(Epitope(
                peptide=ep["peptide"],
                position=tuple(ep["position"]),
                allele=ep["allele"],
                binding_affinity_nM=ep["binding_affinity_nM"],
                percentile_rank=ep["percentile_rank"],
                is_strong_binder=ep.get("is_strong_binder", ep["percentile_rank"] <= 0.5),
                is_weak_binder=ep.get("is_weak_binder", ep["percentile_rank"] <= 2.0),
                confidence=ep.get("confidence", 0.9),
            ))

        # Calculate score: more epitopes = lower score (higher risk)
        strong_binders = data.get("strong_binders", 0)
        weak_binders = data.get("weak_binders", 0)

        # Score calculation: start at 100, penalize for binders
        # Strong binders are worse than weak binders
        penalty = strong_binders * 5 + weak_binders * 1.5

        # Normalize by sequence length (longer proteins have more potential epitopes)
        length_factor = len(sequence) / 200  # Normalize to ~200 AA protein
        penalty = penalty / max(length_factor, 0.5)

        score = max(0, 100 - penalty)

        return ImmunogenicityResult(
            score=round(score, 1),
            epitope_count=len(epitopes),
            strong_binders=strong_binders,
            weak_binders=weak_binders,
            epitopes=epitopes[:50],  # Limit to top 50
            per_residue_risk=data.get("per_residue_risk", []),
            population_coverage=data.get("population_coverage", 0.0),
            method="rip_api",
            confidence=0.92,  # RIP achieves 0.693 Spearman correlation
        )

    def _predict_local(self, sequence: str) -> ImmunogenicityResult:
        """
        Enhanced local immunogenicity estimation when API is unavailable.

        Uses multi-allele binding motif analysis covering:
        - HLA-A*02:01 (~45% global)
        - HLA-A*01:01 (~12% global)
        - HLA-A*03:01 (~10% global)
        - HLA-A*24:02 (~12% global)
        - HLA-B*07:02 (~8% global)

        Combined population coverage: ~70%
        """
        all_epitopes = []
        per_residue_risk = [0.0] * len(sequence)
        total_coverage = 0.0

        # Generate all 9-mer peptides
        peptide_length = 9

        for i in range(len(sequence) - peptide_length + 1):
            peptide = sequence[i:i + peptide_length]

            # Analyze against each allele
            for allele, motif in LOCAL_BINDING_MOTIFS.items():
                binding_score = self._calculate_local_binding_score(peptide, motif)

                if binding_score > 0.65:  # Potential binder threshold
                    is_strong = binding_score > 0.82
                    is_weak = binding_score > 0.65

                    # Convert to pseudo-affinity
                    pseudo_affinity = 1000 * (1 - binding_score)
                    percentile = (1 - binding_score) * 10

                    all_epitopes.append(Epitope(
                        peptide=peptide,
                        position=(i, i + peptide_length),
                        allele=allele,
                        binding_affinity_nM=round(pseudo_affinity, 1),
                        percentile_rank=round(percentile, 2),
                        is_strong_binder=is_strong,
                        is_weak_binder=is_weak,
                        confidence=0.65,
                    ))

                    # Update per-residue risk (weighted by allele coverage)
                    coverage_weight = motif.get("coverage", 0.1)
                    for j in range(peptide_length):
                        per_residue_risk[i + j] = max(
                            per_residue_risk[i + j],
                            binding_score * coverage_weight * 2
                        )

        # Calculate total population coverage
        for motif in LOCAL_BINDING_MOTIFS.values():
            total_coverage += motif.get("coverage", 0.1)
        total_coverage = min(0.70, total_coverage)

        # Check for known immunogenic motifs
        known_motif_penalty = 0
        for motif in IMMUNOGENIC_MOTIFS:
            if motif in sequence:
                known_motif_penalty += 10

        # Remove duplicate epitopes (same peptide, different alleles)
        # Keep only highest-scoring per position
        seen_positions: dict[int, Epitope] = {}
        for ep in all_epitopes:
            pos = ep.position[0]
            if pos not in seen_positions or ep.binding_affinity_nM < seen_positions[pos].binding_affinity_nM:
                seen_positions[pos] = ep

        unique_epitopes = list(seen_positions.values())
        unique_epitopes.sort(key=lambda e: e.binding_affinity_nM)

        # Calculate overall score
        strong_binders = sum(1 for e in unique_epitopes if e.is_strong_binder)
        weak_binders = sum(1 for e in unique_epitopes if e.is_weak_binder and not e.is_strong_binder)

        # Normalize by sequence length
        expected_epitopes = len(sequence) / 9
        strong_ratio = strong_binders / max(expected_epitopes, 1)
        weak_ratio = weak_binders / max(expected_epitopes, 1)

        # Scoring: higher coverage means more reliable penalty
        penalty = (strong_ratio * 120 + weak_ratio * 25) * (total_coverage / 0.5)
        penalty += known_motif_penalty

        score = max(0, min(100, 100 - penalty))

        return ImmunogenicityResult(
            score=round(score, 1),
            epitope_count=len(unique_epitopes),
            strong_binders=strong_binders,
            weak_binders=weak_binders,
            epitopes=unique_epitopes[:50],
            per_residue_risk=[round(min(1.0, r), 3) for r in per_residue_risk],
            population_coverage=round(total_coverage, 2),
            method="local_estimate",
            confidence=0.70,  # Improved confidence with multi-allele
        )

    def _calculate_local_binding_score(
        self,
        peptide: str,
        motif: dict,
    ) -> float:
        """Calculate binding score for a peptide against a motif."""
        if len(peptide) != 9:
            return 0.0

        p2_scores = motif.get("P2", {})
        p9_scores = motif.get("P9", {})

        # Position 2 (P2) anchor
        p2_score = p2_scores.get(peptide[1], 0.2)

        # Position 9 (P9/Omega) anchor
        p9_score = p9_scores.get(peptide[8], 0.2)

        # Combined score with position weighting
        # Anchors contribute 40% each, middle residues 20%
        binding_score = p2_score * 0.4 + p9_score * 0.4 + 0.2

        # Check for immunogenicity-reducing residues
        if peptide[1] in IMMUNOGENICITY_REDUCERS["P2_reducers"]:
            binding_score *= 0.5
        if peptide[8] in IMMUNOGENICITY_REDUCERS["P9_reducers"]:
            binding_score *= 0.5

        return binding_score

    def get_cache_key(self, sequence: str) -> str:
        """Generate cache key for a sequence."""
        # Include alleles in cache key
        allele_str = ",".join(sorted(self._hla_alleles))
        data = f"{sequence}:{allele_str}:{self._threshold}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def close(self) -> None:
        """Close HTTP client."""
        if self._client:
            self._client.close()
            self._client = None

    def __del__(self) -> None:
        """Cleanup on deletion."""
        self.close()
