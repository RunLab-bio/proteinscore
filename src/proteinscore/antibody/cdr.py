"""
CDR Detection and Parsing for Antibodies

Implements Chothia CDR numbering scheme for antibody variable regions.
Supports both heavy (VH) and light (VL) chain analysis.

References:
    - Chothia C & Lesk AM (1987). Canonical structures for the hypervariable regions
    - Al-Lazikani B et al. (1997). Standard conformations for the canonical structures
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import NamedTuple


# =============================================================================
# Chothia CDR Definitions
# =============================================================================

# CDR positions according to Chothia numbering
# Format: (start, end) - inclusive, 1-indexed

CHOTHIA_CDR_HEAVY = {
    "CDR-H1": (26, 32),   # Can extend to 35 in some cases
    "CDR-H2": (52, 56),   # Core H2
    "CDR-H3": (95, 102),  # Highly variable
}

CHOTHIA_CDR_LIGHT = {
    "CDR-L1": (24, 34),   # Variable length
    "CDR-L2": (50, 56),   # Core L2
    "CDR-L3": (89, 97),   # Moderately variable
}

# Kabat CDR definitions (alternative)
KABAT_CDR_HEAVY = {
    "CDR-H1": (31, 35),
    "CDR-H2": (50, 65),
    "CDR-H3": (95, 102),
}

KABAT_CDR_LIGHT = {
    "CDR-L1": (24, 34),
    "CDR-L2": (50, 56),
    "CDR-L3": (89, 97),
}

# IMGT CDR definitions
IMGT_CDR_HEAVY = {
    "CDR-H1": (27, 38),
    "CDR-H2": (56, 65),
    "CDR-H3": (105, 117),
}

IMGT_CDR_LIGHT = {
    "CDR-L1": (27, 38),
    "CDR-L2": (56, 65),
    "CDR-L3": (105, 117),
}


# =============================================================================
# Framework Conserved Motifs for Chain Detection
# =============================================================================

# Heavy chain conserved motifs
VH_MOTIFS = [
    r"^[EQ]V[QK]L",          # Start of VH (EVQL, QVQL, EVKL)
    r"WV[RK]Q",              # FR2 start
    r"L[ET][VW][WM]",        # FR3 after CDR-H2
    r"[YF]C[AS][RK]",        # Before CDR-H3
    r"WG[QK]G",              # After CDR-H3 (J region)
]

# Light chain conserved motifs (kappa and lambda)
VL_MOTIFS = [
    r"^[DE]I[QV][ML]T",      # Start of VL kappa (DIQMT, DIVMT)
    r"^[SQ][YS][VE][LS]T",   # Start of VL lambda
    r"WY[QR][QK]",           # FR2 start
    r"G[VI]P[SD]RF",         # FR3
    r"[YF][YF]C[QA]",        # Before CDR-L3
    r"FG[QG]GT",             # J region
]


# =============================================================================
# Data Structures
# =============================================================================

class CDRPosition(NamedTuple):
    """Position of a CDR region."""
    name: str
    start: int  # 0-indexed
    end: int    # exclusive
    sequence: str


@dataclass
class CDRRegions:
    """Detected CDR regions for an antibody chain."""
    chain_type: str  # "heavy" or "light"
    sequence: str
    cdrs: list[CDRPosition] = field(default_factory=list)
    framework_regions: list[CDRPosition] = field(default_factory=list)
    numbering_scheme: str = "chothia"

    @property
    def cdr_sequences(self) -> dict[str, str]:
        """Get CDR sequences as dict."""
        return {cdr.name: cdr.sequence for cdr in self.cdrs}

    @property
    def total_cdr_length(self) -> int:
        """Total length of all CDRs."""
        return sum(len(cdr.sequence) for cdr in self.cdrs)

    @property
    def cdr_residues(self) -> str:
        """Concatenated CDR residues."""
        return "".join(cdr.sequence for cdr in self.cdrs)

    def get_cdr(self, name: str) -> CDRPosition | None:
        """Get a specific CDR by name."""
        for cdr in self.cdrs:
            if cdr.name == name:
                return cdr
        return None

    def as_dict(self) -> dict[str, tuple[int | None, int | None]]:
        """
        Get CDR regions as dict of name -> (start, end) for compatibility.

        Returns:
            Dict mapping CDR names to (start, end) tuples (0-indexed, end exclusive)
        """
        result = {}
        for cdr in self.cdrs:
            result[cdr.name] = (cdr.start, cdr.end)
        return result


# =============================================================================
# CDR Detector
# =============================================================================

class CDRDetector:
    """
    Detect CDR regions in antibody variable domain sequences.

    Uses conserved motifs and position-based rules to identify CDRs
    without requiring full antibody numbering.
    """

    def __init__(self, numbering_scheme: str = "chothia"):
        """
        Initialize CDR detector.

        Args:
            numbering_scheme: CDR definition scheme - "chothia", "kabat", or "imgt"
        """
        self.numbering_scheme = numbering_scheme.lower()

        if self.numbering_scheme == "chothia":
            self._cdr_heavy = CHOTHIA_CDR_HEAVY
            self._cdr_light = CHOTHIA_CDR_LIGHT
        elif self.numbering_scheme == "kabat":
            self._cdr_heavy = KABAT_CDR_HEAVY
            self._cdr_light = KABAT_CDR_LIGHT
        elif self.numbering_scheme == "imgt":
            self._cdr_heavy = IMGT_CDR_HEAVY
            self._cdr_light = IMGT_CDR_LIGHT
        else:
            raise ValueError(f"Unknown numbering scheme: {numbering_scheme}")

    def detect_chain_type(self, sequence: str) -> str:
        """
        Detect if sequence is heavy (VH) or light (VL) chain.

        Args:
            sequence: Antibody variable domain sequence

        Returns:
            "heavy" or "light"
        """
        sequence = sequence.upper()

        # Check heavy chain motifs
        vh_score = sum(1 for motif in VH_MOTIFS if re.search(motif, sequence))

        # Check light chain motifs
        vl_score = sum(1 for motif in VL_MOTIFS if re.search(motif, sequence))

        if vh_score > vl_score:
            return "heavy"
        elif vl_score > vh_score:
            return "light"
        else:
            # Use heuristics based on sequence patterns
            # Heavy chains often start with Q/E and have W at position ~36
            if sequence.startswith(("Q", "E")) and len(sequence) > 40:
                if "W" in sequence[30:45]:
                    return "heavy"
            return "light"  # Default to light if uncertain

    def detect_cdrs(
        self,
        sequence: str,
        chain_type: str | None = None
    ) -> CDRRegions:
        """
        Detect CDR regions in an antibody variable domain.

        Args:
            sequence: Variable domain sequence (VH or VL)
            chain_type: "heavy" or "light". Auto-detected if None.

        Returns:
            CDRRegions object with detected CDRs
        """
        sequence = sequence.upper().strip()

        # Detect chain type if not provided
        if chain_type is None:
            chain_type = self.detect_chain_type(sequence)

        # Get CDR definitions for this chain type
        cdr_defs = self._cdr_heavy if chain_type == "heavy" else self._cdr_light

        # Detect CDRs using motif-based alignment
        cdrs = []
        seq_len = len(sequence)

        for cdr_name, (start, end) in cdr_defs.items():
            # Adjust positions for variable sequence lengths
            # Standard VH ~120aa, VL ~110aa
            expected_len = 120 if chain_type == "heavy" else 110
            scale = seq_len / expected_len

            adj_start = max(0, int((start - 1) * scale))
            adj_end = min(seq_len, int(end * scale))

            # Refine using motif detection
            adj_start, adj_end = self._refine_cdr_boundaries(
                sequence, cdr_name, adj_start, adj_end, chain_type
            )

            if adj_end > adj_start:
                cdr_seq = sequence[adj_start:adj_end]
                cdrs.append(CDRPosition(
                    name=cdr_name,
                    start=adj_start,
                    end=adj_end,
                    sequence=cdr_seq
                ))

        # Identify framework regions
        frameworks = self._extract_framework_regions(sequence, cdrs)

        return CDRRegions(
            chain_type=chain_type,
            sequence=sequence,
            cdrs=cdrs,
            framework_regions=frameworks,
            numbering_scheme=self.numbering_scheme,
        )

    def _refine_cdr_boundaries(
        self,
        sequence: str,
        cdr_name: str,
        start: int,
        end: int,
        chain_type: str
    ) -> tuple[int, int]:
        """Refine CDR boundaries using conserved motifs."""
        # CDR-H3 refinement: Look for CAR/CAS before and WGxG after
        if cdr_name == "CDR-H3":
            # Find C[AS]R/K motif before H3
            match = re.search(r"C[AS][RK]", sequence[max(0, start-10):end])
            if match:
                start = max(0, start - 10) + match.end()

            # Find WG[QK]G motif after H3
            match = re.search(r"WG[QK]G", sequence[start:min(len(sequence), end+15)])
            if match:
                end = start + match.start()

        # CDR-L3 refinement: Look for YYC/FYC before
        elif cdr_name == "CDR-L3":
            match = re.search(r"[YF][YF]C", sequence[max(0, start-10):end])
            if match:
                start = max(0, start - 10) + match.end()

            # Find FG[QG]G after
            match = re.search(r"FG[QG]G", sequence[start:min(len(sequence), end+10)])
            if match:
                end = start + match.start()

        # CDR-H1 refinement: Look for CXXXX pattern
        elif cdr_name == "CDR-H1":
            # Typically starts after first C
            match = re.search(r"C[A-Z]{4,}", sequence[:start+15])
            if match:
                start = match.start() + 5

            # Ends before WV/WI
            match = re.search(r"W[VI]", sequence[start:min(len(sequence), start+20)])
            if match:
                end = start + match.start()

        return start, end

    def _extract_framework_regions(
        self,
        sequence: str,
        cdrs: list[CDRPosition]
    ) -> list[CDRPosition]:
        """Extract framework regions between CDRs."""
        frameworks = []
        prev_end = 0

        for i, cdr in enumerate(sorted(cdrs, key=lambda c: c.start)):
            if cdr.start > prev_end:
                fr_name = f"FR{i + 1}"
                frameworks.append(CDRPosition(
                    name=fr_name,
                    start=prev_end,
                    end=cdr.start,
                    sequence=sequence[prev_end:cdr.start]
                ))
            prev_end = cdr.end

        # Final framework
        if prev_end < len(sequence):
            frameworks.append(CDRPosition(
                name=f"FR{len(cdrs) + 1}",
                start=prev_end,
                end=len(sequence),
                sequence=sequence[prev_end:]
            ))

        return frameworks

    def detect_paired(
        self,
        heavy_sequence: str,
        light_sequence: str
    ) -> tuple[CDRRegions, CDRRegions]:
        """
        Detect CDRs for a paired VH/VL antibody.

        Args:
            heavy_sequence: VH sequence
            light_sequence: VL sequence

        Returns:
            Tuple of (VH CDRRegions, VL CDRRegions)
        """
        vh_cdrs = self.detect_cdrs(heavy_sequence, "heavy")
        vl_cdrs = self.detect_cdrs(light_sequence, "light")
        return vh_cdrs, vl_cdrs
