"""
ProteinScore Input Validation

SOTA validation for protein sequences and HLA alleles with detailed error messages.
"""

from __future__ import annotations

import re
from functools import lru_cache

from proteinscore.exceptions import InvalidAlleleError, InvalidSequenceError


# Standard amino acid one-letter codes
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

# Extended amino acid codes (including ambiguous)
EXTENDED_AA = STANDARD_AA | {"B", "J", "O", "U", "X", "Z"}

# HLA allele pattern: HLA-[A/B/C/etc]*[group]:[protein]
HLA_PATTERN = re.compile(
    r"^HLA-[A-Z]+\d?\*\d{2,4}(:\d{2,4}){0,3}[NLSCAQ]?$",
    re.IGNORECASE
)

# Common HLA supertype groups for validation hints
HLA_SUPERTYPES = {
    "A1": ["HLA-A*01:01", "HLA-A*26:01", "HLA-A*32:01"],
    "A2": ["HLA-A*02:01", "HLA-A*02:02", "HLA-A*02:03", "HLA-A*02:06", "HLA-A*68:02"],
    "A3": ["HLA-A*03:01", "HLA-A*11:01", "HLA-A*31:01", "HLA-A*33:01", "HLA-A*68:01"],
    "A24": ["HLA-A*23:01", "HLA-A*24:02"],
    "B7": ["HLA-B*07:02", "HLA-B*35:01", "HLA-B*51:01", "HLA-B*53:01"],
    "B27": ["HLA-B*14:02", "HLA-B*27:05"],
    "B44": ["HLA-B*18:01", "HLA-B*37:01", "HLA-B*44:02", "HLA-B*44:03"],
    "B58": ["HLA-B*57:01", "HLA-B*58:01"],
    "B62": ["HLA-B*15:01", "HLA-B*46:01"],
}


def validate_sequence(
    sequence: str,
    *,
    min_length: int = 1,
    max_length: int = 10000,
    allow_extended: bool = False,
    strip_whitespace: bool = True,
) -> str:
    """
    Validate and normalize a protein sequence.

    Args:
        sequence: Input protein sequence
        min_length: Minimum allowed length
        max_length: Maximum allowed length
        allow_extended: Allow extended amino acid codes (B, J, O, U, X, Z)
        strip_whitespace: Remove whitespace from sequence

    Returns:
        Normalized sequence (uppercase, no whitespace)

    Raises:
        InvalidSequenceError: If sequence is invalid
    """
    if not sequence:
        raise InvalidSequenceError("Sequence cannot be empty")

    # Normalize
    if strip_whitespace:
        sequence = "".join(sequence.split())
    sequence = sequence.upper()

    # Check length
    if len(sequence) < min_length:
        raise InvalidSequenceError(
            f"Sequence too short: {len(sequence)} < {min_length}",
            sequence=sequence,
        )

    if len(sequence) > max_length:
        raise InvalidSequenceError(
            f"Sequence too long: {len(sequence)} > {max_length}",
            sequence=sequence,
        )

    # Check characters
    valid_chars = EXTENDED_AA if allow_extended else STANDARD_AA
    invalid_chars = [c for c in sequence if c not in valid_chars]

    if invalid_chars:
        unique_invalid = sorted(set(invalid_chars))
        raise InvalidSequenceError(
            f"Sequence contains invalid characters: {unique_invalid}",
            sequence=sequence,
            invalid_chars=unique_invalid,
        )

    return sequence


def validate_peptide(
    peptide: str,
    *,
    min_length: int = 8,
    max_length: int = 15,
) -> str:
    """
    Validate a peptide for MHC binding prediction.

    Args:
        peptide: Input peptide sequence
        min_length: Minimum length (default: 8 for MHC-I)
        max_length: Maximum length (default: 15 for MHC-I/II)

    Returns:
        Normalized peptide sequence

    Raises:
        InvalidSequenceError: If peptide is invalid
    """
    return validate_sequence(
        peptide,
        min_length=min_length,
        max_length=max_length,
        allow_extended=False,
        strip_whitespace=True,
    )


def validate_allele(allele: str) -> str:
    """
    Validate and normalize an HLA allele name.

    Args:
        allele: HLA allele name (e.g., "HLA-A*02:01")

    Returns:
        Normalized allele name

    Raises:
        InvalidAlleleError: If allele format is invalid
    """
    if not allele:
        raise InvalidAlleleError("Allele cannot be empty")

    # Normalize
    allele = allele.strip().upper()

    # Add HLA- prefix if missing
    if not allele.startswith("HLA-"):
        if allele.startswith(("A*", "B*", "C*")):
            allele = f"HLA-{allele}"
        else:
            raise InvalidAlleleError(
                f"Invalid allele format: {allele}",
                allele=allele,
                suggestions=_get_allele_suggestions(allele),
            )

    # Validate pattern
    if not HLA_PATTERN.match(allele):
        raise InvalidAlleleError(
            f"Invalid HLA allele format: {allele}. Expected format: HLA-A*02:01",
            allele=allele,
            suggestions=_get_allele_suggestions(allele),
        )

    return allele


def validate_alleles(alleles: list[str]) -> list[str]:
    """
    Validate a list of HLA alleles.

    Args:
        alleles: List of HLA allele names

    Returns:
        List of normalized allele names

    Raises:
        InvalidAlleleError: If any allele is invalid
    """
    if not alleles:
        raise InvalidAlleleError("At least one allele is required")

    return [validate_allele(a) for a in alleles]


@lru_cache(maxsize=1024)
def _get_allele_suggestions(invalid_allele: str) -> list[str]:
    """Get suggestions for a potentially misspelled allele."""
    suggestions: list[str] = []

    # Try to extract the gene and group
    parts = invalid_allele.replace("HLA-", "").replace("*", "").replace(":", "").upper()

    for supertype, alleles in HLA_SUPERTYPES.items():
        for allele in alleles:
            allele_simple = allele.replace("HLA-", "").replace("*", "").replace(":", "")
            if parts[:2] == allele_simple[:2]:
                suggestions.append(allele)

    return suggestions[:5]


def calculate_sequence_properties(sequence: str) -> dict[str, float]:
    """
    Calculate basic sequence properties.

    Args:
        sequence: Validated protein sequence

    Returns:
        Dictionary of sequence properties
    """
    # Amino acid groups
    hydrophobic = set("AILMFVPWG")
    polar = set("STNQCY")
    charged_pos = set("KRH")
    charged_neg = set("DE")
    aromatic = set("FWY")

    length = len(sequence)

    # Calculate fractions
    def fraction(aa_set: set[str]) -> float:
        return sum(1 for aa in sequence if aa in aa_set) / length

    # Calculate molecular weight (simplified)
    aa_weights = {
        "A": 89.1, "R": 174.2, "N": 132.1, "D": 133.1, "C": 121.2,
        "E": 147.1, "Q": 146.2, "G": 75.1, "H": 155.2, "I": 131.2,
        "L": 131.2, "K": 146.2, "M": 149.2, "F": 165.2, "P": 115.1,
        "S": 105.1, "T": 119.1, "W": 204.2, "Y": 181.2, "V": 117.1,
    }
    mw = sum(aa_weights.get(aa, 110.0) for aa in sequence) - (length - 1) * 18.02

    # Calculate isoelectric point (simplified Henderson-Hasselbalch)
    pka = {
        "K": 10.5, "R": 12.5, "H": 6.0,  # positive
        "D": 3.9, "E": 4.3, "C": 8.3, "Y": 10.1,  # negative
    }
    n_term_pka = 9.6
    c_term_pka = 2.3

    def net_charge(ph: float) -> float:
        charge = 10 ** (n_term_pka - ph) / (1 + 10 ** (n_term_pka - ph))  # N-terminus
        charge -= 10 ** (ph - c_term_pka) / (1 + 10 ** (ph - c_term_pka))  # C-terminus

        for aa in sequence:
            if aa in {"K", "R", "H"}:
                charge += 10 ** (pka[aa] - ph) / (1 + 10 ** (pka[aa] - ph))
            elif aa in {"D", "E", "C", "Y"}:
                charge -= 10 ** (ph - pka[aa]) / (1 + 10 ** (ph - pka[aa]))

        return charge

    # Binary search for pI
    low, high = 0.0, 14.0
    while high - low > 0.01:
        mid = (low + high) / 2
        if net_charge(mid) > 0:
            low = mid
        else:
            high = mid
    pi = (low + high) / 2

    # GRAVY (Grand Average of Hydropathy)
    gravy_values = {
        "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
        "E": -3.5, "Q": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
        "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
        "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
    }
    gravy = sum(gravy_values.get(aa, 0) for aa in sequence) / length

    return {
        "length": length,
        "molecular_weight": round(mw, 2),
        "isoelectric_point": round(pi, 2),
        "gravy": round(gravy, 3),
        "hydrophobic_fraction": round(fraction(hydrophobic), 3),
        "polar_fraction": round(fraction(polar), 3),
        "charged_fraction": round(fraction(charged_pos | charged_neg), 3),
        "aromatic_fraction": round(fraction(aromatic), 3),
        "net_charge_at_ph7": round(net_charge(7.0), 2),
    }


def estimate_expression_difficulty(properties: dict[str, float]) -> float:
    """
    Estimate expression difficulty based on sequence properties.

    Returns a score from 0 (easy) to 100 (difficult).
    """
    score = 0.0

    # Penalize extreme length
    length = properties["length"]
    if length < 50:
        score += 20  # Very short, may be unstable
    elif length > 500:
        score += 10 * min((length - 500) / 500, 1)  # Long proteins harder

    # Penalize extreme pI
    pi = properties["isoelectric_point"]
    if pi < 5 or pi > 9:
        score += 15  # Extreme pI can cause solubility issues

    # Penalize high hydrophobicity
    gravy = properties["gravy"]
    if gravy > 0.5:
        score += 20 * min(gravy, 1)  # Hydrophobic proteins aggregate

    # Penalize high aromatic content
    aromatic = properties["aromatic_fraction"]
    if aromatic > 0.15:
        score += 15  # High aromatic content can cause aggregation

    return min(score, 100)


def validate_structure(
    structure: str,
    *,
    require_complete: bool = False,
) -> str:
    """
    Validate a PDB structure string.

    Args:
        structure: PDB format structure string
        require_complete: If True, require ATOM records

    Returns:
        Normalized structure string

    Raises:
        ValueError: If structure is invalid
    """
    if not structure:
        raise ValueError("Structure cannot be empty")

    # Basic validation: should have ATOM or HETATM records
    lines = structure.strip().split("\n")
    has_atoms = any(
        line.startswith(("ATOM", "HETATM"))
        for line in lines
    )

    if require_complete and not has_atoms:
        raise ValueError("Structure must contain ATOM or HETATM records")

    return structure.strip()


def detect_protein_type(sequence: str) -> str:
    """
    Detect the likely protein type from sequence features.

    Returns one of: 'antibody', 'enzyme', 'peptide', 'general'
    """
    length = len(sequence)

    # Peptides: < 50 amino acids
    if length < 50:
        return "peptide"

    # Antibody detection: look for conserved motifs
    antibody_vh_motifs = ["CXXW", "WVRQ", "WFRQ", "YYCAR", "WGXG"]
    antibody_vl_motifs = ["CXXY", "WYQQ", "WFQQ", "FGXG"]

    # Check for immunoglobulin-like patterns
    has_vh_like = any(
        _motif_match(sequence, motif)
        for motif in antibody_vh_motifs
    )
    has_vl_like = any(
        _motif_match(sequence, motif)
        for motif in antibody_vl_motifs
    )

    if has_vh_like or has_vl_like:
        return "antibody"

    # Enzyme detection: look for catalytic motifs
    catalytic_motifs = [
        "HExxH",   # Metalloprotease
        "SxHxxE", # Serine protease
        "GxSxG",   # Lipase/esterase
        "GxGxxG",  # Rossmann fold (NAD binding)
        "DxDxT",   # Kinase
    ]

    has_catalytic = any(
        _motif_match(sequence, motif)
        for motif in catalytic_motifs
    )

    if has_catalytic:
        return "enzyme"

    return "general"


def _motif_match(sequence: str, motif: str) -> bool:
    """
    Check if a motif matches anywhere in the sequence.

    'x' or 'X' matches any amino acid.
    """
    motif = motif.upper()
    seq = sequence.upper()

    for i in range(len(seq) - len(motif) + 1):
        window = seq[i:i + len(motif)]
        match = True
        for m, s in zip(motif, window):
            if m != "X" and m != s:
                match = False
                break
        if match:
            return True

    return False
