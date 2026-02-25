"""
Tests for Input Validation Module

Validates sequence and allele validation logic.
"""

from __future__ import annotations

import pytest

from proteinscore.exceptions import InvalidAlleleError, InvalidSequenceError
from proteinscore.validation import (
    calculate_sequence_properties,
    validate_allele,
    validate_alleles,
    validate_peptide,
    validate_sequence,
)


class TestSequenceValidation:
    """Tests for sequence validation."""

    def test_valid_sequence(self) -> None:
        """Test validation of a valid sequence."""
        seq = "MKTAYIAKQRQISFVK"
        result = validate_sequence(seq)
        assert result == seq

    def test_lowercase_normalized(self) -> None:
        """Test that lowercase is converted to uppercase."""
        result = validate_sequence("mktayiak")
        assert result == "MKTAYIAK"

    def test_whitespace_removed(self) -> None:
        """Test that whitespace is removed."""
        result = validate_sequence("MKT AYI AK")
        assert result == "MKTAYIAK"

    def test_newlines_removed(self) -> None:
        """Test that newlines are removed."""
        result = validate_sequence("MKT\nAYI\nAK")
        assert result == "MKTAYIAK"

    def test_empty_sequence_raises(self) -> None:
        """Test that empty sequence raises error."""
        with pytest.raises(InvalidSequenceError, match="cannot be empty"):
            validate_sequence("")

    def test_invalid_characters_raises(self) -> None:
        """Test that invalid characters raise error."""
        with pytest.raises(InvalidSequenceError, match="invalid characters"):
            validate_sequence("MKTX123")

    def test_invalid_characters_listed(self) -> None:
        """Test that invalid characters are listed in error."""
        try:
            validate_sequence("MKTX1Z")
        except InvalidSequenceError as e:
            assert "X" in e.invalid_chars
            assert "1" in e.invalid_chars
            assert "Z" in e.invalid_chars

    def test_too_short_raises(self) -> None:
        """Test that too-short sequence raises error."""
        with pytest.raises(InvalidSequenceError, match="too short"):
            validate_sequence("MK", min_length=10)

    def test_too_long_raises(self) -> None:
        """Test that too-long sequence raises error."""
        with pytest.raises(InvalidSequenceError, match="too long"):
            validate_sequence("M" * 100, max_length=50)

    def test_extended_amino_acids_blocked(self) -> None:
        """Test that extended AA codes are blocked by default."""
        with pytest.raises(InvalidSequenceError):
            validate_sequence("MKTBJU")

    def test_extended_amino_acids_allowed(self) -> None:
        """Test that extended AA codes can be allowed."""
        result = validate_sequence("MKTBXZ", allow_extended=True)
        assert result == "MKTBXZ"


class TestPeptideValidation:
    """Tests for peptide validation (MHC binding)."""

    def test_valid_peptide(self) -> None:
        """Test validation of valid peptide."""
        result = validate_peptide("YLQPRTFLL")
        assert result == "YLQPRTFLL"

    def test_too_short_peptide(self) -> None:
        """Test that short peptides are rejected."""
        with pytest.raises(InvalidSequenceError, match="too short"):
            validate_peptide("YLQPR")  # 5 AA, minimum is 8

    def test_too_long_peptide(self) -> None:
        """Test that long peptides are rejected."""
        with pytest.raises(InvalidSequenceError, match="too long"):
            validate_peptide("YLQPRTFLLAAAAAAA")  # 16 AA, maximum is 15


class TestAlleleValidation:
    """Tests for HLA allele validation."""

    def test_valid_allele(self) -> None:
        """Test validation of valid HLA allele."""
        result = validate_allele("HLA-A*02:01")
        assert result == "HLA-A*02:01"

    def test_lowercase_normalized(self) -> None:
        """Test that lowercase is normalized."""
        result = validate_allele("hla-a*02:01")
        assert result == "HLA-A*02:01"

    def test_prefix_added(self) -> None:
        """Test that HLA- prefix is added if missing."""
        result = validate_allele("A*02:01")
        assert result == "HLA-A*02:01"

    def test_empty_raises(self) -> None:
        """Test that empty allele raises error."""
        with pytest.raises(InvalidAlleleError, match="cannot be empty"):
            validate_allele("")

    def test_invalid_format_raises(self) -> None:
        """Test that invalid format raises error."""
        with pytest.raises(InvalidAlleleError, match="Invalid"):
            validate_allele("INVALID")

    def test_suggestions_provided(self) -> None:
        """Test that suggestions are provided for invalid allele."""
        try:
            validate_allele("A02")
        except InvalidAlleleError as e:
            assert e.suggestions is not None
            assert any("HLA-A*02" in s for s in e.suggestions)

    def test_validate_alleles_list(self) -> None:
        """Test validation of multiple alleles."""
        alleles = ["HLA-A*02:01", "HLA-B*07:02"]
        result = validate_alleles(alleles)
        assert result == ["HLA-A*02:01", "HLA-B*07:02"]

    def test_empty_list_raises(self) -> None:
        """Test that empty list raises error."""
        with pytest.raises(InvalidAlleleError, match="(?i)at least one"):
            validate_alleles([])


class TestSequenceProperties:
    """Tests for sequence property calculation."""

    def test_basic_properties(self, gfp_sequence: str) -> None:
        """Test calculation of basic sequence properties."""
        props = calculate_sequence_properties(gfp_sequence)

        assert "length" in props
        assert props["length"] == len(gfp_sequence)
        assert "molecular_weight" in props
        assert props["molecular_weight"] > 0
        assert "isoelectric_point" in props
        assert 0 < props["isoelectric_point"] < 14
        assert "gravy" in props
        assert "hydrophobic_fraction" in props
        assert 0 <= props["hydrophobic_fraction"] <= 1

    def test_hydrophobic_protein(self) -> None:
        """Test properties of hydrophobic protein."""
        hydrophobic = "AAAVVVLLLIIIFFFMMM"
        props = calculate_sequence_properties(hydrophobic)

        assert props["gravy"] > 0  # Should be positive (hydrophobic)
        assert props["hydrophobic_fraction"] > 0.5

    def test_charged_protein(self) -> None:
        """Test properties of highly charged protein."""
        charged = "KKKRRREEEDDDKKK"
        props = calculate_sequence_properties(charged)

        assert props["charged_fraction"] > 0.8

    def test_short_sequence(self) -> None:
        """Test properties of short sequence."""
        short = "MKTA"
        props = calculate_sequence_properties(short)

        assert props["length"] == 4
        assert props["molecular_weight"] > 0
