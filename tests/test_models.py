"""
Tests for Data Models

Tests for Pydantic models and their validation.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from proteinscore.models import (
    Epitope,
    Recommendation,
    Region,
    RiskLevel,
    ScoringWeights,
    SolubilityClass,
)


class TestRiskLevel:
    """Tests for RiskLevel enum."""

    def test_from_score_high(self) -> None:
        """Test high risk classification."""
        assert RiskLevel.from_score(30) == RiskLevel.HIGH
        assert RiskLevel.from_score(49) == RiskLevel.HIGH

    def test_from_score_medium(self) -> None:
        """Test medium risk classification."""
        assert RiskLevel.from_score(50) == RiskLevel.MEDIUM
        assert RiskLevel.from_score(69) == RiskLevel.MEDIUM

    def test_from_score_low(self) -> None:
        """Test low risk classification."""
        assert RiskLevel.from_score(70) == RiskLevel.LOW
        assert RiskLevel.from_score(100) == RiskLevel.LOW


class TestSolubilityClass:
    """Tests for SolubilityClass enum."""

    def test_from_score_high(self) -> None:
        """Test high solubility classification."""
        assert SolubilityClass.from_score(80) == SolubilityClass.HIGH
        assert SolubilityClass.from_score(100) == SolubilityClass.HIGH

    def test_from_score_medium(self) -> None:
        """Test medium solubility classification."""
        assert SolubilityClass.from_score(60) == SolubilityClass.MEDIUM
        assert SolubilityClass.from_score(79) == SolubilityClass.MEDIUM

    def test_from_score_low(self) -> None:
        """Test low solubility classification."""
        assert SolubilityClass.from_score(40) == SolubilityClass.LOW
        assert SolubilityClass.from_score(59) == SolubilityClass.LOW

    def test_from_score_aggregation_prone(self) -> None:
        """Test aggregation-prone classification."""
        assert SolubilityClass.from_score(39) == SolubilityClass.AGGREGATION_PRONE
        assert SolubilityClass.from_score(0) == SolubilityClass.AGGREGATION_PRONE


class TestRegion:
    """Tests for Region model."""

    def test_valid_region(self) -> None:
        """Test valid region creation."""
        region = Region(start=10, end=20, score=75.5, sequence="MKTAYIAK")
        assert region.start == 10
        assert region.end == 20
        assert region.score == 75.5
        assert region.sequence == "MKTAYIAK"

    def test_length_computed(self) -> None:
        """Test that length is computed correctly."""
        region = Region(start=10, end=25, score=50)
        assert region.length == 15

    def test_annotation_optional(self) -> None:
        """Test that annotation is optional."""
        region = Region(start=0, end=10, score=50)
        assert region.annotation is None

        region_with_annotation = Region(
            start=0, end=10, score=50, annotation="Test annotation"
        )
        assert region_with_annotation.annotation == "Test annotation"

    def test_score_validation(self) -> None:
        """Test that score must be in valid range."""
        with pytest.raises(ValidationError):
            Region(start=0, end=10, score=-1)

        with pytest.raises(ValidationError):
            Region(start=0, end=10, score=101)


class TestEpitope:
    """Tests for Epitope model."""

    def test_valid_epitope(self) -> None:
        """Test valid epitope creation."""
        epitope = Epitope(
            peptide="YLQPRTFLL",
            position=(10, 19),
            allele="HLA-A*02:01",
            binding_affinity_nM=12.5,
            percentile_rank=0.15,
            is_strong_binder=True,
            is_weak_binder=True,
            confidence=0.92,
        )
        assert epitope.peptide == "YLQPRTFLL"
        assert epitope.allele == "HLA-A*02:01"
        assert epitope.is_strong_binder is True

    def test_peptide_length_validation(self) -> None:
        """Test that peptide length is validated."""
        # Too short
        with pytest.raises(ValidationError):
            Epitope(
                peptide="YLQPRT",  # 6 AA, min is 8
                position=(0, 6),
                allele="HLA-A*02:01",
                binding_affinity_nM=12.5,
                percentile_rank=0.15,
                is_strong_binder=True,
                is_weak_binder=True,
                confidence=0.92,
            )

    def test_confidence_range(self) -> None:
        """Test that confidence must be 0-1."""
        with pytest.raises(ValidationError):
            Epitope(
                peptide="YLQPRTFLL",
                position=(0, 9),
                allele="HLA-A*02:01",
                binding_affinity_nM=12.5,
                percentile_rank=0.15,
                is_strong_binder=True,
                is_weak_binder=True,
                confidence=1.5,  # Invalid
            )


class TestRecommendation:
    """Tests for Recommendation model."""

    def test_valid_recommendation(self) -> None:
        """Test valid recommendation creation."""
        rec = Recommendation(
            category="stability",
            severity="warning",
            message="Consider optimizing unstable regions.",
        )
        assert rec.category == "stability"
        assert rec.severity == "warning"
        assert "unstable" in rec.message

    def test_with_region(self) -> None:
        """Test recommendation with associated region."""
        region = Region(start=10, end=20, score=30)
        rec = Recommendation(
            category="aggregation",
            severity="critical",
            message="High aggregation risk in region.",
            region=region,
        )
        assert rec.region is not None
        assert rec.region.start == 10

    def test_with_mutations(self) -> None:
        """Test recommendation with suggested mutations."""
        rec = Recommendation(
            category="immunogenicity",
            severity="warning",
            message="Consider deimmunization.",
            suggested_mutations=["S45A", "T67V"],
        )
        assert rec.suggested_mutations is not None
        assert len(rec.suggested_mutations) == 2


class TestScoringWeights:
    """Tests for ScoringWeights model."""

    def test_default_weights(self) -> None:
        """Test default equal weights."""
        weights = ScoringWeights()
        assert weights.stability == 0.25
        assert weights.solubility == 0.25
        assert weights.aggregation == 0.25
        assert weights.immunogenicity == 0.25

    def test_custom_weights(self) -> None:
        """Test custom weights."""
        weights = ScoringWeights(
            stability=0.4,
            solubility=0.2,
            aggregation=0.2,
            immunogenicity=0.2,
        )
        assert weights.stability == 0.4

    def test_weights_must_sum_to_one(self) -> None:
        """Test that weights must sum to 1.0."""
        with pytest.raises(ValueError, match="sum to 1.0"):
            ScoringWeights(
                stability=0.5,
                solubility=0.5,
                aggregation=0.5,
                immunogenicity=0.5,
            )

    def test_enzyme_preset(self) -> None:
        """Test enzyme preset weights."""
        weights = ScoringWeights.for_enzyme()
        assert weights.stability == 0.4
        total = sum([
            weights.stability,
            weights.solubility,
            weights.aggregation,
            weights.immunogenicity,
        ])
        assert abs(total - 1.0) < 0.001

    def test_antibody_preset(self) -> None:
        """Test antibody preset weights."""
        weights = ScoringWeights.for_antibody()
        total = sum([
            weights.stability,
            weights.solubility,
            weights.aggregation,
            weights.immunogenicity,
        ])
        assert abs(total - 1.0) < 0.001

    def test_vaccine_preset(self) -> None:
        """Test vaccine preset weights."""
        weights = ScoringWeights.for_vaccine()
        assert weights.immunogenicity == 0.3
        total = sum([
            weights.stability,
            weights.solubility,
            weights.aggregation,
            weights.immunogenicity,
        ])
        assert abs(total - 1.0) < 0.001

    def test_weights_immutable(self) -> None:
        """Test that weights are immutable (frozen)."""
        weights = ScoringWeights()
        with pytest.raises(ValidationError):
            weights.stability = 0.5  # type: ignore
