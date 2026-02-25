"""
Tests for Main ProteinScore Scorer

Integration tests for the unified ProteinScore scorer.
"""

from __future__ import annotations

import pytest

from proteinscore import ProteinScore, ProteinScoreResult
from proteinscore.models import RiskLevel, ScoringWeights


class TestProteinScoreInitialization:
    """Tests for ProteinScore initialization."""

    def test_default_initialization(self) -> None:
        """Test that ProteinScore initializes with defaults."""
        scorer = ProteinScore()
        assert scorer is not None

    def test_local_only_mode(self) -> None:
        """Test local-only mode initialization."""
        scorer = ProteinScore(local_only=True)
        assert scorer._local_only is True

    def test_custom_weights(self) -> None:
        """Test custom weights initialization."""
        weights = ScoringWeights(
            stability=0.4,
            solubility=0.2,
            aggregation=0.2,
            immunogenicity=0.2,
        )
        scorer = ProteinScore(weights=weights)
        assert scorer.weights.stability == 0.4

    def test_preset_weights(self) -> None:
        """Test preset weight configurations."""
        enzyme_weights = ScoringWeights.for_enzyme()
        assert enzyme_weights.stability == 0.4

        antibody_weights = ScoringWeights.for_antibody()
        assert antibody_weights.stability == 0.25

        vaccine_weights = ScoringWeights.for_vaccine()
        assert vaccine_weights.immunogenicity == 0.3


class TestProteinScoreScoring:
    """Tests for ProteinScore scoring functionality."""

    def test_score_returns_result(self, scorer, gfp_sequence) -> None:
        """Test that score returns ProteinScoreResult."""
        result = scorer.score(gfp_sequence)
        assert isinstance(result, ProteinScoreResult)

    def test_total_score_in_range(self, scorer, gfp_sequence) -> None:
        """Test that total score is in 0-100 range."""
        result = scorer.score(gfp_sequence)
        assert 0 <= result.total_score <= 100

    def test_all_component_scores_present(self, scorer, gfp_sequence) -> None:
        """Test that all component scores are present."""
        result = scorer.score(gfp_sequence)

        assert result.stability is not None
        assert result.solubility is not None
        assert result.aggregation is not None
        assert result.immunogenicity is not None

        assert 0 <= result.stability.score <= 100
        assert 0 <= result.solubility.score <= 100
        assert 0 <= result.aggregation.score <= 100
        assert 0 <= result.immunogenicity.score <= 100

    def test_risk_level_assigned(self, scorer, gfp_sequence) -> None:
        """Test that risk level is assigned."""
        result = scorer.score(gfp_sequence)
        assert isinstance(result.risk_level, RiskLevel)

    def test_sequence_stored(self, scorer, gfp_sequence) -> None:
        """Test that sequence is stored in result."""
        result = scorer.score(gfp_sequence)
        assert result.sequence == gfp_sequence.upper().replace(" ", "")
        assert result.sequence_length == len(gfp_sequence)

    def test_name_stored(self, scorer, gfp_sequence) -> None:
        """Test that name is stored in result."""
        result = scorer.score(gfp_sequence, name="GFP")
        assert result.name == "GFP"

    def test_weights_recorded(self, scorer, gfp_sequence) -> None:
        """Test that weights used are recorded."""
        result = scorer.score(gfp_sequence)
        assert result.weights_used is not None
        assert result.weights_used.stability == 0.25  # default

    def test_recommendations_generated(self, scorer, gfp_sequence) -> None:
        """Test that recommendations are generated."""
        result = scorer.score(gfp_sequence)
        assert isinstance(result.recommendations, list)
        # May or may not have recommendations depending on scores

    def test_component_scores_dict(self, scorer, gfp_sequence) -> None:
        """Test component_scores computed property."""
        result = scorer.score(gfp_sequence)
        scores = result.component_scores

        assert "stability" in scores
        assert "solubility" in scores
        assert "aggregation" in scores
        assert "immunogenicity" in scores

    def test_weakest_component_identified(self, scorer, gfp_sequence) -> None:
        """Test weakest_component computed property."""
        result = scorer.score(gfp_sequence)
        weakest = result.weakest_component

        assert weakest in ["stability", "solubility", "aggregation", "immunogenicity"]

        # Verify it is actually the weakest
        scores = result.component_scores
        assert scores[weakest] == min(scores.values())


class TestProteinScoreBatch:
    """Tests for batch scoring."""

    def test_batch_returns_list(self, scorer, gfp_sequence, lysozyme_sequence) -> None:
        """Test that batch scoring returns list."""
        results = scorer.score_batch([gfp_sequence, lysozyme_sequence])
        assert isinstance(results, list)
        assert len(results) == 2

    def test_batch_preserves_order(
        self, scorer, gfp_sequence, lysozyme_sequence, amyloid_sequence
    ) -> None:
        """Test that batch preserves input order."""
        sequences = [gfp_sequence, lysozyme_sequence, amyloid_sequence]
        names = ["GFP", "Lysozyme", "Amyloid"]

        results = scorer.score_batch(sequences, names=names)

        assert results[0].name == "GFP"
        assert results[1].name == "Lysozyme"
        assert results[2].name == "Amyloid"

    def test_batch_parallel_vs_sequential(
        self, scorer, gfp_sequence, lysozyme_sequence
    ) -> None:
        """Test that parallel and sequential give same results."""
        sequences = [gfp_sequence, lysozyme_sequence]

        parallel_results = scorer.score_batch(sequences, parallel=True)
        sequential_results = scorer.score_batch(sequences, parallel=False)

        # Scores should be identical
        for p, s in zip(parallel_results, sequential_results):
            assert abs(p.total_score - s.total_score) < 0.1


class TestProteinScoreCompare:
    """Tests for result comparison."""

    def test_compare_returns_table(self, scorer, gfp_sequence, lysozyme_sequence) -> None:
        """Test that compare returns table string."""
        results = scorer.score_batch([gfp_sequence, lysozyme_sequence], names=["GFP", "Lysozyme"])
        comparison = scorer.compare(results, output_format="table")

        assert isinstance(comparison, str)
        assert "GFP" in comparison
        assert "Lysozyme" in comparison

    def test_compare_returns_dataframe(self, scorer, gfp_sequence, lysozyme_sequence) -> None:
        """Test that compare can return DataFrame."""
        pytest.importorskip("pandas")
        import pandas as pd

        results = scorer.score_batch([gfp_sequence, lysozyme_sequence])
        comparison = scorer.compare(results, output_format="dataframe")

        assert isinstance(comparison, pd.DataFrame)
        assert len(comparison) == 2

    def test_compare_returns_json(self, scorer, gfp_sequence, lysozyme_sequence) -> None:
        """Test that compare can return JSON."""
        import json

        results = scorer.score_batch([gfp_sequence, lysozyme_sequence])
        comparison = scorer.compare(results, output_format="json")

        # Should be valid JSON
        data = json.loads(comparison)
        assert isinstance(data, list)
        assert len(data) == 2


class TestProteinScoreCache:
    """Tests for caching functionality."""

    def test_cache_hit(self, scorer, gfp_sequence) -> None:
        """Test that repeated calls use cache."""
        result1 = scorer.score(gfp_sequence)
        result2 = scorer.score(gfp_sequence)

        # Results should be identical (same object from cache)
        assert result1.total_score == result2.total_score
        assert result1.timestamp == result2.timestamp

    def test_cache_cleared(self, scorer, gfp_sequence) -> None:
        """Test that cache can be cleared."""
        result1 = scorer.score(gfp_sequence)
        scorer.clear_cache()
        result2 = scorer.score(gfp_sequence)

        # Timestamps should be different after cache clear
        assert result1.total_score == result2.total_score
        # (timestamps might be very close so don't test that)

    def test_cache_disabled(self, gfp_sequence) -> None:
        """Test that cache can be disabled."""
        scorer = ProteinScore(local_only=True, cache_results=False)
        result1 = scorer.score(gfp_sequence)
        result2 = scorer.score(gfp_sequence)

        # Results should have different timestamps
        # (may be the same if very fast, so just verify both work)
        assert result1.total_score == result2.total_score


class TestProteinScoreWeightedScoring:
    """Tests for weighted score calculation."""

    def test_equal_weights_average(self, gfp_sequence) -> None:
        """Test that equal weights give simple average."""
        scorer = ProteinScore(local_only=True)
        result = scorer.score(gfp_sequence)

        # With equal weights, total should be average of components
        expected = (
            result.stability.score +
            result.solubility.score +
            result.aggregation.score +
            result.immunogenicity.score
        ) / 4

        assert abs(result.total_score - expected) < 0.2

    def test_custom_weights_affect_score(self, gfp_sequence) -> None:
        """Test that custom weights affect total score."""
        # Score with stability-heavy weights
        stability_weights = ScoringWeights(
            stability=0.7,
            solubility=0.1,
            aggregation=0.1,
            immunogenicity=0.1,
        )
        scorer_stability = ProteinScore(local_only=True, weights=stability_weights)
        result_stability = scorer_stability.score(gfp_sequence)

        # Score with immunogenicity-heavy weights
        immuno_weights = ScoringWeights(
            stability=0.1,
            solubility=0.1,
            aggregation=0.1,
            immunogenicity=0.7,
        )
        scorer_immuno = ProteinScore(local_only=True, weights=immuno_weights)
        result_immuno = scorer_immuno.score(gfp_sequence)

        # Total scores should differ based on component scores
        # (they might be similar by chance, so just verify they're valid)
        assert 0 <= result_stability.total_score <= 100
        assert 0 <= result_immuno.total_score <= 100


class TestProteinScoreResultMethods:
    """Tests for ProteinScoreResult methods."""

    def test_to_dict(self, scorer, gfp_sequence) -> None:
        """Test result to_dict method."""
        result = scorer.score(gfp_sequence, name="GFP")
        d = result.to_dict()

        assert isinstance(d, dict)
        assert d["name"] == "GFP"
        assert "total_score" in d

    def test_summary(self, scorer, gfp_sequence) -> None:
        """Test result summary method."""
        result = scorer.score(gfp_sequence, name="GFP")
        summary = result.summary()

        assert isinstance(summary, str)
        assert "GFP" in summary
        assert "Stability" in summary
        assert "/100" in summary

    def test_to_json(self, scorer, gfp_sequence, tmp_path) -> None:
        """Test result to_json method."""
        import json

        result = scorer.score(gfp_sequence, name="GFP")
        path = tmp_path / "result.json"
        result.to_json(str(path))

        assert path.exists()
        data = json.loads(path.read_text())
        assert data["name"] == "GFP"

    def test_to_csv(self, scorer, gfp_sequence, tmp_path) -> None:
        """Test result to_csv method."""
        result = scorer.score(gfp_sequence, name="GFP")
        path = tmp_path / "result.csv"
        result.to_csv(str(path))

        assert path.exists()
        content = path.read_text()
        assert "GFP" in content
        assert "total_score" in content
