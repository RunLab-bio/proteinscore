"""
Tests for Enzyme Module

Tests for thermostability, expression, and unified enzyme scoring.
"""

from __future__ import annotations

import pytest

from proteinscore.enzyme import (
    EnzymeScorer,
    EnzymeResult,
    EnzymeThermostabilityPredictor,
    ThermostabilityResult,
    ExpressionPredictor,
    ExpressionResult,
    ExpressionHost,
)
from proteinscore.enzyme.thermostability import ThermostabilityClass


# Test sequences
@pytest.fixture
def lipase_sequence() -> str:
    """Lipase B from Candida antarctica - known thermostable enzyme."""
    return (
        "LPSGSDPAFSQPKSVLDAGLTCQGASPSSVSKPILLVPGTGTTGPQSFDSNWIPLSTQL"
        "GYTPCWISPPPFMLNDTQVNTEYMVNAITALYAGSGNNKLPVLTWSQGGLVAQWGLTFF"
        "PSIRSKVDRLMAAVKPDKLVLFSHSAGASTTVIPKGKTGGKWNTLPSQNGMVSTQYFEP"
        "DVKAACKLVLNRNLKVTDNSANLSIILNLYPTSYVLNKAQGDFVIVPGGLNNNIQMLPW"
        "ALKQGLPVLRAISDNGVLADNKITLTGIAYYSQAASKTLKPTLAVALQQAGQKGPVVVV"
        "SNPTSKAVLQTIPITDVNKQ"
    )


@pytest.fixture
def lysozyme_sequence() -> str:
    """Hen egg white lysozyme - well characterized enzyme."""
    return (
        "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQIN"
        "SRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGT"
        "DVQAWIRGCRL"
    )


@pytest.fixture
def unstable_enzyme() -> str:
    """Synthetic sequence with destabilizing features."""
    return (
        "MNNNNNQQQQQSSSSSGGGGGCCCCCMMMMM"
        "NNNNQQQSSSGGCCMMNNNQQQSSSGGCCMM"
        "NNNQQSSGCMNNNQQSSGCMNNNQQSSGCM"
    )


@pytest.fixture
def thermophile_enzyme() -> str:
    """Thermophilic enzyme-like sequence (high E, K, P content)."""
    return (
        "MEKPVLIEKPEKPVLIKEPKEPVLVKEPKEPVLVKEPKEPVLVKEPKEPVLVKEPK"
        "EKPVLVKEPKEPVLVKEPKEPVLVKEPKEPVLVKEPKEPVLVKEPKEPVLVKEPKE"
        "PVLVKEPKEPVLVKEPKEPVLVKEPKEPVLVKEPKEPVLVKEPKEPVLVKEPKEPV"
    )


class TestEnzymeThermostabilityPredictor:
    """Tests for thermostability prediction."""

    def test_predict_returns_result(self, lipase_sequence: str) -> None:
        """Test that predict returns ThermostabilityResult."""
        predictor = EnzymeThermostabilityPredictor()
        result = predictor.predict(lipase_sequence)
        assert isinstance(result, ThermostabilityResult)

    def test_score_in_valid_range(self, lipase_sequence: str) -> None:
        """Test that score is in 0-100 range."""
        predictor = EnzymeThermostabilityPredictor()
        result = predictor.predict(lipase_sequence)
        assert 0 <= result.score <= 100

    def test_tm_in_realistic_range(self, lipase_sequence: str) -> None:
        """Test that Tm estimate is in realistic range."""
        predictor = EnzymeThermostabilityPredictor()
        result = predictor.predict(lipase_sequence)
        # Most enzymes have Tm between 25-95°C
        assert 25 <= result.tm_estimate <= 95

    def test_confidence_interval_valid(self, lipase_sequence: str) -> None:
        """Test that confidence interval is valid."""
        predictor = EnzymeThermostabilityPredictor()
        result = predictor.predict(lipase_sequence)
        assert result.tm_confidence_interval[0] < result.tm_estimate
        assert result.tm_confidence_interval[1] > result.tm_estimate

    def test_thermostability_class_assigned(self, lipase_sequence: str) -> None:
        """Test that thermostability class is assigned."""
        predictor = EnzymeThermostabilityPredictor()
        result = predictor.predict(lipase_sequence)
        assert isinstance(result.thermostability_class, ThermostabilityClass)

    def test_process_storage_scores_valid(self, lipase_sequence: str) -> None:
        """Test that process and storage scores are valid."""
        predictor = EnzymeThermostabilityPredictor()
        result = predictor.predict(lipase_sequence)
        assert 0 <= result.process_stability_score <= 100
        assert 0 <= result.storage_stability_score <= 100

    def test_feature_contributions_present(self, lipase_sequence: str) -> None:
        """Test that feature contributions are calculated."""
        predictor = EnzymeThermostabilityPredictor()
        result = predictor.predict(lipase_sequence)
        assert len(result.feature_contributions) > 0
        assert "shannon_entropy" in result.feature_contributions
        assert "thermophile_composition" in result.feature_contributions

    def test_thermophile_sequence_higher_tm(
        self, thermophile_enzyme: str, unstable_enzyme: str
    ) -> None:
        """Test that thermophile-like sequence has higher Tm."""
        predictor = EnzymeThermostabilityPredictor()
        thermo_result = predictor.predict(thermophile_enzyme)
        unstable_result = predictor.predict(unstable_enzyme)
        # Thermophile should have higher Tm
        assert thermo_result.tm_estimate > unstable_result.tm_estimate

    def test_unstable_sequence_lower_score(
        self, lipase_sequence: str, unstable_enzyme: str
    ) -> None:
        """Test that unstable sequence has lower score."""
        predictor = EnzymeThermostabilityPredictor()
        lipase_result = predictor.predict(lipase_sequence)
        unstable_result = predictor.predict(unstable_enzyme)
        # Unstable sequence should have lower score
        assert unstable_result.score < lipase_result.score


class TestExpressionPredictor:
    """Tests for expression prediction."""

    def test_predict_returns_result(self, lipase_sequence: str) -> None:
        """Test that predict returns ExpressionResult."""
        predictor = ExpressionPredictor()
        result = predictor.predict(lipase_sequence)
        assert isinstance(result, ExpressionResult)

    def test_score_in_valid_range(self, lipase_sequence: str) -> None:
        """Test that score is in 0-100 range."""
        predictor = ExpressionPredictor()
        result = predictor.predict(lipase_sequence)
        assert 0 <= result.score <= 100

    def test_host_is_set(self, lipase_sequence: str) -> None:
        """Test that host is correctly set."""
        predictor = ExpressionPredictor(host=ExpressionHost.E_COLI)
        result = predictor.predict(lipase_sequence)
        assert result.host == ExpressionHost.E_COLI

    def test_different_hosts(self, lipase_sequence: str) -> None:
        """Test prediction with different hosts."""
        for host in [ExpressionHost.E_COLI, ExpressionHost.CHO, ExpressionHost.PICHIA]:
            predictor = ExpressionPredictor(host=host)
            result = predictor.predict(lipase_sequence)
            assert result.host == host
            assert 0 <= result.score <= 100

    def test_solubility_score_valid(self, lipase_sequence: str) -> None:
        """Test that solubility score is valid."""
        predictor = ExpressionPredictor()
        result = predictor.predict(lipase_sequence)
        assert 0 <= result.solubility_score <= 100

    def test_problem_regions_identified(self, unstable_enzyme: str) -> None:
        """Test that problem regions are identified."""
        predictor = ExpressionPredictor()
        result = predictor.predict(unstable_enzyme)
        # Unstable sequence should have some problems
        assert isinstance(result.problem_regions, list)

    def test_recommendations_generated(self, lipase_sequence: str) -> None:
        """Test that recommendations are generated."""
        predictor = ExpressionPredictor()
        result = predictor.predict(lipase_sequence)
        assert isinstance(result.recommendations, list)

    def test_interpretation_not_empty(self, lipase_sequence: str) -> None:
        """Test that interpretation is provided."""
        predictor = ExpressionPredictor()
        result = predictor.predict(lipase_sequence)
        assert result.interpretation
        assert len(result.interpretation) > 10


class TestEnzymeScorer:
    """Tests for unified enzyme scorer."""

    def test_score_returns_result(self, lipase_sequence: str) -> None:
        """Test that score returns EnzymeResult."""
        scorer = EnzymeScorer()
        result = scorer.score(lipase_sequence)
        assert isinstance(result, EnzymeResult)

    def test_total_score_in_range(self, lipase_sequence: str) -> None:
        """Test that total score is in valid range."""
        scorer = EnzymeScorer()
        result = scorer.score(lipase_sequence)
        assert 0 <= result.total_score <= 100

    def test_all_components_present(self, lipase_sequence: str) -> None:
        """Test that all component results are present."""
        scorer = EnzymeScorer()
        result = scorer.score(lipase_sequence)
        assert result.thermostability is not None
        assert result.expression is not None
        assert result.solubility is not None
        assert result.aggregation is not None

    def test_component_scores_dict(self, lipase_sequence: str) -> None:
        """Test component_scores property."""
        scorer = EnzymeScorer()
        result = scorer.score(lipase_sequence)
        scores = result.component_scores
        assert "thermostability" in scores
        assert "expression" in scores
        assert "solubility" in scores
        assert "aggregation" in scores

    def test_weakest_component_identified(self, lipase_sequence: str) -> None:
        """Test that weakest component is identified."""
        scorer = EnzymeScorer()
        result = scorer.score(lipase_sequence)
        assert result.weakest_component in [
            "thermostability", "expression", "solubility", "aggregation"
        ]

    def test_tm_estimate_accessible(self, lipase_sequence: str) -> None:
        """Test that Tm estimate is accessible."""
        scorer = EnzymeScorer()
        result = scorer.score(lipase_sequence)
        assert 25 <= result.tm_estimate <= 95

    def test_expression_level_accessible(self, lipase_sequence: str) -> None:
        """Test that expression level is accessible."""
        scorer = EnzymeScorer()
        result = scorer.score(lipase_sequence)
        assert result.expression_level in ["high", "medium", "low", "insoluble"]

    def test_name_stored(self, lipase_sequence: str) -> None:
        """Test that name is stored."""
        scorer = EnzymeScorer()
        result = scorer.score(lipase_sequence, name="CalB")
        assert result.name == "CalB"

    def test_host_stored(self, lipase_sequence: str) -> None:
        """Test that host is stored."""
        scorer = EnzymeScorer(host=ExpressionHost.CHO)
        result = scorer.score(lipase_sequence)
        assert result.expression_host == ExpressionHost.CHO

    def test_recommendations_generated(self, lipase_sequence: str) -> None:
        """Test that recommendations are generated."""
        scorer = EnzymeScorer()
        result = scorer.score(lipase_sequence)
        assert isinstance(result.recommendations, list)

    def test_summary_generated(self, lipase_sequence: str) -> None:
        """Test that summary is generated."""
        scorer = EnzymeScorer()
        result = scorer.score(lipase_sequence, name="CalB")
        summary = result.summary()
        assert "CalB" in summary
        assert "Total Score" in summary
        assert "Melting Temperature" in summary

    def test_batch_scoring(self, lipase_sequence: str, lysozyme_sequence: str) -> None:
        """Test batch scoring."""
        scorer = EnzymeScorer()
        sequences = [
            {"sequence": lipase_sequence, "name": "CalB"},
            {"sequence": lysozyme_sequence, "name": "Lysozyme"},
        ]
        results = scorer.score_batch(sequences)
        assert len(results) == 2
        assert results[0].name == "CalB"
        assert results[1].name == "Lysozyme"

    def test_custom_weights(self, lipase_sequence: str) -> None:
        """Test custom weight configuration."""
        weights = {
            "thermostability": 0.5,
            "expression": 0.2,
            "solubility": 0.2,
            "aggregation": 0.1,
            "immunogenicity": 0.0,
        }
        scorer = EnzymeScorer(weights=weights)
        result = scorer.score(lipase_sequence)
        assert 0 <= result.total_score <= 100


class TestEnzymeModuleIntegration:
    """Integration tests for the enzyme module."""

    def test_lysozyme_reasonable_scores(self, lysozyme_sequence: str) -> None:
        """Test that lysozyme gets reasonable scores."""
        scorer = EnzymeScorer()
        result = scorer.score(lysozyme_sequence, name="Lysozyme")

        # Lysozyme has 8 cysteines forming 4 disulfide bonds
        # In E. coli cytoplasm, this leads to low expression scores
        # This is scientifically accurate - lysozyme requires special strains
        assert result.total_score >= 40
        assert result.thermostability.score >= 30
        # Expression is low in E. coli due to disulfide issues
        # The model correctly identifies this as problematic
        assert result.expression.problem_regions[0]["type"] == "multiple_cysteines"

    def test_lipase_scores_well(self, lipase_sequence: str) -> None:
        """Test that lipase B scores well."""
        scorer = EnzymeScorer()
        result = scorer.score(lipase_sequence, name="CalB")

        # CalB is known to be a good enzyme for expression
        assert result.total_score >= 40

    def test_all_hosts_work(self, lipase_sequence: str) -> None:
        """Test that all expression hosts work."""
        for host in ExpressionHost:
            scorer = EnzymeScorer(host=host)
            result = scorer.score(lipase_sequence)
            assert 0 <= result.total_score <= 100
            assert result.expression_host == host
