"""
Pytest Configuration and Fixtures

Shared fixtures for ProteinScore test suite.
"""

from __future__ import annotations

import pytest

# Well-characterized test sequences
TEST_SEQUENCES = {
    # GFP - well-expressed, stable protein
    "gfp": (
        "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLV"
        "TTLTYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNR"
        "IELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQ"
        "QNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITLGMDELYK"
    ),
    # Insulin B chain - therapeutic peptide
    "insulin_b": "FVNQHLCGSHLVEALYLVCGERGFFYTPKT",
    # Amyloid beta fragment - aggregation-prone
    "amyloid_beta": "DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA",
    # T4 lysozyme - well-studied stability
    "t4_lysozyme": (
        "MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKSELDKAIGRNTNGVITKD"
        "EAEKLFNQDVDAAVRGILRNAKLKPVYDSLDAVRRAALINMVFQMGETGVAGFTNSLRMLQ"
        "QKRWDEAAVNLAKSRWYNQTPNRAKRVITTFRTGTWDAYKNL"
    ),
    # Short test peptide
    "short_peptide": "YLQPRTFLL",
    # HLA-A*02:01 binding epitope
    "epitope_a0201": "GILGFVFTL",  # Flu M1
}


@pytest.fixture
def gfp_sequence() -> str:
    """Green Fluorescent Protein sequence."""
    return TEST_SEQUENCES["gfp"]


@pytest.fixture
def insulin_sequence() -> str:
    """Insulin B chain sequence."""
    return TEST_SEQUENCES["insulin_b"]


@pytest.fixture
def amyloid_sequence() -> str:
    """Amyloid beta fragment (aggregation-prone)."""
    return TEST_SEQUENCES["amyloid_beta"]


@pytest.fixture
def lysozyme_sequence() -> str:
    """T4 lysozyme sequence."""
    return TEST_SEQUENCES["t4_lysozyme"]


@pytest.fixture
def short_peptide() -> str:
    """Short peptide for testing."""
    return TEST_SEQUENCES["short_peptide"]


@pytest.fixture
def epitope_sequence() -> str:
    """Known HLA-A*02:01 binding epitope."""
    return TEST_SEQUENCES["epitope_a0201"]


@pytest.fixture
def scorer():
    """ProteinScore instance in local-only mode."""
    from proteinscore import ProteinScore
    return ProteinScore(local_only=True)


@pytest.fixture
def scorer_with_api(monkeypatch):
    """ProteinScore instance with mock API key."""
    from proteinscore import ProteinScore
    monkeypatch.setenv("RUNLAB_API_KEY", "rl_test_key_12345")
    return ProteinScore()


@pytest.fixture
def stability_predictor():
    """Stability predictor instance."""
    from proteinscore.predictors import StabilityPredictor
    return StabilityPredictor()


@pytest.fixture
def solubility_predictor():
    """Solubility predictor instance."""
    from proteinscore.predictors import SolubilityPredictor
    return SolubilityPredictor()


@pytest.fixture
def aggregation_predictor():
    """Aggregation predictor instance."""
    from proteinscore.predictors import AggregationPredictor
    return AggregationPredictor()


@pytest.fixture
def immunogenicity_predictor():
    """Immunogenicity predictor instance (local mode)."""
    from proteinscore.predictors import ImmunogenicityPredictor
    return ImmunogenicityPredictor(api_key=None)
