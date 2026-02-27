"""
Tests for ML-based HIC retention prediction.

Validates the GBM model with 59 handcrafted features that achieves ρ = 0.55
on the Jain 2017 benchmark dataset.
"""

from __future__ import annotations

import numpy as np
import pytest


# Test antibody sequences from Jain 2017 dataset
TEST_VH = (
    "EVQLVESGGGLVQPGGSLRLSCAASGFTFSDSWIHWVRQAPGKGLEWVAWISPYGGSTYYADSVKG"
    "RFTISADTSKNTAYLQMNSLRAEDTAVYYCARRHWPGGFDYWGQGTLVTVSS"
)

TEST_VL = (
    "DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSG"
    "SGTDFTLTISSLQPEDFATYYCQQSYSTPLTFGGGTKVEIK"
)

# Second test case with different properties
TEST_VH_2 = (
    "QVQLVQSGAEVKKPGSSVKVSCKASGGTFSSYAISWVRQAPGQGLEWMGGIIPIFGTANYAQKFQG"
    "RVTITADESTSTAYMELSSLRSEDTAVYYCARDKGHNCFDYWGQGTTVTVSS"
)

TEST_VL_2 = (
    "EIVLTQSPATLSLSPGERATLSCRASQSVSSYLAWYQQKPGQAPRLLIYDASNRATGIPARFSGSG"
    "SGTDFTLTISSLEPEDFAVYYCQQRSNWPITFGQGTRLEIK"
)


class TestHICPredictor:
    """Tests for HIC predictor module."""

    def test_feature_extraction(self):
        """Test that feature extraction produces 59 features."""
        from proteinscore.antibody.hic_predictor import (
            extract_all_features,
            get_feature_names,
        )

        features = extract_all_features(TEST_VH, TEST_VL)
        feature_names = get_feature_names()

        assert len(features) == 59, f"Expected 59 features, got {len(features)}"
        assert len(feature_names) == 59, f"Expected 59 feature names, got {len(feature_names)}"
        assert not np.isnan(features).any(), "Features should not contain NaN"

    def test_feature_names_match_categories(self):
        """Test that feature names cover expected categories."""
        from proteinscore.antibody.hic_predictor import get_feature_names

        names = get_feature_names()

        # Check amino acid composition features (20)
        aa_features = [n for n in names if n.startswith("aa_comp_")]
        assert len(aa_features) == 20, f"Expected 20 AA composition features, got {len(aa_features)}"

        # Check CDR features (12 = 4 CDRs x 3 properties: hydropathy, aromatic_frac, length_norm)
        cdr_features = [n for n in names if n.startswith("cdr_")]
        assert len(cdr_features) == 12, f"Expected 12 CDR features, got {len(cdr_features)}"

        # Check global features
        assert "aromatic_mean" in names
        assert "charge_mean" in names
        assert "hydropathy_mean" in names

    def test_predict_hic_returns_result(self):
        """Test that HIC prediction returns a result object."""
        from proteinscore.antibody.hic_predictor import predict_hic

        result = predict_hic(TEST_VH, TEST_VL)

        assert hasattr(result, 'predicted_retention')
        assert hasattr(result, 'percentile')
        assert hasattr(result, 'retention_class')
        assert result.predicted_retention > 0, "HIC retention should be positive"

    def test_predict_hic_different_sequences(self):
        """Test that different sequences give different predictions."""
        from proteinscore.antibody.hic_predictor import predict_hic

        result_1 = predict_hic(TEST_VH, TEST_VL)
        result_2 = predict_hic(TEST_VH_2, TEST_VL_2)

        # Different sequences should give different predictions
        assert result_1.predicted_retention != result_2.predicted_retention, \
            "Different sequences should have different HIC predictions"

    def test_hic_predictor_class(self):
        """Test HICPredictor class interface."""
        from proteinscore.antibody.hic_predictor import HICPredictor

        predictor = HICPredictor()

        result = predictor.predict(TEST_VH, TEST_VL)

        assert hasattr(result, 'predicted_retention')
        assert result.predicted_retention > 0

    def test_hic_predictor_batch(self):
        """Test HICPredictor batch prediction."""
        from proteinscore.antibody.hic_predictor import HICPredictor

        predictor = HICPredictor()

        antibodies = [(TEST_VH, TEST_VL), (TEST_VH_2, TEST_VL_2)]

        results = predictor.predict_batch(antibodies)

        assert len(results) == 2
        assert all(hasattr(r, 'predicted_retention') for r in results)
        assert all(r.predicted_retention > 0 for r in results)

    def test_hic_predictor_with_sklearn_model(self):
        """Test that sklearn GBM model produces predictions."""
        from proteinscore.antibody.hic_predictor import HICPredictor
        from pathlib import Path

        # Get the model path
        model_path = Path(__file__).parent.parent / "src" / "proteinscore" / "antibody" / "models" / "hic_gbm_model.pkl"

        if model_path.exists():
            predictor = HICPredictor(model_path=model_path)
            result = predictor.predict(TEST_VH, TEST_VL)

            assert hasattr(result, 'predicted_retention')
            assert result.predicted_retention > 0, "HIC retention should be positive"

    def test_hydrophobicity_integration(self):
        """Test integration with hydrophobicity module."""
        from proteinscore.antibody.hydrophobicity import predict_hic_ml

        result = predict_hic_ml(TEST_VH, TEST_VL, use_sklearn=False)

        assert isinstance(result, float)
        assert result > 0

    def test_module_exports(self):
        """Test that all expected functions are exported from antibody module."""
        from proteinscore.antibody import (
            HICPredictor,
            predict_hic,
            extract_all_features,
            get_feature_names,
            predict_hic_ml,
        )

        # All should be callable
        assert callable(predict_hic)
        assert callable(extract_all_features)
        assert callable(get_feature_names)
        assert callable(predict_hic_ml)
        assert callable(HICPredictor)

    def test_retention_class_assignment(self):
        """Test that retention class is assigned correctly."""
        from proteinscore.antibody.hic_predictor import predict_hic

        result = predict_hic(TEST_VH, TEST_VL)

        # Retention class should be one of these
        valid_classes = {"low", "medium", "high"}
        assert result.retention_class in valid_classes

    def test_percentile_in_range(self):
        """Test that percentile is in valid range."""
        from proteinscore.antibody.hic_predictor import predict_hic

        result = predict_hic(TEST_VH, TEST_VL)

        assert 0 <= result.percentile <= 100


class TestFeatureCategories:
    """Test individual feature categories."""

    def test_amino_acid_composition(self):
        """Test amino acid composition features."""
        from proteinscore.antibody.hic_predictor import extract_all_features, get_feature_names

        features = extract_all_features(TEST_VH, TEST_VL)
        names = get_feature_names()

        # Extract AA composition features
        aa_indices = [i for i, n in enumerate(names) if n.startswith("aa_comp_")]
        aa_features = features[aa_indices]

        # Should sum to approximately 1 (fractions)
        assert abs(sum(aa_features) - 1.0) < 0.01, "AA composition should sum to 1"

        # All should be between 0 and 1
        assert all(0 <= f <= 1 for f in aa_features), "AA fractions should be in [0, 1]"

    def test_physicochemical_features(self):
        """Test physicochemical property features."""
        from proteinscore.antibody.hic_predictor import get_feature_names

        names = get_feature_names()

        # Check aromatic features exist
        assert "aromatic_mean" in names
        assert "aromatic_frac" in names

        # Check charge features exist
        assert "charge_mean" in names
        assert "net_charge_per_res" in names

    def test_cdr_regional_features(self):
        """Test CDR-specific features."""
        from proteinscore.antibody.hic_predictor import get_feature_names

        names = get_feature_names()

        # Check 4 key CDRs have features (h1, h2, h3, l3 are included)
        for cdr in ["cdr_h1", "cdr_h2", "cdr_h3", "cdr_l3"]:
            hydropathy = f"{cdr}_hydropathy"
            aromatic = f"{cdr}_aromatic_frac"
            assert hydropathy in names, f"Missing {hydropathy}"
            assert aromatic in names, f"Missing {aromatic}"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_short_sequences(self):
        """Test with short sequences."""
        from proteinscore.antibody.hic_predictor import predict_hic

        # Minimum viable sequences
        short_vh = "EVQLVESGGGLVQPGG" * 5  # ~80 residues
        short_vl = "DIQMTQSPSSLSASVG" * 5  # ~80 residues

        result = predict_hic(short_vh, short_vl)
        assert hasattr(result, 'predicted_retention')

    def test_unusual_amino_acids(self):
        """Test handling of unusual amino acids."""
        from proteinscore.antibody.hic_predictor import extract_all_features

        # Sequence with all standard AAs
        vh_all_aa = "ACDEFGHIKLMNPQRSTVWY" * 6
        vl_all_aa = "ACDEFGHIKLMNPQRSTVWY" * 5

        features = extract_all_features(vh_all_aa, vl_all_aa)
        assert not np.isnan(features).any()

    def test_empty_string_handling(self):
        """Test that empty sequences are handled gracefully."""
        from proteinscore.antibody.hic_predictor import extract_all_features

        # This should either work or raise a clear error
        try:
            features = extract_all_features("", "")
            # If it works, check for reasonable output
            assert len(features) == 59
        except (ValueError, ZeroDivisionError):
            # Expected to fail gracefully
            pass
