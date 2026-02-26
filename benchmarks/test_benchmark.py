#!/usr/bin/env python3
"""
Quick Test for ProteinScore Benchmark Suite

Tests that the benchmark infrastructure works correctly without
requiring the full ProteinGym download.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from proteinscore import ProteinScore


def test_proteinscore_basic():
    """Test basic ProteinScore functionality."""
    print("Testing ProteinScore basic functionality...")

    scorer = ProteinScore(local_only=True)

    # Test sequences with known characteristics
    test_cases = [
        # (name, sequence, expected_risk_level)
        ("Stable_HSA", "DAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLVNEVTEFAKTCVADESAENCDKS", "low_medium"),
        ("Aggregation_prone", "KLVFFAEDVGSNKGAIIGLMVGGVVIA", "high"),  # Abeta-like
        ("Charged_soluble", "DKEKEKDKEKEKDKEKEKDK", "low_medium"),
    ]

    results = []
    for name, seq, expected in test_cases:
        result = scorer.score(seq, name=name)
        results.append((name, result.total_score, result.risk_level.value))
        print(f"  {name}: {result.total_score:.1f}/100 ({result.risk_level.value})")

    print("\nBasic functionality: OK")
    return True


def test_batch_scoring():
    """Test batch scoring."""
    print("\nTesting batch scoring...")

    scorer = ProteinScore(local_only=True)

    sequences = [
        "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQV",
        "DAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLVNEVTEFAKTCVADESAENCDKS",
        "SKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQ",
    ]

    results = scorer.score_batch(sequences, parallel=True)

    print(f"  Scored {len(results)} sequences")
    for i, r in enumerate(results):
        print(f"    Seq {i+1}: {r.total_score:.1f}/100")

    assert len(results) == len(sequences), "Batch size mismatch"
    print("\nBatch scoring: OK")
    return True


def test_predictors():
    """Test individual predictors."""
    print("\nTesting individual predictors...")

    from proteinscore.predictors import (
        StabilityPredictor,
        SolubilityPredictor,
        AggregationPredictor,
        ImmunogenicityPredictor,
    )

    test_seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQV"

    predictors = {
        "Stability": StabilityPredictor(),
        "Solubility": SolubilityPredictor(),
        "Aggregation": AggregationPredictor(),
        "Immunogenicity": ImmunogenicityPredictor(),
    }

    for name, predictor in predictors.items():
        result = predictor.predict(test_seq)
        assert 0 <= result.score <= 100, f"{name} score out of range"
        print(f"  {name}: {result.score:.1f}/100 (confidence: {result.confidence:.2f})")

    print("\nIndividual predictors: OK")
    return True


def test_correlation_functions():
    """Test statistical functions."""
    print("\nTesting correlation functions...")

    # Import from benchmark module
    from benchmark_public import spearman_correlation, pearson_correlation, calculate_auc

    # Perfect positive correlation
    x = [1, 2, 3, 4, 5]
    y = [1, 2, 3, 4, 5]
    rho = spearman_correlation(x, y)
    assert abs(rho - 1.0) < 0.01, f"Expected 1.0, got {rho}"
    print(f"  Perfect positive: ρ = {rho:.3f}")

    # Perfect negative correlation
    y_neg = [5, 4, 3, 2, 1]
    rho_neg = spearman_correlation(x, y_neg)
    assert abs(rho_neg - (-1.0)) < 0.01, f"Expected -1.0, got {rho_neg}"
    print(f"  Perfect negative: ρ = {rho_neg:.3f}")

    # AUC for perfect classifier
    preds = [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]
    labels = [True, True, True, False, False, False]
    auc = calculate_auc(preds, labels)
    assert abs(auc - 1.0) < 0.01, f"Expected AUC 1.0, got {auc}"
    print(f"  Perfect AUC: {auc:.3f}")

    print("\nCorrelation functions: OK")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("ProteinScore Benchmark Quick Test")
    print("=" * 60)

    tests = [
        test_proteinscore_basic,
        test_batch_scoring,
        test_predictors,
        test_correlation_functions,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n  ERROR: {e}")
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed > 0:
        print("\n❌ Some tests failed")
        return 1

    print("\n✅ All tests passed - Benchmark suite is ready")
    print("\nNext steps:")
    print("  1. Download ProteinGym: bash benchmarks/download_proteingym.sh")
    print("  2. Run full benchmark: python benchmarks/proteingym_benchmark.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
