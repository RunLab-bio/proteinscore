#!/usr/bin/env python3
"""
ProteinScore Benchmark Suite

Validates that ProteinScore correctly predicts protein developability by testing
against proteins with known experimental properties.

This benchmark demonstrates:
1. Individual predictor accuracy (stability, solubility, aggregation, immunogenicity)
2. Integrated ProteinScore scoring
3. Differentiation between good and poor developability candidates
4. Performance metrics

Reference proteins:
- Well-characterized therapeutic proteins with known developability profiles
- Proteins known to have expression/stability issues
- Human germline antibody sequences (low immunogenicity baseline)
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Add src to path for local testing
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from proteinscore import ProteinScore
from proteinscore.models import ProteinScoreResult, RiskLevel
from proteinscore.predictors import (
    AggregationPredictor,
    ImmunogenicityPredictor,
    SolubilityPredictor,
    StabilityPredictor,
)
from proteinscore.validation import validate_sequence


# =============================================================================
# Reference Proteins with Known Developability Profiles
# =============================================================================

@dataclass
class ReferenceProtein:
    """Reference protein with expected developability characteristics."""
    name: str
    sequence: str
    description: str
    expected_stability: str  # "high", "medium", "low"
    expected_solubility: str  # "high", "medium", "low"
    expected_aggregation: str  # "low", "medium", "high" (low APR = better)
    expected_immunogenicity: str  # "low", "medium", "high" (low = better for therapeutics)
    notes: str = ""


# Well-characterized reference proteins
REFERENCE_PROTEINS = [
    # Human serum albumin - highly stable, soluble, low immunogenicity (human origin)
    ReferenceProtein(
        name="HSA_fragment",
        sequence="DAHKSEVAHRFKDLGEENFKALVLIAFAQYLQQCPFEDHVKLVNEVTEFAKTCVADESAENCDKS",
        description="Human Serum Albumin fragment (Domain I partial)",
        expected_stability="high",
        expected_solubility="high",
        expected_aggregation="low",
        expected_immunogenicity="low",
        notes="Highly evolved human protein, excellent developability baseline"
    ),

    # Green Fluorescent Protein - stable but can aggregate, non-human
    ReferenceProtein(
        name="GFP_core",
        sequence="SKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTFSYGVQCFSRYPDHMKQ",
        description="Green Fluorescent Protein core barrel region",
        expected_stability="high",
        expected_solubility="medium",
        expected_aggregation="medium",
        expected_immunogenicity="medium",
        notes="Jellyfish origin, beta-barrel structure, can form aggregates"
    ),

    # Insulin B chain - small, well-studied, can aggregate (amyloid)
    ReferenceProtein(
        name="Insulin_B",
        sequence="FVNQHLCGSHLVEALYLVCGERGFFYTPKT",
        description="Human Insulin B chain",
        expected_stability="medium",
        expected_solubility="medium",
        expected_aggregation="high",  # Known to form amyloid fibrils
        expected_immunogenicity="low",
        notes="Known aggregation-prone, forms amyloid at injection sites"
    ),

    # Alpha-synuclein fragment - intrinsically disordered, highly aggregation-prone
    ReferenceProtein(
        name="aSyn_NAC",
        sequence="EQVTNVGGAVVTGVTAVAQKTVEGAGSIAAATGFVKKDQLGKNEEGAPQEGILEDMPVDPDNEAYEMPSEEGYQDYEPEA",
        description="Alpha-synuclein NAC region + flanking",
        expected_stability="low",
        expected_solubility="low",
        expected_aggregation="high",  # Parkinson's disease aggregates
        expected_immunogenicity="low",
        notes="Intrinsically disordered, highly aggregation-prone"
    ),

    # Trastuzumab heavy chain variable region - therapeutic antibody
    ReferenceProtein(
        name="Trastuzumab_VH",
        sequence="EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYADSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS",
        description="Trastuzumab (Herceptin) VH domain",
        expected_stability="high",
        expected_solubility="high",
        expected_aggregation="low",
        expected_immunogenicity="low",  # Humanized
        notes="FDA-approved therapeutic, humanized antibody"
    ),

    # Amyloid beta 42 - notoriously aggregation-prone
    ReferenceProtein(
        name="Abeta42",
        sequence="DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA",
        description="Amyloid beta 1-42 peptide",
        expected_stability="low",
        expected_solubility="low",
        expected_aggregation="high",  # Alzheimer's plaques
        expected_immunogenicity="low",
        notes="Highly aggregation-prone, forms Alzheimer's plaques"
    ),

    # Lysozyme - extremely stable enzyme
    ReferenceProtein(
        name="Lysozyme",
        sequence="KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL",
        description="Hen egg white lysozyme",
        expected_stability="high",
        expected_solubility="high",
        expected_aggregation="low",
        expected_immunogenicity="medium",  # Non-human
        notes="One of the most stable proteins known"
    ),

    # Poorly designed synthetic peptide (control - should score badly)
    ReferenceProtein(
        name="Synthetic_bad",
        sequence="WWWWWFFFFFIIIIIILLLLLLVVVVVVAAAAA",
        description="Synthetic hydrophobic peptide (negative control)",
        expected_stability="low",
        expected_solubility="low",
        expected_aggregation="high",
        expected_immunogenicity="medium",
        notes="Designed to be aggregation-prone and insoluble"
    ),
]


# =============================================================================
# Benchmark Functions
# =============================================================================

def run_validation_tests() -> dict[str, Any]:
    """Test sequence validation."""
    print("\n" + "="*60)
    print("VALIDATION TESTS")
    print("="*60)

    results = {"passed": 0, "failed": 0, "tests": []}

    # Valid sequences
    valid_sequences = [
        ("Standard amino acids", "ACDEFGHIKLMNPQRSTVWY"),
        ("Lowercase (should normalize)", "acdefghiklmnpqrstvwy"),
        ("With spaces (should strip)", "ACDEF GHIKL MNPQR STVWY"),
        ("Real protein", REFERENCE_PROTEINS[0].sequence),
    ]

    for name, seq in valid_sequences:
        try:
            validated = validate_sequence(seq)
            assert validated.isupper()
            assert " " not in validated
            print(f"  [PASS] {name}")
            results["passed"] += 1
            results["tests"].append({"name": name, "status": "pass"})
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            results["failed"] += 1
            results["tests"].append({"name": name, "status": "fail", "error": str(e)})

    # Invalid sequences
    invalid_sequences = [
        ("Empty string", ""),
        ("Too short", "AC"),
        ("Invalid characters", "ACDEFGHIKLMNPQRSTVWYX123"),
        ("Numbers only", "12345"),
    ]

    for name, seq in invalid_sequences:
        try:
            validate_sequence(seq)
            print(f"  [FAIL] {name}: Should have raised exception")
            results["failed"] += 1
            results["tests"].append({"name": name, "status": "fail", "error": "No exception raised"})
        except Exception:
            print(f"  [PASS] {name}: Correctly rejected")
            results["passed"] += 1
            results["tests"].append({"name": name, "status": "pass"})

    return results


def run_predictor_tests() -> dict[str, Any]:
    """Test individual predictors."""
    print("\n" + "="*60)
    print("INDIVIDUAL PREDICTOR TESTS")
    print("="*60)

    results = {"predictors": {}}

    # Initialize predictors
    predictors = {
        "stability": StabilityPredictor(),
        "solubility": SolubilityPredictor(),
        "aggregation": AggregationPredictor(),
        "immunogenicity": ImmunogenicityPredictor(),  # Local mode (no API key)
    }

    test_sequence = REFERENCE_PROTEINS[0].sequence  # HSA fragment

    for name, predictor in predictors.items():
        print(f"\n  Testing {name} predictor...")
        start_time = time.time()

        try:
            result = predictor.predict(test_sequence)
            elapsed = time.time() - start_time

            # Validate result structure
            assert hasattr(result, 'score'), "Result missing 'score'"
            assert 0 <= result.score <= 100, f"Score {result.score} out of range"
            assert hasattr(result, 'confidence'), "Result missing 'confidence'"

            print(f"    Score: {result.score:.1f}/100")
            print(f"    Confidence: {result.confidence:.2f}")
            print(f"    Time: {elapsed*1000:.1f}ms")
            print(f"    [PASS]")

            results["predictors"][name] = {
                "status": "pass",
                "score": result.score,
                "confidence": result.confidence,
                "time_ms": elapsed * 1000,
            }

        except Exception as e:
            print(f"    [FAIL] {e}")
            results["predictors"][name] = {
                "status": "fail",
                "error": str(e),
            }

    return results


def run_integrated_scoring_tests() -> dict[str, Any]:
    """Test integrated ProteinScore scoring on reference proteins."""
    print("\n" + "="*60)
    print("INTEGRATED PROTEINSCORE TESTS")
    print("="*60)

    results = {"proteins": [], "summary": {}}

    # Initialize ProteinScore (local mode for benchmark)
    scorer = ProteinScore(local_only=True)

    score_results = []

    for protein in REFERENCE_PROTEINS:
        print(f"\n  Scoring {protein.name}...")
        print(f"    {protein.description}")

        start_time = time.time()
        result = scorer.score(protein.sequence, name=protein.name)
        elapsed = time.time() - start_time

        # Check if scores align with expectations
        checks = {
            "stability": check_expectation(result.stability.score, protein.expected_stability),
            "solubility": check_expectation(result.solubility.score, protein.expected_solubility),
            "aggregation": check_expectation(result.aggregation.score, protein.expected_aggregation),
            "immunogenicity": check_expectation(result.immunogenicity.score, protein.expected_immunogenicity),
        }

        all_aligned = all(checks.values())

        print(f"    Total Score: {result.total_score:.1f}/100 ({result.risk_level.value})")
        print(f"    Stability: {result.stability.score:.1f} (expected: {protein.expected_stability}) {'[OK]' if checks['stability'] else '[MISMATCH]'}")
        print(f"    Solubility: {result.solubility.score:.1f} (expected: {protein.expected_solubility}) {'[OK]' if checks['solubility'] else '[MISMATCH]'}")
        print(f"    Aggregation: {result.aggregation.score:.1f} (expected: {protein.expected_aggregation}) {'[OK]' if checks['aggregation'] else '[MISMATCH]'}")
        print(f"    Immunogenicity: {result.immunogenicity.score:.1f} (expected: {protein.expected_immunogenicity}) {'[OK]' if checks['immunogenicity'] else '[MISMATCH]'}")
        print(f"    Time: {elapsed*1000:.1f}ms")
        print(f"    Recommendations: {len(result.recommendations)}")

        protein_result = {
            "name": protein.name,
            "description": protein.description,
            "total_score": result.total_score,
            "risk_level": result.risk_level.value,
            "stability": {"score": result.stability.score, "aligned": checks["stability"]},
            "solubility": {"score": result.solubility.score, "aligned": checks["solubility"]},
            "aggregation": {"score": result.aggregation.score, "aligned": checks["aggregation"]},
            "immunogenicity": {"score": result.immunogenicity.score, "aligned": checks["immunogenicity"]},
            "all_aligned": all_aligned,
            "time_ms": elapsed * 1000,
            "recommendations": len(result.recommendations),
        }

        results["proteins"].append(protein_result)
        score_results.append((protein.name, result.total_score, all_aligned))

    # Summary statistics
    total_proteins = len(REFERENCE_PROTEINS)
    aligned_count = sum(1 for p in results["proteins"] if p["all_aligned"])
    avg_time = sum(p["time_ms"] for p in results["proteins"]) / total_proteins

    results["summary"] = {
        "total_proteins": total_proteins,
        "aligned_with_expectations": aligned_count,
        "alignment_rate": aligned_count / total_proteins * 100,
        "average_time_ms": avg_time,
    }

    return results


def run_differentiation_tests() -> dict[str, Any]:
    """Test that ProteinScore can differentiate good vs bad candidates."""
    print("\n" + "="*60)
    print("DIFFERENTIATION TESTS")
    print("="*60)

    results = {"tests": [], "passed": 0, "failed": 0}

    scorer = ProteinScore(local_only=True)

    # Test pairs: (good candidate, bad candidate)
    test_pairs = [
        # Stable vs unstable
        (
            "HSA_fragment",
            "Synthetic_bad",
            "Stable human protein should score higher than synthetic aggregation-prone peptide"
        ),
        # Therapeutic antibody vs aggregation-prone peptide
        (
            "Trastuzumab_VH",
            "Abeta42",
            "Therapeutic antibody should score higher than amyloid-forming peptide"
        ),
        # Stable enzyme vs disordered protein
        (
            "Lysozyme",
            "aSyn_NAC",
            "Stable enzyme should score higher than intrinsically disordered protein"
        ),
    ]

    for good_name, bad_name, description in test_pairs:
        print(f"\n  Test: {description}")

        good_protein = next(p for p in REFERENCE_PROTEINS if p.name == good_name)
        bad_protein = next(p for p in REFERENCE_PROTEINS if p.name == bad_name)

        good_result = scorer.score(good_protein.sequence, name=good_name)
        bad_result = scorer.score(bad_protein.sequence, name=bad_name)

        passed = good_result.total_score > bad_result.total_score

        print(f"    {good_name}: {good_result.total_score:.1f}")
        print(f"    {bad_name}: {bad_result.total_score:.1f}")
        print(f"    Difference: {good_result.total_score - bad_result.total_score:.1f}")
        print(f"    {'[PASS]' if passed else '[FAIL]'}")

        results["tests"].append({
            "description": description,
            "good_candidate": good_name,
            "bad_candidate": bad_name,
            "good_score": good_result.total_score,
            "bad_score": bad_result.total_score,
            "difference": good_result.total_score - bad_result.total_score,
            "passed": passed,
        })

        if passed:
            results["passed"] += 1
        else:
            results["failed"] += 1

    return results


def run_batch_performance_tests() -> dict[str, Any]:
    """Test batch processing performance."""
    print("\n" + "="*60)
    print("BATCH PERFORMANCE TESTS")
    print("="*60)

    results = {}

    scorer = ProteinScore(local_only=True)
    sequences = [p.sequence for p in REFERENCE_PROTEINS]
    names = [p.name for p in REFERENCE_PROTEINS]

    # Sequential processing
    print("\n  Sequential processing...")
    start_time = time.time()
    sequential_results = scorer.score_batch(sequences, names=names, parallel=False)
    sequential_time = time.time() - start_time
    print(f"    Time: {sequential_time*1000:.1f}ms ({len(sequences)} proteins)")

    # Parallel processing
    print("\n  Parallel processing...")
    start_time = time.time()
    parallel_results = scorer.score_batch(sequences, names=names, parallel=True)
    parallel_time = time.time() - start_time
    print(f"    Time: {parallel_time*1000:.1f}ms ({len(sequences)} proteins)")

    speedup = sequential_time / parallel_time if parallel_time > 0 else 1.0
    print(f"\n  Speedup: {speedup:.2f}x")

    results = {
        "sequence_count": len(sequences),
        "sequential_time_ms": sequential_time * 1000,
        "parallel_time_ms": parallel_time * 1000,
        "speedup": speedup,
        "per_protein_sequential_ms": sequential_time * 1000 / len(sequences),
        "per_protein_parallel_ms": parallel_time * 1000 / len(sequences),
    }

    return results


def check_expectation(score: float, expected: str) -> bool:
    """
    Check if a score aligns with expected level.

    For most metrics, higher score = better.
    For aggregation, higher score = lower aggregation propensity = better.
    """
    if expected == "high":
        return score >= 60  # High performance
    elif expected == "medium":
        return 35 <= score <= 75  # Middle range
    elif expected == "low":
        return score <= 55  # Low performance (but allow some tolerance)
    return False


def generate_report(all_results: dict[str, Any]) -> str:
    """Generate benchmark report."""
    lines = [
        "="*70,
        "PROTEINSCORE BENCHMARK REPORT",
        f"Generated: {datetime.utcnow().isoformat()}Z",
        "="*70,
        "",
    ]

    # Validation summary
    val = all_results["validation"]
    lines.append("VALIDATION TESTS")
    lines.append(f"  Passed: {val['passed']}/{val['passed'] + val['failed']}")
    lines.append("")

    # Predictor summary
    pred = all_results["predictors"]
    lines.append("INDIVIDUAL PREDICTORS")
    for name, data in pred["predictors"].items():
        status = "[OK]" if data["status"] == "pass" else "[FAIL]"
        if data["status"] == "pass":
            lines.append(f"  {name}: {data['score']:.1f}/100 ({data['time_ms']:.1f}ms) {status}")
        else:
            lines.append(f"  {name}: {data.get('error', 'Unknown error')} {status}")
    lines.append("")

    # Integrated scoring summary
    integ = all_results["integrated"]
    lines.append("INTEGRATED SCORING")
    lines.append(f"  Proteins tested: {integ['summary']['total_proteins']}")
    lines.append(f"  Aligned with expectations: {integ['summary']['aligned_with_expectations']}/{integ['summary']['total_proteins']} ({integ['summary']['alignment_rate']:.1f}%)")
    lines.append(f"  Average time: {integ['summary']['average_time_ms']:.1f}ms/protein")
    lines.append("")

    # Score table
    lines.append("  Scores:")
    lines.append("  " + "-"*66)
    lines.append(f"  {'Protein':<20} {'Total':>6} {'Stab':>6} {'Sol':>6} {'Agg':>6} {'Imm':>6} {'Risk':<8}")
    lines.append("  " + "-"*66)
    for p in integ["proteins"]:
        lines.append(
            f"  {p['name']:<20} {p['total_score']:>6.1f} "
            f"{p['stability']['score']:>6.1f} {p['solubility']['score']:>6.1f} "
            f"{p['aggregation']['score']:>6.1f} {p['immunogenicity']['score']:>6.1f} "
            f"{p['risk_level']:<8}"
        )
    lines.append("")

    # Differentiation tests
    diff = all_results["differentiation"]
    lines.append("DIFFERENTIATION TESTS")
    lines.append(f"  Passed: {diff['passed']}/{diff['passed'] + diff['failed']}")
    for test in diff["tests"]:
        status = "[PASS]" if test["passed"] else "[FAIL]"
        lines.append(f"  {test['good_candidate']} ({test['good_score']:.1f}) > {test['bad_candidate']} ({test['bad_score']:.1f}) {status}")
    lines.append("")

    # Performance
    perf = all_results["performance"]
    lines.append("PERFORMANCE")
    lines.append(f"  Sequential: {perf['sequential_time_ms']:.1f}ms ({perf['per_protein_sequential_ms']:.1f}ms/protein)")
    lines.append(f"  Parallel: {perf['parallel_time_ms']:.1f}ms ({perf['per_protein_parallel_ms']:.1f}ms/protein)")
    lines.append(f"  Speedup: {perf['speedup']:.2f}x")
    lines.append("")

    # Final verdict
    lines.append("="*70)
    total_tests = val["passed"] + val["failed"] + diff["passed"] + diff["failed"]
    passed_tests = val["passed"] + diff["passed"]
    predictor_ok = all(p["status"] == "pass" for p in pred["predictors"].values())

    if passed_tests == total_tests and predictor_ok and integ["summary"]["alignment_rate"] >= 50:
        verdict = "PASS - ProteinScore is functioning correctly"
    else:
        verdict = "NEEDS REVIEW - Some tests did not meet expectations"

    lines.append(f"VERDICT: {verdict}")
    lines.append("="*70)

    return "\n".join(lines)


def main():
    """Run all benchmarks."""
    print("="*70)
    print("PROTEINSCORE BENCHMARK SUITE")
    print(f"Started: {datetime.utcnow().isoformat()}Z")
    print("="*70)

    all_results = {}

    # Run all test suites
    all_results["validation"] = run_validation_tests()
    all_results["predictors"] = run_predictor_tests()
    all_results["integrated"] = run_integrated_scoring_tests()
    all_results["differentiation"] = run_differentiation_tests()
    all_results["performance"] = run_batch_performance_tests()

    # Generate and print report
    report = generate_report(all_results)
    print("\n\n")
    print(report)

    # Save detailed results
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"benchmark_{timestamp}.json"
    report_path = output_dir / f"benchmark_{timestamp}.txt"

    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    with open(report_path, "w") as f:
        f.write(report)

    print(f"\nDetailed results saved to: {json_path}")
    print(f"Report saved to: {report_path}")

    # Return exit code based on results
    val = all_results["validation"]
    diff = all_results["differentiation"]
    if val["failed"] > 0 or diff["failed"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
