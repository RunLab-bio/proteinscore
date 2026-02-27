#!/usr/bin/env python3
"""
ProteinScore Library Validation Benchmark

Comprehensive validation of all library modules:
- Core predictors (stability, solubility, aggregation, immunogenicity)
- Enzyme module (thermostability, expression, scorer)
- Peptide module (stability, chemical liability, scorer)
- Antibody module (CDR, TAP, liabilities, HIC, scorer)

Usage:
    python benchmarks/validate_library.py
"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@dataclass
class ValidationResult:
    """Result of a validation test."""
    module: str
    component: str
    test_name: str
    passed: bool
    details: str
    execution_time: float


# =============================================================================
# Test Sequences
# =============================================================================

# Well-characterized protein sequences
GFP = (
    "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLV"
    "TTLTYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNR"
    "IELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQ"
    "QNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITLGMDELYK"
)

T4_LYSOZYME = (
    "MNIFEMLRIDEGLRLKIYKDTEGYYTIGIGHLLTKSPSLNAAKSELDKAIGRNTNGVITKD"
    "EAEKLFNQDVDAAVRGILRNAKLKPVYDSLDAVRRAALINMVFQMGETGVAGFTNSLRMLQ"
    "QKRWDEAAVNLAKSRWYNQTPNRAKRVITTFRTGTWDAYKNL"
)

INSULIN_B = "FVNQHLCGSHLVEALYLVCGERGFFYTPKT"

# Therapeutic peptide - GLP-1 analog
SEMAGLUTIDE = "HXEGTFTSDVSSYLEGQAAKEFIAWLVKGRG"

# Antibody sequences (Herceptin/Trastuzumab)
HERCEPTIN_VH = (
    "EVQLVESGGGLVQPGGSLRLSCAASGFNIKDTYIHWVRQAPGKGLEWVARIYPTNGYTRYA"
    "DSVKGRFTISADTSKNTAYLQMNSLRAEDTAVYYCSRWGGDGFYAMDYWGQGTLVTVSS"
)

HERCEPTIN_VL = (
    "DIQMTQSPSSLSASVGDRVTITCRASQDVNTAVAWYQQKPGKAPKLLIYSASFLYSGVPSR"
    "FSGSRSGTDFTLTISSLQPEDFATYYCQQHYTTPPTFGQGTKVEIK"
)


def run_validation(func, *args, **kwargs) -> tuple:
    """Run a validation function and return (result, time, error)."""
    start = time.perf_counter()
    try:
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        return result, elapsed, None
    except Exception as e:
        elapsed = time.perf_counter() - start
        return None, elapsed, str(e)


def main():
    print("=" * 80)
    print("ProteinScore Library Validation")
    print("=" * 80)

    results: list[ValidationResult] = []

    # =========================================================================
    # CORE PREDICTORS
    # =========================================================================
    print("\n" + "=" * 80)
    print("CORE PREDICTORS")
    print("=" * 80)

    # --- Stability Predictor ---
    print("\n" + "-" * 60)
    print("StabilityPredictor")
    print("-" * 60)

    from proteinscore.predictors import StabilityPredictor

    predictor = StabilityPredictor()
    result, elapsed, error = run_validation(predictor.predict, GFP)

    if error:
        print(f"  ❌ predict(): {error}")
        results.append(ValidationResult("core", "StabilityPredictor", "predict", False, error, elapsed))
    else:
        passed = (
            0 <= result.score <= 100 and
            0 <= result.confidence <= 1 and
            result.method is not None
        )
        print(f"  ✅ predict(): score={result.score:.1f}, confidence={result.confidence:.2f}")
        print(f"     Method: {result.method}")
        print(f"     Interpretation: {result.interpretation[:50]}...")
        results.append(ValidationResult("core", "StabilityPredictor", "predict", passed, f"score={result.score:.1f}", elapsed))

    # --- Solubility Predictor ---
    print("\n" + "-" * 60)
    print("SolubilityPredictor")
    print("-" * 60)

    from proteinscore.predictors import SolubilityPredictor

    predictor = SolubilityPredictor()
    result, elapsed, error = run_validation(predictor.predict, GFP)

    if error:
        print(f"  ❌ predict(): {error}")
        results.append(ValidationResult("core", "SolubilityPredictor", "predict", False, error, elapsed))
    else:
        passed = (
            0 <= result.score <= 100 and
            0 <= result.confidence <= 1 and
            result.solubility_class is not None
        )
        print(f"  ✅ predict(): score={result.score:.1f}, class={result.solubility_class.value}")
        print(f"     Confidence: {result.confidence:.2f}")
        results.append(ValidationResult("core", "SolubilityPredictor", "predict", passed, f"score={result.score:.1f}", elapsed))

    # --- Aggregation Predictor ---
    print("\n" + "-" * 60)
    print("AggregationPredictor")
    print("-" * 60)

    from proteinscore.predictors import AggregationPredictor

    predictor = AggregationPredictor()
    result, elapsed, error = run_validation(predictor.predict, GFP)

    if error:
        print(f"  ❌ predict(): {error}")
        results.append(ValidationResult("core", "AggregationPredictor", "predict", False, error, elapsed))
    else:
        passed = (
            0 <= result.score <= 100 and
            0 <= result.confidence <= 1
        )
        print(f"  ✅ predict(): score={result.score:.1f}, confidence={result.confidence:.2f}")
        print(f"     APR regions: {len(result.aggregation_prone_regions)}")
        results.append(ValidationResult("core", "AggregationPredictor", "predict", passed, f"score={result.score:.1f}", elapsed))

    # --- Immunogenicity Predictor ---
    print("\n" + "-" * 60)
    print("ImmunogenicityPredictor")
    print("-" * 60)

    from proteinscore.predictors import ImmunogenicityPredictor

    predictor = ImmunogenicityPredictor(api_key=None)  # Local mode
    result, elapsed, error = run_validation(predictor.predict, GFP)

    if error:
        print(f"  ❌ predict(): {error}")
        results.append(ValidationResult("core", "ImmunogenicityPredictor", "predict", False, error, elapsed))
    else:
        passed = (
            0 <= result.score <= 100 and
            0 <= result.confidence <= 1
        )
        print(f"  ✅ predict(): score={result.score:.1f}, confidence={result.confidence:.2f}")
        print(f"     Epitopes: {len(result.epitopes)}")
        results.append(ValidationResult("core", "ImmunogenicityPredictor", "predict", passed, f"score={result.score:.1f}", elapsed))

    # --- ProteinScore (Unified Scorer) ---
    print("\n" + "-" * 60)
    print("ProteinScore (Unified)")
    print("-" * 60)

    from proteinscore import ProteinScore

    scorer = ProteinScore(local_only=True)
    result, elapsed, error = run_validation(scorer.score, GFP)

    if error:
        print(f"  ❌ score(): {error}")
        results.append(ValidationResult("core", "ProteinScore", "score", False, error, elapsed))
    else:
        passed = (
            0 <= result.total_score <= 100 and
            result.stability is not None and
            result.solubility is not None
        )
        print(f"  ✅ score(): total={result.total_score:.1f}")
        print(f"     Stability: {result.stability.score:.1f}")
        print(f"     Solubility: {result.solubility.score:.1f}")
        print(f"     Aggregation: {result.aggregation.score:.1f}")
        print(f"     Immunogenicity: {result.immunogenicity.score:.1f}")
        results.append(ValidationResult("core", "ProteinScore", "score", passed, f"total={result.total_score:.1f}", elapsed))

    # =========================================================================
    # ENZYME MODULE
    # =========================================================================
    print("\n" + "=" * 80)
    print("ENZYME MODULE")
    print("=" * 80)

    # --- Thermostability Predictor ---
    print("\n" + "-" * 60)
    print("EnzymeThermostabilityPredictor")
    print("-" * 60)

    from proteinscore.enzyme import EnzymeThermostabilityPredictor

    predictor = EnzymeThermostabilityPredictor()
    result, elapsed, error = run_validation(predictor.predict, T4_LYSOZYME)

    if error:
        print(f"  ❌ predict(): {error}")
        results.append(ValidationResult("enzyme", "ThermostabilityPredictor", "predict", False, error, elapsed))
    else:
        passed = (
            0 <= result.score <= 100 and
            20 <= result.tm_estimate <= 120  # Reasonable Tm range
        )
        print(f"  ✅ predict(): score={result.score:.1f}, Tm={result.tm_estimate:.1f}°C")
        print(f"     Class: {result.thermostability_class}")
        print(f"     Confidence: [{result.tm_confidence_interval[0]:.1f}, {result.tm_confidence_interval[1]:.1f}]")
        results.append(ValidationResult("enzyme", "ThermostabilityPredictor", "predict", passed, f"Tm={result.tm_estimate:.1f}°C", elapsed))

    # --- Expression Predictor ---
    print("\n" + "-" * 60)
    print("ExpressionPredictor")
    print("-" * 60)

    from proteinscore.enzyme import ExpressionPredictor, ExpressionHost

    predictor = ExpressionPredictor()
    result, elapsed, error = run_validation(predictor.predict, T4_LYSOZYME, ExpressionHost.E_COLI)

    if error:
        print(f"  ❌ predict(): {error}")
        results.append(ValidationResult("enzyme", "ExpressionPredictor", "predict", False, error, elapsed))
    else:
        passed = (
            0 <= result.score <= 100 and
            result.host == ExpressionHost.E_COLI
        )
        print(f"  ✅ predict(): score={result.score:.1f}, host={result.host.value}")
        print(f"     Expression level: {result.expression_level}")
        print(f"     Solubility score: {result.solubility_score:.1f}")
        results.append(ValidationResult("enzyme", "ExpressionPredictor", "predict", passed, f"score={result.score:.1f}", elapsed))

    # --- Enzyme Scorer ---
    print("\n" + "-" * 60)
    print("EnzymeScorer")
    print("-" * 60)

    from proteinscore.enzyme import EnzymeScorer

    scorer = EnzymeScorer()
    result, elapsed, error = run_validation(scorer.score, T4_LYSOZYME, name="T4 Lysozyme")

    if error:
        print(f"  ❌ score(): {error}")
        results.append(ValidationResult("enzyme", "EnzymeScorer", "score", False, error, elapsed))
    else:
        passed = (
            0 <= result.total_score <= 100 and
            result.thermostability is not None and
            result.expression is not None
        )
        print(f"  ✅ score(): total={result.total_score:.1f}")
        print(f"     Thermostability: {result.thermostability.score:.1f}")
        print(f"     Expression: {result.expression.score:.1f}")
        print(f"     Weakest: {result.weakest_component}")
        results.append(ValidationResult("enzyme", "EnzymeScorer", "score", passed, f"total={result.total_score:.1f}", elapsed))

    # =========================================================================
    # PEPTIDE MODULE
    # =========================================================================
    print("\n" + "=" * 80)
    print("PEPTIDE MODULE")
    print("=" * 80)

    # --- Peptide Stability Predictor ---
    print("\n" + "-" * 60)
    print("PeptideStabilityPredictor")
    print("-" * 60)

    from proteinscore.peptide import PeptideStabilityPredictor

    predictor = PeptideStabilityPredictor()
    result, elapsed, error = run_validation(predictor.predict, INSULIN_B)

    if error:
        print(f"  ❌ predict(): {error}")
        results.append(ValidationResult("peptide", "PeptideStabilityPredictor", "predict", False, error, elapsed))
    else:
        passed = (
            0 <= result.score <= 100 and
            result.total_cleavage_sites >= 0
        )
        print(f"  ✅ predict(): score={result.score:.1f}")
        print(f"     Cleavage sites: {result.total_cleavage_sites} (high risk: {result.high_risk_sites})")
        print(f"     Estimated half-life: {result.estimated_half_life_category}")
        results.append(ValidationResult("peptide", "PeptideStabilityPredictor", "predict", passed, f"score={result.score:.1f}", elapsed))

    # --- Chemical Liability Scanner ---
    print("\n" + "-" * 60)
    print("ChemicalLiabilityScanner")
    print("-" * 60)

    from proteinscore.peptide import ChemicalLiabilityScanner

    scanner = ChemicalLiabilityScanner()
    result, elapsed, error = run_validation(scanner.scan, INSULIN_B)

    if error:
        print(f"  ❌ scan(): {error}")
        results.append(ValidationResult("peptide", "ChemicalLiabilityScanner", "scan", False, error, elapsed))
    else:
        passed = (
            0 <= result.score <= 100 and
            result.total_liabilities >= 0
        )
        print(f"  ✅ scan(): score={result.score:.1f}")
        print(f"     Liabilities: {result.total_liabilities} (high risk: {result.high_risk_sites})")
        print(f"     Oxidation: {result.oxidation_sites}, Deamidation: {result.deamidation_sites}")
        results.append(ValidationResult("peptide", "ChemicalLiabilityScanner", "scan", passed, f"score={result.score:.1f}", elapsed))

    # --- Peptide Scorer ---
    print("\n" + "-" * 60)
    print("PeptideScorer")
    print("-" * 60)

    from proteinscore.peptide import PeptideScorer

    scorer = PeptideScorer()
    result, elapsed, error = run_validation(scorer.score, INSULIN_B)

    if error:
        print(f"  ❌ score(): {error}")
        results.append(ValidationResult("peptide", "PeptideScorer", "score", False, error, elapsed))
    else:
        passed = (
            0 <= result.total_score <= 100 and
            result.stability is not None
        )
        print(f"  ✅ score(): total={result.total_score:.1f}")
        print(f"     Proteolytic stability: {result.stability.score:.1f}")
        print(f"     Chemical liability: {result.chemical_liability.score:.1f}")
        results.append(ValidationResult("peptide", "PeptideScorer", "score", passed, f"total={result.total_score:.1f}", elapsed))

    # =========================================================================
    # ANTIBODY MODULE
    # =========================================================================
    print("\n" + "=" * 80)
    print("ANTIBODY MODULE")
    print("=" * 80)

    # --- CDR Detector ---
    print("\n" + "-" * 60)
    print("CDRDetector")
    print("-" * 60)

    from proteinscore.antibody import CDRDetector

    detector = CDRDetector()
    result, elapsed, error = run_validation(detector.detect_cdrs, HERCEPTIN_VH, chain_type="heavy")

    if error:
        print(f"  ❌ detect(): {error}")
        results.append(ValidationResult("antibody", "CDRDetector", "detect", False, error, elapsed))
    else:
        cdr_seqs = result.cdr_sequences
        passed = len(result.cdrs) >= 3
        print(f"  ✅ detect_cdrs(): {len(result.cdrs)} CDRs detected")
        for cdr in result.cdrs:
            print(f"     {cdr.name}: {cdr.sequence[:15]}...")
        results.append(ValidationResult("antibody", "CDRDetector", "detect_cdrs", passed, f"{len(result.cdrs)} CDRs", elapsed))

    # --- TAP Metrics ---
    print("\n" + "-" * 60)
    print("TAPMetrics")
    print("-" * 60)

    from proteinscore.antibody import TAPMetrics

    tap = TAPMetrics()
    result, elapsed, error = run_validation(tap.calculate, HERCEPTIN_VH, HERCEPTIN_VL)

    if error:
        print(f"  ❌ calculate(): {error}")
        results.append(ValidationResult("antibody", "TAPMetrics", "calculate", False, error, elapsed))
    else:
        passed = (
            0 <= result.developability_score <= 100 and
            result.psh is not None
        )
        print(f"  ✅ calculate(): developability={result.developability_score:.1f}")
        print(f"     PSH: {result.psh.value:.2f}, PPC: {result.ppc.value:.2f}, PNC: {result.pnc.value:.2f}")
        print(f"     Flags: {result.n_flags} (red: {result.n_red_flags})")
        results.append(ValidationResult("antibody", "TAPMetrics", "calculate", passed, f"score={result.developability_score:.1f}", elapsed))

    # --- Liability Scanner ---
    print("\n" + "-" * 60)
    print("LiabilityScanner")
    print("-" * 60)

    from proteinscore.antibody import scan_antibody

    result, elapsed, error = run_validation(scan_antibody, HERCEPTIN_VH, HERCEPTIN_VL)

    if error:
        print(f"  ❌ scan_antibody(): {error}")
        results.append(ValidationResult("antibody", "LiabilityScanner", "scan_antibody", False, error, elapsed))
    else:
        passed = (
            0 <= result.liability_score <= 100 and
            result.liabilities is not None
        )
        print(f"  ✅ scan_antibody(): score={result.liability_score:.1f}")
        print(f"     Liabilities found: {len(result.liabilities)}")
        print(f"     High severity: {result.summary.high_severity}, CDR: {result.summary.cdr_liabilities}")
        results.append(ValidationResult("antibody", "LiabilityScanner", "scan_antibody", passed, f"score={result.liability_score:.1f}", elapsed))

    # --- HIC Predictor ---
    print("\n" + "-" * 60)
    print("HICPredictor")
    print("-" * 60)

    from proteinscore.antibody import HICPredictor, predict_hic

    predictor = HICPredictor()
    result, elapsed, error = run_validation(predictor.predict, HERCEPTIN_VH, HERCEPTIN_VL)

    if error:
        print(f"  ❌ predict(): {error}")
        results.append(ValidationResult("antibody", "HICPredictor", "predict", False, error, elapsed))
    else:
        passed = (
            result.predicted_retention > 0 and
            0 <= result.percentile <= 100 and
            result.retention_class in ("low", "medium", "high")
        )
        print(f"  ✅ predict(): retention={result.predicted_retention:.2f}, class={result.retention_class}")
        print(f"     Percentile: {result.percentile:.1f}")
        results.append(ValidationResult("antibody", "HICPredictor", "predict", passed, f"retention={result.predicted_retention:.2f}", elapsed))

    # Test convenience function
    result2, elapsed2, error2 = run_validation(predict_hic, HERCEPTIN_VH, HERCEPTIN_VL)
    if error2:
        print(f"  ❌ predict_hic(): {error2}")
        results.append(ValidationResult("antibody", "HICPredictor", "predict_hic", False, error2, elapsed2))
    else:
        print(f"  ✅ predict_hic(): retention={result2.predicted_retention:.2f}")
        results.append(ValidationResult("antibody", "HICPredictor", "predict_hic", True, f"retention={result2.predicted_retention:.2f}", elapsed2))

    # --- Hydrophobicity Analysis ---
    print("\n" + "-" * 60)
    print("Hydrophobicity Analysis")
    print("-" * 60)

    from proteinscore.antibody import analyze_antibody_hydrophobicity, predict_hic_retention, get_combined_hic_score

    result, elapsed, error = run_validation(analyze_antibody_hydrophobicity, HERCEPTIN_VH, HERCEPTIN_VL)

    if error:
        print(f"  ❌ analyze_antibody_hydrophobicity(): {error}")
        results.append(ValidationResult("antibody", "Hydrophobicity", "analyze", False, error, elapsed))
    else:
        passed = "heavy" in result and "light" in result
        vh_analysis = result["heavy"]
        vl_analysis = result["light"]
        combined_hic = get_combined_hic_score(vh_analysis, vl_analysis)
        print(f"  ✅ analyze_antibody_hydrophobicity()")
        print(f"     Heavy chain HIC: {vh_analysis.hic_prediction.hic_score:.2f}")
        print(f"     Light chain HIC: {vl_analysis.hic_prediction.hic_score:.2f}")
        print(f"     Combined HIC: {combined_hic:.2f}")
        results.append(ValidationResult("antibody", "Hydrophobicity", "analyze", passed, f"HIC={combined_hic:.2f}", elapsed))

    # --- Antibody Scorer ---
    print("\n" + "-" * 60)
    print("AntibodyScorer")
    print("-" * 60)

    from proteinscore.antibody import AntibodyScorer

    scorer = AntibodyScorer()
    result, elapsed, error = run_validation(scorer.score, HERCEPTIN_VH, HERCEPTIN_VL)

    if error:
        print(f"  ❌ score(): {error}")
        results.append(ValidationResult("antibody", "AntibodyScorer", "score", False, error, elapsed))
    else:
        passed = (
            0 <= result.total_score <= 100 and
            result.tap_result is not None
        )
        print(f"  ✅ score(): total={result.total_score:.1f}")
        print(f"     TAP score: {result.tap_score:.1f}")
        print(f"     Liability score: {result.liability_score:.1f}")
        print(f"     HIC proxy: {result.hic_proxy:.2f}")
        results.append(ValidationResult("antibody", "AntibodyScorer", "score", passed, f"total={result.total_score:.1f}", elapsed))

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    # Group by module
    modules = {}
    for r in results:
        if r.module not in modules:
            modules[r.module] = {"passed": 0, "failed": 0, "total_time": 0}
        modules[r.module]["total_time"] += r.execution_time
        if r.passed:
            modules[r.module]["passed"] += 1
        else:
            modules[r.module]["failed"] += 1

    print("\nBy Module:")
    print("-" * 60)
    total_passed = 0
    total_failed = 0

    for module, stats in modules.items():
        status = "✅" if stats["failed"] == 0 else "⚠️"
        print(f"  {status} {module.upper()}: {stats['passed']}/{stats['passed']+stats['failed']} passed ({stats['total_time']*1000:.0f}ms)")
        total_passed += stats["passed"]
        total_failed += stats["failed"]

    print("\n" + "-" * 60)
    print(f"TOTAL: {total_passed}/{total_passed+total_failed} passed")

    # Failed tests details
    failed = [r for r in results if not r.passed]
    if failed:
        print("\n⚠️ FAILED TESTS:")
        for r in failed:
            print(f"  - {r.module}/{r.component}/{r.test_name}: {r.details}")

    # Performance summary
    print("\n" + "-" * 60)
    print("Performance (execution time):")
    for r in sorted(results, key=lambda x: -x.execution_time)[:5]:
        print(f"  {r.module}/{r.component}: {r.execution_time*1000:.0f}ms")

    print("\n" + "=" * 80)

    if total_failed == 0:
        print("✅ ALL VALIDATIONS PASSED")
        return 0
    else:
        print(f"⚠️ {total_failed} VALIDATION(S) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
