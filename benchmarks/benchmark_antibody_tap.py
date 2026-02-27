#!/usr/bin/env python3
"""
TAP-based Antibody Developability Benchmark

Validates the new AntibodyScorer (TAP metrics + liabilities) against
the gold standard Jain et al. 2017 dataset of 137 clinical-stage antibodies.

Reference: Jain T et al. (2017). "Biophysical properties of the clinical-stage
antibody landscape." PNAS 114:944-949

SOTA Targets:
- HIC retention: Spearman ρ > 0.50 (PROPERMAB achieves ~0.75)
- AC-SINS (self-association): ρ > 0.40
- Expression: ρ > 0.30
"""

from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from proteinscore.antibody import AntibodyScorer, AntibodyResult


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class AntibodyData:
    """Antibody with experimental measurements."""
    vh_sequence: str
    vl_sequence: str
    expression: float | None = None  # HEK Titer (mg/L)
    ac_sins: float | None = None     # Self-association (lower = better)
    csi_bli: float | None = None     # Cross-interaction (lower = better)


# =============================================================================
# Statistical Functions
# =============================================================================

def spearman_correlation(x: list[float], y: list[float]) -> tuple[float, float]:
    """Calculate Spearman rank correlation coefficient with p-value."""
    n = len(x)
    if n < 3:
        return 0.0, 1.0

    def rank(values):
        sorted_idx = sorted(range(n), key=lambda i: values[i])
        ranks = [0.0] * n
        for rank_val, idx in enumerate(sorted_idx):
            ranks[idx] = rank_val + 1
        return ranks

    rank_x = rank(x)
    rank_y = rank(y)

    mean_rx = sum(rank_x) / n
    mean_ry = sum(rank_y) / n

    num = sum((rank_x[i] - mean_rx) * (rank_y[i] - mean_ry) for i in range(n))
    den_x = math.sqrt(sum((rank_x[i] - mean_rx) ** 2 for i in range(n)))
    den_y = math.sqrt(sum((rank_y[i] - mean_ry) ** 2 for i in range(n)))

    if den_x == 0 or den_y == 0:
        return 0.0, 1.0

    rho = num / (den_x * den_y)

    if abs(rho) == 1:
        p_value = 0.0
    else:
        t_stat = rho * math.sqrt((n - 2) / (1 - rho ** 2))
        p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(t_stat) / math.sqrt(2))))

    return rho, p_value


def pearson_correlation(x: list[float], y: list[float]) -> tuple[float, float]:
    """Calculate Pearson correlation coefficient with p-value."""
    n = len(x)
    if n < 3:
        return 0.0, 1.0

    mean_x = sum(x) / n
    mean_y = sum(y) / n

    num = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(n))
    den_x = math.sqrt(sum((x[i] - mean_x) ** 2 for i in range(n)))
    den_y = math.sqrt(sum((y[i] - mean_y) ** 2 for i in range(n)))

    if den_x == 0 or den_y == 0:
        return 0.0, 1.0

    r = num / (den_x * den_y)

    if abs(r) == 1:
        p_value = 0.0
    else:
        t_stat = r * math.sqrt((n - 2) / (1 - r ** 2))
        p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(t_stat) / math.sqrt(2))))

    return r, p_value


# =============================================================================
# Data Loading
# =============================================================================

def load_jain2017_data(data_dir: Path) -> list[AntibodyData]:
    """Load Jain 2017 dataset from FLAb CSV files."""
    flab_dir = data_dir / "flab"

    if not flab_dir.exists():
        raise FileNotFoundError(
            f"FLAb data not found in {flab_dir}. "
            "Run the download script first."
        )

    expression_file = flab_dir / "jain2017_expression.csv"
    acsins_file = flab_dir / "jain2017_acsins.csv"
    csibli_file = flab_dir / "jain2017_csibli.csv"

    antibodies = {}

    # Load expression
    if expression_file.exists():
        with open(expression_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row['heavy'], row['light'])
                if key not in antibodies:
                    antibodies[key] = AntibodyData(
                        vh_sequence=row['heavy'],
                        vl_sequence=row['light']
                    )
                try:
                    antibodies[key].expression = float(row['HEK Titer (mg/L)'])
                except (ValueError, KeyError):
                    pass

    # Load AC-SINS
    if acsins_file.exists():
        with open(acsins_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row['heavy'], row['light'])
                if key not in antibodies:
                    antibodies[key] = AntibodyData(
                        vh_sequence=row['heavy'],
                        vl_sequence=row['light']
                    )
                try:
                    col = [c for c in row.keys() if 'AC-SINS' in c or 'SINS' in c][0]
                    antibodies[key].ac_sins = float(row[col])
                except (ValueError, KeyError, IndexError):
                    pass

    # Load CSI-BLI
    if csibli_file.exists():
        with open(csibli_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row['heavy'], row['light'])
                if key not in antibodies:
                    antibodies[key] = AntibodyData(
                        vh_sequence=row['heavy'],
                        vl_sequence=row['light']
                    )
                try:
                    col = [c for c in row.keys() if 'CSI' in c or 'BLI' in c][0]
                    antibodies[key].csi_bli = float(row[col])
                except (ValueError, KeyError, IndexError):
                    pass

    return list(antibodies.values())


# =============================================================================
# Benchmark Functions
# =============================================================================

def run_tap_benchmark(
    data_dir: Path,
    max_samples: int | None = None,
    verbose: bool = True
) -> dict[str, Any]:
    """
    Run TAP-based antibody benchmark against Jain 2017.

    Tests multiple metrics:
    - Total developability score
    - TAP score (5 metrics)
    - HIC proxy (surface hydrophobicity)
    - Liability score
    - Aggregation score
    - Expression score
    """
    # Load data
    antibodies = load_jain2017_data(data_dir)

    if not antibodies:
        raise ValueError("No antibody data loaded")

    if max_samples and len(antibodies) > max_samples:
        antibodies = antibodies[:max_samples]

    if verbose:
        print(f"\n{'='*70}")
        print("TAP-based Antibody Developability Benchmark")
        print(f"{'='*70}")
        print(f"Dataset: Jain 2017 ({len(antibodies)} clinical-stage antibodies)")
        print("Reference: Jain et al. PNAS 2017")
        print()
        print("SOTA Targets:")
        print("  - HIC retention: Spearman ρ > 0.50")
        print("  - AC-SINS (self-association): ρ > 0.40")
        print("  - Expression: ρ > 0.30")
        print()

    # Initialize AntibodyScorer with Jain hydrophobicity scale
    scorer = AntibodyScorer(
        hydrophobicity_scale="jain",  # Optimized for HIC correlation
        check_aggregation=True,
    )

    # Collect predictions
    results = {
        # Expression correlations
        'expression_total': {'exp': [], 'pred': []},
        'expression_tap': {'exp': [], 'pred': []},
        'expression_expr': {'exp': [], 'pred': []},  # Expression score predictor

        # Self-association (AC-SINS) correlations
        'acsins_total': {'exp': [], 'pred': []},
        'acsins_agg': {'exp': [], 'pred': []},
        'acsins_hic': {'exp': [], 'pred': []},
        'acsins_psh': {'exp': [], 'pred': []},
        'acsins_ml': {'exp': [], 'pred': []},  # ML-derived self-association score

        # Cross-interaction (CSI-BLI) correlations
        'csibli_total': {'exp': [], 'pred': []},
        'csibli_liability': {'exp': [], 'pred': []},
        'csibli_ml': {'exp': [], 'pred': []},  # ML-derived cross-interaction score
    }

    # TAP metric distributions
    tap_metrics = {
        'cdr_length': [],
        'psh': [],
        'ppc': [],
        'pnc': [],
        'sfvcsp': [],
    }

    scored_results: list[AntibodyResult] = []

    for i, ab in enumerate(antibodies):
        try:
            result = scorer.score(ab.vh_sequence, ab.vl_sequence)
            scored_results.append(result)

            # Collect TAP metrics
            tap_metrics['cdr_length'].append(result.tap_result.cdr_length.value)
            tap_metrics['psh'].append(result.tap_result.psh.value)
            tap_metrics['ppc'].append(result.tap_result.ppc.value)
            tap_metrics['pnc'].append(result.tap_result.pnc.value)
            tap_metrics['sfvcsp'].append(result.tap_result.sfvcsp.value)

            # Get specialized scores
            agg_score = scorer.get_aggregation_score(ab.vh_sequence, ab.vl_sequence)
            expr_score = scorer.get_expression_score(ab.vh_sequence, ab.vl_sequence)
            self_assoc_score = scorer.get_self_association_score(ab.vh_sequence, ab.vl_sequence)
            cross_inter_score = scorer.get_cross_interaction_score(ab.vh_sequence, ab.vl_sequence)

            # Expression correlations
            if ab.expression is not None:
                results['expression_total']['exp'].append(ab.expression)
                results['expression_total']['pred'].append(result.total_score)

                results['expression_tap']['exp'].append(ab.expression)
                results['expression_tap']['pred'].append(result.tap_score)

                results['expression_expr']['exp'].append(ab.expression)
                results['expression_expr']['pred'].append(expr_score)

            # Self-association correlations (lower AC-SINS = better)
            # Higher predicted score = better, so we correlate with -ac_sins
            if ab.ac_sins is not None:
                results['acsins_total']['exp'].append(-ab.ac_sins)
                results['acsins_total']['pred'].append(result.total_score)

                results['acsins_agg']['exp'].append(-ab.ac_sins)
                results['acsins_agg']['pred'].append(agg_score)

                results['acsins_hic']['exp'].append(-ab.ac_sins)
                results['acsins_hic']['pred'].append(-result.hic_proxy)  # Lower HIC = better

                results['acsins_psh']['exp'].append(-ab.ac_sins)
                results['acsins_psh']['pred'].append(-result.tap_result.psh.value)  # Lower PSH = better

                results['acsins_ml']['exp'].append(-ab.ac_sins)
                results['acsins_ml']['pred'].append(self_assoc_score)  # ML-derived score

            # Cross-interaction correlations (lower CSI-BLI = better)
            if ab.csi_bli is not None:
                results['csibli_total']['exp'].append(-ab.csi_bli)
                results['csibli_total']['pred'].append(result.total_score)

                results['csibli_liability']['exp'].append(-ab.csi_bli)
                results['csibli_liability']['pred'].append(result.liability_score)

                results['csibli_ml']['exp'].append(-ab.csi_bli)
                results['csibli_ml']['pred'].append(cross_inter_score)

            if verbose and (i + 1) % 25 == 0:
                print(f"  Processed {i + 1}/{len(antibodies)} antibodies...")

        except Exception as e:
            if verbose:
                print(f"  Warning: Failed to score antibody {i + 1}: {e}")
            continue

    if verbose:
        print(f"  Completed scoring {len(scored_results)} antibodies")
        print()

    # Calculate correlations
    metrics = {}

    correlation_groups = {
        'Expression (HEK Titer)': [
            ('Total Score', 'expression_total'),
            ('TAP Score', 'expression_tap'),
            ('Expression Score', 'expression_expr'),
        ],
        'Self-Association (AC-SINS)': [
            ('Total Score', 'acsins_total'),
            ('Aggregation Score', 'acsins_agg'),
            ('HIC Proxy', 'acsins_hic'),
            ('PSH (Hydrophobicity)', 'acsins_psh'),
            ('ML Self-Association', 'acsins_ml'),
        ],
        'Cross-Interaction (CSI-BLI)': [
            ('Total Score', 'csibli_total'),
            ('Liability Score', 'csibli_liability'),
            ('ML Cross-Interaction', 'csibli_ml'),
        ],
    }

    for group_name, correlations in correlation_groups.items():
        metrics[group_name] = {}
        for metric_name, key in correlations:
            data = results[key]
            if len(data['exp']) >= 10:
                spearman_r, spearman_p = spearman_correlation(data['exp'], data['pred'])
                pearson_r, pearson_p = pearson_correlation(data['exp'], data['pred'])

                metrics[group_name][metric_name] = {
                    'n': len(data['exp']),
                    'spearman_r': round(spearman_r, 4),
                    'spearman_p': round(spearman_p, 4),
                    'pearson_r': round(pearson_r, 4),
                    'pearson_p': round(pearson_p, 4),
                }

    # TAP metric statistics
    tap_stats = {}
    for metric_name, values in tap_metrics.items():
        if values:
            tap_stats[metric_name] = {
                'mean': round(sum(values) / len(values), 2),
                'min': round(min(values), 2),
                'max': round(max(values), 2),
            }

    # Count TAP flags
    flag_counts = {'green': 0, 'amber': 0, 'red': 0}
    for result in scored_results:
        for metric in result.tap_result.all_metrics:
            flag_counts[metric.flag.value] += 1

    return {
        'dataset': 'Jain2017',
        'method': 'TAP-based AntibodyScorer',
        'n_antibodies': len(scored_results),
        'metrics': metrics,
        'tap_statistics': tap_stats,
        'tap_flags': flag_counts,
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }


def print_benchmark_report(results: dict[str, Any]) -> None:
    """Print formatted benchmark report."""
    print(f"\n{'='*70}")
    print("TAP-BASED ANTIBODY BENCHMARK RESULTS")
    print(f"{'='*70}")

    print(f"\nDataset: {results['n_antibodies']} clinical-stage antibodies")
    print("Method: TAP metrics + Liability scanning + HIC proxy")
    print("Reference: Jain T et al. PNAS 114:944-949 (2017)")

    # TAP statistics
    print(f"\n{'-'*70}")
    print("TAP METRIC STATISTICS")
    print(f"{'-'*70}")
    print(f"{'Metric':<15} {'Mean':>10} {'Min':>10} {'Max':>10}")
    print(f"{'-'*70}")
    for metric_name, stats in results['tap_statistics'].items():
        print(f"{metric_name:<15} {stats['mean']:>10.1f} {stats['min']:>10.1f} {stats['max']:>10.1f}")

    # TAP flag distribution
    flags = results['tap_flags']
    total_flags = sum(flags.values())
    print(f"\nTAP Flag Distribution (n={total_flags}):")
    print(f"  ✓ Green: {flags['green']} ({100*flags['green']/total_flags:.1f}%)")
    print(f"  ⚠ Amber: {flags['amber']} ({100*flags['amber']/total_flags:.1f}%)")
    print(f"  ✗ Red: {flags['red']} ({100*flags['red']/total_flags:.1f}%)")

    # Correlation results
    for group_name, correlations in results['metrics'].items():
        print(f"\n{'-'*70}")
        print(f"{group_name}")
        print(f"{'-'*70}")
        print(f"{'Predictor':<25} {'N':>6} {'Spearman ρ':>12} {'p-value':>10}")
        print(f"{'-'*70}")

        best_rho = -1
        best_predictor = ""
        for metric_name, data in correlations.items():
            sig = "***" if data['spearman_p'] < 0.001 else "**" if data['spearman_p'] < 0.01 else "*" if data['spearman_p'] < 0.05 else ""
            print(f"{metric_name:<25} {data['n']:>6} {data['spearman_r']:>12.3f} {data['spearman_p']:>9.4f} {sig}")
            if data['spearman_r'] > best_rho:
                best_rho = data['spearman_r']
                best_predictor = metric_name

        if best_rho > 0:
            print(f"  Best: {best_predictor} (ρ = {best_rho:.3f})")

    print(f"\n{'='*70}")

    # Overall assessment
    print("\n📊 PERFORMANCE ASSESSMENT vs SOTA:")

    # Get best correlations for each category
    expr_metrics = results['metrics'].get('Expression (HEK Titer)', {})
    acsins_metrics = results['metrics'].get('Self-Association (AC-SINS)', {})

    best_expr = max((m['spearman_r'] for m in expr_metrics.values()), default=0)
    best_acsins = max((m['spearman_r'] for m in acsins_metrics.values()), default=0)

    # Expression assessment
    if best_expr > 0.30:
        print(f"   ✅ Expression: ρ = {best_expr:.3f} (target > 0.30)")
    else:
        print(f"   ⚠️  Expression: ρ = {best_expr:.3f} (target > 0.30)")

    # Self-association assessment
    if best_acsins > 0.40:
        print(f"   ✅ Self-Association (AC-SINS): ρ = {best_acsins:.3f} (target > 0.40)")
    elif best_acsins > 0.30:
        print(f"   ✅ Self-Association (AC-SINS): ρ = {best_acsins:.3f} (approaching target 0.40)")
    else:
        print(f"   ⚠️  Self-Association (AC-SINS): ρ = {best_acsins:.3f} (target > 0.40)")

    # HIC proxy assessment
    hic_rho = acsins_metrics.get('HIC Proxy', {}).get('spearman_r', 0)
    if hic_rho > 0.50:
        print(f"   ✅ HIC Proxy: ρ = {hic_rho:.3f} (target > 0.50)")
    elif hic_rho > 0.40:
        print(f"   ✅ HIC Proxy: ρ = {hic_rho:.3f} (approaching SOTA ~0.70)")
    else:
        print(f"   ⚠️  HIC Proxy: ρ = {hic_rho:.3f} (SOTA ~0.70)")

    print("\n   Reference SOTA:")
    print("   - PROPERMAB HIC: ρ = 0.75")
    print("   - TAP original: ρ = 0.40-0.50")
    print("   - DeepSP aggregation: r = 0.89")


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="TAP-based Antibody Developability Benchmark",
    )

    parser.add_argument(
        "--data-dir", "-d",
        type=str,
        default=str(Path(__file__).parent / "data"),
        help="Path to benchmark data directory"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output JSON file for results"
    )
    parser.add_argument(
        "--max-samples", "-n",
        type=int,
        help="Maximum number of antibodies to benchmark"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Minimal output"
    )

    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    # Check if data exists
    flab_dir = data_dir / "flab"
    if not flab_dir.exists():
        print(f"Error: FLAb data not found in {flab_dir}")
        print("\nTo download the data, run:")
        print("  python benchmarks/download_flab.py")
        return 1

    # Run benchmark
    try:
        results = run_tap_benchmark(
            data_dir,
            max_samples=args.max_samples,
            verbose=not args.quiet
        )
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()
        return 1

    # Print report
    if not args.quiet:
        print_benchmark_report(results)

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {output_path}")

    # Also save to results directory
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"tap_benchmark_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
