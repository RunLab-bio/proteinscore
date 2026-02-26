#!/usr/bin/env python3
"""
Hydrophobicity Module Benchmark against Jain 2017 Dataset

Validates the sequence-based HIC and self-association predictions
against experimental AC-SINS and HIC retention data.

Reference: Jain T et al. (2017). Biophysical properties of clinical-stage antibodies.
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


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class AntibodyData:
    """Antibody with experimental measurements."""
    heavy_chain: str
    light_chain: str
    hic_retention: float | None = None  # HIC RT (min)
    ac_sins: float | None = None  # Self-association (lower = better)


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

    antibodies = {}

    # Load AC-SINS data
    acsins_file = flab_dir / "jain2017_acsins.csv"
    if acsins_file.exists():
        with open(acsins_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row['heavy'], row['light'])
                if key not in antibodies:
                    antibodies[key] = AntibodyData(
                        heavy_chain=row['heavy'],
                        light_chain=row['light']
                    )
                try:
                    col = [c for c in row.keys() if 'AC-SINS' in c or 'SINS' in c][0]
                    antibodies[key].ac_sins = float(row[col])
                except (ValueError, KeyError, IndexError):
                    pass

    # Load HIC data (if available - may be in separate file or same as acsins)
    hic_file = flab_dir / "jain2017_hic.csv"
    if hic_file.exists():
        with open(hic_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (row['heavy'], row['light'])
                if key not in antibodies:
                    antibodies[key] = AntibodyData(
                        heavy_chain=row['heavy'],
                        light_chain=row['light']
                    )
                try:
                    col = [c for c in row.keys() if 'HIC' in c or 'RT' in c][0]
                    antibodies[key].hic_retention = float(row[col])
                except (ValueError, KeyError, IndexError):
                    pass

    return list(antibodies.values())


# =============================================================================
# Benchmark Functions
# =============================================================================

def run_hydrophobicity_benchmark(
    data_dir: Path,
    max_samples: int | None = None,
    verbose: bool = True
) -> dict[str, Any]:
    """
    Run hydrophobicity benchmark against Jain 2017 experimental data.

    Returns correlation metrics comparing predictions to experimental measurements.
    """
    from proteinscore.antibody import CDRDetector, TAPMetrics
    from proteinscore.antibody.hydrophobicity import (
        analyze_hydrophobicity,
        predict_hic_retention,
        predict_self_association,
        get_combined_hic_score,
        calculate_cdr_surface_hydrophobicity,
        calculate_specificity_index,
    )

    # Load data
    antibodies = load_jain2017_data(data_dir)

    if not antibodies:
        raise ValueError("No antibody data loaded")

    if max_samples and len(antibodies) > max_samples:
        antibodies = antibodies[:max_samples]

    if verbose:
        print(f"\n{'='*70}")
        print("Hydrophobicity Module Benchmark - Jain 2017 Clinical Antibodies")
        print(f"{'='*70}")
        print(f"Dataset: {len(antibodies)} antibodies")
        print()

    # Initialize CDR detector
    cdr_detector = CDRDetector()

    # Collect predictions and experimental values
    results = {
        'hic': {'experimental': [], 'predicted': []},
        'ac_sins': {'experimental': [], 'predicted': []},
        'ac_sins_cdr_surface': {'experimental': [], 'predicted': []},
        'ac_sins_psh': {'experimental': [], 'predicted': []},
        'ac_sins_jain_hic': {'experimental': [], 'predicted': []},
        'ac_sins_specificity': {'experimental': [], 'predicted': []},
    }

    # Initialize TAP metrics (Jain scale)
    tap_jain = TAPMetrics(hydrophobicity_scale="jain")

    # Detailed metrics
    hic_scores = []
    self_assoc_scores = []
    dev_scores = []
    patch_counts = []
    cdr_surface_scores = []
    psh_scores = []
    jain_hic_scores = []
    specificity_scores = []

    for i, ab in enumerate(antibodies):
        try:
            # Detect CDRs
            vh_cdrs = cdr_detector.detect_cdrs(ab.heavy_chain, "heavy")
            vl_cdrs = cdr_detector.detect_cdrs(ab.light_chain, "light")

            # Run hydrophobicity analysis
            vh_analysis = analyze_hydrophobicity(ab.heavy_chain, "heavy", vh_cdrs)
            vl_analysis = analyze_hydrophobicity(ab.light_chain, "light", vl_cdrs)

            # Combined scores
            combined_hic = get_combined_hic_score(vh_analysis, vl_analysis)
            combined_self_assoc = (
                vh_analysis.self_association.self_association_score * 0.6 +
                vl_analysis.self_association.self_association_score * 0.4
            )
            combined_dev = (
                vh_analysis.developability_score * 0.6 +
                vl_analysis.developability_score * 0.4
            )
            total_patches = (
                len(vh_analysis.patches) + len(vl_analysis.patches)
            )

            # CDR surface hydrophobicity (direct calculation)
            cdr_surface_vh = calculate_cdr_surface_hydrophobicity(ab.heavy_chain, vh_cdrs)
            cdr_surface_vl = calculate_cdr_surface_hydrophobicity(ab.light_chain, vl_cdrs)
            combined_cdr_surface = cdr_surface_vh * 0.6 + cdr_surface_vl * 0.4

            # TAP metrics (PSH and Jain HIC proxy)
            tap_result = tap_jain.calculate(ab.heavy_chain, ab.light_chain)
            psh_value = tap_result.psh.value
            jain_hic = tap_jain.calculate_hic_proxy(ab.heavy_chain, ab.light_chain)

            # Specificity index
            spec_vh = calculate_specificity_index(ab.heavy_chain, vh_cdrs)
            spec_vl = calculate_specificity_index(ab.light_chain, vl_cdrs)
            combined_spec = spec_vh * 0.6 + spec_vl * 0.4

            hic_scores.append(combined_hic)
            self_assoc_scores.append(combined_self_assoc)
            dev_scores.append(combined_dev)
            patch_counts.append(total_patches)
            cdr_surface_scores.append(combined_cdr_surface)
            psh_scores.append(psh_value)
            jain_hic_scores.append(jain_hic)
            specificity_scores.append(combined_spec)

            # Collect for correlation (HIC)
            if ab.hic_retention is not None:
                results['hic']['experimental'].append(ab.hic_retention)
                results['hic']['predicted'].append(combined_hic)

            # Collect for correlation (AC-SINS)
            # Note: Higher AC-SINS = more self-association (worse)
            # Our score: Higher = more self-association prone
            if ab.ac_sins is not None:
                results['ac_sins']['experimental'].append(ab.ac_sins)
                results['ac_sins']['predicted'].append(combined_self_assoc)

                # Also test CDR surface hydrophobicity directly
                results['ac_sins_cdr_surface']['experimental'].append(ab.ac_sins)
                results['ac_sins_cdr_surface']['predicted'].append(combined_cdr_surface)

                # PSH from TAP
                results['ac_sins_psh']['experimental'].append(ab.ac_sins)
                results['ac_sins_psh']['predicted'].append(psh_value)

                # Jain HIC proxy
                results['ac_sins_jain_hic']['experimental'].append(ab.ac_sins)
                results['ac_sins_jain_hic']['predicted'].append(jain_hic)

                # Specificity index
                results['ac_sins_specificity']['experimental'].append(ab.ac_sins)
                results['ac_sins_specificity']['predicted'].append(combined_spec)

            if verbose and (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}/{len(antibodies)} antibodies...")

        except Exception as e:
            if verbose:
                print(f"  Warning: Failed to analyze antibody {i + 1}: {e}")
            continue

    # Calculate correlations
    metrics = {}

    for metric_name, data in results.items():
        if len(data['experimental']) >= 10:
            spearman_r, spearman_p = spearman_correlation(
                data['experimental'], data['predicted']
            )
            pearson_r, pearson_p = pearson_correlation(
                data['experimental'], data['predicted']
            )

            metrics[metric_name] = {
                'n': len(data['experimental']),
                'spearman_r': spearman_r,
                'spearman_p': spearman_p,
                'pearson_r': pearson_r,
                'pearson_p': pearson_p,
            }

    # Calculate distribution statistics
    def calc_stats(values):
        if not values:
            return {}
        n = len(values)
        mean = sum(values) / n
        sorted_vals = sorted(values)
        return {
            'mean': mean,
            'min': min(values),
            'max': max(values),
            'p25': sorted_vals[n // 4],
            'p50': sorted_vals[n // 2],
            'p75': sorted_vals[3 * n // 4],
        }

    return {
        'dataset': 'Jain2017_Hydrophobicity',
        'n_antibodies': len(antibodies),
        'n_analyzed': len(hic_scores),
        'correlations': metrics,
        'distributions': {
            'hic_score': calc_stats(hic_scores),
            'self_association_score': calc_stats(self_assoc_scores),
            'cdr_surface_hydrophobicity': calc_stats(cdr_surface_scores),
            'psh_tap': calc_stats(psh_scores),
            'jain_hic_proxy': calc_stats(jain_hic_scores),
            'specificity_index': calc_stats(specificity_scores),
            'developability_score': calc_stats(dev_scores),
            'patch_count': calc_stats(patch_counts),
        },
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }


def print_hydrophobicity_report(results: dict[str, Any]) -> None:
    """Print formatted benchmark report."""
    print(f"\n{'='*70}")
    print("HYDROPHOBICITY BENCHMARK RESULTS")
    print("Sequence-Based HIC/AC-SINS Prediction")
    print(f"{'='*70}")

    print(f"\nDataset: {results['n_analyzed']}/{results['n_antibodies']} clinical-stage antibodies")

    # Correlation results
    if results['correlations']:
        print(f"\n{'-'*70}")
        print(f"{'Metric':<20} {'N':>8} {'Spearman ρ':>12} {'Pearson r':>12} {'p-value':>12}")
        print(f"{'-'*70}")

        for metric_name, data in results['correlations'].items():
            name = metric_name.upper().replace('_', '-')
            p_str = f"{data['spearman_p']:.2e}" if data['spearman_p'] < 0.01 else f"{data['spearman_p']:.3f}"
            print(f"{name:<20} {data['n']:>8} {data['spearman_r']:>12.3f} {data['pearson_r']:>12.3f} {p_str:>12}")

    # Distribution statistics
    print(f"\n{'-'*70}")
    print("Score Distributions:")
    print(f"{'-'*70}")

    for metric_name, stats in results['distributions'].items():
        if stats:
            name = metric_name.replace('_', ' ').title()
            print(f"\n{name}:")
            print(f"  Mean: {stats['mean']:.1f}  Min: {stats['min']:.1f}  Max: {stats['max']:.1f}")
            print(f"  P25: {stats['p25']:.1f}  P50: {stats['p50']:.1f}  P75: {stats['p75']:.1f}")

    print(f"\n{'='*70}")

    # Interpretation
    print("\n📊 Interpretation:")

    if not results['correlations']:
        print("   ⚠️  No experimental data available for correlation")
        print("   ℹ️  Score distributions computed successfully")
        return

    for metric, data in results['correlations'].items():
        rho = data['spearman_r']
        metric_name = metric.upper().replace('_', '-')

        if rho > 0.4:
            print(f"   ✅ {metric_name}: Strong positive correlation (ρ = {rho:.3f})")
        elif rho > 0.2:
            print(f"   ✅ {metric_name}: Moderate positive correlation (ρ = {rho:.3f})")
        elif rho > 0:
            print(f"   ⚠️  {metric_name}: Weak positive correlation (ρ = {rho:.3f})")
        elif rho > -0.2:
            print(f"   ⚠️  {metric_name}: No significant correlation (ρ = {rho:.3f})")
        else:
            print(f"   ⚠️  {metric_name}: Negative correlation (ρ = {rho:.3f})")

    print("\n   Note: Sequence-only predictions have inherent limitations.")
    print("   Best achievable correlation without 3D structure: r² ~ 0.4-0.6")


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Hydrophobicity Module Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        results = run_hydrophobicity_benchmark(
            data_dir,
            max_samples=args.max_samples,
            verbose=not args.quiet
        )
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Print report
    if not args.quiet:
        print_hydrophobicity_report(results)

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
    results_file = results_dir / f"hydrophobicity_benchmark_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
