#!/usr/bin/env python3
"""
ProteinScore Jain 2017 Benchmark

Validates ProteinScore against the gold standard Jain et al. 2017 dataset
of 137 clinical-stage antibodies with biophysical property measurements.

Reference: Jain T et al. (2017). "Biophysical properties of the clinical-stage
antibody landscape." PNAS 114:944-949

This is the foundational benchmark dataset for antibody developability.
"""

from __future__ import annotations

import csv
import json
import math
import os
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
    expression: float | None = None  # HEK Titer (mg/L)
    ac_sins: float | None = None  # Self-association (lower = better)
    csi_bli: float | None = None  # Cross-interaction (lower = better)


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

    # Load expression data
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
                        heavy_chain=row['heavy'],
                        light_chain=row['light']
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
                        heavy_chain=row['heavy'],
                        light_chain=row['light']
                    )
                try:
                    # Column name from the file
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
                        heavy_chain=row['heavy'],
                        light_chain=row['light']
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

def run_jain2017_benchmark(
    data_dir: Path,
    max_samples: int | None = None,
    verbose: bool = True
) -> dict[str, Any]:
    """
    Run Jain 2017 benchmark against ProteinScore predictions.

    Returns correlation metrics comparing ProteinScore to experimental measurements.
    """
    from proteinscore import ProteinScore

    # Load data
    antibodies = load_jain2017_data(data_dir)

    if not antibodies:
        raise ValueError("No antibody data loaded")

    if max_samples and len(antibodies) > max_samples:
        antibodies = antibodies[:max_samples]

    if verbose:
        print(f"\n{'='*70}")
        print("Jain 2017 Benchmark - Clinical-Stage Antibodies")
        print(f"{'='*70}")
        print(f"Dataset: {len(antibodies)} antibodies")
        print("Reference: Jain et al. PNAS 2017")
        print()

    # Initialize ProteinScore
    scorer = ProteinScore(local_only=True)

    # Score each antibody
    results = {
        'expression': {'experimental': [], 'predicted': []},
        'aggregation': {'experimental': [], 'predicted': []},
        'self_association': {'experimental': [], 'predicted': []},
    }

    for i, ab in enumerate(antibodies):
        try:
            # Combine heavy and light chains
            # For antibodies, we score VH and VL separately and combine
            vh_result = scorer.score(ab.heavy_chain)
            vl_result = scorer.score(ab.light_chain)

            # Combined score (weighted average)
            combined_score = (vh_result.total_score + vl_result.total_score) / 2

            # For aggregation, access the numeric score from the result
            vh_agg = vh_result.aggregation.score if hasattr(vh_result.aggregation, 'score') else vh_result.aggregation
            vl_agg = vl_result.aggregation.score if hasattr(vl_result.aggregation, 'score') else vl_result.aggregation
            combined_agg = (vh_agg + vl_agg) / 2

            # Collect expression correlation
            if ab.expression is not None:
                results['expression']['experimental'].append(ab.expression)
                results['expression']['predicted'].append(combined_score)

            # Collect aggregation correlation (AC-SINS)
            # Note: Lower AC-SINS = better, Higher ProteinScore aggregation = better
            if ab.ac_sins is not None:
                results['aggregation']['experimental'].append(-ab.ac_sins)  # Invert for correlation
                results['aggregation']['predicted'].append(combined_agg)

            # Self-association (CSI-BLI)
            if ab.csi_bli is not None:
                results['self_association']['experimental'].append(-ab.csi_bli)
                results['self_association']['predicted'].append(combined_score)

            if verbose and (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}/{len(antibodies)} antibodies...")

        except Exception as e:
            if verbose:
                print(f"  Warning: Failed to score antibody {i + 1}: {e}")
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

    return {
        'dataset': 'Jain2017',
        'n_antibodies': len(antibodies),
        'metrics': metrics,
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }


def print_jain2017_report(results: dict[str, Any]) -> None:
    """Print formatted benchmark report."""
    print(f"\n{'='*70}")
    print("JAIN 2017 BENCHMARK RESULTS")
    print("Gold Standard for Antibody Developability")
    print(f"{'='*70}")

    print(f"\nDataset: {results['n_antibodies']} clinical-stage antibodies")
    print("Reference: Jain T et al. PNAS 114:944-949 (2017)")

    print(f"\n{'-'*70}")
    print(f"{'Metric':<25} {'N':>8} {'Spearman ρ':>12} {'Pearson r':>12}")
    print(f"{'-'*70}")

    for metric_name, data in results['metrics'].items():
        name = metric_name.replace('_', ' ').title()
        print(f"{name:<25} {data['n']:>8} {data['spearman_r']:>12.3f} {data['pearson_r']:>12.3f}")

    print(f"{'='*70}")

    # Interpretation
    print("\n📊 Interpretation:")

    if not results['metrics']:
        print("   ⚠️  No metrics could be calculated")
        return

    avg_spearman = sum(m['spearman_r'] for m in results['metrics'].values()) / len(results['metrics'])

    if avg_spearman > 0.4:
        print("   ✅ Strong correlation with experimental developability metrics")
    elif avg_spearman > 0.2:
        print("   ✅ Moderate correlation - ProteinScore captures developability trends")
    elif avg_spearman > 0:
        print("   ⚠️  Weak positive correlation")
    else:
        print("   ⚠️  No significant correlation detected")

    print("\n   Note: Jain 2017 is the foundational benchmark for antibody")
    print("   developability, used by pharma industry and regulatory bodies.")


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="ProteinScore Jain 2017 Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full benchmark
  python benchmark_jain2017.py

  # Run with limited samples for quick test
  python benchmark_jain2017.py --max-samples 50

  # Save results to JSON
  python benchmark_jain2017.py --output results.json
        """
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
        results = run_jain2017_benchmark(
            data_dir,
            max_samples=args.max_samples,
            verbose=not args.quiet
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1

    # Print report
    if not args.quiet:
        print_jain2017_report(results)

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
    results_file = results_dir / f"jain2017_benchmark_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
