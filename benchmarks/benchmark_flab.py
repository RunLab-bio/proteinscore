#!/usr/bin/env python3
"""
ProteinScore FLAb2 Benchmark

Validates ProteinScore against the FLAb2 (Fitness Landscape for Antibodies 2)
benchmark - the largest public therapeutic antibody design benchmark.

Reference: Chungyoun M & Gray J. (2025). "FLAb2: Benchmarking Reveals That
Protein AI Models Cannot Yet Consistently Predict Developability Properties."

FLAb2 contains >4 million antibodies across 32 studies covering:
- Thermostability
- Expression
- Aggregation
- Binding affinity
- Pharmacokinetics
- Polyreactivity
- Immunogenicity
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class FLAbDataset:
    """A single FLAb dataset."""
    name: str
    category: str  # thermostability, expression, aggregation, etc.
    file_path: Path
    value_column: str
    n_samples: int = 0
    higher_is_better: bool = True


@dataclass
class BenchmarkResult:
    """Results for a single dataset."""
    dataset_name: str
    category: str
    n_samples: int
    spearman_r: float
    spearman_p: float
    pearson_r: float
    pearson_p: float


# =============================================================================
# Statistical Functions
# =============================================================================

def spearman_correlation(x: list[float], y: list[float]) -> tuple[float, float]:
    """Calculate Spearman rank correlation coefficient."""
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
    """Calculate Pearson correlation coefficient."""
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
# FLAb Dataset Configuration
# =============================================================================

# Datasets available in our FLAb download
FLAB_DATASETS = {
    'thermostability': {
        'jain2017_tm': {
            'file': 'jain2017_tm.csv',
            'higher_is_better': True,  # Higher Tm = more stable
        },
        'garbinski2023_tm1': {
            'file': 'garbinski2023_tm1.csv',
            'higher_is_better': True,
        },
    },
    'expression': {
        'jain2017_expression': {
            'file': 'jain2017_expression.csv',
            'higher_is_better': True,  # Higher titer = better
        },
        'garbinski2023_exp': {
            'file': 'garbinski2023_exp.csv',
            'higher_is_better': True,
        },
    },
    'aggregation': {
        'jain2017_acsins': {
            'file': 'jain2017_acsins.csv',
            'higher_is_better': False,  # Lower AC-SINS = less self-association
        },
        'jain2017_csibli': {
            'file': 'jain2017_csibli.csv',
            'higher_is_better': False,  # Lower = less cross-interaction
        },
    },
}


# =============================================================================
# Data Loading
# =============================================================================

def load_flab_dataset(
    file_path: Path,
    max_samples: int | None = None
) -> tuple[list[tuple[str, str]], list[float]]:
    """
    Load a FLAb CSV file.

    Returns:
        sequences: List of (heavy, light) tuples
        values: List of experimental values
    """
    sequences = []
    values = []

    with open(file_path) as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames or []

        # Find the value column (not heavy, light, or fitness)
        value_col = None
        for col in columns:
            if col not in ['heavy', 'light', 'fitness']:
                value_col = col
                break

        if value_col is None:
            value_col = 'fitness'

        for row in reader:
            try:
                heavy = row.get('heavy', '')
                light = row.get('light', '')
                value = float(row.get(value_col, row.get('fitness', 'nan')))

                if heavy and not math.isnan(value):
                    sequences.append((heavy, light))
                    values.append(value)

                    if max_samples and len(sequences) >= max_samples:
                        break
            except (ValueError, KeyError):
                continue

    return sequences, values


# =============================================================================
# Benchmark Functions
# =============================================================================

def benchmark_flab_dataset(
    scorer,
    sequences: list[tuple[str, str]],
    experimental_values: list[float],
    higher_is_better: bool = True,
    verbose: bool = False
) -> tuple[list[float], list[float]]:
    """
    Score sequences and return experimental vs predicted values.
    """
    experimental = []
    predicted = []

    for i, (heavy, light) in enumerate(sequences):
        try:
            # Score both chains
            vh_result = scorer.score(heavy)

            if light:
                vl_result = scorer.score(light)
                score = (vh_result.total_score + vl_result.total_score) / 2
            else:
                score = vh_result.total_score

            exp_val = experimental_values[i]

            # Adjust for directionality
            if not higher_is_better:
                exp_val = -exp_val  # Invert so positive correlation expected

            experimental.append(exp_val)
            predicted.append(score)

        except Exception:
            continue

    return experimental, predicted


def run_flab_benchmark(
    data_dir: Path,
    categories: list[str] | None = None,
    max_samples_per_dataset: int = 100,
    verbose: bool = True
) -> dict[str, Any]:
    """
    Run FLAb2 benchmark across multiple datasets.
    """
    from proteinscore import ProteinScore

    flab_dir = data_dir / "flab"

    if not flab_dir.exists():
        raise FileNotFoundError(f"FLAb data not found in {flab_dir}")

    if verbose:
        print(f"\n{'='*70}")
        print("FLAb2 Benchmark - Therapeutic Antibody Developability")
        print(f"{'='*70}")
        print("Reference: Chungyoun & Gray, bioRxiv 2025")
        print()

    # Initialize ProteinScore
    scorer = ProteinScore(local_only=True)

    results = []
    categories_to_run = categories or list(FLAB_DATASETS.keys())

    for category in categories_to_run:
        if category not in FLAB_DATASETS:
            continue

        if verbose:
            print(f"\n📊 Category: {category.upper()}")
            print("-" * 50)

        for dataset_name, config in FLAB_DATASETS[category].items():
            file_path = flab_dir / config['file']

            if not file_path.exists():
                if verbose:
                    print(f"   ⚠️  {dataset_name}: file not found")
                continue

            # Load data
            sequences, values = load_flab_dataset(file_path, max_samples_per_dataset)

            if len(sequences) < 10:
                if verbose:
                    print(f"   ⚠️  {dataset_name}: insufficient data ({len(sequences)} samples)")
                continue

            # Benchmark
            experimental, predicted = benchmark_flab_dataset(
                scorer,
                sequences,
                values,
                config['higher_is_better'],
                verbose
            )

            if len(experimental) < 10:
                continue

            # Calculate correlations
            spearman_r, spearman_p = spearman_correlation(experimental, predicted)
            pearson_r, pearson_p = pearson_correlation(experimental, predicted)

            result = BenchmarkResult(
                dataset_name=dataset_name,
                category=category,
                n_samples=len(experimental),
                spearman_r=spearman_r,
                spearman_p=spearman_p,
                pearson_r=pearson_r,
                pearson_p=pearson_p,
            )
            results.append(result)

            if verbose:
                status = "✓" if spearman_r > 0.1 else "○"
                print(f"   {status} {dataset_name}: ρ={spearman_r:.3f} (n={len(experimental)})")

    # Aggregate by category
    by_category = {}
    for result in results:
        if result.category not in by_category:
            by_category[result.category] = []
        by_category[result.category].append(result)

    category_stats = {}
    for category, cat_results in by_category.items():
        spearman_values = [r.spearman_r for r in cat_results]
        category_stats[category] = {
            'n_datasets': len(cat_results),
            'mean_spearman': sum(spearman_values) / len(spearman_values),
            'n_samples_total': sum(r.n_samples for r in cat_results),
        }

    return {
        'benchmark': 'FLAb2',
        'n_datasets': len(results),
        'n_samples_total': sum(r.n_samples for r in results),
        'by_category': category_stats,
        'per_dataset': [
            {
                'name': r.dataset_name,
                'category': r.category,
                'n': r.n_samples,
                'spearman_r': r.spearman_r,
                'pearson_r': r.pearson_r,
            }
            for r in results
        ],
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }


def print_flab_report(results: dict[str, Any]) -> None:
    """Print formatted benchmark report."""
    print(f"\n{'='*70}")
    print("FLAb2 BENCHMARK RESULTS")
    print("Fitness Landscape for Antibodies 2")
    print(f"{'='*70}")

    print(f"\nTotal: {results['n_datasets']} datasets, {results['n_samples_total']} samples")
    print("Reference: Chungyoun & Gray, bioRxiv 2025")

    print(f"\n{'-'*70}")
    print(f"{'Category':<20} {'Datasets':>10} {'Samples':>10} {'Mean ρ':>12}")
    print(f"{'-'*70}")

    for category, stats in results['by_category'].items():
        print(f"{category.capitalize():<20} {stats['n_datasets']:>10} {stats['n_samples_total']:>10} {stats['mean_spearman']:>12.3f}")

    print(f"\n{'-'*70}")
    print(f"{'Dataset':<35} {'Category':<15} {'N':>8} {'ρ':>8}")
    print(f"{'-'*70}")

    for r in sorted(results['per_dataset'], key=lambda x: -x['spearman_r']):
        print(f"{r['name']:<35} {r['category']:<15} {r['n']:>8} {r['spearman_r']:>8.3f}")

    print(f"{'='*70}")

    # Overall interpretation
    all_spearman = [r['spearman_r'] for r in results['per_dataset']]
    avg_spearman = sum(all_spearman) / len(all_spearman) if all_spearman else 0

    print("\n📊 Overall Performance:")
    print(f"   Average Spearman ρ: {avg_spearman:.3f}")

    if avg_spearman > 0.3:
        print("   ✅ Strong correlation with FLAb2 developability metrics")
    elif avg_spearman > 0.15:
        print("   ✅ Moderate correlation - captures developability trends")
    elif avg_spearman > 0:
        print("   ⚠️  Weak positive correlation")
    else:
        print("   ⚠️  No significant correlation")


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="ProteinScore FLAb2 Benchmark"
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
        help="Output JSON file"
    )
    parser.add_argument(
        "--max-samples", "-n",
        type=int,
        default=100,
        help="Max samples per dataset (default: 100)"
    )
    parser.add_argument(
        "--category", "-c",
        type=str,
        choices=['thermostability', 'expression', 'aggregation'],
        help="Run specific category only"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Minimal output"
    )

    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    # Check data
    flab_dir = data_dir / "flab"
    if not flab_dir.exists():
        print(f"Error: FLAb data not found in {flab_dir}")
        print("\nTo download, run:")
        print("  python benchmarks/download_flab.py")
        return 1

    # Run benchmark
    categories = [args.category] if args.category else None

    try:
        results = run_flab_benchmark(
            data_dir,
            categories=categories,
            max_samples_per_dataset=args.max_samples,
            verbose=not args.quiet
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1

    # Print report
    if not args.quiet:
        print_flab_report(results)

    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")

    # Save to results directory
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"flab_benchmark_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
