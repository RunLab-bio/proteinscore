#!/usr/bin/env python3
"""
Download FLAb2 Benchmark Data

Downloads antibody developability datasets from FLAb2 (Fitness Landscape for Antibodies 2).
Repository: https://github.com/Graylab/FLAb

Includes:
- Jain 2017 clinical antibodies (gold standard)
- Garbinski 2023 thermostability/expression
- Additional developability datasets
"""

import os
import sys
import urllib.request
from pathlib import Path


BASE_URL = "https://raw.githubusercontent.com/Graylab/FLAb/main/data"

# Datasets to download organized by category
DATASETS = {
    'thermostability': [
        'garbinski2023_tm1.csv',
    ],
    'expression': [
        'jain2017biophysical_HEK.csv',
        'garbinski2023_exp.csv',
    ],
    'aggregation': [
        'jain2017biophysical_ACSINS.csv',
        'jain2017biophysical_CSIBLI.csv',
    ],
}

# File renaming for consistency
RENAME_MAP = {
    'jain2017biophysical_HEK.csv': 'jain2017_expression.csv',
    'jain2017biophysical_ACSINS.csv': 'jain2017_acsins.csv',
    'jain2017biophysical_CSIBLI.csv': 'jain2017_csibli.csv',
}


def download_file(url: str, dest: Path) -> bool:
    """Download a file from URL."""
    try:
        print(f"  Downloading {dest.name}...", end=" ", flush=True)
        urllib.request.urlretrieve(url, dest)
        print("✓")
        return True
    except Exception as e:
        print(f"✗ ({e})")
        return False


def main():
    # Determine output directory
    if len(sys.argv) > 1:
        output_dir = Path(sys.argv[1])
    else:
        output_dir = Path(__file__).parent / "data" / "flab"

    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("FLAb2 Data Download")
    print("=" * 60)
    print(f"\nTarget directory: {output_dir}")
    print(f"Source: {BASE_URL}")
    print()

    success_count = 0
    fail_count = 0

    for category, files in DATASETS.items():
        print(f"\n📁 {category.upper()}")

        for filename in files:
            url = f"{BASE_URL}/{category}/{filename}"
            dest_name = RENAME_MAP.get(filename, filename)
            dest_path = output_dir / dest_name

            if dest_path.exists():
                print(f"  {dest_name} already exists, skipping")
                success_count += 1
                continue

            if download_file(url, dest_path):
                success_count += 1
            else:
                fail_count += 1

    print()
    print("=" * 60)
    print(f"Download complete: {success_count} files downloaded, {fail_count} failed")
    print("=" * 60)
    print()
    print("To run benchmarks:")
    print("  python benchmarks/benchmark_jain2017.py")
    print("  python benchmarks/benchmark_flab.py")

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
