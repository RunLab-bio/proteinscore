#!/usr/bin/env python3
"""
ProteinScore Benchmark Dataset Downloader

Downloads public datasets for validating ProteinScore predictions.
For FLAb2 antibody data, use download_flab.py instead.

Usage:
    python download_datasets.py                    # Download all datasets
    python download_datasets.py --category stability
    python download_datasets.py --category aggregation
    python download_datasets.py --category peptide
    python download_datasets.py --list             # List available datasets

Datasets:
    - WALTZ-DB 2.0: Amyloidogenic hexapeptides
    - ProThermDB: Protein thermostability
    - FireProtDB: Curated stability data
    - GLP-1 analogs: Peptide half-life validation
    - DPP-4 substrates: Proteolytic cleavage validation
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any


# =============================================================================
# Dataset Definitions
# =============================================================================

# Embedded datasets (no download required - curated from literature)
EMBEDDED_DATASETS = {
    "glp1_analogs": {
        "category": "peptide",
        "description": "GLP-1 receptor agonists with known half-lives",
        "reference": "Lau et al. Front Endocrinol 2019; FDA labels",
        "data": [
            {
                "name": "GLP-1 (7-37)",
                "sequence": "HAEGTFTSDVSSYLEGQAAKEFIAWLVKGR",
                "half_life_hours": 0.033,  # <2 min
                "half_life_category": "very_short",
                "dpp4_susceptible": True,
                "position_2": "Ala",
                "modifications": "none",
                "source": "Native hormone",
            },
            {
                "name": "GLP-1 (7-36)amide",
                "sequence": "HAEGTFTSDVSSYLEGQAAKEFIAWLVKGR",
                "half_life_hours": 0.033,
                "half_life_category": "very_short",
                "dpp4_susceptible": True,
                "position_2": "Ala",
                "modifications": "C-terminal amide",
                "source": "Native hormone",
            },
            {
                "name": "Exenatide",
                "sequence": "HGEGTFTSDLSKQMEEEAVRLFIEWLKNGGPSSGAPPPS",
                "half_life_hours": 2.4,
                "half_life_category": "short",
                "dpp4_susceptible": False,
                "position_2": "Gly",
                "modifications": "Exendin-4 based",
                "source": "FDA label - Byetta",
            },
            {
                "name": "Lixisenatide",
                "sequence": "HGEGTFTSDLSKQMEEEAVRLFIEWLKNGGPSSGAPPSKKKKKK",
                "half_life_hours": 3.0,
                "half_life_category": "short",
                "dpp4_susceptible": False,
                "position_2": "Gly",
                "modifications": "C-terminal Lys tail",
                "source": "FDA label - Adlyxin",
            },
            {
                "name": "Liraglutide",
                "sequence": "HAEGTFTSDVSSYLEGQAAKEFIAWLVRGRG",
                "half_life_hours": 13.0,
                "half_life_category": "medium",
                "dpp4_susceptible": False,  # Albumin binding protects
                "position_2": "Ala",
                "modifications": "C16 fatty acid at Lys26",
                "source": "FDA label - Victoza",
            },
            {
                "name": "Dulaglutide",
                "sequence": "HGEGTFTSDVSSYLEEQAAKEFIAWLVKGGGGGGGSGGGGSGGGGSAEPKSSDKTHTCPPCPAPELLGGPSVFLFPPKPK",
                "half_life_hours": 120.0,  # 5 days
                "half_life_category": "long",
                "dpp4_susceptible": False,
                "position_2": "Gly",
                "modifications": "Fc fusion, Gly8Glu substitution",
                "source": "FDA label - Trulicity",
            },
            {
                "name": "Semaglutide",
                "sequence": "HXEGTFTSDVSSYLEGQAAKEFIAWLVRGRG",  # X = Aib
                "half_life_hours": 168.0,  # 7 days
                "half_life_category": "long",
                "dpp4_susceptible": False,
                "position_2": "Aib",
                "modifications": "Aib8, C18 diacid at Lys26",
                "source": "FDA label - Ozempic",
            },
            {
                "name": "Tirzepatide",
                "sequence": "YXEGTFTSDYSIXLDKIAQKAFVQWLIAGGPSSGAPPPS",  # X = Aib
                "half_life_hours": 120.0,  # 5 days
                "half_life_category": "long",
                "dpp4_susceptible": False,
                "position_2": "Aib",
                "modifications": "GIP/GLP-1 dual agonist, C20 diacid",
                "source": "FDA label - Mounjaro",
            },
        ],
    },
    "dpp4_substrates": {
        "category": "peptide",
        "description": "DPP-4 cleavage substrates and products",
        "reference": "Mentlein R. Regul Pept 1999; Lambeir et al. Chem Rev 2003",
        "data": [
            {
                "name": "GLP-1 (7-36)",
                "sequence": "HAEGTFTSDVSSYLEGQAAKEFIAWLVKGR",
                "dpp4_cleavable": True,
                "cleavage_site": "HA|EG",
                "product": "GLP-1 (9-36)",
                "kcat_km": 9.0,  # s^-1 mM^-1
            },
            {
                "name": "GIP (1-42)",
                "sequence": "YAEGTFISDYSIAMDKIHQQDFVNWLLAQKGKKNDWKHNITQ",
                "dpp4_cleavable": True,
                "cleavage_site": "YA|EG",
                "product": "GIP (3-42)",
                "kcat_km": 6.0,
            },
            {
                "name": "Glucagon",
                "sequence": "HSQGTFTSDYSKYLDSRRAQDFVQWLMNT",
                "dpp4_cleavable": True,
                "cleavage_site": "HS|QG",
                "product": "Glucagon (3-29)",
                "kcat_km": 0.8,
            },
            {
                "name": "NPY",
                "sequence": "YPSKPDNPGEDAPAEDMARYYSALRHYINLITRQRY",
                "dpp4_cleavable": True,
                "cleavage_site": "YP|SK",
                "product": "NPY (3-36)",
                "kcat_km": 2.5,
            },
            {
                "name": "PYY (1-36)",
                "sequence": "YPIKPEAPGEDASPEELNRYYASLRHYLNLVTRQRY",
                "dpp4_cleavable": True,
                "cleavage_site": "YP|IK",
                "product": "PYY (3-36)",
                "kcat_km": 3.2,
            },
            {
                "name": "Substance P",
                "sequence": "RPKPQQFFGLM",
                "dpp4_cleavable": True,
                "cleavage_site": "RP|KP",
                "product": "SP (3-11)",
                "kcat_km": 0.4,
            },
            {
                "name": "Exendin-4",
                "sequence": "HGEGTFTSDLSKQMEEEAVRLFIEWLKNGGPSSGAPPPS",
                "dpp4_cleavable": False,
                "cleavage_site": "HG|EG",
                "product": "N/A (resistant)",
                "kcat_km": 0.01,  # Very low
            },
        ],
    },
    "oxidation_sensitive_peptides": {
        "category": "peptide",
        "description": "Peptides with known oxidation liabilities",
        "reference": "Manning et al. Pharm Res 2010",
        "data": [
            {
                "name": "Oxytocin",
                "sequence": "CYIQNCPLG",
                "met_positions": [],
                "cys_positions": [1, 6],
                "oxidation_sensitive": True,
                "notes": "Disulfide bridge required for activity",
            },
            {
                "name": "Vasopressin",
                "sequence": "CYFQNCPRG",
                "met_positions": [],
                "cys_positions": [1, 6],
                "oxidation_sensitive": True,
                "notes": "Disulfide bridge required",
            },
            {
                "name": "Calcitonin",
                "sequence": "CGNLSTCMLGTYTQDFNKFHTFPQTAIGVGAP",
                "met_positions": [8],
                "cys_positions": [1, 7],
                "oxidation_sensitive": True,
                "notes": "Met8 oxidation reduces activity",
            },
            {
                "name": "PTH (1-34)",
                "sequence": "SVSEIQLMHNLGKHLNSMERVEWLRKKLQDVHNF",
                "met_positions": [8, 18],
                "cys_positions": [],
                "oxidation_sensitive": True,
                "notes": "Met8, Met18 susceptible",
            },
        ],
    },
    "deamidation_sensitive_peptides": {
        "category": "peptide",
        "description": "Peptides with known deamidation liabilities",
        "reference": "Robinson NE. PNAS 2002; Capasso S. J Pept Sci 2000",
        "data": [
            {
                "name": "Model NG peptide",
                "sequence": "GSNGRG",
                "asn_positions": [3],
                "motif": "NG",
                "deamidation_t_half_days": 1.0,  # Very fast
                "notes": "Fastest deamidation motif",
            },
            {
                "name": "Model NS peptide",
                "sequence": "GSNSRG",
                "asn_positions": [3],
                "motif": "NS",
                "deamidation_t_half_days": 10.0,
                "notes": "Fast deamidation",
            },
            {
                "name": "Model NT peptide",
                "sequence": "GSNTRG",
                "asn_positions": [3],
                "motif": "NT",
                "deamidation_t_half_days": 25.0,
                "notes": "Moderate deamidation",
            },
            {
                "name": "Model NH peptide",
                "sequence": "GSNHRG",
                "asn_positions": [3],
                "motif": "NH",
                "deamidation_t_half_days": 20.0,
                "notes": "Moderate-fast deamidation",
            },
            {
                "name": "Model NA peptide",
                "sequence": "GSNARG",
                "asn_positions": [3],
                "motif": "NA",
                "deamidation_t_half_days": 100.0,
                "notes": "Slow deamidation",
            },
        ],
    },
    "waltz_core": {
        "category": "aggregation",
        "description": "Core amyloidogenic sequences from WALTZ-DB",
        "reference": "Louros et al. Nucleic Acids Res 2020",
        "data": [
            {"sequence": "KLVFFA", "amyloidogenic": True, "source": "Abeta core"},
            {"sequence": "KLVFFAE", "amyloidogenic": True, "source": "Abeta 16-22"},
            {"sequence": "NFGAIL", "amyloidogenic": True, "source": "hIAPP core"},
            {"sequence": "GVATVA", "amyloidogenic": True, "source": "Alpha-synuclein"},
            {"sequence": "VEALYL", "amyloidogenic": True, "source": "Insulin B"},
            {"sequence": "STVIIE", "amyloidogenic": True, "source": "Tau PHF6"},
            {"sequence": "VQIVYK", "amyloidogenic": True, "source": "Tau PHF6*"},
            {"sequence": "MVGGVV", "amyloidogenic": True, "source": "Abeta C-term"},
            {"sequence": "NNQQNY", "amyloidogenic": True, "source": "Sup35 prion"},
            {"sequence": "GSGSGS", "amyloidogenic": False, "source": "Gly-Ser linker"},
            {"sequence": "DKEKEK", "amyloidogenic": False, "source": "Charged peptide"},
            {"sequence": "AAEAAK", "amyloidogenic": False, "source": "Alpha-helix"},
            {"sequence": "PPPPPP", "amyloidogenic": False, "source": "Poly-Pro"},
            {"sequence": "KKKKRR", "amyloidogenic": False, "source": "NLS-like"},
        ],
    },
    "protherm_representatives": {
        "category": "stability",
        "description": "Representative proteins from ProThermDB",
        "reference": "Nikam et al. Nucleic Acids Res 2021",
        "data": [
            {
                "protein": "Lysozyme",
                "sequence": "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSS",
                "tm_celsius": 72.1,
                "deltag_kcal": 9.6,
                "organism": "Chicken",
            },
            {
                "protein": "RNase A",
                "sequence": "KETAAAKFERQHMDSSTSAASSSNYCNQMMKSRNLTKDRCKPVNTFVHESLADVQAVCSQKNVACKNGQTNCYQSYSTMSITDCRETGSSKYPNCAYKTTQANKHIIVACEGNPYVPVHFDASV",
                "tm_celsius": 63.0,
                "deltag_kcal": 8.4,
                "organism": "Bovine",
            },
            {
                "protein": "Ubiquitin",
                "sequence": "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG",
                "tm_celsius": 87.0,
                "deltag_kcal": 7.5,
                "organism": "Human",
            },
            {
                "protein": "Barnase",
                "sequence": "AQVINTFDGVADYLQTYHKLPDNYITKSEAQALGWVASKGNLADVAPGKSIGGDIFSNREGKLPGKSGRTWREADINYTSGFRNSDRILYSSDWLIYKTTDHYQTFTKIR",
                "tm_celsius": 55.0,
                "deltag_kcal": 10.5,
                "organism": "Bacillus",
            },
            {
                "protein": "Myoglobin",
                "sequence": "GLSDGEWQQVLNVWGKVEADIAGHGQEVLIRLFTGHPETLEKFDKFKHLKTEAEMKASEDLKKHGTVVLTALGGILKKKGHHEAELKPLAQSHATKHKIPIKYLEFISDAIIHVLHSKHPGDFGADAQGAMTKALELFRNDIAAKYKELGFQG",
                "tm_celsius": 76.0,
                "deltag_kcal": 8.0,
                "organism": "Sperm whale",
            },
        ],
    },
}

# Remote datasets (require download)
REMOTE_DATASETS = {
    "waltz_db": {
        "category": "aggregation",
        "description": "WALTZ-DB 2.0 hexapeptide database",
        "url": "http://waltzdb.switchlab.org/",
        "download_method": "web_export",
        "format": "csv",
        "size_estimate": "~100 KB",
        "entries": 1416,
        "note": "Requires manual export from web interface",
    },
    "fireprotdb": {
        "category": "stability",
        "description": "FireProtDB curated stability data",
        "url": "https://loschmidt.chemi.muni.cz/fireprotdb/download/",
        "download_method": "web_export",
        "format": "csv",
        "size_estimate": "~5 MB",
        "note": "Multiple export formats available",
    },
    "prothermdb": {
        "category": "stability",
        "description": "ProThermDB thermodynamic database",
        "url": "https://web.iitm.ac.in/bioinfo2/prothermdb/",
        "download_method": "web_export",
        "format": "csv",
        "size_estimate": "~10 MB",
        "entries": 32000,
        "note": "Requires web interface query and export",
    },
    "iedb_mhc1": {
        "category": "immunogenicity",
        "description": "IEDB MHC Class I binding data",
        "url": "https://www.iedb.org/",
        "download_method": "api",
        "format": "csv",
        "size_estimate": "Variable",
        "note": "Use IEDB REST API or web export",
    },
}


# =============================================================================
# Download Functions
# =============================================================================

def save_embedded_dataset(name: str, data_dir: Path) -> Path:
    """Save an embedded dataset to CSV file."""
    if name not in EMBEDDED_DATASETS:
        raise ValueError(f"Unknown embedded dataset: {name}")

    dataset = EMBEDDED_DATASETS[name]
    category = dataset["category"]
    category_dir = data_dir / category
    category_dir.mkdir(parents=True, exist_ok=True)

    output_file = category_dir / f"{name}.csv"

    # Write CSV
    data = dataset["data"]
    if not data:
        return output_file

    fieldnames = list(data[0].keys())

    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)

    return output_file


def save_all_embedded_datasets(data_dir: Path, category: str | None = None) -> list[Path]:
    """Save all embedded datasets to CSV files."""
    saved_files = []

    for name, dataset in EMBEDDED_DATASETS.items():
        if category and dataset["category"] != category:
            continue

        try:
            output_file = save_embedded_dataset(name, data_dir)
            saved_files.append(output_file)
            print(f"  ✓ Saved {name} ({len(dataset['data'])} entries)")
        except Exception as e:
            print(f"  ✗ Failed to save {name}: {e}")

    return saved_files


def generate_metadata(data_dir: Path) -> None:
    """Generate metadata file for all datasets."""
    metadata = {
        "embedded_datasets": {},
        "remote_datasets": REMOTE_DATASETS,
        "generated_at": str(Path(__file__).name),
    }

    for name, dataset in EMBEDDED_DATASETS.items():
        metadata["embedded_datasets"][name] = {
            "category": dataset["category"],
            "description": dataset["description"],
            "reference": dataset["reference"],
            "entries": len(dataset["data"]),
        }

    metadata_file = data_dir / "datasets_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  ✓ Saved metadata to {metadata_file}")


def print_dataset_info() -> None:
    """Print information about available datasets."""
    print("\n" + "=" * 70)
    print("AVAILABLE DATASETS")
    print("=" * 70)

    print("\n📦 EMBEDDED DATASETS (no download required)")
    print("-" * 70)
    print(f"{'Name':<30} {'Category':<15} {'Entries':>8}")
    print("-" * 70)

    for name, dataset in EMBEDDED_DATASETS.items():
        print(f"{name:<30} {dataset['category']:<15} {len(dataset['data']):>8}")

    print("\n🌐 REMOTE DATASETS (require manual download)")
    print("-" * 70)
    print(f"{'Name':<20} {'Category':<15} {'Size':<12} {'URL'}")
    print("-" * 70)

    for name, dataset in REMOTE_DATASETS.items():
        print(f"{name:<20} {dataset['category']:<15} {dataset['size_estimate']:<12} {dataset['url']}")

    print("\n" + "=" * 70)


def print_download_instructions() -> None:
    """Print instructions for manually downloading remote datasets."""
    print("\n" + "=" * 70)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("=" * 70)

    print("""
📥 WALTZ-DB 2.0 (Aggregation)
   1. Visit: http://waltzdb.switchlab.org/
   2. Click "Browse" or "Search"
   3. Select all entries
   4. Export as CSV
   5. Save to: benchmarks/data/aggregation/waltz_db_full.csv

📥 FireProtDB (Stability)
   1. Visit: https://loschmidt.chemi.muni.cz/fireprotdb/download/
   2. Choose "CSV Export"
   3. Download the file
   4. Save to: benchmarks/data/stability/fireprotdb.csv

📥 ProThermDB (Stability)
   1. Visit: https://web.iitm.ac.in/bioinfo2/prothermdb/
   2. Use "Browse" or "Search" to query data
   3. Export results as CSV
   4. Save to: benchmarks/data/stability/prothermdb.csv

📥 IEDB (Immunogenicity)
   1. Visit: https://www.iedb.org/
   2. Use "Search" with filters for MHC Class I
   3. Export results
   4. Save to: benchmarks/data/immunogenicity/iedb_mhc1.csv

📥 FLAb2 (Antibody)
   Use the dedicated script:
   $ python benchmarks/download_flab.py
""")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Download ProteinScore benchmark datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python download_datasets.py                    # Download all embedded datasets
  python download_datasets.py --category peptide # Download peptide datasets only
  python download_datasets.py --list             # List available datasets
  python download_datasets.py --instructions     # Show manual download instructions
        """
    )

    parser.add_argument(
        "--category", "-c",
        choices=["aggregation", "stability", "solubility", "immunogenicity", "peptide"],
        help="Download specific category only"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available datasets"
    )
    parser.add_argument(
        "--instructions", "-i",
        action="store_true",
        help="Show manual download instructions"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=str(Path(__file__).parent / "data"),
        help="Output directory (default: benchmarks/data)"
    )

    args = parser.parse_args()

    if args.list:
        print_dataset_info()
        return 0

    if args.instructions:
        print_download_instructions()
        return 0

    # Create output directory
    data_dir = Path(args.output)
    data_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PROTEINSCORE BENCHMARK DATASET DOWNLOADER")
    print("=" * 70)

    if args.category:
        print(f"\nDownloading {args.category} datasets...")
    else:
        print("\nDownloading all embedded datasets...")

    print(f"Output directory: {data_dir}")
    print()

    # Save embedded datasets
    saved_files = save_all_embedded_datasets(data_dir, args.category)

    # Generate metadata
    generate_metadata(data_dir)

    print()
    print("=" * 70)
    print(f"SUMMARY: Saved {len(saved_files)} dataset files")
    print("=" * 70)

    print("\n⚠️  NOTE: Some datasets require manual download.")
    print("   Run with --instructions for manual download guide.")
    print("   Run with --list to see all available datasets.")

    if not args.category or args.category == "antibody":
        print("\n📌 For antibody datasets (FLAb2, Jain 2017), use:")
        print("   python benchmarks/download_flab.py")

    return 0


if __name__ == "__main__":
    sys.exit(main())
