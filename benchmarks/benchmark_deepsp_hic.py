#!/usr/bin/env python3
"""
Benchmark HIC prediction with DeepSP spatial features.

This script:
1. Loads HIC antibody data
2. Runs ANARCI alignment
3. Uses DeepSP (PyTorch port) to generate 30 spatial descriptors
4. Combines with handcrafted features
5. Benchmarks GBM model performance
"""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# Import our PyTorch DeepSP
from deepsp_pytorch import DeepSPPredictor

# Check sklearn
try:
    from sklearn.model_selection import KFold
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
except ImportError:
    print("Error: sklearn required")
    sys.exit(1)

try:
    from Bio import SeqIO
    from Bio.Seq import Seq
    from Bio.SeqRecord import SeqRecord
except ImportError:
    print("Error: biopython required")
    sys.exit(1)


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class AntibodyHIC:
    name: str
    vh_sequence: str
    vl_sequence: str
    hic_rt: float


# =============================================================================
# Amino Acid Properties
# =============================================================================

AA_PROPERTIES = {
    'A': {'hydropathy': 1.8, 'charge': 0, 'aromatic': 0, 'volume': 88.6, 'polar': 0},
    'R': {'hydropathy': -4.5, 'charge': 1, 'aromatic': 0, 'volume': 173.4, 'polar': 1},
    'N': {'hydropathy': -3.5, 'charge': 0, 'aromatic': 0, 'volume': 114.1, 'polar': 1},
    'D': {'hydropathy': -3.5, 'charge': -1, 'aromatic': 0, 'volume': 111.1, 'polar': 1},
    'C': {'hydropathy': 2.5, 'charge': 0, 'aromatic': 0, 'volume': 108.5, 'polar': 0},
    'Q': {'hydropathy': -3.5, 'charge': 0, 'aromatic': 0, 'volume': 143.8, 'polar': 1},
    'E': {'hydropathy': -3.5, 'charge': -1, 'aromatic': 0, 'volume': 138.4, 'polar': 1},
    'G': {'hydropathy': -0.4, 'charge': 0, 'aromatic': 0, 'volume': 60.1, 'polar': 0},
    'H': {'hydropathy': -3.2, 'charge': 0.5, 'aromatic': 1, 'volume': 153.2, 'polar': 1},
    'I': {'hydropathy': 4.5, 'charge': 0, 'aromatic': 0, 'volume': 166.7, 'polar': 0},
    'L': {'hydropathy': 3.8, 'charge': 0, 'aromatic': 0, 'volume': 166.7, 'polar': 0},
    'K': {'hydropathy': -3.9, 'charge': 1, 'aromatic': 0, 'volume': 168.6, 'polar': 1},
    'M': {'hydropathy': 1.9, 'charge': 0, 'aromatic': 0, 'volume': 162.9, 'polar': 0},
    'F': {'hydropathy': 2.8, 'charge': 0, 'aromatic': 1, 'volume': 189.9, 'polar': 0},
    'P': {'hydropathy': -1.6, 'charge': 0, 'aromatic': 0, 'volume': 112.7, 'polar': 0},
    'S': {'hydropathy': -0.8, 'charge': 0, 'aromatic': 0, 'volume': 89.0, 'polar': 1},
    'T': {'hydropathy': -0.7, 'charge': 0, 'aromatic': 0, 'volume': 116.1, 'polar': 1},
    'W': {'hydropathy': -0.9, 'charge': 0, 'aromatic': 1, 'volume': 227.8, 'polar': 0},
    'Y': {'hydropathy': -1.3, 'charge': 0, 'aromatic': 1, 'volume': 193.6, 'polar': 1},
    'V': {'hydropathy': 4.2, 'charge': 0, 'aromatic': 0, 'volume': 140.0, 'polar': 0},
}

HYDROPHOBIC_AAS = set('AILMFVW')
AROMATIC_AAS = set('FWY')


# =============================================================================
# Handcrafted Features
# =============================================================================

def extract_handcrafted_features(vh: str, vl: str) -> np.ndarray:
    """Extract all handcrafted features (59 total)."""
    full_seq = vh + vl
    features = []

    # AA composition (20)
    for aa in sorted(AA_PROPERTIES.keys()):
        features.append(full_seq.count(aa) / len(full_seq))

    # Physicochemical properties (23)
    for prop in ['hydropathy', 'charge', 'aromatic', 'volume', 'polar']:
        values = [AA_PROPERTIES.get(aa, {}).get(prop, 0) for aa in full_seq]
        features.extend([np.mean(values), np.std(values), np.min(values), np.max(values)])

    hydro_count = sum(1 for aa in full_seq if aa in HYDROPHOBIC_AAS)
    features.append(hydro_count / len(full_seq))
    arom_count = sum(1 for aa in full_seq if aa in AROMATIC_AAS)
    features.append(arom_count / len(full_seq))
    net_charge = sum(AA_PROPERTIES.get(aa, {}).get('charge', 0) for aa in full_seq)
    features.append(net_charge / len(full_seq))

    # Regional features (12)
    cdr_h1 = vh[26:35] if len(vh) > 35 else vh[26:]
    cdr_h2 = vh[50:65] if len(vh) > 65 else vh[50:]
    cdr_h3 = vh[95:110] if len(vh) > 110 else vh[95:]
    cdr_l1 = vl[24:34] if len(vl) > 34 else vl[24:]
    cdr_l2 = vl[50:56] if len(vl) > 56 else vl[50:]
    cdr_l3 = vl[89:97] if len(vl) > 97 else vl[89:]

    for cdr in [cdr_h1, cdr_h2, cdr_h3, cdr_l1, cdr_l2, cdr_l3]:
        if len(cdr) > 0:
            hydro = sum(AA_PROPERTIES.get(aa, {}).get('hydropathy', 0) for aa in cdr) / len(cdr)
            charge = sum(AA_PROPERTIES.get(aa, {}).get('charge', 0) for aa in cdr) / len(cdr)
        else:
            hydro, charge = 0, 0
        features.extend([hydro, charge])

    # Surface exposure features (4)
    max_hydro_patch = 0
    current_patch = 0
    for aa in full_seq:
        if aa in HYDROPHOBIC_AAS:
            current_patch += 1
            max_hydro_patch = max(max_hydro_patch, current_patch)
        else:
            current_patch = 0
    features.append(max_hydro_patch)

    hydro_clusters = 0
    in_cluster = False
    for aa in full_seq:
        if aa in HYDROPHOBIC_AAS:
            if not in_cluster:
                hydro_clusters += 1
                in_cluster = True
        else:
            in_cluster = False
    features.append(hydro_clusters / len(full_seq))

    arom_positions = [i for i, aa in enumerate(full_seq) if aa in AROMATIC_AAS]
    if len(arom_positions) > 1:
        avg_dist = np.mean(np.diff(sorted(arom_positions)))
        features.append(1 / (avg_dist + 1))
    else:
        features.append(0)

    surface_hydro = sum(AA_PROPERTIES.get(aa, {}).get('hydropathy', 0)
                        for aa in full_seq if AA_PROPERTIES.get(aa, {}).get('hydropathy', 0) > 0)
    features.append(surface_hydro / len(full_seq))

    return np.array(features, dtype=np.float32)


# =============================================================================
# ANARCI Alignment
# =============================================================================

def run_anarci_alignment(names: list[str], vh_seqs: list[str], vl_seqs: list[str],
                          work_dir: Path) -> list[str]:
    """Run ANARCI to align antibody sequences using anarcii."""
    # Write FASTA files
    h_fasta = work_dir / 'seq_H.fasta'
    l_fasta = work_dir / 'seq_L.fasta'

    with open(h_fasta, 'w') as f:
        for name, seq in zip(names, vh_seqs):
            f.write(f'>{name}\n{seq}\n')

    with open(l_fasta, 'w') as f:
        for name, seq in zip(names, vl_seqs):
            f.write(f'>{name}\n{seq}\n')

    # Run anarcii (new pip-installable version)
    h_out = work_dir / 'seq_aligned_H.csv'
    l_out = work_dir / 'seq_aligned_KL.csv'

    try:
        result = subprocess.run(
            ['anarcii', str(h_fasta), '--scheme', 'imgt', '-o', str(h_out)],
            capture_output=True, text=True, check=True, cwd=work_dir
        )
        result = subprocess.run(
            ['anarcii', str(l_fasta), '--scheme', 'imgt', '-o', str(l_out)],
            capture_output=True, text=True, check=True, cwd=work_dir
        )
    except subprocess.CalledProcessError as e:
        print(f"  ANARCI error: {e.stderr}")
        return []
    except FileNotFoundError:
        print("  ANARCI not found. Install with: pip install anarcii")
        return []

    return parse_aligned_sequences(work_dir, names)


def parse_aligned_sequences(work_dir: Path, names: list[str]) -> list[str]:
    """Parse ANARCI output and create aligned HL sequences for DeepSP.

    DeepSP expects exactly 272 positions:
    - VH: 145 positions (1-128 with insertion codes at 111A-H and 112I-A)
    - VL: 127 positions (1-127)
    """
    h_file = work_dir / 'seq_aligned_H.csv'
    l_file = work_dir / 'seq_aligned_KL.csv'

    if not h_file.exists() or not l_file.exists():
        print(f"  ANARCI output files not found")
        return []

    h_df = pd.read_csv(h_file)
    l_df = pd.read_csv(l_file)

    # New anarcii format uses 'Name' column instead of 'Id'
    name_col = 'Name' if 'Name' in h_df.columns else 'Id'

    # DeepSP expected positions (exactly matching original code)
    H_positions = ['1','2','3','4','5','6','7','8','9','10',
                   '11','12','13','14','15','16','17','18','19','20',
                   '21','22','23','24','25','26','27','28','29','30',
                   '31','32','33','34','35','36','37','38','39','40',
                   '41','42','43','44','45','46','47','48','49','50',
                   '51','52','53','54','55','56','57','58','59','60',
                   '61','62','63','64','65','66','67','68','69','70',
                   '71','72','73','74','75','76','77','78','79','80',
                   '81','82','83','84','85','86','87','88','89','90',
                   '91','92','93','94','95','96','97','98','99','100',
                   '101','102','103','104','105','106','107','108','109','110',
                   '111','111A','111B','111C','111D','111E','111F','111G','111H',
                   '112I','112H','112G','112F','112E','112D','112C','112B','112A','112',
                   '113','114','115','116','117','118','119','120',
                   '121','122','123','124','125','126','127','128']  # 145 positions

    L_positions = ['1','2','3','4','5','6','7','8','9','10',
                   '11','12','13','14','15','16','17','18','19','20',
                   '21','22','23','24','25','26','27','28','29','30',
                   '31','32','33','34','35','36','37','38','39','40',
                   '41','42','43','44','45','46','47','48','49','50',
                   '51','52','53','54','55','56','57','58','59','60',
                   '61','62','63','64','65','66','67','68','69','70',
                   '71','72','73','74','75','76','77','78','79','80',
                   '81','82','83','84','85','86','87','88','89','90',
                   '91','92','93','94','95','96','97','98','99','100',
                   '101','102','103','104','105','106','107','108','109','110',
                   '111','112','113','114','115','116','117','118','119','120',
                   '121','122','123','124','125','126','127']  # 127 positions

    # Map names to row indices
    h_name_to_idx = {str(row[name_col]): i for i, row in h_df.iterrows()}
    l_name_to_idx = {str(row[name_col]): i for i, row in l_df.iterrows()}

    aligned_seqs = []
    for name in names:
        h_idx = h_name_to_idx.get(name)
        l_idx = l_name_to_idx.get(name)

        if h_idx is None or l_idx is None:
            # Use gap-filled sequence if not found
            aligned_seqs.append('-' * 272)
            continue

        # Build H chain aligned sequence using DeepSP positions
        h_seq = ''
        for pos in H_positions:
            if pos in h_df.columns:
                aa = h_df.iloc[h_idx][pos]
                if pd.notna(aa) and aa != '' and str(aa) != '-':
                    h_seq += str(aa)
                else:
                    h_seq += '-'
            else:
                h_seq += '-'

        # Build L chain aligned sequence using DeepSP positions
        l_seq = ''
        for pos in L_positions:
            if pos in l_df.columns:
                aa = l_df.iloc[l_idx][pos]
                if pd.notna(aa) and aa != '' and str(aa) != '-':
                    l_seq += str(aa)
                else:
                    l_seq += '-'
            else:
                l_seq += '-'

        aligned_seqs.append(h_seq + l_seq)

    return aligned_seqs


# =============================================================================
# Metrics
# =============================================================================

def spearman_correlation(x: list[float], y: list[float]) -> float:
    """Calculate Spearman correlation coefficient."""
    n = len(x)
    if n < 3:
        return 0.0

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
        return 0.0

    return num / (den_x * den_y)


# =============================================================================
# Data Loading
# =============================================================================

def load_hic_data(data_dir: Path) -> list[AntibodyHIC]:
    """Load HIC retention time data."""
    flab_dir = data_dir / "flab"
    hicrt_file = flab_dir / "jain2017biophyscial_HICRT.csv"

    antibodies = []
    if hicrt_file.exists():
        with open(hicrt_file) as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                try:
                    col = [c for c in row.keys() if 'HIC' in c and 'Retention' in c][0]
                    hic_rt = float(row[col])
                    antibodies.append(AntibodyHIC(
                        name=f"Ab_{i+1:03d}",
                        vh_sequence=row['heavy'],
                        vl_sequence=row['light'],
                        hic_rt=hic_rt
                    ))
                except (ValueError, KeyError, IndexError):
                    continue

    return antibodies


# =============================================================================
# Benchmark
# =============================================================================

def run_cv_experiment(name: str, X: np.ndarray, y: np.ndarray,
                      n_folds: int = 5) -> dict:
    """Run cross-validation experiment."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42
    )

    fold_rhos = []
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_val_scaled)

        rho = spearman_correlation(y_val.tolist(), y_pred.tolist())
        fold_rhos.append(rho)

    return {
        'name': name,
        'rho': np.mean(fold_rhos),
        'rho_std': np.std(fold_rhos),
        'fold_rhos': fold_rhos,
    }


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("HIC Prediction with DeepSP Spatial Features (PyTorch)")
    print("=" * 70)

    # Paths
    benchmark_dir = Path(__file__).parent
    data_dir = benchmark_dir / "data"
    deepsp_dir = Path("/tmp/DeepSP/DeepSP_models")

    # Check DeepSP models
    if not deepsp_dir.exists():
        print("\nDeepSP models not found. Cloning repository...")
        subprocess.run(['git', 'clone', '--depth', '1',
                       'https://github.com/Lailabcode/DeepSP.git', '/tmp/DeepSP'],
                      capture_output=True)

    # Load data
    print("\n1. Loading HIC data...")
    antibodies = load_hic_data(data_dir)
    print(f"   Loaded {len(antibodies)} antibodies")

    if len(antibodies) == 0:
        print("   No data found!")
        return 1

    y = np.array([ab.hic_rt for ab in antibodies])

    # Extract handcrafted features
    print("\n2. Extracting handcrafted features...")
    X_handcrafted = np.array([
        extract_handcrafted_features(ab.vh_sequence, ab.vl_sequence)
        for ab in antibodies
    ])
    print(f"   Shape: {X_handcrafted.shape}")

    # Run ANARCI alignment
    print("\n3. Running ANARCI alignment...")
    work_dir = Path(tempfile.mkdtemp())
    names = [ab.name for ab in antibodies]
    vh_seqs = [ab.vh_sequence for ab in antibodies]
    vl_seqs = [ab.vl_sequence for ab in antibodies]

    aligned_seqs = run_anarci_alignment(names, vh_seqs, vl_seqs, work_dir)

    if len(aligned_seqs) == 0:
        print("   ANARCI alignment failed!")
        print("   Running benchmark with handcrafted features only...")

        result = run_cv_experiment("Handcrafted_only", X_handcrafted, y)
        print(f"\n   Result: ρ = {result['rho']:.3f} ± {result['rho_std']:.3f}")

        # Cleanup
        import shutil
        shutil.rmtree(work_dir, ignore_errors=True)
        return 0

    print(f"   Aligned {len(aligned_seqs)} sequences")
    print(f"   Aligned length: {len(aligned_seqs[0])}")

    # Pad/trim to 272 if needed
    aligned_seqs = [seq[:272].ljust(272, '-') for seq in aligned_seqs]

    # Load DeepSP and predict
    print("\n4. Loading DeepSP models (PyTorch)...")
    predictor = DeepSPPredictor(str(deepsp_dir))

    print("\n5. Predicting DeepSP features...")
    X_deepsp = predictor.predict(aligned_seqs)
    print(f"   Shape: {X_deepsp.shape}")
    print(f"   Features: {predictor.feature_names[:3]}...")

    # Combine features
    X_combined = np.hstack([X_handcrafted, X_deepsp])
    print(f"\n6. Combined features shape: {X_combined.shape}")

    # Run benchmarks
    print("\n" + "=" * 70)
    print("BENCHMARKS")
    print("=" * 70)

    results = []

    # Handcrafted only
    print("\n[1] Handcrafted features only...")
    result = run_cv_experiment("Handcrafted_only", X_handcrafted, y)
    results.append(result)
    print(f"    ρ = {result['rho']:.3f} ± {result['rho_std']:.3f}")
    print(f"    Folds: {[f'{r:.3f}' for r in result['fold_rhos']]}")

    # DeepSP only
    print("\n[2] DeepSP features only (30 spatial properties)...")
    result = run_cv_experiment("DeepSP_only", X_deepsp, y)
    results.append(result)
    print(f"    ρ = {result['rho']:.3f} ± {result['rho_std']:.3f}")
    print(f"    Folds: {[f'{r:.3f}' for r in result['fold_rhos']]}")

    # Combined
    print("\n[3] Combined features (Handcrafted + DeepSP)...")
    result = run_cv_experiment("Combined", X_combined, y)
    results.append(result)
    print(f"    ρ = {result['rho']:.3f} ± {result['rho_std']:.3f}")
    print(f"    Folds: {[f'{r:.3f}' for r in result['fold_rhos']]}")

    # Try optimized GBM
    print("\n[4] Optimized GBM with combined features...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    best_rho = 0
    best_params = {}

    for n_est in [200, 300]:
        for depth in [3, 4, 5]:
            for lr in [0.03, 0.05]:
                model = GradientBoostingRegressor(
                    n_estimators=n_est, max_depth=depth, learning_rate=lr,
                    subsample=0.8, min_samples_leaf=3, random_state=42
                )

                fold_rhos = []
                for train_idx, val_idx in kf.split(X_combined):
                    X_train, X_val = X_combined[train_idx], X_combined[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]

                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_val_scaled = scaler.transform(X_val)

                    model.fit(X_train_scaled, y_train)
                    y_pred = model.predict(X_val_scaled)
                    fold_rhos.append(spearman_correlation(y_val.tolist(), y_pred.tolist()))

                avg_rho = np.mean(fold_rhos)
                if avg_rho > best_rho:
                    best_rho = avg_rho
                    best_params = {'n_est': n_est, 'depth': depth, 'lr': lr}
                    best_fold_rhos = fold_rhos

    print(f"    Best params: {best_params}")
    print(f"    ρ = {best_rho:.3f} ± {np.std(best_fold_rhos):.3f}")
    print(f"    Folds: {[f'{r:.3f}' for r in best_fold_rhos]}")

    results.append({
        'name': 'GBM_optimized',
        'rho': best_rho,
        'rho_std': np.std(best_fold_rhos),
        'fold_rhos': best_fold_rhos,
        'params': best_params,
    })

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"\n{'Model':<25} {'ρ':>8} {'± std':>8}")
    print("-" * 45)
    for r in sorted(results, key=lambda x: x['rho'], reverse=True):
        print(f"{r['name']:<25} {r['rho']:>8.3f} {r['rho_std']:>8.3f}")

    best = max(results, key=lambda x: x['rho'])
    print(f"\n✅ Best: {best['name']} with ρ = {best['rho']:.3f}")

    print("\n" + "-" * 50)
    print("Comparison with published methods:")
    print("-" * 50)
    print(f"  Current ML HIC:     ρ = 0.351")
    print(f"  Previous best:      ρ = 0.553 (GBM + handcrafted)")
    print(f"  With DeepSP:        ρ = {best['rho']:.3f}")
    print(f"  PROPERMAB (target): ρ = 0.75")

    improvement = (best['rho'] - 0.553) / 0.553 * 100
    gap_to_sota = 0.75 - best['rho']
    print(f"\n  Improvement from baseline: {improvement:+.1f}%")
    print(f"  Gap to PROPERMAB: {gap_to_sota:.3f}")

    # Save results
    results_file = benchmark_dir / "deepsp_hic_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_file}")

    # Cleanup
    import shutil
    shutil.rmtree(work_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
