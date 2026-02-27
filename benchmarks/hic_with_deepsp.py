#!/usr/bin/env python3
"""
HIC Prediction with DeepSP Structural Features

Goal: Reach PROPERMAB-level correlation (ρ ≈ 0.75) by combining:
1. Our handcrafted sequence features (current best: ρ = 0.553)
2. DeepSP's 30 spatial properties (SAP, SCM scores predicted from sequence)

DeepSP: https://github.com/Lailabcode/DeepSP
- CNN model that predicts spatial properties directly from sequence
- No MD simulation or 3D structure required
- Correlates 0.76-0.96 with MD-derived scores

Reference: Lailab et al. 2024 "DeepSP: Deep learning-based spatial properties
to predict monoclonal antibody stability"
"""

from __future__ import annotations

import csv
import json
import math
import os
import pickle
import subprocess
import sys
import tempfile
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

warnings.filterwarnings('ignore')

# Check for sklearn
try:
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: sklearn not available, some experiments will be skipped")


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class AntibodyHIC:
    vh_sequence: str
    vl_sequence: str
    hic_rt: float


@dataclass
class ExperimentResult:
    name: str
    rho: float
    rho_std: float = 0.0
    fold_rhos: list = field(default_factory=list)
    params: dict = field(default_factory=dict)
    notes: str = ""


# =============================================================================
# Amino Acid Properties (from hic_sota_experiments.py)
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
CHARGED_AAS = set('DEKRH')


# =============================================================================
# Feature Extraction Functions (from hic_sota_experiments.py)
# =============================================================================

def extract_basic_features(vh: str, vl: str) -> np.ndarray:
    """Basic AA composition features (20 features)."""
    full_seq = vh + vl
    features = []
    for aa in sorted(AA_PROPERTIES.keys()):
        features.append(full_seq.count(aa) / len(full_seq))
    return np.array(features, dtype=np.float32)


def extract_physicochemical_features(vh: str, vl: str) -> np.ndarray:
    """Physicochemical property features (23 features)."""
    full_seq = vh + vl
    features = []

    # Global properties
    for prop in ['hydropathy', 'charge', 'aromatic', 'volume', 'polar']:
        values = [AA_PROPERTIES.get(aa, {}).get(prop, 0) for aa in full_seq]
        features.extend([np.mean(values), np.std(values), np.min(values), np.max(values)])

    # Hydrophobic ratio
    hydro_count = sum(1 for aa in full_seq if aa in HYDROPHOBIC_AAS)
    features.append(hydro_count / len(full_seq))

    # Aromatic ratio
    arom_count = sum(1 for aa in full_seq if aa in AROMATIC_AAS)
    features.append(arom_count / len(full_seq))

    # Net charge
    net_charge = sum(AA_PROPERTIES.get(aa, {}).get('charge', 0) for aa in full_seq)
    features.append(net_charge / len(full_seq))

    return np.array(features, dtype=np.float32)


def extract_regional_features(vh: str, vl: str) -> np.ndarray:
    """Regional features - CDR approximations (12 features)."""
    features = []

    # Approximate CDR regions (Kabat numbering approximation)
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

    return np.array(features, dtype=np.float32)


def extract_surface_exposure_features(vh: str, vl: str) -> np.ndarray:
    """Surface exposure approximation features (4 features)."""
    full_seq = vh + vl
    features = []

    # Exposed hydrophobic patches (consecutive hydrophobic residues)
    max_hydro_patch = 0
    current_patch = 0
    for aa in full_seq:
        if aa in HYDROPHOBIC_AAS:
            current_patch += 1
            max_hydro_patch = max(max_hydro_patch, current_patch)
        else:
            current_patch = 0
    features.append(max_hydro_patch)

    # Hydrophobic cluster density
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

    # Aromatic clustering
    arom_positions = [i for i, aa in enumerate(full_seq) if aa in AROMATIC_AAS]
    if len(arom_positions) > 1:
        avg_dist = np.mean(np.diff(sorted(arom_positions)))
        features.append(1 / (avg_dist + 1))  # Higher if clustered
    else:
        features.append(0)

    # Surface hydrophobicity score (simplified)
    surface_hydro = sum(AA_PROPERTIES.get(aa, {}).get('hydropathy', 0)
                        for aa in full_seq if AA_PROPERTIES.get(aa, {}).get('hydropathy', 0) > 0)
    features.append(surface_hydro / len(full_seq))

    return np.array(features, dtype=np.float32)


def extract_all_handcrafted_features(vh: str, vl: str) -> np.ndarray:
    """Extract all handcrafted features (59 total)."""
    return np.concatenate([
        extract_basic_features(vh, vl),           # 20
        extract_physicochemical_features(vh, vl), # 23
        extract_regional_features(vh, vl),        # 12
        extract_surface_exposure_features(vh, vl), # 4
    ])


# =============================================================================
# DeepSP Integration - Simplified SAP/SCM Approximation
# =============================================================================

# Since installing full DeepSP requires ANARCI and TensorFlow, we'll implement
# a simplified version that captures the key spatial aggregation concepts

# Eisenberg hydrophobicity scale (used by SAP)
EISENBERG_SCALE = {
    'A': 0.62, 'R': -2.53, 'N': -0.78, 'D': -0.90, 'C': 0.29,
    'Q': -0.85, 'E': -0.74, 'G': 0.48, 'H': -0.40, 'I': 1.38,
    'L': 1.06, 'K': -1.50, 'M': 0.64, 'F': 1.19, 'P': 0.12,
    'S': -0.18, 'T': -0.05, 'W': 0.81, 'Y': 0.26, 'V': 1.08,
}

# Black & Mould hydrophobicity scale
BLACK_MOULD_SCALE = {
    'A': 0.616, 'R': -0.693, 'N': -0.385, 'D': -0.447, 'C': 0.680,
    'Q': -0.339, 'E': -0.447, 'G': 0.501, 'H': -0.078, 'I': 1.245,
    'L': 1.191, 'K': -0.894, 'M': 0.878, 'F': 1.245, 'P': 0.385,
    'S': -0.093, 'T': 0.108, 'W': 1.214, 'Y': 0.447, 'V': 1.107,
}

# Kyte-Doolittle scale
KD_SCALE = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2,
}


def compute_local_sap(sequence: str, window: int = 7, scale: dict = None) -> list[float]:
    """
    Compute local SAP-like scores using sliding window.

    SAP = sum of (hydrophobicity * exposure factor) for residues in window
    We approximate exposure by position in the window (center more exposed).
    """
    if scale is None:
        scale = EISENBERG_SCALE

    scores = []
    half_win = window // 2

    for i in range(len(sequence)):
        start = max(0, i - half_win)
        end = min(len(sequence), i + half_win + 1)
        window_seq = sequence[start:end]

        # Compute weighted hydrophobicity
        total = 0
        for j, aa in enumerate(window_seq):
            hydro = scale.get(aa, 0)
            # Weight by distance from center (Gaussian-like)
            center = len(window_seq) // 2
            dist = abs(j - center)
            weight = 1.0 / (1 + dist * 0.5)
            total += hydro * weight

        scores.append(total / len(window_seq))

    return scores


def compute_sap_features(vh: str, vl: str) -> np.ndarray:
    """
    Compute SAP-inspired features for the antibody.

    Returns features that approximate DeepSP's spatial aggregation propensity scores:
    - Regional SAP scores (VH, VL, CDRs)
    - Multiple hydrophobicity scales (Eisenberg, Black-Mould, Kyte-Doolittle)
    - Aggregation hotspot detection
    """
    features = []

    # Full Fv region
    fv_sequence = vh + vl

    # Compute SAP scores with different scales
    for scale_name, scale in [('eisenberg', EISENBERG_SCALE),
                               ('bm', BLACK_MOULD_SCALE),
                               ('kd', KD_SCALE)]:
        sap_scores = compute_local_sap(fv_sequence, window=7, scale=scale)

        # Global statistics
        features.extend([
            np.mean(sap_scores),
            np.std(sap_scores),
            np.max(sap_scores),
            np.min(sap_scores),
            np.percentile(sap_scores, 90),  # High SAP regions
            np.percentile(sap_scores, 10),  # Low SAP regions
        ])

        # Count hotspots (high SAP residues)
        threshold = np.mean(sap_scores) + np.std(sap_scores)
        hotspots = sum(1 for s in sap_scores if s > threshold)
        features.append(hotspots / len(sap_scores))

    # Regional SAP for VH and VL separately
    for chain, seq in [('vh', vh), ('vl', vl)]:
        sap_scores = compute_local_sap(seq, window=7, scale=EISENBERG_SCALE)
        features.extend([np.mean(sap_scores), np.std(sap_scores), np.max(sap_scores)])

    # CDR-specific SAP (most important for aggregation)
    cdr_h3 = vh[95:110] if len(vh) > 110 else vh[95:]
    cdr_l3 = vl[89:97] if len(vl) > 97 else vl[89:]

    for cdr_name, cdr in [('cdrh3', cdr_h3), ('cdrl3', cdr_l3)]:
        if len(cdr) >= 3:
            sap_scores = compute_local_sap(cdr, window=3, scale=EISENBERG_SCALE)
            features.extend([np.mean(sap_scores), np.max(sap_scores)])
        else:
            features.extend([0.0, 0.0])

    return np.array(features, dtype=np.float32)


def compute_scm_features(vh: str, vl: str) -> np.ndarray:
    """
    Compute SCM (Spatial Charge Map) inspired features.

    SCM measures the distribution of charged residues on the surface.
    Important for viscosity and colloidal stability.
    """
    features = []
    fv_sequence = vh + vl

    # Charge distribution
    charges = [AA_PROPERTIES.get(aa, {}).get('charge', 0) for aa in fv_sequence]

    # Global charge features
    features.extend([
        np.mean(charges),
        np.std(charges),
        sum(1 for c in charges if c > 0) / len(charges),  # Positive ratio
        sum(1 for c in charges if c < 0) / len(charges),  # Negative ratio
    ])

    # Charge clustering (spatial charge map concept)
    # Look for clusters of same-sign charges
    pos_clusters = 0
    neg_clusters = 0
    in_pos = False
    in_neg = False

    for c in charges:
        if c > 0:
            if not in_pos:
                pos_clusters += 1
                in_pos = True
            in_neg = False
        elif c < 0:
            if not in_neg:
                neg_clusters += 1
                in_neg = True
            in_pos = False
        else:
            in_pos = in_neg = False

    features.extend([
        pos_clusters / len(charges),
        neg_clusters / len(charges),
    ])

    # Charge asymmetry between VH and VL
    vh_charge = sum(AA_PROPERTIES.get(aa, {}).get('charge', 0) for aa in vh)
    vl_charge = sum(AA_PROPERTIES.get(aa, {}).get('charge', 0) for aa in vl)
    features.append((vh_charge - vl_charge) / (abs(vh_charge) + abs(vl_charge) + 1e-6))

    # Dipole moment approximation (charge * position)
    dipole = sum(c * i for i, c in enumerate(charges))
    features.append(dipole / (len(charges) ** 2))

    return np.array(features, dtype=np.float32)


def compute_psh_features(vh: str, vl: str) -> np.ndarray:
    """
    Compute PSH (Patches of Surface Hydrophobicity) inspired features.

    PSH identifies large hydrophobic patches that could cause aggregation.
    """
    features = []
    fv_sequence = vh + vl

    # Hydrophobicity profile using Kyte-Doolittle
    hydro = [KD_SCALE.get(aa, 0) for aa in fv_sequence]

    # Sliding window to find patches
    window_size = 9
    patch_scores = []
    for i in range(len(hydro) - window_size + 1):
        window = hydro[i:i + window_size]
        # Only count if mostly hydrophobic
        if np.mean(window) > 0:
            patch_scores.append(np.mean(window))

    if patch_scores:
        features.extend([
            np.mean(patch_scores),
            np.max(patch_scores),
            len([p for p in patch_scores if p > 1.5]) / len(patch_scores),  # Strong patches ratio
        ])
    else:
        features.extend([0.0, 0.0, 0.0])

    # Largest continuous hydrophobic patch
    max_patch_size = 0
    current_patch = 0
    current_sum = 0
    max_patch_sum = 0

    for h in hydro:
        if h > 0:  # Hydrophobic
            current_patch += 1
            current_sum += h
            if current_patch > max_patch_size:
                max_patch_size = current_patch
                max_patch_sum = current_sum
        else:
            current_patch = 0
            current_sum = 0

    features.extend([
        max_patch_size / len(fv_sequence),
        max_patch_sum / len(fv_sequence),
    ])

    return np.array(features, dtype=np.float32)


def compute_ripley_k_features(vh: str, vl: str) -> np.ndarray:
    """
    Compute Ripley's K-inspired features (used by PROPERMAB).

    Ripley's K measures spatial clustering of features (charges, aromatics).
    """
    features = []
    fv_sequence = vh + vl

    # Find positions of charged and aromatic residues
    pos_charged = [i for i, aa in enumerate(fv_sequence) if AA_PROPERTIES.get(aa, {}).get('charge', 0) > 0]
    neg_charged = [i for i, aa in enumerate(fv_sequence) if AA_PROPERTIES.get(aa, {}).get('charge', 0) < 0]
    aromatic = [i for i, aa in enumerate(fv_sequence) if aa in AROMATIC_AAS]

    # Ripley's K: count pairs within distance threshold
    def ripley_k(positions: list, threshold: int = 5) -> float:
        if len(positions) < 2:
            return 0.0
        count = 0
        for i, p1 in enumerate(positions):
            for p2 in positions[i+1:]:
                if abs(p1 - p2) <= threshold:
                    count += 1
        n = len(positions)
        return count / (n * (n - 1) / 2) if n > 1 else 0

    # K values at different thresholds
    for threshold in [3, 5, 10]:
        features.append(ripley_k(pos_charged, threshold))
        features.append(ripley_k(neg_charged, threshold))
        features.append(ripley_k(aromatic, threshold))

    # Clustering index (deviation from random)
    seq_len = len(fv_sequence)
    for positions, name in [(pos_charged, 'pos'), (neg_charged, 'neg'), (aromatic, 'arom')]:
        if len(positions) >= 2:
            avg_dist = np.mean(np.diff(sorted(positions)))
            expected_dist = seq_len / (len(positions) + 1)
            clustering = expected_dist / (avg_dist + 1e-6)  # >1 means clustered
            features.append(clustering)
        else:
            features.append(1.0)

    return np.array(features, dtype=np.float32)


def extract_all_structural_features(vh: str, vl: str) -> np.ndarray:
    """
    Extract all structural-like features (SAP, SCM, PSH, Ripley's K).

    These approximate DeepSP's 30 spatial properties without requiring
    actual structure prediction or MD simulation.

    Total: 50+ features capturing spatial aggregation propensity.
    """
    return np.concatenate([
        compute_sap_features(vh, vl),      # ~31 features
        compute_scm_features(vh, vl),       # ~8 features
        compute_psh_features(vh, vl),       # ~5 features
        compute_ripley_k_features(vh, vl),  # ~12 features
    ])


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
    """Load HIC retention time data from Jain 2017."""
    flab_dir = data_dir / "flab"
    hicrt_file = flab_dir / "jain2017biophyscial_HICRT.csv"

    antibodies = []
    if hicrt_file.exists():
        with open(hicrt_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    col = [c for c in row.keys() if 'HIC' in c and 'Retention' in c][0]
                    hic_rt = float(row[col])
                    antibodies.append(AntibodyHIC(
                        vh_sequence=row['heavy'],
                        vl_sequence=row['light'],
                        hic_rt=hic_rt
                    ))
                except (ValueError, KeyError, IndexError):
                    continue

    return antibodies


# =============================================================================
# Experiments
# =============================================================================

def run_cv_experiment(
    name: str,
    X: np.ndarray,
    y: np.ndarray,
    model,
    n_folds: int = 5,
) -> ExperimentResult:
    """Run cross-validation experiment and return results."""
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_rhos = []
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Fit and predict
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_val_scaled)

        rho = spearman_correlation(y_val.tolist(), y_pred.tolist())
        fold_rhos.append(rho)

    return ExperimentResult(
        name=name,
        rho=np.mean(fold_rhos),
        rho_std=np.std(fold_rhos),
        fold_rhos=fold_rhos,
    )


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("HIC Prediction with Structural Features (DeepSP-inspired)")
    print("=" * 70)
    print("\nGoal: Reach PROPERMAB-level correlation (ρ ≈ 0.75)")
    print("Current best (handcrafted features): ρ = 0.553")
    print("=" * 70)

    # Load data
    data_dir = Path(__file__).parent / "data"
    antibodies = load_hic_data(data_dir)
    print(f"\nLoaded {len(antibodies)} antibodies with HIC retention times")

    if len(antibodies) == 0:
        print("No data found! Please run download_datasets.py first.")
        return 1

    # Prepare labels
    y = np.array([ab.hic_rt for ab in antibodies])

    results = []

    # ==========================================================================
    # Experiment 1: Handcrafted features only (baseline)
    # ==========================================================================
    print("\n" + "-" * 50)
    print("Experiment 1: Handcrafted Features Only (Baseline)")
    print("-" * 50)

    X_handcrafted = np.array([
        extract_all_handcrafted_features(ab.vh_sequence, ab.vl_sequence)
        for ab in antibodies
    ])
    print(f"  Features: {X_handcrafted.shape[1]}")

    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42
    )

    result = run_cv_experiment("Handcrafted_only", X_handcrafted, y, model)
    results.append(result)
    print(f"  ρ = {result.rho:.3f} ± {result.rho_std:.3f}")
    print(f"  Folds: {[f'{r:.3f}' for r in result.fold_rhos]}")

    # ==========================================================================
    # Experiment 2: Structural features only (SAP, SCM, etc.)
    # ==========================================================================
    print("\n" + "-" * 50)
    print("Experiment 2: Structural Features Only (SAP/SCM/PSH)")
    print("-" * 50)

    X_structural = np.array([
        extract_all_structural_features(ab.vh_sequence, ab.vl_sequence)
        for ab in antibodies
    ])
    print(f"  Features: {X_structural.shape[1]}")

    result = run_cv_experiment("Structural_only", X_structural, y, model)
    results.append(result)
    print(f"  ρ = {result.rho:.3f} ± {result.rho_std:.3f}")
    print(f"  Folds: {[f'{r:.3f}' for r in result.fold_rhos]}")

    # ==========================================================================
    # Experiment 3: Combined features (handcrafted + structural)
    # ==========================================================================
    print("\n" + "-" * 50)
    print("Experiment 3: Combined Features (Handcrafted + Structural)")
    print("-" * 50)

    X_combined = np.hstack([X_handcrafted, X_structural])
    print(f"  Features: {X_combined.shape[1]}")

    result = run_cv_experiment("Combined_features", X_combined, y, model)
    results.append(result)
    print(f"  ρ = {result.rho:.3f} ± {result.rho_std:.3f}")
    print(f"  Folds: {[f'{r:.3f}' for r in result.fold_rhos]}")

    # ==========================================================================
    # Experiment 4: Optimized GBM with combined features
    # ==========================================================================
    print("\n" + "-" * 50)
    print("Experiment 4: Optimized GBM with Combined Features")
    print("-" * 50)

    for n_est, depth, lr in [(300, 3, 0.03), (300, 4, 0.05), (500, 3, 0.02)]:
        model_opt = GradientBoostingRegressor(
            n_estimators=n_est, max_depth=depth, learning_rate=lr,
            subsample=0.8, min_samples_leaf=5, random_state=42
        )

        result = run_cv_experiment(
            f"GBM_combined_n{n_est}_d{depth}_lr{lr}",
            X_combined, y, model_opt
        )
        results.append(result)
        print(f"  n={n_est}, d={depth}, lr={lr}: ρ = {result.rho:.3f} ± {result.rho_std:.3f}")

    # ==========================================================================
    # Experiment 5: Random Forest with combined features
    # ==========================================================================
    print("\n" + "-" * 50)
    print("Experiment 5: Random Forest with Combined Features")
    print("-" * 50)

    model_rf = RandomForestRegressor(
        n_estimators=300, max_depth=8, min_samples_leaf=3,
        random_state=42, n_jobs=-1
    )

    result = run_cv_experiment("RF_combined", X_combined, y, model_rf)
    results.append(result)
    print(f"  ρ = {result.rho:.3f} ± {result.rho_std:.3f}")
    print(f"  Folds: {[f'{r:.3f}' for r in result.fold_rhos]}")

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    # Sort by performance
    results.sort(key=lambda x: x.rho, reverse=True)

    print(f"\n{'Model':<45} {'ρ':>8} {'± std':>8}")
    print("-" * 65)
    for r in results:
        print(f"{r.name:<45} {r.rho:>8.3f} {r.rho_std:>8.3f}")

    best = results[0]
    print(f"\n✅ Best model: {best.name}")
    print(f"   Correlation: ρ = {best.rho:.3f} ± {best.rho_std:.3f}")

    # Comparison with SOTA
    print("\n" + "-" * 50)
    print("Comparison with Published Methods:")
    print("-" * 50)
    print(f"  Current ML HIC:     ρ = 0.351")
    print(f"  Our best (prev):    ρ = 0.553 (GBM + handcrafted)")
    print(f"  Our best (new):     ρ = {best.rho:.3f} (+ structural features)")
    print(f"  PROPERMAB (target): ρ = 0.75")

    improvement = (best.rho - 0.553) / 0.553 * 100
    gap_to_sota = 0.75 - best.rho
    print(f"\n  Improvement from baseline: {improvement:+.1f}%")
    print(f"  Gap to PROPERMAB: {gap_to_sota:.3f}")

    # Save results
    results_file = Path(__file__).parent / "structural_features_results.json"
    with open(results_file, 'w') as f:
        json.dump([{
            'name': r.name,
            'rho': r.rho,
            'rho_std': r.rho_std,
            'fold_rhos': r.fold_rhos,
        } for r in results], f, indent=2)
    print(f"\nResults saved to {results_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
