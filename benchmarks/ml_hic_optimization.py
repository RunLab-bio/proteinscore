#!/usr/bin/env python3
"""
ML-Based HIC Retention Time Prediction Optimization

This script implements two approaches to improve HIC prediction:
1. CDR-Focused Feature Engineering + ML Models
2. SAP Proxy with Surface Exposure Weights

Goal: Improve from current ρ = -0.13 to ρ ≈ 0.25-0.40 (theoretical max for sequence-only)

Reference SOTA:
- PROPERMAB (structure-based): ρ = 0.75
- FLAb2 shows sequence-only max: ρ ≈ 0.35-0.40
"""

import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from proteinscore.antibody.cdr import CDRDetector

# =============================================================================
# Hydrophobicity Scales
# =============================================================================

KYTE_DOOLITTLE = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
    'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
    'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
    'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}

WIMLEY_WHITE = {
    'A': 0.17, 'R': -0.81, 'N': -0.42, 'D': -1.23, 'C': 0.24,
    'Q': -0.58, 'E': -2.02, 'G': 0.01, 'H': -0.96, 'I': 0.31,
    'L': 0.56, 'K': -0.99, 'M': 0.23, 'F': 1.13, 'P': -0.45,
    'S': -0.13, 'T': -0.14, 'W': 1.85, 'Y': 0.94, 'V': -0.07
}

EISENBERG = {
    'A': 0.62, 'R': -2.53, 'N': -0.78, 'D': -0.90, 'C': 0.29,
    'Q': -0.85, 'E': -0.74, 'G': 0.48, 'H': -0.40, 'I': 1.38,
    'L': 1.06, 'K': -1.50, 'M': 0.64, 'F': 1.19, 'P': 0.12,
    'S': -0.18, 'T': -0.05, 'W': 0.81, 'Y': 0.26, 'V': 1.08
}

# Aromatic residues (very hydrophobic in surface context)
AROMATIC_AA = set('FYW')
POSITIVE_AA = set('KRH')
NEGATIVE_AA = set('DE')
HYDROPHOBIC_AA = set('VILMFYW')


# =============================================================================
# Surface Accessibility Proxy (SAP) Weights
# =============================================================================

# Based on structural analysis of antibody Fv regions
# Chothia numbering - positions known to be surface-exposed
# Higher weight = more exposed to solvent
SAP_WEIGHTS_HEAVY = {
    # CDR-H1 (always exposed)
    **{i: 1.0 for i in range(31, 36)},
    # CDR-H2 (always exposed)
    **{i: 1.0 for i in range(50, 66)},
    # CDR-H3 (always exposed, variable length)
    **{i: 1.0 for i in range(95, 103)},
    # Framework exposed positions (from structural analysis)
    28: 0.8, 30: 0.8,  # Near CDR-H1
    47: 0.6, 49: 0.6,  # Near CDR-H2
    73: 0.5, 74: 0.5, 76: 0.5,  # Outer framework
    # Buried core (low weight)
    **{i: 0.1 for i in [4, 22, 36, 69, 78, 88, 92]},
}

SAP_WEIGHTS_LIGHT = {
    # CDR-L1 (always exposed)
    **{i: 1.0 for i in range(24, 35)},
    # CDR-L2 (always exposed)
    **{i: 1.0 for i in range(50, 57)},
    # CDR-L3 (always exposed)
    **{i: 1.0 for i in range(89, 98)},
    # Framework exposed positions
    46: 0.7, 49: 0.7,
    # Buried core
    **{i: 0.1 for i in [4, 22, 35, 71, 88]},
}


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class AntibodyHIC:
    """Antibody with HIC retention data."""
    vh_sequence: str
    vl_sequence: str
    hic_rt: float  # HIC Retention Time (minutes)


@dataclass
class CDRFeatures:
    """Features extracted from CDR regions."""
    # Per-CDR features
    cdr_lengths: list[int]  # 6 CDRs
    cdr_hydro_mean: list[float]  # Mean hydrophobicity per CDR
    cdr_hydro_max: list[float]  # Max hydrophobicity per CDR
    cdr_aromatic_frac: list[float]  # Aromatic fraction per CDR
    cdr_positive_frac: list[float]  # Positive charge fraction
    cdr_negative_frac: list[float]  # Negative charge fraction

    # Global features
    total_length: int
    total_hydro_mean: float
    total_aromatic_frac: float
    net_charge_frac: float
    gravy: float  # Grand Average of Hydropathy


# =============================================================================
# Feature Extraction
# =============================================================================

def calculate_hydrophobicity(sequence: str, scale: dict = KYTE_DOOLITTLE) -> float:
    """Calculate mean hydrophobicity of a sequence."""
    if not sequence:
        return 0.0
    return sum(scale.get(aa, 0) for aa in sequence.upper()) / len(sequence)


def calculate_max_hydrophobicity(sequence: str, window: int = 5, scale: dict = KYTE_DOOLITTLE) -> float:
    """Calculate maximum hydrophobicity in sliding window."""
    if len(sequence) < window:
        return calculate_hydrophobicity(sequence, scale)

    max_hydro = float('-inf')
    for i in range(len(sequence) - window + 1):
        window_seq = sequence[i:i+window]
        hydro = calculate_hydrophobicity(window_seq, scale)
        max_hydro = max(max_hydro, hydro)

    return max_hydro


def calculate_aa_fraction(sequence: str, aa_set: set) -> float:
    """Calculate fraction of amino acids in a set."""
    if not sequence:
        return 0.0
    return sum(1 for aa in sequence.upper() if aa in aa_set) / len(sequence)


def extract_cdr_features(vh: str, vl: str) -> CDRFeatures:
    """Extract CDR-focused features for HIC prediction."""
    detector = CDRDetector()

    # Detect CDRs
    vh_cdrs = detector.detect_cdrs(vh, chain_type="heavy")
    vl_cdrs = detector.detect_cdrs(vl, chain_type="light")

    # Get CDR sequences using the correct API
    vh_cdr_dict = vh_cdrs.cdr_sequences
    vl_cdr_dict = vl_cdrs.cdr_sequences

    cdr_sequences = [
        vh_cdr_dict.get("CDR1", vh_cdr_dict.get("H1", "")),
        vh_cdr_dict.get("CDR2", vh_cdr_dict.get("H2", "")),
        vh_cdr_dict.get("CDR3", vh_cdr_dict.get("H3", "")),
        vl_cdr_dict.get("CDR1", vl_cdr_dict.get("L1", "")),
        vl_cdr_dict.get("CDR2", vl_cdr_dict.get("L2", "")),
        vl_cdr_dict.get("CDR3", vl_cdr_dict.get("L3", "")),
    ]

    # Extract per-CDR features
    cdr_lengths = [len(s) for s in cdr_sequences]
    cdr_hydro_mean = [calculate_hydrophobicity(s, KYTE_DOOLITTLE) for s in cdr_sequences]
    cdr_hydro_max = [calculate_max_hydrophobicity(s, 3, KYTE_DOOLITTLE) for s in cdr_sequences]
    cdr_aromatic_frac = [calculate_aa_fraction(s, AROMATIC_AA) for s in cdr_sequences]
    cdr_positive_frac = [calculate_aa_fraction(s, POSITIVE_AA) for s in cdr_sequences]
    cdr_negative_frac = [calculate_aa_fraction(s, NEGATIVE_AA) for s in cdr_sequences]

    # Global features
    full_seq = vh + vl
    total_length = len(full_seq)
    total_hydro_mean = calculate_hydrophobicity(full_seq, KYTE_DOOLITTLE)
    total_aromatic_frac = calculate_aa_fraction(full_seq, AROMATIC_AA)

    pos_count = sum(1 for aa in full_seq if aa in POSITIVE_AA)
    neg_count = sum(1 for aa in full_seq if aa in NEGATIVE_AA)
    net_charge_frac = (pos_count - neg_count) / total_length if total_length > 0 else 0

    gravy = calculate_hydrophobicity(full_seq, KYTE_DOOLITTLE)

    return CDRFeatures(
        cdr_lengths=cdr_lengths,
        cdr_hydro_mean=cdr_hydro_mean,
        cdr_hydro_max=cdr_hydro_max,
        cdr_aromatic_frac=cdr_aromatic_frac,
        cdr_positive_frac=cdr_positive_frac,
        cdr_negative_frac=cdr_negative_frac,
        total_length=total_length,
        total_hydro_mean=total_hydro_mean,
        total_aromatic_frac=total_aromatic_frac,
        net_charge_frac=net_charge_frac,
        gravy=gravy,
    )


def features_to_array(features: CDRFeatures) -> list[float]:
    """Convert CDRFeatures to flat array for ML."""
    return (
        features.cdr_lengths +
        features.cdr_hydro_mean +
        features.cdr_hydro_max +
        features.cdr_aromatic_frac +
        features.cdr_positive_frac +
        features.cdr_negative_frac +
        [
            features.total_length,
            features.total_hydro_mean,
            features.total_aromatic_frac,
            features.net_charge_frac,
            features.gravy,
        ]
    )


def get_feature_names() -> list[str]:
    """Get names for all features."""
    cdr_names = ['H1', 'H2', 'H3', 'L1', 'L2', 'L3']
    names = []
    names += [f'len_{c}' for c in cdr_names]
    names += [f'hydro_mean_{c}' for c in cdr_names]
    names += [f'hydro_max_{c}' for c in cdr_names]
    names += [f'aromatic_{c}' for c in cdr_names]
    names += [f'positive_{c}' for c in cdr_names]
    names += [f'negative_{c}' for c in cdr_names]
    names += ['total_length', 'total_hydro', 'total_aromatic', 'net_charge', 'gravy']
    return names


# =============================================================================
# SAP Proxy Calculation (Solution 2)
# =============================================================================

def calculate_sap_score(vh: str, vl: str) -> float:
    """
    Calculate Surface Accessibility Proxy (SAP) score.

    Uses implicit structural knowledge - certain positions in the Fv
    are always exposed due to the conserved Ig fold.

    Higher score = more hydrophobic surface = higher HIC retention.
    """
    sap_score = 0.0
    total_weight = 0.0

    # Heavy chain SAP
    for i, aa in enumerate(vh.upper()):
        pos = i + 1  # 1-indexed
        weight = SAP_WEIGHTS_HEAVY.get(pos, 0.3)  # Default: moderate exposure
        hydro = WIMLEY_WHITE.get(aa, 0)
        sap_score += weight * hydro
        total_weight += weight

    # Light chain SAP
    for i, aa in enumerate(vl.upper()):
        pos = i + 1
        weight = SAP_WEIGHTS_LIGHT.get(pos, 0.3)
        hydro = WIMLEY_WHITE.get(aa, 0)
        sap_score += weight * hydro
        total_weight += weight

    # Normalize
    if total_weight > 0:
        sap_score /= total_weight

    return sap_score


def calculate_cdr_sap_score(vh: str, vl: str) -> float:
    """
    Calculate SAP score focused on CDR regions only.

    CDRs are always exposed and dominate HIC behavior.
    """
    detector = CDRDetector()

    vh_cdrs = detector.detect_cdrs(vh, chain_type="heavy")
    vl_cdrs = detector.detect_cdrs(vl, chain_type="light")

    # Get CDR sequences using correct API
    vh_cdr_dict = vh_cdrs.cdr_sequences
    vl_cdr_dict = vl_cdrs.cdr_sequences

    # Combine all CDR sequences
    cdr_seq = (
        vh_cdr_dict.get("CDR1", vh_cdr_dict.get("H1", "")) +
        vh_cdr_dict.get("CDR2", vh_cdr_dict.get("H2", "")) +
        vh_cdr_dict.get("CDR3", vh_cdr_dict.get("H3", "")) +
        vl_cdr_dict.get("CDR1", vl_cdr_dict.get("L1", "")) +
        vl_cdr_dict.get("CDR2", vl_cdr_dict.get("L2", "")) +
        vl_cdr_dict.get("CDR3", vl_cdr_dict.get("L3", ""))
    )

    if not cdr_seq:
        return 0.0

    # Weight aromatic residues more heavily (they dominate surface hydrophobicity)
    aromatic_weight = 2.0

    score = 0.0
    for aa in cdr_seq.upper():
        hydro = WIMLEY_WHITE.get(aa, 0)
        if aa in AROMATIC_AA:
            score += hydro * aromatic_weight
        else:
            score += hydro

    return score / len(cdr_seq)


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


# =============================================================================
# Data Loading
# =============================================================================

def load_hic_data(data_dir: Path) -> list[AntibodyHIC]:
    """Load HIC retention time data from all available sources."""
    antibodies = []

    # Jain 2017 biophysical HIC RT
    hic_file = data_dir / "flab" / "jain2017biophyscial_HICRT.csv"
    if hic_file.exists():
        with open(hic_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    hic_col = [c for c in row.keys() if 'HIC' in c][0]
                    antibodies.append(AntibodyHIC(
                        vh_sequence=row['heavy'],
                        vl_sequence=row['light'],
                        hic_rt=float(row[hic_col])
                    ))
                except (ValueError, KeyError, IndexError):
                    pass

    # Jain 2024 HIC data
    hic_file2 = data_dir / "flab" / "jain2024assessment_HIC.csv"
    if hic_file2.exists():
        with open(hic_file2) as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    hic_col = [c for c in row.keys() if 'HIC' in c][0]
                    # Avoid duplicates
                    key = (row['heavy'], row['light'])
                    if not any(a.vh_sequence == row['heavy'] and a.vl_sequence == row['light'] for a in antibodies):
                        antibodies.append(AntibodyHIC(
                            vh_sequence=row['heavy'],
                            vl_sequence=row['light'],
                            hic_rt=float(row[hic_col])
                        ))
                except (ValueError, KeyError, IndexError):
                    pass

    return antibodies


# =============================================================================
# ML Model Training (Solution 1)
# =============================================================================

def train_ridge_model(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> tuple[np.ndarray, float]:
    """
    Train Ridge regression model (L2 regularization).

    Ridge: min ||y - Xw||^2 + alpha * ||w||^2
    Solution: w = (X^T X + alpha * I)^(-1) X^T y
    """
    n_features = X.shape[1]

    # Add regularization
    XtX = X.T @ X + alpha * np.eye(n_features)
    Xty = X.T @ y

    # Solve
    weights = np.linalg.solve(XtX, Xty)

    # Calculate intercept (we use centered data, so intercept = mean(y))
    intercept = np.mean(y) - np.mean(X @ weights)

    return weights, intercept


def cross_validate(X: np.ndarray, y: np.ndarray, n_folds: int = 5, alpha: float = 1.0) -> list[float]:
    """K-fold cross-validation returning Spearman correlations."""
    n = len(y)
    fold_size = n // n_folds
    indices = list(range(n))

    # Shuffle deterministically
    np.random.seed(42)
    np.random.shuffle(indices)

    correlations = []

    for fold in range(n_folds):
        # Split
        test_start = fold * fold_size
        test_end = test_start + fold_size if fold < n_folds - 1 else n
        test_idx = indices[test_start:test_end]
        train_idx = indices[:test_start] + indices[test_end:]

        X_train = X[train_idx]
        y_train = y[train_idx]
        X_test = X[test_idx]
        y_test = y[test_idx]

        # Normalize
        X_mean = np.mean(X_train, axis=0)
        X_std = np.std(X_train, axis=0) + 1e-8
        X_train_norm = (X_train - X_mean) / X_std
        X_test_norm = (X_test - X_mean) / X_std

        # Train
        weights, intercept = train_ridge_model(X_train_norm, y_train, alpha)

        # Predict
        y_pred = X_test_norm @ weights + intercept

        # Evaluate
        rho, _ = spearman_correlation(y_test.tolist(), y_pred.tolist())
        correlations.append(rho)

    return correlations


def grid_search_alpha(X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Find best alpha for Ridge regression."""
    alphas = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]

    best_alpha = 1.0
    best_mean_rho = -1.0

    for alpha in alphas:
        rhos = cross_validate(X, y, n_folds=5, alpha=alpha)
        mean_rho = np.mean(rhos)

        if mean_rho > best_mean_rho:
            best_mean_rho = mean_rho
            best_alpha = alpha

    return best_alpha, best_mean_rho


# =============================================================================
# Main Benchmark
# =============================================================================

def run_hic_ml_benchmark(data_dir: Path):
    """Run comprehensive HIC prediction benchmark."""
    print("=" * 70)
    print("HIC RETENTION TIME PREDICTION - ML OPTIMIZATION")
    print("=" * 70)
    print()

    # Load data
    antibodies = load_hic_data(data_dir)
    print(f"Loaded {len(antibodies)} antibodies with HIC data")
    print()

    if len(antibodies) < 10:
        print("ERROR: Not enough data for ML training")
        return

    # ==========================================================================
    # Solution 1: CDR-Focused Feature Engineering
    # ==========================================================================
    print("-" * 70)
    print("SOLUTION 1: CDR-Focused Feature Engineering + Ridge Regression")
    print("-" * 70)
    print()

    # Extract features
    print("Extracting CDR features...")
    features_list = []
    hic_values = []

    for ab in antibodies:
        try:
            feat = extract_cdr_features(ab.vh_sequence, ab.vl_sequence)
            features_list.append(features_to_array(feat))
            hic_values.append(ab.hic_rt)
        except Exception as e:
            print(f"  Warning: Could not process antibody: {e}")

    X = np.array(features_list)
    y = np.array(hic_values)

    print(f"  Features extracted: {X.shape[1]} features, {X.shape[0]} samples")
    print()

    # Feature names
    feature_names = get_feature_names()

    # Calculate individual feature correlations
    print("Individual feature correlations with HIC RT:")
    correlations = []
    for i, name in enumerate(feature_names):
        rho, p = spearman_correlation(X[:, i].tolist(), y.tolist())
        correlations.append((name, rho, p))

    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("\nTop 10 features:")
    for name, rho, p in correlations[:10]:
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        print(f"  {name:<20} ρ = {rho:+.3f} {sig}")

    print()

    # Grid search for best alpha
    print("Running grid search for regularization parameter...")
    best_alpha, best_cv_rho = grid_search_alpha(X, y)
    print(f"  Best alpha: {best_alpha}")
    print(f"  Best CV Spearman ρ: {best_cv_rho:.3f}")
    print()

    # Final cross-validation with best alpha
    print("5-Fold Cross-Validation Results (Ridge Regression):")
    cv_rhos = cross_validate(X, y, n_folds=5, alpha=best_alpha)
    print(f"  Fold correlations: {[f'{r:.3f}' for r in cv_rhos]}")
    print(f"  Mean ± Std: {np.mean(cv_rhos):.3f} ± {np.std(cv_rhos):.3f}")
    print()

    # Train final model on all data
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std

    weights, intercept = train_ridge_model(X_norm, y, best_alpha)

    # Feature importance
    print("Top 10 Feature Importances (by weight magnitude):")
    importance = [(feature_names[i], abs(weights[i]), weights[i]) for i in range(len(weights))]
    importance.sort(key=lambda x: x[1], reverse=True)
    for name, abs_w, w in importance[:10]:
        print(f"  {name:<20} weight = {w:+.4f}")

    print()

    # ==========================================================================
    # Solution 2: SAP Proxy with Exposure Weights
    # ==========================================================================
    print("-" * 70)
    print("SOLUTION 2: Surface Accessibility Proxy (SAP)")
    print("-" * 70)
    print()

    # Calculate SAP scores
    sap_scores = []
    cdr_sap_scores = []

    for ab in antibodies:
        sap = calculate_sap_score(ab.vh_sequence, ab.vl_sequence)
        cdr_sap = calculate_cdr_sap_score(ab.vh_sequence, ab.vl_sequence)
        sap_scores.append(sap)
        cdr_sap_scores.append(cdr_sap)

    # Evaluate SAP methods
    sap_rho, sap_p = spearman_correlation(sap_scores, hic_values)
    cdr_sap_rho, cdr_sap_p = spearman_correlation(cdr_sap_scores, hic_values)

    print(f"Full SAP Score vs HIC RT:")
    sig = "***" if sap_p < 0.001 else "**" if sap_p < 0.01 else "*" if sap_p < 0.05 else ""
    print(f"  Spearman ρ = {sap_rho:.3f} (p = {sap_p:.4f}) {sig}")
    print()

    print(f"CDR-only SAP Score vs HIC RT:")
    sig = "***" if cdr_sap_p < 0.001 else "**" if cdr_sap_p < 0.01 else "*" if cdr_sap_p < 0.05 else ""
    print(f"  Spearman ρ = {cdr_sap_rho:.3f} (p = {cdr_sap_p:.4f}) {sig}")
    print()

    # ==========================================================================
    # Combined Model
    # ==========================================================================
    print("-" * 70)
    print("COMBINED MODEL: CDR Features + SAP Scores")
    print("-" * 70)
    print()

    # Add SAP scores to features
    X_combined = np.column_stack([X, sap_scores, cdr_sap_scores])
    feature_names_combined = feature_names + ['sap_full', 'sap_cdr']

    # Grid search
    best_alpha_comb, best_cv_rho_comb = grid_search_alpha(X_combined, y)
    print(f"Best alpha: {best_alpha_comb}")
    print()

    # Cross-validation
    cv_rhos_comb = cross_validate(X_combined, y, n_folds=5, alpha=best_alpha_comb)
    print(f"5-Fold CV Results (Combined Model):")
    print(f"  Fold correlations: {[f'{r:.3f}' for r in cv_rhos_comb]}")
    print(f"  Mean ± Std: {np.mean(cv_rhos_comb):.3f} ± {np.std(cv_rhos_comb):.3f}")
    print()

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()

    print("| Method                      | Spearman ρ | Status      |")
    print("|-----------------------------|-----------:|-------------|")
    print(f"| Current HIC Proxy           |     -0.130 | ❌ Baseline |")
    print(f"| Solution 1: CDR ML          |     {np.mean(cv_rhos):+.3f} | {'✅' if np.mean(cv_rhos) > 0.2 else '⚠️'} Improved  |")
    print(f"| Solution 2: SAP Full        |     {sap_rho:+.3f} | {'✅' if sap_rho > 0.2 else '⚠️'} SAP       |")
    print(f"| Solution 2: SAP CDR         |     {cdr_sap_rho:+.3f} | {'✅' if cdr_sap_rho > 0.2 else '⚠️'} SAP-CDR   |")
    print(f"| Combined (CDR + SAP)        |     {np.mean(cv_rhos_comb):+.3f} | {'✅' if np.mean(cv_rhos_comb) > 0.2 else '⚠️'} Best      |")
    print(f"| SOTA (PROPERMAB, 3D)        |     +0.750 | 🎯 Target   |")
    print(f"| Theoretical Max (seq-only)  |  ~0.35-0.40| 📊 FLAb2    |")
    print()

    best_rho = max(np.mean(cv_rhos), sap_rho, cdr_sap_rho, np.mean(cv_rhos_comb))

    if best_rho > 0.30:
        print("✅ SUCCESS: Achieved near-theoretical-maximum for sequence-only prediction!")
    elif best_rho > 0.20:
        print("✅ GOOD: Significant improvement over baseline")
    elif best_rho > 0:
        print("⚠️  MODERATE: Positive correlation achieved, but room for improvement")
    else:
        print("❌ FAILED: Still negative correlation - fundamental limitation")

    print()
    print("Note: SOTA (ρ = 0.75) requires 3D structure. Sequence-only methods are")
    print("      fundamentally limited to ρ ≈ 0.35-0.40 (FLAb2, 2025).")

    # Return best model info
    return {
        'cdr_ml_rho': np.mean(cv_rhos),
        'sap_rho': sap_rho,
        'cdr_sap_rho': cdr_sap_rho,
        'combined_rho': np.mean(cv_rhos_comb),
        'best_alpha': best_alpha,
        'feature_importance': importance[:10],
    }


if __name__ == "__main__":
    data_dir = Path(__file__).parent / "data"
    results = run_hic_ml_benchmark(data_dir)
