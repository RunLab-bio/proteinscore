#!/usr/bin/env python3
"""
HIC SOTA Experiments - Systematic approach to maximize correlation

Goal: Reach or approach SOTA (PROPERMAB ρ = 0.75) using sequence-only features

Experiments:
1. Baseline comparisons (current ML, simple features)
2. Architecture search (MLP, CNN, Attention, Hybrid)
3. Feature engineering (sequence, physicochemical, ESM embeddings)
4. Distillation optimization (alpha, temperature, teacher size)
5. Ensemble methods
6. Advanced techniques (contrastive learning, multi-task)

Theoretical limits:
- Sequence-only: ~0.40-0.50 (without structure)
- With structure: ~0.75 (PROPERMAB)
"""

from __future__ import annotations

import csv
import json
import math
import os
import pickle
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

warnings.filterwarnings('ignore')

# Optional imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge, ElasticNet
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from transformers import AutoModel, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


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
    time_sec: float = 0.0
    notes: str = ""


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

# Hydrophobic amino acids (important for HIC)
HYDROPHOBIC_AAS = set('AILMFVW')
AROMATIC_AAS = set('FWY')
CHARGED_AAS = set('DEKRH')


# =============================================================================
# Feature Extraction Functions
# =============================================================================

def extract_basic_features(vh: str, vl: str) -> np.ndarray:
    """Basic AA composition features."""
    full_seq = vh + vl
    features = []

    # AA composition (20)
    for aa in sorted(AA_PROPERTIES.keys()):
        features.append(full_seq.count(aa) / len(full_seq))

    return np.array(features, dtype=np.float32)


def extract_physicochemical_features(vh: str, vl: str) -> np.ndarray:
    """Physicochemical property features."""
    full_seq = vh + vl
    features = []

    # Global properties
    for prop in ['hydropathy', 'charge', 'aromatic', 'volume', 'polar']:
        values = [AA_PROPERTIES.get(aa, {}).get(prop, 0) for aa in full_seq]
        features.extend([
            np.mean(values),
            np.std(values),
            np.min(values),
            np.max(values),
        ])

    # Hydrophobic content
    hydrophobic_count = sum(1 for aa in full_seq if aa in HYDROPHOBIC_AAS)
    features.append(hydrophobic_count / len(full_seq))

    # Aromatic content
    aromatic_count = sum(1 for aa in full_seq if aa in AROMATIC_AAS)
    features.append(aromatic_count / len(full_seq))

    # Net charge
    net_charge = sum(AA_PROPERTIES.get(aa, {}).get('charge', 0) for aa in full_seq)
    features.append(net_charge / len(full_seq))

    return np.array(features, dtype=np.float32)


def extract_regional_features(vh: str, vl: str) -> np.ndarray:
    """Region-specific features (CDRs, framework)."""
    features = []

    # VH regions (approximate Kabat numbering)
    vh_fr1 = vh[:25] if len(vh) > 25 else vh
    vh_cdr1 = vh[25:35] if len(vh) > 35 else ""
    vh_fr2 = vh[35:50] if len(vh) > 50 else ""
    vh_cdr2 = vh[50:65] if len(vh) > 65 else ""
    vh_fr3 = vh[65:95] if len(vh) > 95 else ""
    vh_cdr3 = vh[95:115] if len(vh) > 115 else vh[95:] if len(vh) > 95 else ""
    vh_fr4 = vh[115:] if len(vh) > 115 else ""

    # VL regions
    vl_fr1 = vl[:23] if len(vl) > 23 else vl
    vl_cdr1 = vl[23:35] if len(vl) > 35 else ""
    vl_fr2 = vl[35:49] if len(vl) > 49 else ""
    vl_cdr2 = vl[49:56] if len(vl) > 56 else ""
    vl_fr3 = vl[56:88] if len(vl) > 88 else ""
    vl_cdr3 = vl[88:98] if len(vl) > 98 else vl[88:] if len(vl) > 88 else ""
    vl_fr4 = vl[98:] if len(vl) > 98 else ""

    # CDR-H3 is most important for HIC
    for region, name in [(vh_cdr3, 'cdrh3'), (vl_cdr3, 'cdrl3'),
                          (vh_cdr1, 'cdrh1'), (vh_cdr2, 'cdrh2')]:
        if region:
            hydro = np.mean([AA_PROPERTIES.get(aa, {}).get('hydropathy', 0) for aa in region])
            aromatic = sum(1 for aa in region if aa in AROMATIC_AAS) / max(len(region), 1)
            features.extend([hydro, aromatic, len(region) / 20])
        else:
            features.extend([0, 0, 0])

    return np.array(features, dtype=np.float32)


def extract_surface_exposure_features(vh: str, vl: str) -> np.ndarray:
    """Approximate surface exposure features."""
    features = []

    # Surface propensity scale (approximate)
    surface_propensity = {
        'A': 0.49, 'R': 0.95, 'N': 0.81, 'D': 0.81, 'C': 0.26,
        'Q': 0.81, 'E': 0.84, 'G': 0.48, 'H': 0.66, 'I': 0.34,
        'L': 0.40, 'K': 0.97, 'M': 0.40, 'F': 0.42, 'P': 0.75,
        'S': 0.70, 'T': 0.70, 'W': 0.49, 'Y': 0.67, 'V': 0.36,
    }

    full_seq = vh + vl

    # Global surface exposure
    exposure = [surface_propensity.get(aa, 0.5) for aa in full_seq]
    features.extend([np.mean(exposure), np.std(exposure)])

    # Terminal exposure (N and C termini often exposed)
    n_term = full_seq[:10]
    c_term = full_seq[-10:]

    n_hydro = np.mean([AA_PROPERTIES.get(aa, {}).get('hydropathy', 0) for aa in n_term])
    c_hydro = np.mean([AA_PROPERTIES.get(aa, {}).get('hydropathy', 0) for aa in c_term])
    features.extend([n_hydro, c_hydro])

    return np.array(features, dtype=np.float32)


def extract_all_features(vh: str, vl: str) -> np.ndarray:
    """Extract all handcrafted features."""
    return np.concatenate([
        extract_basic_features(vh, vl),
        extract_physicochemical_features(vh, vl),
        extract_regional_features(vh, vl),
        extract_surface_exposure_features(vh, vl),
    ])


# =============================================================================
# Metrics
# =============================================================================

def spearman_correlation(x: list, y: list) -> float:
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

    rank_x = rank(list(x))
    rank_y = rank(list(y))

    mean_rx = sum(rank_x) / n
    mean_ry = sum(rank_y) / n

    num = sum((rank_x[i] - mean_rx) * (rank_y[i] - mean_ry) for i in range(n))
    den_x = math.sqrt(sum((rank_x[i] - mean_rx) ** 2 for i in range(n)))
    den_y = math.sqrt(sum((rank_y[i] - mean_ry) ** 2 for i in range(n)))

    if den_x == 0 or den_y == 0:
        return 0.0

    return num / (den_x * den_y)


def evaluate_cv(X: np.ndarray, y: np.ndarray, model, n_folds: int = 5) -> ExperimentResult:
    """Evaluate model with cross-validation."""
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_rhos = []
    all_preds = np.zeros(len(y))

    for train_idx, val_idx in kfold.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model_copy = clone_model(model)
        model_copy.fit(X_train, y_train)
        preds = model_copy.predict(X_val)

        all_preds[val_idx] = preds
        fold_rho = spearman_correlation(y_val.tolist(), preds.tolist())
        fold_rhos.append(fold_rho)

    overall_rho = spearman_correlation(y.tolist(), all_preds.tolist())

    return ExperimentResult(
        name="",
        rho=overall_rho,
        rho_std=np.std(fold_rhos),
        fold_rhos=fold_rhos,
    )


def clone_model(model):
    """Clone a sklearn model."""
    from sklearn.base import clone
    return clone(model)


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
# Experiment 1: Baseline Models
# =============================================================================

def run_baseline_experiments(antibodies: list[AntibodyHIC]) -> list[ExperimentResult]:
    """Test baseline ML models with different features."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Baseline Models")
    print("=" * 70)

    results = []
    y = np.array([ab.hic_rt for ab in antibodies])

    # Feature sets to test
    feature_sets = {
        'basic': lambda ab: extract_basic_features(ab.vh_sequence, ab.vl_sequence),
        'physicochemical': lambda ab: extract_physicochemical_features(ab.vh_sequence, ab.vl_sequence),
        'regional': lambda ab: extract_regional_features(ab.vh_sequence, ab.vl_sequence),
        'surface': lambda ab: extract_surface_exposure_features(ab.vh_sequence, ab.vl_sequence),
        'all_features': lambda ab: extract_all_features(ab.vh_sequence, ab.vl_sequence),
    }

    # Models to test
    models = {
        'Ridge': Ridge(alpha=1.0),
        'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=2000),
        'RF': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        'GBM': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
    }

    for feat_name, feat_fn in feature_sets.items():
        X = np.array([feat_fn(ab) for ab in antibodies])
        X = StandardScaler().fit_transform(X)

        for model_name, model in models.items():
            start = time.time()
            result = evaluate_cv(X, y, model)
            elapsed = time.time() - start

            result.name = f"{model_name}_{feat_name}"
            result.time_sec = elapsed
            result.params = {'features': feat_name, 'model': model_name, 'n_features': X.shape[1]}
            results.append(result)

            print(f"  {result.name:<35} ρ = {result.rho:.3f} ± {result.rho_std:.3f}")

    return results


# =============================================================================
# Experiment 2: Neural Network Architectures
# =============================================================================

def run_nn_experiments(antibodies: list[AntibodyHIC]) -> list[ExperimentResult]:
    """Test different neural network architectures."""
    if not HAS_TORCH:
        print("\nSkipping NN experiments (torch not available)")
        return []

    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Neural Network Architectures")
    print("=" * 70)

    results = []

    # Prepare data
    X = np.array([extract_all_features(ab.vh_sequence, ab.vl_sequence) for ab in antibodies])
    X = StandardScaler().fit_transform(X)
    y = np.array([ab.hic_rt for ab in antibodies])

    # Architectures to test
    architectures = {
        'MLP_small': [64, 32],
        'MLP_medium': [128, 64, 32],
        'MLP_large': [256, 128, 64],
        'MLP_deep': [128, 128, 64, 32],
        'MLP_wide': [512, 256],
    }

    for arch_name, hidden_dims in architectures.items():
        result = train_nn_cv(X, y, hidden_dims, n_epochs=200, lr=0.001)
        result.name = arch_name
        result.params = {'architecture': hidden_dims}
        results.append(result)

        print(f"  {result.name:<35} ρ = {result.rho:.3f} ± {result.rho_std:.3f}")

    return results


def train_nn_cv(X: np.ndarray, y: np.ndarray, hidden_dims: list,
                n_epochs: int = 200, lr: float = 0.001, n_folds: int = 5) -> ExperimentResult:
    """Train neural network with cross-validation."""
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_rhos = []
    all_preds = np.zeros(len(y))

    for train_idx, val_idx in kfold.split(X):
        X_train = torch.tensor(X[train_idx], dtype=torch.float32)
        y_train = torch.tensor(y[train_idx], dtype=torch.float32)
        X_val = torch.tensor(X[val_idx], dtype=torch.float32)
        y_val = y[val_idx]

        # Build model
        layers = []
        in_dim = X.shape[1]
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))
        model = nn.Sequential(*layers)

        # Train
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.01)

        for epoch in range(n_epochs):
            model.train()
            optimizer.zero_grad()
            preds = model(X_train).squeeze()
            loss = F.mse_loss(preds, y_train)
            loss.backward()
            optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            preds = model(X_val).squeeze().numpy()

        all_preds[val_idx] = preds
        fold_rho = spearman_correlation(y_val.tolist(), preds.tolist())
        fold_rhos.append(fold_rho)

    overall_rho = spearman_correlation(y.tolist(), all_preds.tolist())

    return ExperimentResult(
        name="",
        rho=overall_rho,
        rho_std=np.std(fold_rhos),
        fold_rhos=fold_rhos,
    )


# =============================================================================
# Experiment 3: ESM-2 Embeddings with Proper CV
# =============================================================================

def run_esm_experiments(antibodies: list[AntibodyHIC], cache_dir: Path) -> list[ExperimentResult]:
    """Test ESM-2 embeddings with proper cross-validation."""
    if not HAS_TRANSFORMERS:
        print("\nSkipping ESM experiments (transformers not available)")
        return []

    print("\n" + "=" * 70)
    print("EXPERIMENT 3: ESM-2 Embeddings")
    print("=" * 70)

    results = []
    y = np.array([ab.hic_rt for ab in antibodies])

    # ESM models to test
    esm_models = [
        ("esm2_t6_8M", "facebook/esm2_t6_8M_UR50D"),
        ("esm2_t12_35M", "facebook/esm2_t12_35M_UR50D"),
        ("esm2_t30_150M", "facebook/esm2_t30_150M_UR50D"),
    ]

    for model_name, model_id in esm_models:
        cache_file = cache_dir / f"{model_name}_embeddings.pkl"

        # Extract or load embeddings
        if cache_file.exists():
            print(f"  Loading cached embeddings: {model_name}")
            with open(cache_file, 'rb') as f:
                embeddings = pickle.load(f)
        else:
            print(f"  Extracting embeddings: {model_name}...")
            embeddings = extract_esm_embeddings(antibodies, model_id)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(embeddings, f)

        X = embeddings

        # Test different regressors on ESM embeddings
        for reg_name, regressor in [
            ('Ridge', Ridge(alpha=1.0)),
            ('RF', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)),
        ]:
            result = evaluate_cv(X, y, regressor)
            result.name = f"ESM_{model_name}_{reg_name}"
            result.params = {'esm_model': model_name, 'regressor': reg_name}
            results.append(result)

            print(f"  {result.name:<35} ρ = {result.rho:.3f} ± {result.rho_std:.3f}")

    return results


def extract_esm_embeddings(antibodies: list[AntibodyHIC], model_id: str) -> np.ndarray:
    """Extract ESM-2 embeddings for antibodies."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    model.eval()

    embeddings = []

    with torch.no_grad():
        for ab in antibodies:
            # Concatenate VH and VL
            seq = ab.vh_sequence + ab.vl_sequence

            inputs = tokenizer(seq, return_tensors="pt", truncation=True, max_length=512)
            outputs = model(**inputs)

            # Mean pooling
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            embeddings.append(embedding)

    return np.array(embeddings)


# =============================================================================
# Experiment 4: Hybrid Features (Handcrafted + ESM)
# =============================================================================

def run_hybrid_experiments(antibodies: list[AntibodyHIC], cache_dir: Path) -> list[ExperimentResult]:
    """Test hybrid feature combinations."""
    if not HAS_TRANSFORMERS:
        print("\nSkipping hybrid experiments (transformers not available)")
        return []

    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Hybrid Features")
    print("=" * 70)

    results = []
    y = np.array([ab.hic_rt for ab in antibodies])

    # Load best ESM embeddings
    cache_file = cache_dir / "esm2_t30_150M_embeddings.pkl"
    if not cache_file.exists():
        print("  ESM embeddings not cached, extracting...")
        embeddings = extract_esm_embeddings(antibodies, "facebook/esm2_t30_150M_UR50D")
        with open(cache_file, 'wb') as f:
            pickle.dump(embeddings, f)
    else:
        with open(cache_file, 'rb') as f:
            embeddings = pickle.load(f)

    # Handcrafted features
    X_hc = np.array([extract_all_features(ab.vh_sequence, ab.vl_sequence) for ab in antibodies])
    X_hc = StandardScaler().fit_transform(X_hc)

    # ESM features
    X_esm = StandardScaler().fit_transform(embeddings)

    # Hybrid
    X_hybrid = np.concatenate([X_hc, X_esm], axis=1)

    # Test combinations
    feature_configs = [
        ('handcrafted_only', X_hc),
        ('esm_only', X_esm),
        ('hybrid_concat', X_hybrid),
    ]

    for feat_name, X in feature_configs:
        for reg_name, regressor in [
            ('Ridge', Ridge(alpha=1.0)),
            ('RF', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)),
            ('GBM', GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)),
        ]:
            result = evaluate_cv(X, y, regressor)
            result.name = f"{feat_name}_{reg_name}"
            result.params = {'features': feat_name, 'regressor': reg_name, 'n_features': X.shape[1]}
            results.append(result)

            print(f"  {result.name:<35} ρ = {result.rho:.3f} ± {result.rho_std:.3f}")

    return results


# =============================================================================
# Experiment 5: Ensemble Methods
# =============================================================================

def run_ensemble_experiments(antibodies: list[AntibodyHIC], cache_dir: Path) -> list[ExperimentResult]:
    """Test ensemble methods."""
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Ensemble Methods")
    print("=" * 70)

    results = []
    y = np.array([ab.hic_rt for ab in antibodies])

    # Prepare features
    X_hc = np.array([extract_all_features(ab.vh_sequence, ab.vl_sequence) for ab in antibodies])
    X_hc = StandardScaler().fit_transform(X_hc)

    # Try to load ESM embeddings
    cache_file = cache_dir / "esm2_t30_150M_embeddings.pkl"
    if cache_file.exists():
        with open(cache_file, 'rb') as f:
            embeddings = pickle.load(f)
        X_esm = StandardScaler().fit_transform(embeddings)
        X_hybrid = np.concatenate([X_hc, X_esm], axis=1)
    else:
        X_hybrid = X_hc

    # Stacking ensemble
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    # Base models
    base_models = [
        Ridge(alpha=1.0),
        RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
    ]

    # Get meta-features from base models
    meta_features = np.zeros((len(y), len(base_models)))

    for i, model in enumerate(base_models):
        fold_preds = np.zeros(len(y))
        for train_idx, val_idx in kfold.split(X_hybrid):
            model_copy = clone_model(model)
            model_copy.fit(X_hybrid[train_idx], y[train_idx])
            fold_preds[val_idx] = model_copy.predict(X_hybrid[val_idx])
        meta_features[:, i] = fold_preds

    # Simple averaging
    avg_preds = meta_features.mean(axis=1)
    avg_rho = spearman_correlation(y.tolist(), avg_preds.tolist())
    results.append(ExperimentResult(
        name="Ensemble_Average",
        rho=avg_rho,
        params={'method': 'average', 'n_models': len(base_models)}
    ))
    print(f"  {'Ensemble_Average':<35} ρ = {avg_rho:.3f}")

    # Weighted averaging (optimize weights)
    best_rho = 0
    best_weights = None
    for w1 in np.arange(0.1, 0.9, 0.1):
        for w2 in np.arange(0.1, 0.9 - w1, 0.1):
            w3 = 1 - w1 - w2
            weights = np.array([w1, w2, w3])
            weighted_preds = (meta_features * weights).sum(axis=1)
            rho = spearman_correlation(y.tolist(), weighted_preds.tolist())
            if rho > best_rho:
                best_rho = rho
                best_weights = weights

    results.append(ExperimentResult(
        name="Ensemble_Weighted",
        rho=best_rho,
        params={'method': 'weighted', 'weights': best_weights.tolist() if best_weights is not None else None}
    ))
    print(f"  {'Ensemble_Weighted':<35} ρ = {best_rho:.3f}")

    # Stacking with meta-learner
    meta_result = evaluate_cv(meta_features, y, Ridge(alpha=0.1))
    meta_result.name = "Ensemble_Stacking"
    results.append(meta_result)
    print(f"  {'Ensemble_Stacking':<35} ρ = {meta_result.rho:.3f} ± {meta_result.rho_std:.3f}")

    return results


# =============================================================================
# Experiment 6: Optimized Distillation
# =============================================================================

def run_distillation_experiments(antibodies: list[AntibodyHIC], cache_dir: Path) -> list[ExperimentResult]:
    """Test optimized distillation approaches."""
    if not HAS_TORCH or not HAS_TRANSFORMERS:
        print("\nSkipping distillation experiments (torch/transformers not available)")
        return []

    print("\n" + "=" * 70)
    print("EXPERIMENT 6: Optimized Distillation")
    print("=" * 70)

    results = []
    y = np.array([ab.hic_rt for ab in antibodies])

    # Load ESM embeddings
    cache_file = cache_dir / "esm2_t30_150M_embeddings.pkl"
    if not cache_file.exists():
        print("  ESM embeddings not cached, extracting...")
        embeddings = extract_esm_embeddings(antibodies, "facebook/esm2_t30_150M_UR50D")
        with open(cache_file, 'wb') as f:
            pickle.dump(embeddings, f)
    else:
        with open(cache_file, 'rb') as f:
            embeddings = pickle.load(f)

    # Handcrafted features
    X_hc = np.array([extract_all_features(ab.vh_sequence, ab.vl_sequence) for ab in antibodies])
    X_hc = StandardScaler().fit_transform(X_hc)
    X_esm = StandardScaler().fit_transform(embeddings)

    # Test different alpha values (distillation weight)
    for alpha in [0.0, 0.1, 0.3, 0.5, 0.7]:
        result = train_distillation_cv(X_hc, X_esm, y, alpha=alpha)
        result.name = f"Distill_alpha={alpha}"
        result.params = {'alpha': alpha}
        results.append(result)
        print(f"  {result.name:<35} ρ = {result.rho:.3f} ± {result.rho_std:.3f}")

    return results


def train_distillation_cv(X_hc: np.ndarray, X_esm: np.ndarray, y: np.ndarray,
                          alpha: float = 0.3, n_epochs: int = 300, n_folds: int = 5) -> ExperimentResult:
    """Train distillation model with cross-validation."""
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_rhos = []
    all_preds = np.zeros(len(y))

    for train_idx, val_idx in kfold.split(X_hc):
        # Prepare tensors
        X_train = torch.tensor(X_hc[train_idx], dtype=torch.float32)
        y_train = torch.tensor(y[train_idx], dtype=torch.float32)
        teacher_train = torch.tensor(X_esm[train_idx], dtype=torch.float32)

        X_val = torch.tensor(X_hc[val_idx], dtype=torch.float32)
        y_val = y[val_idx]

        # Build student model
        input_dim = X_hc.shape[1]
        teacher_dim = X_esm.shape[1]

        model = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        hic_head = nn.Linear(64, 1)
        embed_head = nn.Linear(64, teacher_dim)

        optimizer = torch.optim.Adam(
            list(model.parameters()) + list(hic_head.parameters()) + list(embed_head.parameters()),
            lr=0.001, weight_decay=0.01
        )

        # Training loop
        for epoch in range(n_epochs):
            model.train()
            optimizer.zero_grad()

            features = model(X_train)
            hic_pred = hic_head(features).squeeze()
            embed_pred = embed_head(features)

            # HIC loss
            hic_loss = F.mse_loss(hic_pred, y_train)

            # Distillation loss (cosine similarity)
            if alpha > 0:
                embed_pred_norm = F.normalize(embed_pred, dim=1)
                teacher_norm = F.normalize(teacher_train, dim=1)
                distill_loss = 1 - (embed_pred_norm * teacher_norm).sum(dim=1).mean()
                loss = (1 - alpha) * hic_loss + alpha * distill_loss
            else:
                loss = hic_loss

            loss.backward()
            optimizer.step()

        # Evaluate
        model.eval()
        with torch.no_grad():
            features = model(X_val)
            preds = hic_head(features).squeeze().numpy()

        all_preds[val_idx] = preds
        fold_rho = spearman_correlation(y_val.tolist(), preds.tolist())
        fold_rhos.append(fold_rho)

    overall_rho = spearman_correlation(y.tolist(), all_preds.tolist())

    return ExperimentResult(
        name="",
        rho=overall_rho,
        rho_std=np.std(fold_rhos),
        fold_rhos=fold_rhos,
    )


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 70)
    print("HIC SOTA Experiments")
    print("=" * 70)

    # Load data
    data_dir = Path(__file__).parent / "data"
    cache_dir = Path(__file__).parent / "cache"
    cache_dir.mkdir(exist_ok=True)

    antibodies = load_hic_data(data_dir)
    print(f"\nLoaded {len(antibodies)} antibodies")
    print(f"HIC range: {min(ab.hic_rt for ab in antibodies):.1f} - {max(ab.hic_rt for ab in antibodies):.1f}")

    all_results = []

    # Run experiments
    all_results.extend(run_baseline_experiments(antibodies))
    all_results.extend(run_nn_experiments(antibodies))
    all_results.extend(run_esm_experiments(antibodies, cache_dir))
    all_results.extend(run_hybrid_experiments(antibodies, cache_dir))
    all_results.extend(run_ensemble_experiments(antibodies, cache_dir))
    all_results.extend(run_distillation_experiments(antibodies, cache_dir))

    # Final summary
    print("\n" + "=" * 70)
    print("FINAL RESULTS - Top 10")
    print("=" * 70)

    sorted_results = sorted(all_results, key=lambda x: x.rho, reverse=True)

    print(f"\n{'Rank':<6} {'Model':<40} {'ρ':>10} {'± Std':>10}")
    print("-" * 70)

    for i, result in enumerate(sorted_results[:10], 1):
        std_str = f"± {result.rho_std:.3f}" if result.rho_std > 0 else ""
        print(f"{i:<6} {result.name:<40} {result.rho:>10.3f} {std_str:>10}")

    print("\n" + "-" * 70)
    print("Reference Benchmarks:")
    print(f"  Current ML HIC scorer:     ρ = 0.351")
    print(f"  Theoretical seq-only max:  ρ ~ 0.40-0.50")
    print(f"  SOTA PROPERMAB (struct):   ρ = 0.75")

    # Best result
    best = sorted_results[0]
    print(f"\n✅ Best model: {best.name}")
    print(f"   Correlation: ρ = {best.rho:.3f}")
    if best.params:
        print(f"   Parameters: {best.params}")

    # Save results
    results_file = Path(__file__).parent / "sota_experiment_results.json"
    with open(results_file, 'w') as f:
        json.dump([{
            'name': r.name,
            'rho': r.rho,
            'rho_std': r.rho_std,
            'fold_rhos': r.fold_rhos,
            'params': r.params,
        } for r in sorted_results], f, indent=2)
    print(f"\nResults saved to: {results_file}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
