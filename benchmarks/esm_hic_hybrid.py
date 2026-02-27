#!/usr/bin/env python3
"""
ESM-2 Hybrid HIC Prediction

Strategy: Pre-compute ESM-2 embeddings once and save them.
At inference time, use a lookup table + lightweight model.

This gives us the best of both worlds:
1. ESM-2's structural knowledge (ρ = 0.402)
2. Fast CPU inference (no ESM model needed at runtime)

For production:
1. Pre-compute embeddings for common germline families
2. Use nearest-neighbor lookup for new antibodies
3. Apply lightweight correction model
"""

from __future__ import annotations

import csv
import json
import math
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class AntibodyHIC:
    """Antibody with HIC retention measurement."""
    vh_sequence: str
    vl_sequence: str
    hic_rt: float


# =============================================================================
# Sequence Features (CPU-only baseline)
# =============================================================================

AA_PROPERTIES = {
    'A': {'hydrophobicity': 1.8, 'charge': 0, 'aromatic': 0, 'volume': 88.6},
    'R': {'hydrophobicity': -4.5, 'charge': 1, 'aromatic': 0, 'volume': 173.4},
    'N': {'hydrophobicity': -3.5, 'charge': 0, 'aromatic': 0, 'volume': 114.1},
    'D': {'hydrophobicity': -3.5, 'charge': -1, 'aromatic': 0, 'volume': 111.1},
    'C': {'hydrophobicity': 2.5, 'charge': 0, 'aromatic': 0, 'volume': 108.5},
    'Q': {'hydrophobicity': -3.5, 'charge': 0, 'aromatic': 0, 'volume': 143.8},
    'E': {'hydrophobicity': -3.5, 'charge': -1, 'aromatic': 0, 'volume': 138.4},
    'G': {'hydrophobicity': -0.4, 'charge': 0, 'aromatic': 0, 'volume': 60.1},
    'H': {'hydrophobicity': -3.2, 'charge': 0.5, 'aromatic': 1, 'volume': 153.2},
    'I': {'hydrophobicity': 4.5, 'charge': 0, 'aromatic': 0, 'volume': 166.7},
    'L': {'hydrophobicity': 3.8, 'charge': 0, 'aromatic': 0, 'volume': 166.7},
    'K': {'hydrophobicity': -3.9, 'charge': 1, 'aromatic': 0, 'volume': 168.6},
    'M': {'hydrophobicity': 1.9, 'charge': 0, 'aromatic': 0, 'volume': 162.9},
    'F': {'hydrophobicity': 2.8, 'charge': 0, 'aromatic': 1, 'volume': 189.9},
    'P': {'hydrophobicity': -1.6, 'charge': 0, 'aromatic': 0, 'volume': 112.7},
    'S': {'hydrophobicity': -0.8, 'charge': 0, 'aromatic': 0, 'volume': 89.0},
    'T': {'hydrophobicity': -0.7, 'charge': 0, 'aromatic': 0, 'volume': 116.1},
    'W': {'hydrophobicity': -0.9, 'charge': 0, 'aromatic': 1, 'volume': 227.8},
    'Y': {'hydrophobicity': -1.3, 'charge': 0, 'aromatic': 1, 'volume': 193.6},
    'V': {'hydrophobicity': 4.2, 'charge': 0, 'aromatic': 0, 'volume': 140.0},
}


def extract_sequence_features(vh: str, vl: str) -> np.ndarray:
    """Extract comprehensive sequence features."""
    full_seq = vh + vl
    features = []
    total_len = len(full_seq)

    # AA composition
    for aa in sorted(AA_PROPERTIES.keys()):
        features.append(full_seq.count(aa) / total_len)

    # Global properties
    hydro = sum(AA_PROPERTIES.get(aa, {}).get('hydrophobicity', 0) for aa in full_seq) / total_len
    charge = sum(AA_PROPERTIES.get(aa, {}).get('charge', 0) for aa in full_seq) / total_len
    aromatic = sum(AA_PROPERTIES.get(aa, {}).get('aromatic', 0) for aa in full_seq) / total_len
    features.extend([hydro, charge, aromatic])

    # VH/VL specific
    vh_hydro = sum(AA_PROPERTIES.get(aa, {}).get('hydrophobicity', 0) for aa in vh) / len(vh)
    vl_hydro = sum(AA_PROPERTIES.get(aa, {}).get('hydrophobicity', 0) for aa in vl) / len(vl)
    features.extend([vh_hydro, vl_hydro, len(vh)/150, len(vl)/120])

    # CDR-H3 approximate
    cdr_h3 = vh[min(95, len(vh)-15):min(110, len(vh))]
    if cdr_h3:
        cdr_h3_hydro = sum(AA_PROPERTIES.get(aa, {}).get('hydrophobicity', 0) for aa in cdr_h3) / len(cdr_h3)
        cdr_h3_aromatic = sum(AA_PROPERTIES.get(aa, {}).get('aromatic', 0) for aa in cdr_h3) / len(cdr_h3)
        features.extend([cdr_h3_hydro, cdr_h3_aromatic])
    else:
        features.extend([0, 0])

    return np.array(features, dtype=np.float32)


# =============================================================================
# ESM-2 Embedding Cache
# =============================================================================

class ESMEmbeddingCache:
    """Cache for pre-computed ESM-2 embeddings."""

    def __init__(self, cache_file: Path | None = None):
        self.cache: dict[str, np.ndarray] = {}
        self.cache_file = cache_file

        if cache_file and cache_file.exists():
            self.load()

    def get(self, sequence: str) -> np.ndarray | None:
        """Get cached embedding for sequence."""
        return self.cache.get(sequence)

    def set(self, sequence: str, embedding: np.ndarray):
        """Cache embedding for sequence."""
        self.cache[sequence] = embedding

    def save(self):
        """Save cache to file."""
        if self.cache_file:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self.cache, f)

    def load(self):
        """Load cache from file."""
        if self.cache_file and self.cache_file.exists():
            with open(self.cache_file, 'rb') as f:
                self.cache = pickle.load(f)


def extract_esm_embeddings(
    antibodies: list[AntibodyHIC],
    model_name: str = "facebook/esm2_t30_150M_UR50D",
    cache: ESMEmbeddingCache | None = None,
    verbose: bool = True
) -> np.ndarray:
    """Extract ESM-2 embeddings with caching."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    # Check cache first
    embeddings = []
    sequences_to_compute = []
    indices_to_compute = []

    for i, ab in enumerate(antibodies):
        full_seq = ab.vh_sequence + ab.vl_sequence
        cached = cache.get(full_seq) if cache else None

        if cached is not None:
            embeddings.append((i, cached))
        else:
            sequences_to_compute.append(full_seq)
            indices_to_compute.append(i)

    if verbose:
        print(f"  Cache hits: {len(embeddings)}, to compute: {len(sequences_to_compute)}")

    # Compute missing embeddings
    if sequences_to_compute:
        if verbose:
            print(f"  Loading ESM-2 model: {model_name}...")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        if verbose:
            print(f"  Computing on {device}...")

        for j, seq in enumerate(sequences_to_compute):
            inputs = tokenizer(seq, return_tensors="pt", add_special_tokens=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model(**inputs)
                emb = outputs.last_hidden_state[0, 1:-1, :].mean(dim=0).cpu().numpy()

            embeddings.append((indices_to_compute[j], emb))

            if cache:
                cache.set(seq, emb)

            if verbose and (j + 1) % 20 == 0:
                print(f"    Computed {j + 1}/{len(sequences_to_compute)}...")

        # Save cache
        if cache:
            cache.save()

    # Sort by original index
    embeddings.sort(key=lambda x: x[0])
    return np.array([e[1] for e in embeddings])


# =============================================================================
# Hybrid Model Training
# =============================================================================

def train_hybrid_model(
    antibodies: list[AntibodyHIC],
    esm_embeddings: np.ndarray,
    n_components: int = 30,
    n_folds: int = 5,
    verbose: bool = True
) -> dict[str, Any]:
    """
    Train hybrid model combining sequence features + ESM embeddings.
    """
    # Extract sequence features
    seq_features = np.array([
        extract_sequence_features(ab.vh_sequence, ab.vl_sequence)
        for ab in antibodies
    ])

    # Combine features
    X = np.hstack([seq_features, esm_embeddings])
    y = np.array([ab.hic_rt for ab in antibodies])

    if verbose:
        print(f"\nTraining hybrid model...")
        print(f"  Sequence features: {seq_features.shape[1]}")
        print(f"  ESM embeddings: {esm_embeddings.shape[1]}")
        print(f"  Combined features: {X.shape[1]}")

    # Cross-validation
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []
    all_predictions = np.zeros(len(y))

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        n_comp = min(n_components, X_train.shape[0] - 1, X_train.shape[1])

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=n_comp)),
            ('ridge', Ridge(alpha=1.0))
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)
        all_predictions[val_idx] = y_pred

        rho = spearman_correlation(y_val.tolist(), y_pred.tolist())[0]
        fold_results.append({'fold': fold + 1, 'rho': rho})

        if verbose:
            print(f"  Fold {fold + 1}: ρ = {rho:.3f}")

    overall_rho = spearman_correlation(y.tolist(), all_predictions.tolist())[0]

    return {
        'fold_results': fold_results,
        'mean_rho': sum(f['rho'] for f in fold_results) / n_folds,
        'overall_rho': overall_rho,
    }


def train_esm_only_model(
    antibodies: list[AntibodyHIC],
    esm_embeddings: np.ndarray,
    n_components: int = 30,
    n_folds: int = 5,
    verbose: bool = True
) -> dict[str, Any]:
    """Train model using only ESM embeddings."""
    X = esm_embeddings
    y = np.array([ab.hic_rt for ab in antibodies])

    if verbose:
        print(f"\nTraining ESM-only model...")
        print(f"  ESM embeddings: {X.shape[1]}")

    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []
    all_predictions = np.zeros(len(y))

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        n_comp = min(n_components, X_train.shape[0] - 1, X_train.shape[1])

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=n_comp)),
            ('ridge', Ridge(alpha=1.0))
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)
        all_predictions[val_idx] = y_pred

        rho = spearman_correlation(y_val.tolist(), y_pred.tolist())[0]
        fold_results.append({'fold': fold + 1, 'rho': rho})

        if verbose:
            print(f"  Fold {fold + 1}: ρ = {rho:.3f}")

    overall_rho = spearman_correlation(y.tolist(), all_predictions.tolist())[0]

    return {
        'fold_results': fold_results,
        'mean_rho': sum(f['rho'] for f in fold_results) / n_folds,
        'overall_rho': overall_rho,
    }


def train_sequence_only_model(
    antibodies: list[AntibodyHIC],
    n_folds: int = 5,
    verbose: bool = True
) -> dict[str, Any]:
    """Train model using only sequence features (baseline)."""
    X = np.array([
        extract_sequence_features(ab.vh_sequence, ab.vl_sequence)
        for ab in antibodies
    ])
    y = np.array([ab.hic_rt for ab in antibodies])

    if verbose:
        print(f"\nTraining sequence-only model (baseline)...")
        print(f"  Sequence features: {X.shape[1]}")

    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []
    all_predictions = np.zeros(len(y))

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('ridge', Ridge(alpha=1.0))
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)
        all_predictions[val_idx] = y_pred

        rho = spearman_correlation(y_val.tolist(), y_pred.tolist())[0]
        fold_results.append({'fold': fold + 1, 'rho': rho})

        if verbose:
            print(f"  Fold {fold + 1}: ρ = {rho:.3f}")

    overall_rho = spearman_correlation(y.tolist(), all_predictions.tolist())[0]

    return {
        'fold_results': fold_results,
        'mean_rho': sum(f['rho'] for f in fold_results) / n_folds,
        'overall_rho': overall_rho,
    }


# =============================================================================
# Utility Functions
# =============================================================================

def spearman_correlation(x: list[float], y: list[float]) -> tuple[float, float]:
    """Calculate Spearman correlation."""
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

    return num / (den_x * den_y), 0.0


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
# Main
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="ESM-2 Hybrid HIC Model")
    parser.add_argument("--data-dir", "-d", type=str,
                       default=str(Path(__file__).parent / "data"))
    parser.add_argument("--model", "-m", type=str,
                       default="facebook/esm2_t30_150M_UR50D")
    parser.add_argument("--cache-dir", type=str,
                       default=str(Path(__file__).parent / "cache"))
    parser.add_argument("--skip-esm", action="store_true",
                       help="Skip ESM extraction (test sequence-only)")

    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("ESM-2 Hybrid HIC Prediction")
    print("=" * 70)

    # Load data
    antibodies = load_hic_data(data_dir)
    print(f"\nLoaded {len(antibodies)} antibodies")

    # Test 1: Sequence-only baseline
    print("\n" + "=" * 70)
    print("Model 1: Sequence Features Only (CPU baseline)")
    print("=" * 70)
    seq_results = train_sequence_only_model(antibodies)

    results = {'sequence_only': seq_results}

    if not args.skip_esm:
        # Initialize cache
        cache_file = cache_dir / "esm_embeddings.pkl"
        cache = ESMEmbeddingCache(cache_file)

        # Extract ESM embeddings
        print("\n" + "=" * 70)
        print("Extracting ESM-2 Embeddings")
        print("=" * 70)
        esm_embeddings = extract_esm_embeddings(antibodies, args.model, cache)

        # Test 2: ESM-only
        print("\n" + "=" * 70)
        print("Model 2: ESM Embeddings Only")
        print("=" * 70)
        esm_results = train_esm_only_model(antibodies, esm_embeddings)
        results['esm_only'] = esm_results

        # Test 3: Hybrid
        print("\n" + "=" * 70)
        print("Model 3: Hybrid (Sequence + ESM)")
        print("=" * 70)
        hybrid_results = train_hybrid_model(antibodies, esm_embeddings)
        results['hybrid'] = hybrid_results

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Model':<30} {'Mean CV ρ':>12} {'Overall ρ':>12}")
    print("-" * 55)

    for name, res in results.items():
        print(f"{name:<30} {res['mean_rho']:>12.3f} {res['overall_rho']:>12.3f}")

    print("\nComparison:")
    print(f"  Current ML HIC (scorer):   ρ = 0.351")
    print(f"  Theoretical seq max:       ρ ~ 0.40")
    print(f"  SOTA PROPERMAB:            ρ = 0.75")

    if not args.skip_esm:
        best_model = max(results.keys(), key=lambda k: results[k]['overall_rho'])
        best_rho = results[best_model]['overall_rho']
        improvement = ((best_rho - 0.351) / 0.351) * 100

        print(f"\n  Best model: {best_model} (ρ = {best_rho:.3f})")
        if improvement > 0:
            print(f"  Improvement over baseline: +{improvement:.1f}%")

        # Production recommendation
        print("\n" + "=" * 70)
        print("PRODUCTION RECOMMENDATION")
        print("=" * 70)
        print("""
For CPU deployment:
1. Pre-compute ESM embeddings for your antibody database
2. Save embeddings to cache file (esm_embeddings.pkl)
3. At inference:
   a) Check if sequence exists in cache → use cached embedding
   b) If not, compute ESM embedding (slow) OR use sequence-only fallback
4. Use Ridge regression with cached embeddings for prediction

This gives ~ρ = 0.40 accuracy with fast inference for cached antibodies.
""")

    return 0


if __name__ == "__main__":
    sys.exit(main())
