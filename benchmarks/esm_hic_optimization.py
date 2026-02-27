#!/usr/bin/env python3
"""
ESM-2 Based HIC Retention Prediction (Hugging Face)

Uses ESM-2 protein language model embeddings from Hugging Face to predict
HIC retention time. The hypothesis is that PLM embeddings capture implicit
structural information that correlates with surface hydrophobicity.

Expected improvement: ρ = 0.45-0.55 (vs current 0.35 sequence-only)

Reference:
- Lin et al. (2023) "Evolutionary-scale prediction of atomic-level protein structure"
- Rives et al. (2021) "Biological structure and function emerge from scaling"
"""

from __future__ import annotations

import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModel, AutoTokenizer

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
    hic_rt: float  # HIC Retention Time (min)


# =============================================================================
# ESM-2 Embedding Extraction (Hugging Face)
# =============================================================================

class ESM2Embedder:
    """Extract embeddings from ESM-2 model via Hugging Face."""

    # Available models on Hugging Face
    MODELS = {
        "esm2_t6_8M": "facebook/esm2_t6_8M_UR50D",      # 8M params, fastest
        "esm2_t12_35M": "facebook/esm2_t12_35M_UR50D",   # 35M params
        "esm2_t30_150M": "facebook/esm2_t30_150M_UR50D", # 150M params
        "esm2_t33_650M": "facebook/esm2_t33_650M_UR50D", # 650M params, best
    }

    def __init__(self, model_name: str = "esm2_t6_8M"):
        """
        Initialize ESM-2 model from Hugging Face.

        Args:
            model_name: Short name or full HF path
        """
        # Resolve model path
        if model_name in self.MODELS:
            model_path = self.MODELS[model_name]
        else:
            model_path = model_name

        print(f"Loading ESM-2 model from Hugging Face: {model_path}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModel.from_pretrained(model_path)
        self.model.eval()

        # Move to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        print(f"Model loaded on {self.device}")

        # Get embedding dimension
        self.embed_dim = self.model.config.hidden_size

    def get_embedding(self, sequence: str) -> torch.Tensor:
        """
        Get per-residue embeddings for a sequence.

        Returns:
            Tensor of shape (seq_len, embed_dim)
        """
        # Tokenize
        inputs = self.tokenizer(sequence, return_tensors="pt", add_special_tokens=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Extract embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Get last hidden state (excluding BOS/EOS tokens)
        # Shape: (batch=1, seq_len+2, hidden_size) -> (seq_len, hidden_size)
        embeddings = outputs.last_hidden_state[0, 1:-1, :]

        return embeddings.cpu()

    def get_pooled_embedding(
        self,
        sequence: str,
        pooling: str = "mean"
    ) -> torch.Tensor:
        """
        Get pooled embedding for entire sequence.

        Args:
            sequence: Amino acid sequence
            pooling: "mean", "max", or "cls"

        Returns:
            Tensor of shape (embed_dim,)
        """
        embeddings = self.get_embedding(sequence)

        if pooling == "mean":
            return embeddings.mean(dim=0)
        elif pooling == "max":
            return embeddings.max(dim=0)[0]
        elif pooling == "cls":
            return embeddings[0]
        else:
            raise ValueError(f"Unknown pooling: {pooling}")


# =============================================================================
# Feature Extraction
# =============================================================================

def extract_esm_features(
    embedder: ESM2Embedder,
    vh: str,
    vl: str
) -> dict[str, Any]:
    """
    Extract features from ESM-2 embeddings.

    Features:
    1. Global mean/max pooling of VH and VL
    2. CDR-specific pooling (approximate positions)
    3. Variance statistics
    """
    features = {}

    # Get full embeddings
    vh_emb = embedder.get_embedding(vh)  # (vh_len, embed_dim)
    vl_emb = embedder.get_embedding(vl)  # (vl_len, embed_dim)

    # Full sequence embedding
    full_seq = vh + vl
    full_emb = embedder.get_embedding(full_seq)

    # Global pooling features
    vh_mean = vh_emb.mean(dim=0)
    vl_mean = vl_emb.mean(dim=0)
    full_mean = full_emb.mean(dim=0)

    vh_max = vh_emb.max(dim=0)[0]
    vl_max = vl_emb.max(dim=0)[0]

    # Store pooled embeddings
    features['vh_mean'] = vh_mean.numpy()
    features['vl_mean'] = vl_mean.numpy()
    features['full_mean'] = full_mean.numpy()
    features['vh_max'] = vh_max.numpy()
    features['vl_max'] = vl_max.numpy()

    # Scalar statistics
    features['vh_mean_norm'] = float(torch.norm(vh_mean))
    features['vl_mean_norm'] = float(torch.norm(vl_mean))
    features['vh_max_norm'] = float(torch.norm(vh_max))
    features['vl_max_norm'] = float(torch.norm(vl_max))

    # Embedding variance
    features['vh_var'] = float(vh_emb.var())
    features['vl_var'] = float(vl_emb.var())

    # CDR-H3 features (typically positions 95-102, most exposed)
    cdr_h3_start = min(95, len(vh) - 10)
    cdr_h3_end = min(cdr_h3_start + 15, len(vh))
    if cdr_h3_end > cdr_h3_start:
        cdr_h3_emb = vh_emb[cdr_h3_start:cdr_h3_end]
        features['cdr_h3_mean'] = cdr_h3_emb.mean(dim=0).numpy()
        features['cdr_h3_norm'] = float(torch.norm(cdr_h3_emb.mean(dim=0)))

    # CDR-L3 features (typically positions 89-97)
    cdr_l3_start = min(89, len(vl) - 10)
    cdr_l3_end = min(cdr_l3_start + 10, len(vl))
    if cdr_l3_end > cdr_l3_start:
        cdr_l3_emb = vl_emb[cdr_l3_start:cdr_l3_end]
        features['cdr_l3_mean'] = cdr_l3_emb.mean(dim=0).numpy()
        features['cdr_l3_norm'] = float(torch.norm(cdr_l3_emb.mean(dim=0)))

    return features


def features_to_vector(features: dict[str, Any], embed_dim: int) -> list[float]:
    """Convert feature dict to flat vector for ML."""
    vector = []

    # Add embedding vectors
    for key in ['vh_mean', 'vl_mean', 'full_mean', 'vh_max', 'vl_max']:
        if key in features:
            vector.extend(features[key].tolist())
        else:
            vector.extend([0.0] * embed_dim)

    # Add CDR embeddings
    for key in ['cdr_h3_mean', 'cdr_l3_mean']:
        if key in features:
            vector.extend(features[key].tolist())
        else:
            vector.extend([0.0] * embed_dim)

    # Add scalar features
    scalar_keys = [
        'vh_mean_norm', 'vl_mean_norm', 'vh_max_norm', 'vl_max_norm',
        'vh_var', 'vl_var', 'cdr_h3_norm', 'cdr_l3_norm'
    ]
    for key in scalar_keys:
        vector.append(features.get(key, 0.0))

    return vector


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


# =============================================================================
# Data Loading
# =============================================================================

def load_hic_data(data_dir: Path) -> list[AntibodyHIC]:
    """Load HIC retention time data from FLAb."""
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
# Model Training
# =============================================================================

def train_simple_esm_model(
    embedder: ESM2Embedder,
    antibodies: list[AntibodyHIC],
    n_folds: int = 5,
    verbose: bool = True
) -> dict[str, Any]:
    """
    Train HIC model using mean embeddings (simple, robust).
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold
    import numpy as np

    if verbose:
        print(f"\nExtracting ESM-2 mean embeddings for {len(antibodies)} antibodies...")

    # Extract mean embeddings
    X = []
    y = []

    for i, ab in enumerate(antibodies):
        try:
            full_seq = ab.vh_sequence + ab.vl_sequence
            emb = embedder.get_pooled_embedding(full_seq, pooling="mean")
            X.append(emb.numpy())
            y.append(ab.hic_rt)

            if verbose and (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}/{len(antibodies)}...")

        except Exception as e:
            if verbose:
                print(f"  Warning: Failed to process antibody {i + 1}: {e}")
            continue

    X = np.array(X)
    y = np.array(y)

    if verbose:
        print(f"  Feature matrix shape: {X.shape}")

    # Cross-validation
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_results = []
    all_predictions = np.zeros(len(y))

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        model = Ridge(alpha=10.0)
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_val_scaled)
        all_predictions[val_idx] = y_pred

        rho, p_value = spearman_correlation(y_val.tolist(), y_pred.tolist())
        fold_results.append({'fold': fold + 1, 'rho': rho, 'p_value': p_value, 'n': len(y_val)})

        if verbose:
            print(f"  Fold {fold + 1}: ρ = {rho:.3f} (p = {p_value:.4f})")

    overall_rho, overall_p = spearman_correlation(y.tolist(), all_predictions.tolist())

    results = {
        'fold_results': fold_results,
        'mean_rho': sum(f['rho'] for f in fold_results) / n_folds,
        'overall_rho': overall_rho,
        'overall_p': overall_p,
        'embed_dim': embedder.embed_dim,
    }

    if verbose:
        print(f"\n  Mean CV ρ = {results['mean_rho']:.3f}")
        print(f"  Overall ρ = {overall_rho:.3f} (p = {overall_p:.4f})")

    return results


def train_esm_hic_model_pca(
    embedder: ESM2Embedder,
    antibodies: list[AntibodyHIC],
    n_components: int = 50,
    n_folds: int = 5,
    verbose: bool = True
) -> dict[str, Any]:
    """
    Train HIC model using full features with PCA reduction.
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.model_selection import KFold
    from sklearn.pipeline import Pipeline
    import numpy as np

    if verbose:
        print(f"\nExtracting ESM-2 full features for {len(antibodies)} antibodies...")

    # Extract features
    X = []
    y = []

    for i, ab in enumerate(antibodies):
        try:
            features = extract_esm_features(embedder, ab.vh_sequence, ab.vl_sequence)
            vector = features_to_vector(features, embedder.embed_dim)
            X.append(vector)
            y.append(ab.hic_rt)

            if verbose and (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}/{len(antibodies)}...")

        except Exception as e:
            if verbose:
                print(f"  Warning: Failed to process antibody {i + 1}: {e}")
            continue

    X = np.array(X)
    y = np.array(y)

    if verbose:
        print(f"  Feature matrix shape: {X.shape}")
        print(f"  Using PCA with {n_components} components")

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

        rho, p_value = spearman_correlation(y_val.tolist(), y_pred.tolist())
        fold_results.append({'fold': fold + 1, 'rho': rho, 'p_value': p_value, 'n': len(y_val)})

        if verbose:
            print(f"  Fold {fold + 1}: ρ = {rho:.3f} (p = {p_value:.4f})")

    overall_rho, overall_p = spearman_correlation(y.tolist(), all_predictions.tolist())

    results = {
        'fold_results': fold_results,
        'mean_rho': sum(f['rho'] for f in fold_results) / n_folds,
        'overall_rho': overall_rho,
        'overall_p': overall_p,
        'feature_dim': X.shape[1],
        'n_components': n_components,
        'embed_dim': embedder.embed_dim,
    }

    if verbose:
        print(f"\n  Mean CV ρ = {results['mean_rho']:.3f}")
        print(f"  Overall ρ = {overall_rho:.3f} (p = {overall_p:.4f})")

    return results


def train_vh_vl_separate_model(
    embedder: ESM2Embedder,
    antibodies: list[AntibodyHIC],
    n_folds: int = 5,
    verbose: bool = True
) -> dict[str, Any]:
    """
    Train HIC model using separate VH and VL embeddings (concatenated).
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold
    import numpy as np

    if verbose:
        print(f"\nExtracting ESM-2 VH+VL embeddings for {len(antibodies)} antibodies...")

    # Extract VH and VL embeddings separately
    X = []
    y = []

    for i, ab in enumerate(antibodies):
        try:
            vh_emb = embedder.get_pooled_embedding(ab.vh_sequence, pooling="mean")
            vl_emb = embedder.get_pooled_embedding(ab.vl_sequence, pooling="mean")

            # Concatenate VH and VL
            combined = torch.cat([vh_emb, vl_emb], dim=0)
            X.append(combined.numpy())
            y.append(ab.hic_rt)

            if verbose and (i + 1) % 20 == 0:
                print(f"  Processed {i + 1}/{len(antibodies)}...")

        except Exception as e:
            if verbose:
                print(f"  Warning: Failed to process antibody {i + 1}: {e}")
            continue

    X = np.array(X)
    y = np.array(y)

    if verbose:
        print(f"  Feature matrix shape: {X.shape}")

    # Cross-validation
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_results = []
    all_predictions = np.zeros(len(y))

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        model = Ridge(alpha=5.0)
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_val_scaled)
        all_predictions[val_idx] = y_pred

        rho, p_value = spearman_correlation(y_val.tolist(), y_pred.tolist())
        fold_results.append({'fold': fold + 1, 'rho': rho, 'p_value': p_value, 'n': len(y_val)})

        if verbose:
            print(f"  Fold {fold + 1}: ρ = {rho:.3f} (p = {p_value:.4f})")

    overall_rho, overall_p = spearman_correlation(y.tolist(), all_predictions.tolist())

    results = {
        'fold_results': fold_results,
        'mean_rho': sum(f['rho'] for f in fold_results) / n_folds,
        'overall_rho': overall_rho,
        'overall_p': overall_p,
        'embed_dim': embedder.embed_dim,
    }

    if verbose:
        print(f"\n  Mean CV ρ = {results['mean_rho']:.3f}")
        print(f"  Overall ρ = {overall_rho:.3f} (p = {overall_p:.4f})")

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="ESM-2 HIC Prediction (Hugging Face)")
    parser.add_argument("--data-dir", "-d", type=str,
                       default=str(Path(__file__).parent / "data"),
                       help="Path to benchmark data directory")
    parser.add_argument("--model", "-m", type=str,
                       default="esm2_t6_8M",
                       choices=["esm2_t6_8M", "esm2_t12_35M", "esm2_t30_150M", "esm2_t33_650M"],
                       help="ESM-2 model size")
    parser.add_argument("--approach", "-a", type=str,
                       default="all",
                       choices=["simple", "vh_vl", "pca", "all"],
                       help="Training approach")

    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    print("=" * 70)
    print("ESM-2 Based HIC Retention Prediction (Hugging Face)")
    print("=" * 70)

    # Load data
    antibodies = load_hic_data(data_dir)
    print(f"\nLoaded {len(antibodies)} antibodies with HIC retention data")

    if len(antibodies) < 20:
        print("Error: Not enough data for training")
        return 1

    # Initialize ESM-2 from Hugging Face
    embedder = ESM2Embedder(args.model)

    results = {}

    # Test different approaches
    if args.approach in ["simple", "all"]:
        print("\n" + "=" * 70)
        print("Approach 1: Simple Mean Embedding (Full Sequence)")
        print("=" * 70)
        results['simple'] = train_simple_esm_model(embedder, antibodies)

    if args.approach in ["vh_vl", "all"]:
        print("\n" + "=" * 70)
        print("Approach 2: Separate VH + VL Embeddings")
        print("=" * 70)
        results['vh_vl'] = train_vh_vl_separate_model(embedder, antibodies)

    if args.approach in ["pca", "all"]:
        print("\n" + "=" * 70)
        print("Approach 3: Full Features with PCA")
        print("=" * 70)
        results['pca'] = train_esm_hic_model_pca(embedder, antibodies, n_components=30)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Approach':<25} {'Mean CV ρ':>12} {'Overall ρ':>12}")
    print("-" * 50)

    for name, res in results.items():
        print(f"{name:<25} {res['mean_rho']:>12.3f} {res['overall_rho']:>12.3f}")

    # Compare with baseline
    print("\n" + "-" * 50)
    print("Comparison:")
    print(f"  Current ML HIC (sequence-only):  ρ = 0.351")
    print(f"  Theoretical sequence-only max:   ρ ~ 0.40")
    print(f"  SOTA PROPERMAB (with structure): ρ = 0.75")

    best_approach = max(results.keys(), key=lambda k: results[k]['overall_rho'])
    best_rho = results[best_approach]['overall_rho']

    print(f"\n  Best ESM-2 approach: {best_approach} (ρ = {best_rho:.3f})")

    improvement = ((best_rho - 0.351) / 0.351) * 100
    if improvement > 0:
        print(f"  Improvement over baseline: +{improvement:.1f}%")
    else:
        print(f"  Change vs baseline: {improvement:.1f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
