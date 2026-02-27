#!/usr/bin/env python3
"""
ESM-2 Knowledge Distillation for HIC Prediction

Strategy:
1. Use large ESM-2 model (150M/650M) to generate embeddings (teacher)
2. Train a lightweight student model that:
   a) Takes sequence features as input
   b) Learns to predict the teacher's embeddings OR HIC directly
3. Deploy only the student model (fast CPU inference)

This allows us to capture the structural knowledge from PLMs
in a model that can run efficiently on CPU.
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
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
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
    hic_rt: float


# =============================================================================
# Sequence Feature Extraction (CPU-friendly)
# =============================================================================

# Amino acid properties for feature extraction
AA_PROPERTIES = {
    'A': {'hydrophobicity': 1.8, 'volume': 88.6, 'charge': 0, 'aromatic': 0},
    'R': {'hydrophobicity': -4.5, 'volume': 173.4, 'charge': 1, 'aromatic': 0},
    'N': {'hydrophobicity': -3.5, 'volume': 114.1, 'charge': 0, 'aromatic': 0},
    'D': {'hydrophobicity': -3.5, 'volume': 111.1, 'charge': -1, 'aromatic': 0},
    'C': {'hydrophobicity': 2.5, 'volume': 108.5, 'charge': 0, 'aromatic': 0},
    'Q': {'hydrophobicity': -3.5, 'volume': 143.8, 'charge': 0, 'aromatic': 0},
    'E': {'hydrophobicity': -3.5, 'volume': 138.4, 'charge': -1, 'aromatic': 0},
    'G': {'hydrophobicity': -0.4, 'volume': 60.1, 'charge': 0, 'aromatic': 0},
    'H': {'hydrophobicity': -3.2, 'volume': 153.2, 'charge': 0.5, 'aromatic': 1},
    'I': {'hydrophobicity': 4.5, 'volume': 166.7, 'charge': 0, 'aromatic': 0},
    'L': {'hydrophobicity': 3.8, 'volume': 166.7, 'charge': 0, 'aromatic': 0},
    'K': {'hydrophobicity': -3.9, 'volume': 168.6, 'charge': 1, 'aromatic': 0},
    'M': {'hydrophobicity': 1.9, 'volume': 162.9, 'charge': 0, 'aromatic': 0},
    'F': {'hydrophobicity': 2.8, 'volume': 189.9, 'charge': 0, 'aromatic': 1},
    'P': {'hydrophobicity': -1.6, 'volume': 112.7, 'charge': 0, 'aromatic': 0},
    'S': {'hydrophobicity': -0.8, 'volume': 89.0, 'charge': 0, 'aromatic': 0},
    'T': {'hydrophobicity': -0.7, 'volume': 116.1, 'charge': 0, 'aromatic': 0},
    'W': {'hydrophobicity': -0.9, 'volume': 227.8, 'charge': 0, 'aromatic': 1},
    'Y': {'hydrophobicity': -1.3, 'volume': 193.6, 'charge': 0, 'aromatic': 1},
    'V': {'hydrophobicity': 4.2, 'volume': 140.0, 'charge': 0, 'aromatic': 0},
}


def extract_sequence_features(vh: str, vl: str) -> np.ndarray:
    """
    Extract comprehensive sequence features for student model.

    Features:
    - Global composition (20 AA frequencies)
    - Physicochemical properties (hydrophobicity, charge, volume, aromaticity)
    - Position-specific features (N-term, C-term, CDR regions)
    - Dipeptide frequencies (top 50)
    - Length features
    """
    full_seq = vh + vl
    features = []

    # 1. Global AA composition (20 features)
    aa_counts = {aa: 0 for aa in AA_PROPERTIES}
    for aa in full_seq:
        if aa in aa_counts:
            aa_counts[aa] += 1
    total_len = len(full_seq)
    for aa in sorted(AA_PROPERTIES.keys()):
        features.append(aa_counts[aa] / total_len)

    # 2. Global physicochemical properties (4 features)
    hydro_sum = sum(AA_PROPERTIES.get(aa, {}).get('hydrophobicity', 0) for aa in full_seq)
    charge_sum = sum(AA_PROPERTIES.get(aa, {}).get('charge', 0) for aa in full_seq)
    volume_sum = sum(AA_PROPERTIES.get(aa, {}).get('volume', 100) for aa in full_seq)
    aromatic_sum = sum(AA_PROPERTIES.get(aa, {}).get('aromatic', 0) for aa in full_seq)

    features.append(hydro_sum / total_len)
    features.append(charge_sum / total_len)
    features.append(volume_sum / total_len)
    features.append(aromatic_sum / total_len)

    # 3. VH-specific properties (4 features)
    vh_hydro = sum(AA_PROPERTIES.get(aa, {}).get('hydrophobicity', 0) for aa in vh) / len(vh)
    vh_charge = sum(AA_PROPERTIES.get(aa, {}).get('charge', 0) for aa in vh) / len(vh)
    vh_aromatic = sum(AA_PROPERTIES.get(aa, {}).get('aromatic', 0) for aa in vh) / len(vh)
    features.extend([vh_hydro, vh_charge, vh_aromatic, len(vh) / 150])

    # 4. VL-specific properties (4 features)
    vl_hydro = sum(AA_PROPERTIES.get(aa, {}).get('hydrophobicity', 0) for aa in vl) / len(vl)
    vl_charge = sum(AA_PROPERTIES.get(aa, {}).get('charge', 0) for aa in vl) / len(vl)
    vl_aromatic = sum(AA_PROPERTIES.get(aa, {}).get('aromatic', 0) for aa in vl) / len(vl)
    features.extend([vl_hydro, vl_charge, vl_aromatic, len(vl) / 120])

    # 5. CDR-H3 approximate features (positions 95-110 in VH)
    cdr_h3_start = min(95, len(vh) - 15)
    cdr_h3_end = min(cdr_h3_start + 15, len(vh))
    cdr_h3 = vh[cdr_h3_start:cdr_h3_end]
    if cdr_h3:
        cdr_h3_hydro = sum(AA_PROPERTIES.get(aa, {}).get('hydrophobicity', 0) for aa in cdr_h3) / len(cdr_h3)
        cdr_h3_aromatic = sum(AA_PROPERTIES.get(aa, {}).get('aromatic', 0) for aa in cdr_h3) / len(cdr_h3)
        features.extend([cdr_h3_hydro, cdr_h3_aromatic, len(cdr_h3) / 20])
    else:
        features.extend([0, 0, 0])

    # 6. CDR-L3 approximate features (positions 89-97 in VL)
    cdr_l3_start = min(89, len(vl) - 10)
    cdr_l3_end = min(cdr_l3_start + 10, len(vl))
    cdr_l3 = vl[cdr_l3_start:cdr_l3_end]
    if cdr_l3:
        cdr_l3_hydro = sum(AA_PROPERTIES.get(aa, {}).get('hydrophobicity', 0) for aa in cdr_l3) / len(cdr_l3)
        cdr_l3_aromatic = sum(AA_PROPERTIES.get(aa, {}).get('aromatic', 0) for aa in cdr_l3) / len(cdr_l3)
        features.extend([cdr_l3_hydro, cdr_l3_aromatic, len(cdr_l3) / 15])
    else:
        features.extend([0, 0, 0])

    # 7. Top dipeptide frequencies (20 features - most informative for structure)
    top_dipeptides = ['YY', 'YF', 'FY', 'WW', 'WY', 'YW', 'FF', 'VV', 'LL', 'II',
                      'GG', 'SS', 'TT', 'AA', 'PP', 'KK', 'RR', 'EE', 'DD', 'NN']
    for dp in top_dipeptides:
        count = full_seq.count(dp)
        features.append(count / max(total_len - 1, 1))

    # 8. Motif features (specific patterns associated with hydrophobicity)
    # Aromatic clusters
    aromatic_cluster = sum(1 for i in range(len(full_seq)-1)
                          if full_seq[i] in 'FYW' and full_seq[i+1] in 'FYW')
    features.append(aromatic_cluster / max(total_len - 1, 1))

    # Hydrophobic stretches
    hydro_stretch = max(len(s) for s in ''.join('H' if aa in 'VILMFYW' else 'P' for aa in full_seq).split('P'))
    features.append(hydro_stretch / 10)

    return np.array(features, dtype=np.float32)


# =============================================================================
# Student Model (Lightweight MLP)
# =============================================================================

class HICStudentModel(nn.Module):
    """
    Lightweight student model for HIC prediction.

    Takes sequence features as input and predicts HIC retention time.
    Can be trained via:
    1. Direct supervision (HIC labels)
    2. Knowledge distillation (teacher embeddings)
    3. Combined loss
    """

    def __init__(self, input_dim: int, hidden_dims: list[int] = [128, 64, 32]):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2),
            ])
            prev_dim = hidden_dim

        # Output layer for HIC prediction
        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class HICDistillationModel(nn.Module):
    """
    Student model with embedding projection for distillation.

    Has two heads:
    1. Embedding head: projects to teacher embedding space
    2. HIC head: predicts HIC directly
    """

    def __init__(
        self,
        input_dim: int,
        teacher_embed_dim: int,
        hidden_dims: list[int] = [256, 128, 64]
    ):
        super().__init__()

        # Shared backbone
        backbone_layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims[:-1]:
            backbone_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2),
            ])
            prev_dim = hidden_dim

        self.backbone = nn.Sequential(*backbone_layers)

        # Embedding projection head (for distillation)
        self.embed_head = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], teacher_embed_dim)
        )

        # HIC prediction head
        self.hic_head = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1], 1)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        backbone_out = self.backbone(x)
        embed_pred = self.embed_head(backbone_out)
        hic_pred = self.hic_head(backbone_out).squeeze(-1)
        return embed_pred, hic_pred


# =============================================================================
# Dataset
# =============================================================================

class HICDataset(Dataset):
    """Dataset for HIC distillation training."""

    def __init__(
        self,
        antibodies: list[AntibodyHIC],
        teacher_embeddings: np.ndarray | None = None
    ):
        self.antibodies = antibodies
        self.teacher_embeddings = teacher_embeddings

        # Pre-extract sequence features
        self.features = np.array([
            extract_sequence_features(ab.vh_sequence, ab.vl_sequence)
            for ab in antibodies
        ])
        self.labels = np.array([ab.hic_rt for ab in antibodies])

    def __len__(self) -> int:
        return len(self.antibodies)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {
            'features': torch.tensor(self.features[idx], dtype=torch.float32),
            'hic': torch.tensor(self.labels[idx], dtype=torch.float32),
        }
        if self.teacher_embeddings is not None:
            item['teacher_embed'] = torch.tensor(
                self.teacher_embeddings[idx], dtype=torch.float32
            )
        return item


# =============================================================================
# Training Functions
# =============================================================================

def extract_teacher_embeddings(
    antibodies: list[AntibodyHIC],
    model_name: str = "facebook/esm2_t30_150M_UR50D",
    verbose: bool = True
) -> np.ndarray:
    """Extract embeddings from teacher model (ESM-2)."""

    if verbose:
        print(f"Loading teacher model: {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if verbose:
        print(f"Teacher model loaded on {device}")
        print(f"Extracting embeddings for {len(antibodies)} antibodies...")

    embeddings = []

    for i, ab in enumerate(antibodies):
        full_seq = ab.vh_sequence + ab.vl_sequence

        inputs = tokenizer(full_seq, return_tensors="pt", add_special_tokens=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            # Mean pooling (excluding special tokens)
            emb = outputs.last_hidden_state[0, 1:-1, :].mean(dim=0)
            embeddings.append(emb.cpu().numpy())

        if verbose and (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(antibodies)}...")

    return np.array(embeddings)


def train_distillation_model(
    antibodies: list[AntibodyHIC],
    teacher_embeddings: np.ndarray,
    n_epochs: int = 100,
    batch_size: int = 16,
    lr: float = 0.001,
    alpha: float = 0.5,  # Weight for distillation loss
    verbose: bool = True
) -> tuple[HICDistillationModel, dict[str, Any]]:
    """
    Train student model with knowledge distillation.

    Loss = alpha * MSE(student_embed, teacher_embed) + (1-alpha) * MSE(hic_pred, hic_true)
    """
    from sklearn.model_selection import train_test_split

    # Create dataset
    dataset = HICDataset(antibodies, teacher_embeddings)

    # Split data
    indices = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

    train_features = dataset.features[train_idx]
    train_labels = dataset.labels[train_idx]
    train_embeds = teacher_embeddings[train_idx]

    val_features = dataset.features[val_idx]
    val_labels = dataset.labels[val_idx]

    # Create model
    input_dim = train_features.shape[1]
    teacher_embed_dim = teacher_embeddings.shape[1]

    model = HICDistillationModel(
        input_dim=input_dim,
        teacher_embed_dim=teacher_embed_dim,
        hidden_dims=[256, 128, 64]
    )

    # Training setup
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)

    best_val_corr = -1
    best_state = None
    history = {'train_loss': [], 'val_loss': [], 'val_corr': []}

    # Convert to tensors
    train_X = torch.tensor(train_features, dtype=torch.float32)
    train_y = torch.tensor(train_labels, dtype=torch.float32)
    train_E = torch.tensor(train_embeds, dtype=torch.float32)

    val_X = torch.tensor(val_features, dtype=torch.float32)
    val_y = torch.tensor(val_labels, dtype=torch.float32)

    if verbose:
        print(f"\nTraining distillation model...")
        print(f"  Input dim: {input_dim}")
        print(f"  Teacher embed dim: {teacher_embed_dim}")
        print(f"  Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")
        print(f"  Alpha (distillation weight): {alpha}")

    for epoch in range(n_epochs):
        # Training
        model.train()

        # Shuffle
        perm = torch.randperm(len(train_X))
        train_X_shuffled = train_X[perm]
        train_y_shuffled = train_y[perm]
        train_E_shuffled = train_E[perm]

        epoch_loss = 0
        n_batches = 0

        for i in range(0, len(train_X), batch_size):
            batch_X = train_X_shuffled[i:i+batch_size]
            batch_y = train_y_shuffled[i:i+batch_size]
            batch_E = train_E_shuffled[i:i+batch_size]

            optimizer.zero_grad()

            embed_pred, hic_pred = model(batch_X)

            # Combined loss
            distill_loss = nn.MSELoss()(embed_pred, batch_E)
            hic_loss = nn.MSELoss()(hic_pred, batch_y)
            loss = alpha * distill_loss + (1 - alpha) * hic_loss

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / n_batches

        # Validation
        model.eval()
        with torch.no_grad():
            _, val_pred = model(val_X)
            val_loss = nn.MSELoss()(val_pred, val_y).item()

            # Spearman correlation
            val_corr = spearman_correlation(val_y.numpy().tolist(), val_pred.numpy().tolist())[0]

        scheduler.step(val_loss)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['val_corr'].append(val_corr)

        if val_corr > best_val_corr:
            best_val_corr = val_corr
            best_state = model.state_dict().copy()

        if verbose and (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch + 1}/{n_epochs}: train_loss={avg_train_loss:.4f}, "
                  f"val_loss={val_loss:.4f}, val_corr={val_corr:.3f}")

    # Load best model
    model.load_state_dict(best_state)

    if verbose:
        print(f"\n  Best validation correlation: ρ = {best_val_corr:.3f}")

    return model, history


def train_direct_student(
    antibodies: list[AntibodyHIC],
    n_epochs: int = 100,
    batch_size: int = 16,
    lr: float = 0.001,
    n_folds: int = 5,
    verbose: bool = True
) -> dict[str, Any]:
    """
    Train student model directly on HIC labels (without distillation).
    Uses cross-validation for fair comparison.
    """
    from sklearn.model_selection import KFold

    # Create dataset
    features = np.array([
        extract_sequence_features(ab.vh_sequence, ab.vl_sequence)
        for ab in antibodies
    ])
    labels = np.array([ab.hic_rt for ab in antibodies])

    input_dim = features.shape[1]

    if verbose:
        print(f"\nTraining direct student model (no distillation)...")
        print(f"  Input features: {input_dim}")
        print(f"  Samples: {len(antibodies)}")

    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []
    all_predictions = np.zeros(len(labels))

    for fold, (train_idx, val_idx) in enumerate(kfold.split(features)):
        train_X = torch.tensor(features[train_idx], dtype=torch.float32)
        train_y = torch.tensor(labels[train_idx], dtype=torch.float32)
        val_X = torch.tensor(features[val_idx], dtype=torch.float32)
        val_y = torch.tensor(labels[val_idx], dtype=torch.float32)

        # Create model
        model = HICStudentModel(input_dim=input_dim, hidden_dims=[128, 64, 32])
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        best_val_corr = -1
        best_predictions = None

        for epoch in range(n_epochs):
            # Training
            model.train()
            perm = torch.randperm(len(train_X))

            for i in range(0, len(train_X), batch_size):
                batch_X = train_X[perm[i:i+batch_size]]
                batch_y = train_y[perm[i:i+batch_size]]

                optimizer.zero_grad()
                pred = model(batch_X)
                loss = nn.MSELoss()(pred, batch_y)
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            with torch.no_grad():
                val_pred = model(val_X)
                val_corr = spearman_correlation(val_y.numpy().tolist(), val_pred.numpy().tolist())[0]

                if val_corr > best_val_corr:
                    best_val_corr = val_corr
                    best_predictions = val_pred.numpy()

        all_predictions[val_idx] = best_predictions
        fold_results.append({'fold': fold + 1, 'rho': best_val_corr})

        if verbose:
            print(f"  Fold {fold + 1}: ρ = {best_val_corr:.3f}")

    overall_rho = spearman_correlation(labels.tolist(), all_predictions.tolist())[0]

    results = {
        'fold_results': fold_results,
        'mean_rho': sum(f['rho'] for f in fold_results) / n_folds,
        'overall_rho': overall_rho,
        'input_dim': input_dim,
    }

    if verbose:
        print(f"\n  Mean CV ρ = {results['mean_rho']:.3f}")
        print(f"  Overall ρ = {overall_rho:.3f}")

    return results


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

    rho = num / (den_x * den_y)
    return rho, 0.0


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

    parser = argparse.ArgumentParser(description="ESM-2 Knowledge Distillation for HIC")
    parser.add_argument("--data-dir", "-d", type=str,
                       default=str(Path(__file__).parent / "data"))
    parser.add_argument("--teacher", "-t", type=str,
                       default="facebook/esm2_t30_150M_UR50D",
                       help="Teacher model (ESM-2)")
    parser.add_argument("--alpha", "-a", type=float, default=0.3,
                       help="Distillation loss weight (0-1)")
    parser.add_argument("--epochs", "-e", type=int, default=100)
    parser.add_argument("--skip-teacher", action="store_true",
                       help="Skip teacher extraction (test direct student)")

    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    print("=" * 70)
    print("ESM-2 Knowledge Distillation for HIC Prediction")
    print("=" * 70)

    # Load data
    antibodies = load_hic_data(data_dir)
    print(f"\nLoaded {len(antibodies)} antibodies with HIC retention data")

    if len(antibodies) < 20:
        print("Error: Not enough data")
        return 1

    # Test 1: Direct student (no distillation)
    print("\n" + "=" * 70)
    print("Test 1: Direct Student Model (No Distillation)")
    print("=" * 70)
    direct_results = train_direct_student(antibodies, n_epochs=args.epochs)

    if not args.skip_teacher:
        # Extract teacher embeddings
        print("\n" + "=" * 70)
        print("Test 2: Distillation from ESM-2 Teacher")
        print("=" * 70)

        teacher_embeddings = extract_teacher_embeddings(antibodies, args.teacher)

        # Train with distillation
        distill_model, history = train_distillation_model(
            antibodies,
            teacher_embeddings,
            n_epochs=args.epochs,
            alpha=args.alpha
        )

        # Final evaluation
        features = np.array([
            extract_sequence_features(ab.vh_sequence, ab.vl_sequence)
            for ab in antibodies
        ])
        labels = np.array([ab.hic_rt for ab in antibodies])

        distill_model.eval()
        with torch.no_grad():
            X = torch.tensor(features, dtype=torch.float32)
            _, predictions = distill_model(X)
            distill_rho = spearman_correlation(labels.tolist(), predictions.numpy().tolist())[0]

        print(f"\n  Distillation model overall ρ = {distill_rho:.3f}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Approach':<35} {'Overall ρ':>12}")
    print("-" * 50)
    print(f"{'Direct Student (no distillation)':<35} {direct_results['overall_rho']:>12.3f}")
    if not args.skip_teacher:
        print(f"{'Distillation Student':<35} {distill_rho:>12.3f}")

    print("\nComparison:")
    print(f"  Current ML HIC (sequence):  ρ = 0.351")
    print(f"  ESM-2 150M (full):          ρ = 0.402")
    print(f"  Theoretical max:            ρ ~ 0.40")

    return 0


if __name__ == "__main__":
    sys.exit(main())
