#!/usr/bin/env python3
"""
ESM-2 Knowledge Distillation with GPU (MPS/CUDA)

Strategy:
1. Use ESM-2 650M on GPU to extract high-quality embeddings (teacher)
2. Train a lightweight MLP student that learns to:
   - Predict HIC directly from sequence features
   - Match teacher embedding patterns (soft targets)
3. Deploy student model on CPU (no ESM needed at inference)

Key improvements over CPU version:
- Larger teacher model (650M vs 150M)
- GPU acceleration for embedding extraction
- Better student architecture with attention
"""

from __future__ import annotations

import csv
import math
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# =============================================================================
# Device Selection
# =============================================================================

def get_device() -> torch.device:
    """Get best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class AntibodyHIC:
    vh_sequence: str
    vl_sequence: str
    hic_rt: float


# =============================================================================
# Sequence Feature Extraction
# =============================================================================

AA_PROPERTIES = {
    'A': [1.8, 0, 0, 88.6, 0], 'R': [-4.5, 1, 0, 173.4, 0],
    'N': [-3.5, 0, 0, 114.1, 0], 'D': [-3.5, -1, 0, 111.1, 0],
    'C': [2.5, 0, 0, 108.5, 0], 'Q': [-3.5, 0, 0, 143.8, 0],
    'E': [-3.5, -1, 0, 138.4, 0], 'G': [-0.4, 0, 0, 60.1, 0],
    'H': [-3.2, 0.5, 1, 153.2, 0], 'I': [4.5, 0, 0, 166.7, 0],
    'L': [3.8, 0, 0, 166.7, 0], 'K': [-3.9, 1, 0, 168.6, 0],
    'M': [1.9, 0, 0, 162.9, 0], 'F': [2.8, 0, 1, 189.9, 0],
    'P': [-1.6, 0, 0, 112.7, 0], 'S': [-0.8, 0, 0, 89.0, 0],
    'T': [-0.7, 0, 0, 116.1, 0], 'W': [-0.9, 0, 1, 227.8, 0],
    'Y': [-1.3, 0, 1, 193.6, 0], 'V': [4.2, 0, 0, 140.0, 0],
}

# One-hot encoding for amino acids
AA_TO_IDX = {aa: i for i, aa in enumerate(sorted(AA_PROPERTIES.keys()))}


def sequence_to_features(seq: str, max_len: int = 150) -> np.ndarray:
    """Convert sequence to feature matrix."""
    features = np.zeros((max_len, 25), dtype=np.float32)  # 20 one-hot + 5 properties

    for i, aa in enumerate(seq[:max_len]):
        if aa in AA_TO_IDX:
            # One-hot encoding
            features[i, AA_TO_IDX[aa]] = 1.0
            # Physicochemical properties
            features[i, 20:25] = AA_PROPERTIES[aa]

    return features


def extract_global_features(vh: str, vl: str) -> np.ndarray:
    """Extract global sequence features."""
    full_seq = vh + vl
    features = []

    # AA composition (20)
    for aa in sorted(AA_PROPERTIES.keys()):
        features.append(full_seq.count(aa) / len(full_seq))

    # Global properties
    for prop_idx in range(5):
        prop_sum = sum(AA_PROPERTIES.get(aa, [0]*5)[prop_idx] for aa in full_seq)
        features.append(prop_sum / len(full_seq))

    # VH/VL specific
    vh_hydro = sum(AA_PROPERTIES.get(aa, [0])[0] for aa in vh) / len(vh)
    vl_hydro = sum(AA_PROPERTIES.get(aa, [0])[0] for aa in vl) / len(vl)
    features.extend([vh_hydro, vl_hydro, len(vh)/150, len(vl)/120])

    # CDR-H3 features
    cdr_h3 = vh[min(95, len(vh)-15):min(110, len(vh))]
    if cdr_h3:
        cdr_h3_hydro = sum(AA_PROPERTIES.get(aa, [0])[0] for aa in cdr_h3) / len(cdr_h3)
        cdr_h3_aromatic = sum(AA_PROPERTIES.get(aa, [0,0,0])[2] for aa in cdr_h3) / len(cdr_h3)
        features.extend([cdr_h3_hydro, cdr_h3_aromatic, len(cdr_h3)/20])
    else:
        features.extend([0, 0, 0])

    return np.array(features, dtype=np.float32)


# =============================================================================
# Student Model with Attention
# =============================================================================

class SequenceEncoder(nn.Module):
    """Encode sequence using 1D convolutions + attention."""

    def __init__(self, input_dim: int = 25, hidden_dim: int = 64, output_dim: int = 128):
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(hidden_dim, output_dim, kernel_size=7, padding=3)

        # Attention
        self.attention = nn.Sequential(
            nn.Linear(output_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

        self.norm = nn.LayerNorm(output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        x = x.transpose(1, 2)  # (batch, input_dim, seq_len)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.transpose(1, 2)  # (batch, seq_len, output_dim)

        # Attention pooling
        attn_weights = F.softmax(self.attention(x), dim=1)  # (batch, seq_len, 1)
        x = (x * attn_weights).sum(dim=1)  # (batch, output_dim)

        return self.norm(x)


class HICDistillationStudent(nn.Module):
    """
    Student model for HIC prediction via distillation.

    Architecture:
    - Sequence encoder (conv + attention) for VH and VL
    - Global feature MLP
    - Combined prediction head
    - Optional embedding projection for distillation
    """

    def __init__(
        self,
        global_feat_dim: int = 32,
        seq_hidden_dim: int = 64,
        seq_output_dim: int = 128,
        teacher_embed_dim: int = 1280,  # ESM-2 650M
        use_distillation: bool = True
    ):
        super().__init__()

        # Sequence encoders
        self.vh_encoder = SequenceEncoder(25, seq_hidden_dim, seq_output_dim)
        self.vl_encoder = SequenceEncoder(25, seq_hidden_dim, seq_output_dim)

        # Global feature processor
        self.global_mlp = nn.Sequential(
            nn.Linear(global_feat_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64)
        )

        # Combined dimension
        combined_dim = seq_output_dim * 2 + 64

        # HIC prediction head
        self.hic_head = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Embedding projection for distillation
        self.use_distillation = use_distillation
        if use_distillation:
            self.embed_projection = nn.Sequential(
                nn.Linear(combined_dim, 512),
                nn.ReLU(),
                nn.Linear(512, teacher_embed_dim)
            )

    def forward(
        self,
        vh_seq: torch.Tensor,
        vl_seq: torch.Tensor,
        global_feat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Encode sequences
        vh_enc = self.vh_encoder(vh_seq)
        vl_enc = self.vl_encoder(vl_seq)

        # Process global features
        global_enc = self.global_mlp(global_feat)

        # Combine
        combined = torch.cat([vh_enc, vl_enc, global_enc], dim=1)

        # HIC prediction
        hic_pred = self.hic_head(combined).squeeze(-1)

        # Embedding projection (for distillation)
        embed_pred = None
        if self.use_distillation:
            embed_pred = self.embed_projection(combined)

        return hic_pred, embed_pred


# =============================================================================
# Dataset
# =============================================================================

class HICDistillDataset(Dataset):
    def __init__(
        self,
        antibodies: list[AntibodyHIC],
        teacher_embeddings: np.ndarray | None = None,
        vh_max_len: int = 150,
        vl_max_len: int = 120
    ):
        self.antibodies = antibodies
        self.teacher_embeddings = teacher_embeddings
        self.vh_max_len = vh_max_len
        self.vl_max_len = vl_max_len

        # Pre-extract features
        self.vh_features = [sequence_to_features(ab.vh_sequence, vh_max_len) for ab in antibodies]
        self.vl_features = [sequence_to_features(ab.vl_sequence, vl_max_len) for ab in antibodies]
        self.global_features = [extract_global_features(ab.vh_sequence, ab.vl_sequence) for ab in antibodies]
        self.labels = [ab.hic_rt for ab in antibodies]

    def __len__(self):
        return len(self.antibodies)

    def __getitem__(self, idx):
        item = {
            'vh_seq': torch.tensor(self.vh_features[idx]),
            'vl_seq': torch.tensor(self.vl_features[idx]),
            'global_feat': torch.tensor(self.global_features[idx]),
            'hic': torch.tensor(self.labels[idx], dtype=torch.float32),
        }
        if self.teacher_embeddings is not None:
            item['teacher_embed'] = torch.tensor(self.teacher_embeddings[idx])
        return item


# =============================================================================
# Teacher Embedding Extraction
# =============================================================================

def extract_teacher_embeddings_gpu(
    antibodies: list[AntibodyHIC],
    model_name: str = "facebook/esm2_t33_650M_UR50D",
    batch_size: int = 4,
    verbose: bool = True
) -> np.ndarray:
    """Extract ESM-2 embeddings using GPU."""
    device = get_device()

    if verbose:
        print(f"Loading teacher model: {model_name}...")
        print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model = model.to(device)
    model.eval()

    if verbose:
        print(f"Model loaded. Extracting embeddings for {len(antibodies)} antibodies...")

    embeddings = []

    for i in range(0, len(antibodies), batch_size):
        batch = antibodies[i:i+batch_size]
        sequences = [ab.vh_sequence + ab.vl_sequence for ab in batch]

        inputs = tokenizer(sequences, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            # Mean pooling over sequence length
            attention_mask = inputs['attention_mask'].unsqueeze(-1)
            hidden_states = outputs.last_hidden_state
            masked_hidden = hidden_states * attention_mask
            summed = masked_hidden.sum(dim=1)
            counts = attention_mask.sum(dim=1)
            mean_pooled = summed / counts

            embeddings.append(mean_pooled.cpu().numpy())

        if verbose and (i + batch_size) % 20 == 0:
            print(f"  Processed {min(i + batch_size, len(antibodies))}/{len(antibodies)}...")

    return np.vstack(embeddings)


# =============================================================================
# Training
# =============================================================================

def train_distillation(
    antibodies: list[AntibodyHIC],
    teacher_embeddings: np.ndarray,
    n_epochs: int = 200,
    batch_size: int = 16,
    lr: float = 0.001,
    alpha: float = 0.3,  # Distillation weight
    temperature: float = 2.0,
    verbose: bool = True
) -> tuple[HICDistillationStudent, dict]:
    """Train student model with knowledge distillation."""
    from sklearn.model_selection import train_test_split

    device = get_device()

    # Create dataset
    dataset = HICDistillDataset(antibodies, teacher_embeddings)

    # Split
    indices = list(range(len(dataset)))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Create model
    global_feat_dim = dataset.global_features[0].shape[0]
    teacher_embed_dim = teacher_embeddings.shape[1]

    model = HICDistillationStudent(
        global_feat_dim=global_feat_dim,
        teacher_embed_dim=teacher_embed_dim,
        use_distillation=True
    )
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    best_val_corr = -1
    best_state = None
    history = {'train_loss': [], 'val_loss': [], 'val_corr': []}

    if verbose:
        print(f"\nTraining distillation model on {device}...")
        print(f"  Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")
        print(f"  Alpha (distillation): {alpha}, Temperature: {temperature}")

    for epoch in range(n_epochs):
        # Training
        model.train()
        epoch_loss = 0
        n_batches = 0

        for batch in train_loader:
            vh_seq = batch['vh_seq'].to(device)
            vl_seq = batch['vl_seq'].to(device)
            global_feat = batch['global_feat'].to(device)
            hic_true = batch['hic'].to(device)
            teacher_embed = batch['teacher_embed'].to(device)

            optimizer.zero_grad()

            hic_pred, embed_pred = model(vh_seq, vl_seq, global_feat)

            # HIC loss
            hic_loss = F.mse_loss(hic_pred, hic_true)

            # Distillation loss (cosine similarity)
            embed_pred_norm = F.normalize(embed_pred, dim=1)
            teacher_embed_norm = F.normalize(teacher_embed, dim=1)
            distill_loss = 1 - (embed_pred_norm * teacher_embed_norm).sum(dim=1).mean()

            # Combined loss
            loss = (1 - alpha) * hic_loss + alpha * distill_loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validation
        model.eval()
        val_preds = []
        val_trues = []

        with torch.no_grad():
            for batch in val_loader:
                vh_seq = batch['vh_seq'].to(device)
                vl_seq = batch['vl_seq'].to(device)
                global_feat = batch['global_feat'].to(device)

                hic_pred, _ = model(vh_seq, vl_seq, global_feat)
                val_preds.extend(hic_pred.cpu().numpy().tolist())
                val_trues.extend(batch['hic'].numpy().tolist())

        val_corr = spearman_correlation(val_trues, val_preds)[0]

        history['train_loss'].append(epoch_loss / n_batches)
        history['val_corr'].append(val_corr)

        if val_corr > best_val_corr:
            best_val_corr = val_corr
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if verbose and (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch + 1}/{n_epochs}: loss={epoch_loss/n_batches:.4f}, val_corr={val_corr:.3f}")

    # Load best model
    model.load_state_dict(best_state)
    model = model.to('cpu')  # Move to CPU for deployment

    if verbose:
        print(f"\n  Best validation correlation: ρ = {best_val_corr:.3f}")

    return model, history


def train_direct_student(
    antibodies: list[AntibodyHIC],
    n_epochs: int = 200,
    batch_size: int = 16,
    lr: float = 0.001,
    n_folds: int = 5,
    verbose: bool = True
) -> dict:
    """Train student model directly (no distillation) for comparison."""
    from sklearn.model_selection import KFold

    device = get_device()
    dataset = HICDistillDataset(antibodies)

    global_feat_dim = dataset.global_features[0].shape[0]

    if verbose:
        print(f"\nTraining direct student (no distillation) on {device}...")

    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []
    all_predictions = np.zeros(len(antibodies))

    for fold, (train_idx, val_idx) in enumerate(kfold.split(range(len(dataset)))):
        train_dataset = torch.utils.data.Subset(dataset, train_idx)
        val_dataset = torch.utils.data.Subset(dataset, val_idx)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        model = HICDistillationStudent(
            global_feat_dim=global_feat_dim,
            use_distillation=False
        )
        model = model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

        best_val_corr = -1
        best_predictions = None

        for epoch in range(n_epochs):
            model.train()
            for batch in train_loader:
                vh_seq = batch['vh_seq'].to(device)
                vl_seq = batch['vl_seq'].to(device)
                global_feat = batch['global_feat'].to(device)
                hic_true = batch['hic'].to(device)

                optimizer.zero_grad()
                hic_pred, _ = model(vh_seq, vl_seq, global_feat)
                loss = F.mse_loss(hic_pred, hic_true)
                loss.backward()
                optimizer.step()

            # Validation
            model.eval()
            val_preds = []
            val_trues = []

            with torch.no_grad():
                for batch in val_loader:
                    vh_seq = batch['vh_seq'].to(device)
                    vl_seq = batch['vl_seq'].to(device)
                    global_feat = batch['global_feat'].to(device)

                    hic_pred, _ = model(vh_seq, vl_seq, global_feat)
                    val_preds.extend(hic_pred.cpu().numpy().tolist())
                    val_trues.extend(batch['hic'].numpy().tolist())

            val_corr = spearman_correlation(val_trues, val_preds)[0]

            if val_corr > best_val_corr:
                best_val_corr = val_corr
                best_predictions = val_preds

        all_predictions[val_idx] = best_predictions
        fold_results.append({'fold': fold + 1, 'rho': best_val_corr})

        if verbose:
            print(f"  Fold {fold + 1}: ρ = {best_val_corr:.3f}")

    labels = [ab.hic_rt for ab in antibodies]
    overall_rho = spearman_correlation(labels, all_predictions.tolist())[0]

    if verbose:
        print(f"\n  Mean CV ρ = {sum(f['rho'] for f in fold_results) / n_folds:.3f}")
        print(f"  Overall ρ = {overall_rho:.3f}")

    return {
        'fold_results': fold_results,
        'mean_rho': sum(f['rho'] for f in fold_results) / n_folds,
        'overall_rho': overall_rho,
    }


# =============================================================================
# Utilities
# =============================================================================

def spearman_correlation(x: list[float], y: list[float]) -> tuple[float, float]:
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

    parser = argparse.ArgumentParser(description="ESM-2 Distillation with GPU")
    parser.add_argument("--data-dir", "-d", type=str,
                       default=str(Path(__file__).parent / "data"))
    parser.add_argument("--teacher", "-t", type=str,
                       default="facebook/esm2_t33_650M_UR50D",
                       help="Teacher model (ESM-2 650M by default)")
    parser.add_argument("--epochs", "-e", type=int, default=200)
    parser.add_argument("--alpha", "-a", type=float, default=0.3)
    parser.add_argument("--skip-teacher", action="store_true")
    parser.add_argument("--save-model", type=str, help="Save trained model")

    args = parser.parse_args()
    data_dir = Path(args.data_dir)

    print("=" * 70)
    print("ESM-2 Knowledge Distillation with GPU")
    print("=" * 70)
    print(f"Device: {get_device()}")

    # Load data
    antibodies = load_hic_data(data_dir)
    print(f"\nLoaded {len(antibodies)} antibodies")

    # Test 1: Direct student (baseline)
    print("\n" + "=" * 70)
    print("Test 1: Direct Student (No Distillation)")
    print("=" * 70)
    direct_results = train_direct_student(antibodies, n_epochs=args.epochs)

    if not args.skip_teacher:
        # Extract teacher embeddings with GPU
        print("\n" + "=" * 70)
        print("Test 2: Distillation from ESM-2 650M")
        print("=" * 70)

        teacher_embeddings = extract_teacher_embeddings_gpu(
            antibodies, args.teacher, batch_size=2
        )

        # Train with distillation
        distill_model, history = train_distillation(
            antibodies, teacher_embeddings,
            n_epochs=args.epochs, alpha=args.alpha
        )

        # Final evaluation
        dataset = HICDistillDataset(antibodies)
        distill_model.eval()

        all_preds = []
        with torch.no_grad():
            for i in range(len(dataset)):
                item = dataset[i]
                vh = item['vh_seq'].unsqueeze(0)
                vl = item['vl_seq'].unsqueeze(0)
                gf = item['global_feat'].unsqueeze(0)
                pred, _ = distill_model(vh, vl, gf)
                all_preds.append(pred.item())

        labels = [ab.hic_rt for ab in antibodies]
        distill_rho = spearman_correlation(labels, all_preds)[0]

        print(f"\n  Distillation model overall ρ = {distill_rho:.3f}")

        if args.save_model:
            torch.save({
                'model_state': distill_model.state_dict(),
                'config': {
                    'global_feat_dim': dataset.global_features[0].shape[0],
                    'teacher_embed_dim': teacher_embeddings.shape[1],
                }
            }, args.save_model)
            print(f"\n  Model saved to {args.save_model}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n{'Model':<35} {'Overall ρ':>12}")
    print("-" * 50)
    print(f"{'Direct Student (no distillation)':<35} {direct_results['overall_rho']:>12.3f}")
    if not args.skip_teacher:
        print(f"{'Distillation from ESM-2 650M':<35} {distill_rho:>12.3f}")

    print("\nComparison:")
    print(f"  Current ML HIC (scorer):   ρ = 0.351")
    print(f"  ESM-2 150M embeddings:     ρ = 0.419")
    print(f"  SOTA PROPERMAB:            ρ = 0.75")

    return 0


if __name__ == "__main__":
    sys.exit(main())
