#!/usr/bin/env python3
"""
DeepSP model recreated in PyTorch with weights loaded from Keras H5 files.

This avoids TensorFlow dependency issues while still using the trained DeepSP models.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepSPModel(nn.Module):
    """
    DeepSP CNN model architecture (recreated from Keras JSON).

    Architecture:
    - Conv1D(21->128, k=5) + BatchNorm + Dropout(0.3)
    - Conv1D(128->96, k=4) + BatchNorm
    - Conv1D(96->32, k=5) + BatchNorm + MaxPool(2)
    - Flatten
    - Dense(112) + ReLU
    - Dense(48) + ReLU
    - Dense(10) (output)
    """

    def __init__(self, model_type: str = 'SAPpos'):
        super().__init__()
        self.model_type = model_type

        # Note: Keras Conv1D uses channels_last, PyTorch uses channels_first
        # Input: (batch, 272, 21) in Keras -> (batch, 21, 272) in PyTorch

        self.conv1 = nn.Conv1d(21, 128, kernel_size=5, padding=0)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.3)

        self.conv2 = nn.Conv1d(128, 96, kernel_size=4, padding=0)
        self.bn2 = nn.BatchNorm1d(96)

        self.conv3 = nn.Conv1d(96, 32, kernel_size=5, padding=0)
        self.bn3 = nn.BatchNorm1d(32)

        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Calculate flatten size: 272 -> 268 (conv1) -> 265 (conv2) -> 261 (conv3) -> 130 (maxpool)
        # 130 * 32 = 4160
        self.fc1 = nn.Linear(4160, 112)
        self.fc2 = nn.Linear(112, 48)
        self.fc3 = nn.Linear(48, 10)  # 10 outputs for each model type

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (batch, seq_len=272, channels=21)
        # Transpose for PyTorch Conv1D: (batch, channels, seq_len)
        x = x.transpose(1, 2)

        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout(x)

        x = F.relu(self.bn2(self.conv2(x)))

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.maxpool(x)

        x = x.flatten(1)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


def load_keras_weights(model: DeepSPModel, h5_path: str) -> None:
    """Load weights from Keras H5 file into PyTorch model."""
    with h5py.File(h5_path, 'r') as f:
        # Navigate H5 structure
        # Keras saves weights under model_weights/layer_name/...

        def get_weights(layer_name: str):
            """Get kernel and bias weights for a layer."""
            try:
                layer_group = f['model_weights'][layer_name][layer_name]
                kernel = np.array(layer_group['kernel:0'])
                bias = np.array(layer_group['bias:0'])
                return kernel, bias
            except KeyError:
                return None, None

        def get_bn_weights(layer_name: str):
            """Get BatchNorm weights."""
            try:
                layer_group = f['model_weights'][layer_name][layer_name]
                gamma = np.array(layer_group['gamma:0'])
                beta = np.array(layer_group['beta:0'])
                moving_mean = np.array(layer_group['moving_mean:0'])
                moving_variance = np.array(layer_group['moving_variance:0'])
                return gamma, beta, moving_mean, moving_variance
            except KeyError:
                return None, None, None, None

        # Conv1D_1 -> conv1
        kernel, bias = get_weights('Conv1D_1')
        if kernel is not None:
            # Keras Conv1D kernel shape: (kernel_size, in_channels, out_channels)
            # PyTorch Conv1d kernel shape: (out_channels, in_channels, kernel_size)
            model.conv1.weight.data = torch.from_numpy(kernel.transpose(2, 1, 0).copy())
            model.conv1.bias.data = torch.from_numpy(bias.copy())

        # batch_normalization -> bn1
        gamma, beta, mean, var = get_bn_weights('batch_normalization')
        if gamma is not None:
            model.bn1.weight.data = torch.from_numpy(gamma.copy())
            model.bn1.bias.data = torch.from_numpy(beta.copy())
            model.bn1.running_mean.data = torch.from_numpy(mean.copy())
            model.bn1.running_var.data = torch.from_numpy(var.copy())

        # Conv1D_2 -> conv2
        kernel, bias = get_weights('Conv1D_2')
        if kernel is not None:
            model.conv2.weight.data = torch.from_numpy(kernel.transpose(2, 1, 0).copy())
            model.conv2.bias.data = torch.from_numpy(bias.copy())

        # batch_normalization_1 -> bn2
        gamma, beta, mean, var = get_bn_weights('batch_normalization_1')
        if gamma is not None:
            model.bn2.weight.data = torch.from_numpy(gamma.copy())
            model.bn2.bias.data = torch.from_numpy(beta.copy())
            model.bn2.running_mean.data = torch.from_numpy(mean.copy())
            model.bn2.running_var.data = torch.from_numpy(var.copy())

        # Conv1D_3 -> conv3
        kernel, bias = get_weights('Conv1D_3')
        if kernel is not None:
            model.conv3.weight.data = torch.from_numpy(kernel.transpose(2, 1, 0).copy())
            model.conv3.bias.data = torch.from_numpy(bias.copy())

        # batch_normalization_2 -> bn3
        gamma, beta, mean, var = get_bn_weights('batch_normalization_2')
        if gamma is not None:
            model.bn3.weight.data = torch.from_numpy(gamma.copy())
            model.bn3.bias.data = torch.from_numpy(beta.copy())
            model.bn3.running_mean.data = torch.from_numpy(mean.copy())
            model.bn3.running_var.data = torch.from_numpy(var.copy())

        # Dense_1 -> fc1
        kernel, bias = get_weights('Dense_1')
        if kernel is not None:
            # Keras Dense kernel shape: (in_features, out_features)
            # PyTorch Linear weight shape: (out_features, in_features)
            model.fc1.weight.data = torch.from_numpy(kernel.T.copy())
            model.fc1.bias.data = torch.from_numpy(bias.copy())

        # Dense_2 -> fc2
        kernel, bias = get_weights('Dense_2')
        if kernel is not None:
            model.fc2.weight.data = torch.from_numpy(kernel.T.copy())
            model.fc2.bias.data = torch.from_numpy(bias.copy())

        # Dense_3 -> fc3
        kernel, bias = get_weights('Dense_3')
        if kernel is not None:
            model.fc3.weight.data = torch.from_numpy(kernel.T.copy())
            model.fc3.bias.data = torch.from_numpy(bias.copy())


class DeepSPPredictor:
    """Wrapper for all DeepSP models (SAP_pos, SCM_neg, SCM_pos)."""

    def __init__(self, model_dir: str):
        self.model_dir = Path(model_dir)
        self.models = {}
        self._load_models()

    def _load_models(self):
        """Load all three DeepSP models."""
        model_files = {
            'SAPpos': 'Conv1D_regression_SAPpos.h5',
            'SCMneg': 'Conv1D_regression_SCMneg.h5',
            'SCMpos': 'Conv1D_regression_SCMpos.h5',
        }

        for model_type, h5_file in model_files.items():
            h5_path = self.model_dir / h5_file
            if h5_path.exists():
                model = DeepSPModel(model_type)
                load_keras_weights(model, str(h5_path))
                model.eval()
                self.models[model_type] = model
                print(f"  Loaded {model_type} model")
            else:
                print(f"  Warning: {model_type} model not found at {h5_path}")

    def one_hot_encode(self, sequence: str) -> np.ndarray:
        """One-hot encode aligned sequence (272 positions, 21 amino acids + gap)."""
        aa_to_idx = {
            'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5, 'H': 6, 'I': 7,
            'K': 8, 'L': 9, 'M': 10, 'N': 11, 'P': 12, 'Q': 13, 'R': 14,
            'S': 15, 'T': 16, 'V': 17, 'W': 18, 'Y': 19, '-': 20
        }

        x = np.zeros((len(sequence), 21), dtype=np.float32)
        for i, aa in enumerate(sequence):
            idx = aa_to_idx.get(aa, 20)  # Unknown -> gap
            x[i, idx] = 1.0

        return x

    @torch.no_grad()
    def predict(self, aligned_sequences: list[str]) -> np.ndarray:
        """
        Predict 30 DeepSP descriptors for aligned sequences.

        Returns: (N, 30) array with SAP_pos(10), SCM_neg(10), SCM_pos(10)
        """
        # One-hot encode
        X = np.array([self.one_hot_encode(seq) for seq in aligned_sequences])
        X_tensor = torch.from_numpy(X)

        results = []
        for model_type in ['SAPpos', 'SCMneg', 'SCMpos']:
            if model_type in self.models:
                preds = self.models[model_type](X_tensor).numpy()
                results.append(preds)
            else:
                results.append(np.zeros((len(X), 10)))

        return np.hstack(results)

    @property
    def feature_names(self) -> list[str]:
        """Return names of all 30 features."""
        names = []
        regions = ['CDRH1', 'CDRH2', 'CDRH3', 'CDRL1', 'CDRL2', 'CDRL3', 'CDR', 'Hv', 'Lv', 'Fv']
        for prefix in ['SAP_pos', 'SCM_neg', 'SCM_pos']:
            for region in regions:
                names.append(f'{prefix}_{region}')
        return names


# =============================================================================
# Test
# =============================================================================

def test_model():
    """Test that models load and predict correctly."""
    model_dir = Path("/tmp/DeepSP/DeepSP_models")

    if not model_dir.exists():
        print("Model directory not found. Please clone DeepSP first:")
        print("  git clone https://github.com/Lailabcode/DeepSP.git /tmp/DeepSP")
        return False

    print("Loading DeepSP models...")
    predictor = DeepSPPredictor(str(model_dir))

    # Test with dummy sequence (272 positions)
    test_seq = "A" * 145 + "-" * 127  # VH (145) + VL (127) = 272
    print(f"\nTest sequence length: {len(test_seq)}")

    preds = predictor.predict([test_seq])
    print(f"Predictions shape: {preds.shape}")
    print(f"Feature names: {predictor.feature_names[:5]}...")
    print(f"Sample predictions: {preds[0][:5]}")

    return True


if __name__ == "__main__":
    success = test_model()
    sys.exit(0 if success else 1)
