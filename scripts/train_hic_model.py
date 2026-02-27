#!/usr/bin/env python3
"""
Train HIC retention prediction model on Jain 2017 dataset.

This script trains a GBM model and saves it for use in the library.
Run from the ProteinScore root directory.

Usage:
    python scripts/train_hic_model.py
"""

import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from proteinscore.antibody.hic_predictor import (
    extract_all_features,
    get_feature_names,
)


def load_jain_hic_data(data_path: Path) -> tuple[list[str], list[str], list[float]]:
    """Load HIC data from Jain 2017 CSV file."""
    vh_sequences = []
    vl_sequences = []
    hic_values = []

    with open(data_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            vh = row.get('heavy', '').strip()
            vl = row.get('light', '').strip()
            hic_str = row.get('HIC Retention Time (Min)a', '')

            # Skip rows with missing data
            if not vh or not vl or not hic_str:
                continue

            try:
                hic = float(hic_str)
            except ValueError:
                continue

            vh_sequences.append(vh)
            vl_sequences.append(vl)
            hic_values.append(hic)

    return vh_sequences, vl_sequences, hic_values


def train_gbm_model(X: np.ndarray, y: np.ndarray) -> tuple:
    """Train GBM model with cross-validation matching benchmark settings."""
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold
    from sklearn.base import clone

    # Create scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    scaler_mean = scaler.mean_
    scaler_std = scaler.scale_

    # GBM with same settings as benchmark (max_depth=5, n_estimators=100)
    base_model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
    )

    # Cross-validation with same fold strategy as benchmark
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    all_preds = np.zeros(len(y))
    fold_rhos = []

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_scaled)):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model_fold = clone(base_model)
        model_fold.fit(X_train, y_train)
        preds = model_fold.predict(X_val)

        all_preds[val_idx] = preds

        # Calculate fold Spearman
        from scipy.stats import spearmanr
        fold_rho, _ = spearmanr(y_val, preds)
        fold_rhos.append(fold_rho)
        print(f"  Fold {fold_idx + 1}: ρ = {fold_rho:.3f}")

    # Overall Spearman (on concatenated CV predictions)
    from scipy.stats import spearmanr
    overall_rho, pval = spearmanr(y, all_preds)
    print(f"\nOverall CV Spearman ρ: {overall_rho:.3f} ± {np.std(fold_rhos):.3f} (p={pval:.2e})")

    # Train final model on all data
    model = clone(base_model)
    model.fit(X_scaled, y)

    return model, scaler_mean, scaler_std


def extract_linear_coefficients(X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> dict:
    """Extract linear coefficients for built-in model."""
    from sklearn.linear_model import Ridge

    # Normalize
    scaler_mean = X.mean(axis=0)
    scaler_std = X.std(axis=0)
    scaler_std[scaler_std == 0] = 1.0
    X_scaled = (X - scaler_mean) / scaler_std

    # Fit Ridge regression
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_scaled, y)

    # Get top coefficients
    coef_abs = np.abs(ridge.coef_)
    top_indices = coef_abs.argsort()[-15:][::-1]

    coefficients = {}
    feature_stats = {}

    for idx in top_indices:
        name = feature_names[idx]
        coef = ridge.coef_[idx]
        coefficients[name] = float(coef)
        feature_stats[name] = {
            'mean': float(scaler_mean[idx]),
            'std': float(scaler_std[idx]),
        }

    return {
        'coefficients': coefficients,
        'intercept': float(ridge.intercept_),
        'feature_stats': feature_stats,
    }


def main():
    print("=" * 60)
    print("Training HIC Retention Prediction Model")
    print("=" * 60)

    # Find data file
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_path = project_root / "benchmarks" / "data" / "flab" / "jain2017biophyscial_HICRT.csv"

    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        print("Please run the benchmark data download first.")
        sys.exit(1)

    # Load data
    print(f"\nLoading data from {data_path}")
    vh_seqs, vl_seqs, hic_values = load_jain_hic_data(data_path)
    print(f"Loaded {len(vh_seqs)} antibodies")

    # Extract features
    print("\nExtracting features...")
    X = np.array([
        extract_all_features(vh, vl)
        for vh, vl in zip(vh_seqs, vl_seqs)
    ])
    y = np.array(hic_values)
    print(f"Feature matrix shape: {X.shape}")

    feature_names = get_feature_names()
    print(f"Number of features: {len(feature_names)}")

    # Train GBM model
    print("\nTraining GBM model...")
    model, scaler_mean, scaler_std = train_gbm_model(X, y)

    # Feature importance
    print("\nTop 10 feature importances:")
    importances = model.feature_importances_
    indices = importances.argsort()[-10:][::-1]
    for i, idx in enumerate(indices):
        print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")

    # Create output directory
    models_dir = project_root / "src" / "proteinscore" / "antibody" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Save full sklearn model
    model_path = models_dir / "hic_gbm_model.pkl"
    print(f"\nSaving full model to {model_path}")
    with open(model_path, 'wb') as f:
        pickle.dump({
            'model': model,
            'scaler_mean': scaler_mean,
            'scaler_std': scaler_std,
            'feature_names': feature_names,
        }, f)

    # Extract and save linear coefficients for built-in model
    print("\nExtracting linear coefficients for built-in model...")
    linear_data = extract_linear_coefficients(X, y, feature_names)

    coef_path = models_dir / "hic_linear_coefficients.json"
    print(f"Saving linear coefficients to {coef_path}")
    with open(coef_path, 'w') as f:
        json.dump(linear_data, f, indent=2)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"  GBM model: {model_path}")
    print(f"  Linear coefficients: {coef_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
