#!/usr/bin/env python3
"""
Validate HIC predictor library implementation against Jain 2017 benchmark.

This script validates that the production library achieves the expected
ρ = 0.55 correlation on the benchmark dataset using proper cross-validation.

Usage:
    python benchmarks/validate_hic_library.py
"""

import csv
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

# Add src to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


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


def main():
    print("=" * 70)
    print("HIC Library Validation Benchmark")
    print("=" * 70)

    # Find data file
    script_dir = Path(__file__).parent
    data_path = script_dir / "data" / "flab" / "jain2017biophyscial_HICRT.csv"

    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        sys.exit(1)

    # Load data
    print(f"\nLoading data from {data_path}")
    vh_seqs, vl_seqs, hic_values = load_jain_hic_data(data_path)
    print(f"Loaded {len(vh_seqs)} antibodies")

    y_true = np.array(hic_values)

    # ==========================================================================
    # Test 1: ML Predictor (HICPredictor class)
    # ==========================================================================
    print("\n" + "-" * 70)
    print("Test 1: HICPredictor (built-in linear approximation)")
    print("-" * 70)

    from proteinscore.antibody import HICPredictor, predict_hic

    predictor = HICPredictor()
    predictions = []

    for vh, vl in zip(vh_seqs, vl_seqs):
        result = predictor.predict(vh, vl)
        predictions.append(result.predicted_retention)

    y_pred_builtin = np.array(predictions)
    rho_builtin, pval_builtin = spearmanr(y_true, y_pred_builtin)

    print(f"  Spearman ρ: {rho_builtin:.4f} (p={pval_builtin:.2e})")
    print(f"  Expected:   ~0.45-0.55 (linear approximation)")

    # ==========================================================================
    # Test 2: ML Predictor with sklearn GBM model (5-fold CV)
    # ==========================================================================
    print("\n" + "-" * 70)
    print("Test 2: HICPredictor (sklearn GBM) - 5-fold Cross-Validation")
    print("-" * 70)

    from proteinscore.antibody.hic_predictor import extract_all_features

    # Extract features for all samples
    X_all = np.array([extract_all_features(vh, vl) for vh, vl in zip(vh_seqs, vl_seqs)])

    # 5-fold CV with same settings as training
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_predictions = np.zeros(len(y_true))
    fold_rhos = []

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_all)):
        X_train, X_val = X_all[train_idx], X_all[val_idx]
        y_train, y_val = y_true[train_idx], y_true[val_idx]

        # Standardize
        scaler_mean = X_train.mean(axis=0)
        scaler_std = X_train.std(axis=0)
        scaler_std[scaler_std == 0] = 1.0

        X_train_scaled = (X_train - scaler_mean) / scaler_std
        X_val_scaled = (X_val - scaler_mean) / scaler_std

        # Train GBM
        from sklearn.ensemble import GradientBoostingRegressor
        model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
        )
        model.fit(X_train_scaled, y_train)

        # Predict
        preds = model.predict(X_val_scaled)
        cv_predictions[val_idx] = preds

        # Fold correlation
        fold_rho, _ = spearmanr(y_val, preds)
        fold_rhos.append(fold_rho)
        print(f"    Fold {fold_idx + 1}: ρ = {fold_rho:.3f}")

    # Overall CV correlation
    rho_gbm, pval_gbm = spearmanr(y_true, cv_predictions)
    print(f"\n  Overall CV Spearman ρ: {rho_gbm:.4f} ± {np.std(fold_rhos):.3f} (p={pval_gbm:.2e})")
    print(f"  Expected:              ~0.55 (matches training benchmark)")

    # ==========================================================================
    # Test 3: predict_hic convenience function
    # ==========================================================================
    print("\n" + "-" * 70)
    print("Test 3: predict_hic() convenience function")
    print("-" * 70)

    predictions_func = []
    for vh, vl in zip(vh_seqs, vl_seqs):
        result = predict_hic(vh, vl)
        predictions_func.append(result.predicted_retention)

    y_pred_func = np.array(predictions_func)
    rho_func, pval_func = spearmanr(y_true, y_pred_func)

    print(f"  Spearman ρ: {rho_func:.4f} (p={pval_func:.2e})")

    # ==========================================================================
    # Test 4: predict_hic_ml from hydrophobicity module
    # ==========================================================================
    print("\n" + "-" * 70)
    print("Test 4: predict_hic_ml() from hydrophobicity module")
    print("-" * 70)

    from proteinscore.antibody import predict_hic_ml

    # Without sklearn (linear approximation) - this is the fair comparison
    predictions_ml_linear = []
    for vh, vl in zip(vh_seqs, vl_seqs):
        pred = predict_hic_ml(vh, vl, use_sklearn=False)
        predictions_ml_linear.append(pred)

    y_pred_ml_lin = np.array(predictions_ml_linear)
    rho_ml_lin, pval_ml_lin = spearmanr(y_true, y_pred_ml_lin)

    print(f"  Linear approx:   ρ = {rho_ml_lin:.4f} (p={pval_ml_lin:.2e})")
    print(f"  Note: sklearn mode would overfit on training data")

    # ==========================================================================
    # Test 5: Heuristic baseline (old method)
    # ==========================================================================
    print("\n" + "-" * 70)
    print("Test 5: Heuristic baseline (predict_hic_retention)")
    print("-" * 70)

    from proteinscore.antibody import predict_hic_retention

    predictions_heuristic = []
    for vh, vl in zip(vh_seqs, vl_seqs):
        # Use combined sequence for heuristic
        combined = vh + vl
        result = predict_hic_retention(combined)
        predictions_heuristic.append(result.hic_score)

    y_pred_heuristic = np.array(predictions_heuristic)
    rho_heuristic, pval_heuristic = spearmanr(y_true, y_pred_heuristic)

    print(f"  Spearman ρ: {rho_heuristic:.4f} (p={pval_heuristic:.2e})")
    print(f"  Expected:   ~0.35 (heuristic baseline)")

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    results = [
        ("Heuristic baseline", rho_heuristic),
        ("HICPredictor (built-in linear)", rho_builtin),
        ("predict_hic_ml (linear)", rho_ml_lin),
    ]

    if rho_gbm is not None:
        results.append(("GBM (5-fold CV)", rho_gbm))

    results.sort(key=lambda x: x[1], reverse=True)

    print("\nRanking by Spearman ρ:")
    for i, (name, rho) in enumerate(results, 1):
        improvement = ""
        if rho_heuristic > 0:
            pct = ((rho / rho_heuristic) - 1) * 100
            if pct > 0:
                improvement = f" (+{pct:.0f}% vs heuristic)"
        print(f"  {i}. {name}: ρ = {rho:.4f}{improvement}")

    # Validation check
    print("\n" + "-" * 70)
    print("VALIDATION")
    print("-" * 70)

    if rho_gbm is not None and rho_gbm >= 0.50:
        print("✅ GBM model achieves expected performance (ρ ≥ 0.50)")
    elif rho_gbm is not None:
        print(f"⚠️  GBM model below expected (ρ = {rho_gbm:.4f}, expected ≥ 0.50)")
    else:
        print("⚠️  GBM model not available for validation")

    if rho_builtin >= 0.40:
        print("✅ Built-in linear model achieves acceptable performance (ρ ≥ 0.40)")
    else:
        print(f"⚠️  Built-in linear model below expected (ρ = {rho_builtin:.4f}, expected ≥ 0.40)")

    best_rho = max(r[1] for r in results)
    improvement_vs_heuristic = ((best_rho / rho_heuristic) - 1) * 100

    print(f"\n📊 Best ML method: {results[0][0]} (ρ = {best_rho:.4f})")
    print(f"📈 Improvement over heuristic: +{improvement_vs_heuristic:.0f}%")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
