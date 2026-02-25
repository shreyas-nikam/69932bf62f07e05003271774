#!/usr/bin/env python3
"""
Simple test script to verify model caching functionality.
Run this script twice to see the caching in action.
"""

from source import (
    load_and_preprocess_data,
    train_and_evaluate_xgboost,
    train_cvae,
    apply_smote_and_train_xgboost
)
import time
import sys
sys.path.insert(0, '/home/user1/QuCreate/QuLabs/69932bf62f07e05003271774')


def test_model_caching():
    """Test that models are cached and loaded correctly."""

    print("=" * 70)
    print("MODEL CACHING TEST")
    print("=" * 70)

    # Load data (smaller dataset for quick testing)
    print("\n1. Loading and preprocessing data...")
    start_time = time.time()
    X_train, X_test, y_train, y_test, defaults_train, non_defaults_train, scaler = \
        load_and_preprocess_data(n_samples=5000, random_state=42)
    data_load_time = time.time() - start_time
    print(f"   Data loaded in {data_load_time:.2f} seconds")

    # Test XGBoost caching
    print("\n2. Testing XGBoost model caching...")
    start_time = time.time()
    model_base, y_prob_base, auc_base, ap_base = train_and_evaluate_xgboost(
        X_train, y_train, X_test, y_test, "Baseline Test"
    )
    xgb_time = time.time() - start_time
    print(f"   XGBoost completed in {xgb_time:.2f} seconds")
    print(f"   (First run trains model, subsequent runs load from cache)")

    # Test SMOTE caching
    print("\n3. Testing SMOTE data caching...")
    start_time = time.time()
    model_smote, y_prob_smote, auc_smote, ap_smote, X_smote, y_smote = \
        apply_smote_and_train_xgboost(X_train, y_train, X_test, y_test)
    smote_time = time.time() - start_time
    print(f"   SMOTE completed in {smote_time:.2f} seconds")
    print(f"   (First run generates data + trains, subsequent runs load from cache)")

    # Test CVAE caching
    print("\n4. Testing CVAE model caching...")
    start_time = time.time()
    cvae_model, encoder, decoder, latent_dim = train_cvae(
        X_train, y_train, latent_dim=8, epochs=20, batch_size=64
    )
    cvae_time = time.time() - start_time
    print(f"   CVAE completed in {cvae_time:.2f} seconds")
    print(f"   (First run trains model, subsequent runs load from cache)")

    # Summary
    print("\n" + "=" * 70)
    print("TEST COMPLETE - Summary:")
    print("=" * 70)
    print(
        f"Total time: {data_load_time + xgb_time + smote_time + cvae_time:.2f} seconds")
    print("\nRun this script again to see cached model loading (much faster!)")
    print("\nTo clear cache and force retraining:")
    print("  rm -rf model_cache/")
    print("=" * 70)


if __name__ == "__main__":
    test_model_caching()
