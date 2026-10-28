import os
import sys
import joblib
import numpy as np
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.data_loader import get_fd003_data
from backend.preprocessing import (
    get_feature_cols,
    fit_scaler,
    transform_dataframe,
    extract_healthy_data,
    prepare_windowed_dataset
)
from backend.models.vae import AnomalyVAE
from backend.models.health_state import HealthStateMapper
from backend.models.bayesian_network import RootCauseBayesianNetwork


def run_training():
    print("=== Step 1: Loading NASA C-MAPSS FD003 Dataset ===")
    train_df, test_df, rul_df, dropped_cols, source_used = get_fd003_data()
    feature_cols = get_feature_cols(train_df)

    print(f"Dataset source: {source_used}")
    print(f"Features count: {len(feature_cols)}, Dropped zero-variance columns: {dropped_cols}")

    save_dir = os.path.join(os.path.dirname(__file__), "saved_models")
    os.makedirs(save_dir, exist_ok=True)

    print("\n=== Step 2: Fitting and Saving Feature Scaler ===")
    scaler = fit_scaler(train_df, feature_cols)
    meta_path = os.path.join(save_dir, "scaler_meta.joblib")
    joblib.dump({
        "scaler": scaler,
        "feature_cols": feature_cols,
        "dropped_cols": dropped_cols,
        "source_used": source_used
    }, meta_path)
    print(f"Saved scaler metadata to {meta_path}")

    scaled_train_df = transform_dataframe(train_df, feature_cols, scaler)

    print("\n=== Step 3: Training Variational Autoencoder (VAE) ===")
    healthy_train_df = extract_healthy_data(scaled_train_df, max_cycle=30)
    X_healthy, _, _ = prepare_windowed_dataset(healthy_train_df, feature_cols, seq_length=30)
    print(f"Healthy sequence windows shape: {X_healthy.shape}")

    vae = AnomalyVAE(seq_len=30, num_features=len(feature_cols), latent_dim=8)
    threshold = vae.fit(X_healthy, epochs=15, batch_size=64)
    print(f"VAE training complete. Learned anomaly threshold: {threshold:.6f}")

    vae_prefix = os.path.join(save_dir, "vae_model")
    vae.save(vae_prefix)
    print(f"Saved VAE model to {vae_prefix}")

    print("\n=== Step 4: Fitting Health State Mapper (HMM) ===")
    X_train, train_engine_ids, train_cycles = prepare_windowed_dataset(scaled_train_df, feature_cols, seq_length=30)
    print("Computing anomaly scores across full training trajectories...")
    train_anomaly_scores = vae.predict_anomaly_score(X_train)

    lengths = [len(g) for _, g in scaled_train_df.groupby("engine_id")]
    hmm_mapper = HealthStateMapper(n_components=4)
    hmm_mapper.fit(train_anomaly_scores, lengths=lengths)

    hmm_path = os.path.join(save_dir, "hmm_model.joblib")
    hmm_mapper.save(hmm_path)
    print(f"Saved HMM model to {hmm_path}")

    print("\n=== Step 5: Fitting Root-Cause Bayesian Network (pgmpy) ===")
    bn = RootCauseBayesianNetwork()
    bn.fit_from_data(train_df)
    bn_path = os.path.join(save_dir, "bn_model.joblib")
    bn.save(bn_path)
    print(f"Saved Bayesian Network model to {bn_path}")

    print("\n=======================================================")
    print("SUCCESS: All models trained and saved to backend/saved_models/")
    print("=======================================================")


if __name__ == "__main__":
    run_training()
