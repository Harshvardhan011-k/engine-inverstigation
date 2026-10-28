import os
import sys
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.data_loader import get_fd003_data
from backend.preprocessing import (
    transform_dataframe,
    create_padded_sequences_for_engine
)
from backend.models.vae import AnomalyVAE
from backend.models.health_state import HealthStateMapper
from backend.models.bayesian_network import RootCauseBayesianNetwork
from backend.models.counterfactual import CounterfactualSimulator


# Caching model instances in memory for fast repeat invocations
MODEL_CACHE = {}


def load_all_models():
    """Loads all trained models from backend/saved_models/."""
    if MODEL_CACHE:
        return MODEL_CACHE

    save_dir = os.path.join(os.path.dirname(__file__), "saved_models")

    meta_path = os.path.join(save_dir, "scaler_meta.joblib")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Scaler metadata not found at {meta_path}. Please run backend/train.py first.")

    meta = joblib.load(meta_path)
    scaler = meta["scaler"]
    feature_cols = meta["feature_cols"]

    vae_prefix = os.path.join(save_dir, "vae_model")
    vae = AnomalyVAE(seq_len=30, num_features=len(feature_cols), latent_dim=8)
    vae.load(vae_prefix)

    hmm_path = os.path.join(save_dir, "hmm_model.joblib")
    hmm_mapper = HealthStateMapper()
    hmm_mapper.load(hmm_path)

    bn_path = os.path.join(save_dir, "bn_model.joblib")
    bn = RootCauseBayesianNetwork()
    bn.load(bn_path)

    counterfactual = CounterfactualSimulator()

    MODEL_CACHE.update({
        "scaler": scaler,
        "feature_cols": feature_cols,
        "vae": vae,
        "hmm": hmm_mapper,
        "bn": bn,
        "counterfactual": counterfactual
    })

    return MODEL_CACHE


def investigate_engine(engine_id: int, n_cycles_earlier: int = 20) -> Dict[str, Any]:
    """
    Orchestrates full investigation pipeline for a target engine_id from test set.
    """
    models = load_all_models()
    scaler = models["scaler"]
    feature_cols = models["feature_cols"]
    vae = models["vae"]
    hmm = models["hmm"]
    bn = models["bn"]
    cf = models["counterfactual"]

    _, test_df, _, _, _ = get_fd003_data()

    engine_data = test_df[test_df["engine_id"] == engine_id].copy()
    if engine_data.empty:
        raise ValueError(f"Engine ID {engine_id} not found in FD003 test dataset.")

    engine_data.sort_values("cycle", inplace=True)
    scaled_engine_df = transform_dataframe(engine_data, feature_cols, scaler)

    # 1. Windowing & VAE Anomaly Scoring
    X_seqs, cycles = create_padded_sequences_for_engine(scaled_engine_df, feature_cols, seq_length=30)
    anomaly_scores = vae.predict_anomaly_score(X_seqs)

    # 2. Health State Mapping
    health_states = hmm.predict_states(anomaly_scores)

    # 3. Timeline Construction
    timeline = []
    detection_cycle = int(cycles[-1])
    first_degraded_found = False

    for c, state, score in zip(cycles, health_states, anomaly_scores):
        c_int = int(c)
        score_flt = float(score)
        timeline.append({
            "cycle": c_int,
            "state": state,
            "anomaly_score": round(score_flt, 6)
        })

        if not first_degraded_found and state in ["Degraded", "Critical", "Failure"]:
            detection_cycle = c_int
            first_degraded_found = True

    # Failure detection flag
    latest_state = health_states[-1]
    latest_score = float(anomaly_scores[-1])
    failure_detected = (latest_state in ["Degraded", "Critical", "Failure"]) or (latest_score > vae.threshold)

    # 4. Bayesian Network Root Cause Diagnosis
    root_cause, confidence, evidence, affected_components = bn.diagnose(engine_data)
    failure_type = root_cause if failure_detected else "None / Normal Degradation"

    # 5. Counterfactual Simulation
    what_if = cf.simulate_early_maintenance(
        cycle_series=cycles,
        anomaly_scores=anomaly_scores,
        health_states=health_states,
        detection_cycle=detection_cycle,
        n_cycles_earlier=n_cycles_earlier
    )

    report_dict = {
        "engine_id": int(engine_id),
        "failure_detected": bool(failure_detected),
        "failure_type": str(failure_type),
        "confidence": round(float(confidence), 4),
        "timeline": timeline,
        "root_cause": str(root_cause),
        "evidence": evidence,
        "affected_components": affected_components,
        "what_if": what_if,
        "vae_threshold": round(float(vae.threshold), 6)
    }

    return report_dict


if __name__ == "__main__":
    test_id = 1
    report = investigate_engine(test_id)
    print(f"Investigation complete for engine {test_id}:")
    print(f"Failure Detected: {report['failure_detected']}")
    print(f"Failure Type: {report['failure_type']} (Confidence: {report['confidence']})")
    print(f"Root Cause: {report['root_cause']}")
    print(f"Evidence: {report['evidence']}")
    print(f"What-If Result: {report['what_if']['result']}")
