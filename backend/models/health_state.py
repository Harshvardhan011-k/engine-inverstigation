import numpy as np
import joblib
import os
from hmmlearn import hmm
from typing import List

HEALTH_STATES = ["Normal", "Degraded", "Critical", "Failure"]


class HealthStateMapper:
    """
    Fits a Gaussian HMM on anomaly score time series and maps anomaly scores to discrete health states:
    Normal -> Degraded -> Critical -> Failure.
    """
    def __init__(self, n_components: int = 4):
        self.n_components = n_components
        self.model = hmm.GaussianHMM(
            n_components=n_components,
            covariance_type="diag",
            n_iter=100,
            random_state=42
        )
        self.state_order_map = {}
        self.thresholds = [0.03, 0.08, 0.15]

    def fit(self, anomaly_scores: np.ndarray, lengths: List[int] = None):
        """Fits HMM and orders states monotonically by mean anomaly score."""
        scores_2d = np.array(anomaly_scores).reshape(-1, 1)
        self.model.fit(scores_2d, lengths=lengths)

        means = self.model.means_.flatten()
        sorted_indices = np.argsort(means)
        self.state_order_map = {raw_idx: rank for rank, raw_idx in enumerate(sorted_indices)}

        self.thresholds = [
            float(np.percentile(anomaly_scores, 50)),
            float(np.percentile(anomaly_scores, 80)),
            float(np.percentile(anomaly_scores, 95))
        ]

    def predict_states(self, anomaly_scores: np.ndarray) -> List[str]:
        """
        Predicts discrete health states for sequence of anomaly scores.
        Enforces monotonic degradation (Normal -> Degraded -> Critical -> Failure).
        """
        scores_2d = np.array(anomaly_scores).reshape(-1, 1)
        try:
            raw_states = self.model.predict(scores_2d)
            ordered_states = [self.state_order_map.get(s, 0) for s in raw_states]
        except Exception:
            ordered_states = []
            t1, t2, t3 = self.thresholds
            for s in anomaly_scores:
                if s < t1:
                    ordered_states.append(0)
                elif s < t2:
                    ordered_states.append(1)
                elif s < t3:
                    ordered_states.append(2)
                else:
                    ordered_states.append(3)

        monotonic_states = []
        curr_max = 0
        for st in ordered_states:
            if st > curr_max:
                curr_max = st
            monotonic_states.append(HEALTH_STATES[curr_max])

        return monotonic_states

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({"model": self.model, "order_map": self.state_order_map, "thresholds": self.thresholds}, filepath)

    def load(self, filepath: str):
        data = joblib.load(filepath)
        self.model = data["model"]
        self.state_order_map = data["order_map"]
        self.thresholds = data["thresholds"]
