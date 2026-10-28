import os
import joblib
import numpy as np
import pandas as pd

try:
    from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
except ImportError:
    from pgmpy.models import BayesianNetwork

from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
from typing import Dict, Any, Tuple, List

CAUSE_MAP = {0: "HPC Degradation", 1: "Fan Degradation"}


class RootCauseBayesianNetwork:
    """
    Bayesian Network for root-cause inference in FD003.
    FD003 Fault Modes:
      0: HPC Degradation
      1: Fan Degradation
    """
    def __init__(self):
        self.model = BayesianNetwork([
            ("root_cause", "HPC_efficiency_drop"),
            ("root_cause", "fan_degradation_signal"),
            ("root_cause", "temperature_anomaly")
        ])
        self.infer_engine = None

    def fit_from_data(self, train_df: pd.DataFrame):
        """Extracts evidence statistics from training trajectories and fits CPTs."""
        rows = []
        for eng_id, group in train_df.groupby("engine_id"):
            if len(group) < 10:
                continue

            first_5 = group.iloc[:5]
            last_5 = group.iloc[-5:]

            # HPC indicators
            hpc_temp_diff = last_5.get("sensor3", pd.Series([0])).mean() - first_5.get("sensor3", pd.Series([0])).mean()
            hpc_press_diff = first_5.get("sensor7", pd.Series([0])).mean() - last_5.get("sensor7", pd.Series([0])).mean()
            hpc_score = max(0, hpc_temp_diff) + max(0, hpc_press_diff)

            # Fan indicators
            fan_speed_diff = first_5.get("sensor12", pd.Series([0])).mean() - last_5.get("sensor12", pd.Series([0])).mean()
            bpr_diff = last_5.get("sensor15", pd.Series([0])).mean() - first_5.get("sensor15", pd.Series([0])).mean()
            fan_score = max(0, fan_speed_diff) + max(0, bpr_diff)

            # Overall temperature anomaly
            temp_diff = last_5.get("sensor4", pd.Series([0])).mean() - first_5.get("sensor4", pd.Series([0])).mean()

            root_cause = 0 if hpc_score >= fan_score else 1

            rows.append({
                "root_cause": root_cause,
                "HPC_efficiency_drop": 1 if hpc_score > 0.05 else 0,
                "fan_degradation_signal": 1 if fan_score > 0.05 else 0,
                "temperature_anomaly": 1 if temp_diff > 0.05 else 0
            })

        data = pd.DataFrame(rows).astype(int)
        self.model.fit(data, estimator=MaximumLikelihoodEstimator)
        self.infer_engine = VariableElimination(self.model)

    def diagnose(self, engine_trajectory_df: pd.DataFrame) -> Tuple[str, float, List[str], List[str]]:
        """
        Diagnoses root cause given an engine's sensor trajectory.
        Returns:
            predicted_cause: str
            confidence: float
            evidence: List[str]
            affected_components: List[str]
        """
        if self.infer_engine is None:
            self.infer_engine = VariableElimination(self.model)

        n_rows = len(engine_trajectory_df)
        start_slice = engine_trajectory_df.iloc[:min(10, n_rows)]
        end_slice = engine_trajectory_df.iloc[max(0, n_rows - 10):]

        evidence_list = []

        # HPC features
        hpc_temp_inc = end_slice.get("sensor3", pd.Series([0])).mean() - start_slice.get("sensor3", pd.Series([0])).mean()
        hpc_press_drop = start_slice.get("sensor7", pd.Series([0])).mean() - end_slice.get("sensor7", pd.Series([0])).mean()
        hpc_flag = 1 if (hpc_temp_inc > 0.02 or hpc_press_drop > 0.02) else 0

        if hpc_temp_inc > 0.02:
            evidence_list.append(f"Elevated HPC outlet temperature T30 (+{hpc_temp_inc:.3f})")
        if hpc_press_drop > 0.02:
            evidence_list.append(f"Drop in HPC pressure ratio P30 (-{hpc_press_drop:.3f})")

        # Fan features
        fan_speed_drop = start_slice.get("sensor12", pd.Series([0])).mean() - end_slice.get("sensor12", pd.Series([0])).mean()
        bpr_inc = end_slice.get("sensor15", pd.Series([0])).mean() - start_slice.get("sensor15", pd.Series([0])).mean()
        fan_flag = 1 if (fan_speed_drop > 0.02 or bpr_inc > 0.02) else 0

        if fan_speed_drop > 0.02:
            evidence_list.append(f"Reduced fan physical speed Nf (-{fan_speed_drop:.3f})")
        if bpr_inc > 0.02:
            evidence_list.append(f"Shift in bypass ratio BPR (+{bpr_inc:.3f})")

        # Overall temp
        temp_inc = end_slice.get("sensor4", pd.Series([0])).mean() - start_slice.get("sensor4", pd.Series([0])).mean()
        temp_flag = 1 if temp_inc > 0.02 else 0
        if temp_inc > 0.02:
            evidence_list.append(f"LPT exhaust temperature T50 elevation (+{temp_inc:.3f})")

        if not evidence_list:
            evidence_list.append("Baseline operational sensor fluctuations within normal tolerances")

        evidence_dict = {
            "HPC_efficiency_drop": hpc_flag,
            "fan_degradation_signal": fan_flag,
            "temperature_anomaly": temp_flag
        }

        try:
            q = self.infer_engine.query(variables=["root_cause"], evidence=evidence_dict)
            values = q.values
            best_idx = int(np.argmax(values))
            # Determine state name or mapped integer
            predicted_cause = CAUSE_MAP.get(best_idx, "HPC Degradation")
            confidence = float(values[best_idx])
        except Exception:
            predicted_cause = "HPC Degradation" if hpc_flag >= fan_flag else "Fan Degradation"
            confidence = 0.85

        if predicted_cause == "HPC Degradation":
            affected_components = ["High Pressure Compressor (HPC)", "Compressor Stator Vanes", "HPC Rotor Blades"]
        else:
            affected_components = ["Fan Assembly", "Low Pressure Compressor (LPC)", "Bypass Duct"]

        return predicted_cause, confidence, evidence_list, affected_components

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)

    def load(self, filepath: str):
        self.model = joblib.load(filepath)
        self.infer_engine = VariableElimination(self.model)
