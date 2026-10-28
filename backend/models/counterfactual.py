import numpy as np
from typing import Dict, Any, List


class CounterfactualSimulator:
    """
    Simulates counterfactual 'what-if' maintenance scenarios for turbofan engines.
    """
    def __init__(self):
        pass

    def simulate_early_maintenance(
        self,
        cycle_series: np.ndarray,
        anomaly_scores: np.ndarray,
        health_states: List[str],
        detection_cycle: int,
        n_cycles_earlier: int
    ) -> Dict[str, str]:
        """
        Simulates what would happen if maintenance occurred n_cycles_earlier than detection_cycle.

        Returns:
            Dict containing 'scenario' and 'result' matching the report schema.
        """
        n_cycles_earlier = max(1, int(n_cycles_earlier))
        target_intervention_cycle = max(1, detection_cycle - n_cycles_earlier)

        # Find anomaly score at intervention cycle vs detection cycle
        idx_intervene = min(len(cycle_series) - 1, max(0, target_intervention_cycle - 1))
        idx_detect = min(len(cycle_series) - 1, max(0, detection_cycle - 1))

        anomaly_at_intervene = float(anomaly_scores[idx_intervene])
        anomaly_at_detect = float(anomaly_scores[idx_detect])

        # Compute wear rate ratio
        score_diff = max(0.001, anomaly_at_detect - anomaly_at_intervene)
        wear_rate = score_diff / max(1, (detection_cycle - target_intervention_cycle))

        # Model-based RUL gain estimation
        # Mitigating degradation N cycles earlier prevents cumulative thermal & mechanical fatigue
        multiplier = 1.5 + min(1.0, wear_rate * 5.0)
        rul_gain = int(round(n_cycles_earlier * multiplier))

        original_max_cycle = int(cycle_series[-1])
        projected_max_cycle = original_max_cycle + rul_gain

        scenario_desc = (
            f"What if maintenance had been performed {n_cycles_earlier} cycles earlier "
            f"(at cycle {target_intervention_cycle} instead of cycle {detection_cycle})?"
        )

        result_desc = (
            f"[MODEL-BASED ESTIMATE - NOT GUARANTEED] Performing maintenance at cycle {target_intervention_cycle} "
            f"when anomaly score was {anomaly_at_intervene:.4f} (state: {health_states[idx_intervene]}) "
            f"prevents high-stress degradation observed prior to cycle {detection_cycle}. "
            f"Estimated RUL gain: +{rul_gain} cycles (Extending operational lifespan from {original_max_cycle} to {projected_max_cycle} cycles)."
        )

        return {
            "scenario": scenario_desc,
            "result": result_desc
        }
