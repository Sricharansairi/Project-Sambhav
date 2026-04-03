import logging
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

# G.09: 4 severity levels (CRITICAL/WARNING/CAUTION/INFO)
SEVERITIES = ["CRITICAL", "WARNING", "CAUTION", "INFO"]

class AuditSystem:
    """
    G.01: Prediction Audit System with 3 Engines.
    Engines: Parameter (1), Prediction (2), Confidence (3).
    """
    def __init__(self):
        # G.02-G.08: ABN flags (7 total)
        self.abn_flags = {
            "ABN-001": "Contradictory Input",
            "ABN-002": "Out of Range (>3 std-devs)",
            "ABN-003": "Surprising Prediction",
            "ABN-004": "Low Confidence (<10% margin)",
            "ABN-005": "High Uncertainty (MC CI >25pp)",
            "ABN-006": "Unreliable Model - CRITICAL block",
            "ABN-007": "Missing Key Parameter"
        }

    def engine_1_parameter_auditor(self, parameters: dict) -> List[dict]:
        """G.01: Engine 1 - Parameter Auditor."""
        flags = []
        if not parameters:
            flags.append({"code": "ABN-007", "severity": "CRITICAL", "message": "Missing all key parameters"})
            
        # G.12: Adversarial Mode — physiological impossibility detection
        if parameters.get("sleep", -1) == 0 and parameters.get("study", -1) >= 10:
            flags.append({"code": "ADV-001", "severity": "CRITICAL", "message": "Physiological impossibility detected"})

        return flags

    def engine_2_prediction_auditor(self, ml_prob: float, llm_prob: float) -> List[dict]:
        """G.01: Engine 2 - Prediction Auditor."""
        flags = []
        gap = abs(ml_prob - llm_prob)
        if gap > 0.40:
            # G.07: ABN-006 blocks output entirely
            flags.append({"code": "ABN-006", "severity": "CRITICAL", "message": "Unreliable Model - Output Blocked"})
        elif gap > 0.25:
            flags.append({"code": "ABN-003", "severity": "WARNING", "message": "Surprising Prediction Gap"})
        return flags

    def engine_3_confidence_auditor(self, ci_width: float) -> List[dict]:
        """G.01/E.06: Engine 3 - Confidence Auditor (feeds from Monte Carlo CI width)."""
        flags = []
        if ci_width > 0.25:
            flags.append({"code": "ABN-005", "severity": "CAUTION", "message": "High Uncertainty (MC CI >25pp)"})
        return flags

    def run_full_audit(self, parameters: dict, ml_prob: float, llm_prob: float, ci_width: float = 0.0) -> dict:
        """Master audit orchestrator."""
        all_flags = []
        all_flags.extend(self.engine_1_parameter_auditor(parameters))
        all_flags.extend(self.engine_2_prediction_auditor(ml_prob, llm_prob))
        all_flags.extend(self.engine_3_confidence_auditor(ci_width))
        
        passed = not any(f["severity"] == "CRITICAL" for f in all_flags)
        
        return {
            "passed": passed,
            "flags": all_flags,
            "severity_counts": {s: len([f for f in all_flags if f["severity"] == s]) for s in SEVERITIES},
            # G.11: block keyword
            "block_output": not passed
        }
