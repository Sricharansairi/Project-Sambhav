"""
core/monte_carlo.py — Monte Carlo Simulation Engine
Section 7.3 Feature 13 — 1,000 variations with Gaussian noise
Produces 95% confidence interval and stability score.
"""

import numpy as np
import logging
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


def monte_carlo_simulate(
    predict_fn,
    parameters: dict,
    n_runs: int = 1000,
    noise_factor: float = 0.05,
) -> dict:
    results = []
    failed  = 0

    str_to_num = {
        "low": 0.2, "medium": 0.5, "high": 0.8, "very_high": 0.95,
        "none": 0.0, "yes": 1.0, "no": 0.0, "true": 1.0, "false": 0.0,
        "strong": 0.9, "weak": 0.2, "moderate": 0.5,
    }

    numeric_params     = {}
    categorical_params = {}
    for k, v in parameters.items():
        if isinstance(v, str):
            num_val = str_to_num.get(v.lower().strip())
            if num_val is not None:
                numeric_params[k] = num_val
            else:
                categorical_params[k] = v
        else:
            try:
                numeric_params[k] = float(v)
            except:
                categorical_params[k] = v

    for _ in range(n_runs):
        try:
            noisy = {**categorical_params}
            for k, v in numeric_params.items():
                noise  = np.random.normal(0, noise_factor * (abs(v) + 1e-8))
                noisy[k] = max(0, v + noise)
            prob = predict_fn(noisy)
            if prob is not None and 0 <= prob <= 1:
                results.append(prob)
        except Exception:
            failed += 1

    if not results:
        return {
            "mean": 0.5, "ci_low": 0.3, "ci_high": 0.7,
            "std": 0.2, "stability": 0.4, "ci_width": 0.4,
            "n_runs": n_runs, "n_failed": failed,
            "distribution": {"very_low": 0, "low": 0, "moderate": 100, "high": 0, "very_high": 0},
            "error": "All simulation runs failed"
        }

    results_arr = np.array(results)
    ci_low      = float(np.percentile(results_arr, 2.5))
    ci_high     = float(np.percentile(results_arr, 97.5))
    ci_width    = ci_high - ci_low
    stability   = max(0.0, 1.0 - ci_width)
    mean        = float(np.mean(results_arr))
    std         = float(np.std(results_arr))

    distribution = {
        "very_low":  int(np.sum(results_arr < 0.20) / len(results_arr) * 100),
        "low":       int(np.sum((results_arr >= 0.20) & (results_arr < 0.40)) / len(results_arr) * 100),
        "moderate":  int(np.sum((results_arr >= 0.40) & (results_arr < 0.60)) / len(results_arr) * 100),
        "high":      int(np.sum((results_arr >= 0.60) & (results_arr < 0.80)) / len(results_arr) * 100),
        "very_high": int(np.sum(results_arr >= 0.80) / len(results_arr) * 100),
    }

    return {
        "mean":         round(mean, 4),
        "ci_low":       round(ci_low, 4),
        "ci_high":      round(ci_high, 4),
        "ci_width":     round(ci_width, 4),
        "std":          round(std, 4),
        "stability":    round(stability, 4),
        "n_runs":       len(results),
        "n_failed":     failed,
        "distribution": distribution,
    }


def generate_failure_scenarios(
    domain: str, parameters: dict, probability: float, shap_values: dict
) -> list:
    """Generate top 3 failure scenarios based on negative SHAP contributors."""
    negative = sorted(
        [(k, float(v)) for k, v in shap_values.items()
         if isinstance(v, (int, float)) and float(v) < 0],
        key=lambda x: x[1]
    )[:3]

    scenarios = []
    for feature, contribution in negative:
        current_val = parameters.get(feature, "current value")
        new_prob    = max(0.05, probability + contribution * 2)
        scenarios.append({
            "trigger":     feature,
            "description": f"If {feature} deteriorates further from {current_val}",
            "new_probability": round(new_prob, 3),
            "change":      round(contribution * 100, 1),
        })

    if not scenarios:
        scenarios.append({
            "trigger":         "External factors",
            "description":     "Unexpected external changes could alter the outcome",
            "new_probability": round(max(0.05, probability - 0.15), 3),
            "change":          -15.0,
        })

    return scenarios


def generate_improvement_suggestions(
    domain: str, parameters: dict, probability: float, shap_values: dict
) -> list:
    """Generate improvement suggestions based on negative SHAP contributors."""
    negative = sorted(
        [(k, float(v)) for k, v in shap_values.items()
         if isinstance(v, (int, float)) and float(v) < 0],
        key=lambda x: x[1]
    )[:3]

    suggestions = []
    for feature, contribution in negative:
        current_val = parameters.get(feature, "current value")
        new_prob    = min(0.97, probability + abs(contribution) * 1.5)
        suggestions.append({
            "action":          f"Improve '{feature}'",
            "current":         current_val,
            "expected_gain":   f"+{abs(round(contribution * 100, 1))}%",
            "new_probability": round(new_prob, 3),
        })

    if not suggestions and probability < 0.8:
        suggestions.append({
            "action":          "Provide more parameters",
            "current":         f"{len(parameters)} parameters provided",
            "expected_gain":   "+5-15%",
            "new_probability": round(min(0.97, probability + 0.10), 3),
        })

    return suggestions


class MonteCarlo:
    """
    MonteCarlo — class interface wrapping monte_carlo_simulate().
    Provides instance-level configuration and result caching.
    """
    def __init__(self, n_runs: int = 1000, noise_factor: float = 0.05):
        self.n_runs       = n_runs
        self.noise_factor = noise_factor
        self._last_result = None

    def simulate(self, predict_fn, parameters: dict) -> dict:
        result = monte_carlo_simulate(
            predict_fn   = predict_fn,
            parameters   = parameters,
            n_runs       = self.n_runs,
            noise_factor = self.noise_factor,
        )
        self._last_result = result
        return result

    def ci_string(self) -> str:
        if not self._last_result:
            return "—"
        lo = (self._last_result.get("ci_low") or 0.0) * 100
        hi = (self._last_result.get("ci_high") or 0.0) * 100
        return f"{lo:.1f}% — {hi:.1f}%"

    def stability_label(self) -> str:
        if not self._last_result:
            return "UNKNOWN"
        s = self._last_result.get("stability", 0)
        if s >= 0.80: return "STABLE"
        if s >= 0.60: return "MODERATE"
        return "UNSTABLE"

    @property
    def result(self) -> dict:
        return self._last_result or {}

    def failure_scenarios(self, domain: str, parameters: dict,
                          probability: float, shap_values: dict) -> list:
        return generate_failure_scenarios(domain, parameters, probability, shap_values)

    def improvement_suggestions(self, domain: str, parameters: dict,
                                probability: float, shap_values: dict) -> list:
        return generate_improvement_suggestions(domain, parameters, probability, shap_values)
