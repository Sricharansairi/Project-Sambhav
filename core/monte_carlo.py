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
    """
    Runs n_runs variations of the same prediction with Gaussian noise.
    Returns 95% CI, stability score, and distribution stats.

    Args:
        predict_fn: callable that takes parameters dict and returns float probability
        parameters: original parameter dict
        n_runs: number of simulation runs (default 1000)
        noise_factor: std dev as fraction of parameter range (default 0.05)
    """
    results = []
    failed  = 0

    str_to_num = {
        "low": 0.2, "medium": 0.5, "high": 0.8, "very_high": 0.95,
        "none": 0.0, "yes": 1.0, "no": 0.0, "true": 1.0, "false": 0.0,
        "strong": 0.9, "weak": 0.2, "moderate": 0.5,
    }

    # Convert all params to numeric for noise injection
    numeric_params = {}
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
                noise = np.random.normal(0, noise_factor * (abs(v) + 1e-8))
                noisy[k] = max(0, v + noise)
            prob = predict_fn(noisy)
            if prob is not None and 0 <= prob <= 1:
                results.append(prob)
        except Exception:
            failed += 1

    if not results:
        return {
            "mean": 0.5, "ci_low": 0.3, "ci_high": 0.7,
            "std": 0.2, "stability": 0.4,
            "n_runs": n_runs, "n_failed": failed,
            "distribution": {"very_low": 0, "low": 0, "moderate": 100, "high": 0, "very_high": 0},
            "error": "All simulation runs failed"
        }

    results_arr = np.array(results)
    ci_low      = float(np.percentile(results_arr, 2.5))
    ci_high     = float(np.percentile(results_arr, 97.5))
    ci_width    = ci_high - ci_low
    stability   = max(0.0, 1.0 - ci_width)

    # Distribution buckets
    distribution = {
        "very_low":  int(np.sum(results_arr < 0.20) / len(results) * 100),
        "low":       int(np.sum((results_arr >= 0.20) & (results_arr < 0.40)) / len(results) * 100),
        "moderate":  int(np.sum((results_arr >= 0.40) & (results_arr < 0.60)) / len(results) * 100),
        "high":      int(np.sum((results_arr >= 0.60) & (results_arr < 0.80)) / len(results) * 100),
        "very_high": int(np.sum(results_arr >= 0.80) / len(results) * 100),
    }

    # Confidence tier from CI width
    if ci_width < 0.10:   ci_tier = "TIGHT"
    elif ci_width < 0.20: ci_tier = "MODERATE"
    elif ci_width < 0.30: ci_tier = "WIDE"
    else:                 ci_tier = "VERY_WIDE"

    logger.info(f"Monte Carlo: mean={np.mean(results_arr):.3f} "
                f"CI=[{ci_low:.3f},{ci_high:.3f}] "
                f"stability={stability:.3f} runs={len(results)}/{n_runs}")

    return {
        "mean":         round(float(np.mean(results_arr)), 4),
        "median":       round(float(np.median(results_arr)), 4),
        "ci_low":       round(ci_low, 4),
        "ci_high":      round(ci_high, 4),
        "ci_width":     round(ci_width, 4),
        "ci_tier":      ci_tier,
        "std":          round(float(np.std(results_arr)), 4),
        "stability":    round(stability, 4),
        "n_runs":       n_runs,
        "n_successful": len(results),
        "n_failed":     failed,
        "distribution": distribution,
    }


def generate_failure_scenarios(
    domain: str,
    parameters: dict,
    final_probability: float,
    shap_values: dict,
) -> list:
    """
    Section 8.4 — Failure scenario generation.
    Generates top 3 failure scenarios based on SHAP values and domain knowledge.
    """
    scenarios = []

    # Get top negative SHAP contributors
    negative_shap = sorted(
        [(k, float(v)) for k, v in shap_values.items() if isinstance(v, (int, float)) and float(v) < 0],
        key=lambda x: x[1]
    )[:3]

    # Get top positive SHAP contributors (these become failure triggers)
    positive_shap = sorted(
        [(k, float(v)) for k, v in shap_values.items() if isinstance(v, (int, float)) and float(v) > 0],
        key=lambda x: x[1], reverse=True
    )[:3]

    # Domain-specific failure scenarios
    domain_scenarios = {
        "student": [
            {
                "id": "F-01",
                "name": "Academic Burnout",
                "trigger": "Stress increases to extreme while study hours drop below 1",
                "probability_drop": round(min(0.45, final_probability * 0.6), 3),
                "trigger_likelihood": "MODERATE (28%)",
                "cascade": "High stress → poor sleep → reduced retention → exam failure"
            },
            {
                "id": "F-02",
                "name": "Attendance Collapse",
                "trigger": "Attendance drops below 40% in final month",
                "probability_drop": round(min(0.35, final_probability * 0.5), 3),
                "trigger_likelihood": "LOW (15%)",
                "cascade": "Missed classes → knowledge gaps → exam underperformance"
            },
            {
                "id": "F-03",
                "name": "Exam Day Underperformance",
                "trigger": "Sleep hours drop to under 4 before exam week",
                "probability_drop": round(min(0.25, final_probability * 0.35), 3),
                "trigger_likelihood": "MODERATE (32%)",
                "cascade": "Sleep deprivation → cognitive impairment → below-average scores"
            },
        ],
        "hr": [
            {
                "id": "F-01",
                "name": "Compensation Dissatisfaction",
                "trigger": "Competitor offers 20%+ salary increase",
                "probability_drop": round(min(0.40, final_probability * 0.55), 3),
                "trigger_likelihood": "MODERATE (35%)",
                "cascade": "Market offer → salary comparison → resignation decision"
            },
            {
                "id": "F-02",
                "name": "Manager Relationship Breakdown",
                "trigger": "Manager relationship deteriorates to poor",
                "probability_drop": round(min(0.30, final_probability * 0.45), 3),
                "trigger_likelihood": "LOW (20%)",
                "cascade": "Poor management → disengagement → active job search"
            },
            {
                "id": "F-03",
                "name": "Career Growth Stagnation",
                "trigger": "Promotion denied for second consecutive year",
                "probability_drop": round(min(0.25, final_probability * 0.35), 3),
                "trigger_likelihood": "MODERATE (28%)",
                "cascade": "Stagnation → demotivation → external opportunity seeking"
            },
        ],
        "disease": [
            {
                "id": "F-01",
                "name": "Metabolic Deterioration",
                "trigger": "Glucose rises above 200 mg/dL consistently",
                "probability_drop": round(min(0.35, final_probability * 0.5), 3),
                "trigger_likelihood": "MODERATE (30%)",
                "cascade": "Hyperglycemia → insulin resistance → cardiovascular strain"
            },
        ],
        "loan": [
            {
                "id": "F-01",
                "name": "Income Shock",
                "trigger": "Job loss or 30%+ income reduction",
                "probability_drop": round(min(0.45, final_probability * 0.6), 3),
                "trigger_likelihood": "LOW (18%)",
                "cascade": "Income loss → missed payments → default cascade"
            },
        ],
        "mental_health": [
            {
                "id": "F-01",
                "name": "Support System Collapse",
                "trigger": "Social support drops to none",
                "probability_drop": round(min(0.35, final_probability * 0.5), 3),
                "trigger_likelihood": "LOW (15%)",
                "cascade": "Isolation → rumination → crisis escalation"
            },
        ],
    }

    # Get domain scenarios or generate from SHAP
    base_scenarios = domain_scenarios.get(domain, [])

    if base_scenarios:
        scenarios = base_scenarios[:3]
    else:
        # Generate from top positive SHAP values (removing them = failure)
        for i, (feature, shap_val) in enumerate(positive_shap[:3]):
            prob_drop = round(min(0.40, abs(shap_val) * 3), 3)
            scenarios.append({
                "id": f"F-0{i+1}",
                "name": f"{feature.replace('_',' ').title()} Deterioration",
                "trigger": f"{feature} drops significantly from current value",
                "probability_drop": prob_drop,
                "trigger_likelihood": "MODERATE (25%)",
                "cascade": f"Reduction in {feature} → weakened positive signal → probability drop"
            })

    # Add SHAP-based severity to each scenario
    for s in scenarios:
        new_prob = max(0.03, final_probability - s["probability_drop"])
        s["resulting_probability"] = round(new_prob, 3)
        s["severity"] = (
            "CRITICAL" if s["probability_drop"] > 0.35 else
            "HIGH"     if s["probability_drop"] > 0.20 else
            "MODERATE"
        )

    return scenarios


def generate_improvement_suggestions(
    domain: str,
    parameters: dict,
    final_probability: float,
    shap_values: dict,
) -> list:
    """
    Section 8.2 — Improvement suggestions powered by SHAP.
    Shows which parameter changes would raise probability most.
    """
    suggestions = []

    # Sort by SHAP impact — negative contributors can be improved
    negative_shap = sorted(
        [(k, float(v)) for k, v in shap_values.items() if isinstance(v, (int, float)) and float(v) < 0],
        key=lambda x: x[1]
    )

    domain_suggestions = {
        "student": {
            "Absences":         ("Reduce absences to below 5", 0.12),
            "StudyTimeWeekly":  ("Increase study time to 8+ hours/week", 0.15),
            "GPA":              ("Focus on GPA improvement through tutoring", 0.18),
            "math score":       ("Practice math problem sets daily", 0.10),
            "Tutoring":         ("Enroll in tutoring sessions", 0.08),
        },
        "hr": {
            "JobSatisfaction":  ("Address job satisfaction through role discussion", 0.20),
            "WorkLifeBalance":  ("Implement flexible work arrangements", 0.15),
            "OverTime":         ("Reduce mandatory overtime", 0.12),
            "MonthlyIncome":    ("Review compensation against market rates", 0.18),
            "YearsAtCompany":   ("Provide retention incentives", 0.10),
        },
        "disease": {
            "Glucose":          ("Follow prescribed diet to control glucose", 0.20),
            "BMI":              ("Achieve healthy BMI through diet and exercise", 0.15),
            "BloodPressure":    ("Monitor and control blood pressure", 0.18),
            "smoking_history":  ("Smoking cessation program", 0.12),
        },
        "mental_health": {
            "sleep_hours":      ("Prioritize 7-8 hours of sleep nightly", 0.20),
            "work_hours":       ("Limit work to 45 hours per week", 0.18),
            "stress_level":     ("Implement stress management techniques", 0.15),
            "social_support":   ("Build social support network", 0.20),
        },
    }

    domain_sugg = domain_suggestions.get(domain, {})

    # Reverse param map so SHAP col names match suggestion keys
    try:
        from core.predictor import DOMAIN_PARAM_MAP
        reverse_map = {v: k for k, v in DOMAIN_PARAM_MAP.get(domain, {}).items()}
    except:
        reverse_map = {}

    for feature, shap_val in negative_shap[:5]:
        lookup_key = reverse_map.get(feature, feature)
        if feature in domain_sugg or lookup_key in domain_sugg:
            action, gain = domain_sugg.get(feature) or domain_sugg.get(lookup_key)
            new_prob = min(0.97, final_probability + gain)
            suggestions.append({
                "feature": feature,
                "current_shap": round(shap_val, 4),
                "action": action,
                "projected_gain": round(gain, 4),
                "new_probability": round(new_prob, 4),
                "feasibility": "HIGH",
            })
    # Fill remaining from SHAP if needed
    if len(suggestions) < 3:
        for feature, shap_val in negative_shap:
            if feature not in [s["feature"] for s in suggestions]:
                gain = min(0.15, abs(shap_val) * 2)
                new_prob = min(0.97, final_probability + gain)
                suggestions.append({
                    "feature":        feature,
                    "current_shap":   round(shap_val, 4),
                    "action":         f"Improve {feature.replace('_',' ')}",
                    "projected_gain": round(gain, 4),
                    "new_probability": round(new_prob, 4),
                    "feasibility":    "MODERATE",
                })
            if len(suggestions) >= 5:
                break

    return sorted(suggestions, key=lambda x: x["projected_gain"], reverse=True)[:5]


if __name__ == "__main__":
    print("Monte Carlo Test\n" + "=" * 40)

    def mock_predict(params):
        base = 0.65
        study = params.get("study_hours", 3) / 10
        return min(0.95, max(0.05, base + study * 0.1))

    result = monte_carlo_simulate(
        mock_predict,
        {"study_hours": 3, "attendance": 70, "stress": 0.5},
        n_runs=1000
    )
    print(f"Mean:       {result['mean']*100:.1f}%")
    print(f"CI 95%:     {result['ci_low']*100:.1f}% — {result['ci_high']*100:.1f}%")
    print(f"Stability:  {result['stability']*100:.1f}%")
    print(f"CI Tier:    {result['ci_tier']}")
    print(f"Runs:       {result['n_successful']}/{result['n_runs']}")
    print(f"Distribution: {result['distribution']}")
