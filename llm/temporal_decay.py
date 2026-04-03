"""
llm/temporal_decay.py — Section 8.6 Temporal Prediction & Probability Decay
Projects how current probability evolves over time toward a deadline (evolution chart).
Uses deadline proximity + resource trajectory + LLM domain knowledge.
"""

import math, logging
from llm.router import route
import json, re

logger = logging.getLogger(__name__)


def compute_decay_curve(
    base_probability: float,
    days_total: int,
    days_elapsed: int,
    domain: str,
    parameters: dict,
) -> dict:
    """
    Compute how probability changes over time mathematically.
    Based on deadline proximity and resource depletion curves.
    """
    if days_total <= 0:
        return {"error": "days_total must be > 0"}

    days_remaining = max(0, days_total - days_elapsed)
    progress_pct   = days_elapsed / days_total

    # Domain-specific decay patterns
    decay_patterns = {
        "student":          {"type": "bell",    "peak_at": 0.4, "decay_rate": 0.3},
        "higher_education": {"type": "sigmoid", "peak_at": 0.5, "decay_rate": 0.4},
        "hr":               {"type": "linear",  "peak_at": 0.5, "decay_rate": 0.2},
        "disease":          {"type": "linear",  "peak_at": 0.5, "decay_rate": 0.1},
        "loan":             {"type": "step",    "peak_at": 0.3, "decay_rate": 0.35},
        "mental_health":    {"type": "sigmoid", "peak_at": 0.4, "decay_rate": 0.4},
        "fitness":          {"type": "bell",    "peak_at": 0.5, "decay_rate": 0.25},
    }

    pattern = decay_patterns.get(domain, {"type": "linear", "peak_at": 0.5, "decay_rate": 0.2})

    # Generate probability at each time point
    n_points = min(days_total, 20)
    curve    = []

    for i in range(n_points + 1):
        t = i / n_points  # 0 to 1

        if pattern["type"] == "bell":
            # Rises then falls — stress peaks before deadline
            adjustment = -pattern["decay_rate"] * (t - pattern["peak_at"]) ** 2 * 4
        elif pattern["type"] == "sigmoid":
            # Slow start, fast middle, plateau
            adjustment = pattern["decay_rate"] * (1 / (1 + math.exp(-10 * (t - 0.5))) - 0.5)
        elif pattern["type"] == "step":
            # Stable then drops sharply near deadline
            if t > 0.7:
                adjustment = -pattern["decay_rate"] * (t - 0.7) * 3
            else:
                adjustment = 0
        else:
            # Linear decay as deadline approaches
            adjustment = -pattern["decay_rate"] * t * 0.5

        prob = max(0.05, min(0.97, base_probability + adjustment))
        day  = round(t * days_total)
        curve.append({
            "day":         day,
            "probability": round(prob, 4),
            "pct":         f"{(prob or 0.0)*100:.1f}%",
            "label":       f"Day {day}",
        })

    # Current position on curve
    current_idx = round(progress_pct * n_points)
    current_idx = max(0, min(current_idx, len(curve) - 1))

    # Trend
    if len(curve) > 1:
        start_prob = curve[0]["probability"]
        end_prob   = curve[-1]["probability"]
        if end_prob > start_prob + 0.05:
            trend = "IMPROVING"
        elif end_prob < start_prob - 0.05:
            trend = "DECLINING"
        else:
            trend = "STABLE"
    else:
        trend = "STABLE"

    return {
        "base_probability":  base_probability,
        "days_total":        days_total,
        "days_elapsed":      days_elapsed,
        "days_remaining":    days_remaining,
        "progress_pct":      round(progress_pct * 100, 1),
        "curve":             curve,
        "current_point":     curve[current_idx],
        "trend":             trend,
        "pattern_type":      pattern["type"],
        "domain":            domain,
    }


def generate_temporal_narrative(
    domain: str,
    parameters: dict,
    base_probability: float,
    days_total: int,
    question: str = None,
) -> dict:
    """
    LLM generates week-by-week narrative showing how probability evolves.
    Section 8.6 — probability decay with reasoning.
    """
    question  = question or f"How will the probability change over {days_total} days?"
    param_str = "\n".join([f"  - {k}: {v}" for k, v in parameters.items()])

    messages = [
        {"role": "system", "content": (
            "You are Project Sambhav's temporal prediction engine (Section 8.6).\n"
            "Given a current probability and timeline, project how it will evolve.\n\n"
            "Consider:\n"
            "1. Deadline proximity effect — probability often changes near deadlines\n"
            "2. Resource depletion — budgets, energy, time all diminish\n"
            "3. Momentum — positive trends tend to continue, negative trends accelerate\n"
            "4. Domain-specific patterns — student stress peaks before exams, etc.\n\n"
            "Respond in JSON only:\n"
            "{\n"
            '  "trajectory": "<IMPROVING|STABLE|DECLINING|VOLATILE>",\n'
            '  "week_by_week": [\n'
            '    {"period": "Week 1", "probability": <0-100>, "key_event": "<what happens>"}\n'
            "  ],\n"
            '  "critical_point": "<when probability changes most dramatically>",\n'
            '  "intervention_window": "<best time to act to improve outcome>",\n'
            '  "final_probability": <0-100>,\n'
            '  "narrative": "<2-3 sentence story of how this plays out>"\n'
            "}"
        )},
        {"role": "user", "content": (
            f"Domain: {domain}\n"
            f"Question: {question}\n"
            f"Parameters:\n{param_str}\n"
            f"Current probability: {(base_probability or 0.0)*100:.1f}%\n"
            f"Timeline: {days_total} days\n\n"
            "Generate temporal probability trajectory."
        )}
    ]

    result = route("llm_predict", messages, max_tokens=500, temperature=0.4)
    raw    = result.get("content", "")
    raw    = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    if "```" in raw:
        raw = re.sub(r"```(?:json)?", "", raw).strip()

    try:
        parsed = json.loads(raw[raw.find("{"):raw.rfind("}")+1])
    except:
        parsed = {
            "trajectory":          "STABLE",
            "week_by_week":        [],
            "critical_point":      "Near the deadline",
            "intervention_window": "Early in the timeline",
            "final_probability":   round(base_probability * 100),
            "narrative":           "Probability remains relatively stable over the timeline.",
        }

    math_curve = compute_decay_curve(
        base_probability, days_total, 0, domain, parameters
    )

    return {
        "base_probability":    base_probability,
        "days_total":          days_total,
        "trajectory":          parsed.get("trajectory", "STABLE"),
        "week_by_week":        parsed.get("week_by_week", []),
        "critical_point":      parsed.get("critical_point", ""),
        "intervention_window": parsed.get("intervention_window", ""),
        "final_probability":   parsed.get("final_probability", round(base_probability * 100)) / 100,
        "narrative":           parsed.get("narrative", ""),
        "math_curve":          math_curve["curve"],
        "trend":               math_curve["trend"],
        "provider":            result.get("provider_used", "unknown"),
    }


if __name__ == "__main__":
    print("Temporal Decay Test\n" + "="*40)

    curve = compute_decay_curve(0.65, 30, 10, "student", {"study_hours": 3})
    print(f"Domain: student | Base: 65% | 30 days | Day 10")
    print(f"Trend: {curve['trend']}")
    print(f"Days remaining: {curve['days_remaining']}")
    print("Curve sample:")
    for pt in curve["curve"][::4]:
        bar = "█" * int(pt["probability"] * 20)
        print(f"  {pt['label']:8s}: {pt['pct']:6s} {bar}")
