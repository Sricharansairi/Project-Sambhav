"""
llm/outcome_simulation.py — Section 8.10 Outcome Simulation Story
For any predicted outcome, generates a week-by-week narrative showing
how that outcome would unfold — primary cause, contributing factors,
inflection point, and intervention window.
"""

import json, re, logging
from llm.router import route

logger = logging.getLogger(__name__)


def generate_outcome_story(
    domain: str,
    parameters: dict,
    outcome: str,
    probability: float,
    question: str = None,
    weeks: int = 8,
) -> dict:
    """
    Section 8.10 — Week-by-week narrative simulation of how an outcome unfolds.
    Identifies: primary cause, inflection point, intervention window.

    Args:
        domain: prediction domain
        parameters: input parameters dict
        outcome: the specific outcome to simulate (e.g. "Student fails exam")
        probability: probability of this outcome (0-1)
        question: original user question
        weeks: number of weeks to simulate (default 8)

    Returns:
        dict with narrative, week_by_week, inflection_point, intervention
    """
    question  = question or f"How would '{outcome}' unfold over time?"
    param_str = "\n".join([f"  - {k}: {v}" for k, v in parameters.items()])
    pct       = round(probability * 100, 1)

    messages = [
        {"role": "system", "content": (
            "You are Project Sambhav's outcome simulation engine (Section 8.10).\n"
            "Given a predicted outcome and parameters, generate a realistic week-by-week\n"
            "narrative showing exactly how this outcome unfolds.\n\n"
            "Your simulation must identify:\n"
            "1. PRIMARY CAUSE — the single most important driver\n"
            "2. SECONDARY FACTORS — 2-3 contributing elements\n"
            "3. INFLECTION POINT — the exact moment the outcome became likely\n"
            "4. INTERVENTION WINDOW — the last opportunity to change trajectory\n"
            "5. WEEK-BY-WEEK — what happens each week leading to the outcome\n\n"
            "Be specific and realistic — cite actual parameter values in your narrative.\n"
            "The story should feel like reading a case study, not a generic template.\n\n"
            "Respond in EXACT JSON:\n"
            "{\n"
            '  "narrative": "<2-3 sentence overall story arc>",\n'
            '  "primary_cause": "<the single biggest driver of this outcome>",\n'
            '  "secondary_factors": ["<factor 1>", "<factor 2>", "<factor 3>"],\n'
            '  "week_by_week": [\n'
            '    {"week": 1, "title": "<short title>", "event": "<what happens>", "probability_shift": "<+/-X%>"},\n'
            '    ...\n'
            '  ],\n'
            '  "inflection_point": {"week": <N>, "event": "<what happens>", "why": "<why this is the turning point>"},\n'
            '  "intervention_window": {"week": <N>, "action": "<what intervention would help>", "impact": "<how much it would change outcome>"},\n'
            '  "could_have_been_prevented": <true|false>,\n'
            '  "prevention_action": "<the single action that would have changed everything>"\n'
            "}"
        )},
        {"role": "user", "content": (
            f"Domain: {domain}\n"
            f"Question: {question}\n"
            f"Parameters:\n{param_str}\n\n"
            f"Outcome to simulate: {outcome}\n"
            f"Probability: {pct}%\n"
            f"Simulate over {weeks} weeks.\n\n"
            "Generate the week-by-week story of how this outcome unfolds."
        )}
    ]

    result = route("outcome_simulation", messages, max_tokens=900, temperature=0.7)
    raw    = result.get("content", "")
    raw    = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    if "```" in raw:
        raw = re.sub(r"```(?:json)?", "", raw).strip()

    try:
        parsed = json.loads(raw[raw.find("{"):raw.rfind("}")+1])
    except:
        parsed = {
            "narrative":             f"The outcome '{outcome}' unfolds over {weeks} weeks driven by the current parameter signals.",
            "primary_cause":         "Combination of negative parameter signals",
            "secondary_factors":     ["Parameter imbalance", "Insufficient positive signals", "Time pressure"],
            "week_by_week":          [{"week": i+1, "title": f"Week {i+1}", "event": "Situation develops", "probability_shift": "0%"} for i in range(min(weeks, 4))],
            "inflection_point":      {"week": weeks//2, "event": "Point of no return reached", "why": "Negative signals compound"},
            "intervention_window":   {"week": weeks//4, "action": "Address primary cause early", "impact": "Could reduce probability by 20-30%"},
            "could_have_been_prevented": True,
            "prevention_action":     "Early intervention on primary negative signals",
        }

    return {
        "outcome":                   outcome,
        "probability":               probability,
        "probability_pct":           f"{pct}%",
        "domain":                    domain,
        "weeks_simulated":           weeks,
        "narrative":                 parsed.get("narrative", ""),
        "primary_cause":             parsed.get("primary_cause", ""),
        "secondary_factors":         parsed.get("secondary_factors", []),
        "week_by_week":              parsed.get("week_by_week", []),
        "inflection_point":          parsed.get("inflection_point", {}),
        "intervention_window":       parsed.get("intervention_window", {}),
        "could_have_been_prevented": parsed.get("could_have_been_prevented", True),
        "prevention_action":         parsed.get("prevention_action", ""),
        "provider":                  result.get("provider_used", "unknown"),
    }


def simulate_both_outcomes(
    domain: str,
    parameters: dict,
    question: str,
    final_probability: float,
    positive_outcome: str = None,
    negative_outcome: str = None,
    weeks: int = 6,
) -> dict:
    """
    Simulate BOTH the dominant and minority outcome stories side by side.
    Used in DETAILED and FULL BREAKDOWN modes.
    """
    pos_label = positive_outcome or "Positive outcome occurs"
    neg_label = negative_outcome or "Negative outcome occurs"

    pos_story = generate_outcome_story(
        domain, parameters, pos_label,
        final_probability, question, weeks
    )
    neg_story = generate_outcome_story(
        domain, parameters, neg_label,
        1.0 - final_probability, question, weeks
    )

    return {
        "dominant_story":  pos_story if final_probability >= 0.5 else neg_story,
        "minority_story":  neg_story if final_probability >= 0.5 else pos_story,
        "domain":          domain,
        "question":        question,
        "weeks_simulated": weeks,
    }


if __name__ == "__main__":
    print("Outcome Simulation Test\n" + "="*40)
    result = generate_outcome_story(
        domain="student",
        parameters={"studytime": 2, "absences": 8, "g1": 10, "g2": 11},
        outcome="Student fails the final exam",
        probability=0.62,
        question="Will this student pass their final exam?",
        weeks=6,
    )
    print(f"Outcome: {result['outcome']} ({result['probability_pct']})")
    print(f"Primary cause: {result['primary_cause']}")
    print(f"Secondary: {result['secondary_factors']}")
    print(f"\nNarrative: {result['narrative']}")
    print(f"\nWeek by week:")
    for w in result["week_by_week"]:
        print(f"  Week {w.get('week','?')}: {w.get('title','')}")
        print(f"    {w.get('event','')}")
        print(f"    Probability shift: {w.get('probability_shift','')}")
    print(f"\nInflection point: Week {result['inflection_point'].get('week','?')}")
    print(f"  {result['inflection_point'].get('event','')}")
    print(f"\nIntervention window: Week {result['intervention_window'].get('week','?')}")
    print(f"  Action: {result['intervention_window'].get('action','')}")
    print(f"  Impact: {result['intervention_window'].get('impact','')}")
    print(f"\nCould have been prevented: {result['could_have_been_prevented']}")
    print(f"Prevention action: {result['prevention_action']}")
