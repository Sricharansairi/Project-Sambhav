"""
llm/comparative_inference.py — Section 8.7 Comparative Inference Mode
Full head-to-head comparison across unlimited scenarios.
Output matrix: scenarios as columns, outcomes as rows, probabilities in cells.
"""

import json, re, logging
from typing import List
from llm.router import route

logger = logging.getLogger(__name__)


def compare_scenarios(
    domain: str,
    scenarios: List[dict],
    outcomes: List[str] = None,
    question: str = None,
) -> dict:
    """
    Section 8.7 — Compare multiple scenarios side by side.
    Each scenario is a dict with a 'label' key + parameter keys.

    Args:
        domain: prediction domain
        scenarios: list of dicts, each with 'label' + parameters
        outcomes: list of outcome labels to compare across
        question: base question being compared

    Returns:
        dict with matrix, winners per outcome, recommendation
    """
    if len(scenarios) < 2:
        return {"error": "Need at least 2 scenarios to compare"}

    question  = question or f"Compare {len(scenarios)} scenarios"
    outcomes  = outcomes or ["Primary outcome", "Risk level", "Success probability"]

    # Build scenario descriptions
    scenario_blocks = []
    for i, sc in enumerate(scenarios):
        label  = sc.get("label", f"Scenario {i+1}")
        params = {k: v for k, v in sc.items() if k != "label"}
        param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
        scenario_blocks.append(f"Scenario {i+1} — {label}: {param_str}")
    scenarios_text = "\n".join(scenario_blocks)
    outcomes_text  = "\n".join([f"  - {o}" for o in outcomes])

    messages = [
        {"role": "system", "content": (
            "You are Project Sambhav's comparative inference engine (Section 8.7).\n"
            "Compare multiple scenarios across outcomes and produce a comparison matrix.\n\n"
            "RULES:\n"
            "- Use INTEGERS for all probabilities (0-100)\n"
            "- Be specific — cite actual parameter values in reasoning\n"
            "- winner per outcome = scenario with highest probability for that outcome\n"
            "- overall_winner = scenario with best combined score\n"
            "- Respond in JSON only, no markdown\n\n"
            "JSON format:\n"
            '{"matrix": ['
            '{"outcome": "outcome name", "probabilities": {"Scenario 1 label": 72, "Scenario 2 label": 45}, '
            '"winner": "Scenario 1 label", "margin": 27, "reasoning": "why winner leads"}'
            '], '
            '"overall_winner": "Scenario label", '
            '"overall_scores": {"Scenario 1 label": 68, "Scenario 2 label": 52}, '
            '"recommendation": "clear recommendation with reasoning", '
            '"risk_profiles": {"Scenario 1 label": "LOW|MEDIUM|HIGH", "Scenario 2 label": "LOW|MEDIUM|HIGH"}, '
            '"key_differentiator": "the single factor that separates the scenarios most"}'
        )},
        {"role": "user", "content": (
            f"Domain: {domain}\n"
            f"Question: {question}\n\n"
            f"Scenarios to compare:\n{scenarios_text}\n\n"
            f"Compare across these outcomes:\n{outcomes_text}\n\n"
            "Produce a full comparison matrix with winner per outcome and overall recommendation."
        )}
    ]

    result = route("llm_predict", messages, max_tokens=1000, temperature=0.2)
    raw    = result.get("content", "")
    raw    = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw    = re.sub(r"```(?:json)?", "", raw).strip()

    try:
        parsed = json.loads(raw[raw.find("{"):raw.rfind("}")+1])
    except Exception as e:
        logger.warning(f"comparative_inference parse error: {e}")
        parsed = _fallback_matrix(scenarios, outcomes)

    # Add metadata
    parsed["domain"]     = domain
    parsed["question"]   = question
    parsed["scenarios"]  = [sc.get("label", f"Scenario {i+1}") for i, sc in enumerate(scenarios)]
    parsed["outcomes"]   = outcomes
    parsed["provider"]   = result.get("provider_used", "unknown")
    parsed["n_scenarios"] = len(scenarios)
    parsed["n_outcomes"]  = len(outcomes)

    return parsed


def _fallback_matrix(scenarios: list, outcomes: list) -> dict:
    labels = [sc.get("label", f"Scenario {i+1}") for i, sc in enumerate(scenarios)]
    matrix = []
    for outcome in outcomes:
        probs  = {label: 50 for label in labels}
        matrix.append({
            "outcome":       outcome,
            "probabilities": probs,
            "winner":        labels[0],
            "margin":        0,
            "reasoning":     "Could not compute comparison",
        })
    return {
        "matrix":           matrix,
        "overall_winner":   labels[0],
        "overall_scores":   {label: 50 for label in labels},
        "recommendation":   "Could not generate recommendation",
        "risk_profiles":    {label: "MEDIUM" for label in labels},
        "key_differentiator": "Could not identify key differentiator",
    }


def format_comparison_table(result: dict) -> str:
    """Format comparison result as readable text table."""
    lines = []
    lines.append(f"\nCOMPARATIVE INFERENCE — {result.get('domain','').upper()}")
    lines.append("=" * 60)
    lines.append(f"Question: {result.get('question','')}")
    lines.append(f"Scenarios: {' | '.join(result.get('scenarios',[]))}")
    lines.append("")

    matrix = result.get("matrix", [])
    for row in matrix:
        lines.append(f"Outcome: {row.get('outcome','')}")
        probs = row.get("probabilities", {})
        for label, prob in probs.items():
            marker = " ← WINNER" if label == row.get("winner") else ""
            lines.append(f"  {label}: {prob}%{marker}")
        lines.append(f"  Margin: {row.get('margin',0)}pp | {row.get('reasoning','')[:80]}")
        lines.append("")

    lines.append(f"OVERALL WINNER: {result.get('overall_winner','')}")
    scores = result.get("overall_scores", {})
    for label, score in scores.items():
        lines.append(f"  {label}: {score}%")
    lines.append(f"\nKey differentiator: {result.get('key_differentiator','')}")
    lines.append(f"Recommendation: {result.get('recommendation','')}")
    lines.append("")
    risks = result.get("risk_profiles", {})
    lines.append("Risk profiles:")
    for label, risk in risks.items():
        lines.append(f"  {label}: {risk}")

    return "\n".join(lines)


if __name__ == "__main__":
    print("Comparative Inference Test\n" + "="*40)

    # HR candidate comparison
    result = compare_scenarios(
        domain="hr",
        question="Which candidate is the best hire for a senior developer role?",
        scenarios=[
            {"label": "Candidate A", "experience_years": 8, "salary_expectation": "high",
             "cultural_fit": 4, "technical_score": 92, "notice_period_weeks": 4},
            {"label": "Candidate B", "experience_years": 5, "salary_expectation": "medium",
             "cultural_fit": 5, "technical_score": 78, "notice_period_weeks": 2},
            {"label": "Candidate C", "experience_years": 10, "salary_expectation": "very_high",
             "cultural_fit": 3, "technical_score": 95, "notice_period_weeks": 8},
        ],
        outcomes=[
            "Job performance probability",
            "12-month retention probability",
            "Cultural fit probability",
            "Early ramp-up probability",
        ]
    )

    print(format_comparison_table(result))
