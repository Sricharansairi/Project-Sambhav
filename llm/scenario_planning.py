"""
llm/scenario_planning.py — Section 8.14 Scenario Planning Mode
Full branching decision tree of conditional probabilities.
Each branch represents a potential future event with probability shifts.
"""

import json, re, logging
from llm.router import route

logger = logging.getLogger(__name__)


def generate_scenario_tree(
    domain: str,
    parameters: dict,
    base_probability: float,
    question: str = None,
    depth: int = 3,
    branches_per_node: int = 2,
) -> dict:
    """
    Section 8.14 — Full branching decision tree of conditional probabilities.
    Base scenario shows current probability. Each branch = potential future event.
    Each node shows: event, probability shift, new probability, reasoning.
    """
    question  = question or "What are the possible scenarios?"
    param_str = "\n".join([f"  - {k}: {v}" for k, v in parameters.items()])
    base_pct  = round(base_probability * 100, 1)

    messages = [
        {"role": "system", "content": (
            "You are Project Sambhav's scenario planning engine (Section 8.14).\n"
            "Generate a branching decision tree of conditional probabilities.\n\n"
            "IMPORTANT: Use INTEGERS for all probabilities (0-100). NOT decimals.\n\n"
            "Each branch represents a realistic future event that could occur.\n"
            "Show how each event shifts the base probability up or down.\n\n"
            "TREE STRUCTURE RULES:\n"
            "1. Root = current situation with base probability\n"
            "2. Level 1 = 2-3 major scenarios that could happen\n"
            "3. Level 2 = for each L1 scenario, 2 sub-scenarios\n"
            "4. Level 3 = final outcomes with cumulative probability\n"
            "5. Each node must cite specific parameters in its reasoning\n"
            "6. Probability shifts must be realistic (+-5 to +-35)\n\n"
            "Respond in JSON only, no markdown:\n"
            '{"root": {"situation": "...", "probability": 64, "branches": ['
            '{"event": "...", "likelihood": "HIGH|MEDIUM|LOW", '
            '"probability_shift": 15, "new_probability": 79, "reasoning": "...", "branches": []}'
            ']},'
            '"best_case": {"path": ["event1"], "final_probability": 89, "description": "..."},'
            '"worst_case": {"path": ["event1"], "final_probability": 29, "description": "..."},'
            '"most_likely_path": {"path": ["event1"], "final_probability": 64, "description": "..."},'
            '"key_decision_point": "...", "recommended_action": "..."}'
        )},
        {"role": "user", "content": (
            f"Domain: {domain}\n"
            f"Question: {question}\n"
            f"Parameters:\n{param_str}\n\n"
            f"Base probability: {int(base_pct)}%\n"
            f"Generate a {depth}-level branching scenario tree with "
            f"{branches_per_node} branches per node."
        )}
    ]

    result = route("outcome_simulation", messages, max_tokens=1200, temperature=0.5)
    raw    = result.get("content", "")
    raw    = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw    = re.sub(r"```(?:json)?", "", raw).strip()

    try:
        parsed = json.loads(raw[raw.find("{"):raw.rfind("}")+1])
    except:
        parsed = _fallback_tree(base_probability, question)

    all_paths = _flatten_tree(parsed.get("root", {}), [], base_probability)

    return {
        "domain":              domain,
        "question":            question,
        "base_probability":    base_probability,
        "base_pct":            f"{base_pct}%",
        "tree":                parsed.get("root", {}),
        "best_case":           parsed.get("best_case", {}),
        "worst_case":          parsed.get("worst_case", {}),
        "most_likely_path":    parsed.get("most_likely_path", {}),
        "key_decision_point":  parsed.get("key_decision_point", ""),
        "recommended_action":  parsed.get("recommended_action", ""),
        "all_paths":           all_paths,
        "probability_range": {
            "min": min((p["final_probability"] for p in all_paths), default=base_pct),
            "max": max((p["final_probability"] for p in all_paths), default=base_pct),
            "base": base_pct,
        },
        "provider": result.get("provider_used", "unknown"),
    }


def _flatten_tree(node: dict, path: list, base_prob: float) -> list:
    """Flatten tree into list of all root-to-leaf paths."""
    if not node:
        return []
    paths    = []
    branches = node.get("branches", [])
    if not branches:
        return [{"path": path, "final_probability": node.get("new_probability", base_prob * 100)}]
    for branch in branches:
        new_path  = path + [branch.get("event", "")]
        sub_paths = _flatten_tree(branch, new_path, base_prob)
        if sub_paths:
            paths.extend(sub_paths)
        else:
            paths.append({
                "path": new_path,
                "final_probability": branch.get("new_probability", base_prob * 100)
            })
    return paths


def _fallback_tree(base_probability: float, question: str) -> dict:
    base_pct = round(base_probability * 100)
    return {
        "root": {
            "situation": question,
            "probability": base_pct,
            "branches": [
                {
                    "event": "Positive developments occur",
                    "likelihood": "MEDIUM",
                    "probability_shift": 15,
                    "new_probability": min(99, base_pct + 15),
                    "reasoning": "Positive signals strengthen",
                    "branches": [
                        {"event": "Momentum continues", "likelihood": "MEDIUM",
                         "probability_shift": 10, "new_probability": min(99, base_pct + 25),
                         "reasoning": "Compounding positive effect", "branches": []},
                        {"event": "Progress slows", "likelihood": "MEDIUM",
                         "probability_shift": -5, "new_probability": min(99, base_pct + 10),
                         "reasoning": "Initial boost fades", "branches": []}
                    ]
                },
                {
                    "event": "Risk factors worsen",
                    "likelihood": "MEDIUM",
                    "probability_shift": -20,
                    "new_probability": max(1, base_pct - 20),
                    "reasoning": "Negative signals compound",
                    "branches": [
                        {"event": "Intervention applied", "likelihood": "MEDIUM",
                         "probability_shift": 10, "new_probability": max(1, base_pct - 10),
                         "reasoning": "Partial recovery", "branches": []},
                        {"event": "No intervention", "likelihood": "MEDIUM",
                         "probability_shift": -15, "new_probability": max(1, base_pct - 35),
                         "reasoning": "Decline accelerates", "branches": []}
                    ]
                }
            ]
        },
        "best_case": {"path": ["Positive developments", "Momentum continues"],
                      "final_probability": min(99, base_pct + 25),
                      "description": "Best case scenario"},
        "worst_case": {"path": ["Risk factors worsen", "No intervention"],
                       "final_probability": max(1, base_pct - 35),
                       "description": "Worst case scenario"},
        "most_likely_path": {"path": ["Current trajectory"],
                             "final_probability": base_pct,
                             "description": "Most likely outcome"},
        "key_decision_point": "Early intervention before risk factors compound",
        "recommended_action": "Address primary negative signals immediately",
    }


def generate_what_if(
    domain: str,
    parameters: dict,
    base_probability: float,
    what_if_event: str,
    question: str = None,
) -> dict:
    """
    Single what-if analysis — what happens if one specific event occurs?
    Used for quick what-if questions in DETAILED and FULL BREAKDOWN modes.
    """
    question  = question or f"What if: {what_if_event}?"
    param_str = "\n".join([f"  - {k}: {v}" for k, v in parameters.items()])
    base_pct  = int(round(base_probability * 100))

    messages = [
        {"role": "system", "content": (
            "You are Project Sambhav's what-if analyzer.\n"
            "Analyze the impact of a hypothetical event on a prediction.\n\n"
            "IMPORTANT RULES:\n"
            "- Use INTEGERS for probabilities (0-100). NEVER decimals like 0.63.\n"
            "- Respond in JSON only. No markdown. No text outside JSON.\n"
            "- direction must be exactly: POSITIVE, NEGATIVE, or NEUTRAL\n"
            "- confidence must be exactly: HIGH, MEDIUM, or LOW\n\n"
            "JSON format:\n"
            '{"what_if_event": "...", "probability_before": 64, '
            '"probability_after": 81, "probability_shift": 17, '
            '"direction": "POSITIVE", "reasoning": "2-3 sentences.", '
            '"cascade_effects": ["effect1", "effect2", "effect3"], '
            '"confidence": "HIGH", "timeline": "how quickly felt"}'
        )},
        {"role": "user", "content": (
            f"Domain: {domain}\n"
            f"Question: {question}\n"
            f"Parameters:\n{param_str}\n\n"
            f"Current probability: {base_pct}%\n\n"
            f"WHAT IF: {what_if_event}\n\n"
            "How does this change the probability? Give JSON only."
        )}
    ]

    result = route("outcome_simulation", messages, max_tokens=700, temperature=0.4)
    raw    = result.get("content", "")
    raw    = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    raw    = re.sub(r"```(?:json)?", "", raw).strip()

    try:
        parsed = json.loads(raw[raw.find("{"):raw.rfind("}")+1])

        # Normalize decimals to integers if LLM ignores instructions
        for key in ["probability_before", "probability_after"]:
            if key in parsed:
                val = float(parsed[key])
                parsed[key] = round(val * 100, 1) if val <= 1.0 else round(val, 1)

        if "probability_shift" in parsed:
            s = float(parsed["probability_shift"])
            parsed["probability_shift"] = round(s * 100, 1) if abs(s) <= 1.0 else round(s, 1)

        parsed["direction"] = str(parsed.get("direction", "NEUTRAL")).upper()

    except Exception as e:
        logger.warning(f"generate_what_if parse error: {e} | raw: {raw[:200]}")
        parsed = {
            "what_if_event":      what_if_event,
            "probability_before": base_pct,
            "probability_after":  base_pct,
            "probability_shift":  0,
            "direction":          "NEUTRAL",
            "reasoning":          "Could not analyze impact.",
            "cascade_effects":    [],
            "confidence":         "LOW",
            "timeline":           "Unknown",
        }

    return {**parsed, "provider": result.get("provider_used", "unknown")}


if __name__ == "__main__":
    print("Scenario Planning Test\n" + "="*40)

    tree = generate_scenario_tree(
        domain="student",
        parameters={"studytime": 3, "absences": 6, "g1": 12, "g2": 13},
        base_probability=0.637,
        question="Will this student pass their final exam?",
    )
    print(f"Base: {tree['base_pct']}")
    print(f"Best case:  {tree['best_case'].get('final_probability')}% — {tree['best_case'].get('description')}")
    print(f"Worst case: {tree['worst_case'].get('final_probability')}% — {tree['worst_case'].get('description')}")
    print(f"Key decision: {tree['key_decision_point']}")
    print(f"Recommended: {tree['recommended_action']}")
    print(f"Range: {tree['probability_range']['min']}% — {tree['probability_range']['max']}%")

    print("\n" + "="*40)

    wi = generate_what_if(
        domain="student",
        parameters={"studytime": 3, "absences": 6, "g1": 12, "g2": 13},
        base_probability=0.637,
        what_if_event="Student increases study time from 3 to 6 hours per day",
        question="Will this student pass?",
    )
    print(f"What if: {wi['what_if_event']}")
    print(f"Before: {wi['probability_before']}% → After: {wi['probability_after']}%")
    print(f"Shift: {wi['probability_shift']:+.1f}% ({wi['direction']})")
    print(f"Reasoning: {wi['reasoning'][:120]}")
    print(f"Cascade: {wi['cascade_effects']}")
    print(f"Timeline: {wi['timeline']}")