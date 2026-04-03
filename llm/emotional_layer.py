"""
llm/emotional_layer.py — Section 8.13 Emotional Intelligence Layer
Detects emotional charge in the USER'S QUESTION using VADER sentiment analysis patterns.
If high anxiety/urgency detected — switches to empathetic framing.
Same probabilities, same accuracy — different communication style.
"""

import re, logging
logger = logging.getLogger(__name__)

ANXIETY_MARKERS   = ["please","desperate","need","urgent","worried","scared",
                     "anxious","terrified","help me","i'm afraid","panic",
                     "really hoping","so scared","please help","begging"]
URGENCY_MARKERS   = ["asap","immediately","right now","today","tonight",
                     "by tomorrow","deadline","running out of time"]
HIGH_STAKE_WORDS  = ["job","career","life","health","marriage","family",
                     "surgery","diagnosis","loan","mortgage","court","legal"]
NEGATIVE_WORDS    = ["fail","lose","fired","rejected","denied","worst",
                     "terrible","horrible","devastated"]

def detect_emotional_charge(text: str) -> dict:
    """
    Analyze user's question for emotional markers.
    Returns emotional state assessment.
    """
    text_lower = text.lower()
    words      = text_lower.split()

    anxiety_score  = sum(1 for m in ANXIETY_MARKERS  if m in text_lower)
    urgency_score  = sum(1 for m in URGENCY_MARKERS  if m in text_lower)
    stakes_score   = sum(1 for m in HIGH_STAKE_WORDS if m in text_lower)
    negative_score = sum(1 for m in NEGATIVE_WORDS   if m in text_lower)

    # Detect caps (urgency signal)
    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    exclamation_count = text.count("!")

    # Detect first-person investment
    personal_words = ["my","i'm","i am","i have","i need","i'm afraid","my job","my health"]
    personal_score = sum(1 for p in personal_words if p in text_lower)

    # Compute overall emotional charge 0-10
    charge = (
        anxiety_score  * 2.0 +
        urgency_score  * 1.5 +
        stakes_score   * 1.0 +
        negative_score * 1.0 +
        personal_score * 0.5 +
        caps_ratio     * 3.0 +
        exclamation_count * 0.5
    )
    charge = min(10.0, round(charge, 2))

    # Determine emotional state
    if charge >= 6.0:
        state = "HIGH_ANXIETY"
    elif charge >= 3.5:
        state = "MODERATE_CONCERN"
    elif charge >= 1.5:
        state = "MILD_CONCERN"
    else:
        state = "NEUTRAL"

    return {
        "charge":           charge,
        "state":            state,
        "use_empathetic":   charge >= 3.5,
        "anxiety_detected": anxiety_score > 0,
        "urgency_detected": urgency_score > 0,
        "high_stakes":      stakes_score > 0,
        "personal_stakes":  personal_score > 0,
        "markers_found": {
            "anxiety":  [m for m in ANXIETY_MARKERS  if m in text_lower],
            "urgency":  [m for m in URGENCY_MARKERS  if m in text_lower],
            "stakes":   [m for m in HIGH_STAKE_WORDS if m in text_lower],
            "negative": [m for m in NEGATIVE_WORDS   if m in text_lower],
        }
    }


def apply_empathetic_framing(
    prediction_result: dict,
    emotional_state:   dict,
    domain:            str,
) -> dict:
    """
    Rewrites prediction output with empathetic framing.
    Same data, different communication — acknowledges stakes.
    """
    if not emotional_state.get("use_empathetic"):
        return prediction_result

    final_prob = prediction_result.get("final_probability", 0.5)
    pct        = round(final_prob * 100, 1)
    state      = emotional_state["state"]

    # Acknowledgment prefix based on emotional state
    if state == "HIGH_ANXIETY":
        prefix = "I understand this feels urgent and important to you. "
    elif state == "MODERATE_CONCERN":
        prefix = "I can see this matters a lot to you. "
    else:
        prefix = "This is an important question. "

    # Soften negative predictions
    if final_prob < 0.40:
        framing = (
            f"{prefix}The probability is {pct}%, which is below 50%. "
            "However, this is based on current information only — "
            "there are specific actions that could improve this outcome significantly."
        )
    elif final_prob < 0.60:
        framing = (
            f"{prefix}The probability is {pct}% — this is genuinely uncertain territory. "
            "Both outcomes are plausible, and small changes in the situation "
            "could meaningfully shift this in either direction."
        )
    else:
        framing = (
            f"{prefix}The probability is {pct}%, which is favorable. "
            "The signals are generally positive, though no prediction is certain."
        )

    # Domain-specific professional disclaimer
    domain_disclaimers = {
        "disease":      "Please consult a qualified medical professional for any health decisions.",
        "mental_health":"If you are in distress, please reach out to a mental health professional.",
        "loan":         "Please consult a certified financial advisor before making financial decisions.",
        "hr":           "This is a probabilistic estimate — human decisions involve many factors beyond data.",
    }
    disclaimer = domain_disclaimers.get(domain, "")

    result = dict(prediction_result)
    result["empathetic_framing"] = {
        "enabled":     True,
        "emotional_state": state,
        "message":     framing,
        "disclaimer":  disclaimer,
        "original_probability": final_prob,
    }
    return result


if __name__ == "__main__":
    print("Emotional Intelligence Layer Test\n" + "="*40)
    tests = [
        "Will my student pass?",
        "Please help me I'm desperate — will I lose my job? I need to know ASAP!!",
        "I'm so scared about my health diagnosis results, what are the chances?",
        "What is the probability this loan will be approved?",
        "I'm terrified I'll fail my final exam tomorrow please help me",
    ]
    for text in tests:
        result = detect_emotional_charge(text)
        print(f"Text: {text[:60]}")
        print(f"  State:    {result['state']} (charge={result['charge']})")
        print(f"  Empathetic: {result['use_empathetic']}")
        print(f"  Markers: {result['markers_found']}")
        print()
