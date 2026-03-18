import asyncio, logging
from llm.groq_client import call_groq

logger = logging.getLogger(__name__)

# ── Agent Prompts ─────────────────────────────────────────────
OPTIMIST_SYSTEM = """You are an optimistic analyst. Your job is to find ONLY positive signals.
Argue for the HIGHEST REASONABLE probability using only supporting evidence.
CALIBRATION: Your probability must be between 55-88%. Never output below 55 or above 88.
Be precise — not 70 or 80, but 67 or 73 based on actual signal strength.
Respond in this exact format:
PROBABILITY: <55-88, precise not rounded>
ARGUMENT: <2-3 sentences arguing for high probability with specific evidence>
EVIDENCE: <top 3 positive signals with magnitude, comma separated>"""

PESSIMIST_SYSTEM = """You are a risk analyst. Your job is to find ONLY risk factors.
Argue for the LOWEST REASONABLE probability using only negative signals.
CALIBRATION: Your probability must be between 12-45%. Never output below 12 or above 45.
Be precise — not 20 or 30, but 17 or 34 based on actual risk severity.
Respond in this exact format:
PROBABILITY: <12-45, precise not rounded>
ARGUMENT: <2-3 sentences arguing for low probability with specific risks>
EVIDENCE: <top 3 risk factors with severity, comma separated>"""

REALIST_SYSTEM = """You are a realist arbitrator. You will receive two arguments — one optimistic, one pessimistic.
Weigh both by evidence quality and produce a reconciled, calibrated probability.
CALIBRATION RULES:
- Weight evidence quality not just position (optimist/pessimist)
- Strong evidence beats weak evidence regardless of direction
- Base rate adjustment: shift toward 50% if evidence is weak
- Never just average the two — weigh by evidence strength
- Output must be precise: not 60 or 65 but 58 or 63
Respond in this exact format:
PROBABILITY: <0-100, precise>
CONFIDENCE: <HIGH|MODERATE|LOW>
REASONING: <2-3 sentences explaining exact weighting and why>
OPTIMIST_WEIGHT: <0.0-1.0>
PESSIMIST_WEIGHT: <0.0-1.0>"""

DEVILS_ADVOCATE_SYSTEM = """You are a devil's advocate. Given a dominant prediction, find the STRONGEST possible argument AGAINST it.
If the prediction is HIGH probability, argue why it could FAIL.
If the prediction is LOW probability, argue why it could SUCCEED.
Respond in this exact format:
COUNTER_PROBABILITY: <0-100>
COUNTER_ARGUMENT: <2-3 sentences with strongest counter-evidence>
COUNTER_SCORE: <0.0-1.0 how strong this counter-argument is>"""

def _parse_probability(raw: str, key: str = "PROBABILITY") -> float:
    for line in raw.split("\n"):
        if line.strip().startswith(f"{key}:"):
            try:
                return float(line.split(":")[1].strip()) / 100
            except:
                pass
    return 0.5

def _parse_field(raw: str, key: str) -> str:
    for line in raw.split("\n"):
        if line.strip().startswith(f"{key}:"):
            return line.split(":", 1)[1].strip()
    return ""

# ── Individual Agents ─────────────────────────────────────────
def run_optimist(domain: str, parameters: dict, question: str) -> dict:
    param_str = "\n".join([f"  - {k}: {v}" for k, v in parameters.items()])
    messages = [
        {"role": "system", "content": OPTIMIST_SYSTEM},
        {"role": "user",   "content": f"Domain: {domain}\nQuestion: {question}\nParameters:\n{param_str}"}
    ]
    raw = call_groq(messages, temperature=0.4, max_tokens=300)
    return {
        "agent": "Optimist",
        "probability": _parse_probability(raw),
        "argument":    _parse_field(raw, "ARGUMENT"),
        "evidence":    _parse_field(raw, "EVIDENCE"),
        "raw":         raw
    }

def run_pessimist(domain: str, parameters: dict, question: str) -> dict:
    param_str = "\n".join([f"  - {k}: {v}" for k, v in parameters.items()])
    messages = [
        {"role": "system", "content": PESSIMIST_SYSTEM},
        {"role": "user",   "content": f"Domain: {domain}\nQuestion: {question}\nParameters:\n{param_str}"}
    ]
    raw = call_groq(messages, temperature=0.4, max_tokens=300)
    return {
        "agent":       "Pessimist",
        "probability": _parse_probability(raw),
        "argument":    _parse_field(raw, "ARGUMENT"),
        "evidence":    _parse_field(raw, "EVIDENCE"),
        "raw":         raw
    }

def run_realist(domain: str, optimist: dict, pessimist: dict, question: str) -> dict:
    messages = [
        {"role": "system", "content": REALIST_SYSTEM},
        {"role": "user", "content": (
            f"Domain: {domain}\nQuestion: {question}\n\n"
            f"OPTIMIST SAYS ({optimist['probability']*100:.0f}%):\n{optimist['argument']}\n"
            f"Evidence: {optimist['evidence']}\n\n"
            f"PESSIMIST SAYS ({pessimist['probability']*100:.0f}%):\n{pessimist['argument']}\n"
            f"Evidence: {pessimist['evidence']}\n\n"
            "Reconcile and produce final calibrated probability."
        )}
    ]
    raw = call_groq(messages, temperature=0.2, max_tokens=300)
    return {
        "agent":             "Realist",
        "probability":       _parse_probability(raw),
        "confidence":        _parse_field(raw, "CONFIDENCE") or "MODERATE",
        "reasoning":         _parse_field(raw, "REASONING"),
        "optimist_weight":   _parse_field(raw, "OPTIMIST_WEIGHT"),
        "pessimist_weight":  _parse_field(raw, "PESSIMIST_WEIGHT"),
        "raw":               raw
    }

def run_devils_advocate(domain: str, parameters: dict,
                         question: str, dominant_prob: float) -> dict:
    param_str = "\n".join([f"  - {k}: {v}" for k, v in parameters.items()])
    messages = [
        {"role": "system", "content": DEVILS_ADVOCATE_SYSTEM},
        {"role": "user", "content": (
            f"Domain: {domain}\nQuestion: {question}\n"
            f"Dominant prediction: {dominant_prob*100:.0f}%\n"
            f"Parameters:\n{param_str}\n\n"
            "Find the strongest argument AGAINST this prediction."
        )}
    ]
    raw = call_groq(messages, temperature=0.5, max_tokens=250)
    counter_score_str = _parse_field(raw, "COUNTER_SCORE")
    try:
        counter_score = float(counter_score_str)
    except:
        counter_score = 0.3

    return {
        "agent":            "DevilsAdvocate",
        "counter_prob":     _parse_probability(raw, "COUNTER_PROBABILITY"),
        "counter_argument": _parse_field(raw, "COUNTER_ARGUMENT"),
        "counter_score":    counter_score,
        "raw":              raw
    }

# ── Full Debate Engine ────────────────────────────────────────
def run_debate(domain: str, parameters: dict, question: str) -> dict:
    """
    Runs all 4 agents sequentially and returns full debate result.
    Final probability = Realist's estimate, adjusted by Devil's Advocate
    if counter_score > 0.3
    """
    logger.info(f"Starting multi-agent debate for domain={domain}")

    # Step 1 — Optimist + Pessimist
    optimist  = run_optimist(domain, parameters, question)
    pessimist = run_pessimist(domain, parameters, question)
    logger.info(f"Optimist: {optimist['probability']:.2f} | Pessimist: {pessimist['probability']:.2f}")

    # Step 2 — Realist reconciles
    realist = run_realist(domain, optimist, pessimist, question)
    logger.info(f"Realist: {realist['probability']:.2f} ({realist['confidence']})")

    # Step 3 — Devil's Advocate challenges
    devil = run_devils_advocate(domain, parameters, question, realist["probability"])
    logger.info(f"Devil's Advocate counter_score: {devil['counter_score']:.2f}")

    # Step 4 — Adjust final probability if counter is strong
    final_prob = realist["probability"]
    if devil["counter_score"] > 0.3:
        adjustment = devil["counter_score"] * 0.1
        if realist["probability"] > 0.5:
            final_prob = max(0.05, realist["probability"] - adjustment)
        else:
            final_prob = min(0.95, realist["probability"] + adjustment)
        logger.info(f"Adjusted by devil's advocate: {realist['probability']:.2f} → {final_prob:.2f}")

    # Step 5 — Determine gap confidence tier
    gap = abs(optimist["probability"] - pessimist["probability"])
    if gap < 0.10:
        tier = "HIGH"
    elif gap < 0.25:
        tier = "MODERATE"
    elif gap < 0.40:
        tier = "LOW"
    else:
        tier = "CRITICAL"

    return {
        "final_probability": round(final_prob, 4),
        "confidence_tier":   tier,
        "agent_gap":         round(gap, 4),
        "optimist":          optimist,
        "pessimist":         pessimist,
        "realist":           realist,
        "devils_advocate":   devil,
        "adjusted":          final_prob != realist["probability"]
    }

if __name__ == "__main__":
    result = run_debate(
        domain="student",
        parameters={
            "study_hours": 3,
            "stress_level": "high",
            "attendance": 75,
            "past_score": 62
        },
        question="Will this student pass their final exam?"
    )
    print(f"\nFinal Probability : {result['final_probability']*100:.1f}%")
    print(f"Confidence Tier   : {result['confidence_tier']}")
    print(f"Agent Gap         : {result['agent_gap']*100:.1f}%")
    print(f"Optimist          : {result['optimist']['probability']*100:.1f}%")
    print(f"Pessimist         : {result['pessimist']['probability']*100:.1f}%")
    print(f"Realist           : {result['realist']['probability']*100:.1f}%")
    print(f"Devils Adjusted   : {result['adjusted']}")
