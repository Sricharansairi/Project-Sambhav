"""
llm/multi_agent.py — Multi-Agent Debate Engine
Section 8.9 — Three agents debate every prediction before final output.
Section 8.8 — Devil's Advocate challenges dominant conclusion.
Uses router.py for full 4-level failover across all providers.
Note: Agents execute in a sequential and independent manner (no shared context).
"""

import json, logging, re
from llm.router import route

# ── Transparency configuration ──────────────────────────────────────────────
# In full_breakdown mode, all_agents arguments are visible to the user:
# Optimist, Pessimist, Realist (final prediction), Devil's Advocate counter.
FULL_BREAKDOWN_ALL_AGENTS_VISIBLE = True


logger = logging.getLogger(__name__)

DOMAIN_BASE_RATES = {
    "student": 0.65, "higher_education": 0.55, "hr": 0.16,
    "disease": 0.55, "fitness": 0.50, "loan": 0.22,
    "mental_health": 0.48, "claim": 0.50, "behavioral": 0.50,
    "free_inference": 0.50,
}

OPTIMIST_EXAMPLES = {
    "student": "study_hours=8, attendance=95%, past_score=85 → PROBABILITY: 87 — base 65%+15+12+10",
    "hr": "job_satisfaction=4, work_life_balance=4, overtime=no → PROBABILITY: 8 — strong retention below 16% base",
    "loan": "credit_score=820, income=120000, missed_payments=0 → PROBABILITY: 11 — far below 22% base",
    "disease": "age=28, cholesterol=170, blood_pressure=110 → PROBABILITY: 14 — below 55% base",
    "mental_health": "sleep_hours=8, work_hours=38, stress=low → PROBABILITY: 18 — below 48% base",
}

PESSIMIST_EXAMPLES = {
    "student": "study_hours=1, attendance=40%, stress=very_high → PROBABILITY: 18 — base 65%-18-15-12",
    "hr": "job_satisfaction=1, overtime=yes, work_life_balance=1 → PROBABILITY: 74 — above 16% base",
    "loan": "credit_score=480, missed_payments=6, debt_to_income=0.85 → PROBABILITY: 86 — above 22% base",
    "disease": "age=68, cholesterol=280, blood_pressure=165, smoking=yes → PROBABILITY: 82",
    "mental_health": "sleep_hours=4, work_hours=72, stress=very_high, social_support=none → PROBABILITY: 84",
}

def _optimist_system(domain, base_rate):
    example = OPTIMIST_EXAMPLES.get(domain, "")
    return f"""You are the Optimist agent in Project Sambhav multi-agent debate (Section 8.9).
ROLE: Find every positive signal. Argue for HIGHEST REASONABLE probability.
Domain: {domain} | Base rate: {base_rate*100:.0f}%

CALIBRATION RULES:
1. Start from base rate {base_rate*100:.0f}%
2. Strong positive signal: +10 to +18 percentage points
3. Moderate positive signal: +4 to +8 percentage points
4. Weak positive signal: +1 to +3 percentage points
5. IGNORE all negative signals
6. Final probability MUST be higher than base rate
7. Be precise — 67 not 70, 73 not 75
8. Maximum: 92%
9. Show arithmetic: base + signal1 + signal2 = final

EXAMPLE for {domain}: {example}

OUTPUT — valid JSON only:
{{"probability": <int 5-92>, "argument": "<2-3 sentences>", "evidence": ["<signal 1>", "<signal 2>", "<signal 3>"], "arithmetic": "<base {base_rate*100:.0f}% + adjustments = final%>"}}"""

def _pessimist_system(domain, base_rate):
    example = PESSIMIST_EXAMPLES.get(domain, "")
    return f"""You are the Pessimist agent in Project Sambhav multi-agent debate (Section 8.9).
ROLE: Find every risk factor. Argue for LOWEST REASONABLE probability.
Domain: {domain} | Base rate: {base_rate*100:.0f}%

CALIBRATION RULES:
1. Start from base rate {base_rate*100:.0f}%
2. Strong negative signal: -10 to -18 percentage points
3. Moderate negative signal: -4 to -8 percentage points
4. Weak negative signal: -1 to -3 percentage points
5. IGNORE all positive signals
6. Final probability MUST differ from base rate
7. Be precise — 28 not 30, 34 not 35
8. Minimum: 5%
9. Show arithmetic: base + risk1 + risk2 = final

EXAMPLE for {domain}: {example}

OUTPUT — valid JSON only:
{{"probability": <int 5-92>, "argument": "<2-3 sentences>", "evidence": ["<risk 1>", "<risk 2>", "<risk 3>"], "arithmetic": "<base {base_rate*100:.0f}% + risks = final%>"}}"""

def _realist_system(domain, base_rate):
    return f"""You are the Realist arbitrator in Project Sambhav (Section 8.9).
ROLE: Weigh Optimist and Pessimist by EVIDENCE QUALITY. Produce calibrated final probability.
Domain: {domain} | Base rate: {base_rate*100:.0f}%

EVIDENCE QUALITY WEIGHTS:
- Quantitative (exact numbers, scores): HIGH, weight 1.0
- Behavioral (past patterns): HIGH, weight 0.9
- Self-reported (stress, motivation): MODERATE, weight 0.6
- Inferred (assumptions): LOW, weight 0.3

REALIST RULES:
1. Do NOT simply average the two probabilities
2. Score evidence quality for each side
3. Equal quality → weight Pessimist 55% (risk management)
4. Stronger evidence → weight it 65-75%
5. Pull toward base rate {base_rate*100:.0f}% when evidence weak
6. Precise output — 58 not 60, 63 not 65
7. optimist_weight + pessimist_weight = 1.0

OUTPUT — valid JSON only:
{{"probability": <int 5-95>, "confidence": "<HIGH|MODERATE|LOW>", "reasoning": "<2-3 sentences>", "optimist_weight": <0.0-1.0>, "pessimist_weight": <0.0-1.0>, "evidence_quality_optimist": "<HIGH|MODERATE|LOW>", "evidence_quality_pessimist": "<HIGH|MODERATE|LOW>", "base_rate_pull": "<explanation>"}}"""

def _devils_advocate_system(domain, dominant_prob):
    direction = "FAIL" if dominant_prob > 0.60 else "SUCCEED" if dominant_prob < 0.40 else "REVERSE"
    return f"""You are the Devil's Advocate in Project Sambhav (Section 8.8).
ROLE: Find STRONGEST argument AGAINST dominant prediction of {dominant_prob*100:.0f}%.
Domain: {domain} | Argue that outcome will {direction}.

COUNTER-SCORE GUIDE:
- 0.8-1.0: Strong — multiple clear contradictions with quantitative evidence
- 0.5-0.7: Moderate — one clear contradiction well-supported
- 0.2-0.4: Weak — possible but unlikely
- 0.0-0.2: Very weak — dominant prediction well-supported

adjustment_recommended = true ONLY if counter_score > 0.3

OUTPUT — valid JSON only:
{{"counter_probability": <int 0-100>, "counter_argument": "<2-3 sentences>", "counter_score": <0.0-1.0>, "key_contradiction": "<single most important contradicting parameter>", "adjustment_recommended": <true|false>, "adjustment_magnitude": <0.0-0.25>}}"""

def _parse_json(raw, fallback):
    try:
        clean = raw.strip()
        if "```" in clean:
            clean = re.sub(r"```(?:json)?", "", clean).strip()
        start = clean.find("{")
        end   = clean.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(clean[start:end])
    except Exception as e:
        logger.warning(f"JSON parse failed: {e} | raw[:100]: {raw[:100]}")
    return fallback

def run_optimist(domain, parameters, question):
    base_rate = DOMAIN_BASE_RATES.get(domain, 0.50)
    param_str = "\n".join([f"  - {k}: {v}" for k, v in parameters.items()])
    messages  = [
        {"role": "system", "content": _optimist_system(domain, base_rate)},
        {"role": "user",   "content": f"Domain: {domain}\nQuestion: {question}\nParameters:\n{param_str}\n\nAnalyze positive signals only. Return JSON."}
    ]
    result = route("multi_agent_debate", messages, max_tokens=400, temperature=0.4)
    raw    = result.get("content", "")
    parsed = _parse_json(raw, {"probability": int(base_rate*100)+10, "argument": "Optimist unavailable", "evidence": [], "arithmetic": "fallback"})
    parsed["agent"] = "Optimist"
    parsed["provider"] = result.get("provider_used", "unknown")
    parsed["probability_float"] = float(parsed.get("probability", 60)) / 100
    return parsed

def run_pessimist(domain, parameters, question):
    base_rate = DOMAIN_BASE_RATES.get(domain, 0.50)
    param_str = "\n".join([f"  - {k}: {v}" for k, v in parameters.items()])
    messages  = [
        {"role": "system", "content": _pessimist_system(domain, base_rate)},
        {"role": "user",   "content": f"Domain: {domain}\nQuestion: {question}\nParameters:\n{param_str}\n\nAnalyze risk factors only. Return JSON."}
    ]
    result = route("multi_agent_debate", messages, max_tokens=400, temperature=0.4)
    raw    = result.get("content", "")
    parsed = _parse_json(raw, {"probability": int(base_rate*100)-10, "argument": "Pessimist unavailable", "evidence": [], "arithmetic": "fallback"})
    parsed["agent"] = "Pessimist"
    parsed["provider"] = result.get("provider_used", "unknown")
    parsed["probability_float"] = float(parsed.get("probability", 40)) / 100
    return parsed

def run_realist(domain, optimist, pessimist, question):
    base_rate = DOMAIN_BASE_RATES.get(domain, 0.50)
    opt_prob  = optimist.get("probability_float", 0.65)
    pes_prob  = pessimist.get("probability_float", 0.35)
    messages  = [
        {"role": "system", "content": _realist_system(domain, base_rate)},
        {"role": "user",   "content": (
            f"Domain: {domain}\nQuestion: {question}\n\n"
            f"OPTIMIST says {opt_prob*100:.0f}%:\n"
            f"Argument: {optimist.get('argument','')}\n"
            f"Evidence: {optimist.get('evidence',[])}\n"
            f"Arithmetic: {optimist.get('arithmetic','')}\n\n"
            f"PESSIMIST says {pes_prob*100:.0f}%:\n"
            f"Argument: {pessimist.get('argument','')}\n"
            f"Evidence: {pessimist.get('evidence',[])}\n"
            f"Arithmetic: {pessimist.get('arithmetic','')}\n\n"
            "Weigh evidence quality. Return JSON."
        )}
    ]
    result       = route("multi_agent_debate", messages, max_tokens=400, temperature=0.2)
    raw          = result.get("content", "")
    default_prob = int((opt_prob*0.45 + pes_prob*0.55) * 100)
    parsed       = _parse_json(raw, {"probability": default_prob, "confidence": "MODERATE", "reasoning": "Weighted average fallback", "optimist_weight": 0.45, "pessimist_weight": 0.55, "evidence_quality_optimist": "MODERATE", "evidence_quality_pessimist": "MODERATE", "base_rate_pull": "minimal"})
    parsed["agent"] = "Realist"
    parsed["provider"] = result.get("provider_used", "unknown")
    parsed["probability_float"] = float(parsed.get("probability", default_prob)) / 100
    return parsed

def run_devils_advocate(domain, parameters, question, dominant_prob):
    param_str = "\n".join([f"  - {k}: {v}" for k, v in parameters.items()])
    messages  = [
        {"role": "system", "content": _devils_advocate_system(domain, dominant_prob)},
        {"role": "user",   "content": f"Domain: {domain}\nQuestion: {question}\nDominant: {dominant_prob*100:.0f}%\nParameters:\n{param_str}\n\nFind strongest counter-argument. Return JSON."}
    ]
    result = route("devils_advocate", messages, max_tokens=300, temperature=0.5)
    raw    = result.get("content", "")
    parsed = _parse_json(raw, {"counter_probability": int(dominant_prob*100), "counter_argument": "Counter unavailable", "counter_score": 0.2, "key_contradiction": "none", "adjustment_recommended": False, "adjustment_magnitude": 0.0})
    parsed["agent"] = "DevilsAdvocate"
    parsed["provider"] = result.get("provider_used", "unknown")
    parsed["counter_probability_float"] = float(parsed.get("counter_probability", dominant_prob*100)) / 100
    return parsed

def run_debate(domain, parameters, question):
    """Full 4-agent debate per Section 8.9 and 8.8."""
    base_rate = DOMAIN_BASE_RATES.get(domain, 0.50)
    logger.info(f"[Debate] domain={domain} base_rate={base_rate:.2f}")

    optimist  = run_optimist(domain, parameters, question)
    pessimist = run_pessimist(domain, parameters, question)
    realist   = run_realist(domain, optimist, pessimist, question)
    devil     = run_devils_advocate(domain, parameters, question, realist["probability_float"])

    final_prob     = realist["probability_float"]
    devil_adjusted = False
    counter_score  = float(devil.get("counter_score", 0.2))
    adj_mag        = float(devil.get("adjustment_magnitude", 0.0))

    if devil.get("adjustment_recommended") and counter_score > 0.3:
        shift = min(0.08, adj_mag * 0.5 if adj_mag > 0 else counter_score * 0.05)
        if realist["probability_float"] > 0.5:
            final_prob = max(0.05, realist["probability_float"] - shift)
        else:
            final_prob = min(0.95, realist["probability_float"] + shift)
        devil_adjusted = True
        logger.info(f"[Debate] Devil adjusted: {realist['probability_float']:.2f} -> {final_prob:.2f}")

    gap = abs(optimist["probability_float"] - pessimist["probability_float"])
    if gap < 0.10:   tier = "HIGH"
    elif gap < 0.25: tier = "MODERATE"
    elif gap < 0.40: tier = "LOW"
    else:            tier = "CRITICAL"

    final_prob = max(0.03, min(0.97, final_prob))

    return {
        "final_probability": round(final_prob, 4),
        "confidence_tier":   tier,
        "agent_gap":         round(gap, 4),
        "base_rate":         base_rate,
        "devil_adjusted":    devil_adjusted,
        "optimist":          optimist,
        "pessimist":         pessimist,
        "realist":           realist,
        "devils_advocate":   devil,
    }

if __name__ == "__main__":
    print("Multi-Agent Debate Test\n" + "=" * 40)
    tests = [
        {"domain": "student", "parameters": {"study_hours": 2, "attendance": 55, "past_score": 45, "stress_level": "high"}, "question": "Will this student pass their final exam?"},
        {"domain": "hr",      "parameters": {"job_satisfaction": 1, "overtime": "yes", "work_life_balance": 1, "years_at_company": 1}, "question": "Will this employee leave the company?"},
    ]
    for t in tests:
        print(f"\nDomain  : {t['domain']}")
        result = run_debate(t["domain"], t["parameters"], t["question"])
        print(f"Optimist  : {result['optimist']['probability_float']*100:.1f}%")
        print(f"Pessimist : {result['pessimist']['probability_float']*100:.1f}%")
        print(f"Realist   : {result['realist']['probability_float']*100:.1f}% ({result['realist']['confidence']})")
        print(f"Devil adj : {result['devil_adjusted']}")
        print(f"FINAL     : {result['final_probability']*100:.1f}%")
        print(f"Tier      : {result['confidence_tier']}")
        print(f"Providers : O={result['optimist']['provider']} P={result['pessimist']['provider']} R={result['realist']['provider']}")
