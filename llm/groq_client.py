import os, time, random, logging
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Load all Groq keys ────────────────────────────────────────
def _load_keys():
    keys = []
    # Check numbered keys
    for i in range(1, 15):
        k = os.getenv(f"GROQ_API_KEY_{i}")
        if k and k not in keys:
            keys.append(k)
    # Check default key if no numbered keys found
    if not keys:
        k = os.getenv("GROQ_API_KEY")
        if k: keys.append(k)
    return keys

# Pre-load keys but don't crash if empty (Section 12.4 Resilience)
GROQ_KEYS  = _load_keys()
_key_index = 0

def _get_client():
    global _key_index
    current_keys = _load_keys() if not GROQ_KEYS else GROQ_KEYS
    if not current_keys:
        raise ValueError("No Groq API keys found in .env or environment")
    key = current_keys[_key_index % len(current_keys)]
    _key_index += 1
    return Groq(api_key=key)

# ── Core call with retry + key rotation ──────────────────────
def call_groq(
    messages: list,
    temperature: float = 0.3,
    max_tokens: int = 1000,
    model: str = "llama-3.3-70b-versatile",
    retries: int = 4
) -> str:
    # Use faster model for simple probability tasks if not explicitly requested
    if model == "llama-3.3-70b-versatile" and len(messages) < 10:
        model = "llama-3.1-8b-instant"
        
    fallback_model = "deepseek-r1-distill-llama-70b"
    for attempt in range(retries):
        try:
            client = _get_client()
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=10
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            err = str(e).lower()
            wait = 2 ** attempt
            if "rate_limit" in err or "429" in err:
                logger.warning(f"Rate limit hit (attempt {attempt+1}), rotating key, waiting {wait}s")
                time.sleep(wait)
            elif "model" in err and attempt == 0:
                logger.warning(f"Model error, switching to fallback model")
                model = fallback_model
            else:
                logger.warning(f"Groq error attempt {attempt+1}: {e}, retrying in {wait}s")
                time.sleep(wait)
    raise RuntimeError(f"Groq call failed after {retries} attempts")

# ── LLM Probability Prediction ────────────────────────────────
def llm_predict(domain: str, parameters: dict, question: str = None) -> dict:
    param_str = "\n".join([f"  - {k}: {v}" for k, v in parameters.items()])
    q = question or f"What is the probability of a positive outcome in the {domain} domain?"

    # Few-shot calibration examples per domain
    few_shots = {
        "student": [
            {"role": "user",      "content": "Domain: student | Q: Will this student pass? | study_hours:1, attendance:40, stress:very high, past_score:30"},
            {"role": "assistant", "content": "PROBABILITY: 22\nCONFIDENCE: HIGH\nREASONING: Three simultaneous failures: 40% attendance, score of 30, 1hr study. All three are critical negatives far below thresholds.\nKEY_FACTORS: attendance(-20%), past_score(-15%), study_hours(-8%)"},
            {"role": "user",      "content": "Domain: student | Q: Will this student pass? | study_hours:3, attendance:70, stress:medium, past_score:55"},
            {"role": "assistant", "content": "PROBABILITY: 53\nCONFIDENCE: MODERATE\nREASONING: Mixed moderate signals. No strong positives. Past score slightly below average pulls below 65% base rate.\nKEY_FACTORS: attendance(neutral), past_score(-5%), stress(neutral)"},
            {"role": "user",      "content": "Domain: student | Q: Will this student pass? | study_hours:2, attendance:50, stress:high, past_score:42"},
            {"role": "assistant", "content": "PROBABILITY: 35\nCONFIDENCE: MODERATE\nREASONING: Below average attendance, high stress and low past score are three compounding negatives. Only 2 study hours insufficient to offset.\nKEY_FACTORS: attendance(-15%), past_score(-10%), stress(-5%)"},
            {"role": "user",      "content": "Domain: student | Q: Will this student pass? | study_hours:3, attendance:70, stress:medium, past_score:55"},
            {"role": "assistant", "content": "PROBABILITY: 53\nCONFIDENCE: MODERATE\nREASONING: Attendance at 70% is acceptable but not strong. Past score of 55 is below average. Medium stress is neutral. No signal is strongly positive — sits just below 65% base rate at 53%.\nKEY_FACTORS: attendance(-5%), past_score(-7%), stress(neutral)"},
        ],
        "hr": [
            {"role": "user",      "content": "Domain: hr | Q: Will this employee leave? | job_satisfaction:very low, overtime:yes, work_life_balance:bad"},
            {"role": "assistant", "content": "PROBABILITY: 78\nCONFIDENCE: HIGH\nREASONING: Three compounding attrition drivers far above 16% base rate.\nKEY_FACTORS: job_satisfaction(+35%), work_life_balance(+20%), overtime(+7%)"},
            {"role": "user",      "content": "Domain: hr | Q: Will this employee leave? | job_satisfaction:high, overtime:no, work_life_balance:excellent"},
            {"role": "assistant", "content": "PROBABILITY: 11\nCONFIDENCE: HIGH\nREASONING: Three strong retention signals far below 16% base rate.\nKEY_FACTORS: satisfaction(-8%), work_life_balance(-5%), overtime(-3%)"},
            {"role": "user",      "content": "Domain: hr | Q: Will this employee leave? | age:42, jobsatisfaction:4, overtime:0, worklifebalance:4, yearsatcompany:12, monthlyincome:9000, distancefromhome:3, environmentsatisfaction:4, joblevel:4"},
            {"role": "assistant", "content": "PROBABILITY: 8\nCONFIDENCE: HIGH\nREASONING: Senior employee with 12 years tenure, high satisfaction across all dimensions, no overtime, close to office. All signals point to very strong retention. Well below 16% base rate.\nKEY_FACTORS: yearsatcompany(-10%), jobsatisfaction(-6%), worklifebalance(-5%)"},
        ],
        "disease": [
            {"role": "user",      "content": "Domain: disease | Q: Does this patient have heart disease? | age:28, cholesterol:170, blood_pressure:110, chest_pain:none"},
            {"role": "assistant", "content": "PROBABILITY: 18\nCONFIDENCE: HIGH\nREASONING: Young age, normal cholesterol, healthy BP and no chest pain are all protective factors well below 55% base rate.\nKEY_FACTORS: age(-15%), cholesterol(-10%), blood_pressure(-12%)"},
        ],
        "loan": [
            {"role": "user",      "content": "Domain: loan | Q: Will this person default? | credit_score:820, income:120000, missed_payments:0, debt_to_income:0.1"},
            {"role": "assistant", "content": "PROBABILITY: 13\nCONFIDENCE: HIGH\nREASONING: Excellent credit, high income, zero missed payments and very low DTI are four strong protective factors.\nKEY_FACTORS: credit_score(-10%), missed_payments(-8%), debt_to_income(-7%)"},
            {"role": "user",      "content": "Domain: loan | Q: Will this person default? | age:22, income:15000, loan_amount:50000, loan_duration:60, credit_score:480, employment_years:0, existing_loans:4, debt_to_income:0.85, has_mortgage:0, missed_payments:6"},
            {"role": "assistant", "content": "PROBABILITY: 84\nCONFIDENCE: HIGH\nREASONING: Six simultaneous critical risk factors: sub-500 credit score, 6 missed payments, 85% DTI, zero employment, 4 existing loans, and loan amount far exceeds income capacity. All signals point to near-certain default.\nKEY_FACTORS: credit_score(+30%), missed_payments(+25%), debt_to_income(+20%)"},
        ],
    }

    system_msg = {
        "role": "system",
        "content": (
            "You are a precise probabilistic inference engine trained on real-world data. "
            "Given parameters, return a carefully CALIBRATED probability estimate. "
            "CALIBRATION RULES — you MUST follow these:\n"
            "1. Start from domain base rate, then adjust per signal\n"
            "2. Each strong signal shifts by 10-15%\n"
            "3. Each weak signal shifts by 3-5%\n"
            "4. 3+ negative signals → MUST be below 45%\n"
            "5. 3+ positive signals → MUST be above 70%\n"
            "6. Never output 50%, 60%, 70%, 80% exactly — be precise\n"
            "7. Negative signals MUST pull probability DOWN below base rate\n"
            "Follow the examples exactly — they show correct calibration.\n"
            "Always respond in this exact format:\n"
            "PROBABILITY: <number 0-100, NOT rounded>\n"
            "CONFIDENCE: <HIGH|MODERATE|LOW>\n"
            "REASONING: <2-3 sentences explaining shifts from base rate>\n"
            "KEY_FACTORS: <top 3 factors with direction and magnitude>"
        )
    }

    shots  = few_shots.get(domain, [])
    user_msg = {
        "role": "user",
        "content": (
            f"Domain: {domain} | Q: {q} | "
            + " | ".join([f"{k}:{v}" for k,v in parameters.items()])
        )
    }
    messages = [system_msg] + shots + [user_msg]

    # Using 8B for speed in probability estimation (Section 6.1)
    raw = call_groq(messages, temperature=0.1, max_tokens=150, model="llama-3.1-8b-instant")

    # Parse response
    result = {"raw": raw, "probability": None, "confidence": "MODERATE",
              "reasoning": "", "key_factors": []}
    for line in raw.split("\n"):
        if line.startswith("PROBABILITY:"):
            try:
                result["probability"] = float(line.split(":")[1].strip()) / 100
            except:
                pass
        elif line.startswith("CONFIDENCE:"):
            result["confidence"] = line.split(":")[1].strip()
        elif line.startswith("REASONING:"):
            result["reasoning"] = line.split(":", 1)[1].strip()
        elif line.startswith("KEY_FACTORS:"):
            result["key_factors"] = [f.strip() for f in line.split(":",1)[1].split(",")]

    return result


# ── Free Inference Mode ───────────────────────────────────────
def free_inference(text: str, n_outcomes: int = 5) -> dict:
    """
    Advanced Free Inference (Mode 2).
    Extracts entities, signals, domain, hypotheses, and calibrated probabilities.
    """
    messages = [
        {"role": "system", "content": (
            "You are the Sambhav Free Inference Engine (Mode 2).\n"
            "Analyze the provided text and extract structural probabilistic data.\n\n"
            "YOUR TASK:\n"
            "1. Detect the most likely domain (student, hr, disease, loan, etc.)\n"
            "2. Extract relevant entities and parameters\n"
            "3. Identify positive signals (+) and negative signals (-)\n"
            "4. Generate independent outcome hypotheses\n"
            "5. Assign calibrated probabilities (0-100) to each\n"
            "6. Compute a Reliability Index (0.0-1.0) based on info completeness\n"
            "7. List missing information that would improve accuracy\n\n"
            "Respond ONLY in valid JSON format:\n"
            "{\n"
            '  "domain": "<detected_domain>",\n'
            '  "entities": ["<entity1>", "<entity2>"],\n'
            '  "positive_signals": ["<signal1>", "<signal2>"],\n'
            '  "negative_signals": ["<signal1>", "<signal2>"],\n'
            '  "reliability_index": <0.0-1.0>,\n'
            '  "missing_info": ["<info1>", "<info2>"],\n'
            '  "outcomes": [\n'
            '    {"outcome": "<description>", "probability": <0-100>, "reasoning": "<1 sentence>"},\n'
            "    ...\n"
            "  ]\n"
            "}"
        )},
        {"role": "user", "content": f"Analyze this situation:\n\n{text}"}
    ]
    
    raw = call_groq(messages, temperature=0.2, max_tokens=1500)
    
    # Clean JSON if wrapped in markdown
    import re, json
    clean = re.sub(r"```(?:json)?", "", raw).strip()
    try:
        data = json.loads(clean[clean.find("{"):clean.rfind("}")+1])
        # Ensure probabilities are 0.0-1.0 for consistency with other modes
        for o in data.get("outcomes", []):
            if o.get("probability", 0) > 1:
                o["probability"] = o["probability"] / 100.0
        return data
    except Exception as e:
        logger.error(f"Free inference JSON parse failed: {e}")
        # Fallback to simple list if JSON fails
        return {
            "domain": "general",
            "outcomes": [],
            "error": "Failed to parse structured response"
        }

# ── Quick probability helper ──────────────────────────────────
def get_llm_probability(question: str, parameters: dict, domain: str) -> float:
    """
    Fast LLM probability generation for dual-layer prediction.
    Section 6.1 — Optimized for speed.
    """
    res = llm_predict(domain, parameters, question)
    return res.get("probability") or 0.5
def health_check() -> dict:
    try:
        resp = call_groq(
            [{"role": "user", "content": "Reply with only: OK"}],
            max_tokens=5, temperature=0
        )
        return {"status": "ok", "response": resp, "keys_loaded": len(GROQ_KEYS)}
    except Exception as e:
        return {"status": "error", "error": str(e), "keys_loaded": len(GROQ_KEYS)}

if __name__ == "__main__":
    print("Testing Groq client...")
    print(health_check())
