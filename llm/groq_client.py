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
        "fitness": [
            {"role": "user",      "content": "Domain: fitness | Q: Am I obese? | weight_kg:110, height_cm:170, activity_level:0"},
            {"role": "assistant", "content": "PROBABILITY: 88.5\nCONFIDENCE: HIGH\nREASONING: Weight of 110kg at 170cm results in a BMI of 38.0 (Class II Obesity). Sedentary activity level further compounds the risk.\nKEY_FACTORS: bmi(+35%), weight_kg(+20%), activity_level(+5%)"},
            {"role": "user",      "content": "Domain: fitness | Q: Am I fit? | weight_kg:70, height_cm:180, activity_level:3"},
            {"role": "assistant", "content": "PROBABILITY: 12.0\nCONFIDENCE: HIGH\nREASONING: BMI is 21.6 (Healthy range) and daily activity level indicates high cardiovascular health. Risk of obesity is minimal.\nKEY_FACTORS: activity_level(-15%), bmi(-10%), weight_kg(-5%)"},
        ],
        "student": [
            {"role": "user",      "content": "Domain: student | Q: Am I eligible for Infosys? | cgpa:8.5, technical_skills:3, internships_done:2, backlogs:0"},
            {"role": "assistant", "content": "PROBABILITY: 92.4\nCONFIDENCE: HIGH\nREASONING: High CGPA (8.5) exceeds most eligibility cutoffs (usually 6.0-7.0). 2 internships and advanced skills make for a highly competitive profile.\nKEY_FACTORS: cgpa(+25%), internships_done(+15%), backlogs(cleared)"},
            {"role": "user",      "content": "Domain: student | Q: Can I get into a top company? | cgpa:5.5, technical_skills:1, internships_done:0, backlogs:2"},
            {"role": "assistant", "content": "PROBABILITY: 14.8\nCONFIDENCE: HIGH\nREASONING: CGPA of 5.5 is below the standard 60% threshold for top-tier companies. Active backlogs and lack of internships are significant blockers.\nKEY_FACTORS: backlogs(-30%), cgpa(-20%), technical_skills(-10%)"},
        ],
        "financial": [
            {"role": "user",      "content": "Domain: financial | Q: Will this investment yield returns? | annual_income:1200000, transaction_amount:500000, stability_duration:5"},
            {"role": "assistant", "content": "PROBABILITY: 74.2\nCONFIDENCE: MODERATE\nREASONING: Strong income-to-transaction ratio and 5 years of stability provide a solid buffer. High probability of success based on capital liquidity.\nKEY_FACTORS: stability_duration(+15%), annual_income(+10%), capital_ratio(+5%)"},
        ],
        "job_life": [
            {"role": "user",      "content": "Domain: job_life | Q: Will I get promoted? | role_satisfaction:4, tenure_duration:3, growth_opportunity:3"},
            {"role": "assistant", "content": "PROBABILITY: 68.5\nCONFIDENCE: MODERATE\nREASONING: Moderate tenure and positive role satisfaction are good signals. Growth opportunities are present but not guaranteed.\nKEY_FACTORS: tenure_duration(+10%), role_satisfaction(+8%), growth_opportunity(+5%)"},
        ],
        "health": [
            {"role": "user",      "content": "Domain: health | Q: Am I at risk of diabetes? | glucose:140, age:45, health_habits:1"},
            {"role": "assistant", "content": "PROBABILITY: 62.8\nCONFIDENCE: HIGH\nREASONING: Elevated glucose (140 mg/dL) and poor health habits in a 45-year-old indicate significant risk. Immediate lifestyle change recommended.\nKEY_FACTORS: glucose(+25%), health_habits(+15%), age(+5%)"},
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

    # Use high-speed Groq 8B for fast generation
    raw = call_groq(messages, temperature=0.1, max_tokens=250, model="llama-3.1-8b-instant")

    # Robust parsing logic
    result = {"raw": raw, "probability": None, "confidence": "MODERATE",
              "reasoning": "", "key_factors": []}
    
    # Extract PROBABILITY
    prob_match = re.search(r"PROBABILITY:\s*([\d\.]+)", raw)
    if prob_match:
        try:
            result["probability"] = float(prob_match.group(1)) / 100
        except: pass
        
    # Extract CONFIDENCE
    conf_match = re.search(r"CONFIDENCE:\s*(HIGH|MODERATE|LOW|CRITICAL)", raw)
    if conf_match:
        result["confidence"] = conf_match.group(1)
        
    # Extract REASONING
    reason_match = re.search(r"REASONING:\s*(.*?)(?:\n|$|KEY_FACTORS)", raw, re.DOTALL)
    if reason_match:
        result["reasoning"] = reason_match.group(1).strip()
        
    # Extract KEY_FACTORS
    factors_match = re.search(r"KEY_FACTORS:\s*(.*?)(?:\n|$)", raw)
    if factors_match:
        result["key_factors"] = [f.strip() for f in factors_match.group(1).split(",")]

    # Safety Fallback: If parsing fails, use regex for any number
    if result["probability"] is None:
        numbers = re.findall(r"\d+\.?\d*", raw)
        if numbers:
            # Assume first number > 1 and < 100 is the probability
            for n in numbers:
                val = float(n)
                if 0 < val <= 100:
                    result["probability"] = val / 100
                    break

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
