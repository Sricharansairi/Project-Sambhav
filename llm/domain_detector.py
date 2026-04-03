import json
import logging
from llm.groq_client import call_groq

logger = logging.getLogger(__name__)

# Must match domain_registry.yaml keys exactly — all 11 domains
VALID_DOMAINS = [
    "student",
    "high_school",
    "job_life",
    "health",
    "fitness",
    "financial",
    "mental_health",
    "claim",
    "behavioral",
    "pragma",
    "sarvagna",
]

# Aliases — user phrases that map to canonical domain keys
DOMAIN_ALIASES = {
    "behaviour":         "behavioral",
    "behaviour analysis":"behavioral",
    "review":            "behavioral",
    "sentiment":         "behavioral",
    "fake review":       "behavioral",
    "deception":         "behavioral",
    "high school":       "high_school",
    "schooling":         "high_school",
    "university":        "high_school",
    "graduation":        "high_school",
    "dropout":           "high_school",
    "college":           "high_school",
    "mental":            "mental_health",
    "mental health":     "mental_health",
    "depression":        "mental_health",
    "burnout":           "mental_health",
    "anxiety":           "mental_health",
    "stress":            "mental_health",
    "psychological":     "mental_health",
    "medical":           "health",
    "disease":           "health",
    "diagnosis":         "health",
    "health risk":       "health",
    "diabetes":          "health",
    "heart":             "health",
    "doctor":            "health",
    "patient":           "health",
    "forensic":          "pragma",
    "lie detection":     "pragma",
    "deception detection":"pragma",
    "psychological profiling":"pragma",
    "statement analysis":"pragma",
    "vega":              "sarvagna",
    "life oracle":       "sarvagna",
    "life domains":      "sarvagna",
    "oracle":            "sarvagna",
    "life prediction":   "sarvagna",
    "career prediction": "sarvagna",
    "daily life":        "sarvagna",
    "everyday":          "sarvagna",
    "credit":            "financial",
    "mortgage":          "financial",
    "default":           "financial",
    "debt":              "financial",
    "finance":           "financial",
    "investment":        "financial",
    "attrition":         "job_life",
    "employee":          "job_life",
    "resignation":       "job_life",
    "workplace":         "job_life",
    "career":            "job_life",
    "job life":          "job_life",
    "fact check":        "claim",
    "misinformation":    "claim",
    "fake news":         "claim",
    "claim":             "claim",
    "credibility":       "claim",
    "student":           "student",
    "exam":              "student",
    "pass":              "student",
    "academic":          "student",
    "grades":            "student",
    "fitness":           "fitness",
    "exercise":          "fitness",
    "body":              "fitness",
    "weight":            "fitness",
}

DOMAIN_DESCRIPTIONS = {
    "student":          "school/college pass-fail, grades, study habits, attendance, exam performance",
    "high_school":      "high school dropout, graduation, enrollment, tuition, scholarship, secondary education",
    "job_life":         "professional stability, career transitions, employee attrition, resignation, job satisfaction, workplace, retention",
    "health":           "medical diagnosis, health wellness, heart disease, diabetes, symptoms, disease risk",
    "fitness":          "body performance, fitness goals, exercise, weight, physical ability",
    "financial":        "financial scenarios, credit, debt, default, repayment, mortgage, financial risk, investment stability",
    "mental_health":    "depression, burnout, stress, anxiety, psychological wellbeing",
    "claim":            "fact checking, fake news, misinformation, claim credibility, verification",
    "behavioral":       "reviews, sentiment, opinions, consumer behavior, deception detection",
    "pragma":           "forensic psychological profiling, deception detection, statement analysis, lie detection, psychological markers",
    "sarvagna":         "life decision model — career, finance, relationships, health, education, spirituality, family, creativity, location, social impact",
}

SYSTEM_PROMPT = (
    "You are a domain classifier for Project Sambhav, a probabilistic inference engine.\n"
    "Given a user question or situation, identify which domain it belongs to.\n\n"
    "Valid domains and their scope:\n"
    + "\n".join([f"- {d}: {DOMAIN_DESCRIPTIONS[d]}" for d in VALID_DOMAINS])
    + "\n\n"
    "Rules:\n"
    "1. Return ONLY a valid domain key from the list above\n"
    "2. If nothing fits, return: free_inference\n"
    "3. Return JSON only — no explanation, no markdown\n"
    "4. Format: {\"domain\": \"<key>\", \"confidence\": \"HIGH|MODERATE|LOW\", "
    "\"reason\": \"<one sentence>\"}"
)

FEW_SHOTS = [
    {"role": "user",      "content": "Will this student pass their final exams given they study 4 hours a day?"},
    {"role": "assistant", "content": "{\"domain\": \"student\", \"confidence\": \"HIGH\", \"reason\": \"Question is about academic exam performance.\"}"},
    {"role": "user",      "content": "Is this employee likely to quit in the next 6 months?"},
    {"role": "assistant", "content": "{\"domain\": \"job_life\", \"confidence\": \"HIGH\", \"reason\": \"Question is about employee attrition and workplace stability.\"}"},
    {"role": "user",      "content": "Does this patient have a high risk of heart disease based on their cholesterol?"},
    {"role": "assistant", "content": "{\"domain\": \"health\", \"confidence\": \"HIGH\", \"reason\": \"Question is about medical disease risk and wellness.\"}"},
    {"role": "user",      "content": "Will this person default on their loan given their credit score is 480?"},
    {"role": "assistant", "content": "{\"domain\": \"financial\", \"confidence\": \"HIGH\", \"reason\": \"Question is about financial risk and credit repayment.\"}"},
    {"role": "user",      "content": "Is this news article about Article 370 true or fake?"},
    {"role": "assistant", "content": "{\"domain\": \"claim\", \"confidence\": \"HIGH\", \"reason\": \"Question is about verifying the credibility of a factual claim.\"}"},
    {"role": "user",      "content": "Will this university student graduate or drop out?"},
    {"role": "assistant", "content": "{\"domain\": \"high_school\", \"confidence\": \"HIGH\", \"reason\": \"Question is about academic outcomes and graduation.\"}"},
    {"role": "user",      "content": "Is this person likely to be depressed based on their sleep and work hours?"},
    {"role": "assistant", "content": "{\"domain\": \"mental_health\", \"confidence\": \"HIGH\", \"reason\": \"Question is about psychological wellbeing and depression risk.\"}"},
    {"role": "user",      "content": "Will this person achieve their fitness goals given their body fat percentage?"},
    {"role": "assistant", "content": "{\"domain\": \"fitness\", \"confidence\": \"HIGH\", \"reason\": \"Question is about physical fitness and body performance goals.\"}"},
    {"role": "user",      "content": "Is this product review genuine or fake?"},
    {"role": "assistant", "content": "{\"domain\": \"behavioral\", \"confidence\": \"HIGH\", \"reason\": \"Question is about deception detection in consumer reviews.\"}"},
    {"role": "user",      "content": "Is this person telling the truth in their statement?"},
    {"role": "assistant", "content": "{\"domain\": \"pragma\", \"confidence\": \"HIGH\", \"reason\": \"Question is about forensic psychological profiling and deception detection.\"}"},
    {"role": "user",      "content": "What are my career and relationship prospects this year?"},
    {"role": "assistant", "content": "{\"domain\": \"sarvagna\", \"confidence\": \"HIGH\", \"reason\": \"Question spans multiple life domains — career and relationships.\"}"},
    {"role": "user",      "content": "Will it rain tomorrow in Hyderabad?"},
    {"role": "assistant", "content": "{\"domain\": \"free_inference\", \"confidence\": \"HIGH\", \"reason\": \"Weather prediction does not fit any registered domain.\"}"},
]


def detect_domain(question: str) -> dict:
    """
    Detects domain from user question.
    First checks alias table, then calls LLM.
    Returns dict with keys: domain, confidence, reason, is_free_inference
    """
    # Fast alias check before LLM call
    q_lower = question.lower().strip()
    for phrase, domain in DOMAIN_ALIASES.items():
        if phrase in q_lower:
            return {
                "domain":            domain,
                "confidence":        "HIGH",
                "reason":            f"Matched alias '{phrase}' → {domain}",
                "is_free_inference": False,
            }

    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + FEW_SHOTS
        + [{"role": "user", "content": question}]
    )

    try:
        raw = call_groq(messages, temperature=0.1, max_tokens=120)

        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        clean = clean.strip()

        result = json.loads(clean)
        domain = result.get("domain", "free_inference")

        if domain not in VALID_DOMAINS and domain != "free_inference":
            logger.warning(f"LLM returned unknown domain '{domain}' — routing to free_inference")
            domain = "free_inference"

        return {
            "domain":            domain,
            "confidence":        result.get("confidence", "MODERATE"),
            "reason":            result.get("reason", ""),
            "is_free_inference": domain == "free_inference",
        }

    except json.JSONDecodeError as e:
        logger.error(f"Domain detector JSON parse failed: {e}")
        return {
            "domain":            "free_inference",
            "confidence":        "LOW",
            "reason":            "Could not parse domain — routing to free inference.",
            "is_free_inference": True,
        }
    except Exception as e:
        logger.error(f"Domain detector failed: {e}")
        return {
            "domain":            "free_inference",
            "confidence":        "LOW",
            "reason":            f"Detection error: {e}",
            "is_free_inference": True,
        }


if __name__ == "__main__":
    tests = [
        "Will my student pass the semester?",
        "Is this employee going to resign?",
        "Does this person have diabetes risk?",
        "Is this news headline fake?",
        "Will it rain tomorrow?",
        "Is this person lying in their testimony?",
        "What are my career prospects this year?",
        "Will this person achieve their fitness goal?",
    ]
    print("Domain Detector — Test Run\n" + "=" * 40)
    for q in tests:
        result = detect_domain(q)
        print(f"Q: {q}")
        print(f"   Domain: {result['domain']} | Confidence: {result['confidence']}")
        print(f"   Reason: {result['reason']}\n")
