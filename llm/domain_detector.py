import json
import logging
from llm.groq_client import call_groq

logger = logging.getLogger(__name__)

# Must match domain_registry.yaml keys exactly
VALID_DOMAINS = [
    "student",
    "higher_education",
    "hr",
    "disease",
    "fitness",
    "loan",
    "mental_health",
    "claim",
    "behavioral",
]

DOMAIN_DESCRIPTIONS = {
    "student":           "school/college pass-fail, grades, study habits, attendance, exam performance",
    "higher_education":  "university dropout, graduation, enrollment, tuition, scholarship",
    "hr":                "employee attrition, resignation, job satisfaction, workplace, retention",
    "disease":           "medical diagnosis, heart disease, diabetes, symptoms, health risk",
    "fitness":           "body performance, fitness goals, exercise, weight, physical ability",
    "loan":              "credit, debt, default, repayment, mortgage, financial risk",
    "mental_health":     "depression, burnout, stress, anxiety, psychological wellbeing",
    "claim":             "fact checking, fake news, misinformation, claim credibility, verification",
    "behavioral":        "reviews, sentiment, opinions, consumer behavior, deception detection",
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
    {"role": "assistant", "content": "{\"domain\": \"student\", \"confidence\": \"HIGH\", \"reason\": \"Question is about academic exam performance and study habits.\"}"},

    {"role": "user",      "content": "Is this employee likely to quit in the next 6 months?"},
    {"role": "assistant", "content": "{\"domain\": \"hr\", \"confidence\": \"HIGH\", \"reason\": \"Question is about employee attrition and workplace retention.\"}"},

    {"role": "user",      "content": "Does this patient have a high risk of heart disease based on their cholesterol?"},
    {"role": "assistant", "content": "{\"domain\": \"disease\", \"confidence\": \"HIGH\", \"reason\": \"Question is about medical risk and disease diagnosis.\"}"},

    {"role": "user",      "content": "Will this person default on their loan given their credit score is 480?"},
    {"role": "assistant", "content": "{\"domain\": \"loan\", \"confidence\": \"HIGH\", \"reason\": \"Question is about credit risk and loan repayment probability.\"}"},

    {"role": "user",      "content": "Is this news article about Article 370 true or fake?"},
    {"role": "assistant", "content": "{\"domain\": \"claim\", \"confidence\": \"HIGH\", \"reason\": \"Question is about verifying the credibility of a factual claim.\"}"},

    {"role": "user",      "content": "Will this university student graduate or drop out?"},
    {"role": "assistant", "content": "{\"domain\": \"higher_education\", \"confidence\": \"HIGH\", \"reason\": \"Question is about university dropout or graduation outcome.\"}"},

    {"role": "user",      "content": "Is this person likely to be depressed based on their sleep and work hours?"},
    {"role": "assistant", "content": "{\"domain\": \"mental_health\", \"confidence\": \"HIGH\", \"reason\": \"Question is about psychological wellbeing and depression risk.\"}"},

    {"role": "user",      "content": "Will this person achieve their fitness goals given their body fat percentage?"},
    {"role": "assistant", "content": "{\"domain\": \"fitness\", \"confidence\": \"HIGH\", \"reason\": \"Question is about physical fitness and body performance goals.\"}"},

    {"role": "user",      "content": "Is this product review genuine or fake?"},
    {"role": "assistant", "content": "{\"domain\": \"behavioral\", \"confidence\": \"HIGH\", \"reason\": \"Question is about deception detection in consumer reviews.\"}"},

    {"role": "user",      "content": "Will it rain tomorrow in Hyderabad?"},
    {"role": "assistant", "content": "{\"domain\": \"free_inference\", \"confidence\": \"HIGH\", \"reason\": \"Weather prediction does not fit any registered domain.\"}"},
]


def detect_domain(question: str) -> dict:
    """
    Detects domain from user question.
    Returns dict with keys: domain, confidence, reason, is_free_inference
    """
    messages = (
        [{"role": "system", "content": SYSTEM_PROMPT}]
        + FEW_SHOTS
        + [{"role": "user", "content": question}]
    )

    try:
        raw = call_groq(messages, temperature=0.1, max_tokens=120)

        # Strip markdown fences if present
        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        clean = clean.strip()

        result = json.loads(clean)
        domain = result.get("domain", "free_inference")

        # Validate domain is in registry
        if domain not in VALID_DOMAINS and domain != "free_inference":
            logger.warning(f"LLM returned unknown domain '{domain}' — routing to free_inference")
            domain = "free_inference"

        return {
            "domain":           domain,
            "confidence":       result.get("confidence", "MODERATE"),
            "reason":           result.get("reason", ""),
            "is_free_inference": domain == "free_inference",
        }

    except json.JSONDecodeError as e:
        logger.error(f"Domain detector JSON parse failed: {e} | raw: {raw}")
        return {
            "domain":           "free_inference",
            "confidence":       "LOW",
            "reason":           "Could not parse domain from question — routing to free inference.",
            "is_free_inference": True,
        }
    except Exception as e:
        logger.error(f"Domain detector failed: {e}")
        return {
            "domain":           "free_inference",
            "confidence":       "LOW",
            "reason":           f"Detection error: {e}",
            "is_free_inference": True,
        }


if __name__ == "__main__":
    tests = [
        "Will my student pass the semester?",
        "Is this employee going to resign?",
        "Does this person have diabetes risk?",
        "Is this news headline fake?",
        "Will it rain tomorrow?",
        "Will this university student drop out?",
        "Is this review genuine or fake?",
    ]
    print("Domain Detector — Test Run\n" + "=" * 40)
    for q in tests:
        result = detect_domain(q)
        print(f"Q: {q}")
        print(f"   Domain: {result['domain']} | Confidence: {result['confidence']}")
        print(f"   Reason: {result['reason']}\n")
