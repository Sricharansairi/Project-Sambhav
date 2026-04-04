"""
llm/conversational_mode.py — Section 6.4 Conversational Mode
Multi-turn dialogue that collects parameters one question at a time.
Asks what would raise Reliability Index most (SHAP-powered).
Stops when RI >= 75% or user indicates no more info.
"""

import json, re, logging
from llm.router import route

logger = logging.getLogger(__name__)

# Domain-specific question sequences
DOMAIN_QUESTIONS = {
    "student": [
        ("cgpa", "What is your current CGPA (0-10 scale)?", 
         ["Below 6.0", "6.0-7.0", "7.0-8.5", "Above 8.5"]),
        ("technical_skills", "How would you rate your technical skills proficiency?",
         ["Beginner", "Intermediate", "Advanced", "Expert"]),
        ("internships_done", "How many internships have you completed?",
         ["0", "1", "2", "3+"]),
        ("backlogs", "Do you have any active backlogs?",
         ["No", "Yes"]),
        ("communication_level", "How would you rate your communication & soft skills?",
         ["Poor", "Average", "Good", "Excellent"]),
    ],
    "fitness": [
        ("weight_kg", "What is your current weight in kilograms?",
         ["Below 50kg", "50-70kg", "70-90kg", "Above 90kg"]),
        ("height_cm", "What is your height in centimeters?",
         ["Below 150cm", "150-170cm", "170-190cm", "Above 190cm"]),
        ("activity_level", "What is your physical activity frequency?",
         ["Sedentary", "1–2x/week", "3–4x/week", "Daily"]),
        ("body_fat_pct", "What is your estimated body fat percentage?",
         ["Below 10%", "10-18%", "19-25%", "26-32%", "Above 32%"]),
    ],
    "job_life": [
        ("role_satisfaction",    "How satisfied is the individual with their current role (1-5)?",
         ["1 - Very low", "2 - Low", "3 - Medium", "4 - High", "5 - Very high"]),
        ("lifestyle_balance",    "How is their work-life or lifestyle balance (1-5)?",
         ["1 - Very poor", "2 - Poor", "3 - Fair", "4 - Good", "5 - Excellent"]),
        ("tenure_duration",      "How long has the individual been in their current position?",
         ["Less than 1 yr", "1-2 years", "3-5 years", "6-10 years", "10+ years"]),
        ("growth_opportunity",   "How do they perceive their current growth opportunities?",
         ["None", "Limited", "Moderate", "Significant", "Exceptional"]),
        ("workplace_culture",    "How would they rate their workplace culture or environment?",
         ["Toxic", "Neutral", "Positive", "Excellent"]),
    ],
    "health": [
        ("age",             "What is the individual's age?",
         ["Under 30", "30-45", "45-60", "Above 60"]),
        ("glucose",         "What is their approximate glucose level (mg/dL)?",
         ["Below 100", "100-125", "126-199", "200+"]),
        ("bmi",             "What is their BMI range?",
         ["Under 18.5", "18.5-24.9", "25-29.9", "30+"]),
        ("blood_pressure",  "What is their blood pressure (systolic)?",
         ["Below 120", "120-129", "130-139", "140+"]),
        ("health_habits",    "How would they rate their overall health habits?",
         ["Poor", "Fair", "Good", "Excellent"]),
    ],
    "financial": [
        ("financial_score",  "What is their financial health or credit score?",
         ["Below 580", "580-669", "670-739", "740+"]),
        ("annual_income",    "What is the annual income or revenue?",
         ["Below ₹3 lakh", "₹3 – 6 lakh", "₹6 – 12 lakh", "₹12 – 25 lakh", "₹25 lakh+"]),
        ("transaction_amount","What is the total amount involved in this scenario?",
         ["Below ₹1 lakh", "₹1 – 5 lakh", "₹5 – 10 lakh", "₹10 – 25 lakh", "₹25 lakh+"]),
        ("stability_duration","Duration of current financial or employment stability?",
         ["None", "< 1 year", "1-3 years", "3-7 years", "7+ years"]),
    ],
    "mental_health": [
        ("sleep_hours",       "How many hours of sleep per night?",
         ["Less than 5", "5-6 hours", "7-8 hours", "More than 8"]),
        ("work_hours",        "How many hours do they work per week?",
         ["Less than 40", "40-50", "50-60", "60+"]),
        ("stress_level",      "What is their stress level?",
         ["Low", "Medium", "High", "Very High"]),
        ("social_support",    "How strong is their social support network?",
         ["None", "Weak", "Moderate", "Strong"]),
        ("family_history",    "Is there a family history of mental health issues?",
         ["Yes", "No"]),
    ],
}

# Value mappings for chip selections
CHIP_VALUES = {
    "Below 6.0": 5.5, "6.0-7.0": 6.5, "7.0-8.5": 7.8, "Above 8.5": 9.2,
    "Beginner": 1, "Intermediate": 2, "Advanced": 3, "Expert": 4,
    "0": 0, "1": 1, "2": 2, "3+": 3,
    "No": 0, "Yes": 1,
    "Poor": 1, "Average": 2, "Good": 3, "Excellent": 4,
    "Below 50kg": 45, "50-70kg": 60, "70-90kg": 80, "Above 90kg": 100,
    "Below 150cm": 145, "150-170cm": 160, "170-190cm": 180, "Above 190cm": 200,
    "Sedentary": 0, "1–2x/week": 1, "3–4x/week": 2, "Daily": 3,
    "Below 10%": 8, "10-18%": 14, "19-25%": 22, "26-32%": 29, "Above 32%": 36,
    "1-2 hours": 1.5, "3-4 hours": 3.5, "5-6 hours": 5.5, "7+ hours": 8,
    "Below 50%": 40, "50-70%": 60, "70-85%": 77, "Above 85%": 90,
    "Below 40": 35, "40-60": 50, "60-75": 67, "Above 75": 82,
    "Low": "low", "Medium": "medium", "High": "high", "Very High": "very_high",
    "Less than 5": 4, "5-6 hours": 5.5, "7-8 hours": 7.5, "More than 8": 9,
    "1 - Very low": 1, "2 - Low": 2, "3 - Medium": 3, "4 - High": 4, "5 - Very high": 5,
    "1 - Very poor": 1, "2 - Poor": 2, "3 - Fair": 3, "4 - Good": 4, "5 - Excellent": 5,
    "Less than 1 yr": 0.5, "1-2 years": 1.5, "3-5 years": 4, "6-10 years": 8, "10+ years": 13,
    "None": 0, "Limited": 1, "Moderate": 2, "Significant": 3, "Exceptional": 4,
    "Toxic": 0, "Neutral": 1, "Positive": 2, "Excellent": 3,
    "Under 30": 25, "30-45": 37, "45-60": 52, "Above 60": 65,
    "Below 100": 90, "100-125": 112, "126-199": 160, "200+": 220,
    "Under 18.5": 17, "18.5-24.9": 22, "25-29.9": 27, "30+": 33,
    "Below 120": 110, "120-129": 124, "130-139": 134, "140+": 150,
    "Poor": 0, "Fair": 1, "Good": 2,
    "Below 580": 550, "580-669": 625, "670-739": 705, "740+": 780,
    "Below ₹3 lakh": 200000, "₹3 – 6 lakh": 450000, "₹6 – 12 lakh": 900000, "₹12 – 25 lakh": 1800000, "₹25 lakh+": 3500000,
    "Below ₹1 lakh": 50000, "₹1 – 5 lakh": 250000, "₹5 – 10 lakh": 750000, "₹10 – 25 lakh": 1750000, "₹25 lakh+": 3500000,
    "< 1 year": 0.5,
    "Less than 40": 35, "40-50": 45, "50-60": 55, "60+": 65,
    "None": "none", "Weak": "weak", "Moderate": "moderate", "Strong": "strong",
}


class ConversationalSession:
    """
    Manages a multi-turn conversational prediction session.
    Section 6.4 — one question at a time, stops at RI >= 75%.
    Powered by Groq LLM for dynamic questioning.
    """

    def __init__(self, domain: str, question: str = None):
        self.domain      = domain
        self.question    = question or f"What is the probability of a positive outcome?"
        self.parameters  = {}
        self.history     = []  # List of {"role": "user"|"assistant", "content": "..."}
        self.step        = 0
        self.complete    = False
        self.reliability = 0.0
        self.max_steps   = 8

    def get_next_question(self) -> dict:
        """Returns next question to ask using LLM, or None if session complete."""
        if self.complete or self.step >= self.max_steps:
            self.complete = True
            return None

        # Call LLM to generate the next smart question
        resp = generate_conversational_response(
            domain=self.domain,
            question=self.question,
            conversation_history=self.history,
            current_parameters=self.parameters
        )

        if resp.get("ready_to_predict") or resp.get("reliability", 0) >= 75:
            self.complete = True
            # Even if ready, return the message as final feedback if any
            if not self.history: # First turn
                 return {
                    "step":        1,
                    "total_steps": self.max_steps,
                    "param_key":   resp.get("asking_for") or "context",
                    "question":    resp.get("message"),
                    "options":     resp.get("suggested_options", []),
                    "progress":    "1 of dynamic",
                    "reliability": round(resp.get("reliability", 0), 1),
                    "can_skip":    True,
                }
            return None

        self.reliability = resp.get("reliability", 0) / 100.0

        return {
            "step":        self.step + 1,
            "total_steps": self.max_steps,
            "param_key":   resp.get("asking_for") or f"param_{self.step}",
            "question":    resp.get("message"),
            "options":     resp.get("suggested_options", []),
            "progress":    f"{self.step + 1} of dynamic",
            "reliability": round(self.reliability * 100, 1),
            "can_skip":    True,
        }

    def submit_answer(self, param_key: str, value: str, skipped: bool = False) -> dict:
        """Process user answer and return updated state."""
        if not skipped:
            # Convert chip selection to numeric value if it matches our known mappings
            numeric_val = CHIP_VALUES.get(value, value)
            self.parameters[param_key] = numeric_val
            
            # Update history for LLM context
            self.history.append({"role": "user", "content": value})

        self.step += 1

        # Check if we should stop based on reliability or max steps
        if self.reliability >= 0.75 or self.step >= self.max_steps:
            self.complete = True

        return {
            "parameters_collected": len(self.parameters),
            "reliability":          round(self.reliability * 100, 1),
            "complete":             self.complete,
            "next_question":        self.get_next_question() if not self.complete else None,
        }

    def get_prediction_ready(self) -> dict:
        """Returns final parameters ready for prediction."""
        return {
            "domain":      self.domain,
            "parameters":  self.parameters,
            "question":    self.question,
            "history":     self.history,
            "reliability": round(self.reliability * 100, 1),
            "steps_taken": self.step,
        }


def generate_conversational_response(
    domain: str,
    question: str,
    conversation_history: list,
    current_parameters: dict,
) -> dict:
    """
    LLM-powered conversational mode — asks smart follow-up questions.
    Used when domain-specific question list is exhausted or domain is unknown.
    """
    history_str = "\n".join([
        f"{msg['role'].upper()}: {msg['content']}"
        for msg in conversation_history[-6:]
    ])

    param_str = "\n".join([f"  - {k}: {v}" for k, v in current_parameters.items()])
    n_params  = len(current_parameters)
    reliability = min(100, n_params * 15)

    messages = [
        {"role": "system", "content": (
            "You are Sambhav. Collect 1 info at a time for prediction.\n"
            "Ask ONE focused question. Be concise.\n\n"
            f"Domain: {domain} | Goal: {question}\n"
            f"Params: {param_str or 'None'}\n"
            f"RI: {reliability}%\n"
            "If RI >= 75%, say 'Ready to predict'.\n"
            "JSON ONLY:\n"
            "{\n"
            '  "message": "...",\n'
            '  "asking_for": "param_key",\n'
            '  "ready_to_predict": bool,\n'
            '  "suggested_options": ["opt1", "opt2"]\n'
            "}"
        )},
        {"role": "user", "content": f"History:\n{history_str}\nNext?"}
    ]

    result = route("conversational", messages, max_tokens=100, temperature=0.1)
    raw    = result.get("content", "")
    raw    = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    if "```" in raw:
        raw = re.sub(r"```(?:json)?", "", raw).strip()

    try:
        parsed = json.loads(raw[raw.find("{"):raw.rfind("}")+1])
    except:
        parsed = {
            "message":          "Can you tell me more about the situation?",
            "asking_for":       None,
            "ready_to_predict": len(current_parameters) >= 5,
            "suggested_options": [],
        }

    return {
        **parsed,
        "reliability":  reliability,
        "param_count":  n_params,
        "provider":     result.get("provider_used", "unknown"),
    }


if __name__ == "__main__":
    print("Conversational Mode Test\n" + "=" * 40)

    # Test structured session
    session = ConversationalSession("student", "Will this student pass their exam?")
    print(f"Domain: {session.domain}")
    print(f"Total questions: {len(session.questions)}\n")

    # Simulate 3 answers
    answers = [
        ("study_hours_per_day", "3-4 hours"),
        ("attendance_pct",      "70-85%"),
        ("past_score",          "60-75"),
    ]

    for param, value in answers:
        q = session.get_next_question()
        if q:
            print(f"Q{q['step']}: {q['question']}")
            print(f"  Options: {q['options']}")
            state = session.submit_answer(param, value)
            print(f"  Answer: {value} -> {session.parameters.get(param)}")
            print(f"  Reliability: {state['reliability']}%")
            print(f"  Complete: {state['complete']}\n")

    ready = session.get_prediction_ready()
    print(f"Parameters collected: {ready['parameters']}")
    print(f"Reliability: {ready['reliability']}%")
