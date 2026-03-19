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
        ("study_hours_per_day", "How many hours does the student study per day?",
         ["1-2 hours", "3-4 hours", "5-6 hours", "7+ hours"]),
        ("attendance_pct",      "What is the student's attendance percentage?",
         ["Below 50%", "50-70%", "70-85%", "Above 85%"]),
        ("past_score",          "What were their recent exam scores?",
         ["Below 40", "40-60", "60-75", "Above 75"]),
        ("stress_level",        "What is the student's current stress level?",
         ["Low", "Medium", "High", "Very High"]),
        ("sleep_hours",         "How many hours of sleep do they get?",
         ["Less than 5", "5-6 hours", "7-8 hours", "More than 8"]),
        ("motivation",          "How motivated is the student (1-5)?",
         ["1 - Very low", "2 - Low", "3 - Medium", "4 - High", "5 - Very high"]),
    ],
    "hr": [
        ("job_satisfaction",    "How satisfied is the employee with their job (1-4)?",
         ["1 - Very dissatisfied", "2 - Dissatisfied", "3 - Satisfied", "4 - Very satisfied"]),
        ("work_life_balance",   "How is their work-life balance (1-4)?",
         ["1 - Bad", "2 - Good", "3 - Better", "4 - Best"]),
        ("overtime",            "Does the employee work overtime?",
         ["Yes", "No"]),
        ("years_at_company",    "How many years have they been at the company?",
         ["Less than 1", "1-3 years", "3-7 years", "7+ years"]),
        ("monthly_income",      "What is their approximate monthly income?",
         ["Below 3000", "3000-6000", "6000-10000", "Above 10000"]),
    ],
    "disease": [
        ("age",             "What is the patient's age?",
         ["Under 30", "30-45", "45-60", "Above 60"]),
        ("glucose",         "What is their glucose level (mg/dL)?",
         ["Below 100", "100-125", "126-199", "200+"]),
        ("bmi",             "What is their BMI?",
         ["Under 18.5", "18.5-24.9", "25-29.9", "30+"]),
        ("blood_pressure",  "What is their blood pressure (systolic)?",
         ["Below 120", "120-129", "130-139", "140+"]),
        ("smoking",         "Do they smoke?",
         ["Never", "Former smoker", "Current smoker"]),
    ],
    "loan": [
        ("credit_score",    "What is their credit score?",
         ["Below 580", "580-669", "670-739", "740+"]),
        ("income",          "What is their annual income?",
         ["Below 30000", "30000-60000", "60000-100000", "100000+"]),
        ("loan_amount",     "How much loan are they requesting?",
         ["Below 10000", "10000-30000", "30000-60000", "60000+"]),
        ("missed_payments", "How many missed payments in their history?",
         ["0", "1-2", "3-5", "6+"]),
        ("employment_years","How many years have they been employed?",
         ["0-1 years", "1-3 years", "3-7 years", "7+ years"]),
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
    "1-2 hours": 1.5, "3-4 hours": 3.5, "5-6 hours": 5.5, "7+ hours": 8,
    "Below 50%": 40, "50-70%": 60, "70-85%": 77, "Above 85%": 90,
    "Below 40": 35, "40-60": 50, "60-75": 67, "Above 75": 82,
    "Low": "low", "Medium": "medium", "High": "high", "Very High": "very_high",
    "Less than 5": 4, "5-6 hours": 5.5, "7-8 hours": 7.5, "More than 8": 9,
    "1 - Very low": 1, "2 - Low": 2, "3 - Medium": 3, "4 - High": 4, "5 - Very high": 5,
    "1 - Very dissatisfied": 1, "2 - Dissatisfied": 2, "3 - Satisfied": 3, "4 - Very satisfied": 4,
    "1 - Bad": 1, "2 - Good": 2, "3 - Better": 3, "4 - Best": 4,
    "Yes": 1, "No": 0,
    "Less than 1": 0.5, "1-3 years": 2, "3-7 years": 5, "7+ years": 10,
    "Below 3000": 2000, "3000-6000": 4500, "6000-10000": 8000, "Above 10000": 12000,
    "Under 30": 25, "30-45": 37, "45-60": 52, "Above 60": 65,
    "Below 100": 90, "100-125": 112, "126-199": 160, "200+": 220,
    "Under 18.5": 17, "18.5-24.9": 22, "25-29.9": 27, "30+": 33,
    "Below 120": 110, "120-129": 124, "130-139": 134, "140+": 150,
    "Never": 0, "Former smoker": 0.5, "Current smoker": 1,
    "Below 580": 550, "580-669": 625, "670-739": 705, "740+": 780,
    "Below 30000": 20000, "30000-60000": 45000, "60000-100000": 80000, "100000+": 120000,
    "Below 10000": 7000, "10000-30000": 20000, "30000-60000": 45000, "60000+": 75000,
    "0": 0, "1-2": 1, "3-5": 4, "6+": 8,
    "0-1 years": 0.5, "1-3 years": 2, "3-7 years": 5, "7+ years": 10,
    "Less than 40": 35, "40-50": 45, "50-60": 55, "60+": 65,
    "None": "none", "Weak": "weak", "Moderate": "moderate", "Strong": "strong",
}


class ConversationalSession:
    """
    Manages a multi-turn conversational prediction session.
    Section 6.4 — one question at a time, stops at RI >= 75%.
    """

    def __init__(self, domain: str, question: str = None):
        self.domain      = domain
        self.question    = question or f"What is the probability of a positive outcome?"
        self.parameters  = {}
        self.history     = []
        self.step        = 0
        self.complete    = False
        self.questions   = DOMAIN_QUESTIONS.get(domain, [])
        self.reliability = 0.0

    def get_next_question(self) -> dict:
        """Returns next question to ask, or None if session complete."""
        if self.step >= len(self.questions):
            self.complete = True
            return None

        param_key, question_text, options = self.questions[self.step]
        return {
            "step":        self.step + 1,
            "total_steps": len(self.questions),
            "param_key":   param_key,
            "question":    question_text,
            "options":     options,
            "progress":    f"{self.step + 1} of {len(self.questions)}",
            "reliability": round(self.reliability * 100, 1),
            "can_skip":    True,
        }

    def submit_answer(self, param_key: str, value: str, skipped: bool = False) -> dict:
        """Process user answer and return updated state."""
        if not skipped:
            # Convert chip selection to numeric value
            numeric_val = CHIP_VALUES.get(value, value)
            self.parameters[param_key] = numeric_val
            self.history.append({
                "step":      self.step + 1,
                "param":     param_key,
                "raw_value": value,
                "value":     numeric_val,
            })

        self.step += 1

        # Update reliability
        self.reliability = min(1.0, len(self.parameters) / max(len(self.questions), 1))

        # Check if we should stop
        if self.reliability >= 0.75 or self.step >= len(self.questions):
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
            "You are Sambhav, a probabilistic inference assistant.\n"
            "You collect information through natural conversation to make predictions.\n"
            "Ask ONE focused question at a time — never multiple questions.\n"
            "Be warm and concise. Stop collecting when you have enough info (5+ parameters).\n\n"
            f"Domain: {domain}\n"
            f"Goal: {question}\n"
            f"Parameters collected so far:\n{param_str or 'None yet'}\n"
            f"Reliability Index: {reliability}%\n\n"
            "If reliability >= 75%, say you have enough info and offer to predict.\n"
            "Otherwise ask the most impactful missing parameter.\n\n"
            "Respond in JSON:\n"
            "{\n"
            '  "message": "<your conversational response>",\n'
            '  "asking_for": "<parameter name you are asking about or null>",\n'
            '  "ready_to_predict": <true|false>,\n'
            '  "suggested_options": ["<option1>", "<option2>", "<option3>"]\n'
            "}"
        )},
        {"role": "user", "content": (
            f"Conversation so far:\n{history_str}\n\n"
            "What should Sambhav say next?"
        )}
    ]

    result = route("conversational", messages, max_tokens=300, temperature=0.3)
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
