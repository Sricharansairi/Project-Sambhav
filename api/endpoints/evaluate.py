"""
api/endpoints/evaluate.py — Section 11.4 Evaluation Feature
POST /evaluate — submit actual outcome, compute Brier score, grade, lessons learned
"""

import logging, json, re
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional
from api.endpoints.auth import get_current_user

router = APIRouter()
logger = logging.getLogger(__name__)

# In-memory evaluations store
_evaluations: list = []
_user_brier:  dict = {}  # username -> {domain -> [brier_scores]}


# ── Request schema ────────────────────────────────────────────
class EvaluateRequest(BaseModel):
    prediction_id:     str   = Field(..., example="SMB-2026-00001")
    predicted_prob:    float = Field(..., example=0.71)
    actual_outcome:    bool  = Field(..., example=False)
    domain:            str   = Field(..., example="student")
    what_system_missed: Optional[str] = None
    quality_rating:    Optional[int]  = Field(None, ge=1, le=5)


# ── Brier Score ───────────────────────────────────────────────
def compute_brier(predicted: float, actual: bool) -> float:
    """BS = (predicted - actual)^2. Lower = better. 0=perfect, 0.25=coin flip."""
    return round((predicted - int(actual)) ** 2, 4)


# ── Evaluation Grade ──────────────────────────────────────────
def assign_grade(brier: float) -> str:
    if brier <= 0.04:  return "A+"
    elif brier <= 0.09: return "A"
    elif brier <= 0.16: return "B"
    elif brier <= 0.25: return "C"
    elif brier <= 0.36: return "D"
    else:               return "F"


# ── Lessons Learned via LLM ───────────────────────────────────
def generate_lessons(
    domain: str,
    predicted_prob: float,
    actual_outcome: bool,
    brier: float,
    what_missed: str = None,
) -> str:
    try:
        from llm.router import route
        direction = "overestimated" if predicted_prob > 0.5 and not actual_outcome else \
                    "underestimated" if predicted_prob < 0.5 and actual_outcome else "well-calibrated"
        missed_str = f"User noted: '{what_missed}'" if what_missed else "No additional context provided."

        messages = [
            {"role": "system", "content": (
                "You are Project Sambhav's evaluation engine. "
                "Generate concise, actionable lessons learned from a prediction evaluation. "
                "Be specific — cite the domain and direction of error. "
                "Keep it to 2-3 sentences maximum. No bullet points."
            )},
            {"role": "user", "content": (
                f"Domain: {domain}\n"
                f"Predicted probability: {predicted_prob*100:.1f}%\n"
                f"Actual outcome: {'YES (happened)' if actual_outcome else 'NO (did not happen)'}\n"
                f"Brier Score: {brier} (grade: {assign_grade(brier)})\n"
                f"Prediction direction: {direction}\n"
                f"{missed_str}\n\n"
                "What should the system learn from this? What parameters were likely over/under-weighted?"
            )}
        ]
        result = route("llm_predict", messages, max_tokens=150, temperature=0.3)
        return result.get("content", "").strip()
    except Exception as e:
        logger.warning(f"Lessons generation failed: {e}")
        predicted_pct = round(predicted_prob * 100, 1)
        if predicted_prob > 0.5 and not actual_outcome:
            return f"Sambhav overestimated the probability at {predicted_pct}% when the outcome did not occur. Consider adding more risk-factor parameters for the {domain} domain."
        elif predicted_prob < 0.5 and actual_outcome:
            return f"Sambhav underestimated at {predicted_pct}% when the outcome actually occurred. Positive signals may have been underweighted in the {domain} domain."
        else:
            return f"Prediction was well-calibrated for the {domain} domain with a Brier Score of {brier}."


# ── POST /evaluate ────────────────────────────────────────────
@router.post("")
async def evaluate_prediction(
    req: EvaluateRequest,
    user: dict = Depends(get_current_user)
):
    """
    Section 11.4 — Submit actual outcome for a prediction.
    Computes Brier Score, assigns grade, generates lessons learned.
    Updates personal calibration tracking.
    """
    username = user.get("username", "guest")
    logger.info(f"POST /evaluate prediction_id={req.prediction_id} actual={req.actual_outcome}")

    # Compute Brier Score
    brier  = compute_brier(req.predicted_prob, req.actual_outcome)
    grade  = assign_grade(brier)

    # Generate lessons learned
    lessons = generate_lessons(
        domain=req.domain,
        predicted_prob=req.predicted_prob,
        actual_outcome=req.actual_outcome,
        brier=brier,
        what_missed=req.what_system_missed,
    )

    # Update personal Brier tracking
    if username != "guest":
        if username not in _user_brier:
            _user_brier[username] = {}
        if req.domain not in _user_brier[username]:
            _user_brier[username][req.domain] = []
        _user_brier[username][req.domain].append(brier)

    # Compute personal stats
    domain_scores = _user_brier.get(username, {}).get(req.domain, [brier])
    personal_domain_brier = round(sum(domain_scores) / len(domain_scores), 4)

    all_scores = [s for scores in _user_brier.get(username, {}).values() for s in scores]
    overall_brier = round(sum(all_scores) / len(all_scores), 4) if all_scores else brier

    # Calibration message
    if overall_brier <= 0.09:
        calibration_msg = "Excellent calibration — your predictions are highly accurate."
    elif overall_brier <= 0.16:
        calibration_msg = "Good calibration — your predictions are generally reliable."
    elif overall_brier <= 0.25:
        calibration_msg = "Moderate calibration — there is room to improve prediction accuracy."
    else:
        calibration_msg = "Poor calibration — consider providing more detailed parameters."

    # Save evaluation record
    record = {
        "evaluation_id":    f"EVAL-{len(_evaluations)+1:05d}",
        "prediction_id":    req.prediction_id,
        "username":         username,
        "domain":           req.domain,
        "predicted_prob":   req.predicted_prob,
        "actual_outcome":   req.actual_outcome,
        "brier_score":      brier,
        "grade":            grade,
        "lessons_learned":  lessons,
        "what_missed":      req.what_system_missed,
        "quality_rating":   req.quality_rating,
    }
    _evaluations.append(record)

    return {
        "success":              True,
        "evaluation_id":        record["evaluation_id"],
        "prediction_id":        req.prediction_id,
        "brier_score":          brier,
        "grade":                grade,
        "grade_explanation": {
            "A+": "Brier ≤ 0.04 — near-perfect calibration",
            "A":  "Brier ≤ 0.09 — excellent",
            "B":  "Brier ≤ 0.16 — good",
            "C":  "Brier ≤ 0.25 — coin-flip equivalent",
            "D":  "Brier ≤ 0.36 — poor",
            "F":  "Brier > 0.36 — very poor",
        }.get(grade, ""),
        "lessons_learned":      lessons,
        "personal_stats": {
            "domain_brier_score":  personal_domain_brier,
            "overall_brier_score": overall_brier,
            "total_evaluations":   len(all_scores),
            "calibration_message": calibration_msg,
        },
        "disclaimer": "Sambhav may be incorrect. Always verify important decisions independently.",
    }


# ── GET /evaluate/stats ───────────────────────────────────────
@router.get("/stats")
async def get_evaluation_stats(
    user: dict = Depends(get_current_user)
):
    """Get personal calibration statistics for the current user."""
    username = user.get("username", "guest")
    if username == "guest":
        return {"success": True, "note": "Register to track personal calibration"}

    user_data = _user_brier.get(username, {})
    if not user_data:
        return {"success": True, "message": "No evaluations yet", "stats": {}}

    domain_stats = {}
    for domain, scores in user_data.items():
        avg = round(sum(scores) / len(scores), 4)
        domain_stats[domain] = {
            "brier_score":    avg,
            "grade":          assign_grade(avg),
            "n_evaluations":  len(scores),
            "best":           round(min(scores), 4),
            "worst":          round(max(scores), 4),
            "trend":          "improving" if len(scores) > 1 and scores[-1] < scores[0] else
                              "declining" if len(scores) > 1 and scores[-1] > scores[0] else "stable",
        }

    all_scores = [s for scores in user_data.values() for s in scores]
    overall = round(sum(all_scores) / len(all_scores), 4)

    return {
        "success":         True,
        "username":        username,
        "overall_brier":   overall,
        "overall_grade":   assign_grade(overall),
        "total_evals":     len(all_scores),
        "domain_stats":    domain_stats,
    }


# ── GET /evaluate/history ─────────────────────────────────────
@router.get("/history")
async def get_evaluation_history(
    user: dict = Depends(get_current_user)
):
    """Get all evaluations submitted by this user."""
    username = user.get("username", "guest")
    records = [e for e in _evaluations if e["username"] == username]
    return {
        "success":     True,
        "evaluations": records,
        "total":       len(records),
    }
