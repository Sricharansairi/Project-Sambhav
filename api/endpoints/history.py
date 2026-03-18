import logging
from fastapi import APIRouter, HTTPException, Depends
from typing import Optional
from api.endpoints.auth import get_current_user

router = APIRouter()
logger = logging.getLogger(__name__)

# In-memory store (replaced by DB in Phase 12)
_predictions: list = []
_prediction_counter = 0

def _next_id() -> str:
    global _prediction_counter
    _prediction_counter += 1
    return f"SMB-2026-{_prediction_counter:05d}"

# ── POST /history/save ────────────────────────────────────────
@router.post("/save")
async def save_prediction(
    prediction: dict,
    user: dict = Depends(get_current_user)
):
    """Save a prediction to history."""
    pid = _next_id()
    record = {
        "prediction_id": pid,
        "user":          user.get("username","guest"),
        "prediction":    prediction,
    }
    _predictions.append(record)
    return {"success": True, "prediction_id": pid}

# ── GET /history ──────────────────────────────────────────────
@router.get("")
async def get_history(
    domain:   Optional[str] = None,
    limit:    int            = 20,
    user: dict = Depends(get_current_user)
):
    """Get prediction history for current user."""
    username = user.get("username","guest")
    if username == "guest":
        return {"success": True, "predictions": [],
                "note": "Register to save prediction history"}
    records = [p for p in _predictions if p["user"] == username]
    if domain:
        records = [p for p in records
                   if p.get("prediction",{}).get("domain") == domain]
    return {
        "success":     True,
        "predictions": records[-limit:],
        "total":       len(records),
    }

# ── GET /history/{prediction_id} ──────────────────────────────
@router.get("/{prediction_id}")
async def get_prediction(
    prediction_id: str,
    user: dict = Depends(get_current_user)
):
    """Get a specific prediction by ID."""
    for p in _predictions:
        if p["prediction_id"] == prediction_id:
            return {"success": True, "prediction": p}
    raise HTTPException(status_code=404,
                        detail=f"Prediction {prediction_id} not found")

# ── DELETE /history/{prediction_id} ───────────────────────────
@router.delete("/{prediction_id}")
async def delete_prediction(
    prediction_id: str,
    user: dict = Depends(get_current_user)
):
    """Delete a prediction (right to erasure — GDPR/DPDP compliance)."""
    global _predictions
    before = len(_predictions)
    _predictions = [p for p in _predictions
                    if p["prediction_id"] != prediction_id]
    if len(_predictions) == before:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return {"success": True, "deleted": prediction_id}
