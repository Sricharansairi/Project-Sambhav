import logging
from fastapi import APIRouter, HTTPException, Depends
from typing import Optional
from api.endpoints.auth import get_current_user

router = APIRouter()
logger = logging.getLogger(__name__)

from sqlalchemy.orm import Session
from db.models import get_db, Prediction
from db.database import get_predictions, delete_prediction as db_delete_prediction

# ── GET /history ──────────────────────────────────────────────
@router.get("")
async def get_history(
    domain:   Optional[str] = None,
    limit:    int            = 20,
    user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get prediction history for current user from database."""
    user_id = user.get("user_id")
    if not user_id or user.get("email") == "guest":
        return {"success": True, "predictions": [],
                "note": "Register to save prediction history"}
    
    records = get_predictions(db, user_id, domain, limit)
    # Map SQLAlchemy objects to dicts for JSON response
    preds = []
    for r in records:
        preds.append({
            "prediction_id": r.prediction_id,
            "domain":        r.domain,
            "question":      r.input_text,
            "created_at":    r.created_at.isoformat() if r.created_at else None,
            "final_probability": r.reconciled_prob,
            "reliability_index": r.reliability_index,
            "mode":          r.mode,
            "parameters":    r.parameters
        })
        
    return {
        "success":     True,
        "predictions": preds,
        "total":       len(preds),
    }

# ── DELETE /history/{prediction_id} ───────────────────────────
@router.delete("/{prediction_id}")
async def delete_prediction_endpoint(
    prediction_id: str,
    user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete a prediction (right to erasure — GDPR/DPDP compliance)."""
    user_id = user.get("user_id")
    if not user_id or user.get("email") == "guest":
         raise HTTPException(status_code=403, detail="Guest accounts cannot delete history")
         
    success = db_delete_prediction(db, prediction_id, user_id)
    if not success:
        raise HTTPException(status_code=404, detail="Prediction not found or unauthorized")
        
    return {"success": True, "deleted": prediction_id}


# ── DELETE /history (Clear All) ───────────────────────────────
@router.delete("")
async def clear_history_endpoint(
    user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Clear all history for the current user."""
    user_id = user.get("user_id")
    if not user_id or user.get("email") == "guest":
         raise HTTPException(status_code=403, detail="Guest accounts cannot delete history")
         
    db.query(Prediction).filter(Prediction.user_id == user_id).delete()
    db.commit()
    return {"success": True, "message": "All history cleared"}
