"""
api/endpoints/modes.py
Operating-mode endpoints for Project Sambhav.

Routes
------
POST /modes/whatif            — What-if scenario analysis
POST /modes/comparative       — Multi-scenario comparison
POST /modes/monitoring/start  — Start a live-monitoring session
POST /modes/monitoring/update — Push an update to a monitoring session
POST /modes/adversarial       — Adversarial stress-test
POST /modes/expert            — Expert / deep-dive analysis
POST /modes/retrospective     — Retrospective case analysis
POST /modes/simulation        — Monte-Carlo simulation
POST /modes/document          — Document-based prediction (multipart)
"""

import uuid
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

logger = logging.getLogger(__name__)
router = APIRouter()

# ---------------------------------------------------------------------------
# Auth helper (reuse the same token dependency as other endpoints)
# ---------------------------------------------------------------------------
try:
    from api.endpoints.auth import get_current_user  # may be optional
    _auth_dep = Depends(get_current_user)
except Exception:
    _auth_dep = None  # auth disabled, open endpoints


def _predictor():
    """Lazy-import so the module loads even if core isn't ready yet."""
    try:
        from core.predictor import SambhavPredictor
        return SambhavPredictor()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Predictor unavailable: {exc}")


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class WhatIfRequest(BaseModel):
    domain: str
    parameters: Optional[Dict[str, Any]] = {}
    question: Optional[str] = None
    base_probability: Optional[float] = None
    depth: Optional[int] = 3


class Scenario(BaseModel):
    label: str
    model_config = {"extra": "allow"}


class ComparativeRequest(BaseModel):
    domain: str
    scenarios: List[Dict[str, Any]]
    outcomes: Optional[List[str]] = None
    question: Optional[str] = None


class MonitoringStartRequest(BaseModel):
    name: str
    domain: str
    parameters: Optional[Dict[str, Any]] = {}
    question: Optional[str] = None
    threshold_low: Optional[float] = 0.2
    threshold_high: Optional[float] = 0.8


class MonitoringUpdateRequest(BaseModel):
    session_id: str
    domain: str
    parameters: Dict[str, Any]
    update_text: Optional[str] = None
    question: Optional[str] = None


class AdversarialRequest(BaseModel):
    domain: str
    parameters: Optional[Dict[str, Any]] = {}
    question: Optional[str] = None


class ExpertRequest(BaseModel):
    domain: str
    parameters: Optional[Dict[str, Any]] = {}
    question: Optional[str] = None


class RetrospectiveRequest(BaseModel):
    domain: str
    description: str
    outcome: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = {}


class SimulationRequest(BaseModel):
    domain: str
    parameters: Optional[Dict[str, Any]] = {}
    question: Optional[str] = None
    n_runs: Optional[int] = 500


# ---------------------------------------------------------------------------
# In-memory monitoring session store (lightweight; replace with DB if needed)
# ---------------------------------------------------------------------------
_monitoring_sessions: Dict[str, Dict] = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_prediction(domain: str, parameters: Dict[str, Any], question: str = None) -> Dict[str, Any]:
    """Run a standard prediction and return the full result dict."""
    try:
        from core.predictor import predict
        result = predict(domain=domain, parameters=parameters, question=question)
        return result.to_dict()
    except Exception as exc:
        logger.warning("Prediction failed in modes endpoint: %s", exc)
        return {"final_probability": 0.5, "warning": str(exc), "outcomes": {"outcome": 0.5}}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/whatif")
async def whatif(payload: WhatIfRequest):
    """Run what-if scenario analysis with narrative simulation."""
    base_result = _run_prediction(payload.domain, payload.parameters or {}, payload.question)
    base_prob = base_result.get("final_probability", 0.5)

    from llm.outcome_simulation import simulate_both_outcomes
    from core.predictor import _load_registry
    
    reg = _load_registry()
    domain_cfg = reg.get(payload.domain, {})
    supported = domain_cfg.get("supported_outcomes", ["Outcome A", "Outcome B"])
    
    simulation = simulate_both_outcomes(
        domain=payload.domain,
        parameters=payload.parameters or {},
        question=payload.question or "What if this scenario unfolds?",
        final_probability=base_prob,
        positive_outcome=supported[0],
        negative_outcome=supported[1] if len(supported) > 1 else None
    )

    return {
        "success": True,
        "mode": "what_if",
        "domain": payload.domain,
        "base_probability": base_prob,
        "simulation": simulation,
        "outcomes": base_result.get("outcomes", []),
        "question": payload.question,
    }


@router.post("/comparative")
async def comparative(payload: ComparativeRequest):
    """Compare multiple scenarios side-by-side using comparative inference."""
    from llm.comparative_inference import compare_scenarios
    
    result = compare_scenarios(
        domain=payload.domain,
        scenarios=payload.scenarios,
        outcomes=payload.outcomes,
        question=payload.question
    )

    return {
        "success": True,
        "mode": "comparative",
        "result": result,
    }


@router.post("/monitoring/start")
async def monitoring_start(payload: MonitoringStartRequest):
    """Initialise a live-monitoring session."""
    session_id = str(uuid.uuid4())
    base = _run_prediction(payload.domain, payload.parameters or {}, payload.question)
    session = {
        "session_id": session_id,
        "name": payload.name,
        "domain": payload.domain,
        "parameters": payload.parameters or {},
        "question": payload.question,
        "threshold_low": payload.threshold_low,
        "threshold_high": payload.threshold_high,
        "history": [{"probability": base.get("final_probability"), "event": "Session started"}],
        "current_probability": base.get("final_probability"),
        "status": "active",
    }
    _monitoring_sessions[session_id] = session
    return {"success": True, "mode": "monitoring", **session}


@router.post("/monitoring/update")
async def monitoring_update(payload: MonitoringUpdateRequest):
    """Push a parameter update to an existing monitoring session."""
    session = _monitoring_sessions.get(payload.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Monitoring session not found.")

    session["parameters"].update(payload.parameters)
    updated = _run_prediction(payload.domain, session["parameters"], payload.question)
    prob = updated.get("final_probability")
    session["current_probability"] = prob
    session["history"].append({
        "probability": prob,
        "event": payload.update_text or "Parameters updated",
        "parameters": payload.parameters,
    })

    alert = None
    if prob is not None:
        if prob < (session.get("threshold_low") or 0.2):
            alert = "below_low_threshold"
        elif prob > (session.get("threshold_high") or 0.8):
            alert = "above_high_threshold"

    return {
        "success": True,
        "mode": "monitoring",
        "session_id": payload.session_id,
        "current_probability": prob,
        "alert": alert,
        "history": session["history"],
        "prediction": updated,
    }


@router.post("/adversarial")
async def adversarial(payload: AdversarialRequest):
    """Adversarial stress-test using Devil's Advocate."""
    from llm.multi_agent import run_devils_advocate
    from core.predictor import predict
    
    # Run normal prediction first
    base_result = predict(domain=payload.domain, parameters=payload.parameters or {}, question=payload.question)
    
    # Challenge it
    devil = run_devils_advocate(
        domain=payload.domain,
        parameters=payload.parameters or {},
        question=payload.question,
        dominant_prob=base_result.reconciled_probability
    )

    return {
        "success": True,
        "mode": "adversarial",
        "base_prediction": base_result.to_dict(),
        "devils_advocate": devil,
        "outcomes": base_result.outcomes,
    }


@router.post("/expert")
async def expert(payload: ExpertRequest):
    """Expert consultation mode using 4-agent debate."""
    from llm.multi_agent import run_debate
    
    debate_result = run_debate(
        domain=payload.domain,
        parameters=payload.parameters or {},
        question=payload.question or "Expert analysis required"
    )

    return {
        "success": True,
        "mode": "expert",
        "debate": debate_result,
        "final_probability": debate_result["final_probability"],
    }


@router.post("/retrospective")
async def retrospective(payload: RetrospectiveRequest):
    """Retrospective case analysis — simulate how a past outcome unfolded."""
    from llm.outcome_simulation import generate_outcome_story
    
    story = generate_outcome_story(
        domain=payload.domain,
        parameters=payload.parameters or {},
        outcome=payload.outcome or "Past event",
        probability=1.0, # Retrospective assumes it happened
        question=payload.description
    )

    return {
        "success": True,
        "mode": "retrospective",
        "story": story,
    }


@router.post("/simulation")
async def simulation(payload: SimulationRequest):
    """Monte-Carlo simulation of outcomes."""
    from core.predictor import predict
    
    # Run prediction to get base probabilities
    result = predict(domain=payload.domain, parameters=payload.parameters or {}, question=payload.question)
    
    # In a real MC simulation, we'd perturb inputs, but here we'll use the LLM to simulate the spread
    from llm.outcome_simulation import simulate_both_outcomes
    
    sim = simulate_both_outcomes(
        domain=payload.domain,
        parameters=payload.parameters or {},
        question=payload.question,
        final_probability=result.reconciled_probability
    )

    return {
        "success": True,
        "mode": "simulation",
        "base_result": result.to_dict(),
        "simulation": sim,
        "outcomes": result.outcomes,
    }


@router.post("/document")
async def document_analysis(
    domain: str = Form(...),
    file: UploadFile = File(...),
    question: Optional[str] = Form(None),
    user=Depends(get_current_user)
):
    """
    Advanced Document Analysis (Mode 5).
    Uses NVIDIA NIM for high-accuracy parameter extraction.
    """
    import os, tempfile
    from vision.document_pipeline import analyze_document
    from core.predictor import predict, generate_outcomes
    
    # Save to temp file for processing
    with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Step 1 — Analyze document
        analysis = analyze_document(tmp_path, domain)
        
        if "error" in analysis:
            raise HTTPException(status_code=400, detail=analysis["error"])

        # Step 2 — Check for insufficient info
        params = analysis.get("parameters", {})
        if not params and domain != "sarvagna": # Sarvagna usually has text_input
            return {
                "success": True,
                "insufficient_info": True,
                "reason": f"The document does not contain clear domain-relevant signals for {domain}.",
                "missing_info": ["Core parameter values"],
                "analysis": analysis
            }

        # Step 3 — Predict using extracted parameters
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            pred_future = executor.submit(predict, domain=domain, params=params, question=question or analysis.get("prediction_question"))
            out_future  = executor.submit(generate_outcomes, domain=domain, parameters=params, question=question or analysis.get("prediction_question"))
            
            prediction = pred_future.result()
            outcomes   = out_future.result()

        return {
            "success": True,
            "mode": "document",
            "domain": domain,
            "filename": file.filename,
            "analysis": analysis,
            "prediction": prediction.to_dict(),
            "outcomes": outcomes.get("outcomes", []),
            "insufficient_info": False
        }
    except Exception as e:
        logger.error(f"Document analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
