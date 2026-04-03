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

def _run_prediction(domain: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Run a standard prediction and return the raw result dict."""
    try:
        predictor = _predictor()
        result = predictor.predict(domain=domain, parameters=parameters)
        if isinstance(result, dict):
            return result
        return {"probability": float(result)}
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("Prediction failed in modes endpoint: %s", exc)
        return {"probability": 0.5, "warning": str(exc)}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/whatif")
async def whatif(payload: WhatIfRequest):
    """Run what-if scenario analysis by tweaking parameters."""
    base = _run_prediction(payload.domain, payload.parameters or {})
    base_prob = base.get("probability", payload.base_probability or 0.5)

    # Generate a small set of perturbations
    perturbations = []
    params = payload.parameters or {}
    for key, val in list(params.items())[:payload.depth]:
        try:
            tweaked = {**params}
            if isinstance(val, (int, float)):
                tweaked[key] = val * 1.1
                up = _run_prediction(payload.domain, tweaked)
                tweaked[key] = val * 0.9
                down = _run_prediction(payload.domain, tweaked)
                perturbations.append({
                    "parameter": key,
                    "original":  val,
                    "increase":  {"value": val * 1.1, "probability": up.get("probability")},
                    "decrease":  {"value": val * 0.9, "probability": down.get("probability")},
                })
        except Exception:
            pass

    return {
        "success": True,
        "mode": "what_if",
        "domain": payload.domain,
        "base_probability": base_prob,
        "perturbations": perturbations,
        "question": payload.question,
        "insight": (
            f"What-if analysis for {payload.domain}: base probability is "
            f"{base_prob:.1%}. Explored {len(perturbations)} parameter(s)."
        ),
    }


@router.post("/comparative")
async def comparative(payload: ComparativeRequest):
    """Compare multiple scenarios side-by-side."""
    results = []
    for scenario in payload.scenarios:
        label = scenario.get("label", "Scenario")
        params = {k: v for k, v in scenario.items() if k != "label"}
        prediction = _run_prediction(payload.domain, params)
        results.append({
            "label": label,
            "parameters": params,
            "probability": prediction.get("probability"),
            "prediction": prediction,
        })

    best = max(results, key=lambda r: r.get("probability") or 0) if results else None
    return {
        "success": True,
        "mode": "comparative",
        "domain": payload.domain,
        "scenarios": results,
        "best_scenario": best.get("label") if best else None,
        "question": payload.question,
    }


@router.post("/monitoring/start")
async def monitoring_start(payload: MonitoringStartRequest):
    """Initialise a live-monitoring session."""
    session_id = str(uuid.uuid4())
    base = _run_prediction(payload.domain, payload.parameters or {})
    session = {
        "session_id": session_id,
        "name": payload.name,
        "domain": payload.domain,
        "parameters": payload.parameters or {},
        "question": payload.question,
        "threshold_low": payload.threshold_low,
        "threshold_high": payload.threshold_high,
        "history": [{"probability": base.get("probability"), "event": "Session started"}],
        "current_probability": base.get("probability"),
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
    updated = _run_prediction(payload.domain, session["parameters"])
    prob = updated.get("probability")
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
    """Run adversarial stress-testing — probe worst/best-case inputs."""
    base = _run_prediction(payload.domain, payload.parameters or {})
    base_prob = base.get("probability", 0.5)

    stress_cases = []
    for label, factor in [("optimistic", 1.2), ("pessimistic", 0.8), ("extreme_low", 0.5), ("extreme_high", 1.5)]:
        tweaked = {k: (v * factor if isinstance(v, (int, float)) else v)
                   for k, v in (payload.parameters or {}).items()}
        result = _run_prediction(payload.domain, tweaked)
        stress_cases.append({
            "case": label,
            "factor": factor,
            "probability": result.get("probability"),
        })

    return {
        "success": True,
        "mode": "adversarial",
        "domain": payload.domain,
        "base_probability": base_prob,
        "stress_cases": stress_cases,
        "robustness": "stable" if all(
            abs((c.get("probability") or base_prob) - base_prob) < 0.15 for c in stress_cases
        ) else "sensitive",
        "question": payload.question,
    }


@router.post("/expert")
async def expert(payload: ExpertRequest):
    """Deep-dive expert analysis with feature importance."""
    prediction = _run_prediction(payload.domain, payload.parameters or {})

    feature_importance = []
    try:
        from core.predictor import SambhavPredictor
        p = SambhavPredictor()
        if hasattr(p, "explain"):
            feature_importance = p.explain(payload.domain, payload.parameters or {})
    except Exception:
        pass

    return {
        "success": True,
        "mode": "expert",
        "domain": payload.domain,
        "prediction": prediction,
        "probability": prediction.get("probability"),
        "feature_importance": feature_importance,
        "question": payload.question,
        "analysis": (
            f"Expert analysis for {payload.domain}: probability is "
            f"{prediction.get('probability', 0):.1%}."
        ),
    }


@router.post("/retrospective")
async def retrospective(payload: RetrospectiveRequest):
    """Analyse a historical case and compare actual vs predicted outcome."""
    prediction = _run_prediction(payload.domain, payload.parameters or {})
    prob = prediction.get("probability", 0.5)

    match = None
    if payload.outcome:
        if payload.outcome.lower() in ("positive", "success", "yes", "1", "true"):
            match = prob >= 0.5
        elif payload.outcome.lower() in ("negative", "failure", "no", "0", "false"):
            match = prob < 0.5

    return {
        "success": True,
        "mode": "retrospective",
        "domain": payload.domain,
        "description": payload.description,
        "actual_outcome": payload.outcome,
        "predicted_probability": prob,
        "prediction_aligned": match,
        "prediction": prediction,
        "insight": (
            f"Retrospective: model predicted {prob:.1%} confidence. "
            + (f"Prediction {'aligned' if match else 'did not align'} with actual outcome." if match is not None else "")
        ),
    }


@router.post("/simulation")
async def simulation(payload: SimulationRequest):
    """Monte-Carlo style simulation over parameter distributions."""
    import random

    base = _run_prediction(payload.domain, payload.parameters or {})
    base_prob = base.get("probability", 0.5)

    n = min(payload.n_runs or 500, 1000)
    probs = []
    params = payload.parameters or {}

    for _ in range(n):
        noisy = {
            k: (v + random.gauss(0, abs(v) * 0.05) if isinstance(v, (int, float)) else v)
            for k, v in params.items()
        }
        try:
            r = _run_prediction(payload.domain, noisy)
            p = r.get("probability")
            if p is not None:
                probs.append(float(p))
        except Exception:
            probs.append(base_prob)

    if not probs:
        probs = [base_prob]

    mean_p = sum(probs) / len(probs)
    sorted_p = sorted(probs)
    p5  = sorted_p[int(len(sorted_p) * 0.05)]
    p95 = sorted_p[int(len(sorted_p) * 0.95)]

    return {
        "success": True,
        "mode": "simulation",
        "domain": payload.domain,
        "n_runs": n,
        "base_probability": base_prob,
        "mean_probability": round(mean_p, 4),
        "confidence_interval_5": round(p5, 4),
        "confidence_interval_95": round(p95, 4),
        "std_dev": round((sum((p - mean_p) ** 2 for p in probs) / len(probs)) ** 0.5, 4),
        "question": payload.question,
    }


@router.post("/document")
async def document_analysis(
    file: UploadFile = File(...),
    domain: str = Form(...),
    question: Optional[str] = Form(None),
):
    """
    Advanced Document Analysis (Mode 5).
    Uses NVIDIA NIM Kimi K2.5 for 1M token context window (P.16).
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
        # Step 1 — Analyze document (P.03/P.17/P.18)
        analysis = analyze_document(tmp_path, domain)
        
        if "error" in analysis:
            raise HTTPException(status_code=400, detail=analysis["error"])

        # Step 2 — Check for insufficient info
        # Only flag insufficient if we literally found ZERO parameters
        params = analysis.get("parameters", {})
        if not params:
            return {
                "success": True,
                "insufficient_info": True,
                "reason": f"The document does not contain clear domain-relevant signals for {domain}. Please provide a document with more specific data points.",
                "missing_info": ["Specific values related to the domain's core parameters"],
                "analysis": analysis
            }

        # Step 3 — Predict using extracted parameters
        # Parallelize for speed
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
