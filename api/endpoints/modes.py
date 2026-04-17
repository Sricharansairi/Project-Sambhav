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
    base_pct = round(base_prob * 100, 1)

    from llm.outcome_simulation import simulate_both_outcomes
    from core.predictor import _load_registry
    import json, re

    reg = _load_registry()
    domain_cfg = reg.get(payload.domain, {})
    supported = domain_cfg.get("supported_outcomes", ["Positive Outcome", "Negative Outcome"])

    simulation = simulate_both_outcomes(
        domain=payload.domain,
        parameters=payload.parameters or {},
        question=payload.question or "What if this scenario unfolds?",
        final_probability=base_prob,
        positive_outcome=supported[0],
        negative_outcome=supported[1] if len(supported) > 1 else None
    )

    # Build the tree structure the frontend expects
    branches = []
    sim_data = simulation if isinstance(simulation, dict) else {}

    # Branch 1: Positive outcome scenario
    pos_story = sim_data.get("positive_story") or sim_data.get("positive_outcome") or sim_data.get("story_a") or ""
    neg_story = sim_data.get("negative_story") or sim_data.get("negative_outcome") or sim_data.get("story_b") or ""

    if pos_story or base_pct > 0:
        branches.append({
            "event": f"If {supported[0]} occurs ({base_pct}%)",
            "new_probability": base_pct,
            "probability_shift": 0,
            "reasoning": pos_story,
            "children": []
        })

    neg_pct = round(100 - base_pct, 1)
    if neg_story or neg_pct > 0:
        branches.append({
            "event": f"If {supported[1] if len(supported) > 1 else 'other outcome'} occurs ({neg_pct}%)",
            "new_probability": neg_pct,
            "probability_shift": round(neg_pct - base_pct, 1),
            "reasoning": neg_story,
            "children": []
        })

    # Generate LLM-based scenario branches for the question
    try:
        from llm.router import route
        q = payload.question or f"What if this {payload.domain} scenario changes?"
        msgs = [
            {"role": "system", "content": (
                f"You are an expert in {payload.domain} prediction. Generate 2-3 specific what-if scenario branches."
                f" Base probability: {base_pct}%. Question: {q}"
                " Respond ONLY as JSON: {\"branches\": [{\"event\": \"...\", \"probability_shift\": 15, \"reasoning\": \"...\"}]}"
            )},
            {"role": "user", "content": f"Generate what-if branches for: {q}"}
        ]
        raw = route("conversational", msgs, max_tokens=400, temperature=0.25)
        txt = raw.get("content", "")
        txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.DOTALL).strip()
        if "{" in txt:
            d = json.loads(txt[txt.find("{"):txt.rfind("}")+1])
            for b in d.get("branches", []):
                shift = float(b.get("probability_shift", 0))
                branches.append({
                    "event": b.get("event", "Scenario"),
                    "new_probability": min(99, max(1, round(base_pct + shift, 1))),
                    "probability_shift": round(shift, 1),
                    "reasoning": b.get("reasoning", ""),
                    "children": []
                })
    except Exception as e:
        logger.warning(f"WhatIf branch generation failed: {e}")

    tree = {
        "base_probability": base_pct,
        "description": payload.question or f"{payload.domain} scenario analysis",
        "branches": branches
    }

    return {
        "success": True,
        "mode": "what_if",
        "domain": payload.domain,
        "base_probability": base_prob,
        "tree": tree,
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
    """Retrospective case analysis — analyse why a past outcome occurred."""
    import json, re

    raw_story = None
    try:
        from llm.outcome_simulation import generate_outcome_story
        raw_story = generate_outcome_story(
            domain=payload.domain,
            parameters=payload.parameters or {},
            outcome=payload.outcome or "Past event",
            probability=1.0,
            question=payload.description
        )
    except Exception as e:
        logger.warning(f"generate_outcome_story failed: {e}")

    # Build structured forensic analysis via LLM
    try:
        from llm.router import route
        msgs = [
            {"role": "system", "content": (
                f"You are a forensic analyst for the {payload.domain} domain. "
                "Analyse why a past event occurred. Respond ONLY as JSON with these exact keys: "
                '{"probability_at_time": <number 0-100>, "root_cause": "...", '
                '"prevention_point": "...", "contributing_factors": ["..."], "lessons_learned": [".."]}, '
                'do not include additional keys.'
            )},
            {"role": "user", "content": (
                f"Event description: {payload.description}\n"
                f"Actual outcome: {payload.outcome or 'Unspecified'}\n"
                f"Domain: {payload.domain}\n"
                f"Known parameters: {json.dumps(payload.parameters or {})}\n"
                "Perform retrospective analysis."
            )}
        ]
        raw = route("llm_predict", msgs, max_tokens=500, temperature=0.2)
        txt = raw.get("content", "")
        txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.DOTALL).strip()
        if "{" in txt:
            analysis = json.loads(txt[txt.find("{"):txt.rfind("}")+1])
        else:
            raise ValueError("no JSON")
    except Exception as e:
        logger.warning(f"Retrospective LLM analysis failed: {e}")
        analysis = {
            "probability_at_time": 50,
            "root_cause": raw_story if raw_story else f"The {payload.outcome or 'event'} occurred due to a confluence of factors described in the scenario.",
            "prevention_point": "An earlier intervention could have altered the trajectory at the critical decision point.",
            "contributing_factors": ["Primary environmental conditions", "Behavioural patterns", "External pressures"],
            "lessons_learned": ["Monitor early warning signals", "Adjust parameters proactively"]
        }

    return {
        "success": True,
        "mode": "retrospective",
        "analysis": analysis,
        "story": raw_story,
    }


@router.post("/simulation")
async def simulation(payload: SimulationRequest):
    """Monte-Carlo simulation of outcomes."""
    import random, math
    from core.predictor import predict, generate_outcomes

    # Run base prediction
    result = predict(domain=payload.domain, parameters=payload.parameters or {}, question=payload.question)
    base_prob = result.reconciled_probability or 0.5
    n_runs = payload.n_runs or 500

    # Monte-Carlo: perturb base_prob with Gaussian noise to simulate uncertainty
    runs = []
    for _ in range(n_runs):
        noise = random.gauss(0, 0.08)  # ±8% std deviation
        p = max(0.01, min(0.99, base_prob + noise))
        runs.append(p)

    mean_p   = sum(runs) / len(runs)
    variance = sum((r - mean_p) ** 2 for r in runs) / len(runs)
    std_dev  = math.sqrt(variance)
    sorted_r = sorted(runs)
    ci_low   = sorted_r[int(0.025 * n_runs)]
    ci_high  = sorted_r[int(0.975 * n_runs)]
    stability = std_dev  # lower = more stable

    monte_carlo = {
        "mean":      round(mean_p * 100, 1),
        "ci_low":    round(ci_low * 100, 1),
        "ci_high":   round(ci_high * 100, 1),
        "stability": round(stability, 3),
        "n_runs":    n_runs,
    }

    # LLM narrative
    try:
        from llm.outcome_simulation import simulate_both_outcomes
        sim = simulate_both_outcomes(
            domain=payload.domain,
            parameters=payload.parameters or {},
            question=payload.question,
            final_probability=base_prob
        )
        narrative = {"story": sim.get("positive_story") or sim.get("story_a") or ""} if isinstance(sim, dict) else {"story": ""}
    except Exception as e:
        logger.warning(f"Simulation narrative failed: {e}")
        sim = {}
        narrative = {"story": f"Based on {n_runs} Monte Carlo runs, the mean probability is {monte_carlo['mean']}% with a 95% CI of {monte_carlo['ci_low']}%–{monte_carlo['ci_high']}%."}

    # Generate outcomes
    try:
        outcomes_res = generate_outcomes(domain=payload.domain, parameters=payload.parameters or {}, question=payload.question)
        outcomes_list = outcomes_res.get("outcomes", [])
    except Exception:
        outcomes_list = result.outcomes or []

    return {
        "success":      True,
        "mode":         "simulation",
        "monte_carlo":  monte_carlo,
        "narrative":    narrative,
        "base_result":  result.to_dict(),
        "simulation":   sim,
        "outcomes":     outcomes_list,
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
        _q = question or analysis.get("prediction_question")
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            pred_future = executor.submit(predict, domain=domain, parameters=params, question=_q)
            out_future  = executor.submit(generate_outcomes, domain=domain, parameters=params, question=_q)

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
