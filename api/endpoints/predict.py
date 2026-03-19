import logging
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional
from core.predictor import predict, predict_free
from core.safety import check_hard_blocks, sanitize_input
from core.audit_system import run_full_audit
from core.reliability_index import compute as compute_reliability

router = APIRouter()
logger = logging.getLogger(__name__)

# ── Request schemas ───────────────────────────────────────────
class PredictRequest(BaseModel):
    domain:     str                  = Field(..., example="student")
    parameters: dict                 = Field(..., example={"study_hours":3})
    question:   Optional[str]        = None
    skipped:    Optional[list]       = []
    run_debate: Optional[bool]       = True
    mode:       Optional[str]        = "guided"

class FreeInferRequest(BaseModel):
    text:       str                  = Field(..., example="My startup has 3 engineers and $50k runway")
    n_outcomes: Optional[int]        = Field(5, ge=1, le=10)

# ── POST /predict ─────────────────────────────────────────────
@router.post("")
async def predict_endpoint(req: PredictRequest):
    """
    Main 7-stage Sambhav prediction pipeline.
    Runs ML + LLM independently, cross-validates, debates if needed.
    """
    logger.info(f"POST /predict domain={req.domain}")

    # Safety check on question
    if req.question:
        safety = check_hard_blocks(req.question)
        if not safety["safe"]:
            raise HTTPException(
                status_code=400,
                detail={"blocked": True,
                        "block_id": safety["block_id"],
                        "message":  safety["message"]}
            )

    # Sanitize parameters
    param_str = str(req.parameters)
    san = sanitize_input(param_str)
    if san["adversarial"]:
        raise HTTPException(
            status_code=400,
            detail={"blocked": True, "message": "Adversarial input detected"})

    try:
        result = predict(
            domain     = req.domain,
            parameters = req.parameters,
            question   = req.question,
            skipped    = req.skipped or [],
            run_debate = req.run_debate,
            mode       = req.mode,
        )
        return {
            "success":          True,
            "prediction":       result.to_dict(),
            "disclaimer":       "Sambhav may be incorrect. Always verify important decisions independently.",
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ── POST /predict/free ────────────────────────────────────────
@router.post("/free")
async def free_inference_endpoint(req: FreeInferRequest):
    """
    Free inference mode — no domain, no form.
    LLM generates N independent probability estimates for any text.
    """
    logger.info(f"POST /predict/free text={req.text[:50]}...")

    # Safety check
    safety = check_hard_blocks(req.text)
    if not safety["safe"]:
        raise HTTPException(
            status_code=400,
            detail={"blocked": True, "message": safety["message"]})

    try:
        result = predict_free(req.text, req.n_outcomes)
        return {
            "success":    True,
            "result":     result,
            "disclaimer": "Sambhav may be incorrect. Always verify important decisions independently.",
        }
    except Exception as e:
        logger.error(f"Free inference error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ── GET /predict/domains ──────────────────────────────────────
@router.get("/domains")
async def list_domains():
    """List all available prediction domains."""
    try:
        import yaml, os
        BASE = os.path.expanduser("~/Desktop/Sri_Coding/Project Sambhav")
        reg  = yaml.safe_load(open(f"{BASE}/schemas/domain_registry.yaml"))
        domains = {}
        for key, cfg in reg["domains"].items():
            domains[key] = {
                "name":        cfg.get("name", key),
                "description": cfg.get("description", ""),
                "parameters":  len(cfg.get("parameters", [])),
            }
        return {"success": True, "domains": domains}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── POST /predict/rich ────────────────────────────────────────
class PredictRichRequest(BaseModel):
    domain:     str           = Field(..., example="student")
    parameters: dict          = Field(..., example={"study_hours": 3})
    question:   Optional[str] = None
    skipped:    Optional[list] = []
    mode:       Optional[str] = "guided"

@router.post("/rich")
async def predict_rich_endpoint(req: PredictRichRequest):
    """
    Full rich prediction — ML + LLM + debate + SHAP + Monte Carlo +
    failure scenarios + improvement suggestions.
    Returns three detail levels: simple, detailed, full.
    """
    logger.info(f"POST /predict/rich domain={req.domain}")
    safety = check_hard_blocks(req.question or str(req.parameters))
    if not safety["safe"]:
        raise HTTPException(status_code=400, detail={"blocked": True, "message": safety["message"]})
    try:
        from core.predictor import predict_rich
        result = predict_rich(
            domain=req.domain, parameters=req.parameters,
            question=req.question, skipped=req.skipped, mode=req.mode
        )
        return {"success": True, "result": result,
                "disclaimer": "Sambhav may be incorrect. Always verify important decisions independently."}
    except Exception as e:
        logger.error(f"Rich prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── POST /predict/outcomes ────────────────────────────────────
class OutcomesRequest(BaseModel):
    domain:             str           = Field(..., example="student")
    parameters:         dict          = Field(..., example={"study_hours": 3})
    question:           Optional[str] = None
    n_outcomes:         Optional[int] = Field(5, ge=1, le=10)
    existing_outcomes:  Optional[list] = []
    mode:               Optional[str] = "independent"

@router.post("/outcomes")
async def generate_outcomes_endpoint(req: OutcomesRequest):
    """
    Multi-outcome generation — Section 8.3.
    Generate N independent outcomes. Call again with existing_outcomes for more.
    mode: independent | spectrum | conditional
    """
    logger.info(f"POST /predict/outcomes domain={req.domain} n={req.n_outcomes}")
    try:
        from core.predictor import generate_outcomes
        result = generate_outcomes(
            domain=req.domain, parameters=req.parameters,
            question=req.question, n_outcomes=req.n_outcomes,
            existing_outcomes=req.existing_outcomes, mode=req.mode
        )
        return {"success": True, "result": result,
                "disclaimer": "Sambhav may be incorrect. Always verify important decisions independently."}
    except Exception as e:
        logger.error(f"Outcomes error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── POST /predict/transparency ────────────────────────────────
class TransparencyRequest(BaseModel):
    domain:            str           = Field(..., example="student")
    parameters:        dict          = Field(..., example={"study_hours": 3})
    final_probability: float         = Field(..., example=0.637)
    question:          Optional[str] = None
    outcome:           Optional[str] = None

@router.post("/transparency")
async def transparency_endpoint(req: TransparencyRequest):
    """
    Section 8.4 — WHY this probability / WHY the minority case /
    WHEN minority would occur. Called on demand per prediction.
    """
    logger.info(f"POST /predict/transparency domain={req.domain} prob={req.final_probability}")
    try:
        from core.predictor import explain_prediction_transparency, _get_shap
        shap_vals = _get_shap(req.domain, req.parameters)
        result = explain_prediction_transparency(
            domain=req.domain, parameters=req.parameters,
            final_probability=req.final_probability,
            shap_values=shap_vals, question=req.question
        )
        return {"success": True, "result": result,
                "disclaimer": "Sambhav may be incorrect. Always verify important decisions independently."}
    except Exception as e:
        logger.error(f"Transparency error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── POST /predict/conversational ─────────────────────────────
class ConversationalStartRequest(BaseModel):
    domain:   str           = Field(..., example="student")
    question: Optional[str] = None

class ConversationalAnswerRequest(BaseModel):
    domain:      str  = Field(..., example="student")
    question:    Optional[str] = None
    param_key:   str  = Field(..., example="study_hours_per_day")
    value:       str  = Field(..., example="3-4 hours")
    skipped:     Optional[bool] = False
    step:        int  = Field(..., example=1)
    parameters:  Optional[dict] = {}

@router.post("/conversational/start")
async def conversational_start(req: ConversationalStartRequest):
    """Start a conversational session — returns first question."""
    try:
        from llm.conversational_mode import ConversationalSession
        session = ConversationalSession(req.domain, req.question)
        q = session.get_next_question()
        return {"success": True, "question": q, "session_domain": req.domain}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/conversational/answer")
async def conversational_answer(req: ConversationalAnswerRequest):
    """Submit answer and get next question or prediction-ready signal."""
    try:
        from llm.conversational_mode import ConversationalSession
        session = ConversationalSession(req.domain, req.question)
        session.parameters = req.parameters or {}
        session.step = req.step - 1
        session.reliability = len(session.parameters) / max(len(session.questions), 1)
        state = session.submit_answer(req.param_key, req.value, req.skipped or False)
        result = {"success": True, "state": state, "parameters": session.parameters}
        if state["complete"]:
            result["prediction_ready"] = session.get_prediction_ready()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
