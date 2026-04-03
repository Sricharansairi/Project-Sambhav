import logging
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from typing import Optional
from sqlalchemy.orm import Session

from core.predictor import predict, predict_free, generate_outcomes, explain_prediction_transparency, _get_shap, _load_registry
from core.safety import check_hard_blocks, sanitize_input
from db.models import get_db
from db.database import save_prediction, log_event
from api.endpoints.auth import get_current_user

router = APIRouter()
logger = logging.getLogger(__name__)

# ── Valid domain IDs (must match domain_registry.yaml keys exactly) ──
VALID_DOMAINS = [
    "student", "high_school", "job_life", "health", "fitness",
    "financial", "mental_health", "pragma", "behavioral", "claim", "sarvagna"
]

# ── Input quality gate ────────────────────────────────────────
def _validate_input_quality(question: Optional[str], parameters: dict, domain: str) -> tuple[bool, str]:
    if not question and not parameters:
        return False, "Please provide a question or parameters to analyze."
    if question:
        q = question.strip()
        words = q.split()
        if len(q) < 8:
            return False, "Your question is too short. Please describe what you want to predict in more detail."
        if len(words) < 2:
            return False, f"Please provide a meaningful question (at least 2 words). Single words like '{q}' cannot be analyzed."
        nonsense = {"hello", "hi", "hey", "test", "ok", "okay", "yes", "no", "lol",
                    "haha", "what", "help", "hmm", "uh", "um", "hmmmm", "idk"}
        if q.lower().strip("?!. ") in nonsense:
            return False, f"'{q}' is not a predictable scenario. Please describe a specific situation."
    if domain not in VALID_DOMAINS:
        return False, f"Domain '{domain}' is not supported. Valid domains: {', '.join(VALID_DOMAINS)}"
    return True, "ok"


# ── Request schemas ───────────────────────────────────────────
class PredictRequest(BaseModel):
    domain:     str             = Field(..., example="student")
    parameters: dict            = Field(default={}, example={"study_hours": 3})
    question:   Optional[str]   = None
    skipped:    Optional[list]  = []
    run_debate: Optional[bool]  = True
    mode:       Optional[str]   = "guided"

class FreeInferRequest(BaseModel):
    text:       str             = Field(..., example="My startup has 3 engineers and $50k runway")
    n_outcomes: Optional[int]   = Field(5, ge=1, le=10)

class OutcomesRequest(BaseModel):
    domain:            str            = Field(..., example="student")
    parameters:        dict           = Field(default={})
    question:          Optional[str]  = None
    n_outcomes:        Optional[int]  = Field(5, ge=1, le=10)
    existing_outcomes: Optional[list] = []
    mode:              Optional[str]  = "independent"

class TransparencyRequest(BaseModel):
    domain:            str            = Field(..., example="student")
    parameters:        dict           = Field(default={})
    final_probability: Optional[float]= Field(None, example=0.637)
    question:          Optional[str]  = None
    outcome:           Optional[str]  = None
    level:             Optional[str]  = "detailed"   # ← NEW: simple | detailed | full

class ConversationalStartRequest(BaseModel):
    domain:   str           = Field(..., example="student")
    question: Optional[str] = None

class ConversationalAnswerRequest(BaseModel):
    domain:     str             = Field(..., example="student")
    question:   Optional[str]   = None
    param_key:  str             = Field(..., example="study_hours_per_day")
    value:      str             = Field(..., example="3-4 hours")
    skipped:    Optional[bool]  = False
    step:       int             = Field(..., example=1)
    parameters: Optional[dict]  = {}
    history:    Optional[list]  = []

class BatchPredictRequest(BaseModel):
    domain:     str        = Field(..., example="student")
    batch_data: list[dict] = Field(..., example=[{"study_hours": 3}])


# ── GET /predict/domains ──────────────────────────────────────
@router.get("/domains")
async def get_domains():
    """
    Returns all 11 registered domains with their full parameter schemas.
    Used by the frontend ChipParameterModal to build dynamic parameter collection.

    FIX: Previously read p.get("name","") — YAML uses 'key', not 'name'.
    FIX: Previously passed raw options list — chips were showing [object Object].
         Now returns options as [{label, value}] for correct chip rendering.
    FIX: Added 'values' array alongside 'options' for clean value lookup.
    """
    try:
        registry = _load_registry()
        result = {}
        for domain_key, cfg in registry.items():
            if domain_key not in VALID_DOMAINS:
                continue

            params_raw = cfg.get("parameters", [])
            params_out = {}

            for p in params_raw:
                if not isinstance(p, dict):
                    continue

                # ── FIX BUG 1: YAML uses 'key', not 'name' ───────────────
                param_key = p.get("key") or p.get("name") or ""
                if not param_key:
                    continue   # skip malformed entries

                # Build options — YAML options are [{label: "...", value: ...}]
                # We return them as-is so ChipParameterModal can display label + submit value
                raw_options = p.get("options", [])

                # Normalise to [{label, value}] regardless of source format
                normalised_options = []
                for opt in raw_options:
                    if isinstance(opt, dict):
                        normalised_options.append({
                            "label": str(opt.get("label", opt.get("value", ""))),
                            "value": opt.get("value", opt.get("label", "")),
                        })
                    else:
                        normalised_options.append({"label": str(opt), "value": opt})

                params_out[param_key] = {
                    "type":        p.get("type", "categorical"),
                    # ── FIX: use explicit label from YAML, not generated from key ──
                    "label":       p.get("label", param_key.replace("_", " ").title()),
                    "description": p.get("description", ""),
                    "options":     normalised_options,   # [{label, value}] for ChipParameterModal
                    "range":       p.get("range", []),
                    "weight":      p.get("weight", "medium"),
                    "required":    p.get("required", False),
                    "placeholder": p.get("placeholder", ""),
                }

            result[domain_key] = {
                "name":              cfg.get("display_name", cfg.get("name", domain_key)),
                "description":       cfg.get("description", ""),
                "prediction_label":  cfg.get("prediction_label", "Probability"),
                "disclaimer":        cfg.get("disclaimer"),
                "brier_score":       cfg.get("brier_score"),
                "auc":               cfg.get("auc"),
                "status":            cfg.get("status", "ACTIVE"),
                "parameters":        params_out,
            }
        return result
    except Exception as e:
        logger.error(f"get_domains failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── POST /predict ─────────────────────────────────────────────
@router.post("")
async def predict_endpoint(
    req: PredictRequest,
    db: Session = Depends(get_db),
    user: dict  = Depends(get_current_user)
):
    logger.info(f"POST /predict domain={req.domain}")

    valid, reason = _validate_input_quality(req.question, req.parameters or {}, req.domain)
    if not valid:
        raise HTTPException(status_code=422, detail={"error": reason, "type": "input_quality"})

    if req.question:
        safety = check_hard_blocks(req.question)
        if not safety["safe"]:
            raise HTTPException(status_code=400, detail={"blocked": True, "block_id": safety["block_id"], "message": safety["message"]})

    params = req.parameters or {}
    san = sanitize_input(str(params))
    if san["adversarial"]:
        log_event(db, "adversarial_input", user_id=user.get("user_id"), domain=req.domain, details={"parameters": str(params)})
        raise HTTPException(status_code=400, detail={"blocked": True, "message": "Adversarial input detected"})

    study = params.get("study_hours_per_day", params.get("study_hours", None))
    sleep = params.get("sleep_hours", None)
    if study is not None and sleep is not None:
        try:
            if float(study) >= 10 and float(sleep) <= 2:
                raise HTTPException(status_code=400, detail={
                    "blocked": True, "flag": "ABN-001",
                    "message": "Physiologically impossible combination: study≥10h + sleep≤2h. Please review your parameters."
                })
        except (TypeError, ValueError):
            pass

    try:
        result = predict(
            domain     = req.domain,
            parameters = req.parameters or {},
            question   = req.question,
            skipped    = req.skipped or [],
            run_debate = req.run_debate,
            mode       = req.mode,
            user_id    = user.get("user_id"),
            db         = db
        )
        pred_record = save_prediction(
            db, user.get("user_id"),
            {"domain": req.domain, "question": req.question,
             "raw_parameters": req.parameters, **result.to_dict(), "mode": req.mode}
        )
        return {
            "success":       True,
            "prediction_id": pred_record.prediction_id,
            "prediction":    result.to_dict(),
            "disclaimer":    "Sambhav may be incorrect. Always verify important decisions independently.",
        }
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── POST /predict/free ────────────────────────────────────────
@router.post("/free")
async def free_inference_endpoint(req: FreeInferRequest):
    logger.info(f"POST /predict/free text={req.text[:50]}")
    q = req.text.strip()
    if len(q) < 8 or len(q.split()) < 2:
        raise HTTPException(status_code=422, detail={
            "error": f"'{q}' is too vague for free inference. Please describe a specific scenario.",
            "type":  "input_quality"
        })
    safety = check_hard_blocks(req.text)
    if not safety["safe"]:
        raise HTTPException(status_code=400, detail={"blocked": True, "message": safety["message"]})
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


# ── POST /predict/outcomes ────────────────────────────────────
@router.post("/outcomes")
async def generate_outcomes_endpoint(req: OutcomesRequest):
    logger.info(f"POST /predict/outcomes domain={req.domain} n={req.n_outcomes}")
    valid, reason = _validate_input_quality(req.question, req.parameters or {}, req.domain)
    if not valid:
        raise HTTPException(status_code=422, detail={"error": reason, "type": "input_quality"})
    try:
        result = generate_outcomes(
            domain            = req.domain,
            parameters        = req.parameters or {},
            question          = req.question,
            n_outcomes        = req.n_outcomes,
            existing_outcomes = req.existing_outcomes,
            mode              = req.mode,
        )
        return {
            "success":    True,
            "result":     result,
            "disclaimer": "Probabilities are independent and do not sum to 100%.",
        }
    except Exception as e:
        logger.error(f"Outcomes error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── POST /predict/transparency ────────────────────────────────
@router.post("/transparency")
async def transparency_endpoint(req: TransparencyRequest):
    """
    FIX: Now accepts 'level' field (simple | detailed | full) and
    passes it through to explain_prediction_transparency so the WHY
    button respects the TransparencyToggle value.
    FIX: Now correctly handles transparency for any outcome in multi-outcome results.
    """
    logger.info(f"POST /predict/transparency domain={req.domain} outcome={req.outcome} level={req.level}")
    try:
        # Use simple SHAP logic
        shap_vals = _get_shap(req.domain, req.parameters or {})

        # Probability for the specific outcome requested
        prob = req.final_probability
        
        # Build question that contextualises the specific outcome
        question = req.question
        if req.outcome:
            # If outcome label is provided, the LLM should explain THAT specific outcome
            question = (
                f"Why does the outcome '{req.outcome}' have a {round((prob or 0.5)*100, 1)}% probability? "
                f"Context: {req.question or 'No additional context provided.'}"
            )

        result = explain_prediction_transparency(
            domain            = req.domain,
            parameters        = req.parameters or {},
            final_probability = prob,
            shap_values       = shap_vals,
            question          = question,
            outcome           = req.outcome
        )

        # Filter output to only the requested level to save tokens
        level = req.level or "detailed"
        if level == "simple":
            filtered = {"simple": result.get("simple", {})}
        elif level == "full":
            filtered = result   # all three levels
        else:
            filtered = {
                "simple":   result.get("simple", {}),
                "detailed": result.get("detailed", {}),
            }

        return {
            "success":    True,
            "result":     filtered,
            "level":      level,
            "outcome":    req.outcome,
            "disclaimer": "Sambhav may be incorrect. Always verify important decisions independently.",
        }
    except Exception as e:
        logger.error(f"Transparency error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── POST /predict/conversational/start ───────────────────────
@router.post("/conversational/start")
async def conversational_start(req: ConversationalStartRequest):
    try:
        from llm.conversational_mode import ConversationalSession
        session = ConversationalSession(req.domain, req.question)
        q = session.get_next_question()
        if q is None:
            # Maybe the domain or question is already enough
            return {
                "success": True, 
                "complete": True, 
                "message": "I already have enough information to generate a prediction.",
                "session_domain": req.domain
            }
        return {"success": True, "question": q, "session_domain": req.domain}
    except Exception as e:
        logger.error(f"Conversational start error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start conversation: {str(e)}")


# ── POST /predict/conversational/answer ──────────────────────
@router.post("/conversational/answer")
async def conversational_answer(req: ConversationalAnswerRequest):
    try:
        from llm.conversational_mode import ConversationalSession
        session = ConversationalSession(req.domain, req.question)
        session.parameters = req.parameters or {}
        session.history = req.history or []
        session.step = req.step - 1
        
        # In dynamic mode, reliability is returned by the LLM in submit_answer -> get_next_question
        state = session.submit_answer(req.param_key, req.value, req.skipped or False)
        
        if state is None:
             raise ValueError("Failed to process your answer. Please try again.")

        return {
            "success": True, 
            "state": state, 
            "parameters": session.parameters,
            "history": session.history,
            "prediction_ready": session.get_prediction_ready() if state.get("complete") else None
        }
    except Exception as e:
        logger.error(f"Conversational answer error: {e}")
        raise HTTPException(status_code=500, detail=f"Conversation error: {str(e)}")


# ── POST /predict/discover-params ─────────────────────────────
class DiscoverParamsRequest(BaseModel):
    domain:   str = Field(..., example="student")
    question: str = Field(..., example="Will I pass the math exam if I study 4 hours?")

@router.post("/discover-params")
async def discover_params_endpoint(req: DiscoverParamsRequest):
    """
    DYNAMIC PARAMETER GENERATION (Section 6.1)
    Generates relevant chip parameters on-the-fly based on the user's specific question.
    Ensures parameters are always relevant to the current scenario.
    """
    try:
        from llm.router import route
        import json, re
        
        registry = _load_registry()
        domain_cfg = registry.get(req.domain, {})
        
        # System prompt for dynamic parameter generation
        messages = [
            {"role": "system", "content": (
                f"You are the Sambhav Dynamic Parameter Generator for the '{req.domain}' domain.\n"
                f"User Question: \"{req.question}\"\n\n"
                "YOUR TASK:\n"
                "Generate 4-6 most relevant parameters (features) needed to predict the outcome of this specific scenario.\n"
                "Each parameter must be quantifiable or categorical.\n\n"
                "Respond ONLY in this JSON format:\n"
                "{\n"
                '  "parameters": [\n'
                '    {\n'
                '      "key": "unique_snake_case_key",\n'
                '      "label": "User friendly question?",\n'
                '      "type": "chips",\n'
                '      "options": [\n'
                '        {"label": "Option 1", "value": 1.0},\n'
                '        {"label": "Option 2", "value": 2.0}\n'
                '      ],\n'
                '      "description": "Short explanation of impact",\n'
                '      "weight": "high|medium|low",\n'
                '      "required": true\n'
                '    }\n'
                '  ]\n'
                "}"
            )},
            {"role": "user", "content": f"Generate dynamic parameters for: {req.question}"}
        ]
        
        # Use high-speed Groq 8B for fast generation
        raw_res = route("conversational", messages, max_tokens=1000, temperature=0.2)
        raw = raw_res.get("content", "")
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        if "```" in raw:
            raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`")
            
        data = json.loads(raw[raw.find("{"):raw.rfind("}")+1])
        return {"success": True, "parameters": data.get("parameters", [])}
        
    except Exception as e:
        logger.error(f"Discover params failed: {e}")
        # Fallback to static registry params if generation fails
        return {"success": False, "error": str(e)}


# ── POST /predict/rich ────────────────────────────────────────
class PredictRichRequest(BaseModel):
    domain:     str           = Field(..., example="student")
    parameters: dict          = Field(default={})
    question:   Optional[str] = None
    skipped:    Optional[list]= []
    mode:       Optional[str] = "guided"

@router.post("/rich")
async def predict_rich_endpoint(
    req: PredictRichRequest,
    db: Session = Depends(get_db),
    user: dict  = Depends(get_current_user)
):
    logger.info(f"POST /predict/rich domain={req.domain}")
    safety = check_hard_blocks(req.question or str(req.parameters))
    if not safety["safe"]:
        raise HTTPException(status_code=400, detail={"blocked": True, "message": safety["message"]})
    try:
        from core.predictor import predict_rich
        result = predict_rich(domain=req.domain, parameters=req.parameters or {},
                              question=req.question, skipped=req.skipped, mode=req.mode,
                              user_id=user.get("user_id"), db=db)
        pred_record = save_prediction(
            db, user.get("user_id"),
            {"domain": req.domain, "question": req.question,
             "raw_parameters": req.parameters, **result, "mode": req.mode}
        )
        log_event(db, "predict_rich", user_id=user.get("user_id"), domain=req.domain)
        return {"success": True, "prediction_id": pred_record.prediction_id, "result": result,
                "disclaimer": "Sambhav may be incorrect. Always verify important decisions independently."}
    except Exception as e:
        logger.error(f"Rich prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── POST /predict/batch ───────────────────────────────────────
@router.post("/batch")
async def batch_predict_endpoint(req: BatchPredictRequest):
    logger.info(f"POST /predict/batch domain={req.domain} count={len(req.batch_data)}")
    try:
        results = []
        for params in req.batch_data:
            res = predict(domain=req.domain, parameters=params)
            results.append(res.to_dict())
        return {"success": True, "results": results}
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── GET /predict/domain-outcomes ─────────────────────────────
@router.get("/outcomes")
async def get_domain_outcomes(domain: str):
    try:
        registry = _load_registry()
        if domain not in registry:
            raise HTTPException(status_code=404, detail=f"Domain '{domain}' not found")
        outcomes = registry[domain].get("supported_outcomes", ["Positive", "Negative"])
        return {"domain": domain, "outcomes": outcomes, "count": len(outcomes)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))