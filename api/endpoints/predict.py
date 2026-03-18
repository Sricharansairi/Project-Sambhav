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
