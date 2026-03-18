import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from core.fact_checker import fact_check_claim, fact_check_batch
from core.safety import check_hard_blocks

router = APIRouter()
logger = logging.getLogger(__name__)

# ── Request schemas ───────────────────────────────────────────
class SingleClaimRequest(BaseModel):
    claim: str = Field(..., example="India was the first country to land on the Moon's south pole in 2023")

class BatchRequest(BaseModel):
    text:      str           = Field(..., example="Full article or speech text here...")
    max_claims:Optional[int] = Field(10, ge=1, le=15)

# ── POST /fact-check ──────────────────────────────────────────
@router.post("")
async def fact_check_endpoint(req: SingleClaimRequest):
    """
    Single claim fact-check across 8 dimensions.
    Dual-LLM verdict with web search evidence.
    """
    logger.info(f"POST /fact-check claim={req.claim[:60]}...")

    # Safety check
    safety = check_hard_blocks(req.claim)
    if not safety["safe"]:
        raise HTTPException(
            status_code=400,
            detail={"blocked": True, "message": safety["message"]})

    try:
        result = fact_check_claim(req.claim)
        return {
            "success":    True,
            "result":     result,
            "disclaimer": "Sambhav may be incorrect. Always verify important decisions independently.",
        }
    except Exception as e:
        logger.error(f"Fact-check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ── POST /fact-check/batch ────────────────────────────────────
@router.post("/batch")
async def batch_fact_check_endpoint(req: BatchRequest):
    """
    Batch fact-check an entire article or speech.
    Extracts all claims and checks each independently.
    Target: 30-60 seconds for a full article.
    """
    logger.info(f"POST /fact-check/batch text_len={len(req.text)}")

    safety = check_hard_blocks(req.text[:500])
    if not safety["safe"]:
        raise HTTPException(
            status_code=400,
            detail={"blocked": True, "message": safety["message"]})

    try:
        result = fact_check_batch(req.text)
        return {
            "success":    True,
            "result":     result,
            "disclaimer": "Sambhav may be incorrect. Always verify important decisions independently.",
        }
    except Exception as e:
        logger.error(f"Batch fact-check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
