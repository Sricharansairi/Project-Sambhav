import time, logging
from collections import defaultdict
from fastapi import HTTPException, Request

logger = logging.getLogger(__name__)

# In-memory store (replaced by Redis in production)
_requests: dict = defaultdict(list)

LIMITS = {
    "guest":      {"requests": 10,  "window": 86400},  # 10/day
    "registered": {"requests": 200, "window": 86400},  # 200/day
    "default":    {"requests": 50,  "window": 86400},  # 50/day
}

def _get_ip(request: Request) -> str:
    forwarded = request.headers.get("X-Forwarded-For")
    return forwarded.split(",")[0] if forwarded else request.client.host

def check_rate_limit(request: Request, tier: str = "default"):
    ip      = _get_ip(request)
    limit   = LIMITS.get(tier, LIMITS["default"])
    window  = limit["window"]
    max_req = limit["requests"]
    now     = time.time()

    # Clean old requests outside window
    _requests[ip] = [t for t in _requests[ip] if now - t < window]

    if len(_requests[ip]) >= max_req:
        logger.warning(f"Rate limit hit: {ip} tier={tier}")
        raise HTTPException(
            status_code=429,
            detail=f"Rate limit exceeded. Max {max_req} requests per day.")

    _requests[ip].append(now)

def get_remaining(request: Request, tier: str = "default") -> dict:
    ip      = _get_ip(request)
    limit   = LIMITS.get(tier, LIMITS["default"])
    now     = time.time()
    used    = len([t for t in _requests[ip] if now - t < limit["window"]])
    return {
        "limit":     limit["requests"],
        "used":      used,
        "remaining": max(0, limit["requests"] - used),
    }
