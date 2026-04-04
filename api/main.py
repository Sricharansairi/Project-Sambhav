import os, sys, logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from slowapi import Limiter, _rate_limit_exceeded_handler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Bulk Env Loading (HF Secret Support) ──────────────────────
load_dotenv()
# Force log all env vars starting with BYPASS for debugging
for k, v in os.environ.items():
    if k.startswith("BYPASS"):
        logger.info(f"System ENV check: {k}={v}")

# If ENV_FILE is set as a secret, parse it directly into os.environ
env_file_content = os.getenv("ENV_FILE")
if env_file_content:
    logger.info("Found ENV_FILE secret, parsing...")
    for line in env_file_content.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        try:
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ[key] = value
            # Re-load into dotenv if possible
            if key == "SECRET_KEY": logger.info(f"Loaded SECRET_KEY from ENV_FILE")
            if "BYPASS" in key: logger.info(f"Loaded {key}={value} from ENV_FILE")
        except Exception as e:
            logger.warning(f"Failed to parse line in ENV_FILE: {line} - {e}")

# Final explicit override check
if os.getenv("BYPASS_AUTH") == "true":
    logger.info("CRITICAL: BYPASS_AUTH is confirmed TRUE in final environment")
# ─────────────────────────────────────────────────────────────

from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

# Get absolute path of project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
    title       = "Project Sambhav API",
    description = "Multi-Modal Probabilistic Inference Engine",
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)
# ── CORS ────────────────────────────────────────────────────────
# In production, ALLOW_ORIGINS should be restricted to your Vercel domain.
# e.g. ["https://sambhav.vercel.app"]
# We also include common Vercel preview URL patterns.
ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ALLOW_ORIGINS,
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
    expose_headers    = ["*"],
)
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    import datetime
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "detail":     str(exc),
            "error_code": 500,
            "timestamp":  datetime.datetime.utcnow().isoformat(),
            "path":       str(request.url)
        }
    )
@app.get("/health")
async def health():
    return {
        "status":  "ok",
        "version": "1.0.0",
        "name":    "Project Sambhav",
        "tagline": "Uncertainty, Quantified."
    }
from api.endpoints import predict, factcheck, vision, history, auth, evaluate, export, modes, reports

app.include_router(predict.router,   prefix="/predict",    tags=["Prediction"])
app.include_router(modes.router,     prefix="/modes",      tags=["Operating Modes"])
app.include_router(factcheck.router, prefix="/fact-check", tags=["Fact Check"])
app.include_router(vision.router,    prefix="/vision",     tags=["Vision"])
app.include_router(history.router,   prefix="/history",    tags=["History"])
app.include_router(auth.router,      prefix="/auth",       tags=["Auth"])
app.include_router(evaluate.router,  prefix="/evaluate",   tags=["Evaluation"])
app.include_router(reports.router,   prefix="/v1",         tags=["Reports"])
app.include_router(export.router,    prefix="/export",     tags=["Export"])
from fastapi import APIRouter
config_router = APIRouter()
@config_router.get("/registry")
async def get_registry():
    from core.predictor import _load_registry
    return {"success": True, "registry": _load_registry()}
app.include_router(config_router, prefix="/config", tags=["Config"])
@app.get("/")
async def root():
    return {
        "name":      "Project Sambhav",
        "version":   "1.0.0",
        "docs":      "/docs",
        "health":    "/health",
        "api_root":  "/api",
        "endpoints": ["/api/predict", "/api/fact-check", "/api/vision", "/api/history", "/api/auth", "/api/export", "/api/modes"]
    }
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)