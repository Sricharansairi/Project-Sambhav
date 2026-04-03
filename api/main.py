import os, sys, logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from slowapi import Limiter, _rate_limit_exceeded_handler

# ── Bulk Env Loading (HF Secret Support) ──────────────────────
load_dotenv()
# If ENV_FILE is set as a secret, write it to a temporary .env and reload
env_file_content = os.getenv("ENV_FILE")
if env_file_content:
    with open(".env.hf", "w") as f:
        f.write(env_file_content)
    load_dotenv(".env.hf", override=True)
    os.remove(".env.hf")
# ─────────────────────────────────────────────────────────────

from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
load_dotenv()
sys.path.insert(0, os.path.expanduser("~/Desktop/Sri_Coding/Project Sambhav"))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
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
ALLOW_ORIGINS = os.getenv("ALLOW_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ALLOW_ORIGINS,
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
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
from api.endpoints import predict, factcheck, vision, history, auth, evaluate, reports
from api.endpoints.export import router as export_router
from api.endpoints.modes import router as modes_router
app.include_router(predict.router,   prefix="/api/predict",    tags=["Prediction"])
app.include_router(modes_router,     prefix="/api/modes",      tags=["Operating Modes"])
app.include_router(factcheck.router, prefix="/api/fact-check", tags=["Fact Check"])
app.include_router(vision.router,    prefix="/api/vision",     tags=["Vision"])
app.include_router(history.router,   prefix="/api/history",    tags=["History"])
app.include_router(auth.router,      prefix="/api/auth",       tags=["Auth"])
app.include_router(evaluate.router,  prefix="/api/evaluate",   tags=["Evaluation"])
app.include_router(reports.router,   prefix="/api/v1",         tags=["Reports"])
app.include_router(export_router,    prefix="/api/export",     tags=["Export"])
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