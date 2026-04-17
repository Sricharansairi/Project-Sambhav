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

# ── Mount all API routers under BOTH bare and /api prefix ───────────────────
# The frontend calls /api/* in production (VITE_API_URL defaults to '/api')
# The bare prefixes keep backward compat for direct API testing
for _prefix, _router, _tag in [
    ("/predict",    predict.router,   "Prediction"),
    ("/modes",      modes.router,     "Operating Modes"),
    ("/fact-check", factcheck.router, "Fact Check"),
    ("/vision",     vision.router,    "Vision"),
    ("/history",    history.router,   "History"),
    ("/auth",       auth.router,      "Auth"),
    ("/evaluate",   evaluate.router,  "Evaluation"),
    ("/v1",         reports.router,   "Reports"),
    ("/export",     export.router,    "Export"),
]:
    app.include_router(_router, prefix=_prefix,        tags=[_tag])
    app.include_router(_router, prefix=f"/api{_prefix}", tags=[f"{_tag} (api)"])

from fastapi import APIRouter
config_router = APIRouter()
@config_router.get("/registry")
async def get_registry():
    from core.predictor import _load_registry
    return {"success": True, "registry": _load_registry()}
app.include_router(config_router, prefix="/config",     tags=["Config"])
app.include_router(config_router, prefix="/api/config", tags=["Config (api)"])

# ── Serve built React frontend (frontend/dist/) ──────────────────────────────
import os as _os
from pathlib import Path as _Path
_DIST = _Path(__file__).parent.parent / "frontend" / "dist"
if _DIST.exists():
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse

    # Serve assets (JS/CSS/images) from /assets
    app.mount("/assets", StaticFiles(directory=str(_DIST / "assets")), name="assets")

    # Serve any other static files at root level (favicon, robots.txt, etc.)
    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        p = _DIST / "favicon.ico"
        return FileResponse(str(p)) if p.exists() else JSONResponse({}, status_code=404)

    # SPA catch-all — serve index.html for any route not matched above
    @app.get("/", include_in_schema=False)
    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa_fallback(full_path: str = ""):
        # Don't intercept API routes
        if full_path.startswith(("api/", "predict/", "auth/", "fact-check/",
                                  "vision/", "history/", "evaluate/", "export/",
                                  "modes/", "v1/", "config/", "health", "docs", "redoc",
                                  "openapi.json")):
            return JSONResponse({"detail": "Not Found"}, status_code=404)
        index = _DIST / "index.html"
        return FileResponse(str(index))
else:
    logger.warning(f"Frontend dist not found at {_DIST}. Run: cd frontend && npm run build")
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