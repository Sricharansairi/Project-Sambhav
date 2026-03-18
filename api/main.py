import os, sys, logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.expanduser("~/Desktop/Sri_Coding/Project Sambhav"))
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────
app = FastAPI(
    title       = "Project Sambhav API",
    description = "Multi-Modal Probabilistic Inference Engine",
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

# ── CORS ──────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# ── Global exception handler ──────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": str(exc), "path": str(request.url)}
    )

# ── Health check ──────────────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status":  "ok",
        "version": "1.0.0",
        "name":    "Project Sambhav",
        "tagline": "Uncertainty, Quantified."
    }

# ── Routes ────────────────────────────────────────────────────
from api.endpoints import predict, factcheck, vision, history, auth

app.include_router(predict.router,   prefix="/predict",    tags=["Prediction"])
app.include_router(factcheck.router, prefix="/fact-check", tags=["Fact Check"])
app.include_router(vision.router,    prefix="/vision",     tags=["Vision"])
app.include_router(history.router,   prefix="/history",    tags=["History"])
app.include_router(auth.router,      prefix="/auth",       tags=["Auth"])

# ── Root ──────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "name":      "Project Sambhav",
        "version":   "1.0.0",
        "docs":      "/docs",
        "health":    "/health",
        "endpoints": ["/predict","/fact-check","/vision","/history","/auth"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0",
                port=8000, reload=True)
