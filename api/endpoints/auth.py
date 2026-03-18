import os, logging, hashlib, hmac
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Optional
from jose import jwt, JWTError

router = APIRouter()
logger = logging.getLogger(__name__)

SECRET_KEY         = os.getenv("SECRET_KEY", "sambhav_secret_key_2026")
ALGORITHM          = "HS256"
TOKEN_EXPIRE_HOURS = 24

security = HTTPBearer(auto_error=False)
_users: dict = {}

def _hash_password(password: str) -> str:
    """Simple secure hash — no bcrypt 72-byte limit."""
    return hashlib.pbkdf2_hmac(
        'sha256', password.encode(), SECRET_KEY.encode(), 100000).hex()

def _verify_password(password: str, hashed: str) -> bool:
    return hmac.compare_digest(_hash_password(password), hashed)

def _create_token(username: str, tier: str = "registered") -> str:
    expire  = datetime.utcnow() + timedelta(hours=TOKEN_EXPIRE_HOURS)
    payload = {"sub": username, "tier": tier, "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def _decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

def get_current_user(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> dict:
    if not creds:
        return {"username": "guest", "tier": "guest"}
    return _decode_token(creds.credentials)

class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=30)
    email:    str = Field(..., example="user@example.com")
    password: str = Field(..., min_length=6)

class LoginRequest(BaseModel):
    username: str
    password: str

@router.post("/register")
async def register(req: RegisterRequest):
    if req.username in _users:
        raise HTTPException(status_code=400, detail="Username already exists")
    _users[req.username] = {
        "username":      req.username,
        "email":         req.email,
        "password_hash": _hash_password(req.password),
        "tier":          "registered",
        "created_at":    datetime.utcnow().isoformat(),
    }
    token = _create_token(req.username)
    logger.info(f"New user registered: {req.username}")
    return {"success": True, "token": token,
            "username": req.username, "tier": "registered"}

@router.post("/login")
async def login(req: LoginRequest):
    user = _users.get(req.username)
    if not user or not _verify_password(req.password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = _create_token(req.username, user["tier"])
    return {"success": True, "token": token,
            "username": req.username, "tier": user["tier"]}

@router.post("/guest")
async def guest_login():
    token = _create_token("guest", "guest")
    return {"success": True, "token": token,
            "username": "guest", "tier": "guest",
            "note": "Guest predictions limited to 10/day"}

@router.get("/me")
async def get_me(user: dict = Depends(get_current_user)):
    return {"success": True, "user": user}
