import os, logging, hmac
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Optional
from jose import jwt, JWTError
from sqlalchemy.orm import Session

from db.models import get_db, User
from db.database import create_user, get_user_by_email, log_event

import bcrypt as _bcrypt

router = APIRouter()
logger = logging.getLogger(__name__)

ALGORITHM          = "HS256"
TOKEN_EXPIRE_HOURS = 24

def get_secret_key() -> str:
    return os.getenv("SECRET_KEY", "sambhav_secret_key_2026")

security = HTTPBearer(auto_error=False)

# ---------------------------------------------------------------------------
# Password hashing — native bcrypt; passlib fallback for legacy hashes
# ---------------------------------------------------------------------------

def _hash_password(password: str) -> str:
    """Hash with native bcrypt."""
    return _bcrypt.hashpw(password.encode("utf-8"), _bcrypt.gensalt()).decode("utf-8")


def _verify_password(password: str, hashed: str) -> bool:
    """
    Try native bcrypt first. If it raises ValueError (passlib-style hash prefix
    difference), fall back to passlib. On successful passlib verify we return True
    (the caller is responsible for rehashing if desired).
    """
    try:
        return _bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        # Fallback: passlib format
        try:
            from passlib.context import CryptContext
            _ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
            return _ctx.verify(password, hashed)
        except Exception as e:
            logger.warning(f"Password verification fallback also failed: {e}")
            return False


def _create_token(user_id: str, email: str, tier: str = "registered") -> str:
    expire  = datetime.utcnow() + timedelta(hours=TOKEN_EXPIRE_HOURS)
    payload = {"sub": email, "user_id": user_id, "tier": tier, "exp": expire}
    return jwt.encode(payload, get_secret_key(), algorithm=ALGORITHM)

def _decode_token(token: str) -> dict:
    return jwt.decode(token, get_secret_key(), algorithms=[ALGORITHM])


def get_current_user(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db)
) -> dict:
    if not creds:
        raise HTTPException(status_code=401, detail="Missing authorization token")
    try:
        payload = _decode_token(creds.credentials)
        # Guest — never hit DB
        if payload.get("sub") == "guest" or payload.get("user_id") == "guest":
            return payload
        user = get_user_by_email(db, payload.get("sub"))
        if not user or not user.is_active:
            raise HTTPException(status_code=401, detail="User not found or inactive")
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid authorization token")


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class RegisterRequest(BaseModel):
    username: Optional[str] = Field(None)
    email:    str = Field(..., example="user@example.com")
    password: str = Field(..., min_length=6)

class LoginRequest(BaseModel):
    email:    str = Field(..., example="user@example.com")
    password: str


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@router.post("/register")
async def register(req: RegisterRequest, db: Session = Depends(get_db)):
    if get_user_by_email(db, req.email):
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed_pwd = _hash_password(req.password)
    user = create_user(db, req.email, hashed_pwd, "registered")
    token = _create_token(str(user.user_id), user.email, user.tier)
    return {"success": True, "token": token, "email": user.email,
            "tier": user.tier, "user_id": str(user.user_id)}


@router.post("/guest")
async def guest_login():
    token = _create_token(user_id="guest", email="guest", tier="guest")
    return {"success": True, "token": token, "email": "guest", "tier": "guest",
            "user_id": "guest", "note": "Guest predictions limited to 10/day"}


@router.post("/login")
async def login(req: LoginRequest, db: Session = Depends(get_db)):
    user = get_user_by_email(db, req.email)
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    ok = _verify_password(req.password, user.password_hash)
    if not ok:
        raise HTTPException(status_code=401, detail="Invalid email or password")

    # Transparent rehash: if hash is passlib format, upgrade to native bcrypt
    try:
        _bcrypt.checkpw(req.password.encode("utf-8"), user.password_hash.encode("utf-8"))
    except Exception:
        user.password_hash = _hash_password(req.password)
        db.commit()
        logger.info(f"Rehashed password for user {user.email} to native bcrypt")

    token = _create_token(str(user.user_id), user.email, user.tier)
    log_event(db, "login", user_id=str(user.user_id))
    return {"success": True, "token": token, "email": user.email,
            "tier": user.tier, "user_id": str(user.user_id)}


@router.get("/me")
async def get_me(user: dict = Depends(get_current_user), db: Session = Depends(get_db)):
    db_user = None if user.get("email") == "guest" else get_user_by_email(db, user.get("email"))
    tier_info = getattr(db_user, "tier", "guest") if db_user else "guest"
    return {"success": True, "user": user, "stats": {"tier": tier_info}}


class ResetPasswordRequest(BaseModel):
    email:        str = Field(..., example="user@example.com")
    new_password: str = Field(..., min_length=6)

@router.post("/reset-password")
async def reset_password(req: ResetPasswordRequest, db: Session = Depends(get_db)):
    user = get_user_by_email(db, req.email)
    if not user:
        raise HTTPException(status_code=404, detail="No account found with this email")
    user.password_hash = _hash_password(req.new_password)
    db.commit()
    log_event(db, "password_reset", user_id=str(user.user_id))
    return {"success": True, "message": "Password reset successfully. Please sign in."}


class ChangePasswordRequest(BaseModel):
    old_password: str = Field(..., min_length=6)
    new_password: str = Field(..., min_length=6)

@router.post("/change-password")
async def change_password(req: ChangePasswordRequest, user: dict = Depends(get_current_user),
                           db: Session = Depends(get_db)):
    if user.get("email") == "guest":
        raise HTTPException(status_code=403, detail="Guest accounts cannot change password")
    db_user = get_user_by_email(db, user.get("email"))
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    if not _verify_password(req.old_password, db_user.password_hash):
        raise HTTPException(status_code=400, detail="Current password is incorrect")
    db_user.password_hash = _hash_password(req.new_password)
    db.commit()
    log_event(db, "password_change", user_id=str(db_user.user_id))
    return {"success": True, "message": "Password changed successfully"}


@router.delete("/me")
async def delete_account(user: dict = Depends(get_current_user), db: Session = Depends(get_db)):
    if user.get("email") == "guest":
        raise HTTPException(status_code=403, detail="Guest accounts cannot be deleted")
    db_user = get_user_by_email(db, user.get("email"))
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    db_user.is_active = False
    db.commit()
    log_event(db, "account_deleted", user_id=str(db_user.user_id))
    return {"success": True, "message": "Account deleted. All data will be purged within 30 days."}
