import os, logging, hashlib, hmac
from datetime import datetime, timedelta
from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Optional
from jose import jwt, JWTError
from sqlalchemy.orm import Session

# Import DB dependencies
from db.models import get_db, User
from db.database import create_user, get_user_by_email, log_event

from passlib.context import CryptContext

router = APIRouter()
logger = logging.getLogger(__name__)

ALGORITHM          = "HS256"
TOKEN_EXPIRE_HOURS = 24

def get_secret_key() -> str:
    """Dynamically fetch SECRET_KEY to ensure it's up-to-date with HF secrets."""
    return os.getenv("SECRET_KEY", "sambhav_secret_key_2026")

security = HTTPBearer(auto_error=False)
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def _hash_password(password: str) -> str:
    """Strong bcrypt hashing."""
    return pwd_context.hash(password)

def _verify_password(password: str, hashed: str) -> bool:
    return pwd_context.verify(password, hashed)

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
        # Handle guest users correctly
        if payload.get("sub") == "guest":
            return payload
            
        user = get_user_by_email(db, payload.get("sub"))
        if not user or not user.is_active:
            raise HTTPException(status_code=401, detail="User not found or inactive")
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid authorization token")

class RegisterRequest(BaseModel):
    username: Optional[str] = Field(None, example="legacy_ui_field") # Ignored but kept for UI compat
    email:    str = Field(..., example="user@example.com")
    password: str = Field(..., min_length=6)

class LoginRequest(BaseModel):
    email:    str = Field(..., example="user@example.com")
    password: str

@router.post("/register")
async def register(req: RegisterRequest, db: Session = Depends(get_db)):
    if get_user_by_email(db, req.email):
        raise HTTPException(status_code=400, detail="Email already registered")
        
    hashed_pwd = _hash_password(req.password)
    user = create_user(db, req.email, hashed_pwd, "registered")
    token = _create_token(str(user.user_id), user.email, user.tier)
    
    return {
        "success": True,
        "token": token,
        "email": user.email,
        "tier": user.tier,
        "user_id": str(user.user_id)
    }

# ── Guest Entry ──────────────────────────────────────────────
@router.post("/guest")
async def guest_login():
    """Returns a hardcoded guest token for seamless entry."""
    # Ensure guest token uses the same structure and dynamic secret key
    token = _create_token(user_id="guest", email="guest", tier="guest")
    return {
        "success": True, 
        "token": token, 
        "email": "guest", 
        "tier": "guest",
        "user_id": "guest",
        "note": "Guest predictions limited to 10/day"
    }

@router.post("/login")
async def login(req: LoginRequest, db: Session = Depends(get_db)):
    user = get_user_by_email(db, req.email)
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="Invalid email or password")
        
    if not _verify_password(req.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")
        
    token = _create_token(str(user.user_id), user.email, user.tier)
    log_event(db, "login", user_id=str(user.user_id))
    
    return {
        "success": True,
        "token": token,
        "email": user.email,
        "tier": user.tier,
        "user_id": str(user.user_id)
    }

@router.get("/me")
async def get_me(user: dict = Depends(get_current_user), db: Session = Depends(get_db)):
    db_user = None if user.get('email') == 'guest' else get_user_by_email(db, user.get('email'))
    tier_info = getattr(db_user, 'tier', 'guest') if db_user else 'guest'
    return {"success": True, "user": user, "stats": {"tier": tier_info}}


# ── Password Reset (email-based) ─────────────────────────────
class ResetPasswordRequest(BaseModel):
    email:        str = Field(..., example="user@example.com")
    new_password: str = Field(..., min_length=6)

@router.post("/reset-password")
async def reset_password(req: ResetPasswordRequest, db: Session = Depends(get_db)):
    """Reset password by email — no old password needed (forgot password flow)."""
    user = get_user_by_email(db, req.email)
    if not user:
        raise HTTPException(status_code=404, detail="No account found with this email")
    user.password_hash = _hash_password(req.new_password)
    db.commit()
    log_event(db, "password_reset", user_id=str(user.user_id))
    return {"success": True, "message": "Password reset successfully. Please sign in."}


# ── Change Password (requires current password) ──────────────
class ChangePasswordRequest(BaseModel):
    old_password: str = Field(..., min_length=6)
    new_password: str = Field(..., min_length=6)

@router.post("/change-password")
async def change_password(
    req: ChangePasswordRequest,
    user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Change password — requires old password for verification."""
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


# ── Delete Account (soft delete) ─────────────────────────────
@router.delete("/me")
async def delete_account(
    user: dict = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Soft-delete account — GDPR/DPDP compliance."""
    if user.get("email") == "guest":
        raise HTTPException(status_code=403, detail="Guest accounts cannot be deleted")
    db_user = get_user_by_email(db, user.get("email"))
    if not db_user:
        raise HTTPException(status_code=404, detail="User not found")
    db_user.is_active = False
    db.commit()
    log_event(db, "account_deleted", user_id=str(db_user.user_id))
    return {"success": True, "message": "Account deleted. All data will be purged within 30 days."}
