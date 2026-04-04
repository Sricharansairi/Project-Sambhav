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

SECRET_KEY         = os.getenv("SECRET_KEY", "sambhav_secret_key_2026")
ALGORITHM          = "HS256"
TOKEN_EXPIRE_HOURS = 24

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
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def _decode_token(token: str) -> dict:
    try:
        # First try HS256 (the standard HS256 algorithm)
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError as e:
        # If HS256 fails, it might be an older session or a transient issue
        logger.warning(f"Token decoding failed: {e}")
        raise HTTPException(status_code=401, detail="Invalid or expired token")
    except Exception as e:
        logger.error(f"Unexpected error decoding token: {e}")
        raise HTTPException(status_code=401, detail="Invalid token format")

def get_current_user(
    creds: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db)
) -> dict:
    # ── GUEST ACCESS HANDLER ──────────────────────────────────
    # If no credentials, or credentials match "guest", allow entry
    if not creds or creds.credentials == "guest":
        return {"sub": "guest", "email": "guest", "tier": "guest", "user_id": None}
        
    try:
        payload = _decode_token(creds.credentials)
        return payload
    except Exception as e:
        # Fallback to guest if token is invalid to prevent breaking the dashboard
        logger.info(f"Invalid token ({e}), falling back to guest mode")
        return {"sub": "guest", "email": "guest", "tier": "guest", "user_id": None}

class RegisterRequest(BaseModel):
    username: Optional[str] = Field(None, example="legacy_ui_field") # Ignored but kept for UI compat
    email:    str = Field(..., example="user@example.com")
    password: str = Field(..., min_length=6)

class LoginRequest(BaseModel):
    email:    str = Field(..., example="user@example.com")
    password: str

@router.post("/register")
async def register(req: RegisterRequest, db: Session = Depends(get_db)):
    # Check if email exists
    if get_user_by_email(db, req.email):
        raise HTTPException(status_code=400, detail="Email already registered")
        
    # Postgres handles gen_random_uuid internally
    user = create_user(
        db=db, 
        email=req.email, 
        password_hash=_hash_password(req.password),
        tier="registered"
    )
    
    token = _create_token(str(user.user_id), user.email)
    logger.info(f"New user registered in Supabase: {user.email}")
    
    # Log audit event
    log_event(db, "registration", user_id=str(user.user_id), details={"email": user.email})
    
    return {"success": True, "token": token,
            "email": req.email, "tier": "registered", "user_id": str(user.user_id)}

@router.post("/guest")
async def guest_login():
    """Returns a hardcoded guest token for seamless entry."""
    # Note: user_id is None for guests
    token = _create_token(user_id="guest", email="guest", tier="guest")
    return {"success": True, "token": token, "email": "guest", "tier": "guest"}

@router.post("/login")
async def login(req: LoginRequest, db: Session = Depends(get_db)):
    user = get_user_by_email(db, req.email)
    
    if not user or not _verify_password(req.password, user.password_hash):
        log_event(db, "failed_login", details={"email": req.email})
        raise HTTPException(status_code=401, detail="Invalid credentials")
        
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account disabled")
        
    # Update last login timestamp
    user.last_login = datetime.utcnow()
    db.commit()

    token = _create_token(str(user.user_id), user.email, user.tier)
    log_event(db, "login", user_id=str(user.user_id))
    
    return {"success": True, "token": token,
            "email": user.email, "tier": user.tier, "user_id": str(user.user_id)}

@router.post("/guest")
async def guest_login():
    token = _create_token("guest_id", "guest", "guest")
    return {"success": True, "token": token,
            "email": "guest", "tier": "guest",
            "note": "Guest predictions limited to 10/day"}

@router.get("/me")
async def get_me(user: dict = Depends(get_current_user), db: Session = Depends(get_db)):
    if user.get("email") == "guest":
        return {"success": True, "user": user, "stats": {}}
        
    db_user = get_user_by_email(db, user.get("sub") or user.get("email"))
    if not db_user:
        # Instead of 404, fallback to guest mode if the DB record is missing
        # This prevents breaking the UI if a user is deleted but has a valid token
        logger.warning(f"User {user.get('email')} has valid token but not found in DB. Falling back to guest.")
        guest_payload = {"sub": "guest", "email": "guest", "tier": "guest", "user_id": None}
        return {"success": True, "user": guest_payload, "stats": {}}
        
    # Return safe user profile
    profile = {
        "user_id": str(db_user.user_id),
        "email": db_user.email,
        "tier": db_user.tier,
        "total_preds": db_user.total_preds,
        "created_at": db_user.created_at.isoformat() if db_user.created_at else None,
        "last_login": db_user.last_login.isoformat() if db_user.last_login else None
    }
    return {"success": True, "user": profile}


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
