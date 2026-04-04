import os
import uuid
from datetime import datetime
from sqlalchemy import (create_engine, Column, String, Float, Integer,
                        Boolean, DateTime, Text, ForeignKey)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import text
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

if DATABASE_URL and DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# Default to generic sqlite in-memory or fallback, but warn the user
if not DATABASE_URL:
    print("WARNING: DATABASE_URL not found in .env. Alembic will require this for PostgreSQL.")
    DATABASE_URL = "sqlite:///./sambhav.db" # Local fallback for dev testing

# Supabase / PostgreSQL database configuration (U.08)
# Optimization: Added pool_pre_ping=True to handle disconnected connections gracefully
# Optimization: Added connect_args to force IPv4 and set specific timeout/keepalives
connect_args = {
    "connect_timeout": 10,
    "keepalives": 1,
    "keepalives_idle": 30,
    "keepalives_interval": 10,
    "keepalives_count": 5,
}

# If we are in production (Hugging Face), we try to force IPv4
# by adding gssencmode=disable which sometimes helps with psycopg2 connection issues in containers
if os.getenv("DEPLOY_ENV") == "production":
    connect_args["gssencmode"] = "disable"

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
    pool_timeout=30,
    pool_recycle=1800,
    connect_args=connect_args
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ── 1. Users Table ────────────────────────────────────────────
class User(Base):
    __tablename__ = "users"
    
    user_id          = Column(UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    email            = Column(String(255), unique=True, nullable=False)
    password_hash    = Column(String(255), nullable=False)
    created_at       = Column(DateTime, default=datetime.utcnow)
    last_login       = Column(DateTime, nullable=True)
    tier             = Column(String(20), default="registered")  # registered | power
    personal_brier   = Column(JSONB, nullable=True)
    bias_corrections = Column(JSONB, nullable=True)
    total_preds      = Column(Integer, default=0)
    preferred_mode   = Column(String(50), nullable=True)
    is_active        = Column(Boolean, default=True)

    predictions      = relationship("Prediction", back_populates="user")
    # Using string names for relationship targets to avoid circular loading
    evaluations      = relationship("Evaluation", back_populates="evaluator")

# ── 2. Predictions Table ──────────────────────────────────────
class Prediction(Base):
    __tablename__     = "predictions"
    
    prediction_id     = Column(String(30), primary_key=True)  # SMB-2026-NNNNN
    user_id           = Column(UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=True)
    session_id        = Column(String(100), nullable=True)
    created_at        = Column(DateTime, default=datetime.utcnow)
    mode              = Column(String(30), nullable=False)
    domain            = Column(String(50), nullable=True)
    input_text        = Column(Text, nullable=True)
    parameters        = Column(JSONB, nullable=True)
    reliability_index = Column(Float, nullable=True)
    warning_level     = Column(String(20), nullable=True)
    outcomes          = Column(JSONB, nullable=True) # happened, not_happened, partial, too_early
    ml_probability    = Column(Float, nullable=True)
    llm_probability   = Column(Float, nullable=True)
    reconciled_prob   = Column(Float, nullable=True)
    agreement_gap     = Column(Float, nullable=True)
    audit_status      = Column(String(20), nullable=True)
    audit_flags       = Column(JSONB, nullable=True)
    shap_values       = Column(JSONB, nullable=True)
    pdf_downloaded    = Column(Boolean, default=False)
    share_token       = Column(String(100), unique=True, nullable=True)
    evaluation_done   = Column(Boolean, default=False)

    user              = relationship("User", back_populates="predictions")
    evaluations       = relationship("Evaluation", back_populates="prediction")

# ── 3. Evaluations Table ──────────────────────────────────────
class Evaluation(Base):
    __tablename__      = "evaluations"
    
    evaluation_id      = Column(UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    prediction_id      = Column(String(30), ForeignKey("predictions.prediction_id"))
    evaluator_user_id  = Column(UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=True)
    evaluator_type     = Column(String(20), nullable=True)  # owner|collaborator|peer
    evaluated_at       = Column(DateTime, default=datetime.utcnow)
    actual_outcomes    = Column(JSONB, nullable=True)
    quality_ratings    = Column(JSONB, nullable=True)
    what_system_missed = Column(Text, nullable=True)
    brier_score        = Column(Float, nullable=True)
    accuracy_score     = Column(Float, nullable=True)
    lessons_learned    = Column(Text, nullable=True)
    evaluation_grade   = Column(String(5), nullable=True)

    prediction         = relationship("Prediction", back_populates="evaluations")
    evaluator          = relationship("User", back_populates="evaluations")

# ── 4. Fact Checks Table ──────────────────────────────────────
class FactCheck(Base):
    __tablename__      = "fact_checks"
    
    fact_check_id      = Column(String(30), primary_key=True)  # FCK-2026-NNNNN
    user_id            = Column(UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=True)
    claim              = Column(Text, nullable=False)
    credibility_score  = Column(Integer, nullable=True)
    verdict            = Column(String(50), nullable=True)
    dimensions         = Column(JSONB, nullable=True)
    sources            = Column(JSONB, nullable=True)
    created_at         = Column(DateTime, default=datetime.utcnow)

# ── 5. Monitoring Sessions Table ──────────────────────────────
class MonitoringSession(Base):
    __tablename__   = "monitoring_sessions"
    
    session_id      = Column(String(30), primary_key=True)  # MON-2026-NNNNN
    user_id         = Column(UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=False)
    name            = Column(String(100), nullable=False)
    domain          = Column(String(50), nullable=True)
    parameters      = Column(JSONB, nullable=True)
    threshold_low   = Column(Float, default=0.3)
    threshold_high  = Column(Float, default=0.7)
    check_frequency = Column(String(20), default="daily")
    is_active       = Column(Boolean, default=True)
    created_at      = Column(DateTime, default=datetime.utcnow)

# ── 6. Audit Logs Table ───────────────────────────────────────
class AuditLog(Base):
    __tablename__ = "audit_logs"
    
    log_id        = Column(String(30), primary_key=True)  # LOG-2026-NNNNN
    event_type    = Column(String(50), nullable=False)  # prediction/factcheck/safety_block
    user_id       = Column(UUID(as_uuid=True), ForeignKey("users.user_id"), nullable=True)
    domain        = Column(String(50), nullable=True)
    details       = Column(JSONB, nullable=True)
    ip_address    = Column(String(45), nullable=True)
    created_at    = Column(DateTime, default=datetime.utcnow)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
