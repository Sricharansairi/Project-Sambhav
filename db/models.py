import os
from datetime import datetime
from sqlalchemy import (create_engine, Column, String, Float, Integer,
                        Boolean, DateTime, JSON, Text, ForeignKey)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from dotenv import load_dotenv
load_dotenv()

BASE_DIR = os.path.expanduser("~/Desktop/Sri_Coding/Project Sambhav")
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR}/sambhav.db")

engine       = create_engine(DATABASE_URL, connect_args={"check_same_thread": False}
                             if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base         = declarative_base()

# ── Users ─────────────────────────────────────────────────────
class User(Base):
    __tablename__ = "users"
    user_id        = Column(String, primary_key=True)
    username       = Column(String, unique=True, nullable=False)
    email          = Column(String, unique=True, nullable=False)
    password_hash  = Column(String, nullable=False)
    tier           = Column(String, default="registered")  # guest/registered
    created_at     = Column(DateTime, default=datetime.utcnow)
    personal_brier = Column(JSON, default=dict)
    predictions    = relationship("Prediction", back_populates="user")

# ── Predictions ───────────────────────────────────────────────
class Prediction(Base):
    __tablename__    = "predictions"
    prediction_id    = Column(String, primary_key=True)
    user_id          = Column(String, ForeignKey("users.user_id"), nullable=True)
    domain           = Column(String, nullable=False)
    question         = Column(Text)
    parameters       = Column(JSON)
    ml_probability   = Column(Float)
    llm_probability  = Column(Float)
    final_probability= Column(Float)
    confidence_tier  = Column(String)
    reliability_index= Column(Float)
    audit_flags      = Column(JSON)
    shap_values      = Column(JSON)
    debate           = Column(JSON)
    mode             = Column(String, default="guided")
    created_at       = Column(DateTime, default=datetime.utcnow)
    user             = relationship("User", back_populates="predictions")
    evaluations      = relationship("Evaluation", back_populates="prediction")

# ── Evaluations ───────────────────────────────────────────────
class Evaluation(Base):
    __tablename__   = "evaluations"
    evaluation_id   = Column(String, primary_key=True)
    prediction_id   = Column(String, ForeignKey("predictions.prediction_id"))
    evaluator_id    = Column(String)
    actual_outcome  = Column(Boolean)
    brier_score     = Column(Float)
    lessons_learned = Column(Text)
    grade           = Column(String)  # A+/A/B/C/D/F
    created_at      = Column(DateTime, default=datetime.utcnow)
    prediction      = relationship("Prediction", back_populates="evaluations")

# ── Fact Checks ───────────────────────────────────────────────
class FactCheck(Base):
    __tablename__      = "fact_checks"
    fact_check_id      = Column(String, primary_key=True)
    user_id            = Column(String, nullable=True)
    claim              = Column(Text, nullable=False)
    credibility_score  = Column(Integer)
    credibility_label  = Column(String)
    verdict            = Column(String)
    dimensions         = Column(JSON)
    sources            = Column(JSON)
    created_at         = Column(DateTime, default=datetime.utcnow)

# ── Monitoring Sessions ───────────────────────────────────────
class MonitoringSession(Base):
    __tablename__   = "monitoring_sessions"
    session_id      = Column(String, primary_key=True)
    user_id         = Column(String, nullable=False)
    name            = Column(String, nullable=False)
    domain          = Column(String)
    parameters      = Column(JSON)
    threshold_low   = Column(Float, default=0.3)
    threshold_high  = Column(Float, default=0.7)
    check_frequency = Column(String, default="daily")
    is_active       = Column(Boolean, default=True)
    created_at      = Column(DateTime, default=datetime.utcnow)

# ── Audit Logs ────────────────────────────────────────────────
class AuditLog(Base):
    __tablename__ = "audit_logs"
    log_id        = Column(String, primary_key=True)
    event_type    = Column(String)  # prediction/factcheck/safety_block
    user_id       = Column(String, nullable=True)
    domain        = Column(String, nullable=True)
    details       = Column(JSON)
    ip_address    = Column(String, nullable=True)
    created_at    = Column(DateTime, default=datetime.utcnow)

# ── DB init ───────────────────────────────────────────────────
def init_db():
    Base.metadata.create_all(bind=engine)
    print("✅ Database tables created")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

if __name__ == "__main__":
    init_db()
    print("Tables:", list(Base.metadata.tables.keys()))
