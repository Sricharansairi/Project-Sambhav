import uuid, logging
from datetime import datetime
from sqlalchemy.orm import Session
from db.models import User, Prediction, Evaluation, FactCheck, AuditLog, MonitoringSession

logger = logging.getLogger(__name__)

# Project Sambhav uses Supabase PostgreSQL for high-reliability persistence.
# The connection string is managed via the DATABASE_URL environment variable.

def generate_id(prefix: str = "SMB") -> str:
    """Generate unique ID like SMB-2026-00001."""
    short = str(uuid.uuid4())[:8].upper()
    return f"{prefix}-2026-{short}"

# ── USER OPERATIONS ───────────────────────────────────────────
def create_user(db: Session, email: str,
                password_hash: str, tier: str = "registered") -> User:
    user = User(
        user_id       = uuid.uuid4(),
        email         = email,
        password_hash = password_hash,
        tier          = tier,
        created_at    = datetime.utcnow(),
        personal_brier= {}
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    logger.info(f"User created: {email}")
    return user

# Removed get_user_by_username - use get_user_by_email instead

def get_user_by_email(db: Session, email: str) -> User:
    return db.query(User).filter(User.email == email).first()

def update_user_brier(db: Session, user_id: str,
                      domain: str, brier_score: float):
    user = db.query(User).filter(User.user_id == user_id).first()
    if user:
        brier = user.personal_brier or {}
        if domain not in brier:
            brier[domain] = []
        brier[domain].append(round(brier_score, 4))
        user.personal_brier = brier
        db.commit()

# ── PREDICTION OPERATIONS ─────────────────────────────────────
def save_prediction(db: Session, user_id: str,
                    prediction_result: dict) -> Prediction:
    pred = Prediction(
        prediction_id     = generate_id("SMB"),
        user_id           = user_id,
        domain            = prediction_result.get("domain"),
        input_text        = prediction_result.get("question") or prediction_result.get("input_text"),
        parameters        = prediction_result.get("raw_parameters", {}),
        ml_probability    = prediction_result.get("ml_probability"),
        llm_probability   = prediction_result.get("llm_probability"),
        reconciled_prob   = prediction_result.get("final_probability"),
        warning_level     = prediction_result.get("confidence_tier"),
        agreement_gap     = prediction_result.get("gap"),
        reliability_index = prediction_result.get("reliability_index"),
        audit_flags       = prediction_result.get("audit_flags", []),
        shap_values       = prediction_result.get("shap_values", {}),
        mode              = prediction_result.get("mode", "guided"),
        created_at        = datetime.utcnow(),
    )
    db.add(pred)
    db.commit()
    db.refresh(pred)
    logger.info(f"Prediction saved: {pred.prediction_id}")
    return pred

def get_predictions(db: Session, user_id: str,
                    domain: str = None, limit: int = 20) -> list:
    query = db.query(Prediction).filter(Prediction.user_id == user_id)
    if domain:
        query = query.filter(Prediction.domain == domain)
    return query.order_by(Prediction.created_at.desc()).limit(limit).all()

def get_prediction_by_id(db: Session, prediction_id: str) -> Prediction:
    return db.query(Prediction).filter(
        Prediction.prediction_id == prediction_id).first()

def delete_prediction(db: Session, prediction_id: str,
                      user_id: str) -> bool:
    pred = db.query(Prediction).filter(
        Prediction.prediction_id == prediction_id,
        Prediction.user_id == user_id).first()
    if pred:
        db.delete(pred)
        db.commit()
        return True
    return False

# ── EVALUATION OPERATIONS ─────────────────────────────────────
def save_evaluation(db: Session, prediction_id: str,
                    evaluator_id: str, actual_outcome: bool,
                    lessons: str = "") -> Evaluation:
    # Calculate Brier score
    pred = get_prediction_by_id(db, prediction_id)
    brier = 0.0
    if pred and pred.final_probability is not None:
        brier = (pred.final_probability - int(actual_outcome)) ** 2

    # Grade based on Brier score
    if   brier < 0.05: grade = "A+"
    elif brier < 0.10: grade = "A"
    elif brier < 0.15: grade = "B"
    elif brier < 0.20: grade = "C"
    elif brier < 0.25: grade = "D"
    else:              grade = "F"

    ev = Evaluation(
        prediction_id   = prediction_id,
        evaluator_user_id= evaluator_id,
        actual_outcomes = {"truth": actual_outcome},
        brier_score     = round(brier, 4),
        lessons_learned = lessons,
        evaluation_grade= grade,
        evaluated_at    = datetime.utcnow(),
    )
    db.add(ev)
    db.commit()
    db.refresh(ev)
    logger.info(f"Evaluation saved: {ev.evaluation_id} grade={grade}")
    return ev

# ── FACT CHECK OPERATIONS ─────────────────────────────────────
def save_fact_check(db: Session, user_id: str,
                    result: dict) -> FactCheck:
    fc = FactCheck(
        fact_check_id    = generate_id("FCK"),
        user_id          = user_id,
        claim            = result.get("claim",""),
        credibility_score= result.get("credibility_score", 50),
        credibility_label= result.get("credibility_label","UNCERTAIN"),
        verdict          = result.get("verdict","UNCERTAIN"),
        dimensions       = result.get("dimensions", {}),
        sources          = result.get("sources", []),
        created_at       = datetime.utcnow(),
    )
    db.add(fc)
    db.commit()
    db.refresh(fc)
    return fc

# ── AUDIT LOG OPERATIONS ──────────────────────────────────────
def log_event(db: Session, event_type: str, user_id: str = None,
              domain: str = None, details: dict = None,
              ip_address: str = None):
    log = AuditLog(
        log_id     = generate_id("LOG"),
        event_type = event_type,
        user_id    = user_id,
        domain     = domain,
        details    = details or {},
        ip_address = ip_address,
        created_at = datetime.utcnow(),
    )
    db.add(log)
    db.commit()

# ── MONITORING OPERATIONS ─────────────────────────────────────
def create_monitoring_session(db: Session, user_id: str,
                               name: str, domain: str,
                               parameters: dict,
                               threshold_low: float = 0.3,
                               threshold_high: float = 0.7) -> MonitoringSession:
    session = MonitoringSession(
        session_id      = generate_id("MON"),
        user_id         = user_id,
        name            = name,
        domain          = domain,
        parameters      = parameters,
        threshold_low   = threshold_low,
        threshold_high  = threshold_high,
        is_active       = True,
        created_at      = datetime.utcnow(),
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return session

def get_user_calibration_bias(db: Session, user_id: str) -> float:
    """
    Calculate the user's calibration bias.
    Positive bias means the user's predictions are higher than reality (overconfident).
    Negative bias means predictions are lower than reality (underconfident).
    """
    # Fetch all resolved predictions for this user
    # Prediction has a relationship to Evaluation in models.py
    # We join Prediction and Evaluation
    from db.models import Prediction, Evaluation
    results = db.query(Prediction.reconciled_prob, Evaluation.actual_outcomes)\
        .join(Evaluation, Prediction.prediction_id == Evaluation.prediction_id)\
        .filter(Prediction.user_id == user_id).all()
    
    if not results:
        return 0.0
    
    biases = []
    for pred_prob, actual in results:
        # actual is a dict like {"truth": True}
        actual_val = 1.0 if actual.get("truth") else 0.0
        biases.append(pred_prob - actual_val)
        
    return sum(biases) / len(biases)

def get_user_stats(db: Session, user_id: str) -> dict:
    """Get user statistics — predictions, avg Brier, best domain."""
    preds = db.query(Prediction).filter(
        Prediction.user_id == user_id).all()
    evals = []
    for p in preds:
        evals.extend(p.evaluations)

    total_preds  = len(preds)
    total_evals  = len(evals)
    avg_brier    = round(sum(e.brier_score for e in evals) /
                        max(len(evals), 1), 4)
    domains_used = list(set(p.domain for p in preds))

    return {
        "total_predictions":  total_preds,
        "total_evaluations":  total_evals,
        "avg_brier_score":    avg_brier,
        "domains_used":       domains_used,
        "calibration_grade":  "A+" if avg_brier < 0.05 else
                              "A"  if avg_brier < 0.10 else
                              "B"  if avg_brier < 0.15 else "C",
    }
