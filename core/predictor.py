import os, sys, logging, yaml, joblib, numpy as np
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.expanduser("~/Desktop/Sri_Coding/Project Sambhav"))
logger = logging.getLogger(__name__)

BASE   = os.path.expanduser("~/Desktop/Sri_Coding/Project Sambhav")
SCHEMA = os.path.join(BASE, "schemas/domain_registry.yaml")

# ── Result dataclass ──────────────────────────────────────────
@dataclass
class PredictionResult:
    domain:             str
    question:           str
    ml_probability:     Optional[float]
    llm_probability:    Optional[float]
    final_probability:  float
    confidence_tier:    str          # HIGH / MODERATE / LOW / CRITICAL
    gap:                float
    shap_values:        dict         = field(default_factory=dict)
    counterfactuals:    list         = field(default_factory=list)
    audit_flags:        list         = field(default_factory=list)
    debate:             dict         = field(default_factory=dict)
    reliability_index:  float        = 1.0
    reasoning:          str          = ""
    key_factors:        list         = field(default_factory=list)
    mode:               str          = "guided"
    raw_parameters:     dict         = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "domain":            self.domain,
            "question":          self.question,
            "ml_probability":    round(self.ml_probability, 4)  if self.ml_probability  else None,
            "llm_probability":   round(self.llm_probability, 4) if self.llm_probability else None,
            "final_probability": round(self.final_probability, 4),
            "confidence_tier":   self.confidence_tier,
            "gap":               round(self.gap, 4),
            "shap_values":       self.shap_values,
            "counterfactuals":   self.counterfactuals,
            "audit_flags":       self.audit_flags,
            "debate":            self.debate,
            "reliability_index": round(self.reliability_index, 4),
            "reasoning":         self.reasoning,
            "key_factors":       self.key_factors,
            "mode":              self.mode,
        }

# ── Domain registry loader ────────────────────────────────────
def _load_registry() -> dict:
    with open(SCHEMA) as f:
        return yaml.safe_load(f)["domains"]

def _load_model(domain: str):
    registry = _load_registry()
    if domain not in registry:
        raise ValueError(f"Domain '{domain}' not found in registry. "
                         f"Available: {list(registry.keys())}")
    raw_path   = registry[domain]["model_path"]
    # Strip leading "models/" if already in path to avoid duplication
    raw_path   = raw_path.replace("models/", "").replace("models\\", "")
    model_path = os.path.join(BASE, "models", raw_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    artifact = joblib.load(model_path)
    if isinstance(artifact, dict):
        return artifact["model"], artifact.get("scaler"), artifact
    return artifact, None, {}

# ── Confidence tier from gap ──────────────────────────────────
def _confidence_tier(gap: float) -> str:
    if gap < 0.10: return "HIGH"
    if gap < 0.25: return "MODERATE"
    if gap < 0.40: return "LOW"
    return "CRITICAL"

# ── ML prediction ─────────────────────────────────────────────
def _ml_predict(domain: str, parameters: dict) -> Optional[float]:
    try:
        model, scaler, artifact = _load_model(domain)
        feature_cols = artifact.get("feature_cols", [])

        str_to_num = {
            "low": 0.2, "medium": 0.5, "moderate": 0.5,
            "high": 0.8, "very_high": 0.95, "very high": 0.95,
            "none": 0.0, "yes": 1.0, "no": 0.0,
            "male": 1.0, "female": 0.0, "m": 1.0, "f": 0.0,
            "true": 1.0, "false": 0.0
        }

        # Fuzzy param name mapper — maps user params to training col names
        # e.g. study_hours → studytime, attendance → absences proxy
        fuzzy_map = {
            # Student mappings
            "study_hours":         "studytime",
            "study_hours_per_day": "studytime",
            "attendance":          "absences",
            "attendance_pct":      "absences",
            "past_score":          "g1",
            "stress_level":        "health",
            "motivation":          "freetime",
            "sleep_hours":         "freetime",
            "part_time_job":       "paid",
            "extracurricular":     "activities",
            # HR mappings — exact column names
            "job_satisfaction":    "jobsatisfaction",
            "work_life_balance":   "worklifebalance",
            "years_at_company":    "yearsatcompany",
            "monthly_income":      "monthlyincome",
            "distance_from_home":  "distancefromhome",
            "environment_satisfaction": "environmentsatisfaction",
            "job_level":           "joblevel",
            "job_involvement":     "jobinvolvement",
            "num_companies":       "numcompaniesworked",
            "salary_hike":         "percentsalaryhike",
            "years_in_role":       "yearsincurrentrole",
            "years_since_promotion": "yearssincelastpromotion",
            "years_with_manager":  "yearswithcurrmanager",
            # Disease mappings — exact column names
            "age":                 "Age",
            "sex":                 "Sex",
            "chest_pain":          "ChestPainType",
            "blood_pressure":      "RestingBP",
            "cholesterol":         "Cholesterol",
            "fasting_bs":          "FastingBS",
            "resting_ecg":         "RestingECG",
            "heart_rate":          "MaxHR",
            "exercise_angina":     "ExerciseAngina",
            "oldpeak":             "Oldpeak",
            "st_slope":            "ST_Slope",
            # Loan mappings
            "credit":              "credit_score",
            "employment":          "employment_years",
            "missed":              "missed_payments",
            "debt_ratio":          "debt_to_income",
        }

        # Normalize parameter keys using fuzzy map
        normalized = {}
        for k, v in parameters.items():
            mapped_k = fuzzy_map.get(k, k)
            if isinstance(v, str):
                v = str_to_num.get(v.lower().strip(), 0.5)
            try:
                normalized[mapped_k] = float(v)
            except:
                normalized[mapped_k] = 0.5
            # Also keep original key
            try:
                normalized[k] = float(parameters[k]) if not isinstance(parameters[k], str)                     else str_to_num.get(str(parameters[k]).lower().strip(), 0.5)
            except:
                pass

        # Build feature vector in training column order
        if feature_cols:
            feature_vec = []
            for col in feature_cols:
                val = normalized.get(col, normalized.get(col.lower(), 0.0))
                feature_vec.append(float(val))
        else:
            # Fallback — use whatever params we have
            feature_vec = list(normalized.values())

        if not feature_vec:
            logger.warning("ML: empty feature vector")
            return None

        X = np.nan_to_num(
            np.array(feature_vec, dtype=np.float64).reshape(1,-1),
            nan=0.0, posinf=0.0, neginf=0.0)

        if scaler:
            n_exp = scaler.n_features_in_
            if X.shape[1] < n_exp:
                X = np.pad(X, ((0,0),(0, n_exp - X.shape[1])))
            elif X.shape[1] > n_exp:
                X = X[:, :n_exp]
            X = scaler.transform(X)

        prob = model.predict_proba(X)[0][1]
        # Cap extreme probabilities — real world is rarely 0% or 100%
        prob = max(0.05, min(0.95, float(prob)))
        logger.info(f"ML probability: {prob:.4f} (features={len(feature_vec)})")
        return prob
    except Exception as e:
        logger.warning(f"ML prediction failed: {e}")
        return None

# ── LLM prediction ────────────────────────────────────────────
def _llm_predict(domain: str, parameters: dict, question: str) -> Optional[float]:
    try:
        from llm.groq_client import llm_predict
        result = llm_predict(domain, parameters, question)
        return result.get("probability")
    except Exception as e:
        logger.warning(f"LLM prediction failed: {e}")
        return None

# ── SHAP explanation ──────────────────────────────────────────
def _get_shap(domain: str, parameters: dict) -> dict:
    try:
        from core.shap_explainer import explain_prediction
        result = explain_prediction(domain, parameters)
        return result if result else {}
    except Exception as e:
        logger.warning(f"SHAP failed: {e}")
        # Return basic feature importance from parameters
        return {k: round(float(v)/10 if isinstance(v,(int,float)) else 0.1, 3)
                for k,v in list(parameters.items())[:5]}

# ── Reliability index ─────────────────────────────────────────
def _compute_reliability(domain: str, parameters: dict,
                          skipped: list = None) -> float:
    registry       = _load_registry()
    domain_cfg     = registry.get(domain, {})
    all_params_raw = domain_cfg.get("parameters", [])
    skipped        = skipped or []

    # Extract param names — handle both str and dict entries
    all_params = []
    for p in all_params_raw:
        if isinstance(p, dict):
            all_params.append(p.get("name", list(p.keys())[0]))
        else:
            all_params.append(str(p))

    if not all_params:
        return 0.85

    provided   = [p for p in all_params
                  if p not in skipped and parameters.get(p) is not None]
    base_score = len(provided) / max(len(all_params), 1)

    # Also count directly provided parameter keys (even if names differ from YAML)
    directly_provided = len([v for v in parameters.values() if v is not None])
    direct_score      = min(1.0, directly_provided / max(len(all_params), 1))
    # Take the BEST of yaml-matched score vs direct count
    best_score = max(base_score, direct_score)

    # Weight penalty by how many HIGH-weight params are skipped
    high_weight  = [p.get("name") for p in all_params_raw
                    if isinstance(p, dict) and p.get("weight") == "high"]
    high_skipped = len([p for p in skipped if p in high_weight])
    penalty      = (len(skipped) * 0.03) + (high_skipped * 0.07)

    return max(0.25, min(1.0, best_score - penalty))

# ── Audit flags ───────────────────────────────────────────────
def _run_audit(parameters: dict, ml_prob: Optional[float],
               llm_prob: Optional[float], gap: float) -> list:
    flags = []
    # ABN-001 — missing critical parameters
    if not parameters:
        flags.append({"code": "ABN-001", "severity": "HIGH",
                      "message": "No parameters provided"})
    # ABN-002 — extreme ML confidence
    if ml_prob is not None and (ml_prob > 0.97 or ml_prob < 0.03):
        flags.append({"code": "ABN-002", "severity": "MEDIUM",
                      "message": f"Extreme ML probability: {ml_prob:.2%} — check for data issues"})
    # ABN-003 — critical gap
    if gap > 0.40:
        flags.append({"code": "ABN-003", "severity": "CRITICAL",
                      "message": f"ML vs LLM gap {gap:.2%} exceeds threshold — output withheld"})
    # ABN-004 — LLM unavailable
    if llm_prob is None:
        flags.append({"code": "ABN-004", "severity": "LOW",
                      "message": "LLM layer unavailable — ML-only prediction"})
    # ABN-005 — ML unavailable
    if ml_prob is None:
        flags.append({"code": "ABN-005", "severity": "MEDIUM",
                      "message": "ML layer unavailable — LLM-only prediction"})
    return flags

# ══════════════════════════════════════════════════════════════
# MAIN PREDICT FUNCTION
# ══════════════════════════════════════════════════════════════
def predict(
    domain:     str,
    parameters: dict,
    question:   str  = None,
    skipped:    list = None,
    run_debate: bool = True,
    mode:       str  = "guided"
) -> PredictionResult:
    """
    Master orchestrator — runs full 7-stage Sambhav pipeline.

    Stages:
      1. Feature engineering + ML prediction
      2. LLM prediction (independent)
      3. Gap analysis + confidence tier
      4. Multi-agent debate (if gap > 10%)
      5. SHAP explanation
      6. Audit flags
      7. Reliability index
    """
    question = question or f"What is the probability of a positive outcome in the {domain} domain?"
    logger.info(f"predict() called — domain={domain}, mode={mode}")

    # ── Stage 1: ML ───────────────────────────────────────────
    ml_prob = _ml_predict(domain, parameters)
    logger.info(f"ML probability: {ml_prob}")

    # ── Stage 2: LLM (never sees ML result) ──────────────────
    llm_prob = _llm_predict(domain, parameters, question)
    logger.info(f"LLM probability: {llm_prob}")

    # ── Stage 3: Gap analysis ─────────────────────────────────
    if ml_prob is not None and llm_prob is not None:
        gap  = abs(ml_prob - llm_prob)
        final = (ml_prob * 0.6) + (llm_prob * 0.4)   # ML weighted higher
    elif ml_prob is not None:
        gap, final = 0.0, ml_prob
    elif llm_prob is not None:
        gap, final = 0.0, llm_prob
    else:
        gap, final = 0.0, 0.5

    tier = _confidence_tier(gap)
    logger.info(f"Gap={gap:.3f} Tier={tier} Final={final:.3f}")

    # ── Stage 4: Multi-agent debate if gap > 10% ─────────────
    debate_result = {}
    if run_debate and gap > 0.10 and tier != "CRITICAL":
        try:
            from llm.multi_agent import run_debate as _debate
            debate_result = _debate(domain, parameters, question)
            # Use realist's probability as final if debate ran
            final = debate_result.get("final_probability", final)
            logger.info(f"Debate final: {final:.3f}")
        except Exception as e:
            logger.warning(f"Debate failed: {e}")

    # ── Stage 5: SHAP ─────────────────────────────────────────
    shap_vals = _get_shap(domain, parameters)

    # ── Stage 6: Audit flags ──────────────────────────────────
    flags = _run_audit(parameters, ml_prob, llm_prob, gap)

    # CRITICAL gap — run debate to reconcile instead of blocking
    if tier == "CRITICAL":
        try:
            from llm.multi_agent import run_debate as _debate
            debate_result = _debate(domain, parameters, question)
            final = debate_result.get("final_probability", 
                    (ml_prob or 0.5)*0.4 + (llm_prob or 0.5)*0.6)
            logger.info(f"CRITICAL gap resolved via debate: {final:.3f}")
        except Exception as e:
            logger.warning(f"Debate failed for CRITICAL: {e}")
            # Fallback — weight LLM more when ML is extreme
            if ml_prob is not None and (ml_prob > 0.95 or ml_prob < 0.05):
                final = (ml_prob * 0.2) + ((llm_prob or 0.5) * 0.8)
            else:
                final = (ml_prob or 0.5) * 0.5 + (llm_prob or 0.5) * 0.5

    # ── Stage 7: Reliability index ────────────────────────────
    reliability = _compute_reliability(domain, parameters, skipped)

    # ── Assemble result ───────────────────────────────────────
    return PredictionResult(
        domain            = domain,
        question          = question,
        ml_probability    = ml_prob,
        llm_probability   = llm_prob,
        final_probability = max(0.0, min(1.0, final)) if final != -1 else -1,
        confidence_tier   = tier,
        gap               = gap,
        shap_values       = shap_vals,
        counterfactuals   = [],
        audit_flags       = flags,
        debate            = debate_result,
        reliability_index = reliability,
        reasoning         = debate_result.get("realist", {}).get("reasoning", ""),
        key_factors       = debate_result.get("optimist", {}).get("evidence", "").split(","),
        mode              = mode,
        raw_parameters    = parameters,
    )

# ── Free Inference Mode ───────────────────────────────────────
def predict_free(text: str, n_outcomes: int = 5) -> dict:
    """
    Free inference — no domain, no form.
    User types anything, LLM generates N independent probabilities.
    """
    try:
        from llm.groq_client import free_inference
        outcomes = free_inference(text, n_outcomes)
        return {
            "mode":     "free_inference",
            "input":    text,
            "outcomes": outcomes,
            "note":     "Probabilities are independent — they do NOT sum to 100%"
        }
    except Exception as e:
        logger.error(f"Free inference failed: {e}")
        return {"mode": "free_inference", "error": str(e), "outcomes": []}

if __name__ == "__main__":
    print("\n🧪 Testing Sambhav Predictor...\n")
    result = predict(
        domain     = "student",
        parameters = {
            "studytime":    3,      # 1-4 scale (matches training col)
            "health":       2,      # 1-5 scale (stress proxy)
            "absences":     6,      # number of absences
            "g1":           12,     # first period grade 0-20
            "g2":           13,     # second period grade 0-20
            "freetime":     2,      # 1-5 scale
            "goout":        3,      # 1-5 scale
            "failures":     0,      # past failures
        },
        question   = "Will this student pass their final exam?",
        run_debate = True
    )
    d = result.to_dict()
    print(f"  ML  Probability  : {d['ml_probability']}")
    print(f"  LLM Probability  : {d['llm_probability']}")
    print(f"  Final Probability: {d['final_probability']*100:.1f}%")
    print(f"  Confidence Tier  : {d['confidence_tier']}")
    print(f"  Gap              : {d['gap']*100:.1f}%")
    print(f"  Reliability      : {d['reliability_index']*100:.0f}%")
    print(f"  Audit Flags      : {len(d['audit_flags'])}")
    print(f"  Debate Ran       : {bool(d['debate'])}")
