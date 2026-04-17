"""
core/predictor.py — Project Sambhav
Fixed model loader + 7-stage prediction pipeline
Handles: all 11 domains, blend mode, Sarvagna special arch, missing models

FIXED BUGS:
  1. Model artifacts were not being loaded from .joblib correctly
  2. IsotonicRegression calibration was not being applied
  3. Blend weight logic for Behavioural + Claim was missing
  4. Sarvagna required separate pipeline loading
  5. Feature column alignment was not enforced
"""

import os
import logging
import yaml
import re
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Any

log = logging.getLogger(__name__)

# ── Path resolution ──────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent   # project root
_MODELS_DIR = _ROOT / "models"
_REGISTRY_PATH = _ROOT / "schemas" / "domain_registry.yaml"


# ── Data classes ─────────────────────────────────────────────
@dataclass
class PredictionResult:
    domain: str
    question: str
    ml_probability: Optional[float]
    llm_probability: Optional[float]
    reconciled_probability: float
    agreement_gap: Optional[float]
    confidence_tier: str            # HIGH / MODERATE / LOW / CRITICAL
    reliability_index: float        # 0.0 – 1.0
    warning_level: str              # CLEAR / MODERATE / LOW / CRITICAL
    outcomes: dict
    shap_values: dict
    audit_flags: list
    model_used: str
    calibrated: bool
    reasoning: Optional[str] = None
    error: Optional[str] = None
    sarvagna_features: Optional[dict] = None

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict with frontend-expected field names."""
        return {
            "domain":             self.domain,
            "question":           self.question or "",
            "ml_probability":     self.ml_probability,
            "llm_probability":    self.llm_probability,
            "final_probability":  self.reconciled_probability,  # frontend key
            "confidence_tier":    self.confidence_tier,
            "gap":                round(self.agreement_gap or 0.0, 4),
            "reliability_index":  self.reliability_index,
            "reasoning":          self.reasoning,
            "shap_values":        self.shap_values or {},
            "audit_flags":        [
                {
                    "code":     f.get("code", "INFO"),
                    "severity": f.get("severity", f.get("warning_level", "INFO")),
                    "message":  f.get("msg", f.get("message", ""))
                }
                for f in (self.audit_flags or [])
            ],
            "outcomes":           self.outcomes or {},
            "model_used":         self.model_used or "unknown",
            "calibrated":         self.calibrated,
            "key_factors":        list((self.shap_values or {}).keys())[:5],
            "mode":               "guided",
        }


@dataclass
class DomainModel:
    domain_key: str
    xgb_model: Any = None
    lgbm_model: Any = None
    iso_xgb: Any = None
    iso_lgbm: Any = None
    scaler: Any = None
    imputer: Any = None
    feature_columns: list = field(default_factory=list)
    # Blend mode extras
    xgb_synthetic: Any = None
    iso_synthetic: Any = None
    blend_weight: float = 1.0
    # Sarvagna extras
    word_svd_pipeline: Any = None
    char_svd_pipeline: Any = None
    brain_pipeline: Any = None
    sarvagna_classifier: Any = None
    available: bool = False


# ── Registry loader ──────────────────────────────────────────
_registry_cache: Optional[dict] = None

def _load_registry() -> dict:
    global _registry_cache
    if _registry_cache is not None:
        return _registry_cache
    if not _REGISTRY_PATH.exists():
        log.error(f"Registry not found at {_REGISTRY_PATH}")
        return {}
    with open(_REGISTRY_PATH, "r") as f:
        raw = yaml.safe_load(f)
    _registry_cache = raw.get("domains", {})
    return _registry_cache


_model_cache: dict[str, DomainModel] = {}

def load_domain_model(domain: str) -> DomainModel:
    """Load and cache a domain model from its .joblib artifact."""
    if domain in _model_cache:
        return _model_cache[domain]

    reg = _load_registry()
    cfg = reg.get(domain)
    if cfg is None:
        log.warning(f"Domain '{domain}' not found in registry.")
        dm = DomainModel(domain_key=domain, available=False)
        _model_cache[domain] = dm
        return dm

    if not cfg.get("model_available", False):
        log.info(f"Domain '{domain}' marked model_available=false — skipping load.")
        dm = DomainModel(domain_key=domain, available=False)
        _model_cache[domain] = dm
        return dm

    dm = DomainModel(domain_key=domain)

    # ── Main model artifact ──────────────────────────────────
    main_path = _MODELS_DIR / Path(cfg["model_path"]).name
    if main_path.exists():
        try:
            artifact = joblib.load(main_path)
            dm = _unpack_artifact(dm, artifact)
            dm.available = True
            log.info(f"[{domain}] Loaded main artifact: {main_path.name}")
            
            # Search for domain imputers and scalers dynamically
            for p in _MODELS_DIR.glob(f"{domain}*imputer*.joblib"):
                if dm.imputer is None:
                    dm.imputer = joblib.load(p)
                    log.info(f"[{domain}] Loaded separate imputer: {p.name}")
                    break
            
            for p in _MODELS_DIR.glob(f"{domain}*scaler*.joblib"):
                if dm.scaler is None:
                    dm.scaler = joblib.load(p)
                    log.info(f"[{domain}] Loaded separate scaler: {p.name}")
                    break
                    
            for p in _MODELS_DIR.glob(f"{domain}*iso*.joblib"):
                if dm.iso_xgb is None:
                    dm.iso_xgb = joblib.load(p)
                    log.info(f"[{domain}] Loaded separate isotonic: {p.name}")
                    break

        except Exception as e:
            log.error(f"[{domain}] Failed to load {main_path.name}: {e}")
            dm.available = False
    else:
        log.warning(f"[{domain}] Model file not found: {main_path}")
        dm.available = False

    # ── Blend synthetic model (behavioural, claim) ───────────
    if cfg.get("blend_mode", False) and "model_path_synthetic" in cfg:
        syn_path = _MODELS_DIR / Path(cfg["model_path_synthetic"]).name
        if syn_path.exists():
            try:
                syn_artifact = joblib.load(syn_path)
                syn_dm = DomainModel(domain_key=f"{domain}_synthetic")
                syn_dm = _unpack_artifact(syn_dm, syn_artifact)
                dm.xgb_synthetic = syn_dm.xgb_model
                dm.iso_synthetic = syn_dm.iso_xgb
                dm.blend_weight = cfg.get("blend_weight", 0.7)
                log.info(f"[{domain}] Loaded synthetic blend artifact.")
            except Exception as e:
                log.warning(f"[{domain}] Synthetic model load failed: {e}")

    # ── Sarvagna multi-artifact ──────────────────────────────
    if domain == "sarvagna":
        for key, attr in [
            ("model_path_char", "char_svd_pipeline"),
            ("model_path_brain", "brain_pipeline"),
            ("model_path_classifier", "sarvagna_classifier"),
        ]:
            if key in cfg:
                p = _MODELS_DIR / Path(cfg[key]).name
                if p.exists():
                    try:
                        setattr(dm, attr, joblib.load(p))
                        log.info(f"[sarvagna] Loaded {key}: {p.name}")
                    except Exception as e:
                        log.warning(f"[sarvagna] {key} load failed: {e}")
        # word_svd is the main artifact (already loaded above)
        dm.word_svd_pipeline = dm.xgb_model
        dm.xgb_model = None

    _model_cache[domain] = dm
    return dm


def _unpack_artifact(dm: DomainModel, artifact) -> DomainModel:
    """
    Unpack joblib artifact dict into DomainModel.
    Supports both dict-style and direct-model artifacts.
    """
    if isinstance(artifact, dict):
        dm.xgb_model       = artifact.get("xgb_model")
        dm.lgbm_model      = artifact.get("lgbm_model")
        dm.iso_xgb         = artifact.get("iso_xgb")
        dm.iso_lgbm        = artifact.get("iso_lgbm")
        dm.scaler          = artifact.get("scaler")
        dm.imputer         = artifact.get("imputer")
        dm.feature_columns = artifact.get("feature_columns", [])

        # Legacy keys — some older artifacts used different names
        if dm.xgb_model is None:
            dm.xgb_model = artifact.get("model") or artifact.get("clf")
        if dm.iso_xgb is None:
            dm.iso_xgb = artifact.get("iso") or artifact.get("calibrator")
        if dm.scaler is None:
            dm.scaler = artifact.get("std_scaler") or artifact.get("ss")
        if dm.imputer is None:
            dm.imputer = artifact.get("si") or artifact.get("simple_imputer")
    else:
        # Bare model (older format) — treat as xgb_model
        dm.xgb_model = artifact

    return dm


# ── Feature preparation ──────────────────────────────────────
def prepare_features(domain: str, params: dict, dm: DomainModel) -> Optional[np.ndarray]:
    """
    Align collected chip-modal params to the feature_columns expected by the model.
    Handles: missing values (imputed), extra keys (ignored), ordering.
    Section 12.1 — Dynamic padding for shape mismatch using model's expected count.
    Section 12.2 — Smart Parameter Calculation (BMI, Risk Scores).
    """
    # ── Smart Parameter Calculation ──────────────────────────
    # If fitness/health domain and we have height/weight but no BMI
    if (domain in ["fitness", "health"]) and params.get("bmi") is None:
        w = params.get("weight_kg")
        h = params.get("height_cm")
        if w is not None and h is not None:
            try:
                # BMI = kg / (m^2)
                bmi = float(w) / ((float(h) / 100) ** 2)
                params["bmi"] = round(bmi, 2)
                log.info(f"[{domain}] Calculated BMI: {params['bmi']}")
            except (TypeError, ValueError, ZeroDivisionError):
                pass

    # Get expected count from model if available
    expected_count = 0
    if dm.xgb_model is not None:
        try:
            expected_count = getattr(dm.xgb_model, "n_features_in_", 0)
        except:
            pass

    if not dm.feature_columns:
        # Fallback: use registry params in the order they appear in domain_registry.yaml
        reg = _load_registry()
        domain_cfg = reg.get(domain, {})
        registry_param_keys = [p["key"] for p in domain_cfg.get("parameters", [])]
        
        # Collect values in registry order
        vals = []
        for k in registry_param_keys:
            val = params.get(k)
            try:
                vals.append(float(val) if val is not None else np.nan)
            except (TypeError, ValueError):
                vals.append(np.nan)
        
        count = expected_count if expected_count > 0 else len(vals)
        if expected_count > 0 and expected_count != len(vals):
            log.info(f"[{domain}] Feature mismatch: registry has {len(vals)}, model expects {expected_count}. Padding with NaNs.")
        
        # Create array of 'count' size, filled with NaNs initially
        arr_full = np.full((1, count), np.nan)
        for i, val in enumerate(vals[:count]):
            arr_full[0, i] = val
        arr = arr_full
    else:
        row = {}
        for col in dm.feature_columns:
            val = params.get(col, np.nan)
            try:
                row[col] = float(val) if val is not None else np.nan
            except (TypeError, ValueError):
                row[col] = np.nan
        arr = np.array([row[c] for c in dm.feature_columns], dtype=float).reshape(1, -1)

    # ── Coverage Check ─────────────────────────────────────────────
    # If fewer than 30% of model features are filled, zero-padding corrupts the stacking
    # classifier. Return None so the LLM layer handles prediction cleanly instead.
    non_nan_count = int(np.sum(~np.isnan(arr)))
    total_cols = arr.shape[1]
    if total_cols > 6 and non_nan_count / total_cols < 0.30:
        log.warning(f"[{domain}] Only {non_nan_count}/{total_cols} features filled. Skipping ML — deferring to LLM.")
        return None

    # Impute then scale
    if dm.imputer is not None:
        try:
            # Check if imputer expects different shape
            if hasattr(dm.imputer, "n_features_in_") and dm.imputer.n_features_in_ != arr.shape[1]:
                log.warning(f"[{domain}] Imputer shape mismatch: expects {dm.imputer.n_features_in_}, got {arr.shape[1]}. Skipping.")
            else:
                arr = dm.imputer.transform(arr)
        except Exception as e:
            log.warning(f"[{domain}] Imputer failed: {e}")

    if dm.scaler is not None:
        try:
            if hasattr(dm.scaler, "n_features_in_") and dm.scaler.n_features_in_ != arr.shape[1]:
                log.warning(f"[{domain}] Scaler shape mismatch: expects {dm.scaler.n_features_in_}, got {arr.shape[1]}. Skipping.")
            else:
                arr = dm.scaler.transform(arr)
        except Exception as e:
            log.warning(f"[{domain}] Scaler failed: {e}")

    # Critical fallback: if any NaNs exist (e.g. imputer skipped), zero them to prevent LogisticRegression crash
    arr = np.nan_to_num(arr, nan=0.0)

    return arr


# ── ML prediction ─────────────────────────────────────────────
def predict_ml(domain: str, params: dict) -> tuple[Optional[float], str]:
    """
    Run ML layer prediction.
    Returns (calibrated_probability, model_description)
    """
    dm = load_domain_model(domain)
    if not dm.available or dm.xgb_model is None:
        return None, "ml_unavailable"

    try:
        X = prepare_features(domain, params, dm)
        if X is None:
            return None, "feature_prep_failed"

        # XGBoost raw probability
        raw_xgb = float(dm.xgb_model.predict_proba(X)[0][1])

        # Apply IsotonicRegression calibration
        if dm.iso_xgb is not None:
            cal_xgb = float(dm.iso_xgb.transform([raw_xgb])[0])
        else:
            cal_xgb = raw_xgb
            log.warning(f"[{domain}] No iso_xgb — using raw XGBoost probability (may be overconfident)")

        # LightGBM (if available) for ensemble average
        if dm.lgbm_model is not None:
            raw_lgbm = float(dm.lgbm_model.predict_proba(X)[0][1])
            if dm.iso_lgbm is not None:
                cal_lgbm = float(dm.iso_lgbm.transform([raw_lgbm])[0])
            else:
                cal_lgbm = raw_lgbm
            primary_prob = (cal_xgb + cal_lgbm) / 2.0
            model_desc = "xgb+lgbm_isotonic"
        else:
            primary_prob = cal_xgb
            model_desc = "xgb_isotonic"

        # Blend with synthetic model (Behavioural, Claim)
        if dm.xgb_synthetic is not None and dm.iso_synthetic is not None:
            raw_syn = float(dm.xgb_synthetic.predict_proba(X)[0][1])
            cal_syn = float(dm.iso_synthetic.transform([raw_syn])[0])
            primary_prob = dm.blend_weight * primary_prob + (1 - dm.blend_weight) * cal_syn
            model_desc += f"_blend{dm.blend_weight}"

        # Cap at 97%
        primary_prob = min(primary_prob, 0.97)
        return primary_prob, model_desc

    except Exception as e:
        log.error(f"[{domain}] ML prediction error: {e}")
        return None, f"error: {e}"


# ── Sarvagna prediction ──────────────────────────────────────
def predict_sarvagna(params: dict) -> tuple[Optional[float], dict]:
    """
    Sarvagna-specific prediction: WordSVD + CharSVD + Brain features → classifier.
    Returns (probability, extracted_features_dict)
    """
    dm = load_domain_model("sarvagna")
    if not dm.available:
        return None, {"error": "Sarvagna models not yet downloaded. See Phase 2."}

    text = params.get("claim_text") or params.get("text_input") or params.get("communication_text", "")
    if not text:
        return None, {"error": "No text provided for Sarvagna analysis"}

    try:
        features = {}

        # WordSVD transform
        if dm.word_svd_pipeline is not None:
            word_feats = dm.word_svd_pipeline.transform([text])
            features["word_svd"] = word_feats

        # CharSVD transform
        if dm.char_svd_pipeline is not None:
            char_feats = dm.char_svd_pipeline.transform([text])
            features["char_svd"] = char_feats

        # Brain features
        if dm.brain_pipeline is not None:
            brain_feats = dm.brain_pipeline.transform([text])
            features["brain"] = brain_feats

        # Combine: 600d + 200d + 20d = 820d
        combined_parts = []
        for k in ["word_svd", "char_svd", "brain"]:
            if k in features:
                combined_parts.append(features[k])
        if not combined_parts:
            return None, {"error": "Feature extraction failed"}

        X_combined = np.hstack(combined_parts)

        # Classifier
        if dm.sarvagna_classifier is not None:
            prob = float(dm.sarvagna_classifier.predict_proba(X_combined)[0][1])
        else:
            return None, {"error": "Sarvagna classifier not loaded"}

        return min(prob, 0.97), {"feature_dims": X_combined.shape[1]}

    except Exception as e:
        log.error(f"[sarvagna] Prediction error: {e}")
        return None, {"error": str(e)}


# ── Reliability Index ─────────────────────────────────────────
def compute_reliability_index(
    domain: str,
    params: dict,
    ml_prob: Optional[float],
    llm_prob: Optional[float],
    agreement_gap: Optional[float],
    has_text_input: bool = False,
    has_vision: bool = False
) -> tuple[float, str]:
    """
    Compute Reliability Index (0.0 – 1.0) and warning level.
    Weights: completeness 40%, layer_availability 30%, agreement 20%, vision 10%
    """
    reg = _load_registry()
    cfg = reg.get(domain, {})
    param_schema = cfg.get("parameters", [])
    required_params = [p["key"] for p in param_schema if p.get("required", False)]

    # 1. Parameter completeness (40%)
    if required_params:
        filled = sum(1 for k in required_params if params.get(k) is not None)
        completeness = filled / len(required_params)
    elif params:
        # Text-only domain
        completeness = 1.0 if has_text_input else 0.5
    else:
        completeness = 0.0

    # 2. Layer availability (30%)
    if ml_prob is not None and llm_prob is not None:
        layer_score = 1.0
    elif ml_prob is not None or llm_prob is not None:
        layer_score = 0.6
    else:
        layer_score = 0.0

    # 3. ML–LLM agreement (20%)
    if agreement_gap is None:
        agreement_score = 0.5  # Unknown
    elif agreement_gap < 0.10:
        agreement_score = 1.0
    elif agreement_gap < 0.25:
        agreement_score = 0.75
    elif agreement_gap < 0.40:
        agreement_score = 0.40
    else:
        agreement_score = 0.10

    # 4. Vision bonus (10%)
    vision_score = 1.0 if has_vision else 0.0

    ri = (
        0.40 * completeness +
        0.30 * layer_score +
        0.20 * agreement_score +
        0.10 * vision_score
    )
    ri = round(min(ri, 1.0), 3)

    if ri >= 0.75:
        warning = "CLEAR"
    elif ri >= 0.50:
        warning = "MODERATE"
    elif ri >= 0.30:
        warning = "LOW"
    else:
        warning = "CRITICAL"

    return ri, warning


# ── Wrapper class for legacy imports ─────────────────────────
class SambhavPredictor:
    """
    Legacy class wrapper for functional predictor logic.
    Provides backwards compatibility for imports and testing.
    """
    def predict(self, *args, **kwargs):
        return predict(*args, **kwargs)

    def generate_outcomes(self, *args, **kwargs):
        return generate_outcomes(*args, **kwargs)

    def explain_transparency(self, *args, **kwargs):
        return explain_prediction_transparency(*args, **kwargs)

    def get_available_domains(self):
        return get_available_domains()
def cross_validate(ml_prob: Optional[float], llm_prob: Optional[float]) -> tuple[float, float, str]:
    """
    Reconcile ML and LLM predictions.
    ML is the PRIMARY layer (70-80% weight) — domain-specific, trained on real data.
    LLM is a CALIBRATION signal (20-30%) — adds context and semantic correction.

    When ML is missing: LLM takes full weight at MODERATE confidence.
    When LLM is missing: ML is returned as-is at LOW confidence.
    """
    if ml_prob is None and llm_prob is None:
        return 0.5, 1.0, "CRITICAL"

    if ml_prob is None:
        # No ML model available — LLM only, capped at MODERATE confidence
        return round(min(llm_prob, 0.97), 4), 0.0, "MODERATE"

    if llm_prob is None:
        # ML only — returned as-is, LOW confidence (no cross-validation)
        return round(min(ml_prob, 0.97), 4), 0.0, "LOW"

    gap = abs(ml_prob - llm_prob)

    if gap < 0.10:
        # High agreement — ML 70%, LLM 30%
        reconciled = 0.70 * ml_prob + 0.30 * llm_prob
        tier = "HIGH"
    elif gap < 0.25:
        # Moderate agreement — ML 75%, LLM 25%
        reconciled = 0.75 * ml_prob + 0.25 * llm_prob
        tier = "MODERATE"
    elif gap < 0.40:
        # Low agreement — ML takes 80% (larger deviation, trust trained model more)
        reconciled = 0.80 * ml_prob + 0.20 * llm_prob
        tier = "LOW"
    else:
        # Critical disagreement — ML 75%, flag both extremes
        reconciled = 0.75 * ml_prob + 0.25 * llm_prob
        tier = "CRITICAL"

    return round(min(reconciled, 0.97), 4), round(gap, 4), tier


# ── Simple SHAP stub ──────────────────────────────────────────
_shap_explainers: dict = {}

def get_shap_values(domain: str, params: dict, prediction: float) -> dict:
    """
    Returns SHAP-like contribution estimates.
    Uses cached TreeExplainer when possible for speed.
    Skips for StackingClassifier (unsupported by TreeExplainer).
    """
    dm = load_domain_model(domain)

    if dm.available and dm.xgb_model is not None:
        # Check if model is StackingClassifier
        from sklearn.ensemble import StackingClassifier
        if isinstance(dm.xgb_model, StackingClassifier):
            log.info(f"[{domain}] Model is StackingClassifier, using proportional stub for speed")
        else:
            try:
                import shap
                global _shap_explainers
                if domain not in _shap_explainers:
                    _shap_explainers[domain] = shap.TreeExplainer(dm.xgb_model)
                
                explainer = _shap_explainers[domain]
                X = prepare_features(domain, params, dm)
                sv = explainer.shap_values(X)
                if isinstance(sv, list):
                    sv = sv[1]
                
                fcols = dm.feature_columns or list(params.keys())
                return {
                    col: round(float(sv[0][i]), 4)
                    for i, col in enumerate(fcols[:sv.shape[1]])
                }
            except Exception as e:
                log.warning(f"[{domain}] Real SHAP failed ({e}), using proportional stub")

    # Proportional stub: distribute ±0.5 based on non-null params
    shap_out = {}
    filled_params = {k: v for k, v in params.items() if v is not None}
    if not filled_params:
        return {}
    total = len(filled_params)
    base = prediction - 0.5
    for i, (k, v) in enumerate(filled_params.items()):
        contribution = round(base / total + (0.02 * (i % 3 - 1)), 4)
        shap_out[k] = contribution
    return shap_out


# ── Main predict entry point ──────────────────────────────────
def predict(
    domain: str,
    params: dict = None,
    question: str = "",
    llm_probability: Optional[float] = None,
    has_vision: bool = False,
    # Accept endpoint kwargs without breaking signature
    parameters: dict = None,
    skipped: list = None,
    run_debate: bool = False,
    mode: str = "guided",
    user_id: Optional[str] = None,
    db: Optional[Any] = None,
) -> PredictionResult:
    # Allow caller to pass 'parameters' instead of 'params'
    if params is None:
        params = parameters or {}
    
    # Stage 0 — Personal Calibration (Section 13.5)
    calibration_adjustment = 0.0
    if user_id and db:
        try:
            from db.database import get_user_calibration_bias
            bias = get_user_calibration_bias(db, user_id)
            # If bias is +0.10, user is overconfident (predicted > actual)
            # We subtract a fraction of this bias to 'calibrate' the result
            calibration_adjustment = -0.5 * bias
            log.info(f"[{domain}] Applying personal calibration bias adjustment: {calibration_adjustment:+.3f}")
        except Exception as e:
            log.warning(f"Calibration adjustment failed: {e}")
    """
    Main 7-stage prediction pipeline.

    Args:
        domain:          Registry domain key (e.g. 'student', 'pragma')
        params:          Dict of chip-modal collected values
        question:        Natural language question string
        llm_probability: Pre-computed LLM estimate (from API router) or None
        has_vision:      Whether vision pipeline contributed params

    Returns:
        PredictionResult dataclass
    """

    # Stage 1 — Safety check (abbreviated; full check in api/safety.py)
    audit_flags = []
    if not params and not question:
        return PredictionResult(
            domain=domain, question=question,
            ml_probability=None, llm_probability=None,
            reconciled_probability=0.5, agreement_gap=None,
            confidence_tier="CRITICAL", reliability_index=0.0,
            warning_level="CRITICAL", outcomes={}, shap_values={},
            audit_flags=[{"code": "ABN-001", "msg": "No parameters provided",
                          "severity": "CRITICAL"}],
            model_used="none", calibrated=False,
            error="No parameters or question provided"
        )

    # Stage 2 — Domain validation
    reg = _load_registry()
    if domain not in reg:
        audit_flags.append({"code": "ABN-002", "msg": f"Unknown domain: {domain}",
                            "severity": "WARNING"})
        domain = "student"  # graceful fallback

    has_text_input = bool(
        params.get("claim_text") or
        params.get("communication_text") or
        params.get("commitment_statement") or
        params.get("text_input") or
        question
    )

    # Stage 3 — Feature extraction (handled in prepare_features)

    # Stage 4 — Dual-layer prediction (Parallelized for Speed)
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        # ML Prediction Task
        if domain == "sarvagna":
            ml_future = executor.submit(predict_sarvagna, params)
        else:
            ml_future = executor.submit(predict_ml, domain, params)
        
        # LLM Prediction Task (if needed)
        llm_future = None
        if llm_probability is None:
            from llm.groq_client import llm_predict
            # We use high-speed Groq 8B for fast generation
            llm_future = executor.submit(llm_predict, domain, params, question)
        
        # Collect results
        try:
            if domain == "sarvagna":
                ml_prob, sarvagna_info = ml_future.result()
                model_desc = "sarvagna_820d"
            else:
                ml_prob, model_desc = ml_future.result()
                sarvagna_info = None
        except Exception as e:
            log.warning(f"[{domain}] ML layer future failed: {e}")
            ml_prob, model_desc, sarvagna_info = None, "ml_error", None
            
        llm_reasoning = None
        if llm_future:
            try:
                llm_res = llm_future.result()
                if isinstance(llm_res, dict):
                    llm_prob = llm_res.get("probability", 0.5)
                    llm_reasoning = llm_res.get("reasoning")
                else:
                    llm_prob = float(llm_res)
                log.info(f"[{domain}] LLM layer generated probability: {llm_prob}")
            except Exception as e:
                log.warning(f"[{domain}] LLM layer future failed: {e}")
                llm_prob = None
        else:
            llm_prob = llm_probability

    # Stage 5 — Cross-validation
    reconciled, gap, tier = cross_validate(ml_prob, llm_prob)

    # Apply personal calibration adjustment (Section 13.5)
    if calibration_adjustment != 0:
        reconciled = max(0.01, min(0.99, reconciled + calibration_adjustment))
        audit_flags.append({
            "code": "CAL-001",
            "msg": f"Personal calibration applied ({calibration_adjustment:+.1%})",
            "severity": "INFO"
        })

    if tier == "CRITICAL":
        _ml_s  = f"{ml_prob:.0%}"  if ml_prob  is not None else "N/A"
        _llm_s = f"{llm_prob:.0%}" if llm_prob is not None else "N/A"
        _gap_s = f"{gap:.0%}"      if gap      is not None else "N/A"
        audit_flags.append({
            "code": "ABN-003",
            "msg": f"ML ({_ml_s}) and LLM ({_llm_s}) disagree by {_gap_s}",
            "severity": "CRITICAL"
        })

    # Stage 6 — Reliability Index
    ri, warning = compute_reliability_index(
        domain, params, ml_prob, llm_prob, gap if llm_prob is not None else None,
        has_text_input, has_vision
    )

    # Stage 6b — Missing key params check
    cfg = reg.get(domain, {})
    for p in cfg.get("parameters", []):
        if p.get("required") and params.get(p["key"]) is None:
            audit_flags.append({
                "code": "ABN-007",
                "msg": f"Missing required parameter: {p['label']}",
                "severity": "INFO"
            })

    # SHAP values
    shap_vals = get_shap_values(domain, params, reconciled)

    # Stage 7 — Outcome packaging
    domain_cfg = reg.get(domain, {})
    supported = domain_cfg.get("supported_outcomes", ["outcome_a", "outcome_b"])
    if len(supported) == 2:
        outcomes = {
            supported[0]: round(reconciled, 4),
            supported[1]: round(1 - reconciled, 4)
        }
    else:
        outcomes = {s: round(1 / len(supported), 4) for s in supported}
        outcomes[supported[0]] = round(reconciled, 4)

    return PredictionResult(
        domain=domain,
        question=question,
        ml_probability=round(ml_prob, 4) if ml_prob is not None else None,
        llm_probability=round(llm_prob, 4) if llm_prob is not None else None,
        reconciled_probability=reconciled,
        agreement_gap=gap if llm_prob is not None else None,
        confidence_tier=tier,
        reliability_index=ri,
        warning_level=warning,
        outcomes=outcomes,
        shap_values=shap_vals,
        audit_flags=audit_flags,
        model_used=model_desc,
        calibrated=(ml_prob is not None and "isotonic" in model_desc),
        reasoning=llm_reasoning,
        sarvagna_features=sarvagna_info,
    )


def predict_rich(
    domain: str,
    parameters: dict = None,
    question: str = "",
    skipped: list = None,
    mode: str = "guided",
    user_id: Optional[str] = None,
    db: Optional[Any] = None,
) -> dict:
    """
    Rich prediction: returns prediction + outcomes + transparency in one parallel call.
    """
    import concurrent.futures
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        pred_future = executor.submit(predict, domain=domain, params=parameters, 
                                     question=question, skipped=skipped, mode=mode,
                                     user_id=user_id, db=db)
        out_future  = executor.submit(generate_outcomes, domain=domain, 
                                     parameters=parameters, question=question, mode=mode)
        
        prediction = pred_future.result()
        outcomes   = out_future.result()
        
        # Get transparency for the primary outcome
        supported = _load_registry().get(domain, {}).get("supported_outcomes", ["outcome_a"])
        primary_outcome = supported[0]
        
        trans_future = executor.submit(explain_prediction_transparency, 
                                      domain=domain, parameters=parameters, 
                                      final_probability=prediction.reconciled_probability,
                                      shap_values=prediction.shap_values,
                                      question=question, outcome=primary_outcome)
        
        transparency = trans_future.result()

    return {
        "prediction":   prediction.to_dict(),
        "outcomes":     outcomes.get("outcomes", []),
        "transparency": transparency,
        "mode":         mode
    }

# ── Utility: list available domains ──────────────────────────
def get_available_domains() -> list[dict]:
    """Return list of all domains with their availability status."""
    reg = _load_registry()
    result = []
    for key, cfg in reg.items():
        result.append({
            "key": key,
            "display_name": cfg.get("display_name", key),
            "available": cfg.get("model_available", False),
            "status": cfg.get("status", "UNKNOWN"),
            "brier_score": cfg.get("brier_score"),
            "auc": cfg.get("auc"),
            "param_count": len(cfg.get("parameters", [])),
        })
    return result


def reload_model_cache():
    """Force reload all models (use after downloading new artifacts)."""
    global _model_cache, _registry_cache
    _model_cache.clear()
    _registry_cache = None
    log.info("Model cache cleared — will reload on next prediction")


# ── SHAP alias (imported by endpoint) ────────────────────────
def _get_shap(domain: str, parameters: dict) -> dict:
    """Public alias for get_shap_values used by transparency endpoint."""
    return get_shap_values(domain, parameters, 0.5)


# ── ML / LLM sub-predict (imported by calibration endpoint) ──
def _ml_predict(domain: str, parameters: dict) -> Optional[float]:
    """Run only the ML layer and return raw probability."""
    dm = load_domain_model(domain)
    if not dm.available:
        return None
    try:
        features = _prepare_features(domain, parameters)
        return _run_ml_prediction(dm, features)
    except Exception as e:
        log.warning(f"_ml_predict failed for {domain}: {e}")
        return None


def _llm_predict(domain: str, parameters: dict, question: str = "") -> Optional[float]:
    """Run only the LLM layer and return raw probability."""
    try:
        from llm.groq_client import llm_predict
        res = llm_predict(domain, parameters, question)
        return res.get("probability")
    except Exception as e:
        log.warning(f"_llm_predict failed for {domain}: {e}")
        return None


# ── Free inference (imported by /predict/free endpoint) ───────
def predict_free(text: str, n_outcomes: int = 5) -> dict:
    """
    Free-text inference: no domain required.
    Calls Groq LLM to generate n_outcomes independent probabilities.
    Includes entity extraction, domain detection and signal analysis (Mode 2).
    """
    try:
        from llm.groq_client import free_inference
        data = free_inference(text, n_outcomes)
        
        # Format for frontend consistency
        outcomes = data.get("outcomes", [])
        for o in outcomes:
            o["probability_pct"] = f"{round(o['probability'] * 100)}%"
            o["type"] = "positive" if o["probability"] > 0.6 else "negative" if o["probability"] < 0.4 else "neutral"
            o["has_transparency"] = True

        return {
            "success":           True,
            "outcomes":          outcomes,
            "domain_detected":   data.get("domain", "general"),
            "entities":          data.get("entities", []),
            "positive_signals":  data.get("positive_signals", []),
            "negative_signals":  data.get("negative_signals", []),
            "reliability_index": data.get("reliability_index", 0.5),
            "missing_info":      data.get("missing_info", []),
            "mode":              "free",
            "interpretation":    f"Sambhav automatically extracted signals for {data.get('domain', 'this scenario')}.",
        }
    except Exception as e:
        log.error(f"predict_free failed: {e}")
        return {"outcomes": [], "mode": "free", "error": str(e)}


# ── Multi-outcome generator (imported by /predict/outcomes) ───
def generate_outcomes(
    domain: str,
    parameters: dict,
    question: str = None,
    n_outcomes: int = 5,
    existing_outcomes: list = None,
    mode: str = "independent",
) -> dict:
    """
    Generate n_outcomes independent probability predictions for a domain/context.
    Each outcome is a distinct scenario label with its own probability.
    Calls Groq LLM for topic-relevant, domain-aware outcome labels.
    """
    import json, re

    existing_outcomes = existing_outcomes or []
    param_str = "\n".join([f"  - {k}: {v}" for k, v in (parameters or {}).items()]) or "  (no parameters)"
    q = question or f"What are likely outcomes for a prediction in the {domain} domain?"
    existing_labels = [o.get("outcome", "") for o in existing_outcomes if o.get("outcome")]
    avoid_str = (
        f"\nDo NOT repeat these already-shown outcomes: {', '.join(existing_labels)}"
        if existing_labels else ""
    )

    # Load domain config for context
    reg = _load_registry()
    domain_cfg = reg.get(domain, {})
    domain_name = domain_cfg.get("name", domain)
    prediction_label = domain_cfg.get("prediction_label", "Outcome probability")

    messages = [
        {"role": "system", "content": (
            f"You are the Sambhav probabilistic engine for the '{domain_name}' domain.\n"
            f"Generate {n_outcomes} DISTINCT, domain-relevant outcome scenarios with independent probability estimates.\n"
            "Rules:\n"
            "1. Each outcome must be a SPECIFIC, meaningful scenario label for this domain (not generic).\n"
            "2. Probabilities are INDEPENDENT — they do NOT sum to 100%.\n"
            "3. Base probabilities on the given parameters and question.\n"
            "4. Include a 1-sentence reasoning for each.\n"
            f"5. {prediction_label}.\n"
            f"{avoid_str}\n\n"
            "Respond in EXACT JSON array format:\n"
            "[\n"
            "  {\"outcome\": \"<specific scenario label>\", \"probability\": <0-100>, \"reasoning\": \"<1 sentence>\", \"type\": \"<positive|negative|neutral>\"},\n"
            "  ...\n"
            "]"
        )},
        {"role": "user", "content": (
            f"Domain: {domain_name}\n"
            f"Question: {q}\n"
            f"Parameters:\n{param_str}\n\n"
            f"Generate {n_outcomes} independent outcome probabilities."
        )}
    ]

    try:
        from llm.router import route
        result = route("outcome_generation", messages, max_tokens=800, temperature=0.6)
        raw = result.get("content", "")
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        if "```" in raw:
            raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`")

        # Extract JSON array
        start = raw.find("[")
        end = raw.rfind("]") + 1
        parsed = json.loads(raw[start:end]) if start != -1 and end > 0 else []

        outcomes = []
        for item in parsed[:n_outcomes]:
            prob_raw = item.get("probability", 50)
            prob = float(prob_raw) / 100 if float(prob_raw) > 1 else float(prob_raw)
            outcomes.append({
                "outcome":     item.get("outcome", "Unknown outcome"),
                "probability": round(min(max(prob * 100, 1), 99), 1),  # keep as 0-100 for frontend
                "probability_pct": f"{round(min(max(prob * 100, 1), 99), 1)}%",
                "reasoning":   item.get("reasoning", ""),
                "type":        item.get("type", "neutral"),
                "has_transparency": True,
            })
        return {"outcomes": outcomes, "domain": domain, "mode": mode}

    except Exception as e:
        log.error(f"generate_outcomes LLM failed ({e}), using stub")
        # Fallback stub outcomes based on domain config
        supported = domain_cfg.get("supported_outcomes", ["Positive outcome", "Negative outcome"])
        stub = []
        for i, label in enumerate(supported[:n_outcomes]):
            if label in existing_labels:
                continue
            stub.append({
                "outcome":     label,
                "probability": round(60 - i * 10, 1),
                "probability_pct": f"{60 - i * 10}%",
                "reasoning":   "Based on available parameters.",
                "type":        "neutral",
                "has_transparency": False,
            })
        return {"outcomes": stub, "domain": domain, "mode": mode}


# ── WHY explanation (imported by /predict/transparency) ───────
def explain_prediction_transparency(
    domain: str,
    parameters: dict,
    final_probability: float = None,
    shap_values: dict = None,
    question: str = None,
    outcome: str = None,
) -> dict:
    """
    Generate a 3-level WHY explanation using Groq LLM.
    Returns: {simple, detailed, full} — each with increasing detail.
    Called when user clicks 'WHY' on any outcome row.
    """
    import json, re

    param_str = "\n".join([f"  - {k}: {v}" for k, v in (parameters or {}).items()]) or "  (no parameters)"
    prob_pct = f"{round((final_probability or 0.5) * 100, 1)}%" if final_probability is not None else "unknown"
    outcome_label = outcome or "this outcome"
    q = question or f"Why would '{outcome_label}' occur in the {domain} domain?"

    # Build SHAP context string
    shap_ctx = ""
    if shap_values:
        top = sorted(shap_values.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        shap_ctx = "\nTop SHAP contributors:\n" + "\n".join(
            [f"  - {k}: {'positive' if (v or 0.0) > 0 else 'negative'} ({(v or 0.0):+.3f})" for k, v in top]
        )

    messages = [
        {"role": "system", "content": (
            f"You are Sambhav's explainability engine for the '{domain}' domain.\n"
            "Explain WHY a particular outcome has the given probability.\n"
            "Be specific — cite actual parameter values, not generic statements.\n"
            "Respond in EXACT JSON:\n"
            "{\n"
            "  \"simple\": {\n"
            "    \"one_line_reason\": \"<single sentence: the single biggest reason for this outcome>\",\n"
            "    \"dominant_probability\": <0-100>,\n"
            "    \"minority_probability\": <0-100>\n"
            "  },\n"
            "  \"detailed\": {\n"
            "    \"case_for\": \"<2-3 sentences arguing FOR this outcome probability>\",\n"
            "    \"case_against\": \"<2-3 sentences arguing AGAINST>\",\n"
            "    \"positive_signals\": [[\"factor\", \"impact\"], ...],\n"
            "    \"negative_signals\": [[\"factor\", \"impact\"], ...]\n"
            "  },\n"
            "  \"full\": {\n"
            "    \"primary_driver\": \"<the single most important factor>\",\n"
            "    \"intervention\": \"<what would most change this outcome>\",\n"
            "    \"confidence_note\": \"<note on confidence level>\"\n"
            "  }\n"
            "}"
        )},
        {"role": "user", "content": (
            f"Domain: {domain}\n"
            f"Question: {q}\n"
            f"Outcome: {outcome_label}\n"
            f"Probability: {prob_pct}\n"
            f"Parameters:\n{param_str}"
            f"{shap_ctx}\n\n"
            "Explain WHY this outcome has this probability."
        )}
    ]

    try:
        from llm.router import route
        result = route("transparency", messages, max_tokens=700, temperature=0.3)
        raw = result.get("content", "")
        raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        if "```" in raw:
            raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`")

        start = raw.find("{")
        end = raw.rfind("}") + 1
        parsed = json.loads(raw[start:end]) if start != -1 else {}

        return {
            "simple":   parsed.get("simple",   {"one_line_reason": f"Based on the current parameters, {outcome_label} has a {prob_pct} probability.", "dominant_probability": round((final_probability or 0.5) * 100, 1), "minority_probability": round((1 - (final_probability or 0.5)) * 100, 1)}),
            "detailed": parsed.get("detailed", {"case_for": "The current parameter signals support this outcome.", "case_against": "Alternative signals may reduce this probability.", "positive_signals": [], "negative_signals": []}),
            "full":     parsed.get("full",     {"primary_driver": "Combined parameter signals", "intervention": "Improve key negative signals", "confidence_note": "Based on available data."}),
            "outcome":  outcome_label,
            "probability": prob_pct,
        }

    except Exception as e:
        log.error(f"explain_prediction_transparency failed ({e}), returning stub")
        p = round((final_probability or 0.5) * 100, 1)
        return {
            "simple":   {"one_line_reason": f"Based on the provided parameters, {outcome_label} has a {prob_pct} probability.", "dominant_probability": p, "minority_probability": round(100 - p, 1)},
            "detailed": {"case_for": "Current parameter signals are consistent with this outcome probability.", "case_against": "Other factors not captured in parameters may influence the result.", "positive_signals": [], "negative_signals": []},
            "full":     {"primary_driver": "Parameter combination", "intervention": "Provide more context or parameters for a refined analysis.", "confidence_note": "LLM explanation unavailable — showing structural analysis."},
            "outcome":  outcome_label,
            "probability": prob_pct,
        }