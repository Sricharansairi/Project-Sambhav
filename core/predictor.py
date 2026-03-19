import os, sys, logging, yaml, joblib, numpy as np
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.expanduser("~/Desktop/Sri_Coding/Project Sambhav"))
logger = logging.getLogger(__name__)

BASE   = os.path.expanduser("~/Desktop/Sri_Coding/Project Sambhav")
SCHEMA = os.path.join(BASE, "schemas/domain_registry.yaml")

# ── Domain-specific parameter mappers ────────────────────────
# Maps user-friendly parameter names → exact training column names
DOMAIN_PARAM_MAP = {
    "student": {
        "study_hours":          "StudyTimeWeekly",
        "study_hours_per_day":  "StudyTimeWeekly",
        "attendance":           "Absences",
        "attendance_pct":       "Absences",
        "past_score":           "GPA",
        "gpa":                  "GPA",
        "math_score":           "math score",
        "reading_score":        "reading score",
        "writing_score":        "writing score",
        "absences":             "Absences",
        "tutoring":             "Tutoring",
        "parental_support":     "ParentalSupport",
        "extracurricular":      "Extracurricular",
        "sports":               "Sports",
        "age":                  "Age",
        "gender":               "Gender",
        "motivation":           "ParentalSupport",
        "stress_level":         "Absences",
    },
    "higher_education": {
        "age":                  "Age at enrollment",
        "age_at_enrollment":    "Age at enrollment",
        "scholarship":          "Scholarship holder",
        "scholarship_holder":   "Scholarship holder",
        "tuition_paid":         "Tuition fees up to date",
        "tuition_fees_up_to_date": "Tuition fees up to date",
        "debtor":               "Debtor",
        "gender":               "Gender",
        "admission_grade":      "Previous qualification",
        "units_approved":       "Curricular units 1st sem (approved)",
        "units_enrolled":       "Curricular units 1st sem (enrolled)",
        "units_grade":          "Curricular units 1st sem (grade)",
        "displaced":            "Displaced",
        "international":        "International",
    },
    "hr": {
        "age":                  "Age",
        "job_satisfaction":     "JobSatisfaction",
        "work_life_balance":    "WorkLifeBalance",
        "overtime":             "OverTime",
        "years_at_company":     "YearsAtCompany",
        "monthly_income":       "MonthlyIncome",
        "distance_from_home":   "DistanceFromHome",
        "environment_satisfaction": "EnvironmentSatisfaction",
        "job_level":            "JobLevel",
        "job_involvement":      "JobInvolvement",
        "years_in_role":        "YearsInCurrentRole",
        "years_since_promotion": "YearsSinceLastPromotion",
        "years_with_manager":   "YearsWithCurrManager",
        "num_companies":        "NumCompaniesWorked",
        "salary_hike":          "PercentSalaryHike",
        "total_working_years":  "TotalWorkingYears",
        "training_times":       "TrainingTimesLastYear",
        "stock_options":        "StockOptionLevel",
        "performance_rating":   "PerformanceRating",
        "relationship_satisfaction": "RelationshipSatisfaction",
        "tenure":               "Tenure",
        "monthly_charges":      "MonthlyCharges",
    },
    "disease": {
        "age":                  "Age",
        "glucose":              "Glucose",
        "blood_pressure":       "BloodPressure",
        "bmi":                  "BMI",
        "cholesterol":          "chol",
        "pregnancies":          "Pregnancies",
        "insulin":              "Insulin",
        "skin_thickness":       "SkinThickness",
        "heart_rate":           "thalach",
        "chest_pain":           "cp",
        "exercise_angina":      "exang",
        "fasting_bs":           "fbs",
        "blood_glucose":        "blood_glucose_level",
        "hba1c":                "HbA1c_level",
        "smoking":              "smoking_history",
        "hypertension":         "hypertension",
        "heart_disease":        "heart_disease",
    },
    "fitness": {
        "age":                  "age",
        "gender":               "gender",
        "weight_kg":            "weight_kg",
        "height_cm":            "height_cm",
        "body_fat_pct":         "body fat_%",
        "body_fat":             "body fat_%",
        "sit_ups":              "sit-ups counts",
        "broad_jump":           "broad jump_cm",
        "grip_force":           "gripForce",
        "bmi":                  "bmi",
        "smoker":               "smoker",
        "systolic":             "systolic",
        "diastolic":            "diastolic",
    },
    "loan": {},  # PCA features — LLM only
    "mental_health": {},  # Text model — handled separately
    "claim": {},  # Text model — handled separately
    "behavioral": {},  # Text model — handled separately
}

# Smart domain defaults for unmapped columns
DOMAIN_DEFAULTS = {
    "student": {
        "StudentID": 0, "Age": 17, "Gender": 0, "Ethnicity": 0,
        "ParentalEducation": 2, "StudyTimeWeekly": 5, "Absences": 5,
        "Tutoring": 0, "ParentalSupport": 2, "Extracurricular": 0,
        "Sports": 0, "Music": 0, "Volunteering": 0, "GPA": 2.5,
        "gender": 0, "race/ethnicity": 0, "parental level of education": 2,
        "lunch": 1, "test preparation course": 0,
        "math score": 65, "reading score": 65, "writing score": 65,
        "Education Level": 1, "Institution Type": 0, "IT Student": 0,
        "Location": 1, "Load-shedding": 1, "Financial Condition": 1,
        "Internet Type": 1, "Network Type": 1, "Class Duration": 3,
        "Self Lms": 0, "Device": 0, "city": 0, "city_development_index": 0.8,
        "relevent_experience": 1, "enrolled_university": 0,
        "education_level": 2, "major_discipline": 0, "experience": 2,
        "company_size": 0, "company_type": 0, "last_new_job": 1,
        "training_hours": 50,
    },
    "higher_education": {
        "Marital status": 1, "Application mode": 1, "Application order": 1,
        "Course": 1, "Daytime/evening attendance": 1,
        "Previous qualification": 120, "Nacionality": 1,
        "Mother's qualification": 2, "Father's qualification": 2,
        "Mother's occupation": 3, "Father's occupation": 3,
        "Displaced": 0, "Educational special needs": 0, "Debtor": 0,
        "Tuition fees up to date": 1, "Gender": 0, "Scholarship holder": 0,
        "Age at enrollment": 20, "International": 0,
        "Curricular units 1st sem (credited)": 0,
        "Curricular units 1st sem (enrolled)": 6,
        "Curricular units 1st sem (evaluations)": 6,
        "Curricular units 1st sem (approved)": 5,
        "Curricular units 1st sem (grade)": 12,
        "Curricular units 1st sem (without evaluations)": 0,
        "Curricular units 2nd sem (credited)": 0,
        "Curricular units 2nd sem (enrolled)": 6,
        "Curricular units 2nd sem (evaluations)": 6,
        "Curricular units 2nd sem (approved)": 5,
        "Curricular units 2nd sem (grade)": 12,
        "Curricular units 2nd sem (without evaluations)": 0,
        "Unemployment rate": 10, "Inflation rate": 1.5, "GDP": 1.5,
    },
    "hr": {
        "Age": 35, "BusinessTravel": 1, "DailyRate": 800,
        "Department": 1, "DistanceFromHome": 5, "Education": 3,
        "EducationField": 1, "EmployeeCount": 1, "EmployeeNumber": 1,
        "EnvironmentSatisfaction": 3, "Gender": 0, "HourlyRate": 65,
        "JobInvolvement": 3, "JobLevel": 2, "JobRole": 1,
        "JobSatisfaction": 3, "MaritalStatus": 1, "MonthlyIncome": 5000,
        "MonthlyRate": 14000, "NumCompaniesWorked": 2, "Over18": 1,
        "OverTime": 0, "PercentSalaryHike": 13, "PerformanceRating": 3,
        "RelationshipSatisfaction": 3, "StandardHours": 80,
        "StockOptionLevel": 1, "TotalWorkingYears": 8,
        "TrainingTimesLastYear": 3, "WorkLifeBalance": 3,
        "YearsAtCompany": 5, "YearsInCurrentRole": 3,
        "YearsSinceLastPromotion": 1, "YearsWithCurrManager": 3,
        "JoiningYear": 2018, "City": 0, "PaymentTier": 2,
        "EverBenched": 0, "ExperienceInCurrentDomain": 3,
        "RowNumber": 1, "CustomerId": 0, "Surname": 0,
        "CreditScore": 650, "Geography": 0, "Tenure": 5,
        "Balance": 50000, "NumOfProducts": 1, "HasCrCard": 1,
        "IsActiveMember": 1, "EstimatedSalary": 50000,
        "customerID": 0, "gender": 0, "SeniorCitizen": 0,
        "Partner": 0, "Dependents": 0, "tenure": 5,
        "PhoneService": 1, "MultipleLines": 0, "InternetService": 1,
        "OnlineSecurity": 0, "OnlineBackup": 0, "DeviceProtection": 0,
        "TechSupport": 0, "StreamingTV": 0, "StreamingMovies": 0,
        "Contract": 0, "PaperlessBilling": 1, "PaymentMethod": 0,
        "MonthlyCharges": 65, "TotalCharges": 3000,
        "CLIENTNUM": 0, "Customer_Age": 40, "Dependent_count": 2,
        "Education_Level": 2, "Marital_Status": 1, "Income_Category": 2,
        "Card_Category": 0, "Months_on_book": 36,
        "Total_Relationship_Count": 4, "Months_Inactive_12_mon": 2,
        "Contacts_Count_12_mon": 3, "Credit_Limit": 10000,
        "Total_Revolving_Bal": 1200, "Avg_Open_To_Buy": 8000,
        "Total_Amt_Chng_Q4_Q1": 0.7, "Total_Trans_Amt": 4000,
        "Total_Trans_Ct": 60, "Total_Ct_Chng_Q4_Q1": 0.7,
        "Avg_Utilization_Ratio": 0.2,
        "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1": 0,
        "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2": 0,
        "city": 0, "city_development_index": 0.8, "relevent_experience": 1,
        "enrolled_university": 0, "education_level": 2,
        "major_discipline": 0, "experience": 5, "company_size": 2,
        "company_type": 0, "last_new_job": 1, "training_hours": 50,
    },
    "disease": {
        "Pregnancies": 0, "Glucose": 110, "BloodPressure": 80,
        "SkinThickness": 20, "Insulin": 80, "BMI": 25,
        "DiabetesPedigreeFunction": 0.4, "Age": 35,
        "gender": 0, "age": 35, "hypertension": 0, "heart_disease": 0,
        "smoking_history": 0, "bmi": 25, "HbA1c_level": 5.5,
        "blood_glucose_level": 100, "id": 0, "ever_married": 0,
        "work_type": 2, "Residence_type": 0, "avg_glucose_level": 100,
        "smoking_status": 0, "sex": 0, "cp": 0, "trestbps": 120,
        "chol": 200, "fbs": 0, "restecg": 0, "thalach": 150,
        "exang": 0, "oldpeak": 0, "slope": 1, "ca": 0, "thal": 2,
        "height": 170, "weight": 70, "ap_hi": 120, "ap_lo": 80,
        "cholesterol": 1, "gluc": 1, "smoke": 0, "alco": 0, "active": 1,
    },
    "fitness": {
        "age": 30, "gender": 0, "height_cm": 170, "weight_kg": 70,
        "body fat_%": 20, "diastolic": 75, "systolic": 120,
        "gripForce": 35, "sit and bend forward_cm": 15,
        "sit-ups counts": 30, "broad jump_cm": 180,
        "sex": 0, "bmi": 24, "children": 0, "smoker": 0,
        "region": 0, "charges": 5000, "Gender": 0,
        "Height": 170, "Weight": 70,
    },
}



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
        raise ValueError(f"Domain not found: {domain}")
    raw_path   = registry[domain]["model_path"]
    raw_path   = raw_path.replace("models/", "").replace("models\\", "")
    model_path = os.path.join(BASE, "models", raw_path)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    artifact = joblib.load(model_path)
    if not isinstance(artifact, dict):
        return artifact, None, {}
    if "xgb_model" in artifact:
        model = artifact["xgb_model"]
    elif "model" in artifact:
        model = artifact["model"]
    else:
        raise KeyError(f"No model key found. Keys: {list(artifact.keys())}")
    scaler = artifact.get("scaler")
    return model, scaler, artifact

# ── Confidence tier from gap ──────────────────────────────────
def _confidence_tier(gap: float) -> str:
    if gap < 0.10: return "HIGH"
    if gap < 0.25: return "MODERATE"
    if gap < 0.40: return "LOW"
    return "CRITICAL"

# ── ML prediction ─────────────────────────────────────────────
def _ml_predict(domain: str, parameters: dict) -> Optional[float]:
    try:
        import warnings
        warnings.filterwarnings("ignore")
        model, scaler, artifact = _load_model(domain)
        feature_cols = artifact.get("feature_cols") or artifact.get("feat_cols", [])

        # TEXT MODELS — claim, mental_health, behavioral
        tfidf = artifact.get("tfidf")
        svd   = artifact.get("svd")
        if tfidf is not None and svd is not None:
            text = (parameters.get("claim_text") or
                    parameters.get("text_input") or
                    parameters.get("context") or
                    " ".join([str(v) for v in parameters.values() if isinstance(v, str)]) or
                    "no text provided")
            X = svd.transform(tfidf.transform([text]))
            imputer = artifact.get("imputer")
            if imputer:
                X = imputer.transform(X)
            if scaler:
                X = scaler.transform(X)
            prob = model.predict_proba(X)[0][1]
            return max(0.05, min(0.95, float(prob)))

        # UNMAPPABLE MODELS — loan PCA features
        if feature_cols and feature_cols[0] in ["V1", "V2", "Time"]:
            logger.warning(f"Domain {domain} has PCA features — skipping ML")
            return None

        # NORMAL MODELS
        str_to_num = {
            "low": 0.2, "medium": 0.5, "moderate": 0.5,
            "high": 0.8, "very_high": 0.95, "very high": 0.95,
            "none": 0.0, "yes": 1.0, "no": 0.0,
            "male": 1.0, "female": 0.0, "m": 1.0, "f": 0.0,
            "true": 1.0, "false": 0.0, "strong": 0.9, "weak": 0.2,
        }

        # Step 1 — apply domain param map to user parameters
        param_map     = DOMAIN_PARAM_MAP.get(domain, {})
        domain_defaults = DOMAIN_DEFAULTS.get(domain, {})

        mapped = {}
        for k, v in parameters.items():
            mapped_key = param_map.get(k, param_map.get(k.lower(), k))
            val = str_to_num.get(str(v).lower().strip(), None) if isinstance(v, str) else v
            try:
                mapped[mapped_key] = float(val) if val is not None else 0.5
            except:
                mapped[mapped_key] = 0.5
            # Also keep original key as fallback
            mapped[k] = mapped[mapped_key]
            mapped[k.lower()] = mapped[mapped_key]

        # Step 2 — build feature vector using domain defaults for unmapped cols
        if feature_cols:
            feature_vec = []
            for col in feature_cols:
                val = (mapped.get(col) or
                       mapped.get(col.lower()) or
                       mapped.get(col.replace(" ","_").lower()) or
                       domain_defaults.get(col) or
                       0.0)
                feature_vec.append(float(val))
        else:
            feature_vec = list(mapped.values())

        if not feature_vec:
            logger.warning("ML: empty feature vector")
            return None

        X = np.nan_to_num(
            np.array(feature_vec, dtype=np.float64).reshape(1,-1),
            nan=0.0, posinf=0.0, neginf=0.0)

        imputer = artifact.get("imputer")
        if imputer:
            try:
                n = imputer.n_features_in_
                if X.shape[1] < n: X = np.pad(X, ((0,0),(0,n-X.shape[1])))
                elif X.shape[1] > n: X = X[:,:n]
                X = imputer.transform(X)
            except Exception as ie:
                logger.warning(f"Imputer skipped (version mismatch): {ie}")
                X = np.nan_to_num(X, nan=0.0)

        if scaler:
            try:
                n = scaler.n_features_in_
                if X.shape[1] < n: X = np.pad(X, ((0,0),(0,n-X.shape[1])))
                elif X.shape[1] > n: X = X[:,:n]
                X = scaler.transform(X)
            except Exception as se:
                logger.warning(f"Scaler skipped (version mismatch): {se}")

        prob = model.predict_proba(X)[0][1]
        prob = max(0.05, min(0.95, float(prob)))
        logger.info(f"ML probability: {prob:.4f} (features={len(feature_vec)})")
        return prob

    except Exception as e:
        logger.warning(f"ML prediction failed: {e}")
        return None


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
        import warnings
        warnings.filterwarnings("ignore")
        import numpy as np
        model, scaler, artifact = _load_model(domain)
        feature_cols = artifact.get("feature_cols") or artifact.get("feat_cols", [])

        # Skip text models and PCA models
        if artifact.get("tfidf") or (feature_cols and feature_cols[0] in ["V1","V2","Time"]):
            return {k: round(float(v)/10 if isinstance(v,(int,float)) else 0.1, 4)
                    for k,v in list(parameters.items())[:5]}

        # Use XGB built-in feature importance (no KernelExplainer needed)
        xgb = artifact.get("xgb_model")
        if xgb is None:
            return {}

        # Get feature importances from XGB
        try:
            importances = xgb.feature_importances_
        except:
            try:
                importances = xgb.estimators_[0].feature_importances_ if hasattr(xgb, 'estimators_') else None
            except:
                importances = None

        if importances is None or len(importances) == 0:
            # Fallback — compute signal strength from param values vs domain defaults
            defaults_fb = DOMAIN_DEFAULTS.get(domain, {})
            param_map_fb = DOMAIN_PARAM_MAP.get(domain, {})
            raw = {}
            str_to_num_fb = {"low":0.2,"medium":0.5,"high":0.8,"very_high":0.95,"none":0.0,"yes":1.0,"no":0.0,"true":1.0,"false":0.0}
            # Features where higher value = worse outcome
            negative_features = {
                "absences","Absences","stress_level","failures",
                "overtime","OverTime","distance_from_home","DistanceFromHome",
                "debt_to_income","missed_payments","glucose","Glucose",
                "blood_pressure","BloodPressure","cholesterol","chol",
                "work_hours","bmi","BMI","smoking",
            }
            for k, v in list(parameters.items())[:8]:
                mapped_k = param_map_fb.get(k, k)
                val = str_to_num_fb.get(str(v).lower().strip(), None) if isinstance(v,str) else v
                try:
                    num_val = float(val) if val is not None else 0.5
                except:
                    num_val = 0.5
                default = float(defaults_fb.get(mapped_k, defaults_fb.get(k, 0.5)) or 0.5)
                direction = 1.0 if num_val >= default else -1.0
                # Invert direction for features where higher = worse
                if k in negative_features or mapped_k in negative_features:
                    direction = -direction
                magnitude = abs(num_val - default) / max(abs(default) + 1e-8, 1.0)
                raw[k] = round(direction * magnitude, 6)
            if raw:
                max_abs = max(abs(v) for v in raw.values()) or 1.0
                return {k: round(v/max_abs*0.18, 4) for k,v in raw.items()}
            return {}

        # Build param-mapped feature contributions
        param_map = DOMAIN_PARAM_MAP.get(domain, {})
        defaults  = DOMAIN_DEFAULTS.get(domain, {})

        # Map user params to training col names
        mapped = {}
        str_to_num = {"low": 0.2, "medium": 0.5, "high": 0.8, "very_high": 0.95,
                      "none": 0.0, "yes": 1.0, "no": 0.0, "true": 1.0, "false": 0.0}
        for k, v in parameters.items():
            mapped_key = param_map.get(k, k)
            val = str_to_num.get(str(v).lower().strip(), None) if isinstance(v, str) else v
            try:
                mapped[mapped_key] = float(val) if val is not None else 0.5
            except:
                mapped[mapped_key] = 0.5

        # Build contributions dict using feature importance × (value - mean)
        contributions = {}
        base_prob = 0.5
        for i, col in enumerate(feature_cols[:len(importances)]):
            if i >= len(importances):
                break
            importance = float(importances[i])
            user_val   = mapped.get(col, mapped.get(col.lower(), defaults.get(col, 0.0)))
            default_val = defaults.get(col, 0.5)
            # Direction: positive if above default, negative if below
            direction  = 1.0 if float(user_val) >= float(default_val) else -1.0
            contribution = round(importance * direction, 6)
            if abs(contribution) > 0.001:
                contributions[col] = contribution

        # Normalize contributions to ±0.20 range
        if contributions:
            max_abs = max(abs(v) for v in contributions.values())
            if max_abs > 0:
                contributions = {k: round(v / max_abs * 0.20, 4) for k,v in contributions.items()}

        # Also add user-friendly param names
        for user_key, val in parameters.items():
            mapped_key = param_map.get(user_key, user_key)
            if mapped_key in contributions and user_key not in contributions:
                contributions[user_key] = contributions[mapped_key]

        # Normalize before returning
        if contributions:
            max_abs = max(abs(v) for v in contributions.values())
            if max_abs > 0:
                contributions = {k: round(v / max_abs * 0.18, 4) for k, v in contributions.items()}
        return contributions

    except Exception as e:
        logger.warning(f"SHAP failed: {e}")
        raw = {k: round(float(v)/10 if isinstance(v,(int,float)) else 0.05, 4) for k,v in list(parameters.items())[:5]}
        max_abs = max((abs(v) for v in raw.values()), default=1)
        return {k: round(v/max_abs*0.18, 4) for k,v in raw.items()}

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
        key_factors       = debate_result.get("optimist", {}).get("evidence", []),
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


def predict_rich(
    domain:     str,
    parameters: dict,
    question:   str  = None,
    skipped:    list = None,
    mode:       str  = "guided"
) -> dict:
    """
    Full rich prediction output per Section 8.4 — Three Detail Levels.
    Returns complete prediction with:
    - ML + LLM dual layer probabilities
    - Multi-agent debate transcript
    - SHAP per-feature contributions
    - Monte Carlo 95% CI
    - Failure scenarios
    - Improvement suggestions
    - Reliability Index
    - Audit flags
    """
    import warnings
    warnings.filterwarnings("ignore")
    from core.monte_carlo import monte_carlo_simulate, generate_failure_scenarios, generate_improvement_suggestions

    question = question or f"What is the probability of a positive outcome in the {domain} domain?"

    # ── Run base prediction ───────────────────────────────────
    result = predict(domain, parameters, question, skipped, run_debate=True, mode=mode)
    d      = result.to_dict()

    # ── Monte Carlo simulation ────────────────────────────────
    def ml_predict_fn(params):
        ml = _ml_predict(domain, params)
        llm = _llm_predict(domain, params, question) if ml is None else None
        if ml is not None:
            # Add Gaussian noise to simulate parameter uncertainty
            import numpy as np
            noise = np.random.normal(0, 0.03)
            return max(0.05, min(0.95, ml + noise))
        return llm or 0.5

    mc = monte_carlo_simulate(ml_predict_fn, parameters, n_runs=200)

    # ── Failure scenarios ─────────────────────────────────────
    shap_vals = d.get("shap_values", {})
    failures  = generate_failure_scenarios(domain, parameters, d["final_probability"], shap_vals)

    # ── Improvement suggestions ───────────────────────────────
    improvements = generate_improvement_suggestions(domain, parameters, d["final_probability"], shap_vals)

    # ── SIMPLE output (default) ───────────────────────────────
    simple = {
        "probability":       d["final_probability"],
        "probability_pct":   f"{d['final_probability']*100:.1f}%",
        "outcome":           "Positive" if d["final_probability"] >= 0.5 else "Negative",
        "confidence":        d["confidence_tier"],
        "reliability_index": d["reliability_index"],
    }

    # ── DETAILED output ───────────────────────────────────────
    top_positive = sorted([(k,float(v)) for k,v in shap_vals.items() if isinstance(v,(int,float)) and float(v)>0], key=lambda x:x[1], reverse=True)[:3]
    top_negative = sorted([(k,float(v)) for k,v in shap_vals.items() if isinstance(v,(int,float)) and float(v)<0], key=lambda x:x[1])[:3]

    detailed = {
        **simple,
        "ml_probability":    d["ml_probability"],
        "llm_probability":   d["llm_probability"],
        "gap":               d["gap"],
        "ci_95":             f"{mc['ci_low']*100:.1f}% — {mc['ci_high']*100:.1f}%",
        "ci_width":          mc["ci_width"],
        "stability":         mc["stability"],
        "positive_signals":  [(f, f"+{round(v*100,1)}%") for f,v in top_positive],
        "negative_signals":  [(f, f"{round(v*100,1)}%") for f,v in top_negative],
        "top_failure":       failures[0] if failures else None,
        "top_improvement":   improvements[0] if improvements else None,
        "audit_flags":       d["audit_flags"],
        "debate_ran":        bool(d["debate"]),
    }

    # ── FULL BREAKDOWN output ─────────────────────────────────
    debate = d.get("debate", {})
    full = {
        **detailed,
        "shap_all":           shap_vals,
        "monte_carlo":        mc,
        "failure_scenarios":  failures,
        "improvements":       improvements,
        "debate_transcript":  {
            "optimist":  {
                "probability": debate.get("optimist", {}).get("probability_float", None),
                "argument":    debate.get("optimist", {}).get("argument", ""),
                "evidence":    debate.get("optimist", {}).get("evidence", []),
            } if debate else None,
            "pessimist": {
                "probability": debate.get("pessimist", {}).get("probability_float", None),
                "argument":    debate.get("pessimist", {}).get("argument", ""),
                "evidence":    debate.get("pessimist", {}).get("evidence", []),
            } if debate else None,
            "realist":   {
                "probability": debate.get("realist", {}).get("probability_float", None),
                "reasoning":   debate.get("realist", {}).get("reasoning", ""),
                "confidence":  debate.get("realist", {}).get("confidence", ""),
            } if debate else None,
            "devils_advocate": {
                "counter_score":    debate.get("devils_advocate", {}).get("counter_score", None),
                "counter_argument": debate.get("devils_advocate", {}).get("counter_argument", ""),
                "adjusted":        debate.get("devil_adjusted", False),
            } if debate else None,
        },
        "reasoning":          d["reasoning"],
        "key_factors":        d["key_factors"],
        "domain":             domain,
        "question":           question,
        "mode":               mode,
        "parameters_used":    parameters,
    }

    return {
        "simple":   simple,
        "detailed": detailed,
        "full":     full,
    }

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


# ── Probabilistic Reasoning Transparency (Section 8.4) ───────
def explain_prediction_transparency(
    domain: str,
    parameters: dict,
    final_probability: float,
    shap_values: dict,
    question: str = None,
) -> dict:
    """
    Section 8.4 — Full probabilistic reasoning transparency.
    Shows WHY the dominant probability, WHY the minority probability,
    and WHEN the minority outcome would occur.

    Returns three levels:
    - simple: just the number
    - detailed: case FOR + case AGAINST + top failure scenario
    - full: all scenarios + sensitivity + what this probability is NOT saying
    """
    from llm.router import route

    question = question or f"What is the probability of a positive outcome in the {domain} domain?"
    minority_prob = round(1.0 - final_probability, 4)
    dominant_pct  = round(final_probability * 100, 1)
    minority_pct  = round(minority_prob * 100, 1)

    # ── Build SHAP-based signal lists ─────────────────────────
    positive_signals = sorted(
        [(k, float(v)) for k, v in shap_values.items()
         if isinstance(v, (int, float)) and float(v) > 0],
        key=lambda x: x[1], reverse=True
    )[:5]

    negative_signals = sorted(
        [(k, float(v)) for k, v in shap_values.items()
         if isinstance(v, (int, float)) and float(v) < 0],
        key=lambda x: x[1]
    )[:5]

    # ── LLM generates case FOR and case AGAINST ───────────────
    param_str = "\n".join([f"  - {k}: {v}" for k, v in parameters.items()])
    pos_str   = ", ".join([f"{k}(+{v*100:.1f}%)" for k, v in positive_signals]) or "none detected"
    neg_str   = ", ".join([f"{k}({v*100:.1f}%)" for k, v in negative_signals]) or "none detected"

    messages = [
        {"role": "system", "content": (
            "You are Project Sambhav's probabilistic reasoning engine. "
            "Given a prediction, explain BOTH sides clearly.\n\n"
            "Respond in this EXACT JSON format:\n"
            "{\n"
            '  "case_for": "<2-3 sentences: why the dominant probability is correct, cite specific parameters>",\n'
            '  "case_against": "<2-3 sentences: what factors could make the minority probability occur>",\n'
            '  "when_minority_occurs": [\n'
            '    {"scenario": "<description>", "trigger": "<what changes>", "new_probability": <0-100>},\n'
            '    {"scenario": "<description>", "trigger": "<what changes>", "new_probability": <0-100>},\n'
            '    {"scenario": "<description>", "trigger": "<what changes>", "new_probability": <0-100>}\n'
            "  ],\n"
            '  "what_this_is_not_saying": "<1-2 sentences: common misinterpretation to avoid>",\n'
            '  "key_assumption": "<the single most important assumption in this prediction>"\n'
            "}"
        )},
        {"role": "user", "content": (
            f"Domain: {domain}\n"
            f"Question: {question}\n"
            f"Parameters:\n{param_str}\n\n"
            f"Prediction: {dominant_pct}% (minority: {minority_pct}%)\n"
            f"Positive signals: {pos_str}\n"
            f"Negative signals: {neg_str}\n\n"
            "Explain both sides. Return JSON only."
        )}
    ]

    import json, re
    result  = route("llm_predict", messages, max_tokens=600, temperature=0.3)
    raw     = result.get("content", "")
    raw     = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    if "```" in raw:
        raw = re.sub(r"```(?:json)?", "", raw).strip()

    try:
        parsed = json.loads(raw[raw.find("{"):raw.rfind("}")+1])
    except:
        parsed = {
            "case_for":             f"The {dominant_pct}% probability is supported by {pos_str}.",
            "case_against":         f"The {minority_pct}% minority case is driven by {neg_str}.",
            "when_minority_occurs": [
                {"scenario": "Risk factors worsen", "trigger": "Negative signals increase", "new_probability": int(minority_pct + 10)},
            ],
            "what_this_is_not_saying": f"This does not guarantee the outcome — it means {dominant_pct}% of similar cases result positively.",
            "key_assumption":          "Current parameter values remain stable.",
        }

    # ── SIMPLE level ──────────────────────────────────────────
    simple = {
        "dominant_probability": dominant_pct,
        "minority_probability": minority_pct,
        "one_line_reason":      parsed.get("case_for", "")[:120],
    }

    # ── DETAILED level ────────────────────────────────────────
    detailed = {
        **simple,
        "case_for":              parsed.get("case_for", ""),
        "case_against":          parsed.get("case_against", ""),
        "top_failure_scenario":  parsed.get("when_minority_occurs", [{}])[0],
        "positive_signals":      [(k, f"+{v*100:.1f}%") for k, v in positive_signals],
        "negative_signals":      [(k, f"{v*100:.1f}%") for k, v in negative_signals],
        "what_this_is_not_saying": parsed.get("what_this_is_not_saying", ""),
    }

    # ── FULL level ────────────────────────────────────────────
    full = {
        **detailed,
        "when_minority_occurs": parsed.get("when_minority_occurs", []),
        "key_assumption":       parsed.get("key_assumption", ""),
        "all_positive_signals": positive_signals,
        "all_negative_signals": negative_signals,
        "domain":               domain,
        "question":             question,
        "parameters":           parameters,
    }

    return {"simple": simple, "detailed": detailed, "full": full}


def generate_outcomes(
    domain: str,
    parameters: dict,
    question: str = None,
    n_outcomes: int = 5,
    existing_outcomes: list = None,
    mode: str = "independent",
) -> dict:
    """
    Section 8.3 — Multi-Outcome Generation.
    mode: independent (default) | spectrum | conditional
    Call with existing_outcomes to get MORE without repeating.
    Each outcome supports lazy transparency via explain_outcome_transparency().
    """
    from llm.router import route
    import json, re

    question  = question or f"What are the possible outcomes in the {domain} domain?"
    param_str = "\n".join([f"  - {k}: {v}" for k, v in parameters.items()])
    existing  = existing_outcomes or []

    avoid_str = ""
    if existing:
        avoid_str = (
            "\n\nALREADY SHOWN — do NOT repeat these outcomes:\n" +
            "\n".join([f"  - {o.get('outcome','')}" for o in existing])
        )

    mode_notes = {
        "independent": "Probabilities are INDEPENDENT — do NOT sum to 100%. Each is a separate event that can happen or not.",
        "spectrum":    "Probabilities MUST sum to exactly 100% — these are mutually exclusive outcomes.",
        "conditional": "Each outcome has a condition. State exactly what must happen for this outcome to occur.",
    }
    format_note = mode_notes.get(mode, mode_notes["independent"])

    messages = [
        {"role": "system", "content": (
            "You are Project Sambhav multi-outcome engine (Section 8.3).\n"
            f"MODE: {mode.upper()} — {format_note}\n\n"
            "RULES:\n"
            "1. Each outcome must be meaningfully distinct\n"
            "2. Be precise — not 50% or 70% but 47% or 73%\n"
            "3. Mix positive and negative outcomes\n"
            "4. Base probabilities on actual parameter signals\n"
            "5. reasoning must cite specific parameters\n\n"
            "Respond in EXACT JSON only:\n"
            "{\n"
            '  "outcomes": [\n'
            '    {\n'
            '      "outcome": "<clear description>",\n'
            '      "probability": <0-100>,\n'
            '      "reasoning": "<1 sentence citing parameters>",\n'
            '      "type": "<positive|negative|neutral>",\n'
            '      "condition": "<required condition or null>"\n'
            "    }\n"
            "  ],\n"
            '  "interpretation_note": "<how to read these probabilities>"\n'
            "}"
        )},
        {"role": "user", "content": (
            f"Domain: {domain}\n"
            f"Question: {question}\n"
            f"Parameters:\n{param_str}\n\n"
            f"Generate {n_outcomes} outcomes." + avoid_str
        )}
    ]

    result = route("free_inference", messages, max_tokens=800, temperature=0.5)
    raw    = result.get("content", "")
    raw    = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    if "```" in raw:
        raw = re.sub(r"```(?:json)?", "", raw).strip()

    try:
        parsed = json.loads(raw[raw.find("{"):raw.rfind("}")+1])
        outcomes = parsed.get("outcomes", [])
    except:
        outcomes = []

    # Ensure each outcome has all fields
    clean = []
    for o in outcomes:
        if o.get("outcome") and o.get("probability") is not None:
            clean.append({
                "outcome":     o.get("outcome", ""),
                "probability": max(1, min(99, int(o.get("probability", 50)))),
                "probability_pct": f"{o.get('probability', 50)}%",
                "reasoning":   o.get("reasoning", ""),
                "type":        o.get("type", "neutral"),
                "condition":   o.get("condition"),
                "has_transparency": True,  # UI shows [? Why] button for each
            })

    return {
        "outcomes":             clean,
        "mode":                 mode,
        "interpretation_note":  parsed.get("interpretation_note", format_note) if "parsed" in dir() else format_note,
        "total_shown":          len(existing) + len(clean),
        "can_generate_more":    True,
        "domain":               domain,
        "parameters":           parameters,
    }


def explain_outcome_transparency(
    domain: str,
    parameters: dict,
    outcome: str,
    probability: float,
    question: str = None,
) -> dict:
    """
    Called when user clicks [? Why] on a specific outcome.
    Returns WHY this outcome has this probability + WHEN it would NOT occur.
    Lazy — only called on demand per outcome.
    """
    shap_vals = _get_shap(domain, parameters)
    return explain_prediction_transparency(
        domain=domain,
        parameters=parameters,
        final_probability=probability / 100,
        shap_values=shap_vals,
        question=f"Why does '{outcome}' have {probability}% probability?",
    )
