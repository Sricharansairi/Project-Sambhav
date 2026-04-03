import joblib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

print("=" * 55)
print("SAMBHAV — Full Retraining Session (~30 mins)")
print("=" * 55)

# ── helpers ──────────────────────────────────────────────
def brier(model, X, y):
    return brier_score_loss(y, model.predict_proba(X)[:, 1])

def build_stack():
    return StackingClassifier(
        estimators=[
            ("lr",   LogisticRegression(max_iter=1000, random_state=42)),
            ("rf",   RandomForestClassifier(random_state=42, n_jobs=2)),
            ("xgb",  XGBClassifier(eval_metric="logloss", random_state=42,
                                   verbosity=0, n_jobs=2)),
            ("lgbm", LGBMClassifier(random_state=42, verbose=-1, n_jobs=2)),
        ],
        final_estimator=LogisticRegression(max_iter=1000),
        cv=5,
        passthrough=True,
        n_jobs=2,
    )

def calibrate(stack, X_cal, y_cal):
    cal = CalibratedClassifierCV(stack, method="isotonic", cv="prefit")
    cal.fit(X_cal, y_cal)
    return cal

def save(artifact, path):
    joblib.dump(artifact, path)
    print(f"  Saved → {path}")

# ── XGB param grid (shared) ───────────────────────────────
xgb_params = {
    "xgb__n_estimators":  [200, 300, 400],
    "xgb__max_depth":     [3, 4, 5, 6],
    "xgb__learning_rate": [0.01, 0.05, 0.1],
    "xgb__subsample":     [0.7, 0.8, 0.9],
    "xgb__colsample_bytree": [0.7, 0.8, 1.0],
}

lgbm_params = {
    "lgbm__n_estimators":  [200, 300, 400],
    "lgbm__max_depth":     [3, 4, 5, 6],
    "lgbm__learning_rate": [0.01, 0.05, 0.1],
    "lgbm__num_leaves":    [15, 31, 63],
}

all_params = {**xgb_params, **lgbm_params}


# ════════════════════════════════════════════════════════
# 1. STUDENT
# ════════════════════════════════════════════════════════
print("\n[1/5] Student — retraining...")
try:
    old   = joblib.load("models/student_stacking_v3.joblib")
    df    = pd.read_csv("data/processed/student_final.csv")
    cols  = old["feature_cols"]
    imp   = SimpleImputer(strategy="median")
    X     = imp.fit_transform(df[cols])
    y     = df["pass"].values
    scaler = StandardScaler()
    X     = scaler.fit_transform(X)

    X_tr, X_test, y_tr, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    X_tr, X_cal, y_tr, y_cal = train_test_split(
        X_tr, y_tr, test_size=0.15, stratify=y_tr, random_state=42)

    stack = build_stack()
    rs = RandomizedSearchCV(stack, all_params, n_iter=20, scoring="neg_brier_score",
                            cv=3, random_state=42, n_jobs=2, verbose=0)
    rs.fit(X_tr, y_tr)
    best  = rs.best_estimator_
    cal   = calibrate(best, X_cal, y_cal)

    old_b = brier(joblib.load("models/student_stacking_v3.joblib")["model"], X_test, y_test)
    new_b = brier(cal, X_test, y_test)
    print(f"  Brier  old={old_b:.4f}  new={new_b:.4f}  {'✓ improved' if new_b < old_b else '— no change'}")

    save({"model": cal, "scaler": scaler, "feature_cols": cols, "target_col": "pass"},
         "models/student_stacking_v4.joblib")
except Exception as e:
    print(f"  ✗ FAILED — {e}")


# ════════════════════════════════════════════════════════
# 2. HIGHER ED
# ════════════════════════════════════════════════════════
print("\n[2/5] Higher Ed — retraining...")
try:
    old   = joblib.load("models/student_dropout_stacking_v4.joblib")
    df    = pd.read_csv("data/processed/student_dropout_final.csv")
    cols  = old["feature_cols"]
    imp   = SimpleImputer(strategy="median")
    X     = imp.fit_transform(df[cols])
    y     = df["pass"].values
    scaler = StandardScaler()
    X     = scaler.fit_transform(X)

    X_tr, X_test, y_tr, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    X_tr, X_cal, y_tr, y_cal = train_test_split(
        X_tr, y_tr, test_size=0.15, stratify=y_tr, random_state=42)

    stack = build_stack()
    rs = RandomizedSearchCV(stack, all_params, n_iter=20, scoring="neg_brier_score",
                            cv=3, random_state=42, n_jobs=2, verbose=0)
    rs.fit(X_tr, y_tr)
    best  = rs.best_estimator_
    cal   = calibrate(best, X_cal, y_cal)

    old_b = brier(joblib.load("models/student_dropout_stacking_v4.joblib")["model"], X_test, y_test)
    new_b = brier(cal, X_test, y_test)
    print(f"  Brier  old={old_b:.4f}  new={new_b:.4f}  {'✓ improved' if new_b < old_b else '— no change'}")

    save({"model": cal, "scaler": scaler, "feature_cols": cols, "target_col": "pass"},
         "models/student_dropout_stacking_v5.joblib")
except Exception as e:
    print(f"  ✗ FAILED — {e}")


# ════════════════════════════════════════════════════════
# 3. HR ATTRITION
# ════════════════════════════════════════════════════════
print("\n[3/5] HR Attrition — retraining...")
try:
    old   = joblib.load("models/hr_stacking_v3.joblib")
    df    = pd.read_csv("data/processed/hr_final.csv")
    cols  = old["feature_cols"]
    imp   = SimpleImputer(strategy="median")
    X     = imp.fit_transform(df[cols])
    y     = df["attrition"].values
    scaler = StandardScaler()
    X     = scaler.fit_transform(X)

    X_tr, X_test, y_tr, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    X_tr, X_cal, y_tr, y_cal = train_test_split(
        X_tr, y_tr, test_size=0.15, stratify=y_tr, random_state=42)

    stack = build_stack()
    rs = RandomizedSearchCV(stack, all_params, n_iter=20, scoring="neg_brier_score",
                            cv=3, random_state=42, n_jobs=2, verbose=0)
    rs.fit(X_tr, y_tr)
    best  = rs.best_estimator_
    cal   = calibrate(best, X_cal, y_cal)

    old_b = brier(joblib.load("models/hr_stacking_v3.joblib")["model"], X_test, y_test)
    new_b = brier(cal, X_test, y_test)
    print(f"  Brier  old={old_b:.4f}  new={new_b:.4f}  {'✓ improved' if new_b < old_b else '— no change'}")

    save({"model": cal, "scaler": scaler, "feature_cols": cols, "target_col": "attrition"},
         "models/hr_stacking_v4.joblib")
except Exception as e:
    print(f"  ✗ FAILED — {e}")


# ════════════════════════════════════════════════════════
# 4. BEHAVIORAL
# ════════════════════════════════════════════════════════
print("\n[4/5] Behavioral — retraining...")
try:
    old   = joblib.load("models/behavioral_stacking_v5.joblib")
    df    = pd.read_csv("data/processed/behavioral_final.csv")
    cols  = old["feat_cols"]
    imp   = SimpleImputer(strategy="median")
    X     = imp.fit_transform(df[cols])
    y     = df["deceptive"].values
    scaler = StandardScaler()
    X     = scaler.fit_transform(X)

    X_tr, X_test, y_tr, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    X_tr, X_cal, y_tr, y_cal = train_test_split(
        X_tr, y_tr, test_size=0.15, stratify=y_tr, random_state=42)

    stack = build_stack()
    rs = RandomizedSearchCV(stack, all_params, n_iter=20, scoring="neg_brier_score",
                            cv=3, random_state=42, n_jobs=2, verbose=0)
    rs.fit(X_tr, y_tr)
    best  = rs.best_estimator_
    cal   = calibrate(best, X_cal, y_cal)

    new_b = brier(cal, X_test, y_test)
    print(f"  Brier  new={new_b:.4f}")

    save({"model": cal, "scaler": scaler, "feat_cols": cols, "target_col": "deceptive"},
         "models/behavioral_stacking_v6.joblib")
except Exception as e:
    print(f"  ✗ FAILED — {e}")


# ════════════════════════════════════════════════════════
# 5. CLAIM
# ════════════════════════════════════════════════════════
print("\n[5/5] Claim — retraining...")
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD

    old    = joblib.load("models/claim_stacking_v5.joblib")
    df     = pd.read_csv("data/processed/claim_final.csv")

    tfidf  = TfidfVectorizer(max_features=8000, ngram_range=(1,2),
                              sublinear_tf=True, min_df=2)
    svd    = TruncatedSVD(n_components=120, random_state=42)
    tfidf_mat = tfidf.fit_transform(df["statement"].astype(str))
    svd_mat   = svd.fit_transform(tfidf_mat)

    num_cols = old["num_cols"]
    num_imp  = SimpleImputer(strategy="median")
    num_mat  = num_imp.fit_transform(df[num_cols])

    X      = np.hstack([svd_mat, num_mat])
    y      = df["credible"].values
    scaler = StandardScaler()
    X      = scaler.fit_transform(X)

    X_tr, X_test, y_tr, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    X_tr, X_cal, y_tr, y_cal = train_test_split(
        X_tr, y_tr, test_size=0.15, stratify=y_tr, random_state=42)

    stack = build_stack()
    rs = RandomizedSearchCV(stack, all_params, n_iter=20, scoring="neg_brier_score",
                            cv=3, random_state=42, n_jobs=2, verbose=0)
    rs.fit(X_tr, y_tr)
    best  = rs.best_estimator_
    cal   = calibrate(best, X_cal, y_cal)

    old_b = brier(joblib.load("models/claim_stacking_v5.joblib")["model"], X_test, y_test)
    new_b = brier(cal, X_test, y_test)
    print(f"  Brier  old={old_b:.4f}  new={new_b:.4f}  {'✓ improved' if new_b < old_b else '— no change'}")

    save({"model": cal, "scaler": scaler, "tfidf": tfidf, "svd": svd,
          "num_cols": num_cols, "target_col": "credible"},
         "models/claim_stacking_v6.joblib")
except Exception as e:
    print(f"  ✗ FAILED — {e}")


print("\n" + "=" * 55)
print("Retraining complete. New versions saved.")
print("Student v4 | Higher Ed v5 | HR v4 | Behavioral v6 | Claim v6")
print("=" * 55)
