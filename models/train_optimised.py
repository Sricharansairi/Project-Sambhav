
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

os.makedirs("models", exist_ok=True)

def tune_xgb(X, y):
    print("  Tuning XGBoost...")
    params = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 0.8, 0.9],
        "colsample_bytree": [0.7, 0.8, 0.9],
    }
    search = RandomizedSearchCV(
        XGBClassifier(eval_metric="logloss", verbosity=0, random_state=42),
        params, n_iter=30, cv=3, scoring="neg_brier_score",
        random_state=42, n_jobs=-1
    )
    search.fit(X, y)
    print(f"  Best XGB: {search.best_params_}")
    return search.best_estimator_

def tune_lgb(X, y):
    print("  Tuning LightGBM...")
    params = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 4, 5, 6],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.7, 0.8, 0.9],
        "num_leaves": [15, 31, 63],
    }
    search = RandomizedSearchCV(
        LGBMClassifier(verbose=-1, random_state=42),
        params, n_iter=30, cv=3, scoring="neg_brier_score",
        random_state=42, n_jobs=-1
    )
    search.fit(X, y)
    print(f"  Best LGB: {search.best_params_}")
    return search.best_estimator_

def train_optimised(name, target):
    print(f"\n{'='*40}")
    print(f"Training: {name.upper()}")
    print(f"{'='*40}")
    df = pd.read_csv(f"data/processed/{name}_final.csv")
    df = df.select_dtypes(include=[np.number])
    df = df.drop(columns=["synthetic"], errors="ignore").fillna(df.median())
    if target not in df.columns:
        print(f"Target {target} not found")
        return None
    X = df.drop(columns=[target]).values
    y = df[target].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_s, y_train)
    lr_brier = brier_score_loss(y_test, lr.predict_proba(X_test_s)[:, 1])
    print(f"LR Brier (scaled): {lr_brier:.4f}")
    best_xgb = tune_xgb(X_train_s, y_train)
    best_lgb = tune_lgb(X_train_s, y_train)
    estimators = [
        ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ("rf", RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)),
        ("xgb", best_xgb),
        ("lgb", best_lgb),
    ]
    calibrated = CalibratedClassifierCV(
        StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=1000),
            cv=5, passthrough=True, n_jobs=-1
        ),
        method="isotonic", cv=5
    )
    calibrated.fit(X_train_s, y_train)
    brier = brier_score_loss(y_test, calibrated.predict_proba(X_test_s)[:, 1])
    improvement = ((lr_brier - brier) / lr_brier) * 100
    print(f"Optimised Brier:   {brier:.4f}")
    print(f"Improvement:       {improvement:.1f}%")
    joblib.dump({"model": calibrated, "scaler": scaler}, f"models/{name}_stacking_v2.joblib")
    print(f"Saved: models/{name}_stacking_v2.joblib")
    return brier

def train_claim_tfidf():
    print(f"\n{'='*40}")
    print("Training: CLAIM (TF-IDF)")
    print(f"{'='*40}")
    df = pd.read_csv("data/processed/claim_final.csv")
    target = "credible"
    texts = df["statement"].fillna("").astype(str)
    tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1,2), min_df=2, sublinear_tf=True)
    X_tfidf = tfidf.fit_transform(texts)
    svd = TruncatedSVD(n_components=100, random_state=42)
    X_text = svd.fit_transform(X_tfidf)
    df_num = df.select_dtypes(include=[np.number]).drop(columns=[target,"synthetic"], errors="ignore").fillna(0)
    X = np.hstack([X_text, df_num.values])
    y = df[target].fillna(0).values
    print(f"Features: {X.shape}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_s, y_train)
    lr_brier = brier_score_loss(y_test, lr.predict_proba(X_test_s)[:, 1])
    print(f"LR Brier (with text): {lr_brier:.4f}")
    best_xgb = tune_xgb(X_train_s, y_train)
    best_lgb = tune_lgb(X_train_s, y_train)
    estimators = [
        ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ("rf", RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)),
        ("xgb", best_xgb),
        ("lgb", best_lgb),
    ]
    calibrated = CalibratedClassifierCV(
        StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=1000),
            cv=5, passthrough=True, n_jobs=-1
        ),
        method="isotonic", cv=5
    )
    calibrated.fit(X_train_s, y_train)
    brier = brier_score_loss(y_test, calibrated.predict_proba(X_test_s)[:, 1])
    print(f"Optimised Brier: {brier:.4f}")
    joblib.dump({"model": calibrated, "scaler": scaler, "tfidf": tfidf, "svd": svd,
                 "feature_cols": list(df_num.columns)}, "models/claim_stacking_v2.joblib")
    print("Saved: models/claim_stacking_v2.joblib")
    return brier

original = {"student": 0.1025, "hr": 0.1462, "behavioral": 0.0007, "claim": 0.1852}
results = {}
results["student"] = train_optimised("student", "pass")
results["hr"] = train_optimised("hr", "attrition")
results["behavioral"] = train_optimised("behavioral", "deceptive")
results["claim"] = train_claim_tfidf()

print(f"\n{'='*40}")
print("FINAL RESULTS")
print(f"{'='*40}")
for name, brier in results.items():
    if brier is not None:
        orig = original[name]
        pct = ((orig - brier) / orig) * 100
        print(f"{name}: {orig:.4f} -> {brier:.4f} ({pct:+.1f}%) {'✅' if brier < orig else '⚠'}")
print("\n=== OPTIMISATION COMPLETE ===")
