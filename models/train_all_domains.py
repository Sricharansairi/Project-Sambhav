import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

os.makedirs("models", exist_ok=True)

DOMAIN_CONFIGS = {
    "student": {
        "data": "data/processed/student_final.csv",
        "target": "pass",
        "model_out": "models/student_stacking_v1.joblib"
    },
    "hr": {
        "data": "data/processed/hr_final.csv",
        "target": "attrition",
        "model_out": "models/hr_stacking_v1.joblib"
    },
    "behavioral": {
        "data": "data/processed/behavioral_final.csv",
        "target": "deceptive",
        "model_out": "models/behavioral_stacking_v1.joblib"
    },
    "claim": {
        "data": "data/processed/claim_final.csv",
        "target": "credible",
        "model_out": "models/claim_stacking_v1.joblib"
    },
}

def train_domain(name, config):
    print(f"\n{'='*40}")
    print(f"Training: {name.upper()}")
    print(f"{'='*40}")

    df = pd.read_csv(config["data"])
    target = config["target"]

    df = df.select_dtypes(include=[np.number])
    df = df.drop(columns=["synthetic"], errors="ignore")

    if target not in df.columns:
        print(f"⚠ Target '{target}' not found — skipping")
        return None, None

    X = df.drop(columns=[target])
    y = df[target]

    # Fill any NaN values
    X = X.fillna(X.median())
    y = y.fillna(0)

    print(f"Data: {X.shape[0]} rows | {X.shape[1]} features")
    print(f"Target balance: {y.value_counts().to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    lr_base = LogisticRegression(max_iter=1000, random_state=42)
    lr_base.fit(X_train, y_train)
    lr_brier = brier_score_loss(y_test, lr_base.predict_proba(X_test)[:, 1])
    print(f"Baseline LR Brier:      {lr_brier:.4f}")

    nb_base = GaussianNB()
    nb_base.fit(X_train, y_train)
    nb_brier = brier_score_loss(y_test, nb_base.predict_proba(X_test)[:, 1])
    print(f"Baseline NB Brier:      {nb_brier:.4f}")

    estimators = [
        ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ("rf", RandomForestClassifier(
            n_estimators=200, max_depth=8,
            random_state=42, n_jobs=-1)),
        ("xgb", XGBClassifier(
            n_estimators=200, max_depth=5,
            learning_rate=0.05, eval_metric="logloss",
            verbosity=0, random_state=42)),
        ("lgb", LGBMClassifier(
            n_estimators=200, max_depth=5,
            learning_rate=0.05, verbose=-1,
            random_state=42)),
    ]

    calibrated = CalibratedClassifierCV(
        StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=1000),
            cv=5,
            passthrough=True,
            n_jobs=-1
        ),
        method="sigmoid",
        cv=5
    )

    calibrated.fit(X_train, y_train)
    brier = brier_score_loss(y_test, calibrated.predict_proba(X_test)[:, 1])
    improvement = ((lr_brier - brier) / lr_brier) * 100
    print(f"Calibrated Stack Brier: {brier:.4f}")
    print(f"Improvement over LR:    {improvement:.1f}%")

    joblib.dump(calibrated, config["model_out"])
    print(f"Saved: {config['model_out']}")
    print(f"Status: {'✅ PASS' if brier < 0.2 else '⚠ WARNING'}")

    return calibrated, brier

if __name__ == "__main__":
    results = {}
    for name, config in DOMAIN_CONFIGS.items():
        model, brier = train_domain(name, config)
        if brier is not None:
            results[name] = brier

    print(f"\n{'='*40}")
    print("FINAL RESULTS")
    print(f"{'='*40}")
    for name, brier in results.items():
        status = "✅" if brier < 0.2 else "⚠"
        print(f"{status} {name}: Brier={brier:.4f}")

    print("\n=== Phase 4 COMPLETE ===")