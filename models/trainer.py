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

def train_student_model():
    print("=== Training Student Model ===")
    df = pd.read_csv("data/processed/student_final.csv")
    X = df.drop(columns=["pass"])
    y = df["pass"]
    print(f"Loaded: {X.shape[0]} rows, {X.shape[1]} features")

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_train, X_cal, y_train, y_cal = train_test_split(X_temp, y_temp, test_size=0.125, stratify=y_temp, random_state=42)
    print(f"Train: {len(X_train)} | Cal: {len(X_cal)} | Test: {len(X_test)}")

    lr_base = LogisticRegression(max_iter=1000, random_state=42)
    lr_base.fit(X_train, y_train)
    print(f"LR Brier: {brier_score_loss(y_test, lr_base.predict_proba(X_test)[:,1]):.4f}")

    rf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    print(f"RF Brier: {brier_score_loss(y_test, rf.predict_proba(X_test)[:,1]):.4f}")

    xgb = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, eval_metric="logloss", verbosity=0, random_state=42)
    xgb.fit(X_train, y_train)
    print(f"XGB Brier: {brier_score_loss(y_test, xgb.predict_proba(X_test)[:,1]):.4f}")

    lgb = LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, verbose=-1, random_state=42)
    lgb.fit(X_train, y_train)
    print(f"LGB Brier: {brier_score_loss(y_test, lgb.predict_proba(X_test)[:,1]):.4f}")

    estimators = [
        ("lr", LogisticRegression(max_iter=1000, random_state=42)),
        ("rf", RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)),
        ("xgb", XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, eval_metric="logloss", verbosity=0, random_state=42)),
        ("lgb", LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, verbose=-1, random_state=42)),
    ]
    stack = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=1000), cv=5, passthrough=True, n_jobs=-1)
    stack.fit(X_train, y_train)
    stack_brier = brier_score_loss(y_test, stack.predict_proba(X_test)[:,1])
    print(f"Stack Brier: {stack_brier:.4f}")

    calibrated = CalibratedClassifierCV(stack, method="isotonic", cv="prefit")
    calibrated.fit(X_cal, y_cal)
    cal_brier = brier_score_loss(y_test, calibrated.predict_proba(X_test)[:,1])
    print(f"Calibrated Brier: {cal_brier:.4f}")
    print(f"Improvement: {((stack_brier - cal_brier)/stack_brier)*100:.1f}%")

    os.makedirs("models", exist_ok=True)
    joblib.dump(calibrated, "models/student_stacking_v1.joblib")
    print("Model saved to models/student_stacking_v1.joblib")
    return calibrated, cal_brier

if __name__ == "__main__":
    model, brier = train_student_model()
    print(f"Final Brier: {brier:.4f}")
    print("PASS" if brier < 0.15 else "WARNING: high brier score")
