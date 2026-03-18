import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings("ignore")

def quick_model(X_train, y_train, X_test, y_test):
    estimators = [
        ("lr", LogisticRegression(max_iter=2000, random_state=42)),
        ("rf", RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, n_jobs=-1)),
        ("xgb", XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.05,
                               eval_metric="logloss", verbosity=0, random_state=42)),
        ("lgb", LGBMClassifier(n_estimators=100, max_depth=4,
                                learning_rate=0.05, verbose=-1, random_state=42)),
    ]
    model = CalibratedClassifierCV(
        StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=2000),
            cv=3, passthrough=True, n_jobs=-1
        ),
        method="sigmoid", cv=3
    )
    model.fit(X_train, y_train)
    return brier_score_loss(y_test, model.predict_proba(X_test)[:, 1])

def run_ablation(domain="student", target="pass"):
    print(f"\n{'='*50}")
    print(f"ABLATION STUDY — {domain.upper()}")
    print(f"{'='*50}")

    df = pd.read_csv(f"data/processed/{domain}_final.csv")
    df = df.select_dtypes(include=[np.number])
    df = df.drop(columns=["synthetic"], errors="ignore")
    df = df.fillna(df.median())

    if target not in df.columns:
        print(f"Target {target} not found")
        return

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    results = {}

    # Experiment 1 — Full system (baseline)
    print("Running Exp 1: Full system...")
    results["1_full_system"] = quick_model(X_train, y_train, X_test, y_test)
    print(f"  Brier: {results['1_full_system']:.4f}")

    # Experiment 2 — Remove G1/G2 (past performance proxy)
    past_cols = [c for c in X.columns if any(
        k in c.lower() for k in ["g1", "g2", "grade", "past", "failures"]
    )]
    if past_cols:
        print(f"Running Exp 2: No past performance ({past_cols})...")
        X_tr2 = X_train.drop(columns=past_cols, errors="ignore")
        X_te2 = X_test.drop(columns=past_cols, errors="ignore")
        results["2_no_past_performance"] = quick_model(X_tr2, y_tr2 if False else y_train, X_te2, y_test)
        results["2_no_past_performance"] = quick_model(X_tr2, y_train, X_te2, y_test)
        print(f"  Brier: {results['2_no_past_performance']:.4f}")
    else:
        results["2_no_past_performance"] = results["1_full_system"]
        print("  No past performance cols found — skipped")

    # Experiment 3 — Remove behavioral/lifestyle features
    behavioral_cols = [c for c in X.columns if any(
        k in c.lower() for k in ["studytime", "absences", "health",
                                  "dalc", "walc", "goout", "freetime",
                                  "overtime", "worklife", "satisfaction"]
    )]
    if behavioral_cols:
        print(f"Running Exp 3: No behavioral features ({len(behavioral_cols)} cols)...")
        X_tr3 = X_train.drop(columns=behavioral_cols, errors="ignore")
        X_te3 = X_test.drop(columns=behavioral_cols, errors="ignore")
        results["3_no_behavioral"] = quick_model(X_tr3, y_train, X_te3, y_test)
        print(f"  Brier: {results['3_no_behavioral']:.4f}")
    else:
        results["3_no_behavioral"] = results["1_full_system"]
        print("  No behavioral cols found — skipped")

    # Experiment 4 — Remove family/support features
    support_cols = [c for c in X.columns if any(
        k in c.lower() for k in ["famsup", "schoolsup", "famrel",
                                  "paid", "higher", "internet",
                                  "jobsatisfaction", "environment",
                                  "relationship"]
    )]
    if support_cols:
        print(f"Running Exp 4: No support features ({len(support_cols)} cols)...")
        X_tr4 = X_train.drop(columns=support_cols, errors="ignore")
        X_te4 = X_test.drop(columns=support_cols, errors="ignore")
        results["4_no_support"] = quick_model(X_tr4, y_train, X_te4, y_test)
        print(f"  Brier: {results['4_no_support']:.4f}")
    else:
        results["4_no_support"] = results["1_full_system"]
        print("  No support cols found — skipped")

    # Experiment 5 — Single feature only (most important)
    print("Running Exp 5: Single best feature only...")
    rf_quick = RandomForestClassifier(n_estimators=50, random_state=42)
    rf_quick.fit(X_train, y_train)
    best_feature = X.columns[rf_quick.feature_importances_.argmax()]
    print(f"  Best feature: {best_feature}")
    X_tr5 = X_train[[best_feature]]
    X_te5 = X_test[[best_feature]]
    lr_single = LogisticRegression(max_iter=1000)
    lr_single.fit(X_tr5, y_train)
    results["5_single_feature"] = brier_score_loss(
        y_test, lr_single.predict_proba(X_te5)[:, 1]
    )
    print(f"  Brier: {results['5_single_feature']:.4f}")

    # Print summary
    print(f"\n--- Ablation Results: {domain.upper()} ---")
    baseline = results["1_full_system"]
    for exp, brier in results.items():
        delta = brier - baseline
        direction = "↑ worse" if delta > 0.005 else ("↓ better" if delta < -0.005 else "≈ same")
        print(f"  {exp}: {brier:.4f} ({direction})")

    print(f"\n✅ Ablation complete for {domain}")
    return results

if __name__ == "__main__":
    all_results = {}

    all_results["student"] = run_ablation("student", "pass")
    all_results["hr"] = run_ablation("hr", "attrition")
    all_results["behavioral"] = run_ablation("behavioral", "deceptive")

    print(f"\n{'='*50}")
    print("ABLATION SUMMARY ACROSS ALL DOMAINS")
    print(f"{'='*50}")
    for domain, results in all_results.items():
        if results:
            baseline = results.get("1_full_system", 0)
            worst = max(results.values())
            print(f"{domain}: baseline={baseline:.4f} | worst_ablation={worst:.4f} | "
                  f"max_degradation={worst-baseline:.4f}")

    print("\n=== Phase 5 Ablation Studies COMPLETE ===")