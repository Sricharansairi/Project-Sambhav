import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer

results = []

def get_metrics(y_test, y_prob, y_pred):
    acc   = accuracy_score(y_test, y_pred)
    auc   = roc_auc_score(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)
    frac, mean_pred = calibration_curve(y_test, y_prob, n_bins=10)
    ece   = float(np.mean(np.abs(frac - mean_pred)))
    return acc, auc, brier, ece

def add_result(name, acc, auc, brier, ece):
    results.append({
        "Domain":     name,
        "Accuracy":   f"{acc*100:.1f}%",
        "AUC-ROC":    f"{auc:.4f}",
        "Brier":      f"{brier:.4f}",
        "ECE":        f"{ece:.4f}",
        "Brier<0.12": "✓" if brier < 0.12 else "✗",
        "ECE<0.08":   "✓" if ece   < 0.08 else "✗",
    })

def add_error(name, e):
    print(f"✗  {name} FAILED — {e}")
    results.append({"Domain": name, "Accuracy": "ERR", "AUC-ROC": "ERR",
                    "Brier": "ERR", "ECE": "ERR", "Brier<0.12": "✗", "ECE<0.08": "✗"})

# --- Student (has NaNs, uses feature_cols) ---
try:
    a   = joblib.load("models/student_stacking_v3.joblib")
    df  = pd.read_csv("data/processed/student_final.csv")
    imp = SimpleImputer(strategy="median")
    X   = imp.fit_transform(df[a["feature_cols"]])
    X   = a["scaler"].transform(X)
    y   = df["pass"].values
    split = int(len(X)*0.8)
    y_prob = a["model"].predict_proba(X[split:])[:,1]
    y_pred = a["model"].predict(X[split:])
    add_result("Student", *get_metrics(y[split:], y_prob, y_pred))
    print("✓  Student")
except Exception as e:
    add_error("Student", e)

# --- Higher Ed (clean, uses feature_cols) ---
try:
    a   = joblib.load("models/student_dropout_stacking_v4.joblib")
    df  = pd.read_csv("data/processed/student_dropout_final.csv")
    imp = SimpleImputer(strategy="median")
    X   = imp.fit_transform(df[a["feature_cols"]])
    X   = a["scaler"].transform(X)
    y   = df["pass"].values
    split = int(len(X)*0.8)
    y_prob = a["model"].predict_proba(X[split:])[:,1]
    y_pred = a["model"].predict(X[split:])
    add_result("Higher Ed", *get_metrics(y[split:], y_prob, y_pred))
    print("✓  Higher Ed")
except Exception as e:
    add_error("Higher Ed", e)

# --- HR Attrition (has NaNs, uses feature_cols) ---
try:
    a   = joblib.load("models/hr_stacking_v3.joblib")
    df  = pd.read_csv("data/processed/hr_final.csv")
    imp = SimpleImputer(strategy="median")
    X   = imp.fit_transform(df[a["feature_cols"]])
    X   = a["scaler"].transform(X)
    y   = df["attrition"].values
    split = int(len(X)*0.8)
    y_prob = a["model"].predict_proba(X[split:])[:,1]
    y_pred = a["model"].predict(X[split:])
    add_result("HR Attrition", *get_metrics(y[split:], y_prob, y_pred))
    print("✓  HR Attrition")
except Exception as e:
    add_error("HR Attrition", e)

# --- Behavioral (uses feat_cols) ---
try:
    a   = joblib.load("models/behavioral_stacking_v5.joblib")
    df  = pd.read_csv("data/processed/behavioral_final.csv")
    imp = SimpleImputer(strategy="median")
    X   = imp.fit_transform(df[a["feat_cols"]])
    X   = a["scaler"].transform(X)
    y   = df["deceptive"].values
    split = int(len(X)*0.8)
    y_prob = a["model"].predict_proba(X[split:])[:,1]
    y_pred = a["model"].predict(X[split:])
    add_result("Behavioral", *get_metrics(y[split:], y_prob, y_pred))
    print("✓  Behavioral")
except Exception as e:
    add_error("Behavioral", e)

# --- Claim (uses tfidf + svd + num_cols) ---
try:
    a   = joblib.load("models/claim_stacking_v5.joblib")
    df  = pd.read_csv("data/processed/claim_final.csv")
    tfidf_mat = a["tfidf"].transform(df["statement"].astype(str))
    svd_mat   = a["svd"].transform(tfidf_mat)
    num_imp   = SimpleImputer(strategy="median")
    num_mat   = num_imp.fit_transform(df[a["num_cols"]])
    X         = np.hstack([svd_mat, num_mat])
    X         = a["scaler"].transform(X)
    y         = df["credible"].values
    split     = int(len(X)*0.8)
    y_prob    = a["model"].predict_proba(X[split:])[:,1]
    y_pred    = a["model"].predict(X[split:])
    add_result("Claim", *get_metrics(y[split:], y_prob, y_pred))
    print("✓  Claim")
except Exception as e:
    add_error("Claim", e)

# --- Print results ---
print("\n")
df_out = pd.DataFrame(results)
print(df_out.to_string(index=False))
df_out.to_csv("model_scorecard_full.csv", index=False)
print("\nSaved → model_scorecard_full.csv")
