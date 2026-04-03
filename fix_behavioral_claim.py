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
from sklearn.metrics import brier_score_loss, accuracy_score, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

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
        cv=5, passthrough=True, n_jobs=2,
    )

all_params = {
    "xgb__n_estimators":     [200, 300, 400],
    "xgb__max_depth":        [3, 4, 5, 6],
    "xgb__learning_rate":    [0.01, 0.05, 0.1],
    "xgb__subsample":        [0.7, 0.8, 0.9],
    "lgbm__n_estimators":    [200, 300, 400],
    "lgbm__max_depth":       [3, 4, 5, 6],
    "lgbm__learning_rate":   [0.01, 0.05, 0.1],
    "lgbm__num_leaves":      [15, 31, 63],
}

def print_metrics(name, model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    acc   = accuracy_score(y_test, y_pred)
    auc   = roc_auc_score(y_test, y_prob)
    brier = brier_score_loss(y_test, y_prob)
    frac, mean_pred = calibration_curve(y_test, y_prob, n_bins=10)
    ece   = float(np.mean(np.abs(frac - mean_pred)))
    print(f"  {name}")
    print(f"    Accuracy : {acc*100:.1f}%")
    print(f"    AUC-ROC  : {auc:.4f}")
    print(f"    Brier    : {brier:.4f}  {'✓' if brier < 0.12 else '✗'}")
    print(f"    ECE      : {ece:.4f}  {'✓' if ece < 0.08 else '✗'}")


# ════════════════════════════════════════════
# BEHAVIORAL — retrain from scratch
# ════════════════════════════════════════════
print("\n[1/2] Behavioral — retraining from scratch...")
try:
    df   = pd.read_csv("data/processed/behavioral_final.csv")
    feat_cols = ['exclamation_count', 'has_superlative', 'word_count',
                 'caps_ratio', 'avg_sentence_length', 'sentiment_score', 'polarity']

    imp    = SimpleImputer(strategy="median")
    X      = imp.fit_transform(df[feat_cols])
    y      = df["deceptive"].values
    scaler = StandardScaler()
    X      = scaler.fit_transform(X)

    X_tr, X_test, y_tr, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    X_tr, X_cal, y_tr, y_cal   = train_test_split(
        X_tr, y_tr, test_size=0.15, stratify=y_tr, random_state=42)

    stack = build_stack()
    rs    = RandomizedSearchCV(stack, all_params, n_iter=20,
                               scoring="neg_brier_score", cv=3,
                               random_state=42, n_jobs=2, verbose=0)
    rs.fit(X_tr, y_tr)
    best  = rs.best_estimator_
    cal   = CalibratedClassifierCV(best, method="isotonic", cv="prefit")
    cal.fit(X_cal, y_cal)

    print_metrics("Behavioral v6", cal, X_test, y_test)
    joblib.dump({"model": cal, "scaler": scaler, "feat_cols": feat_cols,
                 "target_col": "deceptive"},
                "models/behavioral_stacking_v6.joblib")
    print("  Saved → models/behavioral_stacking_v6.joblib ✓")

except Exception as e:
    print(f"  ✗ FAILED — {e}")


# ════════════════════════════════════════════
# CLAIM — retrain from scratch
# ════════════════════════════════════════════
print("\n[2/2] Claim — retraining from scratch...")
try:
    df       = pd.read_csv("data/processed/claim_final.csv")
    num_cols = ['barely_true', 'false', 'half_true', 'mostly_true', 'pants_fire']

    tfidf    = TfidfVectorizer(max_features=8000, ngram_range=(1, 2),
                               sublinear_tf=True, min_df=2)
    svd      = TruncatedSVD(n_components=100, random_state=42)
    tfidf_mat = tfidf.fit_transform(df["statement"].astype(str))
    svd_mat   = svd.fit_transform(tfidf_mat)

    num_imp  = SimpleImputer(strategy="median")
    num_mat  = num_imp.fit_transform(df[num_cols])

    X        = np.hstack([svd_mat, num_mat])
    y        = df["credible"].values
    scaler   = StandardScaler()
    X        = scaler.fit_transform(X)

    X_tr, X_test, y_tr, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    X_tr, X_cal, y_tr, y_cal   = train_test_split(
        X_tr, y_tr, test_size=0.15, stratify=y_tr, random_state=42)

    stack = build_stack()
    rs    = RandomizedSearchCV(stack, all_params, n_iter=20,
                               scoring="neg_brier_score", cv=3,
                               random_state=42, n_jobs=2, verbose=0)
    rs.fit(X_tr, y_tr)
    best  = rs.best_estimator_
    cal   = CalibratedClassifierCV(best, method="isotonic", cv="prefit")
    cal.fit(X_cal, y_cal)

    print_metrics("Claim v6", cal, X_test, y_test)
    joblib.dump({"model": cal, "scaler": scaler, "tfidf": tfidf,
                 "svd": svd, "num_cols": num_cols, "target_col": "credible"},
                "models/claim_stacking_v6.joblib")
    print("  Saved → models/claim_stacking_v6.joblib ✓")

except Exception as e:
    print(f"  ✗ FAILED — {e}")

print("\nDone.")
