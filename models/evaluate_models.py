"""
evaluate_models.py  —  Project Sambhav
Evaluates claim (v2) and behavioral (v22) models.
Run from project root:  python evaluate_models.py
"""

import os, warnings, sys
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    brier_score_loss, roc_auc_score, accuracy_score,
    f1_score, classification_report, confusion_matrix
)
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────
# PATHS — adjust if needed
# ─────────────────────────────────────────────────
BASE     = os.path.expanduser("~/Desktop/Sri_Coding/Project Sambhav")
MODELS   = os.path.join(BASE, "models")
DATA     = os.path.join(BASE, "data/processed")
REPORTS  = os.path.join(BASE, "eval_reports")
os.makedirs(REPORTS, exist_ok=True)

CLAIM_MODEL   = os.path.join(MODELS, "claim_model_v2-2.pkl")
CLAIM_TFIDF   = os.path.join(MODELS, "claim_tfidf_v2.pkl")
CLAIM_SVD     = os.path.join(MODELS, "claim_svd_v2.pkl")
BEHAVIORAL    = os.path.join(MODELS, "behavioral_model_v22-2.pkl")

# ─────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────
def grade(brier):
    if brier <= 0.04:  return "A+"
    if brier <= 0.09:  return "A"
    if brier <= 0.16:  return "B"
    if brier <= 0.25:  return "C"
    if brier <= 0.36:  return "D"
    return "F"

def banner(title):
    print("\n" + "═"*60)
    print(f"  {title}")
    print("═"*60)

def metrics_report(y_true, y_prob, y_pred, name):
    brier = brier_score_loss(y_true, y_prob)
    auc   = roc_auc_score(y_true, y_prob)
    acc   = accuracy_score(y_true, y_pred)
    f1    = f1_score(y_true, y_pred, average='binary')
    print(f"\n  {'Metric':<22} {'Value':>10}")
    print(f"  {'─'*34}")
    print(f"  {'Brier Score':<22} {brier:>10.4f}  ← PRIMARY  grade={grade(brier)}")
    print(f"  {'ROC-AUC':<22} {auc:>10.4f}")
    print(f"  {'Accuracy':<22} {acc:>10.4f}  ({acc*100:.1f}%)")
    print(f"  {'F1 Score':<22} {f1:>10.4f}")
    print(f"\n{classification_report(y_true, y_pred, target_names=['Negative','Positive'])}")
    return {"domain": name, "brier": brier, "auc": auc, "accuracy": acc, "f1": f1, "grade": grade(brier)}

def plot_calibration(y_true, y_prob, name, outpath):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#08080A')
    for ax in axes:
        ax.set_facecolor('#141419')
        ax.tick_params(colors='#73717D')
        ax.xaxis.label.set_color('#EBE9F2')
        ax.yaxis.label.set_color('#EBE9F2')
        ax.title.set_color('#C2CD93')
        for spine in ax.spines.values():
            spine.set_edgecolor('#26242E')

    # Calibration curve
    ax = axes[0]
    fraction_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='uniform')
    ax.plot([0,1],[0,1], 'w--', lw=1.5, alpha=0.5, label='Perfect calibration')
    ax.plot(mean_pred, fraction_pos, 'o-', color='#C2CD93', lw=2, ms=6, label='Model')
    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives')
    ax.set_title(f'{name} — Reliability Diagram')
    ax.legend(facecolor='#1A1A21', labelcolor='#EBE9F2')
    ax.set_xlim([0,1]); ax.set_ylim([0,1])

    # Probability distribution
    ax = axes[1]
    ax.hist(y_prob[y_true==0], bins=30, alpha=0.7, color='#C891AA', label='Negative class', density=True)
    ax.hist(y_prob[y_true==1], bins=30, alpha=0.7, color='#C2CD93', label='Positive class', density=True)
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Density')
    ax.set_title(f'{name} — Probability Distribution')
    ax.legend(facecolor='#1A1A21', labelcolor='#EBE9F2')

    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight', facecolor='#08080A')
    plt.close()
    print(f"  → Chart saved: {outpath}")

# ─────────────────────────────────────────────────
# CLAIM MODEL EVALUATION
# ─────────────────────────────────────────────────
def eval_claim():
    banner("CLAIM CREDIBILITY — v2")

    # Load artifacts
    print("\n  Loading model artifacts...")
    loaded = joblib.load(CLAIM_MODEL)
    model = loaded['cal_xgb'] if isinstance(loaded, dict) else loaded
    tfidf = joblib.load(CLAIM_TFIDF)
    svd   = joblib.load(CLAIM_SVD)
    print(f"  Model type: {type(model).__name__}")
    print(f"  TF-IDF vocab: {len(tfidf.vocabulary_):,} features")
    print(f"  SVD components: {svd.n_components}")

    # Load test data — tries multiple known locations
    df = None
    candidates = [
        os.path.join(DATA, "claim_test.csv"),
        os.path.join(DATA, "claim_final.csv"),
        os.path.join(DATA, "claim_balanced.csv"),
        os.path.join(BASE, "data/raw/WELFake_Dataset.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"  Loaded: {path}  ({len(df):,} rows)")
            break

    if df is None:
        print("\n  ⚠  No test CSV found. Trying LIAR dataset from HuggingFace cache...")
        try:
            from datasets import load_dataset
            ds = load_dataset("liar", split="test")
            df = pd.DataFrame({"text": ds["statement"], "label": [1 if l in ["true","mostly-true","half-true"] else 0 for l in ds["label"]]})
            print(f"  Loaded LIAR test set: {len(df):,} rows")
        except:
            print("  ✗ Could not load LIAR. Place claim_final.csv in data/processed/ and retry.")
            return None

    # Find text and label columns
    text_col  = next((c for c in df.columns if 'text' in c.lower() or 'claim' in c.lower() or 'statement' in c.lower()), df.columns[0])
    label_col = next((c for c in df.columns if 'label' in c.lower() or 'credible' in c.lower() or 'fake' in c.lower()), df.columns[-1])
    print(f"  Using text_col='{text_col}', label_col='{label_col}'")

    df = df[[text_col, label_col]].dropna()
    df[label_col] = df[label_col].astype(int)
    X_text = df[text_col].astype(str).tolist()
    y_true = df[label_col].values

    # Keep only up to 50k rows for speed
    if len(df) > 50000:
        idx = np.random.RandomState(42).choice(len(df), 50000, replace=False)
        X_text = [X_text[i] for i in idx]
        y_true = y_true[idx]
        print(f"  Subsampled to 50,000 rows for speed")

    # Transform
    print(f"  Transforming {len(X_text):,} texts → TF-IDF → SVD...")
    X_tfidf = tfidf.transform(X_text)
    X_svd   = svd.transform(X_tfidf)
    X_svd = np.hstack([X_svd, np.zeros((X_svd.shape[0], 6))])

    # Also check if model needs isotonic calibrator
    if hasattr(model, 'predict_proba'):
        y_prob  = model.predict_proba(X_svd)[:,1]
        y_pred  = model.predict(X_svd)
    else:
        y_prob  = model.decision_function(X_svd)
        y_prob  = 1 / (1 + np.exp(-y_prob))
        y_pred  = (y_prob > 0.5).astype(int)

    result = metrics_report(y_true, y_prob, y_pred, "Claim")
    plot_calibration(y_true, y_prob, "Claim v2", os.path.join(REPORTS, "claim_v2_calibration.png"))
    return result

# ─────────────────────────────────────────────────
# BEHAVIORAL MODEL EVALUATION
# ─────────────────────────────────────────────────
def eval_behavioral():
    banner("BEHAVIORAL CHOICE — v22")

    print("\n  Loading model artifacts...")
    model = joblib.load(BEHAVIORAL)
    print(f"  Model type: {type(model).__name__}")
    if hasattr(model, 'n_features_in_'):
        print(f"  n_features_in_: {model.n_features_in_}")

    # Check if behavioral uses the same TF-IDF pipeline or separate
    # Try to find a behavioral TF-IDF (v22 might be self-contained)
    behav_tfidf_path = os.path.join(MODELS, "behavioral_tfidf_v22.pkl")
    behav_svd_path   = os.path.join(MODELS, "behavioral_svd_v22.pkl")

    has_tfidf = os.path.exists(behav_tfidf_path) and os.path.exists(behav_svd_path)
    if has_tfidf:
        b_tfidf = joblib.load(behav_tfidf_path)
        b_svd   = joblib.load(behav_svd_path)
        print(f"  Found behavioral TF-IDF: {len(b_tfidf.vocabulary_):,} features")
    else:
        print("  No separate behavioral TF-IDF found — checking if model is a pipeline...")
        from sklearn.pipeline import Pipeline
        if isinstance(model, Pipeline):
            print(f"  Pipeline steps: {[s[0] for s in model.steps]}")
            has_tfidf = True
            b_tfidf = b_svd = None  # pipeline handles it

    # Load test data
    df = None
    candidates = [
        os.path.join(DATA, "behavioral_test.csv"),
        os.path.join(DATA, "behavioral_final.csv"),
        os.path.join(DATA, "behavioral_balanced.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"  Loaded: {path}  ({len(df):,} rows)")
            break

    if df is None:
        print("\n  ⚠  No behavioral CSV found. Trying Cornell Deceptive Spam / Yelp reviews...")
        try:
            from datasets import load_dataset
            ds = load_dataset("yelp_review_full", split="test[:5000]")
            df = pd.DataFrame({
                "text":  ds["text"],
                "label": [1 if s >= 3 else 0 for s in ds["label"]]
            })
            print(f"  Loaded Yelp test sample: {len(df):,} rows")
        except:
            print("  ✗ Could not load behavioral test data.")
            print("  Place behavioral_final.csv in data/processed/ and retry.")
            return None

    text_col  = next((c for c in df.columns if 'text' in c.lower() or 'review' in c.lower() or 'comment' in c.lower()), df.columns[0])
    label_col = next((c for c in df.columns if 'label' in c.lower() or 'fake' in c.lower() or 'genuine' in c.lower()), df.columns[-1])
    print(f"  Using text_col='{text_col}', label_col='{label_col}'")

    df = df[[text_col, label_col]].dropna()
    df[label_col] = df[label_col].astype(int)
    X_text = df[text_col].astype(str).tolist()
    y_true = df[label_col].values

    if len(df) > 50000:
        idx = np.random.RandomState(42).choice(len(df), 50000, replace=False)
        X_text = [X_text[i] for i in idx]
        y_true = y_true[idx]

    print(f"  Evaluating {len(X_text):,} samples...")

    from sklearn.pipeline import Pipeline
   # 🔥 HANDLE DICT MODEL FIRST
    loaded = model
    if isinstance(loaded, dict):
        model = loaded[list(loaded.keys())[0]]  # pick first model safely


    # 🔥 NORMAL PREDICTION FLOW
    if isinstance(model, Pipeline):
        probs = model.predict_proba(X_text)[:,1]
        y_pred = model.predict(X_text)

    elif has_tfidf and b_tfidf is not None:
        X_tfidf = b_tfidf.transform(X_text)
        X_svd   = b_svd.transform(X_tfidf)
        probs   = model.predict_proba(X_svd)[:,1]
        y_pred  = model.predict(X_svd)

    else:
        print("  ⚠  Unclear feature format — attempting direct predict on text features")
        probs  = model.predict_proba(X_text)[:,1]
        y_pred = model.predict(X_text)


    # 🔥 APPLY CALIBRATION (VERY IMPORTANT)
    if iso is not None:
        y_prob = iso.transform(probs)
    else:
        y_prob = probs

    result = metrics_report(y_true, y_prob, y_pred, "Behavioral")
    plot_calibration(y_true, y_prob, "Behavioral v22", os.path.join(REPORTS, "behavioral_v22_calibration.png"))
    return result

# ─────────────────────────────────────────────────
# SUMMARY TABLE
# ─────────────────────────────────────────────────
def summary_table(results):
    banner("SUMMARY — BOTH MODELS")
    header = f"\n  {'Domain':<18} {'Brier':>8} {'AUC':>8} {'Accuracy':>10} {'F1':>8} {'Grade':>8}"
    print(header)
    print("  " + "─"*56)
    for r in results:
        if r:
            print(f"  {r['domain']:<18} {r['brier']:>8.4f} {r['auc']:>8.4f} {r['accuracy']:>10.4f} {r['f1']:>8.4f} {r['grade']:>8}")

    print(f"\n  Reports saved to: {REPORTS}/")
    print("\n  Grading scale: A+ ≤0.04 | A ≤0.09 | B ≤0.16 | C ≤0.25 | D ≤0.36 | F >0.36")
    print("  Primary metric = Brier Score (measures probability calibration)\n")

# ─────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "═"*60)
    print("  PROJECT SAMBHAV — MODEL EVALUATION SCRIPT")
    print("  Brier Score · AUC · Accuracy · Calibration Charts")
    print("═"*60)

    results = []

    if os.path.exists(CLAIM_MODEL):
        try:
            r = eval_claim()
            results.append(r)
        except Exception as e:
            print(f"\n  ERROR in claim eval: {e}")
            import traceback; traceback.print_exc()
    else:
        print(f"\n  ⚠  claim_model_v2-2.pkl not found at {CLAIM_MODEL}")
        print("  Copy the uploaded models to ~/Desktop/Sri_Coding/Project Sambhav/models/")

    if os.path.exists(BEHAVIORAL):
        try:
            r = eval_behavioral()
            results.append(r)
        except Exception as e:
            print(f"\n  ERROR in behavioral eval: {e}")
            import traceback; traceback.print_exc()
    else:
        print(f"\n  ⚠  behavioral_model_v22-2.pkl not found at {BEHAVIORAL}")

    summary_table(results)