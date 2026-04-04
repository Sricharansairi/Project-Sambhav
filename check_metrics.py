import os, sys, warnings
warnings.filterwarnings('ignore')
# Get project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import joblib, numpy as np, pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score, accuracy_score
from sklearn.calibration import calibration_curve

# Only domains with local processed data
DOMAINS = [
    ("Student",    "models/student_stacking_v8.joblib",         "data/processed/student_final.csv",         "pass"),
    ("Higher Ed",  "models/student_dropout_stacking_v8.joblib", "data/processed/student_dropout_final.csv", "pass"),
    ("HR",         "models/hr_stacking_v8.joblib",              "data/processed/hr_final.csv",              "attrition"),
    ("Claim",      "models/claim_stacking_v8.joblib",           "data/processed/claim_final.csv",           "label"),
    ("Behavioral", "models/behavioral_minilm_v1.joblib",        "data/processed/behavioral_final.csv",      "label"),
]

# V8 Kaggle results (from handover doc) — for domains without local data
KAGGLE_RESULTS = [
    ("Disease",     "0.0817", "0.9513", "88.8%", "v6=0.0776 kept"),
    ("Fitness",     "0.0815", "0.9505", "89.2%", "v8 winner"),
    ("Loan",        "0.0337", "0.9184", "86.3%", "v5 kept — v8 worse"),
    ("MentalHealth","0.0359", "0.9912", "95.1%", "v8 winner — world class"),
]

def test_domain(name, model_path, data_path, target_override=None):
    try:
        art = joblib.load(model_path)
        df  = pd.read_csv(data_path).dropna()
        df  = df.sample(min(3000, len(df)), random_state=99)

        # Get target column
        target = target_override or art.get('target_col', None)
        if target is None:
            # Try common names
            for t in ['label','target','Attrition','dropout','outcome','sentiment']:
                if t in df.columns:
                    target = t
                    break
        if target not in df.columns:
            return None, f"target col '{target}' not in CSV"

        y = df[target].astype(int).values

        # Text models (tfidf+svd)
        if 'tfidf' in art:
            text_col = next((c for c in ['text','review','Text'] if c in df.columns), None)
            if text_col is None:
                text_col = df.select_dtypes('object').columns[0]
            X_tfidf = art['tfidf'].transform(df[text_col].astype(str))
            X_svd   = art['svd'].transform(X_tfidf)
            try:
                X = art['scaler'].transform(art['imputer'].transform(X_svd))
            except:
                X = art['scaler'].transform(X_svd)
            prob = (art['xgb_model'].predict_proba(X)[:,1] +
                    art['lgbm_model'].predict_proba(X)[:,1]) / 2

        # Normal models
        else:
            feats = art.get('feature_cols', [])
            feats = [f for f in feats if f in df.columns]
            if len(feats) == 0:
                return None, f"no matching features in CSV"
            X = df[feats].values
            try:
                X = art['imputer'].transform(X)
            except: pass
            X    = art['scaler'].transform(X)
            prob = art['xgb_model'].predict_proba(X)[:,1]

        brier = brier_score_loss(y, prob)
        auc   = roc_auc_score(y, prob)
        acc   = accuracy_score(y, (prob >= 0.5).astype(int))
        frac, mean_p = calibration_curve(y, prob, n_bins=10)
        ece   = float(np.mean(np.abs(frac - mean_p)))

        return {"brier":brier,"auc":auc,"acc":acc,"ece":ece}, None

    except Exception as e:
        return None, str(e)[:80]

print("\n" + "="*70)
print("  PROJECT SAMBHAV — MODEL PERFORMANCE SCORECARD")
print("  Target: Brier < 0.12 | ✓ Pass  ~ Close(0.12-0.15)  ✗ Fail")
print("="*70)
print(f"\n  {'Domain':<15} {'Accuracy':>8} {'AUC':>7} {'Brier':>7} {'ECE':>7}  Status")
print(f"  {'-'*60}")

results = []
for name, model_path, data_path, target in DOMAINS:
    metrics, err = test_domain(name, model_path, data_path, target)
    if metrics:
        b = metrics['brier']
        status = "✓ PASS" if b < 0.12 else "~ CLOSE" if b < 0.15 else "✗ FAIL"
        print(f"  {name:<15} {metrics['acc']*100:>7.1f}% {metrics['auc']:>7.4f} {b:>7.4f} {metrics['ece']:>7.4f}  {status}")
        results.append((name, metrics['acc'], metrics['auc'], b, metrics['ece']))
    else:
        print(f"  {name:<15} {'ERR':>8} {'ERR':>7} {'ERR':>7} {'ERR':>7}  ✗ {err}")

print(f"\n  {'--- Kaggle V8 Results (no local data) ---':}")
print(f"  {'Domain':<15} {'Accuracy':>8} {'AUC':>7} {'Brier':>7}  Notes")
print(f"  {'-'*60}")
for name, brier, auc, acc, note in KAGGLE_RESULTS:
    status = "✓ PASS" if float(brier) < 0.12 else "~ CLOSE"
    print(f"  {name:<15} {acc:>8} {auc:>7} {brier:>7}  {status} — {note}")

print("\n" + "="*70)
local_pass = sum(1 for _, _, _, b, _ in results if b < 0.12)
kaggle_pass = sum(1 for _, b, _, _, _ in KAGGLE_RESULTS if float(b) < 0.12)
print(f"  Local tested:  {local_pass}/{len(results)} pass Brier < 0.12")
print(f"  Kaggle V8:     {kaggle_pass}/4 pass Brier < 0.12")
print(f"  MentalHealth Brier=0.0359 — publication-grade performance")
print(f"  Claim: LLM layer is primary — ML is fast filter only")
print("="*70 + "\n")
