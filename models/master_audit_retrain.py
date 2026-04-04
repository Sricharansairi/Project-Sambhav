import joblib, numpy as np, pandas as pd, warnings, os
warnings.filterwarnings('ignore')
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import xgboost as xgb
import lightgbm as lgb

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results = {}

def grade(b):
    if   b < 0.05: return "🔥🔥 GODTIER"
    elif b < 0.07: return "🔥 BEAST"
    elif b < 0.09: return "✅ EXCELLENT"
    elif b < 0.12: return "✅ TARGET MET"
    else:          return "🔴 NEEDS FIX"

def clean(X):
    """Fill NaN + Inf safely"""
    X = np.array(X, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X

def build_stack():
    estimators = [
        ('lr',  LogisticRegression(max_iter=1000, C=0.1, random_state=42)),
        ('rf',  RandomForestClassifier(n_estimators=300, max_depth=8,
                    min_samples_leaf=4, random_state=42, n_jobs=2)),
        ('xgb', xgb.XGBClassifier(n_estimators=300, max_depth=5,
                    learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                    eval_metric='logloss', random_state=42,
                    n_jobs=2, verbosity=0)),
        ('lgb', lgb.LGBMClassifier(n_estimators=300, max_depth=5,
                    learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                    random_state=42, n_jobs=2, verbose=-1)),
    ]
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(C=0.5, max_iter=1000),
        cv=5, passthrough=True, n_jobs=1)
    return CalibratedClassifierCV(stack, method='isotonic', cv=5)

def train_eval_save(X, y, save_path, label):
    X = clean(X)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s  = sc.transform(X_te)
    cal = build_stack()
    cal.fit(X_tr_s, y_tr)
    b = brier_score_loss(y_te, cal.predict_proba(X_te_s)[:,1])
    joblib.dump({'model': cal, 'scaler': sc}, save_path)
    sz = os.path.getsize(save_path)/1e6
    print(f"    Brier → {b:.4f}  {grade(b)}")
    print(f"    💾 Saved → {os.path.basename(save_path)} ({sz:.1f} MB)")
    results[label] = b
    return b, cal, sc

print("\n" + "="*60)
print("  PROJECT SAMBHAV — BEAST RETRAIN (NaN-safe)")
print("="*60)

# ── 1. STUDENT ───────────────────────────────────────────────
print("\n[1/9] STUDENT...")
df = pd.read_csv(f"{BASE}/data/processed/student_final.csv")
y  = df['pass'].values
X  = df.drop(columns=['pass']).select_dtypes(include=[np.number]).values
train_eval_save(X, y, f"{BASE}/models/student_stacking_v3.joblib", "Student")

# ── 2. HIGHER EDUCATION ──────────────────────────────────────
print("\n[2/9] HIGHER EDUCATION...")
df = pd.read_csv(f"{BASE}/data/processed/student_dropout_final.csv")
y  = df['pass'].values
X  = df.drop(columns=['pass']).select_dtypes(include=[np.number]).values
train_eval_save(X, y, f"{BASE}/models/student_dropout_stacking_v4.joblib", "Higher Edu")

# ── 3. HR ATTRITION ──────────────────────────────────────────
print("\n[3/9] HR ATTRITION...")
df = pd.read_csv(f"{BASE}/data/processed/hr_final.csv")
y  = df['attrition'].values
X  = df.drop(columns=['attrition']).select_dtypes(include=[np.number]).values
train_eval_save(X, y, f"{BASE}/models/hr_stacking_v3.joblib", "HR Attrition")

# ── 4. BEHAVIORAL ────────────────────────────────────────────
print("\n[4/9] BEHAVIORAL (correct features)...")
df = pd.read_csv(f"{BASE}/data/processed/behavioral_final.csv")
y  = df['deceptive'].values
feat_cols = [c for c in ['exclamation_count','has_superlative','word_count',
             'caps_ratio','avg_sentence_length','sentiment_score','polarity']
             if c in df.columns]
print(f"    Using features: {feat_cols}")
X  = df[feat_cols].values
train_eval_save(X, y, f"{BASE}/models/behavioral_stacking_v4.joblib", "Behavioral")

# ── 5. CLAIM (TF-IDF + SVD + numeric) ────────────────────────
print("\n[5/9] CLAIM (TF-IDF 150 SVD components + 5 numeric)...")
df = pd.read_csv(f"{BASE}/data/processed/claim_final.csv")
y  = df['credible'].values
num_cols = ['barely_true','false','half_true','mostly_true','pants_fire']
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1,2), sublinear_tf=True)
X_text = tfidf.fit_transform(df['statement'].fillna(''))
svd    = TruncatedSVD(n_components=150, random_state=42)
X_svd  = svd.fit_transform(X_text)
X_num  = df[num_cols].fillna(0).values
X_all  = np.hstack([X_svd, X_num])
X_all  = clean(X_all)
X_tr, X_te, y_tr, y_te = train_test_split(X_all, y, test_size=0.2, stratify=y, random_state=42)
sc = StandardScaler()
X_tr_s = sc.fit_transform(X_tr); X_te_s = sc.transform(X_te)
cal = build_stack(); cal.fit(X_tr_s, y_tr)
b = brier_score_loss(y_te, cal.predict_proba(X_te_s)[:,1])
joblib.dump({'model': cal, 'scaler': sc, 'tfidf': tfidf, 'svd': svd, 'num_cols': num_cols},
            f"{BASE}/models/claim_stacking_v4.joblib")
sz = os.path.getsize(f"{BASE}/models/claim_stacking_v4.joblib")/1e6
print(f"    Brier → {b:.4f}  {grade(b)}")
print(f"    💾 Saved → claim_stacking_v4.joblib ({sz:.1f} MB)")
results['Claim'] = b

# ── 6. DISEASE (encode categoricals) ─────────────────────────
print("\n[6/9] DISEASE (with categorical encoding)...")
df = pd.read_csv(f"{BASE}/data/raw/medical/heart.csv")
for col in ['Sex','ChestPainType','RestingECG','ExerciseAngina','ST_Slope']:
    if col in df.columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
y  = df['HeartDisease'].values
X  = df.drop(columns=['HeartDisease']).values
train_eval_save(X, y, f"{BASE}/models/disease_stacking_v2.joblib", "Disease")

# ── 7. MENTAL HEALTH (ordinal encode all) ────────────────────
print("\n[7/9] MENTAL HEALTH (50k sample + ordinal encoding)...")
df = pd.read_csv(f"{BASE}/data/raw/mental_health/Mental Health Dataset.csv")
df = df.sample(n=50000, random_state=42)
y  = (df['treatment'].str.lower() == 'yes').astype(int).values
drop_cols = ['treatment','Timestamp']
feat_df = df.drop(columns=[c for c in drop_cols if c in df.columns])
enc = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
X   = enc.fit_transform(feat_df.astype(str))
X   = clean(X)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
sc = StandardScaler()
X_tr_s = sc.fit_transform(X_tr); X_te_s = sc.transform(X_te)
cal = build_stack(); cal.fit(X_tr_s, y_tr)
b = brier_score_loss(y_te, cal.predict_proba(X_te_s)[:,1])
joblib.dump({'model': cal, 'scaler': sc, 'encoder': enc,
             'feature_cols': list(feat_df.columns)},
            f"{BASE}/models/mental_health_stacking_v3.joblib")
sz = os.path.getsize(f"{BASE}/models/mental_health_stacking_v3.joblib")/1e6
print(f"    Brier → {b:.4f}  {grade(b)}")
print(f"    💾 Saved → mental_health_stacking_v3.joblib ({sz:.1f} MB)")
results['Mental Health'] = b

# ── 8. LOAN ───────────────────────────────────────────────────
print("\n[8/9] LOAN...")
df = pd.read_csv(f"{BASE}/data/raw/finance/loan_synthetic.csv")
y  = df['default'].values
X  = df.drop(columns=['default']).values
train_eval_save(X, y, f"{BASE}/models/loan_stacking_v2.joblib", "Loan")

# ── 9. FITNESS (encode gender) ────────────────────────────────
print("\n[9/9] FITNESS (with gender encoding)...")
df = pd.read_csv(f"{BASE}/data/raw/fitness/bodyPerformance.csv")
df['gender_enc'] = (df['gender'] == 'M').astype(int)
df['fit']        = df['class'].isin(['A','B']).astype(int)
feat_cols = ['age','gender_enc','height_cm','weight_kg','body fat_%',
             'diastolic','systolic','gripForce',
             'sit and bend forward_cm','sit-ups counts','broad jump_cm']
y  = df['fit'].values
X  = df[feat_cols].values
train_eval_save(X, y, f"{BASE}/models/fitness_stacking_v2.joblib", "Fitness")

# ══ FINAL LEADERBOARD ════════════════════════════════════════
print("\n" + "="*60)
print("  🏆  FINAL LEADERBOARD — PROJECT SAMBHAV")
print("="*60)
print(f"  {'Domain':<18} {'Brier':>8}   Grade")
print("-"*60)
for domain, b in sorted(results.items(), key=lambda x: x[1]):
    print(f"  {domain:<18} {b:>8.4f}   {grade(b)}")
print("-"*60)
avg = np.mean(list(results.values()))
print(f"  {'AVERAGE':<18} {avg:>8.4f}   {grade(avg)}")
print("="*60 + "\n")
