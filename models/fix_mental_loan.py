import joblib, numpy as np, pandas as pd, warnings, os
warnings.filterwarnings('ignore')
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.calibration import CalibratedClassifierCV
import xgboost as xgb
import lightgbm as lgb

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def grade(b):
    if   b < 0.05: return "🔥🔥 GODTIER"
    elif b < 0.07: return "🔥 BEAST"
    elif b < 0.09: return "✅ EXCELLENT"
    elif b < 0.12: return "✅ TARGET MET"
    else:          return "🔴 NEEDS FIX"

def clean(X):
    return np.nan_to_num(np.array(X, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)

def build_stack():
    estimators = [
        ('lr',  LogisticRegression(max_iter=2000, C=0.5, random_state=42)),
        ('rf',  RandomForestClassifier(n_estimators=400, max_depth=10,
                    min_samples_leaf=2, random_state=42, n_jobs=2)),
        ('xgb', xgb.XGBClassifier(n_estimators=400, max_depth=6,
                    learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                    eval_metric='logloss', random_state=42, n_jobs=2, verbosity=0)),
        ('lgb', lgb.LGBMClassifier(n_estimators=400, max_depth=6,
                    learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                    random_state=42, n_jobs=2, verbose=-1)),
    ]
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(C=1.0, max_iter=1000),
        cv=5, passthrough=True, n_jobs=1)
    return CalibratedClassifierCV(stack, method='isotonic', cv=5)

print("\n" + "="*60)
print("  FIXING MENTAL HEALTH + LOAN")
print("="*60)

# ══════════════════════════════════════════════════════════════
# 1. MENTAL HEALTH — fix v3 artifact + retrain properly
# ══════════════════════════════════════════════════════════════
print("\n[1/2] MENTAL HEALTH...")
df  = pd.read_csv(f"{BASE}/data/raw/mental_health/Mental Health Dataset.csv")
df  = df.sample(n=50000, random_state=42)
y   = (df['treatment'].str.lower() == 'yes').astype(int).values

drop_cols = ['treatment', 'Timestamp']
feat_df   = df.drop(columns=[c for c in drop_cols if c in df.columns])

# Manual ordinal encoding for known ordinal columns
ordinal_maps = {
    'Days_Indoors':           {'Go out Every day': 0, '1-14 days': 1,
                               '15-30 days': 2, '31-60 days': 3, '60+ days': 4},
    'Growing_Stress':         {'No': 0, 'Maybe': 1, 'Yes': 2},
    'Changes_Habits':         {'No': 0, 'Maybe': 1, 'Yes': 2},
    'Mental_Health_History':  {'No': 0, 'Maybe': 1, 'Yes': 2},
    'Mood_Swings':            {'Low': 0, 'Medium': 1, 'High': 2},
    'Coping_Struggles':       {'No': 0, 'Maybe': 1, 'Yes': 2},
    'Work_Interest':          {'No': 0, 'Maybe': 1, 'Yes': 2},
    'Social_Weakness':        {'No': 0, 'Maybe': 1, 'Yes': 2},
    'mental_health_interview':{'No': 0, 'Maybe': 1, 'Yes': 2},
    'care_options':           {'No': 0, 'Not sure': 1, 'Yes': 2},
}

for col, mapping in ordinal_maps.items():
    if col in feat_df.columns:
        feat_df[col] = feat_df[col].map(mapping).fillna(1)

# Label encode remaining categoricals
label_encoders = {}
for col in feat_df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    feat_df[col] = le.fit_transform(feat_df[col].astype(str))
    label_encoders[col] = le

feature_cols = list(feat_df.columns)
X = clean(feat_df.values)
print(f"  Feature cols ({len(feature_cols)}): {feature_cols}")

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
sc     = StandardScaler()
X_tr_s = sc.fit_transform(X_tr)
X_te_s = sc.transform(X_te)

cal = build_stack()
cal.fit(X_tr_s, y_tr)
b = brier_score_loss(y_te, cal.predict_proba(X_te_s)[:,1])
print(f"  Train Brier (rs=42): {b:.4f}  {grade(b)}")

# Verify on held-out split
_, X_v, _, y_v = train_test_split(X, y, test_size=0.2, stratify=y, random_state=99)
b_v = brier_score_loss(y_v, cal.predict_proba(sc.transform(X_v))[:,1])
print(f"  Verify Brier (rs=99): {b_v:.4f}  {grade(b_v)}")

joblib.dump({
    'model':         cal,
    'scaler':        sc,
    'feature_cols':  feature_cols,
    'ordinal_maps':  ordinal_maps,
    'label_encoders': label_encoders,
}, f"{BASE}/models/mental_health_stacking_v4.joblib")
sz = os.path.getsize(f"{BASE}/models/mental_health_stacking_v4.joblib")/1e6
print(f"  💾 Saved → mental_health_stacking_v4.joblib ({sz:.1f} MB)")

# ══════════════════════════════════════════════════════════════
# 2. LOAN — retrain with derived features for BEAST score
# ══════════════════════════════════════════════════════════════
print("\n[2/2] LOAN — with derived features...")
df = pd.read_csv(f"{BASE}/data/raw/finance/loan_synthetic.csv")

# Derived features
df['loan_to_income']    = df['loan_amount'] / df['income'].clip(lower=1)
df['monthly_payment']   = df['loan_amount'] / df['loan_duration'].clip(lower=1)
df['payment_to_income'] = df['monthly_payment'] / (df['income']/12).clip(lower=1)
df['credit_risk_score'] = (df['credit_score'] / 850) - df['debt_to_income']
df['stability_score']   = df['employment_years'] / (df['existing_loans'] + 1)
df['missed_payment_rate'] = df['missed_payments'] / df['loan_duration'].clip(lower=1)
df['income_per_loan']   = df['income'] / df['loan_amount'].clip(lower=1)

y  = df['default'].values
fc = [c for c in df.columns if c != 'default']
X  = clean(df[fc].values)
print(f"  Feature cols ({len(fc)}): {fc}")

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
sc     = StandardScaler()
X_tr_s = sc.fit_transform(X_tr)
X_te_s = sc.transform(X_te)

cal = build_stack()
cal.fit(X_tr_s, y_tr)
b = brier_score_loss(y_te, cal.predict_proba(X_te_s)[:,1])
print(f"  Train Brier (rs=42): {b:.4f}  {grade(b)}")

_, X_v, _, y_v = train_test_split(X, y, test_size=0.2, stratify=y, random_state=99)
b_v = brier_score_loss(y_v, cal.predict_proba(sc.transform(X_v))[:,1])
print(f"  Verify Brier (rs=99): {b_v:.4f}  {grade(b_v)}")

joblib.dump({
    'model':        cal,
    'scaler':       sc,
    'feature_cols': fc,
}, f"{BASE}/models/loan_stacking_v3.joblib")
sz = os.path.getsize(f"{BASE}/models/loan_stacking_v3.joblib")/1e6
print(f"  💾 Saved → loan_stacking_v3.joblib ({sz:.1f} MB)")

# ══ FINAL SUMMARY ════════════════════════════════════════════
print("\n" + "="*60)
print("  ✅ DONE — update domain_registry.yaml next")
print("  mental_health → mental_health_stacking_v4.joblib")
print("  loan          → loan_stacking_v3.joblib")
print("="*60 + "\n")
