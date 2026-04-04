import joblib, numpy as np, pandas as pd, warnings, os
warnings.filterwarnings('ignore')
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
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
    return np.nan_to_num(np.array(X, dtype=np.float64),
                         nan=0.0, posinf=0.0, neginf=0.0)

def build_stack():
    estimators = [
        ('lr',  LogisticRegression(max_iter=1000, C=0.5, random_state=42)),
        ('rf',  RandomForestClassifier(n_estimators=300, max_depth=10,
                    min_samples_leaf=2, random_state=42, n_jobs=2)),
        ('xgb', xgb.XGBClassifier(n_estimators=300, max_depth=6,
                    learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                    eval_metric='logloss', random_state=42, n_jobs=2, verbosity=0)),
        ('lgb', lgb.LGBMClassifier(n_estimators=300, max_depth=6,
                    learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                    random_state=42, n_jobs=2, verbose=-1)),
    ]
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(C=1.0, max_iter=1000),
        cv=5, passthrough=True, n_jobs=1)
    return CalibratedClassifierCV(stack, method='isotonic', cv=5)

def save(model, scaler, path, extra=None):
    art = {'model': model, 'scaler': scaler}
    if extra: art.update(extra)
    joblib.dump(art, path)
    sz = os.path.getsize(path)/1e6
    print(f"  💾 Saved → {os.path.basename(path)} ({sz:.1f} MB)")

print("\n" + "="*60)
print("  FAST BEAST — MENTAL HEALTH + LOAN")
print("="*60)

# ══════════════════════════════════════════════════════════════
# MENTAL HEALTH — 30k sample + interaction features
# ══════════════════════════════════════════════════════════════
print("\n[1/2] MENTAL HEALTH (30k + interactions)...")
df  = pd.read_csv(f"{BASE}/data/raw/mental_health/Mental Health Dataset.csv")
df  = df.sample(n=30000, random_state=42)
y   = (df['treatment'].str.lower() == 'yes').astype(int).values

drop_cols = ['treatment', 'Timestamp']
feat_df   = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()

ordinal_maps = {
    'Days_Indoors':           {'Go out Every day':0,'1-14 days':1,
                               '15-30 days':2,'31-60 days':3,'60+ days':4},
    'Growing_Stress':         {'No':0,'Maybe':1,'Yes':2},
    'Changes_Habits':         {'No':0,'Maybe':1,'Yes':2},
    'Mental_Health_History':  {'No':0,'Maybe':1,'Yes':2},
    'Mood_Swings':            {'Low':0,'Medium':1,'High':2},
    'Coping_Struggles':       {'No':0,'Maybe':1,'Yes':2},
    'Work_Interest':          {'No':0,'Maybe':1,'Yes':2},
    'Social_Weakness':        {'No':0,'Maybe':1,'Yes':2},
    'mental_health_interview':{'No':0,'Maybe':1,'Yes':2},
    'care_options':           {'No':0,'Not sure':1,'Yes':2},
}
for col, mapping in ordinal_maps.items():
    if col in feat_df.columns:
        feat_df[col] = feat_df[col].map(mapping).fillna(1)

label_encoders = {}
for col in feat_df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    feat_df[col] = le.fit_transform(feat_df[col].astype(str))
    label_encoders[col] = le

base_cols = list(feat_df.columns)
F = feat_df.values.astype(np.float64)
col_idx = {c:i for i,c in enumerate(base_cols)}
def ci(n): return col_idx.get(n, 0)

# Key interaction features
interactions = np.column_stack([
    F[:,ci('Growing_Stress')] + F[:,ci('Mood_Swings')] +
    F[:,ci('Coping_Struggles')] + F[:,ci('Social_Weakness')],   # risk score
    F[:,ci('Work_Interest')] + F[:,ci('care_options')],          # protective
    F[:,ci('Growing_Stress')] * F[:,ci('Mood_Swings')],          # stress×mood
    F[:,ci('Mental_Health_History')] * F[:,ci('family_history')],# history×family
    F[:,ci('Days_Indoors')] * F[:,ci('Coping_Struggles')],       # isolation×coping
    2 - F[:,ci('Work_Interest')],                                 # inv work interest
    (F[:,ci('Growing_Stress')]*0.3 + F[:,ci('Mood_Swings')]*0.25 +
     F[:,ci('Coping_Struggles')]*0.25 + F[:,ci('Social_Weakness')]*0.2), # weighted risk
    F[:,ci('Growing_Stress')] ** 2,                               # stress squared
    F[:,ci('Social_Weakness')] + F[:,ci('Days_Indoors')],         # isolation composite
])

X = clean(np.hstack([F, interactions]))
feature_cols = base_cols + ['risk_score','protective','stress_x_mood',
    'history_x_family','isolation_x_coping','inv_work',
    'weighted_risk','stress_sq','isolation_composite']
print(f"  Features: {X.shape[1]} | Samples: {X.shape[0]}")

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
sc = StandardScaler()
X_tr_s = sc.fit_transform(X_tr)
X_te_s = sc.transform(X_te)
cal = build_stack()
cal.fit(X_tr_s, y_tr)
b = brier_score_loss(y_te, cal.predict_proba(X_te_s)[:,1])
_, X_v, _, y_v = train_test_split(X, y, test_size=0.2, stratify=y, random_state=99)
b_v = brier_score_loss(y_v, cal.predict_proba(sc.transform(X_v))[:,1])
print(f"  Train Brier: {b:.4f}  {grade(b)}")
print(f"  Verify Brier: {b_v:.4f}  {grade(b_v)}")
save(cal, sc, f"{BASE}/models/mental_health_stacking_v5.joblib", {
    'feature_cols': feature_cols, 'base_cols': base_cols,
    'ordinal_maps': ordinal_maps, 'label_encoders': label_encoders})

# ══════════════════════════════════════════════════════════════
# LOAN — rich derived features
# ══════════════════════════════════════════════════════════════
print("\n[2/2] LOAN (rich features)...")
df = pd.read_csv(f"{BASE}/data/raw/finance/loan_synthetic.csv")
y  = df['default'].values
df['loan_to_income']      = df['loan_amount'] / df['income'].clip(lower=1)
df['monthly_payment']     = df['loan_amount'] / df['loan_duration'].clip(lower=1)
df['payment_to_income']   = df['monthly_payment'] / (df['income']/12).clip(lower=1)
df['credit_risk_score']   = (df['credit_score']/850) - df['debt_to_income']
df['stability_score']     = df['employment_years'] / (df['existing_loans']+1)
df['missed_payment_rate'] = df['missed_payments'] / df['loan_duration'].clip(lower=1)
df['income_per_loan']     = df['income'] / df['loan_amount'].clip(lower=1)
df['credit_x_income']     = (df['credit_score']/850) * np.log1p(df['income'])
df['risk_composite']      = (df['debt_to_income']*0.4 +
                             df['missed_payments']/10*0.4 +
                             (1-df['credit_score']/850)*0.2)
df['affordability']       = df['income'] / df['monthly_payment'].clip(lower=1)
df['high_risk_flag']      = ((df['credit_score']<580) &
                             (df['missed_payments']>2)).astype(int)
df['safe_flag']           = ((df['credit_score']>750) &
                             (df['missed_payments']==0)).astype(int)
fc = [c for c in df.columns if c != 'default']
X  = clean(df[fc].values)
print(f"  Features: {len(fc)} | Samples: {len(X)}")
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
sc = StandardScaler()
X_tr_s = sc.fit_transform(X_tr)
X_te_s = sc.transform(X_te)
cal = build_stack()
cal.fit(X_tr_s, y_tr)
b = brier_score_loss(y_te, cal.predict_proba(X_te_s)[:,1])
_, X_v, _, y_v = train_test_split(X, y, test_size=0.2, stratify=y, random_state=99)
b_v = brier_score_loss(y_v, cal.predict_proba(sc.transform(X_v))[:,1])
print(f"  Train Brier: {b:.4f}  {grade(b)}")
print(f"  Verify Brier: {b_v:.4f}  {grade(b_v)}")
save(cal, sc, f"{BASE}/models/loan_stacking_v4.joblib", {'feature_cols': fc})

print("\n" + "="*60)
print("  ✅ DONE!")
print("  mental_health → mental_health_stacking_v5.joblib")
print("  loan          → loan_stacking_v4.joblib")
print("="*60 + "\n")
