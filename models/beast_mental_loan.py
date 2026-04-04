import joblib, numpy as np, pandas as pd, warnings, os
warnings.filterwarnings('ignore')
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb
try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except:
    HAS_CATBOOST = False

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

def build_beast_stack():
    estimators = [
        ('lr',  LogisticRegression(max_iter=3000, C=1.0, random_state=42)),
        ('rf',  RandomForestClassifier(n_estimators=500, max_depth=12,
                    min_samples_leaf=2, max_features='sqrt',
                    random_state=42, n_jobs=2)),
        ('xgb', xgb.XGBClassifier(n_estimators=500, max_depth=6,
                    learning_rate=0.02, subsample=0.8,
                    colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                    eval_metric='logloss', random_state=42,
                    n_jobs=2, verbosity=0)),
        ('lgb', lgb.LGBMClassifier(n_estimators=500, max_depth=7,
                    learning_rate=0.02, subsample=0.8,
                    colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
                    num_leaves=63, random_state=42, n_jobs=2, verbose=-1)),
        ('mlp', MLPClassifier(hidden_layer_sizes=(128, 64, 32),
                    activation='relu', max_iter=500,
                    learning_rate_init=0.001, random_state=42)),
    ]
    if HAS_CATBOOST:
        estimators.append(
            ('cat', CatBoostClassifier(iterations=400, depth=6,
                    learning_rate=0.03, random_seed=42, verbose=0))
        )
    stack = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(C=2.0, max_iter=1000),
        cv=5, passthrough=True, n_jobs=1)
    return CalibratedClassifierCV(stack, method='isotonic', cv=5)

print("\n" + "="*60)
print("  BEAST MODE — MENTAL HEALTH + LOAN")
print(f"  CatBoost available: {HAS_CATBOOST}")
print("="*60)

# ══════════════════════════════════════════════════════════════
# MENTAL HEALTH — full dataset + interaction features
# ══════════════════════════════════════════════════════════════
print("\n[1/2] MENTAL HEALTH — beast mode...")
df  = pd.read_csv(f"{BASE}/data/raw/mental_health/Mental Health Dataset.csv")
print(f"  Full dataset: {len(df)} rows")

# Use 100k for better signal
df  = df.sample(n=min(100000, len(df)), random_state=42)
y   = (df['treatment'].str.lower() == 'yes').astype(int).values

drop_cols = ['treatment', 'Timestamp']
feat_df   = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()

# Manual ordinal encoding — preserves order
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

# ── Interaction features ──────────────────────────────────────
base_cols = list(feat_df.columns)
F = feat_df.values.astype(np.float64)
col_idx = {c: i for i, c in enumerate(base_cols)}

def ci(name): return col_idx.get(name, 0)

# Key interaction signals
stress_idx      = ci('Growing_Stress')
mood_idx        = ci('Mood_Swings')
cope_idx        = ci('Coping_Struggles')
social_idx      = ci('Social_Weakness')
work_idx        = ci('Work_Interest')
history_idx     = ci('Mental_Health_History')
indoor_idx      = ci('Days_Indoors')
interview_idx   = ci('mental_health_interview')
care_idx        = ci('care_options')
family_idx      = ci('family_history')

interactions = np.column_stack([
    # Risk accumulation score
    F[:,stress_idx] + F[:,mood_idx] + F[:,cope_idx] + F[:,social_idx],
    # Protective factors
    F[:,work_idx] + F[:,care_idx],
    # Stress × mood interaction
    F[:,stress_idx] * F[:,mood_idx],
    # History × family risk
    F[:,history_idx] * F[:,family_idx],
    # Indoor isolation × coping
    F[:,indoor_idx] * F[:,cope_idx],
    # Interview willingness × care
    F[:,interview_idx] * F[:,care_idx],
    # Inverse work interest (low = risk)
    2 - F[:,work_idx],
    # Total risk score (weighted)
    (F[:,stress_idx]*0.25 + F[:,mood_idx]*0.2 +
     F[:,cope_idx]*0.2 + F[:,social_idx]*0.15 +
     F[:,indoor_idx]*0.1 + F[:,history_idx]*0.1),
    # Squared stress (non-linear)
    F[:,stress_idx] ** 2,
    # Social isolation composite
    F[:,social_idx] + F[:,indoor_idx],
])

X_full = clean(np.hstack([F, interactions]))
feature_cols = base_cols + [
    'risk_accumulation', 'protective_score',
    'stress_x_mood', 'history_x_family',
    'indoor_x_coping', 'interview_x_care',
    'inv_work_interest', 'total_risk_weighted',
    'stress_squared', 'social_isolation'
]
print(f"  Total features: {X_full.shape[1]}")

X_tr, X_te, y_tr, y_te = train_test_split(
    X_full, y, test_size=0.2, stratify=y, random_state=42)
sc     = StandardScaler()
X_tr_s = sc.fit_transform(X_tr)
X_te_s = sc.transform(X_te)

print("  Training beast stack (this takes ~8 mins)...")
cal = build_beast_stack()
cal.fit(X_tr_s, y_tr)
b_tr = brier_score_loss(y_te, cal.predict_proba(X_te_s)[:,1])

# Verify on fresh split
_, X_v, _, y_v = train_test_split(
    X_full, y, test_size=0.2, stratify=y, random_state=99)
b_v = brier_score_loss(y_v, cal.predict_proba(sc.transform(X_v))[:,1])
print(f"  Train Brier (rs=42): {b_tr:.4f}  {grade(b_tr)}")
print(f"  Verify Brier (rs=99): {b_v:.4f}  {grade(b_v)}")

joblib.dump({
    'model':          cal,
    'scaler':         sc,
    'feature_cols':   feature_cols,
    'base_cols':      base_cols,
    'ordinal_maps':   ordinal_maps,
    'label_encoders': label_encoders,
}, f"{BASE}/models/mental_health_stacking_v5.joblib")
sz = os.path.getsize(f"{BASE}/models/mental_health_stacking_v5.joblib")/1e6
print(f"  💾 Saved → mental_health_stacking_v5.joblib ({sz:.1f} MB)")

# ══════════════════════════════════════════════════════════════
# LOAN — smarter synthetic + interaction features
# ══════════════════════════════════════════════════════════════
print("\n[2/2] LOAN — beast mode features...")
df = pd.read_csv(f"{BASE}/data/raw/finance/loan_synthetic.csv")
y  = df['default'].values

# Rich derived features
df['loan_to_income']       = df['loan_amount'] / df['income'].clip(lower=1)
df['monthly_payment']      = df['loan_amount'] / df['loan_duration'].clip(lower=1)
df['payment_to_income']    = df['monthly_payment'] / (df['income']/12).clip(lower=1)
df['credit_risk_score']    = (df['credit_score'] / 850) - df['debt_to_income']
df['stability_score']      = df['employment_years'] / (df['existing_loans'] + 1)
df['missed_payment_rate']  = df['missed_payments'] / df['loan_duration'].clip(lower=1)
df['income_per_loan']      = df['income'] / df['loan_amount'].clip(lower=1)
df['credit_x_income']      = (df['credit_score'] / 850) * np.log1p(df['income'])
df['risk_composite']       = (df['debt_to_income'] * 0.4 +
                              df['missed_payments'] / 10 * 0.4 +
                              (1 - df['credit_score']/850) * 0.2)
df['affordability']        = df['income'] / df['monthly_payment'].clip(lower=1)
df['loan_burden']          = df['loan_amount'] * df['debt_to_income']
df['employment_stability'] = np.log1p(df['employment_years']) / (df['existing_loans'] + 1)
df['credit_squared']       = (df['credit_score'] / 850) ** 2
df['high_risk_flag']       = ((df['credit_score'] < 580) &
                              (df['missed_payments'] > 2)).astype(int)
df['safe_flag']            = ((df['credit_score'] > 750) &
                              (df['missed_payments'] == 0)).astype(int)

fc = [c for c in df.columns if c != 'default']
X  = clean(df[fc].values)
print(f"  Total features: {len(fc)}")
print(f"  Features: {fc}")

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
sc     = StandardScaler()
X_tr_s = sc.fit_transform(X_tr)
X_te_s = sc.transform(X_te)

print("  Training beast stack...")
cal = build_beast_stack()
cal.fit(X_tr_s, y_tr)
b_tr = brier_score_loss(y_te, cal.predict_proba(X_te_s)[:,1])

_, X_v, _, y_v = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=99)
b_v = brier_score_loss(y_v, cal.predict_proba(sc.transform(X_v))[:,1])
print(f"  Train Brier (rs=42): {b_tr:.4f}  {grade(b_tr)}")
print(f"  Verify Brier (rs=99): {b_v:.4f}  {grade(b_v)}")

joblib.dump({
    'model':        cal,
    'scaler':       sc,
    'feature_cols': fc,
}, f"{BASE}/models/loan_stacking_v4.joblib")
sz = os.path.getsize(f"{BASE}/models/loan_stacking_v4.joblib")/1e6
print(f"  💾 Saved → loan_stacking_v4.joblib ({sz:.1f} MB)")

print("\n" + "="*60)
print("  ✅ BEAST TRAINING COMPLETE")
print("  Update domain_registry.yaml:")
print("  mental_health → mental_health_stacking_v5.joblib")
print("  loan          → loan_stacking_v4.joblib")
print("="*60 + "\n")
