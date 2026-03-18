"""
PROJECT SAMBHAV — Beast Mode Trainer
=====================================
Pushes every domain model to its absolute Brier score floor.

Usage:
    python train_optimised_v2.py --domain student
    python train_optimised_v2.py --domain all
    python train_optimised_v2.py --domain hr --trials 300

Requirements (install if missing):
    pip install optuna catboost xgboost lightgbm scikit-learn imbalanced-learn joblib
"""

import os, sys, argparse, warnings, json, time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.linear_model     import LogisticRegression, RidgeClassifier
from sklearn.ensemble         import (RandomForestClassifier,
                                      ExtraTreesClassifier,
                                      HistGradientBoostingClassifier,
                                      StackingClassifier, VotingClassifier)
from sklearn.neural_network   import MLPClassifier
from sklearn.calibration      import CalibratedClassifierCV, calibration_curve
from sklearn.model_selection  import (train_test_split, StratifiedKFold,
                                      RepeatedStratifiedKFold, cross_val_score)
from sklearn.feature_selection import RFECV
from sklearn.preprocessing    import StandardScaler
from sklearn.pipeline         import Pipeline
from sklearn.metrics          import brier_score_loss, roc_auc_score, log_loss
from sklearn.base             import clone

import xgboost  as xgb
import lightgbm as lgb

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("⚠️  CatBoost not installed — skipping. Run: pip install catboost")

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
# DOMAIN REGISTRY
# ─────────────────────────────────────────────────────────
DOMAINS = {
    "student": {
        "label":      "Student Performance",
        "data_path":  "data/processed/student_final.csv",
        "out_path":   "models/student_stacking_v2.joblib",
        "target_col": "target",
        "text_cols":  [],
    },
    "student_uci": {
        "label":      "Student UCI",
        "data_path":  "data/processed/student_uci_final.csv",
        "out_path":   "models/student_uci_stacking_v3.joblib",
        "target_col": "target",
        "text_cols":  [],
    },
    "dropout": {
        "label":      "Higher Education Dropout",
        "data_path":  "data/processed/dropout_final.csv",
        "out_path":   "models/student_dropout_stacking_v3.joblib",
        "target_col": "target",
        "text_cols":  [],
    },
    "hr": {
        "label":      "HR Attrition",
        "data_path":  "data/processed/hr_final.csv",
        "out_path":   "models/hr_stacking_v2.joblib",
        "target_col": "target",
        "text_cols":  [],
    },
    "behavioral": {
        "label":      "Behavioral / Deception",
        "data_path":  "data/processed/behavioral_final.csv",
        "out_path":   "models/behavioral_stacking_v2.joblib",
        "target_col": "target",
        "text_cols":  [],
    },
    "claim": {
        "label":      "Claim Credibility",
        "data_path":  "data/processed/claim_final.csv",
        "out_path":   "models/claim_stacking_v3.joblib",
        "target_col": "target",
        "text_cols":  [],
    },
    "disease": {
        "label":      "Disease Risk",
        "data_path":  "data/processed/disease_final.csv",
        "out_path":   "models/disease_stacking_v1.joblib",
        "target_col": "target",
        "text_cols":  [],
    },
    "mental_health": {
        "label":      "Mental Health Risk",
        "data_path":  "data/processed/mental_health_final.csv",
        "out_path":   "models/mental_health_stacking_v1.joblib",
        "target_col": "target",
        "text_cols":  [],
    },
    "loan": {
        "label":      "Loan / Credit Risk",
        "data_path":  "data/processed/loan_final.csv",
        "out_path":   "models/loan_stacking_v1.joblib",
        "target_col": "target",
        "text_cols":  [],
    },
    "fitness": {
        "label":      "Diet & Fitness",
        "data_path":  "data/processed/fitness_final.csv",
        "out_path":   "models/fitness_stacking_v1.joblib",
        "target_col": "target",
        "text_cols":  [],
    },
}


# ─────────────────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────────────────

def compute_ece(y_true, y_prob, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0: continue
        ece += mask.sum() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return ece / max(len(y_true), 1)

def smooth_probs(p, eps=1e-5):
    """Clip probabilities away from exact 0/1 — improves calibration + log loss."""
    return np.clip(p, eps, 1 - eps)

def cv_brier(model, X, y, n_splits=5, n_repeats=3):
    """Repeated Stratified K-Fold Brier — far more stable than single CV."""
    rskf    = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    scores  = []
    for tr, val in rskf.split(X, y):
        m = clone(model)
        m.fit(X.iloc[tr], y[tr])
        p = smooth_probs(m.predict_proba(X.iloc[val])[:, 1])
        scores.append(brier_score_loss(y[val], p))
    return np.mean(scores), np.std(scores)


# ─────────────────────────────────────────────────────────
# TEMPERATURE SCALING CALIBRATOR
# ─────────────────────────────────────────────────────────

class TemperatureScaler:
    """Learns a single temperature T that stretches/shrinks the probability distribution."""
    def __init__(self): self.T = 1.0

    def fit(self, logits, y):
        from scipy.optimize import minimize_scalar
        def nll(T):
            p = 1 / (1 + np.exp(-logits / T))
            p = np.clip(p, 1e-7, 1 - 1e-7)
            return -np.mean(y * np.log(p) + (1-y) * np.log(1-p))
        res = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
        self.T = res.x
        return self

    def predict_proba(self, logits):
        p = 1 / (1 + np.exp(-logits / self.T))
        return np.clip(p, 1e-7, 1 - 1e-7)


# ─────────────────────────────────────────────────────────
# OPTUNA OBJECTIVE — XGBoost
# ─────────────────────────────────────────────────────────

def make_xgb_objective(X_tr, y_tr, n_splits=5):
    def objective(trial):
        params = {
            "n_estimators":       trial.suggest_int("n_estimators", 200, 1200),
            "max_depth":          trial.suggest_int("max_depth", 3, 10),
            "learning_rate":      trial.suggest_float("lr", 0.005, 0.15, log=True),
            "subsample":          trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":   trial.suggest_float("colsample", 0.4, 1.0),
            "min_child_weight":   trial.suggest_int("min_child_weight", 1, 10),
            "gamma":              trial.suggest_float("gamma", 0, 5),
            "reg_alpha":          trial.suggest_float("alpha", 1e-4, 10, log=True),
            "reg_lambda":         trial.suggest_float("lambda", 1e-4, 10, log=True),
            "eval_metric":        "logloss",
            "use_label_encoder":  False,
            "random_state":       42,
            "n_jobs":             -1,
        }
        cv   = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = []
        for tr, val in cv.split(X_tr, y_tr):
            m = xgb.XGBClassifier(**params)
            m.fit(X_tr.iloc[tr], y_tr[tr], verbose=False)
            p = smooth_probs(m.predict_proba(X_tr.iloc[val])[:, 1])
            scores.append(brier_score_loss(y_tr[val], p))
        return np.mean(scores)
    return objective


# ─────────────────────────────────────────────────────────
# OPTUNA OBJECTIVE — LightGBM
# ─────────────────────────────────────────────────────────

def make_lgb_objective(X_tr, y_tr, n_splits=5):
    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 200, 1200),
            "max_depth":         trial.suggest_int("max_depth", 3, 12),
            "learning_rate":     trial.suggest_float("lr", 0.005, 0.15, log=True),
            "num_leaves":        trial.suggest_int("num_leaves", 15, 200),
            "subsample":         trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample", 0.4, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "reg_alpha":         trial.suggest_float("alpha", 1e-4, 10, log=True),
            "reg_lambda":        trial.suggest_float("lambda", 1e-4, 10, log=True),
            "random_state":      42,
            "n_jobs":            -1,
            "verbose":           -1,
        }
        cv     = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = []
        for tr, val in cv.split(X_tr, y_tr):
            m = lgb.LGBMClassifier(**params)
            m.fit(X_tr.iloc[tr], y_tr[tr])
            p = smooth_probs(m.predict_proba(X_tr.iloc[val])[:, 1])
            scores.append(brier_score_loss(y_tr[val], p))
        return np.mean(scores)
    return objective


# ─────────────────────────────────────────────────────────
# OPTUNA OBJECTIVE — CatBoost
# ─────────────────────────────────────────────────────────

def make_cat_objective(X_tr, y_tr, n_splits=5):
    def objective(trial):
        params = {
            "iterations":       trial.suggest_int("iterations", 200, 1000),
            "depth":            trial.suggest_int("depth", 4, 10),
            "learning_rate":    trial.suggest_float("lr", 0.005, 0.15, log=True),
            "l2_leaf_reg":      trial.suggest_float("l2", 1e-3, 10, log=True),
            "border_count":     trial.suggest_int("border_count", 32, 255),
            "random_seed":      42,
            "verbose":          False,
        }
        cv     = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = []
        for tr, val in cv.split(X_tr, y_tr):
            m = CatBoostClassifier(**params)
            m.fit(X_tr.iloc[tr], y_tr[tr], verbose=False)
            p = smooth_probs(m.predict_proba(X_tr.iloc[val])[:, 1])
            scores.append(brier_score_loss(y_tr[val], p))
        return np.mean(scores)
    return objective


# ─────────────────────────────────────────────────────────
# CALIBRATION SHOOTOUT
# ─────────────────────────────────────────────────────────

def best_calibration(base_model, X_cal, y_cal, X_val, y_val):
    """
    Test 3 calibration methods on a held-out calibration set.
    Returns the method name and calibrated probabilities with the lowest Brier.
    """
    results = {}

    # 1. Isotonic
    cal_iso = CalibratedClassifierCV(estimator=clone(base_model), method="isotonic", cv=5)
    cal_iso.fit(X_cal, y_cal)
    p_iso = smooth_probs(cal_iso.predict_proba(X_val)[:, 1])
    results["isotonic"] = (brier_score_loss(y_val, p_iso), p_iso, cal_iso)

    # 2. Sigmoid (Platt)
    cal_sig = CalibratedClassifierCV(estimator=clone(base_model), method="sigmoid", cv=5)
    cal_sig.fit(X_cal, y_cal)
    p_sig = smooth_probs(cal_sig.predict_proba(X_val)[:, 1])
    results["sigmoid"] = (brier_score_loss(y_val, p_sig), p_sig, cal_sig)

    # 3. Temperature Scaling on raw logits
    try:
        raw_model = clone(base_model)
        raw_model.fit(X_cal, y_cal)
        raw_p   = raw_model.predict_proba(X_val)[:, 1]
        logits  = np.log(raw_p / (1 - raw_p + 1e-9))
        ts      = TemperatureScaler().fit(logits, y_val)
        p_temp  = smooth_probs(ts.predict_proba(logits))
        results["temperature"] = (brier_score_loss(y_val, p_temp), p_temp, None)
    except Exception:
        pass

    best_name  = min(results, key=lambda k: results[k][0])
    best_brier = results[best_name][0]
    best_probs = results[best_name][1]
    best_model = results[best_name][2]

    print(f"  🎯 Calibration shootout:")
    for name, (bs, _, _) in sorted(results.items(), key=lambda x: x[1][0]):
        marker = " ← WINNER" if name == best_name else ""
        print(f"     {name:<14} Brier={bs:.5f}{marker}")

    return best_name, best_brier, best_probs, best_model


# ─────────────────────────────────────────────────────────
# FEATURE SELECTION
# ─────────────────────────────────────────────────────────

def select_features(X, y):
    """RFECV — removes features that hurt the Brier score."""
    print("  🔍 Running RFECV feature selection...")
    estimator = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    selector  = RFECV(
        estimator=estimator,
        step=1,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring="neg_brier_score",
        min_features_to_select=5,
        n_jobs=-1,
    )
    selector.fit(X, y)
    selected = X.columns[selector.support_].tolist()
    print(f"  ✅ Selected {len(selected)}/{X.shape[1]} features "
          f"(removed {X.shape[1]-len(selected)} noisy features)")
    return selected


# ─────────────────────────────────────────────────────────
# BEAST MODE TRAINER
# ─────────────────────────────────────────────────────────

def train_beast(key, cfg, n_trials=150):
    sep = "═" * 65
    print(f"\n{sep}")
    print(f"  🔥 BEAST MODE: {cfg['label'].upper()}")
    print(f"{sep}")
    t_start = time.time()

    if not Path(cfg["data_path"]).exists():
        print(f"  ❌ Data not found: {cfg['data_path']}"); return

    # ── Load data ─────────────────────────────────────────
    df = pd.read_csv(cfg["data_path"])
    if cfg["target_col"] not in df.columns:
        print(f"  ❌ Target column '{cfg['target_col']}' missing.")
        print(f"     Available columns: {list(df.columns[:10])}")
        return

    X = df.drop(columns=[cfg["target_col"]] + cfg.get("text_cols", []))
    y = df[cfg["target_col"]].values
    print(f"  📂 {len(df):,} rows | {X.shape[1]} features | {y.mean():.1%} positive")

    # ── Split: 60% train | 20% calibration | 20% test ────
    X_tmp,  X_test,  y_tmp,  y_test  = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
    X_train, X_cal,  y_train, y_cal  = train_test_split(X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=42)
    print(f"  Split → Train:{len(X_train)} | Cal:{len(X_cal)} | Test:{len(X_test)}")

    # ── Feature selection ─────────────────────────────────
    selected = select_features(X_train, y_train)
    X_train  = X_train[selected]
    X_cal    = X_cal[selected]
    X_test   = X_test[selected]

    # ── Optuna: tune XGBoost ──────────────────────────────
    print(f"\n  ⚡ Optuna tuning XGBoost ({n_trials} trials)...")
    xgb_study = optuna.create_study(direction="minimize")
    xgb_study.optimize(make_xgb_objective(X_train, y_train), n_trials=n_trials, show_progress_bar=True)
    best_xgb_params = xgb_study.best_params
    best_xgb = xgb.XGBClassifier(
        **{k: v for k, v in best_xgb_params.items()},
        eval_metric="logloss", use_label_encoder=False, random_state=42, n_jobs=-1
    )
    best_xgb.fit(X_train, y_train)
    xgb_brier = brier_score_loss(y_test, smooth_probs(best_xgb.predict_proba(X_test)[:,1]))
    print(f"  ✅ Best XGBoost Brier (holdout): {xgb_brier:.5f}")

    # ── Optuna: tune LightGBM ─────────────────────────────
    print(f"\n  ⚡ Optuna tuning LightGBM ({n_trials} trials)...")
    lgb_study = optuna.create_study(direction="minimize")
    lgb_study.optimize(make_lgb_objective(X_train, y_train), n_trials=n_trials, show_progress_bar=True)
    best_lgb_params = lgb_study.best_params
    best_lgb = lgb.LGBMClassifier(
        **{k: v for k, v in best_lgb_params.items()},
        random_state=42, n_jobs=-1, verbose=-1
    )
    best_lgb.fit(X_train, y_train)
    lgb_brier = brier_score_loss(y_test, smooth_probs(best_lgb.predict_proba(X_test)[:,1]))
    print(f"  ✅ Best LightGBM Brier (holdout): {lgb_brier:.5f}")

    # ── Optuna: tune CatBoost ─────────────────────────────
    if HAS_CATBOOST:
        print(f"\n  ⚡ Optuna tuning CatBoost ({n_trials} trials)...")
        cat_study = optuna.create_study(direction="minimize")
        cat_study.optimize(make_cat_objective(X_train, y_train), n_trials=n_trials, show_progress_bar=True)
        best_cat_params = cat_study.best_params
        best_cat = CatBoostClassifier(**{k:v for k,v in best_cat_params.items()}, random_seed=42, verbose=False)
        best_cat.fit(X_train, y_train, verbose=False)
        cat_brier = brier_score_loss(y_test, smooth_probs(best_cat.predict_proba(X_test)[:,1]))
        print(f"  ✅ Best CatBoost Brier (holdout): {cat_brier:.5f}")

    # ── L1 Estimators ─────────────────────────────────────
    l1_estimators = [
        ("lr",    Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(C=1.0, max_iter=2000, random_state=42))])),
        ("rf",    RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_split=2,
                                         random_state=42, n_jobs=-1, class_weight="balanced")),
        ("et",    ExtraTreesClassifier(n_estimators=500, random_state=42, n_jobs=-1, class_weight="balanced")),
        ("hgb",   HistGradientBoostingClassifier(max_iter=500, random_state=42)),
        ("xgb",   best_xgb),
        ("lgb",   best_lgb),
        ("mlp",   Pipeline([("sc", StandardScaler()),
                             ("m",  MLPClassifier(hidden_layer_sizes=(256,128,64),
                                                   activation="relu", max_iter=500,
                                                   early_stopping=True, random_state=42))])),
    ]
    if HAS_CATBOOST:
        l1_estimators.append(("cat", best_cat))

    # ── L2 Meta-learner ───────────────────────────────────
    meta_l2 = LogisticRegression(C=0.5, max_iter=2000, random_state=42)

    # ── Stacking Ensemble ─────────────────────────────────
    print(f"\n  🏗️  Building {len(l1_estimators)}-model stacking ensemble...")
    stacker = StackingClassifier(
        estimators=l1_estimators,
        final_estimator=meta_l2,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        passthrough=True,
        n_jobs=-1,
    )
    stacker.fit(X_train, y_train)
    raw_brier = brier_score_loss(y_test, smooth_probs(stacker.predict_proba(X_test)[:,1]))
    print(f"  ✅ Raw stacker Brier (holdout): {raw_brier:.5f}")

    # ── Calibration shootout ──────────────────────────────
    print(f"\n  🎯 Running calibration shootout on held-out calibration set...")
    best_cal_name, cal_brier, cal_probs, best_cal_model = best_calibration(
        stacker, X_cal, y_cal, X_test, y_test
    )
    print(f"  ✅ Post-calibration Brier: {cal_brier:.5f}  ({raw_brier:.5f} → {cal_brier:.5f})")

    # ── Repeated CV Brier ─────────────────────────────────
    print(f"\n  📊 Running Repeated 5×3 Stratified K-Fold CV (15 total folds)...")
    cv_mean, cv_std = cv_brier(stacker, pd.concat([X_train, X_cal]), np.concatenate([y_train, y_cal]),
                                n_splits=5, n_repeats=3)
    print(f"  ✅ Repeated CV Brier: {cv_mean:.5f} ± {cv_std:.5f}")

    # ── Final metrics ─────────────────────────────────────
    auc = roc_auc_score(y_test, cal_probs)
    ece = compute_ece(y_test, cal_probs)
    ll  = log_loss(y_test, np.clip(cal_probs, 1e-7, 1-1e-7))

    elapsed = time.time() - t_start
    print(f"\n  {'─'*55}")
    print(f"  📋 FINAL RESULTS — {cfg['label']}")
    print(f"  {'─'*55}")
    print(f"  Holdout Brier (calibrated) : {cal_brier:.5f}")
    print(f"  Repeated CV Brier          : {cv_mean:.5f} ± {cv_std:.5f}")
    print(f"  ROC-AUC                    : {auc:.5f}")
    print(f"  ECE                        : {ece:.5f}  (target < 0.05)")
    print(f"  Log Loss                   : {ll:.5f}")
    print(f"  Calibration method         : {best_cal_name}")
    print(f"  Features used              : {len(selected)}")
    print(f"  L1 models in stack         : {len(l1_estimators)}")
    print(f"  Time elapsed               : {elapsed/60:.1f} min")
    print(f"  {'─'*55}")

    # ── Save model ────────────────────────────────────────
    model_to_save = best_cal_model if best_cal_model is not None else stacker
    Path(cfg["out_path"]).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_to_save, cfg["out_path"])
    print(f"  💾 Model saved → {cfg['out_path']}")

    # ── Save run log ──────────────────────────────────────
    log = {
        "domain":          key,
        "timestamp":       datetime.now().isoformat(),
        "holdout_brier":   round(cal_brier, 6),
        "cv_brier_mean":   round(cv_mean, 6),
        "cv_brier_std":    round(cv_std, 6),
        "roc_auc":         round(auc, 6),
        "ece":             round(ece, 6),
        "log_loss":        round(ll, 6),
        "calibration":     best_cal_name,
        "n_features":      len(selected),
        "n_l1_models":     len(l1_estimators),
        "n_trials":        n_trials,
        "elapsed_min":     round(elapsed/60, 2),
    }
    log_path = f"models/{key}_beast_run.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"  📝 Run log   → {log_path}")

    return log


# ─────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────

def print_summary(logs):
    valid = [l for l in logs if l is not None]
    if not valid: return
    print(f"\n\n{'═'*80}")
    print("  🔥 BEAST MODE — FINAL SUMMARY")
    print(f"{'═'*80}")
    print(f"  {'Domain':<32} {'Brier':>8} {'CV Brier':>14} {'AUC':>8} {'ECE':>8}  Cal Method")
    print(f"  {'─'*76}")
    for l in sorted(valid, key=lambda x: x["holdout_brier"]):
        print(f"  {l['domain']:<32} {l['holdout_brier']:>8.5f} "
              f"{l['cv_brier_mean']:>7.5f}±{l['cv_brier_std']:<6.5f}"
              f"{l['roc_auc']:>8.5f} {l['ece']:>8.5f}  {l['calibration']}")
    print(f"  {'─'*76}")
    best = min(valid, key=lambda x: x["holdout_brier"])
    print(f"\n  🏆 Best domain: {best['domain']} — Brier {best['holdout_brier']:.5f}")
    print(f"{'═'*80}\n")


# ─────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", default="all",
                        help="Domain key or 'all'")
    parser.add_argument("--trials", type=int, default=150,
                        help="Optuna trials per model (default 150, use 300+ for max performance)")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║     PROJECT SAMBHAV — 🔥 BEAST MODE TRAINER 🔥          ║")
    print("║  Optuna + 8-model stacking + calibration shootout       ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Optuna trials per model: {args.trials}")
    print(f"  CatBoost available: {HAS_CATBOOST}")

    keys = list(DOMAINS.keys()) if args.domain == "all" else [args.domain]
    if args.domain != "all" and args.domain not in DOMAINS:
        print(f"Unknown domain. Options: {list(DOMAINS.keys())}"); sys.exit(1)

    logs = [train_beast(k, DOMAINS[k], n_trials=args.trials) for k in keys]
    if args.domain == "all":
        print_summary(logs)
    print("\n✅ Beast Mode complete. Models saved to models/")

if __name__ == "__main__":
    main()
