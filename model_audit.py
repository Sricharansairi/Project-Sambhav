"""
PROJECT SAMBHAV — Model Accuracy Audit
Run: python model_audit.py
     python model_audit.py --domain student
     python model_audit.py --save-plots
"""

import os, sys, argparse, warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from sklearn.metrics import brier_score_loss, roc_auc_score, accuracy_score, log_loss, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.calibration import calibration_curve

warnings.filterwarnings("ignore")

DOMAINS = {
    "student": {
        "label":          "Student Performance (v2)",
        "model_path":     "models/student_stacking_v2.joblib",
        "data_path":      "data/processed/student_final.csv",
        "target_col":     "target",
        "brier_floor":    0.15,
    },
    "student_uci": {
        "label":          "Student UCI (v3)",
        "model_path":     "models/student_uci_stacking_v3.joblib",
        "data_path":      "data/processed/student_uci_final.csv",
        "target_col":     "target",
        "brier_floor":    0.15,
    },
    "dropout": {
        "label":          "Higher Education Dropout (v3)",
        "model_path":     "models/student_dropout_stacking_v3.joblib",
        "data_path":      "data/processed/dropout_final.csv",
        "target_col":     "target",
        "brier_floor":    0.15,
    },
    "hr": {
        "label":          "HR Attrition (v2)",
        "model_path":     "models/hr_stacking_v2.joblib",
        "data_path":      "data/processed/hr_final.csv",
        "target_col":     "target",
        "brier_floor":    0.15,
    },
    "behavioral": {
        "label":          "Behavioral / Deception (v2)",
        "model_path":     "models/behavioral_stacking_v2.joblib",
        "data_path":      "data/processed/behavioral_final.csv",
        "target_col":     "target",
        "brier_floor":    0.05,
    },
    "claim": {
        "label":          "Claim Credibility (v3)",
        "model_path":     "models/claim_stacking_v3.joblib",
        "data_path":      "data/processed/claim_final.csv",
        "target_col":     "target",
        "brier_floor":    0.20,
    },
    "disease": {
        "label":          "Disease Risk (v1)",
        "model_path":     "models/disease_stacking_v1.joblib",
        "data_path":      "data/processed/disease_final.csv",
        "target_col":     "target",
        "brier_floor":    0.12,
    },
    "mental_health": {
        "label":          "Mental Health Risk (v1)",
        "model_path":     "models/mental_health_stacking_v1.joblib",
        "data_path":      "data/processed/mental_health_final.csv",
        "target_col":     "target",
        "brier_floor":    0.15,
    },
    "loan": {
        "label":          "Loan / Credit Risk (v1)",
        "model_path":     "models/loan_stacking_v1.joblib",
        "data_path":      "data/processed/loan_final.csv",
        "target_col":     "target",
        "brier_floor":    0.14,
    },
    "fitness": {
        "label":          "Diet & Fitness (v1)",
        "model_path":     "models/fitness_stacking_v1.joblib",
        "data_path":      "data/processed/fitness_final.csv",
        "target_col":     "target",
        "brier_floor":    0.16,
    },
}

MISSING_MODELS = {
    "exam": "models/exam_stacking_v1.joblib",
}

def compute_ece(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        ece += mask.sum() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return ece / len(y_true)

def grade(brier, floor):
    if brier < floor * 0.6:   return "🟢 EXCELLENT"
    elif brier < floor:        return "✅ PASS"
    elif brier < floor * 1.2:  return "⚠️  BORDERLINE"
    else:                      return "❌ FAIL"

def plot_reliability(y_true, y_prob, label, save_path):
    fig = plt.figure(figsize=(10, 4))
    gs  = gridspec.GridSpec(1, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0])
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=10)
    ax1.plot([0,1],[0,1],"k--",lw=1,label="Perfect")
    ax1.plot(mean_pred, frac_pos,"o-",color="#6C63FF",lw=2,ms=6,label="Model")
    ax1.set_title(f"{label}\nReliability Diagram"); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)
    ax2 = fig.add_subplot(gs[1])
    ax2.hist(y_prob[y_true==0],bins=20,alpha=0.6,color="#E74C3C",label="Negative")
    ax2.hist(y_prob[y_true==1],bins=20,alpha=0.6,color="#00D4AA",label="Positive")
    ax2.set_title("Probability Distribution"); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  📊 Plot saved → {save_path}")

def audit_domain(key, cfg, save_plots=False):
    sep = "─" * 62
    print(f"\n{sep}\n  DOMAIN: {cfg['label'].upper()}\n{sep}")

    if not Path(cfg["model_path"]).exists():
        print(f"  ❌ Model NOT FOUND: {cfg['model_path']}")
        return None
    if not Path(cfg["data_path"]).exists():
        print(f"  ❌ Data NOT FOUND:  {cfg['data_path']}")
        print(f"     → Check your data/processed/ filenames and update DOMAINS dict.")
        return None

    model = joblib.load(cfg["model_path"])
    df    = pd.read_csv(cfg["data_path"])

    if cfg["target_col"] not in df.columns:
        print(f"  ❌ Column '{cfg['target_col']}' not in dataset. Columns: {list(df.columns[:8])}")
        return None

    X = df.drop(columns=[cfg["target_col"]])
    y = df[cfg["target_col"]].values
    print(f"  ✅ Loaded | {len(df):,} rows | {X.shape[1]} features | {y.mean():.1%} positive class")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    holdout_brier = brier_score_loss(y_test, y_prob)
    holdout_auc   = roc_auc_score(y_test, y_prob)
    holdout_acc   = accuracy_score(y_test, y_pred)
    holdout_ll    = log_loss(y_test, y_prob)
    ece           = compute_ece(y_test, y_prob)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_briers, cv_aucs = [], []
    for tr_idx, val_idx in cv.split(X, y):
        try:
            model.fit(X.iloc[tr_idx], y[tr_idx])
            p = model.predict_proba(X.iloc[val_idx])[:, 1]
            cv_briers.append(brier_score_loss(y[val_idx], p))
            cv_aucs.append(roc_auc_score(y[val_idx], p))
        except Exception as e:
            print(f"  ⚠️  CV fold skipped: {e}")

    cv_brier_mean = np.mean(cv_briers) if cv_briers else float("nan")
    cv_brier_std  = np.std(cv_briers)  if cv_briers else float("nan")
    cv_auc_mean   = np.mean(cv_aucs)   if cv_aucs   else float("nan")

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
    verdict = grade(cv_brier_mean, cfg["brier_floor"])

    print(f"\n  {'METRIC':<30} {'VALUE':>10}   {'TARGET':>8}")
    print(f"  {'─'*52}")
    print(f"  {'Holdout Brier':<30} {holdout_brier:>10.4f}   {'<'+str(cfg['brier_floor']):>8}")
    print(f"  {'CV Brier (5-fold mean ± std)':<30} {cv_brier_mean:>7.4f} ± {cv_brier_std:.4f}")
    print(f"  {'ROC-AUC (holdout)':<30} {holdout_auc:>10.4f}   {'>0.75':>8}")
    print(f"  {'CV ROC-AUC (mean)':<30} {cv_auc_mean:>10.4f}   {'>0.75':>8}")
    print(f"  {'Accuracy (0.5 threshold)':<30} {holdout_acc:>10.4f}")
    print(f"  {'Log Loss':<30} {holdout_ll:>10.4f}")
    print(f"  {'ECE (calibration error)':<30} {ece:>10.4f}   {'<0.05':>8}")
    print(f"  {'─'*52}")
    print(f"  Confusion Matrix  →  TP:{tp}  FP:{fp}  FN:{fn}  TN:{tn}")
    print(f"  VERDICT: {verdict}")

    if ece > 0.05:
        print(f"\n  ⚠️  ECE too high — re-run Phase 5 CalibratedClassifierCV for this domain.")
    if holdout_auc < 0.75:
        print(f"  ⚠️  AUC too low — do NOT wire to LLM cross-validation yet.")

    if save_plots:
        os.makedirs("audit_plots", exist_ok=True)
        model_fresh = joblib.load(cfg["model_path"])
        plot_reliability(y_test, model_fresh.predict_proba(X_test)[:,1],
                         cfg["label"], f"audit_plots/{key}_reliability.png")

    return {
        "domain":        key,
        "label":         cfg["label"],
        "holdout_brier": round(holdout_brier, 4),
        "cv_brier_mean": round(cv_brier_mean, 4),
        "cv_brier_std":  round(cv_brier_std, 4),
        "roc_auc":       round(holdout_auc, 4),
        "cv_auc":        round(cv_auc_mean, 4),
        "accuracy":      round(holdout_acc, 4),
        "ece":           round(ece, 4),
        "verdict":       verdict,
        "phase7_ready":  cv_brier_mean < cfg["brier_floor"] and holdout_auc >= 0.75,
    }

def print_summary(results):
    valid = [r for r in results if r is not None]
    if not valid:
        print("\n  No results to summarise."); return

    print(f"\n\n{'═'*92}")
    print("  SAMBHAV — FULL MODEL AUDIT SUMMARY")
    print(f"{'═'*92}")
    print(f"  {'Domain':<34} {'Holdout Brier':>13} {'CV Brier':>14} {'AUC':>7} {'ECE':>7} {'Phase 7?':>10}  Grade")
    print(f"  {'─'*88}")
    all_ready = True
    for r in valid:
        ready = "✅ YES" if r["phase7_ready"] else "❌ NO"
        if not r["phase7_ready"]: all_ready = False
        print(f"  {r['label']:<34} {r['holdout_brier']:>13.4f} "
              f"{r['cv_brier_mean']:>7.4f}±{r['cv_brier_std']:<6.4f}"
              f"{r['roc_auc']:>7.4f} {r['ece']:>7.4f} {ready:>10}  {r['verdict']}")
    print(f"  {'─'*88}")

    if MISSING_MODELS:
        print(f"\n  ⚠️  MISSING (not yet trained):")
        for name, path in MISSING_MODELS.items():
            print(f"     • {name:<12} → {path}")

    if all_ready:
        print("\n  🟢 ALL AUDITED MODELS PASS — safe to begin Phase 7.")
    else:
        print(f"\n  ❌ Fix failing domains before Phase 7 LLM integration.")
    print(f"\n{'═'*92}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain",     default="all")
    parser.add_argument("--save-plots", action="store_true")
    args = parser.parse_args()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║       PROJECT SAMBHAV — PRE-PHASE 7 MODEL AUDIT         ║")
    print("╚══════════════════════════════════════════════════════════╝")

    keys = list(DOMAINS.keys()) if args.domain == "all" else [args.domain]
    if args.domain != "all" and args.domain not in DOMAINS:
        print(f"Unknown domain. Choose from: {list(DOMAINS.keys())}"); sys.exit(1)

    results = [audit_domain(k, DOMAINS[k], args.save_plots) for k in keys]
    if args.domain == "all":
        print_summary(results)
    print("Audit complete.")

if __name__ == "__main__":
    main()
