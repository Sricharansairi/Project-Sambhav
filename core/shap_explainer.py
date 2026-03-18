import shap
import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

DOMAIN_TARGETS = {
    "student": "pass",
    "higher_education": "pass",
    "hr": "attrition",
    "behavioral": "deceptive",
    "claim": "credible",
}

DOMAIN_DATASETS = {
    "student": "data/processed/student_final.csv",
    "higher_education": "data/processed/student_dropout_final.csv",
    "hr": "data/processed/hr_final.csv",
    "behavioral": "data/processed/behavioral_final.csv",
    "claim": "data/processed/claim_final.csv",
}

DOMAIN_MODELS = {
    "student": "models/student_stacking_v2.joblib",
    "higher_education": "models/student_dropout_stacking_v3.joblib",
    "hr": "models/hr_stacking_v2.joblib",
    "behavioral": "models/behavioral_stacking_v2.joblib",
    "claim": "models/claim_stacking_v2.joblib",
}

def load_model(domain: str):
    path = DOMAIN_MODELS.get(domain)
    bundle = joblib.load(path)
    if isinstance(bundle, dict):
        return bundle["model"], bundle.get("scaler")
    return bundle, None

def get_background_data(domain: str, n=50):
    target = DOMAIN_TARGETS[domain]
    df = pd.read_csv(DOMAIN_DATASETS[domain])
    df = df.select_dtypes(include=[np.number])
    df = df.drop(columns=["synthetic", target], errors="ignore").fillna(0)
    return df.sample(min(n, len(df)), random_state=42)

def explain_prediction(domain: str, input_features: dict) -> dict:
    model, scaler = load_model(domain)
    background = get_background_data(domain)

    input_df = pd.DataFrame([input_features])
    for col in background.columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[background.columns]

    if scaler:
        bg_scaled = scaler.transform(background)
        input_scaled = scaler.transform(input_df)
    else:
        bg_scaled = background.values
        input_scaled = input_df.values

    def predict_fn(X):
        return model.predict_proba(X)[:, 1]

    explainer = shap.KernelExplainer(
        predict_fn,
        shap.sample(bg_scaled, 30)
    )
    shap_values = explainer.shap_values(input_scaled, nsamples=100)

    contributions = dict(zip(background.columns, shap_values[0]))
    sorted_contrib = sorted(
        contributions.items(),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    probability = float(model.predict_proba(input_scaled)[0][1])
    positive = [(f, v) for f, v in sorted_contrib if v > 0][:5]
    negative = [(f, v) for f, v in sorted_contrib if v < 0][:5]

    top_feature = sorted_contrib[0]
    direction = "increased" if top_feature[1] > 0 else "decreased"

    summary = (
        f"Probability is {probability:.1%}. "
        f"Most influential factor: '{top_feature[0]}' "
        f"({direction} probability by {abs(top_feature[1]):.1%})."
    )

    positive_text = ", ".join([f"{f} (+{v:.1%})" for f, v in positive[:3]])
    negative_text = ", ".join([f"{f} ({v:.1%})" for f, v in negative[:3]])
    detailed = (
        f"Positive signals: {positive_text}. "
        f"Risk factors: {negative_text}."
    )

    return {
        "domain": domain,
        "probability": round(probability, 4),
        "summary": summary,
        "detailed": detailed,
        "top_positive": [(f, round(v, 4)) for f, v in positive],
        "top_negative": [(f, round(v, 4)) for f, v in negative],
        "all_contributions": {f: round(v, 4) for f, v in sorted_contrib},
        "feature_count": len(sorted_contrib),
    }

def sensitivity_analysis(
    domain: str,
    input_features: dict,
    param: str,
    values: list
) -> dict:
    model, scaler = load_model(domain)
    background = get_background_data(domain)

    results = {}
    for val in values:
        modified = input_features.copy()
        modified[param] = val
        input_df = pd.DataFrame([modified])
        for col in background.columns:
            if col not in input_df.columns:
                input_df[col] = 0
        input_df = input_df[background.columns]
        if scaler:
            input_scaled = scaler.transform(input_df)
        else:
            input_scaled = input_df.values
        prob = float(model.predict_proba(input_scaled)[0][1])
        results[val] = round(prob, 4)

    best_val = max(results, key=results.get)
    prob_range = max(results.values()) - min(results.values())

    return {
        "parameter": param,
        "sensitivity": results,
        "range": round(prob_range, 4),
        "best_value": best_val,
        "insight": (
            f"Changing '{param}' from {values[0]} to {values[-1]} "
            f"shifts probability by {prob_range:.1%}. "
            f"Best value: {best_val} "
            f"(probability: {results[best_val]:.1%})."
        )
    }

def counterfactual_analysis(
    domain: str,
    input_features: dict,
    target_probability: float = 0.75
) -> dict:
    model, scaler = load_model(domain)
    background = get_background_data(domain)

    input_df = pd.DataFrame([input_features])
    for col in background.columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[background.columns]

    if scaler:
        input_scaled = scaler.transform(input_df)
    else:
        input_scaled = input_df.values

    current_prob = float(model.predict_proba(input_scaled)[0][1])
    suggestions = []

    for col in background.columns[:15]:
        col_max = float(background[col].max())
        col_mean = float(background[col].mean())

        for target_val in [col_max, col_mean + (col_max - col_mean) * 0.5]:
            current_val = input_features.get(col, col_mean)
            if abs(target_val - current_val) < 0.01:
                continue

            modified = input_features.copy()
            modified[col] = target_val
            mod_df = pd.DataFrame([modified])
            for c in background.columns:
                if c not in mod_df.columns:
                    mod_df[c] = 0
            mod_df = mod_df[background.columns]

            if scaler:
                mod_scaled = scaler.transform(mod_df)
            else:
                mod_scaled = mod_df.values

            new_prob = float(model.predict_proba(mod_scaled)[0][1])
            gain = new_prob - current_prob

            if gain > 0.01:
                suggestions.append({
                    "feature": col,
                    "current_value": round(current_val, 3),
                    "suggested_value": round(target_val, 3),
                    "probability_gain": round(gain, 4),
                    "new_probability": round(new_prob, 4),
                })

    suggestions.sort(key=lambda x: x["probability_gain"], reverse=True)

    return {
        "current_probability": round(current_prob, 4),
        "target_probability": target_probability,
        "gap": round(target_probability - current_prob, 4),
        "suggestions": suggestions[:5],
        "achievable": any(
            s["new_probability"] >= target_probability
            for s in suggestions
        )
    }

if __name__ == "__main__":
    print("=== Testing SHAP explainer — borderline student ===\n")

    sample = {
        "studytime": 2,
        "failures": 1,
        "schoolsup": 1,
        "famsup": 0,
        "paid": 0,
        "activities": 0,
        "higher": 1,
        "internet": 1,
        "romantic": 1,
        "famrel": 2,
        "freetime": 4,
        "goout": 4,
        "dalc": 3,
        "walc": 4,
        "health": 2,
        "absences": 12,
    }

    print("1. SHAP explanation...")
    result = explain_prediction("student", sample)
    print(f"Probability: {result['probability']:.1%}")
    print(f"Summary: {result['summary']}")
    print(f"Detailed: {result['detailed']}")

    print("\n2. Sensitivity: studytime...")
    sens = sensitivity_analysis("student", sample, "studytime", [1, 2, 3, 4])
    for val, prob in sens["sensitivity"].items():
        bar = "█" * int(prob * 30)
        print(f"  studytime={val}: {prob:.1%} {bar}")
    print(f"  {sens['insight']}")

    print("\n3. Counterfactual: how to reach 75%...")
    cf = counterfactual_analysis("student", sample, 0.75)
    print(f"Current: {cf['current_probability']:.1%}")
    print(f"Target:  {cf['target_probability']:.1%}")
    print(f"Achievable: {cf['achievable']}")
    for s in cf["suggestions"][:3]:
        print(f"  Change '{s['feature']}' {s['current_value']} → {s['suggested_value']} = {s['new_probability']:.1%} (+{s['probability_gain']:.1%})")

    print("\n✅ Phase 6 SHAP explainer complete!")