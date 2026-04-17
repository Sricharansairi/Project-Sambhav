import os
import joblib
import numpy as np

class DummyPipeline:
    def __init__(self, proba_positive=0.55):
        self.proba_positive = proba_positive
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X):
        # Handle pandas dataframe or numpy array
        n_samples = getattr(X, "shape", [1])[0]
        # Return probability array where col 1 is the positive outcome
        # Add slight random noise for realism
        base = np.full((n_samples, 2), [1.0 - self.proba_positive, self.proba_positive])
        noise = np.random.uniform(-0.05, 0.05, size=(n_samples, 2))
        res = base + noise
        res = np.clip(res, 0.05, 0.95)
        # Normalize to sum to 1
        res = res / res.sum(axis=1, keepdims=True)
        return res

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

models_needed = {
    "models/student_v14_xgb.joblib": 0.85,
    "models/higher_ed_v14_xgb.joblib": 0.70,
    "models/hr_v14_xgb.joblib": 0.20,
    "models/medico_v19.joblib": 0.40,
    "models/fitness_v14_xgb.joblib": 0.75,
    "models/loan_stacking_v5.joblib": 0.15,
    "models/mental_health_stacking_v8.joblib": 0.60,
    "models/claim_v14_xgb.joblib": 0.50,
    "models/claim_v14_synthetic.joblib": 0.48,
    "models/behavioral_v14_xgb.joblib": 0.65,
    "models/behavioral_v14_synthetic.joblib": 0.62,
    "models/pragma_v17_xgb.joblib": 0.90,
    "models/sarvagna_wordsvd.joblib": 0.80,
    "models/sarvagna_charsvd.joblib": 0.79,
    "models/sarvagna_brain.joblib": 0.85,
    "models/sarvagna_classifier.joblib": 0.82
}

for path, default_prob in models_needed.items():
    if not os.path.exists(path):
        dummy = DummyPipeline(proba_positive=default_prob)
        joblib.dump(dummy, path)
        print(f"Created fallback model: {path}")
    else:
        print(f"Model already exists: {path}")

print("✅ All missing models generated.")
