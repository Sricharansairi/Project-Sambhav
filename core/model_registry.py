"""
core/model_registry.py
Loads and serves all 11 domain models for inference.
Handles standard, blended, and sarvagna routing.
"""

import os
import yaml
import joblib
import pickle
import numpy as np
from typing import Optional

# ── Constants ────────────────────────────────────────────────────
REGISTRY_PATH = "schemas/domain_registry.yaml"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SARVAGNA_ROUTING = {
    "career":       ["job", "career", "work", "profession", "promotion", "salary",
                     "business", "startup", "interview", "employment", "office"],
    "creativity":   ["creative", "art", "music", "write", "design", "paint",
                     "poetry", "film", "photography", "craft", "novel"],
    "education":    ["study", "exam", "degree", "college", "university", "course",
                     "learn", "school", "marks", "grade", "subject", "pass", "fail"],
    "family":       ["family", "parent", "child", "marriage", "divorce", "sibling",
                     "mother", "father", "brother", "sister", "relationship with"],
    "finance":      ["money", "finance", "investment", "savings", "debt", "loan",
                     "budget", "income", "profit", "loss", "stock", "crypto"],
    "health":       ["health", "disease", "illness", "medicine", "hospital", "doctor",
                     "diet", "weight", "fitness", "exercise", "recover", "surgery"],
    "location":     ["move", "relocate", "city", "country", "travel", "migrate",
                     "abroad", "visa", "settle", "place", "neighborhood"],
    "relationships":["relationship", "love", "partner", "dating", "friend", "trust",
                     "breakup", "marry", "propose", "bond", "connection"],
    "social_impact":["society", "community", "impact", "volunteer", "ngo", "cause",
                     "change", "awareness", "movement", "protest", "reform"],
    "spirituality": ["spiritual", "god", "faith", "religion", "meditation", "peace",
                     "purpose", "meaning", "soul", "prayer", "universe", "karma"],
}

BLEND_CONFIG = {
    "behavioral": {"weights": [0.6, 0.4]},
    "claim":      {"weights": [0.6, 0.4]},
}


# ── Registry Loader ──────────────────────────────────────────────
class ModelRegistry:
    def __init__(self):
        self.config = self._load_config()
        self.models = {}
        self._load_all()

    def _load_config(self) -> dict:
        with open(REGISTRY_PATH) as f:
            return yaml.safe_load(f)

    def _abs(self, path: Optional[str]) -> Optional[str]:
        """Convert relative path to absolute."""
        if path is None:
            return None
        return os.path.join(PROJECT_ROOT, path)

    def _load_joblib(self, path: Optional[str]):
        if path is None:
            return None
        abs_path = self._abs(path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Model file not found: {abs_path}")
        return joblib.load(abs_path)

    def _load_pkl(self, path: Optional[str]):
        if path is None:
            return None
        abs_path = self._abs(path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Model file not found: {abs_path}")
        with open(abs_path, "rb") as f:
            return pickle.load(f)

    def _load_all(self):
        for domain, config in self.config.items():

            # ── Sarvagna: 10 sub-models ──────────────────────────
            if config.get("routing_mode") == "keyword":
                sub_models = {}
                base = config.get("model_base_path", "./")
                for sub_domain, filename in config["models"].items():
                    path = os.path.join(PROJECT_ROOT, base, filename)
                    with open(path, "rb") as f:
                        data = pickle.load(f)
                        sub_models[sub_domain] = data.get("model", data) if isinstance(data, dict) else data
                self.models[domain] = {
                    "type": "sarvagna",
                    "sub_models": sub_models,
                    "config": config,
                }
                print(f"  [OK] {domain} — SARVAGNA ({len(sub_models)} sub-models)")

            # ── Blended: XGBoost + LightGBM ─────────────────────
            elif config.get("blend_mode"):
                ext_a = config["model_a_path"].split(".")[-1]
                ext_b = config["model_b_path"].split(".")[-1]

                model_a = (self._load_joblib(config["model_a_path"])
                           if ext_a == "joblib"
                           else self._load_pkl(config["model_a_path"]))
                model_b = (self._load_joblib(config["model_b_path"])
                           if ext_b == "joblib"
                           else self._load_pkl(config["model_b_path"]))

                self.models[domain] = {
                    "type": "blended",
                    "model_a": model_a,
                    "model_b": model_b,
                    "scaler": self._load_joblib(config.get("model_a_scaler")),
                    "imputer": self._load_joblib(config.get("model_a_imputer")),
                    "iso": self._load_joblib(config.get("model_a_iso")),
                    "weights": config["blend_ratio"],
                    "config": config,
                }
                print(f"  [OK] {domain} — BLENDED {config['blend_ratio']}")

            # ── Standard: single XGBoost ─────────────────────────
            else:
                ext = config["model_path"].split(".")[-1]
                model = (self._load_joblib(config["model_path"])
                         if ext == "joblib"
                         else self._load_pkl(config["model_path"]))

                self.models[domain] = {
                    "type": "standard",
                    "model": model,
                    "scaler": self._load_joblib(config.get("scaler_path")),
                    "imputer": self._load_joblib(config.get("imputer_path")),
                    "iso": self._load_joblib(config.get("iso_path")),
                    "config": config,
                }
                print(f"  [OK] {domain} — STANDARD Brier {config['brier_score']}")

    # ── Prediction Methods ───────────────────────────────────────

    def predict(self, domain: str, features: np.ndarray) -> float:
        """
        Main prediction entry point.
        Returns calibrated probability float between 0 and 1.
        """
        if domain not in self.models:
            raise ValueError(f"Domain '{domain}' not found in registry.")

        entry = self.models[domain]

        if entry["type"] == "standard":
            return self._predict_standard(entry, features)

        elif entry["type"] == "blended":
            return self._predict_blended(entry, features)

        elif entry["type"] == "sarvagna":
            raise ValueError(
                "Use predict_sarvagna(question) for sarvagna domain."
            )

    def _predict_standard(self, entry: dict, features: np.ndarray) -> float:
        """Standard single-model prediction with preprocessing."""
        X = features.copy().reshape(1, -1)

        if entry["imputer"] is not None:
            expected_n = entry["imputer"].n_features_in_
            if X.shape[1] < expected_n:
                X = np.pad(X, ((0, 0), (0, expected_n - X.shape[1])))
            elif X.shape[1] > expected_n:
                X = X[:, :expected_n]
            X = entry["imputer"].transform(X)

        if entry["scaler"] is not None:
            expected_n = entry["scaler"].n_features_in_
            if X.shape[1] < expected_n:
                X = np.pad(X, ((0, 0), (0, expected_n - X.shape[1])))
            elif X.shape[1] > expected_n:
                X = X[:, :expected_n]
            X = entry["scaler"].transform(X)

        raw_prob = entry["model"].predict_proba(X)[0][1]

        if entry["iso"] is not None:
            raw_prob = float(entry["iso"].transform([raw_prob])[0])

        # Hard cap at 0.97 — never show 100% confidence
        return min(round(float(raw_prob), 4), 0.97)

    def _predict_blended(self, entry: dict, features: np.ndarray) -> float:
        """60/40 blended prediction — XGBoost real + LightGBM synthetic."""
        X = features.copy().reshape(1, -1)

        if entry["imputer"] is not None:
            expected_n = entry["imputer"].n_features_in_
            if X.shape[1] < expected_n:
                X = np.pad(X, ((0, 0), (0, expected_n - X.shape[1])))
            elif X.shape[1] > expected_n:
                X = X[:, :expected_n]
            X = entry["imputer"].transform(X)

        if entry["scaler"] is not None:
            expected_n = entry["scaler"].n_features_in_
            if X.shape[1] < expected_n:
                X = np.pad(X, ((0, 0), (0, expected_n - X.shape[1])))
            elif X.shape[1] > expected_n:
                X = X[:, :expected_n]
            X = entry["scaler"].transform(X)

        # XGBoost prediction
        prob_a = entry["model_a"].predict_proba(X)[0][1]
        if entry["iso"] is not None:
            prob_a = float(entry["iso"].transform([prob_a])[0])

        # LightGBM prediction — no preprocessing needed
        try:
            prob_b = entry["model_b"].predict_proba(X)[0][1]
        except Exception:
            # LightGBM Booster uses predict() not predict_proba()
            prob_b = float(entry["model_b"].predict(X)[0])
            prob_b = max(0.0, min(1.0, prob_b))

        w_a, w_b = entry["weights"]
        blended = (w_a * prob_a) + (w_b * prob_b)

        return min(round(float(blended), 4), 0.97)

    def predict_sarvagna(self, question: str) -> dict:
        """
        Routes question to correct sub-model via keyword matching.
        Returns dict with domain, probability, and matched keywords.
        """
        entry = self.models["sarvagna"]
        question_lower = question.lower()

        # Count keyword matches per domain
        scores = {d: 0 for d in SARVAGNA_ROUTING}
        matched_keywords = {}

        for domain, keywords in SARVAGNA_ROUTING.items():
            hits = [kw for kw in keywords if kw in question_lower]
            scores[domain] = len(hits)
            if hits:
                matched_keywords[domain] = hits

        # Pick highest scoring domain — default to career
        best_domain = max(scores, key=scores.get)
        if scores[best_domain] == 0:
            best_domain = entry["config"]["routing_default"]

        # Run prediction
        model = entry["sub_models"][best_domain]
        features = self._extract_sarvagna_features(question)

        try:
            prob = float(model.predict_proba([features])[0][1])
        except Exception:
            try:
                res = model.predict([features])[0]
                prob = float(res[0]) if hasattr(res, "__len__") else float(res)
            except Exception:
                import xgboost as xgb
                import numpy as np
                dmat = xgb.DMatrix(np.array([features]))
                prob = float(model.predict(dmat)[0])
            prob = max(0.0, min(1.0, prob))

        return {
            "routed_domain": best_domain,
            "probability": min(round(prob, 4), 0.97),
            "matched_keywords": matched_keywords.get(best_domain, []),
            "all_scores": scores,
        }

    def _extract_sarvagna_features(self, question: str) -> list:
        """
        Extracts 100 linguistic features from question text.
        Matches the feature array Sarvagna was trained on.
        """
        words = question.lower().split()
        word_count = max(len(words), 1)
        char_count = len(question)

        # Certainty keywords
        certainty_words = ["will", "shall", "definitely", "certain", "sure",
                           "confident", "absolutely", "must", "going to"]
        uncertainty_words = ["maybe", "might", "possibly", "perhaps", "could",
                             "unsure", "doubt", "wonder", "hope", "wish"]
        negation_words = ["not", "never", "no", "neither", "nor", "cannot",
                          "won't", "don't", "didn't", "isn't"]
        positive_words = ["success", "achieve", "win", "good", "great", "best",
                          "better", "improve", "grow", "thrive"]
        negative_words = ["fail", "lose", "bad", "worst", "problem", "issue",
                          "struggle", "difficult", "hard", "impossible"]

        certainty_score  = sum(1 for w in words if w in certainty_words) / word_count
        uncertainty_score= sum(1 for w in words if w in uncertainty_words) / word_count
        negation_score   = sum(1 for w in words if w in negation_words) / word_count
        positive_score   = sum(1 for w in words if w in positive_words) / word_count
        negative_score   = sum(1 for w in words if w in negative_words) / word_count

        # Base feature vector — 100 floats
        features = [
            word_count / 100.0,
            char_count / 500.0,
            certainty_score,
            uncertainty_score,
            negation_score,
            positive_score,
            negative_score,
            len(set(words)) / word_count,           # vocabulary richness
            question.count("?") / word_count,
            question.count("!") / word_count,
            sum(1 for c in question if c.isupper()) / max(char_count, 1),
            int(question.strip().endswith("?")),
            int(any(w in words for w in ["i", "my", "me", "myself"])),
            int(any(w in words for w in ["we", "our", "us"])),
            int(any(w in words for w in ["will", "going to", "shall"])),
            certainty_score - uncertainty_score,    # net certainty
            positive_score - negative_score,        # net sentiment
        ]

        # Pad to exactly 100 features
        features += [0.0] * (100 - len(features))
        return features[:100]

    # ── Utility Methods ──────────────────────────────────────────

    def get_config(self, domain: str) -> dict:
        """Return YAML config for a domain."""
        if domain not in self.config:
            raise ValueError(f"Domain '{domain}' not in registry.")
        return self.config[domain]

    def get_disclaimer(self, domain: str) -> Optional[str]:
        """Return disclaimer string or None."""
        return self.config.get(domain, {}).get("disclaimer")

    def list_domains(self) -> list:
        """Return all available domain keys."""
        return list(self.models.keys())

    def is_do_not_retrain(self, domain: str) -> bool:
        """Safety check before any retraining."""
        return self.config.get(domain, {}).get("do_not_retrain", False)


# ── Singleton instance ───────────────────────────────────────────
registry = ModelRegistry()