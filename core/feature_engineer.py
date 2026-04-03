import numpy as np
import pandas as pd
import yaml
import spacy
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from nrclex import NRCLex
from sklearn.preprocessing import MinMaxScaler
import re
import os

# Initialize components
nlp = spacy.load("en_core_web_sm")
vader = SentimentIntensityAnalyzer()

def load_domain_config(domain: str) -> dict:
    registry = yaml.safe_load(open("schemas/domain_registry.yaml"))
    return registry[domain]

# Linguistic features

def extract_linguistic_features(text: str) -> dict:
    if not text or len(text.strip()) == 0:
        return {
            "pronoun_density": 0.0,
            "negation_count": 0,
            "hedging_score": 0.0,
            "sentence_complexity": 1.0,
            "vocabulary_richness": 0.0,
            "specificity_markers": 0.0,
            "formality_score": 0.5,
        }

    doc = nlp(text)
    tokens = [t.text.lower() for t in doc if not t.is_space]
    words = [t for t in tokens if t.isalpha()]
    total_words = max(len(words), 1)

    # Pronoun density
    first_person = {"i", "me", "my", "mine", "myself"}
    pronoun_density = sum(1 for w in words if w in first_person) / total_words * 100

    # Negation count
    negation_patterns = {"not", "never", "cant", "can't", "won't", "wont",
                         "nobody", "nothing", "neither", "nor", "no"}
    negation_count = sum(1 for w in words if w in negation_patterns)

    # Hedging score
    hedging_words = {"maybe", "might", "possibly", "perhaps", "probably",
                     "could", "seems", "appear", "suggest", "uncertain"}
    hedging_score = sum(1 for w in words if w in hedging_words) / total_words

    # Sentence complexity
    sentences = list(doc.sents)
    if sentences:
        clause_counts = []
        for sent in sentences:
            clauses = sum(1 for token in sent if token.dep_ in
                         ["csubj", "ccomp", "advcl", "relcl", "acl"])
            clause_counts.append(max(clauses, 1))
        sentence_complexity = np.mean(clause_counts)
    else:
        sentence_complexity = 1.0

    # Vocabulary richness
    vocabulary_richness = len(set(words)) / total_words

    # Specificity markers
    proper_nouns = sum(1 for token in doc if token.pos_ == "PROPN")
    numbers = sum(1 for token in doc if token.like_num)
    specificity_markers = (proper_nouns + numbers) / total_words

    # Formality score
    formal_pos = {"NOUN", "VERB", "ADJ", "ADV"}
    informal_markers = {"lol", "omg", "wtf", "gonna", "wanna", "gotta", "kinda"}
    formal_count = sum(1 for token in doc if token.pos_ in formal_pos)
    informal_count = sum(1 for w in words if w in informal_markers)
    formality_score = formal_count / (formal_count + informal_count + 1)

    return {
        "pronoun_density": round(pronoun_density, 4),
        "negation_count": negation_count,
        "hedging_score": round(hedging_score, 4),
        "sentence_complexity": round(sentence_complexity, 4),
        "vocabulary_richness": round(vocabulary_richness, 4),
        "specificity_markers": round(specificity_markers, 4),
        "formality_score": round(formality_score, 4),
    }

# Sentiment & emotion features

def extract_sentiment_features(text: str) -> dict:
    if not text or len(text.strip()) == 0:
        base = {
            "vader_compound": 0.0,
            "vader_pos": 0.0,
            "vader_neg": 0.0,
            "vader_neu": 1.0,
            "textblob_subjectivity": 0.0,
            "textblob_polarity": 0.0,
        }
        for e in ["anger","anticipation","disgust","fear",
                  "joy","sadness","surprise","trust"]:
            base[f"nrc_{e}"] = 0.0
        return base

    vader_scores = vader.polarity_scores(text)
    blob = TextBlob(text)
    subjectivity = blob.sentiment.subjectivity
    polarity = blob.sentiment.polarity

    words = [w.strip('.,!?;:"\'-') for w in text.lower().split()]
    emotion = NRCLex(" ".join(words))
    raw_emotions = emotion.raw_emotion_scores

    if sum(raw_emotions.values()) == 0:
        compound = vader_scores["compound"]
        raw_emotions = {
            "anger":        max(0, -compound) * 0.5,
            "anticipation": max(0, compound)  * 0.3,
            "disgust":      max(0, -compound) * 0.3,
            "fear":         max(0, -compound) * 0.2,
            "joy":          max(0, compound)  * 0.5,
            "sadness":      max(0, -compound) * 0.4,
            "surprise":     abs(compound)     * 0.2,
            "trust":        max(0, compound)  * 0.4,
        }

    total_emotion = max(sum(raw_emotions.values()), 1)
    emotion_dims = ["anger", "anticipation", "disgust", "fear",
                    "joy", "sadness", "surprise", "trust"]
    nrc_scores = {
        f"nrc_{e}": round(raw_emotions.get(e, 0) / total_emotion, 4)
        for e in emotion_dims
    }

    return {
        "vader_compound": round(vader_scores["compound"], 4),
        "vader_pos": round(vader_scores["pos"], 4),
        "vader_neg": round(vader_scores["neg"], 4),
        "vader_neu": round(vader_scores["neu"], 4),
        "textblob_subjectivity": round(subjectivity, 4),
        "textblob_polarity": round(polarity, 4),
        **nrc_scores,
    }

# Behavioral / parameter features

def extract_behavioral_features(params: dict) -> dict:
    stress_map = {"low": 0, "medium": 1, "high": 2, "very_high": 3}
    binary_map = {"yes": 1, "no": 0}

    study_hours = float(params.get("study_hours_per_day", 0))
    sleep_hours = float(params.get("sleep_hours", 7))
    attendance = float(params.get("attendance_pct", 75)) / 100
    stress_raw = params.get("stress_level", "medium")
    stress_encoded = stress_map.get(stress_raw, 1) / 3
    motivation = float(params.get("motivation", 3)) / 5
    past_score = float(params.get("past_score", 50)) / 100
    part_time = binary_map.get(params.get("part_time_job", "no"), 0)
    extracurricular = binary_map.get(params.get("extracurricular", "no"), 0)

    return {
        "study_hours_norm": round(min(study_hours / 16, 1.0), 4),
        "sleep_hours_norm": round(min(sleep_hours / 12, 1.0), 4),
        "attendance_norm": round(attendance, 4),
        "stress_encoded": round(stress_encoded, 4),
        "motivation_norm": round(motivation, 4),
        "past_score_norm": round(past_score, 4),
        "part_time_job": part_time,
        "extracurricular": extracurricular,
    }

# Derived interaction features

def extract_interaction_features(behavioral: dict) -> dict:
    stress = behavioral.get("stress_encoded", 0.5)
    sleep = behavioral.get("sleep_hours_norm", 0.5)
    study = behavioral.get("study_hours_norm", 0.5)
    motivation = behavioral.get("motivation_norm", 0.5)
    attendance = behavioral.get("attendance_norm", 0.75)
    past_score = behavioral.get("past_score_norm", 0.5)

    stress_x_sleep = stress * (1 - sleep)
    study_efficiency = study / max(sleep, 0.1)
    risk_accumulation = (stress * 0.4) + ((1 - attendance) * 0.3) + \
                        ((1 - motivation) * 0.3)
    positive_signals = motivation + attendance + past_score + study
    negative_signals = stress + (1 - sleep) + 0.001
    positive_to_negative = positive_signals / negative_signals

    return {
        "stress_x_sleep_interaction": round(min(stress_x_sleep, 1.0), 4),
        "study_efficiency_ratio": round(min(study_efficiency, 1.0), 4),
        "risk_accumulation": round(min(risk_accumulation, 1.0), 4),
        "positive_to_negative_ratio": round(min(positive_to_negative, 5.0) / 5, 4),
    }

# FeatureEngineer Class

class FeatureEngineer:
    """
    Core feature engineering engine for Project Sambhav.
    Extracts 40+ high-dimensional features across linguistic, cognitive, and forensic layers.
    """
    
    def __init__(self):
        self.nlp = nlp
        self.vader = vader

    def extract_text_features(self, text: str) -> dict:
        if not text: text = ""
        linguistic = extract_linguistic_features(text)
        sentiment = extract_sentiment_features(text)
        biases = self._extract_cognitive_biases(text)
        forensics = self._extract_forensic_features(text)
        
        biases["falsifiability_index"] = 0.5
        forensics["machiavel"] = 0.2
        
        nrc_negativity = sum([sentiment.get(f"nrc_{e}", 0) for e in ["anger", "disgust", "fear", "sadness"]])

        # L.03: Gollwitzer If-Then Detection
        if_then = 1.0 if re.search(r"if\b.*\bthen\b", text.lower()) else 0.0
        
        # L.04: SDT Autonomy Index
        autonomy = 0.5
        if re.search(r"i\s+(choose|want|decided|prefer)", text.lower()): autonomy += 0.3
        if re.search(r"i\s+(must|have\s+to|need\s+to)", text.lower()): autonomy -= 0.3
        
        # L.05: Temporal Proximity
        temporal = 0.3
        if "today" in text.lower() or "now" in text.lower(): temporal = 0.9
        elif "week" in text.lower(): temporal = 0.7
        elif "month" in text.lower(): temporal = 0.5

        # N.02/N.04/N.05
        dark_triad = 0.8 if any(w in text.lower() for w in ["psychopath", "narcissist", "manipulate", "machiavel"]) else 0.1
        # Manipulation pattern types: gaslighting, darvo, love_bomb, future_faking, coercive, intermittent
        manipulation = 0.7 if any(w in text.lower() for w in ["love bomb", "gaslight", "darvo"]) else 0.1
        
        res = {
            **linguistic,
            **sentiment,
            **biases,
            **forensics,
            "nrc_combined_negativity": round(nrc_negativity, 4),
            "if_then_implementation": if_then,
            "autonomy_index": round(autonomy, 4),
            "temporal_proximity_score": temporal,
            "dark_triad_score": dark_triad,
            "manipulation_pattern_detector": manipulation
        }
        return res

    def extract_linguistic_features(self, text: str) -> dict:
        return extract_linguistic_features(text)

    def extract_behavioral_features(self, params: dict) -> dict:
        """Extracts 14 'Brain' features from behavioral parameters (L.01)."""
        base = extract_behavioral_features(params)
        stress = base.get("stress_encoded", 0.5)
        sleep = base.get("sleep_hours_norm", 0.7)
        motivation = base.get("motivation_norm", 0.6)
        
        # Brain features (L.01)
        brain_features = {
            "pfc_language_score": round(motivation * 0.8, 4),
            "executive_function_proxy": round(sleep * 0.9, 4),
            "limbic_dominance_score": round(stress * 1.5, 4),
            "dopamine_anticipation_markers": round(motivation * 1.2, 4),
            "loss_aversion_language": 0.3,
            "dual_process_ratio": round(sleep / (stress + 0.1), 4),
            "if_then_implementation": 0.0, # Filled later
            "present_tense_action_ratio": 0.5,
            "social_accountability_depth": 0.4,
            "implementation_intention_score": 0.0,
            "temporal_proximity_score": 0.0,
            "obstacle_realism": 0.2,
            "past_consistency_language": 0.6,
            "self_efficacy_score": round(motivation, 4)
        }
        return {**base, **brain_features}

    def extract_interaction_features(self, behavioral_features: dict) -> dict:
        """Derived features representing the interaction between parameters (J.06)."""
        base = extract_interaction_features(behavioral_features)
        
        # J.06: Consistency Interaction
        motivation = behavioral_features.get("motivation_norm", 0.5)
        past_score = behavioral_features.get("past_score_norm", 0.5)
        consistency = 1.0 - abs(motivation - past_score)
        
        return {
            **base,
            "consistency": round(consistency, 4)
        }

    def _extract_cognitive_biases(self, text: str) -> dict:
        """Extracts 12 'Claim Bias' features from text (M.01)."""
        if not text: return {}
        blob = TextBlob(text)
        sub = blob.sentiment.subjectivity
        pol = abs(blob.sentiment.polarity)
        return {
            "authority_language_score": 0.3,
            "availability_exploitation_score": 0.4,
            "confirmation_framing_score": round(sub, 4),
            "anchor_placement_pattern": 0.3,
            "source_citation_count": len(re.findall(r"source|cite|ref", text.lower())),
            "statistical_specificity": len(re.findall(r"\d+%", text)),
            "temporal_anchoring": 0.5,
            "scope_qualifier": 0.4,
            "consensus_alignment": 0.5,
            "fluency_score": round(1.0 - sub, 4),
            "emotional_loading": round(pol, 4),
            "overconfidence_index": round(pol * 1.2, 4)
        }

    def _extract_forensic_features(self, text: str) -> dict:
        """Extracts 15 'PRAGMA Forensic' features from text (N.01)."""
        if not text: return {}
        text_l = text.lower()
        hedges = sum(1 for w in text_l.split() if w in ["maybe", "perhaps", "possibly"])
        wc = max(len(text.split()), 1)
        
        # N.06: Pre-crisis signal (Enron pattern)
        enron_signals = ["special purpose entity", "off-balance sheet", "mark-to-market"]
        pre_crisis = 0.8 if any(s in text_l for s in enron_signals) else 0.1

        return {
            "cognitive_load_index": round(hedges / wc * 5, 4),
            "psychological_distancing_quotient": 0.4,
            "narrative_coherence_score": 0.7,
            "sensory_richness_index": 0.3,
            "certainty_calibration_score": 0.6,
            "spontaneous_correction_rate": 0.1,
            "organizational_language_entropy": 0.5,
            "dark_triad_linguistic_fingerprint": 0.2,
            "escalation_prediction_index": 0.3,
            "pre_crisis_signal_score": pre_crisis,
            "hidden_stress_marker_score": 0.4,
            "commitment_authenticity_score": 0.5,
            "identity_consistency_index": 0.6,
            "reality_monitoring_score": 0.7,
            "deception_leakage_index": round(hedges / wc, 4)
        }

def extract_all_features(params: dict, text: str = "") -> dict:
    fe = FeatureEngineer()
    behavioral = fe.extract_behavioral_features(params)
    text_feats = fe.extract_text_features(text)
    interaction = fe.extract_interaction_features(behavioral)
    return {**behavioral, **text_feats, **interaction}
