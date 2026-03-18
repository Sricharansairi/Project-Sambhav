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

nlp = spacy.load("en_core_web_sm")
vader = SentimentIntensityAnalyzer()

def load_domain_config(domain: str) -> dict:
    registry = yaml.safe_load(open("schemas/domain_registry.yaml"))
    return registry["domains"][domain]

# ── Linguistic features ──────────────────────────────────────────────────────

def extract_linguistic_features(text: str) -> dict:
    if not text or len(text.strip()) == 0:
        return _empty_linguistic_features()

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

    # Sentence complexity (avg clauses per sentence)
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

    # Vocabulary richness (type-token ratio)
    vocabulary_richness = len(set(words)) / total_words if total_words > 0 else 0

    # Specificity markers (numbers, dates, proper nouns)
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

def _empty_linguistic_features() -> dict:
    return {
        "pronoun_density": 0.0,
        "negation_count": 0,
        "hedging_score": 0.0,
        "sentence_complexity": 1.0,
        "vocabulary_richness": 0.0,
        "specificity_markers": 0.0,
        "formality_score": 0.5,
    }

# ── Sentiment & emotion features ─────────────────────────────────────────────

def extract_sentiment_features(text: str) -> dict:
    if not text or len(text.strip()) == 0:
        return _empty_sentiment_features()

    # VADER
    vader_scores = vader.polarity_scores(text)

    # TextBlob
    blob = TextBlob(text)
    subjectivity = blob.sentiment.subjectivity
    polarity = blob.sentiment.polarity

    # NRC Lexicon — preprocess for better word matching
    words = [w.strip('.,!?;:"\'-') for w in text.lower().split()]
    expanded_text = " ".join(words)

    emotion = NRCLex(expanded_text)
    raw_emotions = emotion.raw_emotion_scores

    # If NRC finds nothing, fall back to VADER as proxy
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

def _empty_sentiment_features() -> dict:
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

# ── Behavioral / parameter features ──────────────────────────────────────────

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

# ── Derived interaction features ──────────────────────────────────────────────

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

# ── Master feature extractor ──────────────────────────────────────────────────

def extract_all_features(params: dict, text: str = "") -> dict:
    linguistic = extract_linguistic_features(text)
    sentiment = extract_sentiment_features(text)
    behavioral = extract_behavioral_features(params)
    interaction = extract_interaction_features(behavioral)

    return {
        **linguistic,
        **sentiment,
        **behavioral,
        **interaction,
    }

# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample_params = {
        "study_hours_per_day": 5,
        "sleep_hours": 7,
        "attendance_pct": 85,
        "stress_level": "medium",
        "motivation": 4,
        "past_score": 72,
        "part_time_job": "no",
        "extracurricular": "yes",
    }
    sample_text = "I study hard every day and I am confident I will pass my exams."

    features = extract_all_features(sample_params, sample_text)
    print(f"\n✓ Total features extracted: {len(features)}")
    print("\nFeature breakdown:")
    for k, v in features.items():
        print(f"  {k}: {v}")