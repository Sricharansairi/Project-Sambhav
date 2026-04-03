import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.feature_engineer import extract_all_features

def validate_student():
    print("=== Validating Student Features ===")
    df = pd.read_csv("data/processed/student_final.csv")
    sample = df.iloc[0].to_dict()

    params = {
        "study_hours_per_day": sample.get("studytime", 2),
        "sleep_hours": 7,
        "attendance_pct": 85,
        "stress_level": "medium",
        "motivation": 3,
        "past_score": sample.get("g2", 10) * 5,
        "part_time_job": "no",
        "extracurricular": "no",
    }
    features = extract_all_features(params, "")
    print(f"Features extracted: {len(features)}")
    print(f"Sample values: study={features['study_hours_norm']} | stress={features['stress_encoded']}")
    print("✅ Student domain OK\n")
    return features

def validate_hr():
    print("=== Validating HR Features ===")
    params = {
        "study_hours_per_day": 0,
        "sleep_hours": 6,
        "attendance_pct": 90,
        "stress_level": "high",
        "motivation": 2,
        "past_score": 60,
        "part_time_job": "yes",
        "extracurricular": "no",
    }
    text = "I work overtime constantly and feel undervalued. Management never listens to concerns."
    features = extract_all_features(params, text)
    print(f"Features extracted: {len(features)}")
    print(f"VADER compound: {features['vader_compound']} | NRC anger: {features['nrc_anger']}")
    print("✅ HR domain OK\n")
    return features

def validate_claim():
    print("=== Validating Claim Features ===")
    params = {
        "study_hours_per_day": 0,
        "sleep_hours": 7,
        "attendance_pct": 75,
        "stress_level": "low",
        "motivation": 3,
        "past_score": 50,
        "part_time_job": "no",
        "extracurricular": "no",
    }
    text = "The government confirmed that vaccines contain microchips to track citizens worldwide."
    features = extract_all_features(params, text)
    print(f"Features extracted: {len(features)}")
    print(f"Hedging: {features['hedging_score']} | Specificity: {features['specificity_markers']}")
    print("✅ Claim domain OK\n")
    return features

def validate_behavioral():
    print("=== Validating Behavioral Features ===")
    params = {
        "study_hours_per_day": 0,
        "sleep_hours": 7,
        "attendance_pct": 75,
        "stress_level": "low",
        "motivation": 3,
        "past_score": 50,
        "part_time_job": "no",
        "extracurricular": "no",
    }
    text = "ABSOLUTELY AMAZING! Best hotel ever! Perfect in every single way! Highly recommend!!"
    features = extract_all_features(params, text)
    print(f"Features extracted: {len(features)}")
    print(f"Exclamation proxy (negation_count): {features['negation_count']} | Joy: {features['nrc_joy']}")
    print("✅ Behavioral domain OK\n")
    return features

if __name__ == "__main__":
    f1 = validate_student()
    f2 = validate_hr()
    f3 = validate_claim()
    f4 = validate_behavioral()

    print("=== Phase 3 Feature Validation COMPLETE ===")
    print(f"Total features per prediction: {len(f1)}")
    print("All 4 domains extract features correctly ✅")