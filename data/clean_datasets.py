import pandas as pd
import numpy as np
import os

os.makedirs("data/processed", exist_ok=True)

# ── 1. STUDENT ───────────────────────────────────────────────
def clean_student():
    print("=== Cleaning Student Dataset ===")
    df = pd.read_csv("data/raw/student-mat.csv", sep=";")

    # Standardise column names to snake_case
    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]

    # Create binary target
    df["pass"] = (df["g3"] >= 10).astype(int)
    df = df.drop(columns=["g3"])

    # Fill missing values
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates()
    print(f"Duplicates removed: {before - len(df)}")

    # Encode binary categoricals
    binary_map = {}
    for col in df.select_dtypes(include=["object"]).columns:
        unique = df[col].unique()
        if len(unique) == 2:
            binary_map[col] = {unique[0]: 0, unique[1]: 1}
            df[col] = df[col].map(binary_map[col])

    # Drop remaining non-numeric
    df = df.select_dtypes(include=[np.number])

    df.to_csv("data/processed/student_clean.csv", index=False)
    print(f"Shape: {df.shape} | Pass rate: {df['pass'].mean():.1%}")
    print(f"Saved: data/processed/student_clean.csv\n")
    return df

# ── 2. HR ────────────────────────────────────────────────────
def clean_hr():
    print("=== Cleaning HR Dataset ===")
    df = pd.read_csv("data/raw/hr/ibm_hr.csv")

    # Standardise column names
    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]

    # Binary target
    df["attrition"] = (df["attrition"] == "Yes").astype(int)

    # Drop constant columns
    df = df.drop(columns=["employeecount", "over18", "standardhours"], errors="ignore")

    # Fill missing
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates()
    print(f"Duplicates removed: {before - len(df)}")

    # Encode categoricals
    for col in df.select_dtypes(include=["object"]).columns:
        unique = df[col].unique()
        if len(unique) == 2:
            df[col] = (df[col] == unique[0]).astype(int)
        else:
            df[col] = pd.Categorical(df[col]).codes

    df = df.select_dtypes(include=[np.number])

    df.to_csv("data/processed/hr_clean.csv", index=False)
    print(f"Shape: {df.shape} | Attrition rate: {df['attrition'].mean():.1%}")
    print(f"Saved: data/processed/hr_clean.csv\n")
    return df

# ── 3. CLAIM ─────────────────────────────────────────────────
def clean_claim():
    print("=== Cleaning Claim Dataset ===")
    cols = ["id", "label", "statement", "subject", "speaker",
            "job", "state", "party", "barely_true", "false",
            "half_true", "mostly_true", "pants_fire", "context"]

    train = pd.read_csv("data/raw/claim/liar_train.tsv", sep="\t", header=None, names=cols)
    test  = pd.read_csv("data/raw/claim/liar_test.tsv",  sep="\t", header=None, names=cols)
    valid = pd.read_csv("data/raw/claim/liar_valid.tsv", sep="\t", header=None, names=cols)
    df = pd.concat([train, test, valid], ignore_index=True)

    # Binary target: true/mostly-true = 1, others = 0
    true_labels = ["true", "mostly-true"]
    df["credible"] = df["label"].isin(true_labels).astype(int)

    # Fill missing count columns with 0
    count_cols = ["barely_true", "false", "half_true", "mostly_true", "pants_fire"]
    df[count_cols] = df[count_cols].fillna(0)

    # Fill text columns
    text_cols = ["statement", "subject", "speaker", "job", "state", "party", "context"]
    for col in text_cols:
        df[col] = df[col].fillna("unknown")

    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates(subset=["statement"])
    print(f"Duplicates removed: {before - len(df)}")

    df.to_csv("data/processed/claim_clean.csv", index=False)
    print(f"Shape: {df.shape} | Credible rate: {df['credible'].mean():.1%}")
    print(f"Saved: data/processed/claim_clean.csv\n")
    return df

# ── 4. BEHAVIORAL ─────────────────────────────────────────────
def clean_behavioral():
    print("=== Cleaning Behavioral Dataset ===")
    df = pd.read_csv("data/raw/behavioral/deceptive_opinions.csv")

    # Standardise column names
    df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]

    # Fill missing
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].fillna("unknown")

    # Remove duplicates
    before = len(df)
    df = df.drop_duplicates()
    print(f"Duplicates removed: {before - len(df)}")

    # Encode categoricals
    for col in ["polarity", "hotel", "source"]:
        if col in df.columns:
            df[col] = pd.Categorical(df[col]).codes

    df.to_csv("data/processed/behavioral_clean.csv", index=False)
    print(f"Shape: {df.shape} | Deceptive rate: {df['deceptive'].mean():.1%}")
    print(f"Saved: data/processed/behavioral_clean.csv\n")
    return df

# ── RUN ALL ───────────────────────────────────────────────────
if __name__ == "__main__":
    clean_student()
    clean_hr()
    clean_claim()
    clean_behavioral()
    print("=== Phase 2 Stage 2.1 COMPLETE ===")
    