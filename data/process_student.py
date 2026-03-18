import pandas as pd
import numpy as np
import os

def process_student_data():
    df = pd.read_csv("data/raw/student-mat.csv", sep=";")
    print(f"✓ Loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # Create target column — pass if final grade G3 >= 10
    df["pass"] = (df["G3"] >= 10).astype(int)
    print(f"✓ Class distribution:\n{df['pass'].value_counts(normalize=True).round(3)}")

    # Select and rename useful columns
    df = df.rename(columns={
        "studytime": "study_hours_per_day",
        "absences": "absences",
        "health": "health",
        "famrel": "family_relationship",
        "freetime": "free_time",
        "goout": "go_out",
        "Dalc": "workday_alcohol",
        "Walc": "weekend_alcohol",
        "G1": "grade_period_1",
        "G2": "grade_period_2",
    })

    # Encode binary categorical columns
    binary_cols = ["school", "sex", "address", "famsize",
                   "Pstatus", "schoolsup", "famsup", "paid",
                   "activities", "nursery", "higher", "internet", "romantic"]
    for col in binary_cols:
        if col in df.columns:
            df[col] = (df[col] == df[col].unique()[0]).astype(int)

    # Normalise numeric columns to [0, 1]
    numeric_cols = ["study_hours_per_day", "absences", "health",
                    "family_relationship", "free_time", "go_out",
                    "workday_alcohol", "weekend_alcohol",
                    "grade_period_1", "grade_period_2", "age",
                    "Medu", "Fedu", "traveltime", "failures"]
    for col in numeric_cols:
        if col in df.columns:
            col_min = df[col].min()
            col_max = df[col].max()
            if col_max > col_min:
                df[col] = (df[col] - col_min) / (col_max - col_min)

    # Drop original grade columns and any remaining objects
    df = df.drop(columns=["G3"], errors="ignore")
    df = df.select_dtypes(include=[np.number])
    df = df.dropna()

    os.makedirs("data/processed", exist_ok=True)
    df.to_csv("data/processed/student_final.csv", index=False)

    print(f"✓ Processed: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"✓ Saved to data/processed/student_final.csv")
    print(f"✓ Features: {[c for c in df.columns if c != 'pass']}")
    return df

if __name__ == "__main__":
    process_student_data()
