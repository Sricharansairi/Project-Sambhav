import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
import os

os.makedirs("data/processed", exist_ok=True)

def apply_smote(filepath, target_col, domain_name, threshold=3.0):
    print(f"=== SMOTE: {domain_name} ===")
    df = pd.read_csv(filepath)

    # Select only numeric columns
    df = df.select_dtypes(include=[np.number])

    X = df.drop(columns=[target_col])
    y = df[target_col]

    counts = y.value_counts()
    ratio = counts.max() / counts.min()
    print(f"Before SMOTE: {counts.to_dict()} | Ratio: {ratio:.2f}:1")

    if ratio > threshold:
        X_res, y_res = SMOTE(random_state=42).fit_resample(X, y)
        df_res = pd.DataFrame(X_res, columns=X.columns)
        df_res[target_col] = y_res
        counts_after = y_res.value_counts()
        print(f"After SMOTE:  {counts_after.to_dict()} | Ratio: 1.00:1")
        print(f"Rows added: {len(df_res) - len(df)}")
    else:
        df_res = df
        print(f"Skipped — imbalance {ratio:.2f}:1 is below threshold {threshold}:1")

    out_path = filepath.replace("_clean.csv", "_balanced.csv")
    df_res.to_csv(out_path, index=False)
    print(f"Saved: {out_path}\n")
    return df_res

if __name__ == "__main__":
    # Student — mild imbalance, apply anyway
    apply_smote("data/processed/student_clean.csv", "pass", "Student", threshold=1.5)

    # HR — significant imbalance, must apply
    apply_smote("data/processed/hr_clean.csv", "attrition", "HR", threshold=1.5)

    # Claim — check after cleaning
    apply_smote("data/processed/claim_clean.csv", "credible", "Claim", threshold=1.5)

    # Behavioral — already balanced, will skip
    apply_smote("data/processed/behavioral_clean.csv", "deceptive", "Behavioral", threshold=1.5)

    print("=== Phase 2 Stage 2.2 COMPLETE ===")