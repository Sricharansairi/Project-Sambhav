import pandas as pd
import numpy as np
import os

os.makedirs("data/processed", exist_ok=True)

def merge_student():
    print("=== Merging Student Data ===")
    real = pd.read_csv("data/processed/student_balanced.csv")
    synth = pd.read_csv("data/synthetic/student_synthetic.csv")

    # Align columns — keep only columns in real dataset
    common_cols = [c for c in real.columns if c in synth.columns]
    synth = synth[common_cols]

    df = pd.concat([real, synth], ignore_index=True)
    df = df.drop_duplicates()
    df.to_csv("data/processed/student_final.csv", index=False)
    print(f"Real: {len(real)} | Synthetic: {len(synth)} | Final: {len(df)}")
    print(f"Pass rate: {df['pass'].mean():.1%}")
    print(f"Saved: data/processed/student_final.csv\n")

def merge_hr():
    print("=== Merging HR Data ===")
    real = pd.read_csv("data/processed/hr_balanced.csv")
    synth = pd.read_csv("data/synthetic/hr_synthetic.csv")

    common_cols = [c for c in real.columns if c in synth.columns]
    synth = synth[common_cols]

    df = pd.concat([real, synth], ignore_index=True)
    df = df.drop_duplicates()
    df.to_csv("data/processed/hr_final.csv", index=False)
    print(f"Real: {len(real)} | Synthetic: {len(synth)} | Final: {len(df)}")
    print(f"Attrition rate: {df['attrition'].mean():.1%}")
    print(f"Saved: data/processed/hr_final.csv\n")

def merge_behavioral():
    print("=== Merging Behavioral Data ===")
    real = pd.read_csv("data/processed/behavioral_balanced.csv")
    synth = pd.read_csv("data/synthetic/behavioral_synthetic.csv")

    common_cols = [c for c in real.columns if c in synth.columns]
    synth = synth[common_cols]

    df = pd.concat([real, synth], ignore_index=True)
    df = df.drop_duplicates()
    df.to_csv("data/processed/behavioral_final.csv", index=False)
    print(f"Real: {len(real)} | Synthetic: {len(synth)} | Final: {len(df)}")
    print(f"Deceptive rate: {df['deceptive'].mean():.1%}")
    print(f"Saved: data/processed/behavioral_final.csv\n")

def merge_claim():
    print("=== Merging Claim Data ===")
    # Claim uses only real balanced data — text domain, no numeric synthetic
    df = pd.read_csv("data/processed/claim_balanced.csv")
    df.to_csv("data/processed/claim_final.csv", index=False)
    print(f"Final: {len(df)} rows")
    print(f"Credible rate: {df['credible'].mean():.1%}")
    print(f"Saved: data/processed/claim_final.csv\n")

if __name__ == "__main__":
    merge_student()
    merge_hr()
    merge_behavioral()
    merge_claim()

    print("=== ALL FINAL DATASETS READY ===")
    print("\nFinal dataset summary:")
    for domain, target in [
        ("student", "pass"),
        ("hr", "attrition"),
        ("behavioral", "deceptive"),
        ("claim", "credible")
    ]:
        df = pd.read_csv(f"data/processed/{domain}_final.csv")
        print(f"  {domain}: {len(df)} rows | target={target} | rate={df[target].mean():.1%}")