python -c "
import pandas as pd
import numpy as np
import os

df = pd.read_csv('data/raw/student-mat.csv', sep=';')
print(f'Loaded: {df.shape[0]} rows, {df.shape[1]} columns')

df['pass'] = (df['G3'] >= 10).astype(int)
print(f'Pass rate: {df[\"pass\"].mean():.1%}')

binary_cols = ['school','sex','address','famsize','Pstatus','schoolsup',
               'famsup','paid','activities','nursery','higher','internet','romantic']
for col in binary_cols:
    if col in df.columns:
        df[col] = (df[col] == df[col].unique()[0]).astype(int)

df = df.drop(columns=['G3'], errors='ignore')
df = df.select_dtypes(include=[np.number])
df = df.dropna()

os.makedirs('data/processed', exist_ok=True)
df.to_csv('data/processed/student_final.csv', index=False)
print(f'Processed: {df.shape[0]} rows, {df.shape[1]} columns')
print(f'Saved to data/processed/student_final.csv')
"