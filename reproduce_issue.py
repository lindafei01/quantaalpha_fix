
import pandas as pd
import numpy as np
import os
from pathlib import Path

# Path to the experiment directory
exp_dir = Path("/nfsdata-117/quantaalpha/data/results/workspace_exp_20260225_154717/00b987de0f3f477c854dd957fa4339c8")
parquet_path = exp_dir / "test_combined_factors.parquet"

# Create a dummy dataframe with MultiIndex columns and index
dates = pd.date_range("2020-01-01", "2020-01-10")
instruments = [f"SH{i:06d}" for i in range(10)]
index = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
columns = [f"factor_{i}" for i in range(5)]
df = pd.DataFrame(np.random.randn(len(index), len(columns)), index=index, columns=columns)

# Add MultiIndex columns like in runner.py
new_columns = pd.MultiIndex.from_product([["feature"], df.columns])
df.columns = new_columns

print("DataFrame info:")
print(df.info())

try:
    print(f"Saving to {parquet_path}...")
    df.to_parquet(parquet_path, engine="pyarrow")
    print("Success!")
    
    # Try reading it back
    print("Reading back...")
    df_read = pd.read_parquet(parquet_path)
    print("Read success!")
    print(df_read.info())
except Exception as e:
    print(f"Error: {e}")
