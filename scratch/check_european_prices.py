import pandas as pd
import numpy as np

# Read cached european prices
df = pd.read_csv("cache/european_prices.csv", index_col=0, parse_dates=True)
print("Data shape:", df.shape)
print("Columns:", list(df.columns))
print("\nFirst valid values:")
for col in df.columns:
    first_valid = df[col].first_valid_index()
    if first_valid is not None:
        val = df.loc[first_valid, col]
        print(f"  {col}: {first_valid.strftime('%Y-%m-%d')} -> {val:.4f}")
    else:
        print(f"  {col}: No valid value")

print("\nLast valid values:")
for col in df.columns:
    last_valid = df[col].last_valid_index()
    if last_valid is not None:
        val = df.loc[last_valid, col]
        print(f"  {col}: {last_valid.strftime('%Y-%m-%d')} -> {val:.4f}")
    else:
        print(f"  {col}: No valid value")
