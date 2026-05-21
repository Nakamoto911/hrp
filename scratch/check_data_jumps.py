import pandas as pd
import numpy as np

df = pd.read_csv("cache/european_prices.csv", index_col=0, parse_dates=True)

print("Checking for extreme daily returns (>10% or <-10%)...")
for col in df.columns:
    series = df[col].dropna()
    if len(series) < 2:
        continue
    rets = series.pct_change().dropna()
    extreme_mask = (rets > 0.10) | (rets < -0.10)
    n_extreme = extreme_mask.sum()
    if n_extreme > 0:
        print(f"\nAsset: {col} has {n_extreme} extreme daily returns:")
        extreme_idx = rets[extreme_mask].index
        for idx in extreme_idx[:10]:
            p_prev = series.loc[:idx].iloc[-2] if len(series.loc[:idx]) > 1 else np.nan
            p_curr = series.loc[idx]
            ret_val = rets.loc[idx]
            print(f"  {idx.strftime('%Y-%m-%d')}: {ret_val*100:6.2f}% (Price: {p_prev:.4f} -> {p_curr:.4f})")
        if n_extreme > 10:
            print(f"  ... and {n_extreme - 10} more.")
