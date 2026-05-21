import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath('.'))

# Load cached prices
prices_df = pd.read_csv("cache/european_prices.csv", index_col=0, parse_dates=True)

print(f"Dataset range: {prices_df.index[0].strftime('%Y-%m-%d')} to {prices_df.index[-1].strftime('%Y-%m-%d')}")
print(f"Total days: {len(prices_df)}")

summary = []
for col in prices_df.columns:
    series = prices_df[col].dropna()
    n_valid = len(series)
    n_nan = len(prices_df) - n_valid
    
    if n_valid < 2:
        continue
        
    # Calculate daily returns
    rets = series.pct_change().dropna()
    ann_ret = (series.iloc[-1] / series.iloc[0]) ** (252.0 / len(series)) - 1.0 if len(series) > 0 else 0.0
    ann_vol = rets.std() * np.sqrt(252.0)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    
    summary.append({
        "Ticker": col,
        "Valid Days": n_valid,
        "NaN Days": n_nan,
        "Start Price": f"{series.iloc[0]:.4f}",
        "End Price": f"{series.iloc[-1]:.4f}",
        "Ann. Return": f"{ann_ret*100:.2f}%",
        "Ann. Vol": f"{ann_vol*100:.2f}%",
        "Sharpe": f"{sharpe:.2f}"
    })

df_sum = pd.DataFrame(summary)
print(df_sum.to_string(index=False))
