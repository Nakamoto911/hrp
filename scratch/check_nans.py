import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath('.'))

from hrp_engine.data import fetch_data

prices_df = fetch_data('european')
target_date = pd.Timestamp("2015-12-31")
start_date = target_date - pd.DateOffset(years=4)

window_df = prices_df.loc[start_date:target_date]

print(f"Data analysis for window: {start_date.strftime('%Y-%m-%d')} to {target_date.strftime('%Y-%m-%d')}")
print(f"Total trading days in index: {len(window_df)}")
print(f"{'Ticker':<10} | {'Not-NaN Count':<15} | {'NaN Count':<10} | {'Percentage':<10}")
print("-" * 55)

for col in window_df.columns:
    series = window_df[col]
    valid_count = series.notna().sum()
    nan_count = series.isna().sum()
    pct = valid_count / len(series) if len(series) > 0 else 0.0
    print(f"{col:<10} | {valid_count:>13} | {nan_count:>9} | {pct:>9.2%}")
