import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath('.'))

from hrp_engine.data import fetch_data

prices_df = fetch_data('european')
target_date = pd.Timestamp("2012-03-30")
lookback_years = 4

start_date = target_date - pd.DateOffset(years=lookback_years)
window_df = prices_df.loc[start_date:target_date]

print("INSIDE get_lookback_data TRACE FOR 2012-03-30:")
for col in window_df.columns:
    series = window_df[col]
    valid_count = series.notna().sum()
    total_count = len(series)
    ratio = valid_count / total_count if total_count > 0 else 0.0
    last_val = series.iloc[-1]
    is_last_nan = pd.isna(last_val)
    
    passed_ratio = ratio >= 0.8
    passed_last = not is_last_nan
    
    print(f"  {col:<10}: Ratio={ratio:>7.2%} (Passed: {passed_ratio}), LastVal={last_val} (Passed: {passed_last})")
