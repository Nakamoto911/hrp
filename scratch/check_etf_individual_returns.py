import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath('.'))

from hrp_engine.data import fetch_data

prices_df = fetch_data('etf')
# The ETF backtest runs from 2012-01-02 (after 4 years of lookback from 2008-01-02) to 2026-05-20
prices_sliced = prices_df.loc["2012-01-02":]

print("INDIVIDUAL ASSET PERFORMANCE IN ETF POOL (2012-01-02 to 2026-05-20):")
print(f"{'Ticker':<10} | {'Annualized Return':<18} | {'Annualized Vol':<15} | {'Sharpe Ratio':<12} | {'Start Date':<10}")
print("-" * 70)

for col in prices_sliced.columns:
    series = prices_sliced[col].dropna()
    if series.empty:
        print(f"{col:<10} | EMPTY")
        continue
    
    start_val = series.iloc[0]
    end_val = series.iloc[-1]
    n_years = (series.index[-1] - series.index[0]).days / 365.25
    
    ann_return = (end_val / start_val) ** (1.0 / n_years) - 1.0
    daily_returns = series.pct_change().dropna()
    ann_vol = daily_returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0
    
    print(f"{col:<10} | {ann_return:>17.2%} | {ann_vol:>14.2%} | {sharpe:>12.4f} | {series.index[0].strftime('%Y-%m-%d')}")
