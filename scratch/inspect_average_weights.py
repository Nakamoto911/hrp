import pandas as pd
import numpy as np
import sys
import os

# Set paths
sys.path.append(os.path.abspath('.'))

from hrp_engine.config import StrategyParams
from hrp_engine.backtest import run_strategy_backtest

# Load cached prices
prices_df = pd.read_csv("cache/european_prices.csv", index_col=0, parse_dates=True)

params = StrategyParams(
    lookback_years=4,
    rebalance_frequency='quarterly',
    linkage_method='single',
    drift_threshold=0.015,
    transaction_cost_bps=5.0,
    french_pfu_rate=0.314
)

hrp_cum, hrp_diag, _, _, _ = run_strategy_backtest(
    prices_df=prices_df,
    params=params,
    pool_name='european',
    limit_to_least_history=False
)

# Extract weight history
weight_history = hrp_diag["weight_history"]
df_weights = pd.DataFrame.from_dict(weight_history, orient='index')

# Calculate average weight for each asset
avg_weights = df_weights.mean().sort_values(ascending=False)
print("Average HRP Weights over the entire backtest:")
for ticker, w in avg_weights.items():
    # Also get first valid date and total return in cache
    first_val = prices_df[ticker].first_valid_index()
    last_val = prices_df[ticker].last_valid_index()
    prices_valid = prices_df[ticker].dropna()
    total_ret = prices_valid.iloc[-1] / prices_valid.iloc[0] - 1 if len(prices_valid) > 0 else 0.0
    
    print(f"  {ticker:<10} | Avg Weight: {w*100:5.2f}% | First Valid: {first_val.strftime('%Y-%m-%d')} | Total Return: {total_ret*100:6.2f}%")
