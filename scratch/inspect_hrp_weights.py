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

# Run backtest without FX adjustment first to get weight history
params_at = StrategyParams(
    lookback_years=4,
    rebalance_frequency='quarterly',
    linkage_method='single',
    drift_threshold=0.015,
    transaction_cost_bps=5.0,
    french_pfu_rate=0.314
)

hrp_cum, hrp_diag, _, _, _ = run_strategy_backtest(
    prices_df=prices_df,
    params=params_at,
    pool_name='european',
    limit_to_least_history=False
)

# Look at weights at the last rebalance date
weight_history = hrp_diag["weight_history"]
last_date = sorted(list(weight_history.keys()))[-1]
weights = pd.Series(weight_history[last_date])
print(f"Weights on {last_date}:")
print(weights.sort_values(ascending=False).to_string())
print(f"Sum of weights: {weights.sum():.6f}")
