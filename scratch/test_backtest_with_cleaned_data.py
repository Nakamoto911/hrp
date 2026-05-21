import pandas as pd
import numpy as np
import sys
import os

# Set paths
sys.path.append(os.path.abspath('.'))

from hrp_engine.config import StrategyParams
from hrp_engine.backtest import run_strategy_backtest
from hrp_engine.reporting import compute_metrics

# Load cache
df = pd.read_csv("cache/european_prices.csv", index_col=0, parse_dates=True)

# Clean bad ticks using 3-day rolling median
df_cleaned = df.rolling(window=3, min_periods=1, center=True).median()

params = StrategyParams(
    lookback_years=4,
    rebalance_frequency='quarterly',
    linkage_method='single',
    drift_threshold=0.015,
    transaction_cost_bps=5.0,
    french_pfu_rate=0.314
)

# Run backtest with raw prices
hrp_cum_raw, hrp_diag_raw, sp500_cum, sixty_forty_cum, sf_diag = run_strategy_backtest(
    prices_df=df,
    params=params,
    pool_name='european',
    limit_to_least_history=False
)

# Run backtest with cleaned prices
hrp_cum_cl, hrp_diag_cl, _, _, _ = run_strategy_backtest(
    prices_df=df_cleaned,
    params=params,
    pool_name='european',
    limit_to_least_history=False
)

# Compute metrics
metrics_raw = compute_metrics(
    hrp_cum_raw,
    total_tc=hrp_diag_raw["total_friction_costs_paid"],
    bps=params.transaction_cost_bps,
    initial_capital=100000.0
)

metrics_cl = compute_metrics(
    hrp_cum_cl,
    total_tc=hrp_diag_cl["total_friction_costs_paid"],
    bps=params.transaction_cost_bps,
    initial_capital=100000.0
)

print("RAW DATA BACKTEST RESULTS:")
for k, v in metrics_raw.items():
    if isinstance(v, float):
        print(f"  {k:<35}: {v:.4f}")

print("\nCLEANED DATA BACKTEST RESULTS:")
for k, v in metrics_cl.items():
    if isinstance(v, float):
        print(f"  {k:<35}: {v:.4f}")
