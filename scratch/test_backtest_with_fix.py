import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath('.'))

from hrp_engine.config import StrategyParams
from hrp_engine.data import fetch_data
import hrp_engine.backtest
from hrp_engine.reporting import compute_metrics

# Define the fixed get_lookback_data function
def get_lookback_data_fixed(prices_df: pd.DataFrame, rebalance_date: pd.Timestamp, lookback_years: int):
    start_date = rebalance_date - pd.DateOffset(years=lookback_years)
    window_df = prices_df.loc[start_date:rebalance_date]
    
    active_assets = []
    for col in window_df.columns:
        series = window_df[col]
        valid_count = series.notna().sum()
        total_count = len(series)
        
        # Calculate ratio of non-NaN values
        if total_count > 0 and (valid_count / total_count) >= 0.8:
            first_valid = prices_df[col].first_valid_index()
            if first_valid is not None and rebalance_date >= first_valid:
                ffilled_price = prices_df[col].loc[:rebalance_date].ffill().iloc[-1]
                if not pd.isna(ffilled_price):
                    active_assets.append(col)
                    
    sliced_prices = window_df[active_assets].copy()
    sliced_prices = sliced_prices.ffill().bfill()
    returns_df = sliced_prices.pct_change().dropna(how='all')
    return returns_df, active_assets

# Patch the backtest module directly
hrp_engine.backtest.get_lookback_data = get_lookback_data_fixed

prices_df = fetch_data('european')
params = StrategyParams(
    lookback_years=4,
    rebalance_frequency='quarterly',
    linkage_method='single',
    drift_threshold=0.015,
    transaction_cost_bps=5.0,
    french_pfu_rate=0.314
)

hrp_cum, hrp_diag, sp500_cum, sixty_forty_cum, sf_diag = hrp_engine.backtest.run_strategy_backtest(
    prices_df=prices_df,
    params=params,
    pool_name='european',
    initial_capital=100000.0
)

metrics = compute_metrics(hrp_cum, hrp_diag["total_friction_costs_paid"], params.transaction_cost_bps, 100000.0)

print("\n--- RESULTS WITH CORRECTED PATCH (NO HOLIDAY SHOCKS) ---")
print(f"HRP After-Tax Return: {metrics['annualized_return']*100:.2f}%")
print(f"HRP Volatility:       {metrics['annualized_volatility']*100:.2f}%")
print(f"HRP Sharpe Ratio:     {metrics['sharpe_ratio']:.4f}")
print(f"HRP Sortino Ratio:    {metrics['sortino_ratio']:.4f}")
print(f"HRP Max Drawdown:     {metrics['max_drawdown']*100:.2f}%")
print(f"HRP Annualized Churn: {metrics['annualized_turnover']*100:.2f}%")
print(f"Total Friction Paid:  {hrp_diag['total_friction_costs_paid']:.2f} EUR")
print(f"Total Taxes Paid:     {hrp_diag['total_pfu_taxes_paid']:.2f} EUR")
print(f"Total Rebalance Events: {hrp_diag['total_rebalance_events']}")
