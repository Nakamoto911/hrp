import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath('.'))

from hrp_engine.config import StrategyParams
from hrp_engine.data import fetch_data
from hrp_engine.denoiser import denoise_covariance
from hrp_engine.hrp import optimize_hrp
from hrp_engine.backtest import get_rebalance_dates

prices_df = fetch_data('european')
params = StrategyParams(
    lookback_years=4,
    rebalance_frequency='quarterly',
    linkage_method='single',
    drift_threshold=0.015,
    transaction_cost_bps=5.0,
    french_pfu_rate=0.314
)

prices_clean = prices_df.ffill().bfill()
all_rebalance_dates = get_rebalance_dates(prices_clean.index, params.rebalance_frequency)
start_date = prices_clean.index[0]
valid_rebalance_dates = [
    d for d in all_rebalance_dates 
    if d >= start_date + pd.DateOffset(years=params.lookback_years)
]

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
            # Instead of checking raw prices_df on the holiday-affected rebalance_date,
            # we check if the asset has been launched (first valid date exists)
            # and if the forward-filled price at the rebalance date is valid.
            first_valid = prices_df[col].first_valid_index()
            if first_valid is not None and rebalance_date >= first_valid:
                # Double check that we have a forward-filled price at the rebalance date
                ffilled_price = prices_df[col].loc[:rebalance_date].ffill().iloc[-1]
                if not pd.isna(ffilled_price):
                    active_assets.append(col)
                    
    sliced_prices = window_df[active_assets].copy()
    sliced_prices = sliced_prices.ffill().bfill()
    returns_df = sliced_prices.pct_change().dropna(how='all')
    return returns_df, active_assets

print("REBALANCE DATES WEIGHTS TRACE WITH FIXED GET_LOOKBACK_DATA:")
for T_k in valid_rebalance_dates:
    lookback_returns, active_assets = get_lookback_data_fixed(prices_df, T_k, params.lookback_years)
    cov_emp = lookback_returns.cov().values
    n_obs = len(lookback_returns)
    cov_denoised = denoise_covariance(cov_emp, n_obs)
    cov_denoised_df = pd.DataFrame(cov_denoised, index=active_assets, columns=active_assets)
    w_star = optimize_hrp(cov_denoised_df, params.linkage_method)
    
    max_w = w_star.max()
    max_asset = w_star.idxmax()
    if max_w > 0.6:
        print(f"Date: {T_k.strftime('%Y-%m-%d')} | Max Weight: {max_w:.2%} on {max_asset}")
        non_zero = w_star[w_star > 0.01].sort_values(ascending=False)
        print("  Non-zero weights:")
        for t, val in non_zero.items():
            print(f"    {t}: {val:.2%}")
