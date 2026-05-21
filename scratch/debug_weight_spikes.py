import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath('.'))

from hrp_engine.config import StrategyParams
from hrp_engine.data import fetch_data, get_lookback_data
from hrp_engine.denoiser import denoise_covariance
from hrp_engine.hrp import optimize_hrp
from hrp_engine.backtest import get_rebalance_dates

# We will test the old configuration first (to see if the user's chart was due to the old tickers or data errors)
# The old European pool had 'IGLT.L', 'EXX5.DE', 'EXS1.DE', '5MVW.DE', '36BZ.DE', 'QDVB.DE', 'IUS3.DE'
old_tickers = [
    'SXR8.DE', 'EXX5.DE', 'EXS1.DE', 'SXRT.DE', 'IS3N.DE', 'IUSM.DE', 'IS0L.DE',
    'XJSE.DE', 'IGLT.L', 'IBCQ.DE', 'IHYG.MI', '4GLD.DE', 'CRUD.MI', 'COPA.MI',
    'AIGP.MI', '5MVW.DE', '36BZ.DE', 'QDVB.DE', 'IUS3.DE', 'BTCE.DE'
]

# Fetch data using current pool and verify what the weights look like over time
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

print("REBALANCE DATES WEIGHTS TRACE:")
for T_k in valid_rebalance_dates:
    lookback_returns, active_assets = get_lookback_data(prices_df, T_k, params.lookback_years)
    cov_emp = lookback_returns.cov().values
    n_obs = len(lookback_returns)
    cov_denoised = denoise_covariance(cov_emp, n_obs)
    cov_denoised_df = pd.DataFrame(cov_denoised, index=active_assets, columns=active_assets)
    w_star = optimize_hrp(cov_denoised_df, params.linkage_method)
    
    # Check if there is any asset with weight > 0.8
    max_w = w_star.max()
    max_asset = w_star.idxmax()
    if max_w > 0.6:
        print(f"Date: {T_k.strftime('%Y-%m-%d')} | Max Weight: {max_w:.2%} on {max_asset}")
        # Print other non-zero weights
        non_zero = w_star[w_star > 0.01].sort_values(ascending=False)
        print("  Non-zero weights:")
        for t, val in non_zero.items():
            print(f"    {t}: {val:.2%}")
