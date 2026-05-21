import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath('.'))

from hrp_engine.config import StrategyParams
from hrp_engine.data import fetch_data
from hrp_engine.backtest import run_strategy_backtest

pools = ['etf', 'european']
params = StrategyParams(
    lookback_years=4,
    rebalance_frequency='quarterly',
    linkage_method='single',
    drift_threshold=0.015,
    transaction_cost_bps=5.0,
    french_pfu_rate=0.314
)

for pool in pools:
    prices_df = fetch_data(pool)
    hrp_cum, hrp_diag, sp500_cum, sixty_forty_cum, sf_diag = run_strategy_backtest(
        prices_df=prices_df,
        params=params,
        pool_name=pool,
        initial_capital=100000.0
    )
    
    # Calculate before-tax and after-tax returns
    # hrp_cum is after-tax equity. Let's find before-tax equity
    # Wait, the backtest records both! Let's look at the columns of hrp_cum or hrp_diag
    print(f"\n--- {pool.upper()} POOL TAX AND RETURN BREAKDOWN ---")
    print(f"Final After-Tax Value:  {hrp_cum.iloc[-1]:.2f}")
    # Note: run_strategy_backtest returns hrp_equity, which is after-tax. 
    # Let's inspect the keys of hrp_diag to see if there is before_tax_equity or similar.
    for k in sorted(hrp_diag.keys()):
        if isinstance(hrp_diag[k], (int, float)):
            print(f"  {k}: {hrp_diag[k]:.4f}")
