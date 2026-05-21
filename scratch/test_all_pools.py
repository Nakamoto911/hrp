import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath('.'))

from hrp_engine.config import StrategyParams
from hrp_engine.data import fetch_data
from hrp_engine.backtest import run_strategy_backtest
from hrp_engine.reporting import compute_metrics

pools = ['etf', 'mutual_fund', 'european']
params = StrategyParams(
    lookback_years=4,
    rebalance_frequency='quarterly',
    linkage_method='single',
    drift_threshold=0.015,
    transaction_cost_bps=5.0,
    french_pfu_rate=0.314
)

for pool in pools:
    print(f"\n=================== POOL: {pool.upper()} ===================")
    prices_df = fetch_data(pool)
    
    hrp_cum, hrp_diag, sp500_cum, sixty_forty_cum, sf_diag = run_strategy_backtest(
        prices_df=prices_df,
        params=params,
        pool_name=pool,
        initial_capital=100000.0,
        limit_to_least_history=False
    )
    
    metrics_hrp = compute_metrics(hrp_cum, hrp_diag["total_friction_costs_paid"], params.transaction_cost_bps, 100000.0)
    metrics_sf = compute_metrics(sixty_forty_cum, sf_diag["total_friction_costs_paid"], params.transaction_cost_bps, 100000.0)
    metrics_sp = compute_metrics(sp500_cum, 0.0, params.transaction_cost_bps, 100000.0)
    
    print("  HRP Strategy:")
    for k, v in metrics_hrp.items():
        if isinstance(v, float):
            print(f"    {k}: {v*100:.2f}%" if 'return' in k or 'volatility' in k or 'drawdown' in k else f"    {k}: {v:.4f}")
        else:
            print(f"    {k}: {v}")
            
    print("  60/40 Benchmark:")
    print(f"    annualized_return: {metrics_sf['annualized_return']*100:.2f}%")
    print(f"    annualized_volatility: {metrics_sf['annualized_volatility']*100:.2f}%")
    print(f"    sharpe_ratio: {metrics_sf['sharpe_ratio']:.4f}")
    
    print("  S&P 500 Buy & Hold:")
    print(f"    annualized_return: {metrics_sp['annualized_return']*100:.2f}%")
    print(f"    annualized_volatility: {metrics_sp['annualized_volatility']*100:.2f}%")
    print(f"    sharpe_ratio: {metrics_sp['sharpe_ratio']:.4f}")
