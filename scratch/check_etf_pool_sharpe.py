import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.abspath('.'))

from hrp_engine.config import StrategyParams
from hrp_engine.backtest import run_strategy_backtest
from hrp_engine.reporting import compute_metrics

# Load cached prices for ETF pool
# If not cached, it will download
prices_df = pd.read_csv("cache/etf_prices.csv", index_col=0, parse_dates=True)

params = StrategyParams(
    lookback_years=4,
    rebalance_frequency='quarterly',
    linkage_method='single',
    drift_threshold=0.015,
    transaction_cost_bps=5.0,
    french_pfu_rate=0.314
)

hrp_cum, hrp_diag, sp500_cum, sixty_forty_cum, sf_diag = run_strategy_backtest(
    prices_df=prices_df,
    params=params,
    pool_name='etf',
    limit_to_least_history=False
)

metrics_hrp = compute_metrics(hrp_cum, hrp_diag["total_friction_costs_paid"], params.transaction_cost_bps, 100000.0)
metrics_sf = compute_metrics(sixty_forty_cum, sf_diag["total_friction_costs_paid"], params.transaction_cost_bps, 100000.0)
metrics_sp = compute_metrics(sp500_cum, 0.0, params.transaction_cost_bps, 100000.0)

print("ETF POOL PERFORMANCE SUMMARY:")
print(f"HRP:   Return: {metrics_hrp['annualized_return']*100:.2f}% | Vol: {metrics_hrp['annualized_volatility']*100:.2f}% | Sharpe: {metrics_hrp['sharpe_ratio']:.2f} | MDD: {metrics_hrp['max_drawdown']*100:.2f}%")
print(f"60/40: Return: {metrics_sf['annualized_return']*100:.2f}% | Vol: {metrics_sf['annualized_volatility']*100:.2f}% | Sharpe: {metrics_sf['sharpe_ratio']:.2f} | MDD: {metrics_sf['max_drawdown']*100:.2f}%")
print(f"S&P:   Return: {metrics_sp['annualized_return']*100:.2f}% | Vol: {metrics_sp['annualized_volatility']*100:.2f}% | Sharpe: {metrics_sp['sharpe_ratio']:.2f} | MDD: {metrics_sp['max_drawdown']*100:.2f}%")
