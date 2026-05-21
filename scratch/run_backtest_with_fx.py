import yfinance as yf
import pandas as pd
import numpy as np
import sys
import os

# Set paths
sys.path.append(os.path.abspath('.'))

from hrp_engine.config import StrategyParams
from hrp_engine.backtest import run_strategy_backtest
from hrp_engine.reporting import compute_metrics, generate_markdown_report

# 1. Fetch raw data
tickers = [
    'SXR8.DE', 'EXX5.DE', 'EXS1.DE', 'SXRT.DE', 'IS3N.DE',
    'IUSM.DE', 'IS0L.DE', 'XJSE.DE', 'IGLT.L',
    'IBCQ.DE', 'IHYG.MI',
    '4GLD.DE', 'CRUD.MI', 'COPA.MI', 'AIGP.MI',
    '5MVW.DE', '36BZ.DE', 'QDVB.DE', 'IUS3.DE',
    'BTCE.DE'
]

start_date = '2000-01-01'
end_date = '2026-05-20'

print("Downloading historical data and GBPEUR=X...")
df_raw = yf.download(tickers + ['GBPEUR=X'], start=start_date, end=end_date, progress=False)

if isinstance(df_raw.columns, pd.MultiIndex):
    if 'Adj Close' in df_raw.columns.levels[0]:
        df = df_raw['Adj Close']
    else:
        df = df_raw['Close']
else:
    df = df_raw

# Keep only the columns we need
df_fx = df['GBPEUR=X'].ffill().bfill()
df_prices = df[tickers].copy()

# Convert IGLT.L from GBP to EUR
df_prices['IGLT.L'] = df_prices['IGLT.L'] * df_fx

# Clean prices
prices_clean = df_prices.ffill().bfill()

# Setup backtest params
params_at = StrategyParams(
    lookback_years=4,
    rebalance_frequency='quarterly',
    linkage_method='single',
    drift_threshold=0.015,
    transaction_cost_bps=5.0,
    french_pfu_rate=0.314
)

params_bt = StrategyParams(
    lookback_years=4,
    rebalance_frequency='quarterly',
    linkage_method='single',
    drift_threshold=0.015,
    transaction_cost_bps=5.0,
    french_pfu_rate=0.0
)

# Run backtest
print("Running backtest with FX-adjusted prices...")
hrp_cum, hrp_diag, sp500_cum, sixty_forty_cum, sf_diag = run_strategy_backtest(
    prices_df=prices_clean,
    params=params_at,
    pool_name='european',
    limit_to_least_history=False
)

# Run before-tax backtest for reporting
hrp_cum_bt, hrp_diag_bt, _, sixty_forty_cum_bt, _ = run_strategy_backtest(
    prices_df=prices_clean,
    params=params_bt,
    pool_name='european',
    limit_to_least_history=False
)

# Compute final metrics
hrp_metrics = compute_metrics(
    hrp_cum, 
    total_tc=hrp_diag["total_friction_costs_paid"], 
    bps=params_at.transaction_cost_bps, 
    initial_capital=100000.0
)

sp500_metrics = compute_metrics(sp500_cum, total_tc=0.0, bps=0.0, initial_capital=100000.0)

sixty_forty_metrics = compute_metrics(
    sixty_forty_cum, 
    total_tc=sf_diag["total_friction_costs_paid"], 
    bps=params_at.transaction_cost_bps, 
    initial_capital=100000.0
)

report = generate_markdown_report(hrp_metrics, sp500_metrics, sixty_forty_metrics, hrp_diag)
print("\n" + report + "\n")
