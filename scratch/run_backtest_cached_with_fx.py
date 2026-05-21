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

# 1. Load cached prices
prices_df = pd.read_csv("cache/european_prices.csv", index_col=0, parse_dates=True)
start_date = prices_df.index[0]
end_date = prices_df.index[-1]

print(f"Original cached data range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

# 2. Download GBPEUR=X for the same range
print(f"Downloading GBPEUR=X from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...")
fx_df = yf.download('GBPEUR=X', start=start_date, end=end_date, progress=False)

if isinstance(fx_df.columns, pd.MultiIndex):
    if 'Adj Close' in fx_df.columns.levels[0]:
        fx_series = fx_df['Adj Close']
    else:
        fx_series = fx_df['Close']
else:
    fx_series = fx_df

# yfinance might return a dataframe or series
if isinstance(fx_series, pd.DataFrame):
    fx_series = fx_series.iloc[:, 0]

# Reindex and forward fill exchange rates to match prices_df index
fx_series = fx_series.reindex(prices_df.index).ffill().bfill()

# 3. Create a copy and convert IGLT.L to EUR
prices_fx_adjusted = prices_df.copy()
prices_fx_adjusted['IGLT.L'] = prices_df['IGLT.L'] * fx_series

# Setup backtest params (same as user's run)
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
print("Running backtest with original cache + FX-adjusted IGLT.L...")
hrp_cum, hrp_diag, sp500_cum, sixty_forty_cum, sf_diag = run_strategy_backtest(
    prices_df=prices_fx_adjusted,
    params=params_at,
    pool_name='european',
    limit_to_least_history=False
)

# Run before-tax backtest for reporting
hrp_cum_bt, hrp_diag_bt, _, sixty_forty_cum_bt, _ = run_strategy_backtest(
    prices_df=prices_fx_adjusted,
    params=params_bt,
    pool_name='european',
    limit_to_least_history=False
)

# Compute final metrics (using same calculation as backtest.py)
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

# Let's print first and last wealth values
print(f"HRP Cum Final Value: {hrp_cum.iloc[-1] * 100000.0:.2f}")
print(f"S&P 500 Cum Final Value: {sp500_cum.iloc[-1] * 100000.0:.2f}")
print(f"60/40 Cum Final Value: {sixty_forty_cum.iloc[-1] * 100000.0:.2f}")
