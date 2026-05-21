import pandas as pd
import numpy as np
import sys
import os

# Set paths
sys.path.append(os.path.abspath('.'))

from hrp_engine.config import StrategyParams
from hrp_engine.backtest import run_strategy_backtest
from hrp_engine.reporting import compute_metrics

# Load cached prices
prices_df = pd.read_csv("cache/european_prices.csv", index_col=0, parse_dates=True)

linkages = ['single', 'complete', 'ward']
lookbacks = [1, 2, 3, 4, 5]
frequencies = ['monthly', 'quarterly', 'yearly']

results = []

print("Running parameter sensitivity analysis on European pool...")
for freq in frequencies:
    for lb in lookbacks:
        for link in linkages:
            params = StrategyParams(
                lookback_years=lb,
                rebalance_frequency=freq,
                linkage_method=link,
                drift_threshold=0.015,
                transaction_cost_bps=5.0,
                french_pfu_rate=0.314
            )
            try:
                hrp_cum, hrp_diag, _, _, _ = run_strategy_backtest(
                    prices_df=prices_df,
                    params=params,
                    pool_name='european',
                    limit_to_least_history=False
                )
                metrics = compute_metrics(
                    hrp_cum,
                    total_tc=hrp_diag["total_friction_costs_paid"],
                    bps=params.transaction_cost_bps,
                    initial_capital=100000.0
                )
                ann_ret = metrics["annualized_return"] * 100
                mdd = metrics["max_drawdown"] * 100
                sharpe = metrics["sharpe_ratio"]
                results.append({
                    "Freq": freq,
                    "Lookback": lb,
                    "Linkage": link,
                    "Ann. Return": f"{ann_ret:.2f}%",
                    "Max Drawdown": f"{mdd:.2f}%",
                    "Sharpe": f"{sharpe:.2f}"
                })
            except Exception as e:
                # Some configurations might fail if history is too short for lookback
                pass

df_res = pd.DataFrame(results)
print(df_res.to_string(index=False))
