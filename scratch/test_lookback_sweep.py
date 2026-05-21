import pandas as pd
import sys
import os

# Set paths
sys.path.append(os.path.abspath('.'))

from hrp_engine.config import StrategyParams
from app import run_lookback_years_sweep

def test_sweep():
    print("Loading prices...")
    prices_df = pd.read_csv("cache/european_prices.csv", index_col=0, parse_dates=True)
    
    params = StrategyParams(
        lookback_years=4,
        rebalance_frequency="quarterly",
        linkage_method="ward",
        drift_threshold=0.015,
        transaction_cost_bps=5.0,
        french_pfu_rate=0.314
    )
    
    print("Running lookback window years sweep...")
    df = run_lookback_years_sweep(prices_df, params, "european", 100000.0)
    print(df.to_string())

if __name__ == "__main__":
    test_sweep()
