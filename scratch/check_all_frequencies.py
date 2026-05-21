import pandas as pd
from hrp_engine.config import StrategyParams
from hrp_engine.data import fetch_data
from hrp_engine.backtest import run_strategy_backtest

pools = ["etf", "mutual_fund", "european"]
frequencies = ["monthly", "quarterly", "semi-annually", "yearly"]

for pool in pools:
    print(f"\n================ POOL: {pool} ================")
    try:
        prices_df = fetch_data(pool, force_refresh=False)
    except Exception as e:
        print(f"Skipping pool {pool} due to fetch error: {e}")
        continue
        
    for freq in frequencies:
        params = StrategyParams(
            lookback_years=4,
            rebalance_frequency=freq,
            linkage_method="single",
            drift_threshold=0.015,
            transaction_cost_bps=5.0,
            french_pfu_rate=0.314,
            bisection_method="tree"
        )
        try:
            hrp_cum, hrp_diag, sp500_cum, sixty_forty_cum, sf_diag = run_strategy_backtest(
                prices_df=prices_df,
                params=params,
                pool_name=pool,
                limit_to_least_history=False
            )
            weights_raw_df = pd.DataFrame.from_dict(hrp_diag["weight_history"], orient='index')
            weights_raw_df.index = pd.to_datetime(weights_raw_df.index)
            diffs = pd.Series(weights_raw_df.index).diff().dropna()
            
            print(f"Frequency: {freq:15s} | Row Count: {len(weights_raw_df):3d} | Min Diff: {str(diffs.min()):15s} | Median Diff: {str(diffs.median()):15s}")
        except Exception as e:
            print(f"Frequency: {freq:15s} | Error: {e}")
