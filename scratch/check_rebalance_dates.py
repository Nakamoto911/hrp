import pandas as pd
from hrp_engine.config import StrategyParams
from hrp_engine.data import fetch_data
from hrp_engine.backtest import run_strategy_backtest

params = StrategyParams(
    lookback_years=4,
    rebalance_frequency="quarterly",
    linkage_method="single",
    drift_threshold=0.015,
    transaction_cost_bps=5.0,
    french_pfu_rate=0.314,
    bisection_method="tree"
)

prices_df = fetch_data("etf", force_refresh=False)
hrp_cum, hrp_diag, sp500_cum, sixty_forty_cum, sf_diag = run_strategy_backtest(
    prices_df=prices_df,
    params=params,
    pool_name="etf",
    limit_to_least_history=False
)

weights_raw_df = pd.DataFrame.from_dict(hrp_diag["weight_history"], orient='index')
weights_raw_df.index = pd.to_datetime(weights_raw_df.index)
print("Index length:", len(weights_raw_df.index))
print("Index head:")
print(weights_raw_df.index[:15])
diffs = pd.Series(weights_raw_df.index).diff().dropna()
print("Diffs count:", len(diffs))
print("Diffs description:")
print(diffs.describe())
print("Median diff total seconds:", diffs.median().total_seconds())
print("Median diff in days:", diffs.median().total_seconds() / 86400)
