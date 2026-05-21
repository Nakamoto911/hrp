import pandas as pd
import plotly.express as px
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

# Melt dataframe for plotly express
weights_melted = weights_raw_df.reset_index().rename(columns={'index': 'Date'}).melt(
    id_vars='Date', 
    var_name='Asset Name (Symbol)', 
    value_name='Weight'
)

fig_weights = px.bar(
    weights_melted, 
    x='Date', 
    y='Weight', 
    color='Asset Name (Symbol)',
)

diffs = pd.Series(weights_raw_df.index).diff().dropna()
median_diff_ms = diffs.median().total_seconds() * 1000
bar_width = median_diff_ms * 0.95

print("Computed bar width:", bar_width)
fig_weights.update_traces(width=bar_width, selector=dict(type='bar'))

for i, trace in enumerate(fig_weights.data):
    print(f"Trace {i}: name={trace.name}, type={trace.type}, width={getattr(trace, 'width', None)}")
