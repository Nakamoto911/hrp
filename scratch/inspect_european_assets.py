import pandas as pd
import numpy as np

# Load cached prices
prices_df = pd.read_csv("cache/european_prices.csv", index_col=0, parse_dates=True)

# Let's print metrics for each asset (from 2012-01-02 onwards, to match the backtest trading period)
trade_prices = prices_df.loc['2012-01-02':].ffill().bfill()
returns = trade_prices.pct_change().dropna(how='all')

# Annualized return and volatility
summary_rows = []
for col in trade_prices.columns:
    col_ret = returns[col]
    ann_ret = (1 + col_ret).prod() ** (252 / len(col_ret)) - 1 if len(col_ret) > 0 else 0
    ann_vol = col_ret.std() * np.sqrt(252) if len(col_ret) > 0 else 0
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    
    # Check first and last valid prices
    first_p = trade_prices[col].dropna().iloc[0]
    last_p = trade_prices[col].dropna().iloc[-1]
    
    summary_rows.append({
        "Ticker": col,
        "Start Price": f"{first_p:.2f}",
        "End Price": f"{last_p:.2f}",
        "Ann. Return": f"{ann_ret*100:.2f}%",
        "Ann. Volatility": f"{ann_vol*100:.2f}%",
        "Sharpe": f"{sharpe:.2f}"
    })
    
summary_df = pd.DataFrame(summary_rows)
print(summary_df.to_string(index=False))
