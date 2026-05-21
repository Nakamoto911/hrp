import pandas as pd
import numpy as np

df = pd.read_csv("cache/european_prices.csv", index_col=0, parse_dates=True)

target_events = {
    "SXR8.DE": "2010-11-01",
    "IUS3.DE": "2008-05-26",
    "IGLT.MI": "2008-05-05",
    "COPA.MI": "2025-07-31"
}

for ticker, date_str in target_events.items():
    if ticker in df.columns:
        date = pd.Timestamp(date_str)
        idx = df.index.get_loc(date)
        start_idx = max(0, idx - 3)
        end_idx = min(len(df), idx + 4)
        print(f"\n--- {ticker} around {date_str} ---")
        print(df.iloc[start_idx:end_idx][ticker])
