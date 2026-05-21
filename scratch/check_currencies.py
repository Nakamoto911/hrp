import yfinance as yf
import pandas as pd
import sys
import os

# Set paths
sys.path.append(os.path.abspath('.'))

from hrp_engine.data import EUROPEAN_POOL

print("Fetching currency info for all assets in the European pool...")
for ticker in EUROPEAN_POOL:
    try:
        t = yf.Ticker(ticker)
        info = t.info
        currency = info.get('currency', 'UNKNOWN')
        price_hint = info.get('priceHint', 'N/A')
        exchange = info.get('exchange', 'UNKNOWN')
        long_name = info.get('longName', 'UNKNOWN')
        print(f"  {ticker:<10} | Currency: {currency:<5} | Exchange: {exchange:<5} | Name: {long_name}")
    except Exception as e:
        print(f"  {ticker:<10} | Failed to fetch info: {e}")
