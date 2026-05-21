import yfinance as yf
import pandas as pd

factors = {
    "Value": "IS3S.DE",
    "Momentum": "IS3R.DE",
    "Quality": "IS3Q.DE",
    "Min Vol (Defensive)": "IQQ0.DE"
}

print("Checking factor tickers on yfinance...")
for name, ticker in factors.items():
    try:
        df = yf.download(ticker, start="2000-01-01", progress=False)
        if not df.empty:
            print(f"  {name} ({ticker}): Start={df.index[0].strftime('%Y-%m-%d')}, Rows={len(df)}, Currency={yf.Ticker(ticker).info.get('currency')}")
        else:
            print(f"  {name} ({ticker}): EMPTY")
    except Exception as e:
        print(f"  {name} ({ticker}): FAILED: {e}")
