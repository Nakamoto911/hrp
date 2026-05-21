import yfinance as yf
import pandas as pd

tickers = {
    "FTSE 100": "SXRW.DE",
    "Nikkei 225 (XDJP)": "XDJP.DE",
    "Nikkei 225 (SXRZ)": "SXRZ.DE"
}

print("Checking index tickers on yfinance...")
for name, ticker in tickers.items():
    try:
        df = yf.download(ticker, start="2000-01-01", progress=False)
        if not df.empty:
            print(f"  {name} ({ticker}): Start={df.index[0].strftime('%Y-%m-%d')}, Rows={len(df)}, Currency={yf.Ticker(ticker).info.get('currency')}")
        else:
            print(f"  {name} ({ticker}): EMPTY")
    except Exception as e:
        print(f"  {name} ({ticker}): FAILED: {e}")
