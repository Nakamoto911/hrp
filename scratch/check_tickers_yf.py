import yfinance as yf
import pandas as pd

tickers = [
    'SXR8.DE', 'EXX5.DE', 'EXS1.DE', 'SXRT.DE', 'IS3N.DE',
    'IUSM.DE', 'IS0L.DE', 'XJSE.DE', 'IGLT.L',
    'IBCQ.DE', 'IHYG.MI',
    '4GLD.DE', 'CRUD.MI', 'COPA.MI', 'AIGP.MI',
    '5MVW.DE', '36BZ.DE', 'QDVB.DE', 'IUS3.DE',
    'BTCE.DE'
]

for t in tickers:
    try:
        ticker_obj = yf.Ticker(t)
        info = ticker_obj.info
        name = info.get('longName', 'N/A')
        currency = info.get('currency', 'N/A')
        exchange = info.get('exchange', 'N/A')
        print(f"{t}: {name} | Currency: {currency} | Exchange: {exchange}")
    except Exception as e:
        print(f"Error for {t}: {e}")
