import yfinance as yf

suffixes = ["DE", "F", "MI", "PA", "L", "SW"]
symbols_to_test = {
    "JGB Bonds": ["XJSE", "DBX0N3", "XJGB"],
    "Credit IG": ["LQEE", "SXXF", "IBCQ", "CRPH"],
    "Credit HY": ["IHYE", "SBNK", "HYHE", "IHYG"],
    "Oil": ["CRUD", "OD7F", "WTIO"],
    "Copper": ["COPA", "OD7D", "COPP"],
    "Agriculture": ["AIGP", "OD7A", "AGRI"],
    "Equity Quality": ["QDVB", "IS38", "5MVY", "IUQA"],
    "Equity Defensive": ["IUMV", "IUS3", "MVEA", "IUMV"]
}

print("Running deep search on tickers and suffixes...")
for category, base_symbols in symbols_to_test.items():
    print(f"\n=================== Category: {category} ===================")
    found_any = False
    for base in base_symbols:
        # Also test base symbol without suffix (often LSE in USD/GBP)
        for s in [""] + suffixes:
            ticker = f"{base}.{s}" if s else base
            try:
                df = yf.download(ticker, start="2018-01-01", end="2026-05-20", progress=False)
                if not df.empty and len(df) > 100:
                    print(f"  [FOUND] {ticker} -> Shape: {df.shape}, Start: {df.index[0].strftime('%Y-%m-%d')}, End: {df.index[-1].strftime('%Y-%m-%d')}")
                    found_any = True
            except Exception:
                pass
    if not found_any:
        print(f"  [ERROR] No active tickers found for {category}!")
