import yfinance as yf

alternatives = {
    "UK Gilts": ["IGLT.L", "IGLT.DE", "VGOV.L", "VGOV.DE"],
    "FTSE 100": ["ISF.L", "VUKE.L", "EXS1.DE"],
    "Equity Value": ["VALW.L", "VALW.DE", "IWVL.L", "IWVL.DE", "VVAL.L"]
}

for name, tickers in alternatives.items():
    print(f"\n--- {name} ---")
    for t in tickers:
        try:
            df = yf.download(t, start="2000-01-01", progress=False)
            if not df.empty:
                print(f"  {t}: Start={df.index[0].strftime('%Y-%m-%d')}, Rows={len(df)}")
            else:
                print(f"  {t}: EMPTY")
        except Exception as e:
            print(f"  {t}: FAILED: {e}")
