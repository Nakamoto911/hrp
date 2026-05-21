import pandas as pd
import numpy as np

# Load cache
df = pd.read_csv("cache/european_prices.csv", index_col=0, parse_dates=True)

# Try rolling median filter with window=3 or 5
df_cleaned_3 = df.rolling(window=3, min_periods=1, center=True).median()

print("Comparing Original vs Cleaned (window=3) Volatilities and Extreme Returns:")
for col in df.columns:
    series_org = df[col].dropna()
    series_cl3 = df_cleaned_3[col].dropna()
    
    if len(series_org) < 2:
        continue
        
    rets_org = series_org.pct_change().dropna()
    rets_cl3 = series_cl3.pct_change().dropna()
    
    vol_org = rets_org.std() * np.sqrt(252.0) * 100
    vol_cl3 = rets_cl3.std() * np.sqrt(252.0) * 100
    
    ext_org = ((rets_org > 0.10) | (rets_org < -0.10)).sum()
    ext_cl3 = ((rets_cl3 > 0.10) | (rets_cl3 < -0.10)).sum()
    
    # Check if this is a crypto asset
    is_crypto = 'BTCE' in col
    
    print(f"  {col:<10} | Org Vol: {vol_org:5.2f}% (Ext: {ext_org:2d}) | Cl3 Vol: {vol_cl3:5.2f}% (Ext: {ext_cl3:2d})")
