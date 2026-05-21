import pandas as pd
import os

def analyze():
    csv_path = '/Users/user2/Documents/Code/hrp/scratch/grid_search_results.csv'
    if not os.path.exists(csv_path):
        print("Grid search results not found!")
        return
        
    df = pd.read_csv(csv_path)
    
    # Separate by pool
    for pool in df['pool'].unique():
        print(f"\n==================================================")
        print(f"ANALYSIS FOR POOL: {pool.upper()}")
        print(f"==================================================")
        pool_df = df[df['pool'] == pool]
        
        # 1. Best by Sharpe Ratio
        print("\n--- Top 5 by Sharpe Ratio ---")
        top_sharpe = pool_df.sort_values(by='sharpe_ratio', ascending=False).head(5)
        for idx, row in top_sharpe.iterrows():
            print(f"Sharpe: {row['sharpe_ratio']:.3f} | Ann. Return: {row['ann_return']*100:.2f}% | Ann. Vol: {row['ann_vol']*100:.2f}% | Max DD: {row['max_drawdown']*100:.2f}% | Turnover: {row['ann_turnover']*100:.2f}%")
            print(f"      Params: linkage={row['linkage_method']}, denoise={row['denoise']}, lookback={row['lookback_years']}y, freq={row['rebalance_frequency']}, drift={row['drift_threshold']}")
            
        # 2. Best by Annualized Return
        print("\n--- Top 5 by Annualized Return ---")
        top_ret = pool_df.sort_values(by='ann_return', ascending=False).head(5)
        for idx, row in top_ret.iterrows():
            print(f"Ann. Return: {row['ann_return']*100:.2f}% | Sharpe: {row['sharpe_ratio']:.3f} | Ann. Vol: {row['ann_vol']*100:.2f}% | Max DD: {row['max_drawdown']*100:.2f}% | Turnover: {row['ann_turnover']*100:.2f}%")
            print(f"      Params: linkage={row['linkage_method']}, denoise={row['denoise']}, lookback={row['lookback_years']}y, freq={row['rebalance_frequency']}, drift={row['drift_threshold']}")
            
        # 3. Impact of Denoising
        print("\n--- Denoising vs No Denoising Impact ---")
        denoise_impact = pool_df.groupby('denoise')[['ann_return', 'sharpe_ratio', 'ann_turnover', 'max_drawdown']].mean()
        print(denoise_impact)
        
        # 4. Impact of Linkage Method
        print("\n--- Linkage Method Impact ---")
        linkage_impact = pool_df.groupby('linkage_method')[['ann_return', 'sharpe_ratio', 'ann_turnover', 'max_drawdown']].mean()
        print(linkage_impact)
        
        # 5. Impact of Lookback Years
        print("\n--- Lookback Years Impact ---")
        lookback_impact = pool_df.groupby('lookback_years')[['ann_return', 'sharpe_ratio', 'ann_turnover', 'max_drawdown']].mean()
        print(lookback_impact)
        
        # 6. Impact of Rebalance Frequency
        print("\n--- Rebalance Frequency Impact ---")
        freq_impact = pool_df.groupby('rebalance_frequency')[['ann_return', 'sharpe_ratio', 'ann_turnover', 'max_drawdown']].mean()
        print(freq_impact)
        
        # 7. Impact of Drift Threshold
        print("\n--- Drift Threshold Impact ---")
        drift_impact = pool_df.groupby('drift_threshold')[['ann_return', 'sharpe_ratio', 'ann_turnover', 'max_drawdown']].mean()
        print(drift_impact)

if __name__ == '__main__':
    analyze()
