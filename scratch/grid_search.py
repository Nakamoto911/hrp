import os
import sys
import pandas as pd
import numpy as np
import itertools
import concurrent.futures

# Add root folder to path so we can import hrp_engine
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hrp_engine.config import StrategyParams
from hrp_engine.data import fetch_data
from hrp_engine.backtest import run_strategy_backtest
from hrp_engine.reporting import compute_metrics

def evaluate_config(config):
    pool, linkage, denoise, lookback, freq, drift, prices_df = config
    params = StrategyParams(
        lookback_years=lookback,
        rebalance_frequency=freq,
        linkage_method=linkage,
        drift_threshold=drift,
        denoise=denoise
    )
    try:
        hrp_cum, hrp_diag, sp500_cum, sixty_forty_cum, sf_diag = run_strategy_backtest(
            prices_df=prices_df,
            params=params,
            pool_name=pool
        )
        hrp_metrics = compute_metrics(
            hrp_cum, 
            total_tc=hrp_diag["total_friction_costs_paid"], 
            bps=params.transaction_cost_bps, 
            initial_capital=100000.0
        )
        return {
            'pool': pool,
            'linkage_method': linkage,
            'denoise': denoise,
            'lookback_years': lookback,
            'rebalance_frequency': freq,
            'drift_threshold': drift,
            'ann_return': hrp_metrics['annualized_return'],
            'ann_vol': hrp_metrics['annualized_volatility'],
            'sharpe_ratio': hrp_metrics['sharpe_ratio'],
            'sortino_ratio': hrp_metrics['sortino_ratio'],
            'max_drawdown': hrp_metrics['max_drawdown'],
            'ann_turnover': hrp_metrics['annualized_turnover'],
            'total_friction_costs': hrp_diag['total_friction_costs_paid'],
            'total_taxes': hrp_diag['total_pfu_taxes_paid'],
            'rebalance_events': hrp_diag['total_rebalance_events']
        }
    except Exception as e:
        return {
            'error': str(e),
            'pool': pool,
            'linkage_method': linkage,
            'denoise': denoise,
            'lookback_years': lookback,
            'rebalance_frequency': freq,
            'drift_threshold': drift
        }

def main():
    pools = ['european', 'etf']
    linkages = ['single', 'complete', 'average', 'ward']
    denoises = [True, False]
    lookbacks = [2, 3, 4, 5]
    freqs = ['monthly', 'quarterly', 'semi-annually']
    drifts = [0.0, 0.015, 0.03]
    
    # Load data for pools
    data_dict = {}
    for pool in pools:
        print(f"Fetching data for {pool}...")
        data_dict[pool] = fetch_data(pool)
        
    print("Skipping main grid search as results are already saved in grid_search_results.csv")
    
    # Sensitivity analysis section
    print("\n--- Running Asset Selection Sensitivity Analysis ---")
    sensitivity_results = []
    
    # Configurations for sensitivity analysis:
    european_prices = data_dict['european']
    
    sens_configs = [
        ("All Assets (Baseline)", []),
        ("Exclude Japanese Gov Bonds (XJSE.DE)", ['XJSE.DE']),
        ("Exclude German Bunds & Japanese Gov Bonds (IS0L.DE, XJSE.DE)", ['IS0L.DE', 'XJSE.DE']),
        ("Exclude All Government Bonds (XJSE.DE, IS0L.DE, IGLT.MI, IBCQ.DE)", ['XJSE.DE', 'IS0L.DE', 'IGLT.MI', 'IBCQ.DE'])
    ]
    
    # Baseline configuration
    baseline_params = StrategyParams(
        lookback_years=4,
        rebalance_frequency='quarterly',
        linkage_method='single',
        drift_threshold=0.015,
        denoise=True
    )
    
    for name, exclude_list in sens_configs:
        # Filter active assets for HRP but keep prices_df unfiltered for benchmarks
        active_univ = [c for c in european_prices.columns if c not in exclude_list]
        
        try:
            hrp_cum, hrp_diag, sp500_cum, sixty_forty_cum, sf_diag = run_strategy_backtest(
                prices_df=european_prices,
                params=baseline_params,
                pool_name='european',
                active_universe=active_univ
            )
            hrp_metrics = compute_metrics(
                hrp_cum, 
                total_tc=hrp_diag["total_friction_costs_paid"], 
                bps=baseline_params.transaction_cost_bps, 
                initial_capital=100000.0
            )
            sensitivity_results.append({
                'config_name': name,
                'excluded': ", ".join(exclude_list) if exclude_list else "None",
                'ann_return': hrp_metrics['annualized_return'],
                'ann_vol': hrp_metrics['annualized_volatility'],
                'sharpe_ratio': hrp_metrics['sharpe_ratio'],
                'max_drawdown': hrp_metrics['max_drawdown'],
                'ann_turnover': hrp_metrics['annualized_turnover']
            })
        except Exception as e:
            print(f"Error in sensitivity analysis '{name}': {e}")
            
    df_sens = pd.DataFrame(sensitivity_results)
    sens_output_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'sensitivity_results.csv'))
    df_sens.to_csv(sens_output_path, index=False)
    print(f"Sensitivity analysis completed. Results saved to {sens_output_path}")

if __name__ == '__main__':
    main()
