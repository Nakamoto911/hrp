import argparse
import sys
import pandas as pd
import time
from hrp_engine.config import StrategyParams
from hrp_engine.data import fetch_data
from hrp_engine.backtest import run_strategy_backtest
from hrp_engine.reporting import compute_metrics, generate_markdown_report, generate_json_output

def main():
    parser = argparse.ArgumentParser(description="High-Performance Multi-Asset HRP Allocation & Backtesting Engine")
    parser.add_argument("--pool", type=str, default="etf", choices=["etf", "mutual_fund", "european"],
                        help="Asset pool to backtest: 'etf', 'mutual_fund', or 'european' (default: 'etf')")
    parser.add_argument("--lookback-years", type=int, default=4,
                        help="Lookback window in years for covariance matrix estimation (default: 4)")
    parser.add_argument("--frequency", type=str, default="quarterly", choices=["daily", "monthly", "quarterly", "yearly"],
                        help="Rebalancing frequency (default: 'quarterly')")
    parser.add_argument("--linkage", type=str, default="single", choices=["single", "complete", "ward"],
                        help="Hierarchical clustering linkage method (default: 'single')")
    parser.add_argument("--drift-threshold", type=float, default=0.015,
                        help="Drift band threshold for portfolio weight execution (default: 0.015)")
    parser.add_argument("--transaction-cost", type=float, default=5.0,
                        help="Transaction costs in bps (default: 5.0)")
    parser.add_argument("--pfu-rate", type=float, default=0.314,
                        help="French PFU Flat Tax rate (default: 0.314)")
    parser.add_argument("--force-refresh", action="store_true",
                        help="Force download of daily historical prices from yfinance")
    parser.add_argument("--output-json", type=str, default=None,
                        help="File path to save the structured JSON chart output")
    parser.add_argument("--limit-history", action="store_true",
                        help="Limit backtest start date to the asset with the least history")
    
    args = parser.parse_args()
    
    # Initialize strategy params
    params = StrategyParams(
        lookback_years=args.lookback_years,
        rebalance_frequency=args.frequency,
        linkage_method=args.linkage,
        drift_threshold=args.drift_threshold,
        transaction_cost_bps=args.transaction_cost,
        french_pfu_rate=args.pfu_rate
    )
    
    # 1. Fetch data
    try:
        prices_df = fetch_data(args.pool, force_refresh=args.force_refresh)
    except Exception as e:
        print(f"FATAL ERROR: Failed to retrieve data. {e}", file=sys.stderr)
        sys.exit(1)
        
    # Find the start date of the asset with the least history
    from hrp_engine.data import get_least_history_info
    start_date_least_history, least_history_asset = get_least_history_info(prices_df)
    
    # Slice prices_df to start from this date if requested
    if args.limit_history:
        prices_df = prices_df.loc[start_date_least_history:]
        
    print(f"Running HRP walk-forward backtest on pool: {args.pool.upper()}...")
    if args.limit_history:
        print(f"Simulation start date (Aligned - asset of least history: {least_history_asset}): {start_date_least_history.strftime('%Y-%m-%d')}")
    else:
        print(f"Simulation start date (Full Range): {prices_df.index[0].strftime('%Y-%m-%d')}")
    print(f"Parameters: {params}")
    
    # 2. Run backtest
    start_time = time.perf_counter()
    
    try:
        hrp_cum, hrp_diag, sp500_cum, sixty_forty_cum, sf_diag = run_strategy_backtest(
            prices_df=prices_df,
            params=params,
            pool_name=args.pool,
            limit_to_least_history=args.limit_history
        )
    except Exception as e:
        print(f"FATAL ERROR: Backtest failed. {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    elapsed_time = time.perf_counter() - start_time
    print(f"Backtest completed in {elapsed_time:.4f} seconds.")
    
    # 3. Compute final metrics
    hrp_metrics = compute_metrics(
        hrp_cum, 
        total_tc=hrp_diag["total_friction_costs_paid"], 
        bps=params.transaction_cost_bps, 
        initial_capital=100000.0
    )
    
    # S&P 500 B&H has no rebalance costs or taxes
    sp500_metrics = compute_metrics(sp500_cum, total_tc=0.0, bps=0.0, initial_capital=100000.0)
    
    # 60/40 Portfolio metrics (rebalanced, so it has transaction costs)
    sixty_forty_metrics = compute_metrics(
        sixty_forty_cum, 
        total_tc=sf_diag["total_friction_costs_paid"], 
        bps=params.transaction_cost_bps, 
        initial_capital=100000.0
    )
    
    # 4. Generate & print report
    report = generate_markdown_report(hrp_metrics, sp500_metrics, sixty_forty_metrics, hrp_diag)
    print("\n" + report + "\n")
    
    # 5. Generate & print JSON diagnostics block
    json_output = generate_json_output(params, hrp_cum, sp500_cum, sixty_forty_cum, hrp_diag)
    print("### CHART DATA DIAGNOSTICS JSON")
    print(json_output)
    
    # Save to file if path is specified
    if args.output_json:
        try:
            with open(args.output_json, 'w') as f:
                f.write(json_output)
            print(f"\nJSON output saved to {args.output_json}")
        except Exception as e:
            print(f"Error saving JSON to {args.output_json}: {e}", file=sys.stderr)

if __name__ == "__main__":
    main()
