import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from hrp_engine.config import StrategyParams
from hrp_engine.denoiser import denoise_covariance
from hrp_engine.hrp import optimize_hrp
from hrp_engine.data import get_lookback_data

def get_rebalance_dates(index: pd.DatetimeIndex, frequency: str) -> pd.DatetimeIndex:
    """
    Identifies the rebalancing dates in the historical prices DatetimeIndex
    based on the specified frequency.
    """
    df = pd.Series(index, index=index)
    if frequency == 'daily':
        return index
    elif frequency == 'weekly':
        return pd.DatetimeIndex(df.groupby([index.isocalendar().year, index.isocalendar().week]).last().values)
    elif frequency == 'monthly':
        return pd.DatetimeIndex(df.groupby([index.year, index.month]).last().values)
    elif frequency == 'quarterly':
        return pd.DatetimeIndex(df.groupby([index.year, (index.month - 1) // 3]).last().values)
    elif frequency == 'semi-annually':
        return pd.DatetimeIndex(df.groupby([index.year, (index.month - 1) // 6]).last().values)
    elif frequency == 'yearly':
        return pd.DatetimeIndex(df.groupby(index.year).last().values)
    else:
        raise ValueError(f"Unknown frequency: {frequency}")

class PortfolioState:
    """
    Helper class to track cash, holdings, and cost bases for capital gains tax calculations.
    """
    def __init__(self, initial_cash: float):
        self.cash: float = initial_cash
        self.holdings: Dict[str, float] = {}  # ticker -> shares
        self.cost_basis: Dict[str, float] = {}  # ticker -> total cost basis of the position
        self.tax_loss_carryforward: float = 0.0
        self.total_transaction_costs: float = 0.0
        self.total_taxes_paid: float = 0.0
        
    def get_value(self, prices: Dict[str, float]) -> float:
        val = self.cash
        for ticker, shares in self.holdings.items():
            if shares > 0 and ticker in prices and not pd.isna(prices[ticker]):
                val += shares * prices[ticker]
        return val

def run_strategy_backtest(
    prices_df: pd.DataFrame, 
    params: StrategyParams, 
    pool_name: str,
    initial_capital: float = 100000.0,
    limit_to_least_history: bool = False,
    active_universe: List[str] = None
) -> Tuple[pd.Series, Dict, pd.Series, pd.Series, Dict]:
    """
    Runs the HRP backtest with Marchenko-Pastur denoising, drift bands, transaction costs, and PFU tax.
    Also runs concurrent tracking for S&P 500 Buy & Hold and 60/40 Equity/Bond.
    
    Returns:
    - hrp_equity: Series of daily portfolio values for HRP
    - hrp_diagnostics: Dict of diagnostic metrics (costs, taxes, events)
    - sp500_equity: Series of daily portfolio values for S&P 500 B&H
    - sixty_forty_equity: Series of daily portfolio values for 60/40 Portfolio
    - sixty_forty_diagnostics: Dict of diagnostic metrics for 60/40 Portfolio
    """
    # Slice prices to start at the start date of the asset with the least history if requested
    if limit_to_least_history:
        first_valid_dates = {col: prices_df[col].first_valid_index() for col in prices_df.columns}
        start_date_least_history = max(first_valid_dates.values())
        prices_df = prices_df.loc[start_date_least_history:]

    # Clean prices: forward fill then backward fill to handle any daily gaps
    prices_clean = prices_df.ffill().bfill()
    
    # Track historical target weights at rebalance dates
    hrp_weights_history = {}
    
    # 1. Setup benchmarks
    if pool_name.lower() == 'etf':
        equity_ticker = 'IVV'
        bond_ticker = 'AGG'
    elif pool_name.lower() == 'european':
        equity_ticker = 'SXR8.DE'
        bond_ticker = 'IS0L.DE'
    else:
        equity_ticker = '^SP500TR'
        bond_ticker = 'VBMFX'
        
    # Get rebalancing dates
    all_rebalance_dates = get_rebalance_dates(prices_clean.index, params.rebalance_frequency)
    
    # We can only rebalance if we have lookback_years of data.
    # Find the first rebalance date where we have enough lookback history
    start_date = prices_clean.index[0]
    valid_rebalance_dates = [
        d for d in all_rebalance_dates 
        if d >= start_date + pd.DateOffset(years=params.lookback_years)
    ]
    
    if not valid_rebalance_dates:
        raise ValueError("Prices dataset is too short for the specified lookback_years")
        
    # Slicing the price index from the first valid rebalance date onwards
    backtest_dates = prices_clean.index[prices_clean.index >= valid_rebalance_dates[0]]
    
    # Initialize states
    hrp_state = PortfolioState(initial_capital)
    sixty_forty_state = PortfolioState(initial_capital)
    
    # Track daily equity lines
    hrp_equity = pd.Series(index=backtest_dates, dtype=float)
    sixty_forty_equity = pd.Series(index=backtest_dates, dtype=float)
    sp500_equity = pd.Series(index=backtest_dates, dtype=float)
    
    # Track daily transaction costs, tax, and carryforward
    hrp_tc_series = pd.Series(0.0, index=backtest_dates)
    hrp_tax_series = pd.Series(0.0, index=backtest_dates)
    hrp_tlc_series = pd.Series(0.0, index=backtest_dates)
    
    sf_tc_series = pd.Series(0.0, index=backtest_dates)
    sf_tax_series = pd.Series(0.0, index=backtest_dates)
    sf_tlc_series = pd.Series(0.0, index=backtest_dates)
    
    # Track actual HRP trade dates
    hrp_trade_dates = []
    
    # S&P 500 B&H Initial Setup
    sp500_price_start = prices_clean.loc[valid_rebalance_dates[0], equity_ticker]
    sp500_val_post = initial_capital / (1.0 + params.transaction_cost_bps / 10000.0)
    sp500_shares = sp500_val_post / sp500_price_start
    
    # Diagnostic counts
    hrp_events_triggered = 0
    average_assets_traded = []
    
    # Iterate over rebalancing intervals
    for k in range(len(valid_rebalance_dates)):
        T_k = valid_rebalance_dates[k]
        T_k_plus_1 = valid_rebalance_dates[k+1] if k+1 < len(valid_rebalance_dates) else None
        
        # Slicing daily dates in this interval
        if T_k_plus_1 is not None:
            interval_dates = backtest_dates[(backtest_dates >= T_k) & (backtest_dates < T_k_plus_1)]
        else:
            interval_dates = backtest_dates[backtest_dates >= T_k]
            
        if len(interval_dates) == 0:
            continue
            
        prices_at_T_k = prices_clean.loc[T_k].to_dict()
        
        # Get historical returns for covariance calculation
        lookback_returns, active_assets = get_lookback_data(prices_df, T_k, params.lookback_years)
        
        if active_universe is not None:
            active_assets = [a for a in active_assets if a in active_universe]
            lookback_returns = lookback_returns[active_assets]
        
        # Calculate empirical covariance
        cov_emp = lookback_returns.cov().values
        n_obs = len(lookback_returns)
        
        # Denoise using Marchenko-Pastur if enabled
        if getattr(params, 'denoise', True):
            cov_denoised = denoise_covariance(cov_emp, n_obs)
        else:
            cov_denoised = cov_emp
        cov_denoised_df = pd.DataFrame(cov_denoised, index=active_assets, columns=active_assets)
        
        # Compute HRP target weights W*
        bisection_method = getattr(params, 'bisection_method', 'tree')
        w_star = optimize_hrp(cov_denoised_df, params.linkage_method, bisection_method)
        
        # Get current HRP portfolio value before trades
        hrp_v_prior = hrp_state.get_value(prices_at_T_k)
        
        # Determine weights
        if hrp_events_triggered == 0:
            # First rebalance: allocate fully
            w_new = w_star
            is_first = True
        else:
            # Calculate current weights before trade
            w_curr = pd.Series(0.0, index=active_assets)
            for ticker in active_assets:
                shares = hrp_state.holdings.get(ticker, 0.0)
                price = prices_at_T_k[ticker]
                w_curr[ticker] = (shares * price) / hrp_v_prior
                
            # Apply drift threshold
            w_new = pd.Series(0.0, index=active_assets)
            untraded = []
            traded = []
            
            # Identify which active assets are traded vs untraded
            for ticker in active_assets:
                w_target = w_star.get(ticker, 0.0)
                w_c = w_curr.get(ticker, 0.0)
                drift = abs(w_target - w_c)
                if drift < params.drift_threshold:
                    untraded.append(ticker)
                else:
                    traded.append(ticker)
                    
            # Liquidate any assets currently held but not in the active universe
            for ticker in list(hrp_state.holdings.keys()):
                if hrp_state.holdings[ticker] > 0 and ticker not in active_assets:
                    traded.append(ticker)
                    
            sum_untraded_w_curr = sum(w_curr.get(t, 0.0) for t in untraded)
            
            if sum_untraded_w_curr >= 0.95 or len(traded) == 0:
                # Force trade all active assets to raw target if untraded weights are too large
                w_new = w_star
                traded = list(set(active_assets).union(hrp_state.holdings.keys()))
                is_first = False
            else:
                is_first = False
                # Freeze untraded weights
                for ticker in untraded:
                    w_new[ticker] = w_curr[ticker]
                # Scale traded weights to match remainder
                sum_traded_target = sum(w_star.get(t, 0.0) for t in traded if t in active_assets)
                remainder = 1.0 - sum_untraded_w_curr
                
                for ticker in traded:
                    if ticker in active_assets:
                        target = w_star.get(ticker, 0.0)
                        if sum_traded_target > 0:
                            w_new[ticker] = target * (remainder / sum_traded_target)
                        else:
                            w_new[ticker] = remainder / len(traded)
                    else:
                        w_new[ticker] = 0.0  # Fully liquidate inactive asset
                        
        # Trade execution
        if hrp_events_triggered == 0 or any(t in traded for t in w_new.index if w_new[t] != w_curr.get(t, 0.0)):
            hrp_trade_dates.append(T_k)
            # Calculate first-pass trade values
            if hrp_events_triggered == 0:
                w_curr_temp = pd.Series(0.0, index=w_new.index)
            else:
                w_curr_temp = w_curr
                
            trade_values_temp = {}
            for ticker in w_new.index:
                w_t = w_new[ticker]
                w_c = w_curr_temp.get(ticker, 0.0)
                trade_values_temp[ticker] = hrp_v_prior * (w_t - w_c)
                
            for ticker in hrp_state.holdings.keys():
                if ticker not in trade_values_temp:
                    # Fully sell inactive assets
                    trade_values_temp[ticker] = -hrp_state.holdings[ticker] * prices_at_T_k[ticker]
                    
            # Compute transaction costs and taxes based on first-pass trades
            temp_tc = sum(abs(v) for v in trade_values_temp.values()) * (params.transaction_cost_bps / 10000.0)
            
            # Tax calculations on first-pass sales
            temp_gains = 0.0
            for ticker, v in trade_values_temp.items():
                if v < 0:
                    sold_val = -v
                    old_shares = hrp_state.holdings.get(ticker, 0.0)
                    if old_shares > 0:
                        old_price = prices_at_T_k[ticker]
                        frac = sold_val / (old_shares * old_price)
                        frac = min(1.0, frac)
                        old_basis = hrp_state.cost_basis.get(ticker, 0.0)
                        sold_basis = old_basis * frac
                        gain = sold_val - sold_basis
                        temp_gains += gain
                        
            # Apply tax loss carryforward
            temp_tax = 0.0
            if temp_gains > 0:
                taxable_gain = max(0.0, temp_gains - hrp_state.tax_loss_carryforward)
                temp_tax = taxable_gain * params.french_pfu_rate
            
            # Post-rebalance target portfolio value
            hrp_v_post = hrp_v_prior - temp_tc - temp_tax
            
            # Execute actual trades based on hrp_v_post
            actual_trade_values = {}
            for ticker in w_new.index:
                target_val = hrp_v_post * w_new[ticker]
                curr_val = hrp_state.holdings.get(ticker, 0.0) * prices_at_T_k[ticker]
                actual_trade_values[ticker] = target_val - curr_val
                
            for ticker in list(hrp_state.holdings.keys()):
                if ticker not in actual_trade_values:
                    actual_trade_values[ticker] = -hrp_state.holdings[ticker] * prices_at_T_k[ticker]
                    
            # Actual sales first
            realized_gains_sales = 0.0
            n_traded = 0
            for ticker, v in actual_trade_values.items():
                if abs(v) > 1e-5:
                    n_traded += 1
                if v < 0:
                    sold_val = -v
                    old_shares = hrp_state.holdings.get(ticker, 0.0)
                    if old_shares > 0:
                        old_price = prices_at_T_k[ticker]
                        frac = min(1.0, sold_val / (old_shares * old_price))
                        old_basis = hrp_state.cost_basis.get(ticker, 0.0)
                        sold_basis = old_basis * frac
                        gain = sold_val - sold_basis
                        realized_gains_sales += gain
                        
                        # Update state
                        hrp_state.holdings[ticker] = old_shares * (1.0 - frac)
                        hrp_state.cost_basis[ticker] = old_basis - sold_basis
                        hrp_state.cash += sold_val
                        
            # Calculate final taxes and costs
            actual_tc = sum(abs(v) for v in actual_trade_values.values()) * (params.transaction_cost_bps / 10000.0)
            
            actual_tax = 0.0
            if realized_gains_sales > 0:
                taxable_gain = max(0.0, realized_gains_sales - hrp_state.tax_loss_carryforward)
                actual_tax = taxable_gain * params.french_pfu_rate
                hrp_state.tax_loss_carryforward = max(0.0, hrp_state.tax_loss_carryforward - realized_gains_sales)
            else:
                # Realized a net loss, add to carryforward
                hrp_state.tax_loss_carryforward += abs(realized_gains_sales)
                
            hrp_state.total_transaction_costs += actual_tc
            hrp_state.total_taxes_paid += actual_tax
            hrp_state.cash -= (actual_tc + actual_tax)
            
            # Actual purchases second
            for ticker, v in actual_trade_values.items():
                if v > 0:
                    price = prices_at_T_k[ticker]
                    shares_bought = v / price
                    hrp_state.holdings[ticker] = hrp_state.holdings.get(ticker, 0.0) + shares_bought
                    hrp_state.cost_basis[ticker] = hrp_state.cost_basis.get(ticker, 0.0) + v
                    hrp_state.cash -= v
                    
            hrp_events_triggered += 1
            if not is_first:
                average_assets_traded.append(n_traded)
                
            # Record post-rebalance portfolio weights
            total_val = hrp_state.get_value(prices_at_T_k)
            hrp_weights_history[T_k.strftime('%Y-%m-%d')] = {
                ticker: (hrp_state.holdings.get(ticker, 0.0) * prices_at_T_k[ticker]) / total_val if total_val > 0 else 0.0
                for ticker in prices_df.columns
            }
                
        # --- 60/40 REBALANCE ---
        sf_v_prior = sixty_forty_state.get_value(prices_at_T_k)
        
        # Target weights: 60% Equity / 40% Bond
        sf_w_target = {equity_ticker: 0.6, bond_ticker: 0.4}
        
        # Calculate first-pass trades
        sf_trade_values_temp = {}
        for ticker, w_t in sf_w_target.items():
            curr_val = sixty_forty_state.holdings.get(ticker, 0.0) * prices_at_T_k[ticker]
            sf_trade_values_temp[ticker] = sf_v_prior * w_t - curr_val
            
        sf_temp_tc = sum(abs(v) for v in sf_trade_values_temp.values()) * (params.transaction_cost_bps / 10000.0)
        
        # Tax on first-pass sales
        sf_temp_gains = 0.0
        for ticker, v in sf_trade_values_temp.items():
            if v < 0:
                sold_val = -v
                old_shares = sixty_forty_state.holdings.get(ticker, 0.0)
                if old_shares > 0:
                    old_price = prices_at_T_k[ticker]
                    frac = min(1.0, sold_val / (old_shares * old_price))
                    old_basis = sixty_forty_state.cost_basis.get(ticker, 0.0)
                    sold_basis = old_basis * frac
                    gain = sold_val - sold_basis
                    sf_temp_gains += gain
                    
        sf_temp_tax = 0.0
        if sf_temp_gains > 0:
            sf_taxable_gain = max(0.0, sf_temp_gains - sixty_forty_state.tax_loss_carryforward)
            sf_temp_tax = sf_taxable_gain * params.french_pfu_rate
            
        sf_v_post = sf_v_prior - sf_temp_tc - sf_temp_tax
        
        # Execute actual 60/40 trades
        sf_actual_trade_values = {}
        for ticker, w_t in sf_w_target.items():
            target_val = sf_v_post * w_t
            curr_val = sixty_forty_state.holdings.get(ticker, 0.0) * prices_at_T_k[ticker]
            sf_actual_trade_values[ticker] = target_val - curr_val
            
        # Sales first
        sf_realized_gains_sales = 0.0
        for ticker, v in sf_actual_trade_values.items():
            if v < 0:
                sold_val = -v
                old_shares = sixty_forty_state.holdings.get(ticker, 0.0)
                if old_shares > 0:
                    old_price = prices_at_T_k[ticker]
                    frac = min(1.0, sold_val / (old_shares * old_price))
                    old_basis = sixty_forty_state.cost_basis.get(ticker, 0.0)
                    sold_basis = old_basis * frac
                    gain = sold_val - sold_basis
                    sf_realized_gains_sales += gain
                    
                    # Update
                    sixty_forty_state.holdings[ticker] = old_shares * (1.0 - frac)
                    sixty_forty_state.cost_basis[ticker] = old_basis - sold_basis
                    sixty_forty_state.cash += sold_val
                    
        # Apply tax and costs
        sf_actual_tc = sum(abs(v) for v in sf_actual_trade_values.values()) * (params.transaction_cost_bps / 10000.0)
        sf_actual_tax = 0.0
        if sf_realized_gains_sales > 0:
            sf_taxable_gain = max(0.0, sf_realized_gains_sales - sixty_forty_state.tax_loss_carryforward)
            sf_actual_tax = sf_taxable_gain * params.french_pfu_rate
            sixty_forty_state.tax_loss_carryforward = max(0.0, sixty_forty_state.tax_loss_carryforward - sf_realized_gains_sales)
        else:
            sixty_forty_state.tax_loss_carryforward += abs(sf_realized_gains_sales)
            
        sixty_forty_state.total_transaction_costs += sf_actual_tc
        sixty_forty_state.total_taxes_paid += sf_actual_tax
        sixty_forty_state.cash -= (sf_actual_tc + sf_actual_tax)
        
        # Purchases second
        for ticker, v in sf_actual_trade_values.items():
            if v > 0:
                price = prices_at_T_k[ticker]
                shares_bought = v / price
                sixty_forty_state.holdings[ticker] = sixty_forty_state.holdings.get(ticker, 0.0) + shares_bought
                sixty_forty_state.cost_basis[ticker] = sixty_forty_state.cost_basis.get(ticker, 0.0) + v
                sixty_forty_state.cash -= v
                
        # --- CALCULATE DAILY CUMULATIVE VALUES FOR THE INTERVAL (Vectorized) ---
        # active_holdings index slice to avoid NaNs from unheld columns
        hrp_held = {t: s for t, s in hrp_state.holdings.items() if s > 0.0}
        sf_held = {t: s for t, s in sixty_forty_state.holdings.items() if s > 0.0}
        
        prices_interval = prices_clean.loc[interval_dates]
        
        # Vectorized dot products
        if hrp_held:
            hrp_tickers = list(hrp_held.keys())
            hrp_shares_arr = np.array([hrp_held[t] for t in hrp_tickers])
            hrp_equity.loc[interval_dates] = prices_interval[hrp_tickers].values.dot(hrp_shares_arr) + hrp_state.cash
        else:
            hrp_equity.loc[interval_dates] = hrp_state.cash
            
        if sf_held:
            sf_tickers = list(sf_held.keys())
            sf_shares_arr = np.array([sf_held[t] for t in sf_tickers])
            sixty_forty_equity.loc[interval_dates] = prices_interval[sf_tickers].values.dot(sf_shares_arr) + sixty_forty_state.cash
        else:
            sixty_forty_equity.loc[interval_dates] = sixty_forty_state.cash
            
        # Track daily operational costs
        hrp_tc_series.loc[interval_dates] = hrp_state.total_transaction_costs
        hrp_tax_series.loc[interval_dates] = hrp_state.total_taxes_paid
        hrp_tlc_series.loc[interval_dates] = hrp_state.tax_loss_carryforward
        
        sf_tc_series.loc[interval_dates] = sixty_forty_state.total_transaction_costs
        sf_tax_series.loc[interval_dates] = sixty_forty_state.total_taxes_paid
        sf_tlc_series.loc[interval_dates] = sixty_forty_state.tax_loss_carryforward
            
        # S&P 500 B&H daily tracking
        sp500_equity.loc[interval_dates] = prices_interval[equity_ticker].values * sp500_shares
        
    # Scale all equities to start at 1.0 (cumulative return series)
    hrp_cum = hrp_equity / initial_capital
    sp500_cum = sp500_equity / initial_capital
    sixty_forty_cum = sixty_forty_equity / initial_capital
    
    # Calculate tax drag as a percentage of total returns
    hrp_total_return = hrp_equity.iloc[-1] - initial_capital
    if hrp_total_return > 0:
        tax_drag_pct = hrp_state.total_taxes_paid / hrp_total_return
    else:
        tax_drag_pct = 0.0
        
    hrp_diagnostics = {
        "total_friction_costs_paid": hrp_state.total_transaction_costs,
        "total_pfu_taxes_paid": hrp_state.total_taxes_paid,
        "remaining_tax_loss_carryforward": hrp_state.tax_loss_carryforward,
        "total_rebalance_events": hrp_events_triggered,
        "tax_drag_percentage_of_returns": tax_drag_pct,
        "average_assets_traded_per_event": np.mean(average_assets_traded) if len(average_assets_traded) > 0 else 0.0,
        "weight_history": hrp_weights_history,
        "daily_tc": hrp_tc_series,
        "daily_tax": hrp_tax_series,
        "daily_carryforward": hrp_tlc_series,
        "trade_dates": hrp_trade_dates
    }
    
    sixty_forty_diagnostics = {
        "total_friction_costs_paid": sixty_forty_state.total_transaction_costs,
        "total_pfu_taxes_paid": sixty_forty_state.total_taxes_paid,
        "remaining_tax_loss_carryforward": sixty_forty_state.tax_loss_carryforward,
        "daily_tc": sf_tc_series,
        "daily_tax": sf_tax_series,
        "daily_carryforward": sf_tlc_series
    }
    
    return hrp_cum, hrp_diagnostics, sp500_cum, sixty_forty_cum, sixty_forty_diagnostics
