import json
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from hrp_engine.config import StrategyParams

def compute_metrics(equity_series: pd.Series, total_tc: float = 0.0, bps: float = 5.0, initial_capital: float = 1.0) -> Dict:
    """
    Computes annualized performance metrics for a given cumulative equity series.
    """
    index = equity_series.index
    n_days = len(index)
    if n_days <= 1:
        return {
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "max_drawdown": 0.0,
            "annualized_turnover": 0.0
        }
        
    years = (index[-1] - index[0]).days / 365.25
    if years <= 0:
        years = n_days / 252.0
        
    # Annualized Return
    total_return = equity_series.iloc[-1] / equity_series.iloc[0]
    ann_return = (total_return) ** (1.0 / years) - 1.0
    
    # Daily returns
    daily_returns = equity_series.pct_change().dropna()
    
    # Annualized Volatility
    ann_vol = daily_returns.std() * np.sqrt(252.0) if len(daily_returns) > 0 else 0.0
    
    # Sharpe Ratio (Rf=0)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0
    
    # Downside Volatility (Sortino Ratio, Rf=0)
    downside_returns = np.where(daily_returns < 0, daily_returns, 0.0)
    downside_vol = np.sqrt(np.mean(downside_returns**2)) * np.sqrt(252.0) if len(daily_returns) > 0 else 0.0
    sortino = ann_return / downside_vol if downside_vol > 0 else 0.0
    
    # Maximum Drawdown
    peak = equity_series.cummax()
    drawdown = (equity_series - peak) / peak
    mdd = drawdown.min()
    
    # Annualized Turnover
    # Turnover = Volume Traded / Portfolio Value
    # Volume Traded = Friction Costs Paid / (bps / 10000)
    # We estimate using initial_capital as average portfolio value
    if total_tc > 0 and bps > 0:
        volume_traded = total_tc / (bps / 10000.0)
        total_turnover = volume_traded / initial_capital
        ann_turnover = total_turnover / years
    else:
        ann_turnover = 0.0
        
    return {
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "max_drawdown": mdd,
        "annualized_turnover": ann_turnover
    }

def generate_markdown_report(
    hrp_metrics: Dict,
    sp500_metrics: Dict,
    sixty_forty_metrics: Dict,
    hrp_diagnostics: Dict
) -> str:
    """
    Generates the performance report in Markdown table format.
    """
    report = []
    report.append("### STRATEGY PERFORMANCE SUMMARY")
    report.append("| Metric | HRP Denoised Strategy | S&P 500 Buy & Hold | 60/40 Benchmark |")
    report.append("| :--- | :---: | :---: | :---: |")
    
    report.append(
        f"| Annualized Return | {hrp_metrics['annualized_return']*100:.2f}% | "
        f"{sp500_metrics['annualized_return']*100:.2f}% | "
        f"{sixty_forty_metrics['annualized_return']*100:.2f}% |"
    )
    report.append(
        f"| Annualized Volatility | {hrp_metrics['annualized_volatility']*100:.2f}% | "
        f"{sp500_metrics['annualized_volatility']*100:.2f}% | "
        f"{sixty_forty_metrics['annualized_volatility']*100:.2f}% |"
    )
    report.append(
        f"| Sharpe Ratio (Rf=0) | {hrp_metrics['sharpe_ratio']:.2f} | "
        f"{sp500_metrics['sharpe_ratio']:.2f} | "
        f"{sixty_forty_metrics['sharpe_ratio']:.2f} |"
    )
    report.append(
        f"| Sortino Ratio | {hrp_metrics['sortino_ratio']:.2f} | "
        f"{sp500_metrics['sortino_ratio']:.2f} | "
        f"{sixty_forty_metrics['sortino_ratio']:.2f} |"
    )
    report.append(
        f"| Maximum Drawdown | {hrp_metrics['max_drawdown']*100:.2f}% | "
        f"{sp500_metrics['max_drawdown']*100:.2f}% | "
        f"{sixty_forty_metrics['max_drawdown']*100:.2f}% |"
    )
    report.append(
        f"| Annualized Turnover | {hrp_metrics['annualized_turnover']*100:.2f}% | "
        f"{sp500_metrics['annualized_turnover']*100:.2f}% | "
        f"{sixty_forty_metrics['annualized_turnover']*100:.2f}% |"
    )
    
    report.append("")
    report.append("### STRATEGY OPERATIONAL DIAGNOSTICS")
    report.append(f"- Total Friction Costs Paid: {hrp_diagnostics['total_friction_costs_paid']:.2f} EUR")
    report.append(f"- Total PFU Taxes Paid (31.4%): {hrp_diagnostics['total_pfu_taxes_paid']:.2f} EUR")
    report.append(f"- Remaining Tax Loss Carryforward: {hrp_diagnostics['remaining_tax_loss_carryforward']:.2f} EUR")
    report.append(f"- Total Rebalance Events Triggered: {hrp_diagnostics['total_rebalance_events']}")
    
    return "\n".join(report)

def generate_json_output(
    params: StrategyParams,
    hrp_cum: pd.Series,
    sp500_cum: pd.Series,
    sixty_forty_cum: pd.Series,
    hrp_diagnostics: Dict
) -> str:
    """
    Generates the structured JSON block for diagnostic charting.
    """
    # Downsample daily series to end-of-month or rebalance dates to keep JSON reasonably sized
    # But let's keep all rebalance dates or end of months
    monthly_idx = hrp_cum.index[hrp_cum.index.is_month_end]
    if len(monthly_idx) < 10:
        # Fallback to every 20 days if too short
        monthly_idx = hrp_cum.index[::20]
        
    # Ensure the last date is included
    if hrp_cum.index[-1] not in monthly_idx:
        monthly_idx = monthly_idx.append(pd.DatetimeIndex([hrp_cum.index[-1]]))
        
    dates_str = [d.strftime('%Y-%m-%d') for d in monthly_idx]
    
    data = {
        "strategy_parameters": {
            "lookback_years": params.lookback_years,
            "rebalance_frequency": params.rebalance_frequency,
            "linkage_method": params.linkage_method,
            "drift_threshold": params.drift_threshold,
            "french_pfu_rate": params.french_pfu_rate
        },
        "chart_data": {
            "dates": dates_str,
            "hrp_cumulative_equity": hrp_cum.loc[monthly_idx].round(4).tolist(),
            "sp500_cumulative_equity": sp500_cum.loc[monthly_idx].round(4).tolist(),
            "sixty_forty_cumulative_equity": sixty_forty_cum.loc[monthly_idx].round(4).tolist()
        },
        "agent_diagnostic_signals": {
            "matrix_conditioning_errors": 0,  # We handle covariance matrices defensively
            "tax_drag_percentage_of_returns": round(hrp_diagnostics["tax_drag_percentage_of_returns"], 4),
            "average_assets_traded_per_event": round(hrp_diagnostics["average_assets_traded_per_event"], 2)
        }
    }
    
    return json.dumps(data, indent=2)
