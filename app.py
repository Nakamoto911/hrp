import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from hrp_engine.config import StrategyParams
from dataclasses import replace
from hrp_engine.data import fetch_data, get_lookback_data, get_least_history_info
from hrp_engine.backtest import run_strategy_backtest, get_rebalance_dates
from hrp_engine.denoiser import denoise_covariance
from hrp_engine.reporting import compute_metrics
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
import plotly.figure_factory as ff

# Set Page Config
st.set_page_config(
    page_title="HRP Engine Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Asset Name Mapping
ASSET_NAMES = {
    # ETFs
    'IVV': 'US Large Cap (IVV)',
    'IJH': 'US Mid Cap (IJH)',
    'IWM': 'US Small Cap (IWM)',
    'EFA': 'Developed Markets (EFA)',
    'EEM': 'Emerging Markets (EEM)',
    'AGG': 'US Aggregate Bonds (AGG)',
    'SPTL': 'US Long-Term Treasuries (SPTL)',
    'HYG': 'US High Yield Bonds (HYG)',
    'SPBO': 'US Corporate Bonds (SPBO)',
    'IYR': 'US Real Estate (IYR)',
    'DBC': 'Commodities (DBC)',
    'GLD': 'Gold (GLD)',
    # Mutual Funds
    '^SP500TR': 'US Large Cap (^SP500TR)',
    'VIMSX': 'US Mid Cap (VIMSX)',
    'NAESX': 'US Small Cap (NAESX)',
    'FDIVX': 'Developed Markets (FDIVX)',
    'VEIEX': 'Emerging Markets (VEIEX)',
    'VBMFX': 'US Aggregate Bonds (VBMFX)',
    'VUSTX': 'US Long-Term Treasuries (VUSTX)',
    'VWEHX': 'US High Yield Bonds (VWEHX)',
    'VWESX': 'US Corporate Bonds (VWESX)',
    'FRESX': 'US Real Estate (FRESX)',
    '^SPGSCI': 'Commodities (^SPGSCI)',
    'GC=F': 'Gold (GC=F)',
    # European Investable Pool
    'SXR8.DE': 'S&P 500 (SXR8.DE)',
    'EXX5.DE': 'Nikkei 225 (EXX5.DE)',
    'EXS1.DE': 'FTSE 100 (EXS1.DE)',
    'SXRT.DE': 'EuroSTOXX 50 (SXRT.DE)',
    'IS3N.DE': 'MSCI EM (IS3N.DE)',
    'IUSM.DE': 'US 10Y Treasuries (IUSM.DE)',
    'IS0L.DE': 'Bund Allemand (IS0L.DE)',
    'XJSE.DE': 'JGB Japonais (XJSE.DE)',
    'IGLT.L': 'U.K. Gilts (IGLT.L)',
    'IBCQ.DE': 'Credit IG (IBCQ.DE)',
    'IHYG.MI': 'Credit HY (IHYG.MI)',
    '4GLD.DE': 'Gold (4GLD.DE)',
    'CRUD.MI': 'Oil (CRUD.MI)',
    'COPA.MI': 'Copper (COPA.MI)',
    'AIGP.MI': 'Agriculture (AIGP.MI)',
    '5MVW.DE': 'Equity Value (5MVW.DE)',
    '36BZ.DE': 'Equity Momentum (36BZ.DE)',
    'QDVB.DE': 'Equity Quality (QDVB.DE)',
    'IUS3.DE': 'Equity Defensive (IUS3.DE)',
    'BTCE.DE': 'Bitcoin (BTCE.DE)'
}

# Apply Rich Custom CSS for styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #FF4B4B 0%, #FF8F8F 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #888888;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background-color: #1e2129;
        border: 1px solid #2d3139;
        border-radius: 12px;
        padding: 1.25rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #FF4B4B;
    }
    
    .metric-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        color: #888888;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffffff;
    }
    
    .metric-delta {
        font-size: 0.9rem;
        margin-top: 0.25rem;
    }
    
    .delta-green {
        color: #00E676;
    }
    
    .delta-red {
        color: #FF1744;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to compute drawdown series
def get_drawdown(series):
    peak = series.cummax()
    return (series - peak) / peak

# Helper function to compute yearly indicators
def compute_yearly_indicators(
    hrp_cum_at, hrp_cum_bt, sp500_cum, sixty_forty_cum_at, sixty_forty_cum_bt,
    hrp_diag_at, sf_diag_at, initial_capital
):
    years = sorted(list(set(hrp_cum_at.index.year)))
    
    # Calculate daily returns
    hrp_daily_at = hrp_cum_at.pct_change().fillna(0.0)
    hrp_daily_bt = hrp_cum_bt.pct_change().fillna(0.0)
    sp500_daily = sp500_cum.pct_change().fillna(0.0)
    sf_daily_at = sixty_forty_cum_at.pct_change().fillna(0.0)
    sf_daily_bt = sixty_forty_cum_bt.pct_change().fillna(0.0)
    
    yearly_rows = []
    
    for y in years:
        y_str = str(y)
        
        # Get data slices for this year
        hrp_y_at = hrp_daily_at.loc[y_str]
        hrp_y_bt = hrp_daily_bt.loc[y_str]
        sp500_y = sp500_daily.loc[y_str]
        sf_y_at = sf_daily_at.loc[y_str]
        sf_y_bt = sf_daily_bt.loc[y_str]
        
        # Returns
        hrp_ret_at = (1 + hrp_y_at).prod() - 1.0
        hrp_ret_bt = (1 + hrp_y_bt).prod() - 1.0
        sp500_ret = (1 + sp500_y).prod() - 1.0
        sf_ret_at = (1 + sf_y_at).prod() - 1.0
        sf_ret_bt = (1 + sf_y_bt).prod() - 1.0
        
        # Volatility (After-Tax)
        hrp_vol = hrp_y_at.std() * np.sqrt(252.0) if len(hrp_y_at) > 1 else 0.0
        
        # Sharpe (After-Tax)
        hrp_sharpe = hrp_ret_at / hrp_vol if hrp_vol > 0.0 else 0.0
        
        # Max Drawdown (After-Tax)
        hrp_cum_y = hrp_cum_at.loc[y_str]
        peak = hrp_cum_y.cummax()
        dd = (hrp_cum_y - peak) / peak
        hrp_mdd = dd.min()
        
        # Operational Metrics
        tc_all = hrp_diag_at["daily_tc"]
        tax_all = hrp_diag_at["daily_tax"]
        carry_all = hrp_diag_at["daily_carryforward"]
        
        y_dates = hrp_cum_at.index[hrp_cum_at.index.year == y]
        
        tc_end = tc_all.loc[y_dates[-1]]
        prev_dates = tc_all.index[tc_all.index < y_dates[0]]
        tc_start = tc_all.loc[prev_dates[-1]] if len(prev_dates) > 0 else 0.0
        tc_paid = tc_end - tc_start
        
        tax_end = tax_all.loc[y_dates[-1]]
        tax_start = tax_all.loc[prev_dates[-1]] if len(prev_dates) > 0 else 0.0
        tax_paid = tax_end - tax_start
        
        carryforward = carry_all.loc[y_dates[-1]] if len(y_dates) > 0 else 0.0
        
        trade_dates_in_year = [d for d in hrp_diag_at["trade_dates"] if d.year == y]
        rebalances = len(trade_dates_in_year)
        
        yearly_rows.append({
            "Year": y,
            "HRP Return (Before-Tax)": f"{hrp_ret_bt * 100.0:.2f}%",
            "HRP Return (After-Tax)": f"{hrp_ret_at * 100.0:.2f}%",
            "S&P 500 Return": f"{sp500_ret * 100.0:.2f}%",
            "60/40 Return (After-Tax)": f"{sf_ret_at * 100.0:.2f}%",
            "HRP Volatility (After-Tax)": f"{hrp_vol * 100.0:.2f}%",
            "HRP Sharpe (After-Tax)": f"{hrp_sharpe:.2f}",
            "HRP Max Drawdown": f"{hrp_mdd * 100.0:.2f}%",
            "PFU Tax Paid": f"€{tax_paid:,.2f}",
            "Tax Carryforward": f"€{carryforward:,.2f}",
            "Friction Costs Paid": f"€{tc_paid:,.2f}",
            "Rebalances": rebalances
        })
        
    return pd.DataFrame(yearly_rows).set_index("Year")

@st.cache_data
def precompute_all_rebalance_diagnostics(prices_df: pd.DataFrame, params: StrategyParams, pool_name: str):
    """
    Precomputes and caches diagnostics for all valid rebalance dates in one go.
    """
    prices_clean = prices_df.ffill().bfill()
    all_rebalance_dates = get_rebalance_dates(prices_clean.index, params.rebalance_frequency)
    start_date = prices_clean.index[0]
    valid_rebalance_dates = [
        d for d in all_rebalance_dates 
        if d >= start_date + pd.DateOffset(years=params.lookback_years)
    ]
    
    results = {}
    for d in valid_rebalance_dates:
        diag = get_rebalance_diagnostics(prices_df, params, pool_name, d)
        if diag is not None:
            results[d] = diag
    return results

def get_rebalance_diagnostics(prices_df: pd.DataFrame, params: StrategyParams, pool_name: str, selected_date=None):
    """
    Computes the covariance, denoised covariance, linkage matrix, and weights
    for a specific rebalance date to visualize step-by-step impact.
    """
    prices_clean = prices_df.ffill().bfill()
    all_rebalance_dates = get_rebalance_dates(prices_clean.index, params.rebalance_frequency)
    start_date = prices_clean.index[0]
    valid_rebalance_dates = [
        d for d in all_rebalance_dates 
        if d >= start_date + pd.DateOffset(years=params.lookback_years)
    ]
    if not valid_rebalance_dates:
        return None
    
    if selected_date is None or selected_date not in valid_rebalance_dates:
        selected_date = valid_rebalance_dates[-1]
    
    # Get lookback returns and active assets
    lookback_returns, active_assets = get_lookback_data(prices_df, selected_date, params.lookback_years)
    
    # Empirical covariance and correlation
    cov_emp = lookback_returns.cov().values
    n_obs = len(lookback_returns)
    N = len(active_assets)
    
    std = np.sqrt(np.diag(cov_emp))
    std_safe = np.where(std == 0, 1e-8, std)
    corr_emp = cov_emp / np.outer(std_safe, std_safe)
    corr_emp = np.clip(corr_emp, -1.0, 1.0)
    
    # Eigen decomposition for RMT
    eigenvalues, eigenvectors = np.linalg.eigh(corr_emp)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # RMT denoised covariance and correlation
    cov_denoised = denoise_covariance(cov_emp, n_obs)
    std_denoised = np.sqrt(np.diag(cov_denoised))
    std_denoised_safe = np.where(std_denoised == 0, 1e-8, std_denoised)
    corr_denoised = cov_denoised / np.outer(std_denoised_safe, std_denoised_safe)
    corr_denoised = np.clip(corr_denoised, -1.0, 1.0)
    
    # Denoised eigenvalues
    eigenvalues_denoised, _ = np.linalg.eigh(corr_denoised)
    eigenvalues_denoised = np.sort(eigenvalues_denoised)[::-1]
    
    # Noise boundary lambda_max
    eigenvalues_lte_1 = eigenvalues[eigenvalues <= 1.0]
    sigma2 = np.var(eigenvalues_lte_1) if len(eigenvalues_lte_1) > 0 else 1.0
    if sigma2 <= 0:
        sigma2 = 1e-8
    q = N / n_obs
    lambda_max = sigma2 * (1.0 + np.sqrt(q))**2
    
    # Dendrogram calculations
    dist = np.sqrt(2.0 * np.clip(1.0 - corr_denoised, 0.0, 2.0))
    dist_cols = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            d = np.sqrt(np.sum((dist[i] - dist[j])**2))
            dist_cols[i, j] = d
            dist_cols[j, i] = d
    condensed_dist = squareform(dist_cols)
    Z = sch.linkage(condensed_dist, method=params.linkage_method)
    
    # Leaf ordering
    from hrp_engine.hrp import get_quasi_diag, recursive_bisection
    ordered_indices = get_quasi_diag(Z)
    sort_items = [active_assets[i] for i in ordered_indices]
    
    # Weights comparison
    # 1. HRP weights
    cov_denoised_df = pd.DataFrame(cov_denoised, index=active_assets, columns=active_assets)
    w_hrp = recursive_bisection(cov_denoised_df, sort_items)
    w_hrp = w_hrp.reindex(active_assets)
    
    # 2. Inverse-Variance weights (on denoised cov)
    inv_var = 1.0 / std_denoised_safe**2
    w_inv_var = pd.Series(inv_var / inv_var.sum(), index=active_assets)
    
    # 3. Equal weights
    w_equal = pd.Series(1.0 / N, index=active_assets)
    
    return {
        "date": selected_date,
        "assets": active_assets,
        "cov_emp": pd.DataFrame(cov_emp, index=active_assets, columns=active_assets),
        "cov_denoised": cov_denoised_df,
        "corr_emp": pd.DataFrame(corr_emp, index=active_assets, columns=active_assets),
        "corr_denoised": pd.DataFrame(corr_denoised, index=active_assets, columns=active_assets),
        "eigenvalues": eigenvalues,
        "eigenvalues_denoised": eigenvalues_denoised,
        "lambda_max": lambda_max,
        "linkage": Z,
        "w_hrp": w_hrp,
        "w_inv_var": w_inv_var,
        "w_equal": w_equal,
        "ordered_assets": sort_items
    }

# Helper functions for parameter sensitivity sweeps
def run_rebalance_frequency_sweep(prices_df, base_params, pool_name, initial_capital):
    frequencies = ["weekly", "monthly", "quarterly", "yearly"]
    rows = []
    for freq in frequencies:
        params_sweep = replace(base_params, rebalance_frequency=freq)
        try:
            hrp_cum, hrp_diag, _, _, _ = run_strategy_backtest(prices_df, params_sweep, pool_name, initial_capital)
            metrics = compute_metrics(hrp_cum)
            final_wealth = initial_capital * hrp_cum.iloc[-1]
            rows.append({
                "Rebalance Frequency": freq.capitalize(),
                "Final Wealth (Net)": f"€{final_wealth:,.2f}",
                "Ann. Return (Net)": f"{metrics['annualized_return']*100:.2f}%",
                "Ann. Volatility": f"{metrics['annualized_volatility']*100:.2f}%",
                "Sharpe (Net)": f"{metrics['sharpe_ratio']:.2f}",
                "Sortino (Net)": f"{metrics['sortino_ratio']:.2f}",
                "Max Drawdown": f"{metrics['max_drawdown']*100:.2f}%",
                "Turnover": f"{metrics['annualized_turnover']*100:.2f}%",
                "PFU Taxes Paid": f"€{hrp_diag['total_pfu_taxes_paid']:,.2f}"
            })
        except Exception as e:
            rows.append({"Rebalance Frequency": freq.capitalize(), "Error": str(e)})
    return pd.DataFrame(rows).set_index("Rebalance Frequency")

def run_linkage_sweep(prices_df, base_params, pool_name, initial_capital):
    linkage_methods = ["single", "complete", "average", "ward"]
    rows = []
    for linkage in linkage_methods:
        params_sweep = replace(base_params, linkage_method=linkage)
        try:
            hrp_cum, hrp_diag, _, _, _ = run_strategy_backtest(prices_df, params_sweep, pool_name, initial_capital)
            metrics = compute_metrics(hrp_cum)
            final_wealth = initial_capital * hrp_cum.iloc[-1]
            rows.append({
                "Clustering Linkage": linkage.capitalize(),
                "Final Wealth (Net)": f"€{final_wealth:,.2f}",
                "Ann. Return (Net)": f"{metrics['annualized_return']*100:.2f}%",
                "Ann. Volatility": f"{metrics['annualized_volatility']*100:.2f}%",
                "Sharpe (Net)": f"{metrics['sharpe_ratio']:.2f}",
                "Sortino (Net)": f"{metrics['sortino_ratio']:.2f}",
                "Max Drawdown": f"{metrics['max_drawdown']*100:.2f}%",
                "Turnover": f"{metrics['annualized_turnover']*100:.2f}%",
                "PFU Taxes Paid": f"€{hrp_diag['total_pfu_taxes_paid']:,.2f}"
            })
        except Exception as e:
            rows.append({"Clustering Linkage": linkage.capitalize(), "Error": str(e)})
    return pd.DataFrame(rows).set_index("Clustering Linkage")

def run_drift_band_sweep(prices_df, base_params, pool_name, initial_capital):
    drift_thresholds = [0.0, 0.01, 0.025, 0.05, 0.10]
    rows = []
    for threshold in drift_thresholds:
        params_sweep = replace(base_params, drift_threshold=threshold)
        try:
            hrp_cum, hrp_diag, _, _, _ = run_strategy_backtest(prices_df, params_sweep, pool_name, initial_capital)
            metrics = compute_metrics(hrp_cum)
            final_wealth = initial_capital * hrp_cum.iloc[-1]
            rows.append({
                "Drift Band Threshold": f"{threshold * 100.0:.1f}%",
                "Final Wealth (Net)": f"€{final_wealth:,.2f}",
                "Ann. Return (Net)": f"{metrics['annualized_return']*100:.2f}%",
                "Ann. Volatility": f"{metrics['annualized_volatility']*100:.2f}%",
                "Sharpe (Net)": f"{metrics['sharpe_ratio']:.2f}",
                "Sortino (Net)": f"{metrics['sortino_ratio']:.2f}",
                "Max Drawdown": f"{metrics['max_drawdown']*100:.2f}%",
                "Turnover": f"{metrics['annualized_turnover']*100:.2f}%",
                "PFU Taxes Paid": f"€{hrp_diag['total_pfu_taxes_paid']:,.2f}"
            })
        except Exception as e:
            rows.append({"Drift Band Threshold": f"{threshold * 100.0:.1f}%", "Error": str(e)})
    return pd.DataFrame(rows).set_index("Drift Band Threshold")

# Load Data (Cached to avoid yfinance rate limits and slow reads)
@st.cache_data(show_spinner=False)
def load_prices(pool_name: str) -> pd.DataFrame:
    return fetch_data(pool_name)

# --- SIDEBAR PARAMETERS ---
st.sidebar.markdown("## ⚙️ Configuration Parameters")

pool_selection = st.sidebar.selectbox(
    "Asset Pool",
    options=["ETF", "Mutual Fund", "European Investable"],
    index=0,
    help="Select the asset pool dataset."
)

limit_history = st.sidebar.checkbox(
    "Limit Time Range to Common Inception",
    value=False,
    help="If checked, the simulation starts at the inception date of the asset with the least history, ensuring complete history for all assets without backward-filling."
)

st.sidebar.markdown("---")

lookback_years = st.sidebar.slider(
    "Lookback Window (Years)",
    min_value=1,
    max_value=10,
    value=4,
    step=1,
    help="Size of the rolling window used to estimate asset covariances."
)

rebalance_frequency = st.sidebar.selectbox(
    "Rebalance Frequency",
    options=["monthly", "quarterly", "yearly"],
    index=1,
    help="Frequency of portfolio reoptimizations."
)

linkage_method = st.sidebar.selectbox(
    "Clustering Linkage Method",
    options=["single", "complete", "average", "ward"],
    index=0,
    help="The linkage criteria for hierarchical clustering of assets."
)

drift_threshold_pct = st.sidebar.slider(
    "Drift Band Threshold (%)",
    min_value=0.0,
    max_value=5.0,
    value=1.5,
    step=0.1,
    help="Minimum change in target weight required to trigger a trade."
)

transaction_cost_bps = st.sidebar.slider(
    "Transaction Cost (bps)",
    min_value=0,
    max_value=100,
    value=5,
    step=1,
    help="Execution fee deducted on sales and purchases (1 bps = 0.01%)."
)

french_pfu_rate_pct = st.sidebar.slider(
    "French PFU Tax Rate (%)",
    min_value=0.0,
    max_value=50.0,
    value=31.4,
    step=0.1,
    help="Flat tax rate applied to realized net capital gains."
)

initial_capital = st.sidebar.number_input(
    "Initial Capital (EUR)",
    min_value=1000,
    max_value=10000000,
    value=100000,
    step=5000,
    help="Starting capital for the portfolio simulation."
)

# --- MAIN PAGE DISPLAY ---
st.markdown('<div class="main-title">Hierarchical Risk Parity allocation Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Walk-forward multi-asset engine with denoised covariance & tax modeling</div>', unsafe_allow_html=True)

# Fetch data based on selection
if pool_selection == "ETF":
    pool_key = 'etf'
elif pool_selection == "European Investable":
    pool_key = 'european'
else:
    pool_key = 'mutual_fund'
try:
    prices_df = load_prices(pool_key)
except Exception as e:
    st.error(f"Error loading price data: {e}")
    st.stop()

# Find the start date of the asset with the least history
start_date_least_history, least_history_asset = get_least_history_info(prices_df)
asset_display_name = ASSET_NAMES.get(least_history_asset, least_history_asset)

# Slice prices_df if checkbox is checked
if limit_history:
    prices_df = prices_df.loc[start_date_least_history:]

# Display the start date in a premium styled container
if limit_history:
    st.markdown(f"""
    <div style="background-color: #1e2129; border: 1px solid #2d3139; border-radius: 12px; padding: 1rem; margin-bottom: 2rem; border-left: 5px solid #FF4B4B;">
        <span style="font-weight: 600; color: #ffffff; font-size: 1rem;">📅 Simulation Time Range (Aligned):</span>
        <span style="color: #888888; font-size: 1rem; margin-left: 0.5rem;">
            Starts on <strong>{start_date_least_history.strftime('%B %d, %Y')}</strong> 
            (limited by the inception date of <strong>{asset_display_name}</strong> to ensure complete history for all assets).
        </span>
    </div>
    """, unsafe_allow_html=True)
else:
    dataset_start_date = prices_df.index[0]
    st.markdown(f"""
    <div style="background-color: #1e2129; border: 1px solid #2d3139; border-radius: 12px; padding: 1rem; margin-bottom: 2rem; border-left: 5px solid #2196F3;">
        <span style="font-weight: 600; color: #ffffff; font-size: 1rem;">📅 Simulation Time Range (Full):</span>
        <span style="color: #888888; font-size: 1rem; margin-left: 0.5rem;">
            Starts on <strong>{dataset_start_date.strftime('%B %d, %Y')}</strong>. 
            (Assets with shorter history are backward-filled. Toggle 'Limit Time Range' in the sidebar to align at common inception).
        </span>
    </div>
    """, unsafe_allow_html=True)

# Instantiate StrategyParams for After-Tax
params_at = StrategyParams(
    lookback_years=lookback_years,
    rebalance_frequency=rebalance_frequency,
    linkage_method=linkage_method,
    drift_threshold=drift_threshold_pct / 100.0,
    transaction_cost_bps=transaction_cost_bps,
    french_pfu_rate=french_pfu_rate_pct / 100.0
)

# Instantiate StrategyParams for Before-Tax
params_bt = StrategyParams(
    lookback_years=lookback_years,
    rebalance_frequency=rebalance_frequency,
    linkage_method=linkage_method,
    drift_threshold=drift_threshold_pct / 100.0,
    transaction_cost_bps=transaction_cost_bps,
    french_pfu_rate=0.0  # Set tax to 0.0
)

# Run Backtests
with st.spinner("Running walk-forward backtests..."):
    # 1. Run After-Tax
    hrp_cum_at, hrp_diag_at, sp500_cum_at, sixty_forty_cum_at, sf_diag_at = run_strategy_backtest(
        prices_df, params_at, pool_key, initial_capital, limit_to_least_history=limit_history
    )
    
    # 2. Run Before-Tax (skip if user-selected tax is 0% to keep near-instantaneous)
    if french_pfu_rate_pct > 0.0:
        hrp_cum_bt, hrp_diag_bt, sp500_cum_bt, sixty_forty_cum_bt, sf_diag_bt = run_strategy_backtest(
            prices_df, params_bt, pool_key, initial_capital, limit_to_least_history=limit_history
        )
    else:
        hrp_cum_bt, hrp_diag_bt, sp500_cum_bt, sixty_forty_cum_bt, sf_diag_bt = (
            hrp_cum_at, hrp_diag_at, sp500_cum_at, sixty_forty_cum_at, sf_diag_at
        )

# Compute performance metrics (After-Tax)
hrp_metrics_at = compute_metrics(hrp_cum_at, hrp_diag_at["total_friction_costs_paid"], params_at.transaction_cost_bps, initial_capital)
sp500_metrics_at = compute_metrics(sp500_cum_at, 0.0, 0.0, initial_capital)
sixty_forty_metrics_at = compute_metrics(sixty_forty_cum_at, sf_diag_at["total_friction_costs_paid"], params_at.transaction_cost_bps, initial_capital)

# Compute performance metrics (Before-Tax)
hrp_metrics_bt = compute_metrics(hrp_cum_bt, hrp_diag_bt["total_friction_costs_paid"], params_bt.transaction_cost_bps, initial_capital)
sixty_forty_metrics_bt = compute_metrics(sixty_forty_cum_bt, sf_diag_bt["total_friction_costs_paid"], params_bt.transaction_cost_bps, initial_capital)

# Calculate final portfolio values
hrp_final_wealth_at = initial_capital * hrp_cum_at.iloc[-1]
sp500_final_wealth = initial_capital * sp500_cum_at.iloc[-1]
sixty_forty_final_wealth_at = initial_capital * sixty_forty_cum_at.iloc[-1]

# --- KPI VISUAL CARDS ---
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    # Final Wealth (After-Tax)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Final Wealth (After-Tax)</div>
        <div class="metric-value">€{hrp_final_wealth_at:,.0f}</div>
        <div class="metric-delta">
            SPY: <span class="delta-green">€{sp500_final_wealth:,.0f}</span> | 
            60/40: <span class="delta-green">€{sixty_forty_final_wealth_at:,.0f}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    # Sharpe Ratio (After-Tax)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Sharpe Ratio (Net)</div>
        <div class="metric-value">{hrp_metrics_at['sharpe_ratio']:.2f}</div>
        <div class="metric-delta">
            SPY: <span class="delta-red">{sp500_metrics_at['sharpe_ratio']:.2f}</span> | 
            60/40: <span class="delta-red">{sixty_forty_metrics_at['sharpe_ratio']:.2f}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    # Sortino Ratio (After-Tax)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Sortino Ratio (Net)</div>
        <div class="metric-value">{hrp_metrics_at['sortino_ratio']:.2f}</div>
        <div class="metric-delta">
            SPY: <span class="delta-red">{sp500_metrics_at['sortino_ratio']:.2f}</span> | 
            60/40: <span class="delta-red">{sixty_forty_metrics_at['sortino_ratio']:.2f}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    # Annualized Return (After-Tax)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Ann. Return (Net)</div>
        <div class="metric-value">{hrp_metrics_at['annualized_return']*100:.2f}%</div>
        <div class="metric-delta">
            SPY: <span class="delta-green">{sp500_metrics_at['annualized_return']*100:.2f}%</span> | 
            60/40: <span class="delta-green">{sixty_forty_metrics_at['annualized_return']*100:.2f}%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    # Max Drawdown (After-Tax)
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Max Drawdown</div>
        <div class="metric-value">{hrp_metrics_at['max_drawdown']*100:.2f}%</div>
        <div class="metric-delta">
            SPY: <span class="delta-red">{sp500_metrics_at['max_drawdown']*100:.2f}%</span> | 
            60/40: <span class="delta-red">{sixty_forty_metrics_at['max_drawdown']*100:.2f}%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Pre-compute yearly indicators table
yearly_df = compute_yearly_indicators(
    hrp_cum_at, hrp_cum_bt, sp500_cum_at, sixty_forty_cum_at, sixty_forty_cum_bt,
    hrp_diag_at, sf_diag_at, initial_capital
)

# Create tabs for charts and details
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📈 Cumulative Equity Curves", 
    "📉 Drawdown Profiles", 
    "🍕 Weight Allocations", 
    "🔍 Asset Universe Signals",
    "📊 Performance Table & Diagnostics",
    "📅 Yearly Indicators Breakdown",
    "⚙️ Parameter Sensitivity",
    "🧠 Algorithmic Explainers & Impact"
])

with tab1:
    # Chart 1: Cumulative Equity Curve (Before vs After Tax)
    fig_equity = go.Figure()
    
    # HRP curves
    fig_equity.add_trace(go.Scatter(
        x=hrp_cum_at.index, y=hrp_cum_at, 
        name="HRP Denoised (After Tax)", 
        line=dict(color="#FF4B4B", width=2)
    ))
    if french_pfu_rate_pct > 0.0:
        fig_equity.add_trace(go.Scatter(
            x=hrp_cum_bt.index, y=hrp_cum_bt, 
            name="HRP Denoised (Before Tax)", 
            line=dict(color="#FF8F8F", width=1.5, dash="dash")
        ))
        
    # 60/40 curves
    fig_equity.add_trace(go.Scatter(
        x=sixty_forty_cum_at.index, y=sixty_forty_cum_at, 
        name="60/40 Benchmark (After Tax)", 
        line=dict(color="#4CAF50", width=2)
    ))
    if french_pfu_rate_pct > 0.0:
        fig_equity.add_trace(go.Scatter(
            x=sixty_forty_cum_bt.index, y=sixty_forty_cum_bt, 
            name="60/40 Benchmark (Before Tax)", 
            line=dict(color="#A9DFBF", width=1.5, dash="dash")
        ))
        
    # S&P 500 curve
    fig_equity.add_trace(go.Scatter(
        x=sp500_cum_at.index, y=sp500_cum_at, 
        name="S&P 500 Buy & Hold", 
        line=dict(color="#2196F3", width=2)
    ))
    
    fig_equity.update_layout(
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title="Normalized Value (Base 1.0)",
        title="Equity Growth Curves (Initial Capital Scaled to 1.0)",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    st.plotly_chart(fig_equity, use_container_width=True)

with tab2:
    # Chart 2: Drawdowns over time
    fig_dd = go.Figure()
    
    fig_dd.add_trace(go.Scatter(
        x=hrp_cum_at.index, y=get_drawdown(hrp_cum_at) * 100.0, 
        name="HRP Denoised (After Tax)", 
        line=dict(color="#FF4B4B", width=1.5)
    ))
    fig_dd.add_trace(go.Scatter(
        x=sixty_forty_cum_at.index, y=get_drawdown(sixty_forty_cum_at) * 100.0, 
        name="60/40 Benchmark (After Tax)", 
        line=dict(color="#4CAF50", width=1.5)
    ))
    fig_dd.add_trace(go.Scatter(
        x=sp500_cum_at.index, y=get_drawdown(sp500_cum_at) * 100.0, 
        name="S&P 500 Buy & Hold", 
        line=dict(color="#2196F3", width=1.5)
    ))
    
    fig_dd.update_layout(
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        title="Portfolio Drawdown Profiles (%)",
        hovermode="x unified",
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(yanchor="bottom", y=0.01, xanchor="left", x=0.01)
    )
    st.plotly_chart(fig_dd, use_container_width=True)

with tab3:
    # Chart 3: Weight Allocations
    if "weight_history" in hrp_diag_at and hrp_diag_at["weight_history"]:
        weights_raw_df = pd.DataFrame.from_dict(hrp_diag_at["weight_history"], orient='index')
        weights_raw_df.index = pd.to_datetime(weights_raw_df.index)
        
        # Rename columns to show symbol + asset name
        weights_raw_df = weights_raw_df.rename(columns=ASSET_NAMES)
        
        # Melt dataframe for plotly express
        weights_melted = weights_raw_df.reset_index().rename(columns={'index': 'Date'}).melt(
            id_vars='Date', 
            var_name='Asset Name (Symbol)', 
            value_name='Weight'
        )
        
        # Plot stacked area chart
        fig_weights = px.area(
            weights_melted, 
            x='Date', 
            y='Weight', 
            color='Asset Name (Symbol)',
            title="HRP Dynamic Asset Weight Allocation Over Time",
            color_discrete_sequence=px.colors.qualitative.Plotly
        )
        fig_weights.update_layout(
            template="plotly_dark",
            xaxis_title="Rebalance Date",
            yaxis_title="Weight (1.0 = 100%)",
            legend_title="Asset Universe",
            hovermode="x unified",
            margin=dict(l=40, r=40, t=50, b=40)
        )
        st.plotly_chart(fig_weights, use_container_width=True)
    else:
        st.info("No weight history found in backtest diagnostics.")

with tab4:
    st.markdown("### 🔍 Asset Universe Signals")
    st.markdown("Analyze underlying asset prices, rolling volatility, and rolling yearly returns in raw value or standardized Z-score format.")

    # Controls
    signal_col1, signal_col2, signal_col3 = st.columns(3)
    with signal_col1:
        selected_assets = st.multiselect(
            "Select Assets to Analyze:",
            options=prices_df.columns,
            default=list(prices_df.columns),
            format_func=lambda x: ASSET_NAMES.get(x, x),
            key="signals_assets_select"
        )
    with signal_col2:
        metric_choice = st.radio(
            "Select Metric:",
            options=["Asset Price", "Volatility", "Rolling Yearly Return"],
            horizontal=True,
            key="signals_metric_choice"
        )
    with signal_col3:
        display_mode = st.radio(
            "Display Mode:",
            options=["Value", "Z-Score"],
            horizontal=True,
            key="signals_display_mode"
        )

    # Contextual controls below the main options
    sub_col1, sub_col2 = st.columns(2)
    with sub_col1:
        if metric_choice == "Volatility":
            vol_window = st.slider(
                "Volatility Rolling Window (Business Days):",
                min_value=10,
                max_value=252,
                value=60,
                step=5,
                help="Number of days to compute rolling volatility of daily returns.",
                key="signals_vol_window"
            )
        elif metric_choice == "Asset Price" and display_mode == "Value":
            normalize_price = st.checkbox(
                "Normalize Prices (Base 100)",
                value=True,
                help="Index all assets to start at 100.0 on the first date for easier performance comparison.",
                key="signals_normalize_price"
            )
        else:
            normalize_price = False
    with sub_col2:
        show_ref_lines = st.checkbox(
            "Show Statistical Reference Lines (Mean & ±1 Std)",
            value=True,
            help="Show historical average and standard deviation bands. In Z-Score mode, these are universal (0 and ±1). In Value mode, they are asset-specific.",
            key="signals_show_ref_lines"
        )

    # Filtered and clean prices
    if not selected_assets:
        st.warning("Please select at least one asset to analyze.")
    else:
        # Clean price data
        df_clean = prices_df[selected_assets].ffill().bfill()
        
        # Calculations
        if metric_choice == "Asset Price":
            y_axis_title = "Price"
            if display_mode == "Value":
                if normalize_price:
                    plot_df = (df_clean / df_clean.iloc[0]) * 100.0
                    y_axis_title = "Normalized Price (Base 100)"
                else:
                    plot_df = df_clean
                    y_axis_title = "Raw Price (EUR/Local)"
            else:
                plot_df = (df_clean - df_clean.mean()) / df_clean.std()
                y_axis_title = "Standardized Price (Z-Score)"
                
        elif metric_choice == "Volatility":
            # Returns
            returns_df = df_clean.pct_change()
            # Rolling annualized volatility
            vol_df = returns_df.rolling(window=vol_window).std() * np.sqrt(252)
            y_axis_title = "Annualized Volatility"
            if display_mode == "Value":
                plot_df = vol_df * 100.0  # express as %
                y_axis_title = "Annualized Volatility (%)"
            else:
                plot_df = (vol_df - vol_df.mean()) / vol_df.std()
                y_axis_title = "Standardized Volatility (Z-Score)"
                
        else: # Rolling Yearly Return
            # Rolling 1-year (252 days) return
            ret_df = df_clean.pct_change(periods=252)
            y_axis_title = "Rolling Yearly Return"
            if display_mode == "Value":
                plot_df = ret_df * 100.0  # express as %
                y_axis_title = "Rolling Yearly Return (%)"
            else:
                plot_df = (ret_df - ret_df.mean()) / ret_df.std()
                y_axis_title = "Standardized Rolling Yearly Return (Z-Score)"

        # Generate stacked subplots for each asset
        from plotly.subplots import make_subplots
        
        n_assets = len(selected_assets)
        fig_height = max(400, n_assets * 160)
        
        fig_signals = make_subplots(
            rows=n_assets,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=max(0.01, 0.05 / n_assets),
            subplot_titles=[ASSET_NAMES.get(a, a) for a in selected_assets]
        )
        
        # Color palette definition for clean look
        colors = px.colors.qualitative.Plotly
        for idx, asset in enumerate(selected_assets):
            row = idx + 1
            # Add trace to the specific row
            fig_signals.add_trace(
                go.Scatter(
                    x=plot_df.index,
                    y=plot_df[asset],
                    name=ASSET_NAMES.get(asset, asset),
                    line=dict(width=2, color=colors[idx % len(colors)])
                ),
                row=row,
                col=1
            )
            
            # Draw statistical reference lines for each subplot when checked
            if show_ref_lines:
                if display_mode == "Z-Score":
                    # Universal reference lines at 0, 1, -1
                    fig_signals.add_hline(y=0, line_dash="dash", line_color="#888888", line_width=1.5, row=row, col=1)
                    fig_signals.add_hline(y=1, line_dash="dot", line_color="#FF8F8F", line_width=1, row=row, col=1)
                    fig_signals.add_hline(y=-1, line_dash="dot", line_color="#8F8FFF", line_width=1, row=row, col=1)
                elif display_mode == "Value":
                    series = plot_df[asset]
                    mean_val = series.mean()
                    std_val = series.std()
                    
                    if not pd.isna(mean_val) and not pd.isna(std_val):
                        fig_signals.add_hline(y=mean_val, line_dash="dash", line_color="#888888", line_width=1.5, row=row, col=1)
                        fig_signals.add_hline(y=mean_val + std_val, line_dash="dot", line_color="#FF8F8F", line_width=1, row=row, col=1)
                        fig_signals.add_hline(y=mean_val - std_val, line_dash="dot", line_color="#8F8FFF", line_width=1, row=row, col=1)

        # Style layout
        fig_signals.update_layout(
            template="plotly_dark",
            height=fig_height,
            title=dict(
                text=f"Asset Universe - {metric_choice} ({display_mode} Mode)",
                x=0.5,
                xanchor="center"
            ),
            hovermode="x unified",
            margin=dict(l=40, r=40, t=80, b=40),
            showlegend=False
        )
        
        # Update y-axes titles
        for idx in range(n_assets):
            fig_signals.update_yaxes(title_text=y_axis_title, row=idx+1, col=1)
            
        st.plotly_chart(fig_signals, use_container_width=True)

with tab5:
    # Performance Table Comparison (Before vs After Tax)
    st.markdown("### Performance Comparison Table")
    
    summary_data = {
        "Metric": [
            "Initial Wealth",
            "Final Wealth (Before-Tax)",
            "Final Wealth (After-Tax)",
            "Total Capital Gains Taxes Paid",
            "Annualized Return (Before-Tax)",
            "Annualized Return (After-Tax)",
            "Annualized Volatility (After-Tax)",
            "Sharpe Ratio (After-Tax)",
            "Sortino Ratio (After-Tax)",
            "Maximum Drawdown (After-Tax)",
            "Annualized Turnover"
        ],
        "HRP Denoised Strategy": [
            f"€{initial_capital:,.2f}",
            f"€{initial_capital * hrp_cum_bt.iloc[-1]:,.2f}",
            f"€{initial_capital * hrp_cum_at.iloc[-1]:,.2f}",
            f"€{hrp_diag_at['total_pfu_taxes_paid']:,.2f}",
            f"{hrp_metrics_bt['annualized_return']*100:.2f}%",
            f"{hrp_metrics_at['annualized_return']*100:.2f}%",
            f"{hrp_metrics_at['annualized_volatility']*100:.2f}%",
            f"{hrp_metrics_at['sharpe_ratio']:.2f}",
            f"{hrp_metrics_at['sortino_ratio']:.2f}",
            f"{hrp_metrics_at['max_drawdown']*100:.2f}%",
            f"{hrp_metrics_at['annualized_turnover']*100:.2f}%"
        ],
        "S&P 500 Buy & Hold": [
            f"€{initial_capital:,.2f}",
            f"€{sp500_final_wealth:,.2f}",
            f"€{sp500_final_wealth:,.2f}",  # No rebalances = no taxes paid during period
            "€0.00",
            f"{sp500_metrics_at['annualized_return']*100:.2f}%",
            f"{sp500_metrics_at['annualized_return']*100:.2f}%",
            f"{sp500_metrics_at['annualized_volatility']*100:.2f}%",
            f"{sp500_metrics_at['sharpe_ratio']:.2f}",
            f"{sp500_metrics_at['sortino_ratio']:.2f}",
            f"{sp500_metrics_at['max_drawdown']*100:.2f}%",
            "0.00%"
        ],
        "60/40 Equity/Bond": [
            f"€{initial_capital:,.2f}",
            f"€{initial_capital * sixty_forty_cum_bt.iloc[-1]:,.2f}",
            f"€{sixty_forty_final_wealth_at:,.2f}",
            f"€{sf_diag_at['total_pfu_taxes_paid']:,.2f}",
            f"{sixty_forty_metrics_bt['annualized_return']*100:.2f}%",
            f"{sixty_forty_metrics_at['annualized_return']*100:.2f}%",
            f"{sixty_forty_metrics_at['annualized_volatility']*100:.2f}%",
            f"{sixty_forty_metrics_at['sharpe_ratio']:.2f}",
            f"{sixty_forty_metrics_at['sortino_ratio']:.2f}",
            f"{sixty_forty_metrics_at['max_drawdown']*100:.2f}%",
            f"{sixty_forty_metrics_at['annualized_turnover']*100:.2f}%"
        ]
    }
    
    st.table(pd.DataFrame(summary_data).set_index("Metric"))
    
    # Detailed Operational Diagnostics
    st.markdown("### Operational Details")
    
    diag_col1, diag_col2 = st.columns(2)
    with diag_col1:
        st.markdown("**HRP Strategy Details**")
        st.markdown(f"- **Total Rebalances**: {hrp_diag_at['total_rebalance_events']}")
        st.markdown(f"- **Average Assets Traded per Rebalance**: {hrp_diag_at['average_assets_traded_per_event']:.2f}")
        st.markdown(f"- **Total Friction Costs Paid**: {hrp_diag_at['total_friction_costs_paid']:,.2f} EUR")
        st.markdown(f"- **Total PFU Taxes Paid (31.4%):**: {hrp_diag_at['total_pfu_taxes_paid']:,.2f} EUR")
        st.markdown(f"- **Tax Drag (of total return)**: {hrp_diag_at['tax_drag_percentage_of_returns']*100:.2f}%")
        st.markdown(f"- **Remaining Tax Loss Carryforward**: {hrp_diag_at['remaining_tax_loss_carryforward']:,.2f} EUR")
        
    with diag_col2:
        st.markdown("**60/40 Benchmark Details**")
        st.markdown(f"- **Total Friction Costs Paid**: {sf_diag_at['total_friction_costs_paid']:,.2f} EUR")
        st.markdown(f"- **Total PFU Taxes Paid**: {sf_diag_at['total_pfu_taxes_paid']:,.2f} EUR")
        st.markdown(f"- **Remaining Tax Loss Carryforward**: {sf_diag_at['remaining_tax_loss_carryforward']:,.2f} EUR")

with tab6:
    st.markdown("### 📅 Yearly Indicators Breakdown (HRP Denoised Strategy)")
    st.markdown("This interactive table details the year-by-year financial and operational performance of the HRP portfolio.")
    st.dataframe(yearly_df, use_container_width=True)

with tab7:
    st.markdown("### ⚙️ Parameter Sensitivity Sweep")
    st.markdown("Evaluate how individual parameters affect final wealth, risk-adjusted performance, and operational cost parameters, holding other settings constant.")
    
    if st.button("🔍 Run Parameter Sensitivity Sweep"):
        with st.spinner("Executing sensitivity backtests..."):
            st.markdown("#### 1. Rebalance Frequency Sensitivity")
            st.markdown("Compare the performance of Weekly, Monthly, Quarterly, and Yearly rebalancing frequencies.")
            freq_df = run_rebalance_frequency_sweep(prices_df, params_at, pool_key, initial_capital)
            st.table(freq_df)
            
            st.markdown("#### 2. Clustering Linkage Method Sensitivity")
            st.markdown("Compare different agglomerative dendrogram linkage methods (`Single`, `Complete`, `Average`, `Ward`).")
            linkage_df = run_linkage_sweep(prices_df, params_at, pool_key, initial_capital)
            st.table(linkage_df)
            
            st.markdown("#### 3. Drift Band Threshold Sensitivity")
            st.markdown("Compare drift band tolerance thresholds ranging from 0% (always rebalance) up to 10.0%.")
            drift_df = run_drift_band_sweep(prices_df, params_at, pool_key, initial_capital)
            st.table(drift_df)
    else:
        st.info("Click the button above to run the sensitivity sweeps. (Runs in less than 1 second)")

@st.fragment
def render_algo_explainer_fragment(prices_df, params_at, pool_key, lookback_years, drift_threshold_pct, initial_capital, hrp_cum_at, hrp_diag_at):
    algo_selection = st.radio(
        "Select Algorithm Module:",
        options=[
            "1. Covariance Denoising (RMT)",
            "2. Hierarchical Risk Clustering",
            "3. HRP Weight Bisection",
            "4. Drift-Band Execution Guard"
        ],
        horizontal=True
    )
    
    # 1. Date Selector setup for walk-forward inspection
    prices_clean = prices_df.ffill().bfill()
    all_rebalance_dates = get_rebalance_dates(prices_clean.index, params_at.rebalance_frequency)
    start_date = prices_clean.index[0]
    valid_rebalance_dates = [
        d for d in all_rebalance_dates 
        if d >= start_date + pd.DateOffset(years=params_at.lookback_years)
    ]
    
    selected_date = None
    all_diags = {}
    date_list = []
    date_options = {}
    if valid_rebalance_dates:
        # Precompute diagnostics in one cached step
        all_diags = precompute_all_rebalance_diagnostics(prices_df, params_at, pool_key)
        date_options = {d.strftime("%Y-%m-%d"): d for d in valid_rebalance_dates if d in all_diags}
        date_list = list(date_options.keys())

    # Helper function to render slider in a specific layout context
    def render_date_slider():
        if date_list:
            selected_date_str = st.select_slider(
                "📅 Drag Slider to Visualize Walk-Forward Evolution (Rebalance Date):",
                options=date_list,
                value=date_list[-1],
                key="rebalance_date_slider",
                help="Drag the slider to see how the covariance, risk clusters, and weights transition dynamically at each historical rebalance event."
            )
            return date_options[selected_date_str]
        return None
    
    st.markdown("---")
    
    if algo_selection == "1. Covariance Denoising (RMT)":
        st.markdown("### 1. Covariance Denoising via Random Matrix Theory (RMT)")
        st.markdown("""
        **The Problem**: Standard portfolio optimization depends heavily on the covariance matrix. However, if the lookback window is short (e.g., 4 years) compared to the number of assets, empirical correlations are dominated by noise. This is called the *Markowitz Curse*—it causes extreme, unstable portfolio weights.
        
        **The Solution**: We apply the **Marchenko-Pastur (MP) theorem** from Random Matrix Theory. MP predicts the distribution of eigenvalues for a purely random correlation matrix. By identifying the theoretical noise cutoff threshold $\\lambda_{max}$:
        1. Eigenvalues **above** $\\lambda_{max}$ represent true economic factor signals (like the Market factor).
        2. Eigenvalues **below** $\\lambda_{max}$ represent noise. We replace them with their average value, preserving the trace (total risk) of the matrix while stripping out spurious correlation spikes.
        """)
        
        # Render slider just above charts, below explanatory text
        selected_date = render_date_slider()
        
        # Fetch diagnostics for the selected rebalance date from precomputed dict
        diag_data = all_diags.get(selected_date)
        
        if diag_data is not None:
            col_l, col_r = st.columns([2, 3])
            
            with col_l:
                st.markdown("#### Eigenvalue Spectrum Denoising")
                st.markdown("Notice how eigenvalues below the noise cutoff threshold are flattened to their average, filtering out random noise while leaving the dominant market factors intact.")
                
                # Sorted eigenvalues plot
                e_emp = diag_data["eigenvalues"]
                e_den = diag_data["eigenvalues_denoised"]
                lambda_max = diag_data["lambda_max"]
                
                fig_rmt = go.Figure()
                fig_rmt.add_trace(go.Scatter(
                    y=e_emp, mode='lines+markers', name='Empirical',
                    line=dict(color='#FF8F8F', width=2),
                    marker=dict(size=6)
                ))
                fig_rmt.add_trace(go.Scatter(
                    y=e_den, mode='lines+markers', name='Denoised (RMT)',
                    line=dict(color='#FF4B4B', width=2),
                    marker=dict(size=6)
                ))
                fig_rmt.add_hline(
                    y=lambda_max, line_dash="dash", line_color="#2196F3",
                    annotation_text=f"Noise Cutoff λ_max ({lambda_max:.2f})",
                    annotation_position="bottom right"
                )
                
                fig_rmt.update_layout(
                    template="plotly_dark",
                    xaxis_title="Eigenvalue Rank Index",
                    yaxis_title="Eigenvalue Magnitude",
                    margin=dict(l=40, r=40, t=30, b=40),
                    legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
                )
                st.plotly_chart(fig_rmt, use_container_width=True)
                
            with col_r:
                st.markdown("#### Correlation Matrix Heatmaps")
                st.markdown("Select from the tabs below to compare the raw empirical correlation matrix with the denoised correlation matrix, or to view the specific noise differences removed by RMT.")
                
                # Heatmaps
                heat_tab1, heat_tab2, heat_tab3 = st.tabs(["📊 Empirical Correlation", "🛡️ Denoised Correlation", "🔍 Denoising Difference (Emp. - Den.)"])
                
                # Use asset names for labels
                assets = diag_data["assets"]
                labels = [ASSET_NAMES.get(a, a) for a in assets]
                
                with heat_tab1:
                    fig_heat_emp = px.imshow(
                        diag_data["corr_emp"],
                        x=labels, y=labels,
                        color_continuous_scale='RdBu',
                        zmin=-1.0, zmax=1.0,
                        title="Raw Empirical Correlation Matrix"
                    )
                    fig_heat_emp.update_layout(
                        template="plotly_dark",
                        margin=dict(l=40, r=40, t=40, b=40)
                    )
                    st.plotly_chart(fig_heat_emp, use_container_width=True)
                    
                with heat_tab2:
                    fig_heat_den = px.imshow(
                        diag_data["corr_denoised"],
                        x=labels, y=labels,
                        color_continuous_scale='RdBu',
                        zmin=-1.0, zmax=1.0,
                        title="Denoised Correlation Matrix (RMT)"
                    )
                    fig_heat_den.update_layout(
                        template="plotly_dark",
                        margin=dict(l=40, r=40, t=40, b=40)
                    )
                    st.plotly_chart(fig_heat_den, use_container_width=True)
                    
                with heat_tab3:
                    diff_corr = diag_data["corr_emp"] - diag_data["corr_denoised"]
                    fig_heat_diff = px.imshow(
                        diff_corr,
                        x=labels, y=labels,
                        color_continuous_scale='RdBu',
                        color_continuous_midpoint=0.0,
                        title="Denoising Difference Heatmap (Empirical - Denoised)"
                    )
                    fig_heat_diff.update_layout(
                        template="plotly_dark",
                        margin=dict(l=40, r=40, t=40, b=40)
                    )
                    st.plotly_chart(fig_heat_diff, use_container_width=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            max_diff = np.max(np.abs(diag_data["corr_emp"].values - diag_data["corr_denoised"].values))
            st.markdown(f"""
            > [!TIP]
            > **Why do the Empirical and Denoised Correlation matrices look so similar?**
            > Currently, the Lookback Window is set to **{lookback_years} Years** (~{len(diag_data['cov_emp'])} daily observations) for only {len(diag_data['assets'])} assets. 
            > Because you have a very large number of observations relative to the number of assets, the empirical covariance is already statistically stable and clean. 
            > 
            > As a result, RMT denoising only needs to filter out a tiny amount of noise, resulting in a maximum difference of **{max_diff*100.0:.3f}%** between the correlation coefficients (shown in the *Denoising Difference* tab).
            > 
            > **To see the RMT algorithm make much larger adjustments**, try **reducing the Lookback Window to 1 Year** in the sidebar to simulate a high-noise regime with fewer observations!
            """)
        else:
            st.info("Insufficient lookback data to compute RMT diagnostics.")
            
    elif algo_selection == "2. Hierarchical Risk Clustering":
        st.markdown("### 2. Hierarchical Risk Clustering")
        st.markdown("""
        **The Problem**: Traditional portfolio optimization ignores the fact that assets belong to hierarchical structures (e.g., sectors or asset classes). Treating all assets as independent leads to highly unstable, concentrated portfolios when assets within a sector are highly correlated.
        
        **The Solution**: Instead of forcing rigid predefined sectors, HRP performs **unsupervised hierarchical clustering** on the assets' correlation distance matrix.
        1. **Distance Metric**: Correlation $\\rho_{i,j}$ is converted into distance: $d_{i,j} = \\sqrt{2(1 - \\rho_{i,j})}$. This maps perfectly correlated assets to distance 0, and perfectly anti-correlated assets to distance 2.
        2. **Linkage Tree**: Assets are grouped step-by-step into a dendrogram using the selected linkage method (e.g. `single`, `average`, `ward`).
        3. **Quasi-Diagonalization**: We reorder the rows/columns of the correlation matrix so that similar assets are placed next to each other. This groups risk clusters into clean blocks along the diagonal.
        """)
        
        # Render slider just above charts, below explanatory text
        selected_date = render_date_slider()
        diag_data = all_diags.get(selected_date)
        
        if diag_data is not None:
            col_l, col_r = st.columns([1, 1])
            
            with col_l:
                st.markdown("#### Clustered Asset Tree (Dendrogram)")
                st.markdown("Assets that share a branch are highly correlated. The height of the connection represents their distance.")
                
                # Plotly Dendrogram
                assets = diag_data["assets"]
                labels = [ASSET_NAMES.get(a, a) for a in assets]
                Z = diag_data["linkage"]
                
                try:
                    fig_dendro = ff.create_dendrogram(
                        np.zeros((len(labels), 2)),
                        labels=labels,
                        linkagefun=lambda x: Z,
                        orientation='left'
                    )
                    fig_dendro.update_layout(
                        template="plotly_dark",
                        height=500,
                        margin=dict(l=10, r=40, t=10, b=40),
                        xaxis_title="Cophenetic Distance"
                    )
                    st.plotly_chart(fig_dendro, use_container_width=True)
                except Exception as e:
                    st.error(f"Error drawing dendrogram: {e}")
                    
            with col_r:
                st.markdown("#### Reordered Clustered Correlation Heatmap")
                st.markdown("This heatmap uses the quasi-diagonalized leaf order of the dendrogram. Notice how highly correlated assets group into distinct blocks along the diagonal.")
                
                # Reordered correlation
                ordered_assets = diag_data["ordered_assets"]
                corr_ordered = diag_data["corr_denoised"].loc[ordered_assets, ordered_assets]
                ordered_labels = [ASSET_NAMES.get(a, a) for a in ordered_assets]
                
                fig_heat_ordered = px.imshow(
                    corr_ordered,
                    x=ordered_labels, y=ordered_labels,
                    color_continuous_scale='RdBu',
                    zmin=-1.0, zmax=1.0,
                    title="Clustered Correlation Heatmap (Quasi-Diagonalized)"
                )
                fig_heat_ordered.update_layout(
                    template="plotly_dark",
                    height=500,
                    margin=dict(l=40, r=40, t=40, b=40)
                )
                st.plotly_chart(fig_heat_ordered, use_container_width=True)
        else:
            st.info("Insufficient lookback data to compute risk clusters.")
            
    elif algo_selection == "3. HRP Weight Bisection":
        st.markdown("### 3. HRP Weight Bisection (Recursive Bisection)")
        st.markdown("""
        **The Problem**: Classical portfolio optimization (like Mean-Variance or standard Inverse-Variance) ignores correlation structures. If you have 5 tech stocks and 1 utility stock, and all tech stocks are low-volatility, Inverse-Variance will allocate massive weight to the tech sector collectively, double-counting their risk diversification.
        
        **The Solution**: HRP allocates weights recursively down the dendrogram tree:
        1. **Split**: Starting at the root (top of the tree), it divides the assets into two sub-clusters (left and right).
        2. **Variance Aggregation**: It computes the aggregate variance of each sub-cluster as a single unit.
        3. **Inverse-Variance Allocation**: It divides the weight between the two sub-clusters inversely proportional to their variance:
           $$\\alpha = 1 - \\frac{V_{left}}{V_{left} + V_{right}}$$
        4. **Recurse**: This process is repeated down each branch until it reaches individual assets.
        
        This guarantees that the 5 tech stocks (which cluster together) share their collective allocation, while the utility stock gets its independent share.
        """)
        
        # Render slider just above charts, below explanatory text
        selected_date = render_date_slider()
        diag_data = all_diags.get(selected_date)
        
        if diag_data is not None:
            st.markdown("#### Weight Allocation Comparison")
            st.markdown("Compare the target weights generated by HRP with Inverse-Variance and Equal Weighting. Notice how HRP moderates the weights in clusters of highly correlated assets to achieve true diversification.")
            
            # Combine into a single DataFrame for Plotly Express
            assets = diag_data["assets"]
            labels = [ASSET_NAMES.get(a, a) for a in assets]
            
            df_weights = pd.DataFrame({
                "Asset": labels,
                "HRP Denoised": diag_data["w_hrp"].values,
                "Inverse Variance": diag_data["w_inv_var"].values,
                "Equal Weight": diag_data["w_equal"].values
            })
            
            # Melt for grouped bar chart
            df_weights_melted = df_weights.melt(
                id_vars="Asset",
                var_name="Allocation Method",
                value_name="Weight"
            )
            
            fig_weights_comp = px.bar(
                df_weights_melted,
                x="Asset",
                y="Weight",
                color="Allocation Method",
                barmode="group",
                color_discrete_map={
                    "HRP Denoised": "#FF4B4B",
                    "Inverse Variance": "#2196F3",
                    "Equal Weight": "#4CAF50"
                },
                title="Weights Comparison: HRP vs. Benchmarks (Latest Rebalance Date)"
            )
            fig_weights_comp.update_layout(
                template="plotly_dark",
                xaxis_title="",
                yaxis_title="Weight (1.0 = 100%)",
                margin=dict(l=40, r=40, t=50, b=40)
            )
            st.plotly_chart(fig_weights_comp, use_container_width=True)
            
            # Show interactive data table
            st.markdown("##### Allocation Details Table")
            df_table = df_weights.copy()
            df_table["HRP Denoised"] = df_table["HRP Denoised"].map(lambda x: f"{x*100.0:.2f}%")
            df_table["Inverse Variance"] = df_table["Inverse Variance"].map(lambda x: f"{x*100.0:.2f}%")
            df_table["Equal Weight"] = df_table["Equal Weight"].map(lambda x: f"{x*100.0:.2f}%")
            st.dataframe(df_table.set_index("Asset"), use_container_width=True)
        else:
            st.info("Insufficient lookback data to compute weights bisection.")
            
    elif algo_selection == "4. Drift-Band Execution Guard":
        st.markdown("### 4. Drift-Band Execution Guard")
        st.markdown("""
        **The Problem**: Standard academic portfolio models assume zero transaction costs and zero taxes. In the real world, rebalancing too frequently (e.g., quarterly) or for tiny weight adjustments (e.g., changing a weight from 8.2% to 8.3%) is highly counterproductive. It eats into returns via trading commissions (friction costs) and immediately triggers capital gains taxes (such as the **French PFU tax rate** on realized capital gains), killing the power of compounding.
        
        **The Solution**: We implement a **Drift-Band Guard**. When a rebalance date is reached, we calculate the optimal target weights. However, we only execute trades for assets whose target weights differ from their *current actual weights* by more than the **Drift Band Threshold** (e.g., 1.5%). If the drift is smaller, we bypass trading that asset. If no asset exceeds the threshold, the entire rebalance event is bypassed! This filters out noise trades, defers capital gains realization, and saves trading fees.
        """)
        
        # Run a quick zero-drift comparison backtest
        params_no_drift = replace(params_at, drift_threshold=0.0)
        
        with st.spinner("Running zero-drift comparison..."):
            hrp_cum_nd, hrp_diag_nd, _, _, _ = run_strategy_backtest(
                prices_df, params_no_drift, pool_key, initial_capital
            )
            
        # Metrics comparison
        nd_wealth = initial_capital * hrp_cum_nd.iloc[-1]
        at_wealth = initial_capital * hrp_cum_at.iloc[-1]
        
        col_nd, col_at = st.columns(2)
        
        with col_nd:
            st.markdown(f"""
            <div class="metric-card" style="border-color: #888888;">
                <div class="metric-title">Always Rebalance (0.0% Drift)</div>
                <div class="metric-value">€{nd_wealth:,.2f}</div>
                <div class="metric-delta">
                    Rebalances: <span class="delta-red">{hrp_diag_nd['total_rebalance_events']}</span> | 
                    Taxes Paid: <span class="delta-red">€{hrp_diag_nd['total_pfu_taxes_paid']:,.0f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Show list of details
            st.markdown("**Performance Details:**")
            st.markdown(f"- **Final Net Wealth**: €{nd_wealth:,.2f}")
            st.markdown(f"- **Total Rebalances**: {hrp_diag_nd['total_rebalance_events']}")
            st.markdown(f"- **Total Friction Costs**: €{hrp_diag_nd['total_friction_costs_paid']:,.2f}")
            st.markdown(f"- **Total Taxes Paid**: €{hrp_diag_nd['total_pfu_taxes_paid']:,.2f}")
            st.markdown(f"- **Tax Drag (of return)**: {hrp_diag_nd['tax_drag_percentage_of_returns']*100:.2f}%")
            
        with col_at:
            st.markdown(f"""
            <div class="metric-card" style="border-color: #FF4B4B;">
                <div class="metric-title">Drift Band Enabled ({drift_threshold_pct}%)</div>
                <div class="metric-value">€{at_wealth:,.2f}</div>
                <div class="metric-delta">
                    Rebalances: <span class="delta-green">{hrp_diag_at['total_rebalance_events']}</span> | 
                    Taxes Paid: <span class="delta-green">€{hrp_diag_at['total_pfu_taxes_paid']:,.0f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Show list of details
            st.markdown("**Performance Details:**")
            st.markdown(f"- **Final Net Wealth**: €{at_wealth:,.2f}")
            st.markdown(f"- **Total Rebalances**: {hrp_diag_at['total_rebalance_events']}")
            st.markdown(f"- **Total Friction Costs**: €{hrp_diag_at['total_friction_costs_paid']:,.2f}")
            st.markdown(f"- **Total Taxes Paid**: €{hrp_diag_at['total_pfu_taxes_paid']:,.2f}")
            st.markdown(f"- **Tax Drag (of return)**: {hrp_diag_at['tax_drag_percentage_of_returns']*100:.2f}%")
            
        # Drawdown drag chart (Taxes + Friction paid over time)
        st.markdown("#### Cumulative Financial Drag (Taxes & Friction Paid)")
        st.markdown("This chart shows the cumulative money lost to fees and capital gains taxes. Notice how the drift band flattens the curve, preserving your capital so it can compound.")
        
        drag_nd = hrp_diag_nd["daily_tc"] + hrp_diag_nd["daily_tax"]
        drag_at = hrp_diag_at["daily_tc"] + hrp_diag_at["daily_tax"]
        
        fig_drag = go.Figure()
        fig_drag.add_trace(go.Scatter(
            x=drag_nd.index, y=drag_nd,
            name="Always Rebalance (0.0% Drift)",
            line=dict(color="#FF8F8F", width=2)
        ))
        fig_drag.add_trace(go.Scatter(
            x=drag_at.index, y=drag_at,
            name=f"Drift Band Enabled ({drift_threshold_pct}%)",
            line=dict(color="#FF4B4B", width=2)
        ))
        
        # Calculate savings
        savings = drag_nd.iloc[-1] - drag_at.iloc[-1]
        st.info(f"💡 **Drift Band Savings**: The drift band guard has saved you **€{savings:,.2f}** in total friction fees and capital gains taxes over the backtest period!")
        
        fig_drag.update_layout(
            template="plotly_dark",
            xaxis_title="Date",
            yaxis_title="Total Financial Drag (EUR)",
            hovermode="x unified",
            margin=dict(l=40, r=40, t=30, b=40),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        st.plotly_chart(fig_drag, use_container_width=True)

with tab8:
    st.markdown("## 🧠 HRP Algorithmic Explanations & Impact")
    st.markdown("This interactive lab helps you understand the mathematics and real-world impact of the four key algorithms driving the Hierarchical Risk Parity engine.")
    render_algo_explainer_fragment(prices_df, params_at, pool_key, lookback_years, drift_threshold_pct, initial_capital, hrp_cum_at, hrp_diag_at)

# --- LLM COPYABLE REPORT SECTION ---
st.markdown("<br><hr>", unsafe_allow_html=True)
with st.expander("📋 LLM-Shareable Report (Click to Copy)", expanded=False):
    st.markdown("Use the copy button in the top right of the code block below to copy and share the backtest simulation results with an LLM:")
    
    llm_report = f"""# HRP Backtest Simulation Results (Walk-Forward)

## ⚙️ Strategy Parameters:
- **Asset Pool**: {pool_selection}
- **Simulation Start Date**: {prices_df.index[0].strftime('%Y-%m-%d')} ({'Aligned' if limit_history else 'Full Range'})
- **Lookback Window**: {lookback_years} Years
- **Rebalance Frequency**: {rebalance_frequency}
- **Linkage Method**: {linkage_method}
- **Drift Band Threshold**: {drift_threshold_pct}%
- **Transaction Cost**: {transaction_cost_bps} bps
- **French PFU Tax Rate**: {french_pfu_rate_pct}%
- **Initial Capital**: €{initial_capital:,.2f}

## 📊 Performance Comparison Table:
| Metric | HRP Denoised Strategy | S&P 500 Buy & Hold | 60/40 Equity/Bond |
| :--- | :---: | :---: | :---: |
| Initial Wealth | €{initial_capital:,.2f} | €{initial_capital:,.2f} | €{initial_capital:,.2f} |
| Final Wealth (Before-Tax) | €{initial_capital * hrp_cum_bt.iloc[-1]:,.2f} | €{sp500_final_wealth:,.2f} | €{initial_capital * sixty_forty_cum_bt.iloc[-1]:,.2f} |
| Final Wealth (After-Tax) | €{hrp_final_wealth_at:,.2f} | €{sp500_final_wealth:,.2f} | €{sixty_forty_final_wealth_at:,.2f} |
| Total Taxes Paid | €{hrp_diag_at['total_pfu_taxes_paid']:,.2f} | €0.00 | €{sf_diag_at['total_pfu_taxes_paid']:,.2f} |
| Annualized Return (Before-Tax) | {hrp_metrics_bt['annualized_return']*100:.2f}% | {sp500_metrics_at['annualized_return']*100:.2f}% | {sixty_forty_metrics_bt['annualized_return']*100:.2f}% |
| Annualized Return (After-Tax) | {hrp_metrics_at['annualized_return']*100:.2f}% | {sp500_metrics_at['annualized_return']*100:.2f}% | {sixty_forty_metrics_at['annualized_return']*100:.2f}% |
| Annualized Volatility (After-Tax) | {hrp_metrics_at['annualized_volatility']*100:.2f}% | {sp500_metrics_at['annualized_volatility']*100:.2f}% | {sixty_forty_metrics_at['annualized_volatility']*100:.2f}% |
| Sharpe Ratio (After-Tax) | {hrp_metrics_at['sharpe_ratio']:.2f} | {sp500_metrics_at['sharpe_ratio']:.2f} | {sixty_forty_metrics_at['sharpe_ratio']:.2f} |
| Sortino Ratio (After-Tax) | {hrp_metrics_at['sortino_ratio']:.2f} | {sp500_metrics_at['sortino_ratio']:.2f} | {sixty_forty_metrics_at['sortino_ratio']:.2f} |
| Maximum Drawdown (After-Tax) | {hrp_metrics_at['max_drawdown']*100:.2f}% | {sp500_metrics_at['max_drawdown']*100:.2f}% | {sixty_forty_metrics_at['max_drawdown']*100:.2f}% |
| Annualized Turnover | {hrp_metrics_at['annualized_turnover']*100:.2f}% | 0.00% | {sixty_forty_metrics_at['annualized_turnover']*100:.2f}% |

## 🛠️ HRP Operational Diagnostics:
- **Total Rebalances**: {hrp_diag_at['total_rebalance_events']}
- **Average Assets Traded per Event**: {hrp_diag_at['average_assets_traded_per_event']:.2f}
- **Total Friction Fees Paid**: €{hrp_diag_at['total_friction_costs_paid']:,.2f}
- **Total PFU Taxes Paid (31.4%)**: €{hrp_diag_at['total_pfu_taxes_paid']:,.2f}
- **Tax Drag (% of total return)**: {hrp_diag_at['tax_drag_percentage_of_returns']*100:.2f}%
- **Remaining Tax Loss Carryforward**: €{hrp_diag_at['remaining_tax_loss_carryforward']:,.2f}

## 📅 Yearly Breakdown (HRP Denoised Strategy):
{yearly_df.to_markdown()}
"""
    
    st.code(llm_report, language="markdown")


