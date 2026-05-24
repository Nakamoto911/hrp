# HRP Backtest Experiment Memory Log

### How to Use This File
This file serves as a chronological record of all quant experiments, debugging sessions, and parameter tuning for the Hierarchical Risk Parity (HRP) engine.
- **Sort Order**: Always list entries in reverse-chronological order (newest at the top).
- **Format**: Keep entries extremely short, highlighting the objective, key takeaways, changes, and final parameters/metrics.
- **Detailed Data**: Do not include raw tables, logs, or proofs here. Save those details in the companion [experiment_details.md](experiment_details.md) file and link to it.
- **LLM Context**: When starting a new session or subagent, read this file first to understand the current best configuration and past failures.

---

## [2026-05-21] HRP Engine Bug Fixes, Tree-Based Bisection, and Parameter Grid Search

### Objective
Diagnose and resolve low returns (<1% annualized, 0.07 Sharpe) and incoherent allocation behavior (wild weight spikes pinching to 100% bonds) in the "European Investable" asset pool.

### Root Causes Identified
1. **Low Volatility Drag**: HRP allocated >75% of weight to negative-yield/depreciating JPY bonds (`XJSE.DE`) and interest-rate-sensitive long-duration Bunds (`IS0L.DE`) due to its inverse-variance mechanism.
2. **Holiday Mismatch Bug**: Mismatched exchange calendars (Xetra closed on Milan open days) caused Xetra assets to return `NaN` on holiday rebalance dates. This led the engine to temporarily discard them and allocate 100% of the portfolio to Milan-traded `IGLT.MI` (UK Gilts), causing severe weight spikes.
3. **Weight Starvation**: López de Prado's original index-based bisection split the asset list strictly in half by index, starving high-return assets in unbalanced trees.

### Core Changes Implemented
1. **Math Fix**: Corrected noise variance estimation in `hrp_engine/denoiser.py` to use `np.mean` of eigenvalues $\le 1.0$ (was incorrectly `np.var`).
2. **Holiday Mismatch Fix**: Modified `hrp_engine/data.py` to check forward-filled history instead of raw daily close prices on rebalance dates.
3. **Tree-Based Bisection**: Implemented `recursive_bisection_tree` in `hrp_engine/hrp.py` to split clusters along the branches of the linkage tree rather than arbitrary index medians.
4. **Data Cleaning**: Added automatic stale price segment trimming and a 3-day rolling median filter for bad tick suppression.
5. **Ticker Alignment**: Replaced outdated/mismatched tickers in the European pool with liquid EUR-denominated Xetra ETFs.

### Best Configuration found
- **Asset Exclusions**: Exclude `XJSE.DE` (JPY bonds) and `IS0L.DE` (long-duration Bunds). Keep short-duration `IBCQ.DE` as a stabilizer.
- **Lookback Window**: 2 Years (adapts faster to rate-hiking regimes).
- **Linkage Method**: Ward linkage (creates stable, variance-minimizing clusters).
- **Bisection Method**: Tree-Based bisection (prevents asset starvation).
- **Rebalance Frequency**: Semi-annually (reduces French PFU tax drag and transaction costs).
- **Drift Band**: 1.5% (`drift_threshold = 0.015`).
- **Denoising**: Enable Marchenko-Pastur denoising.

### Net Performance Impact (European Pool)
- **Sharpe Ratio (Rf=0)**: Improved from **0.07 to 0.79** (+10x).
- **Annualized Return**: Increased from **0.37% to 3.67%**.
- **Maximum Drawdown**: Reduced from **-23.66% to -16.02%**.
- **Annualized Turnover**: Reduced from **289.32% to 94.80%**.

*For full comparison tables, parameter sweep details, and mathematical verification, see [experiment_details.md](experiment_details.md#experiment-2026-05-21).*
