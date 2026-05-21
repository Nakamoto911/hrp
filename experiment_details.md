# HRP Backtest Experiment Details

This document houses detailed data tables, parameter sensitivity statistics, mathematical audits, and code architecture summaries for experiments recorded in [memory.md](file:///Users/user2/Documents/Code/hrp/memory.md).

---

## Experiment [2026-05-21]

### 1. Cumulative Performance Metrics (European Pool)
The table below compares the strategy metrics before and after the data cleaning, holiday alignment, and parameter optimizations.

| Metric | Before Fixes | With Holiday Mismatch Fix (Final Baseline) | Optimized Grid Search Config | Optimized Config (No JPY/German Bonds) |
| :--- | :---: | :---: | :---: | :---: |
| **Annualized Return** | 0.37% | **0.89%** | **2.62%** | **3.67%** |
| **Annualized Volatility** | 5.65% | **3.99%** | **4.08%** | **4.62%** |
| **Sharpe Ratio (Rf=0)** | 0.07 | **0.22** | **0.64** | **0.79** |
| **Maximum Drawdown** | -23.66% | **-19.44%** | **-16.02%** | **-16.02%** |
| **Annualized Turnover** | 289.32% | **77.03%** | **109.20%** | **94.80%** |

---

### 2. Asset-Selection Sensitivity Study
This sensitivity study isolates the impact of different bond exclusions on the European Investable pool.
*Parameters: lookback=4y, frequency=quarterly, linkage=single, drift=0.015, denoise=True, bisection=tree*

| Configuration | Excluded Assets | Annualized Return | Annualized Volatility | Sharpe Ratio (Rf=0) | Maximum Drawdown | Annualized Turnover |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **All Assets (Baseline)** | None | 1.16% | 3.96% | 0.29 | -18.21% | 66.90% |
| **Exclude JPY Bonds** | `XJSE.DE` | 1.60% | 3.97% | 0.40 | -18.21% | 70.08% |
| **Exclude JPY + German Bunds** | `XJSE.DE`, `IS0L.DE` | **2.09%** | **4.48%** | **0.47** | **-18.21%** | **61.55%** |
| **Exclude All Govt Bonds** | `XJSE.DE`, `IS0L.DE`, `IGLT.MI`, `IBCQ.DE` | 2.09% | 4.80% | 0.44 | -24.01% | 58.45% |

#### Key Insights:
- Excluding JPY government bonds (`XJSE.DE`) removes a severe currency/yield drag (near-zero yields + 40% JPY depreciation against EUR).
- Excluding long-duration German Bunds (`IS0L.DE`) prevents losses from duration crash during the 2022-2023 rate hike cycle.
- Retaining short-term USD Treasuries (`IBCQ.DE`) is necessary to prevent maximum drawdown from deteriorating from -18.21% to -24.01%.

---

### 3. Grid Search Averages & Parameter Sensitivity
Below are the average results across the entire 576-run grid search parameter sweep for both pools.

#### European Pool parameter averages:
* **Denoising**: 
  - `False`: Return = 1.60% | Sharpe = 0.399 | Turnover = 88.17%
  - `True`: Return = 1.65% | Sharpe = 0.410 | Turnover = 87.21%
  - *Insight*: Denoising consistently clean up random eigenvalues, improving Sharpe and reducing turnover.
* **Linkage Method**:
  - `ward`: Return = **1.79%** | Sharpe = **0.452** | Max DD = **-16.90%**
  - `single`: Return = 1.75% | Sharpe = 0.428 | Max DD = -17.42%
  - `average`: Return = 1.49% | Sharpe = 0.373 | Max DD = -17.82%
  - `complete`: Return = 1.46% | Sharpe = 0.365 | Max DD = -18.39%
  - *Insight*: Ward linkage minimizes intra-cluster variance, leading to more stable risk groupings.
* **Lookback Years**:
  - `2y`: Return = **2.16%** | Sharpe = **0.528** | Turnover = 114.35%
  - `3y`: Return = 2.07% | Sharpe = 0.503 | Turnover = 91.56%
  - `4y`: Return = 1.00% | Sharpe = 0.260 | Turnover = 70.49%
  - `5y`: Return = 1.26% | Sharpe = 0.328 | Turnover = 74.35%
  - *Insight*: Shorter windows (2-3y) adapt much faster to modern rate regimes than 4-5y.
* **Rebalance Frequency**:
  - `semi-annually`: Return = **1.73%** | Sharpe = **0.430** | Turnover = **68.48%**
  - `quarterly`: Return = 1.55% | Sharpe = 0.385 | Turnover = 83.58%
  - `monthly`: Return = 1.58% | Sharpe = 0.399 | Turnover = 111.01%
  - *Insight*: Semi-annual frequency mitigates transaction costs and French PFU capital gains tax drag.
* **Drift Threshold**:
  - `0.000`: Return = 1.60% | Sharpe = 0.402 | Turnover = 97.51%
  - `0.015`: Return = 1.63% | Sharpe = 0.407 | Turnover = 84.88%
  - `0.030`: Return = 1.64% | Sharpe = 0.405 | Turnover = 80.67%
  - *Insight*: Drift thresholds reduce turnover by ~17% with negligible impact on returns.

---

### 4. Mathematical Audit & Code Upgrades

#### A. Correlation-to-Distance Mapping
López de Prado's original paper defines correlation-to-distance mapping as:
$$D_{\text{paper}} = \sqrt{\frac{1}{2}(1 - \rho)}$$
The codebase was using $\sqrt{2(1 - \rho)}$, which is a scaling factor of 2.0. Due to the scale-invariance of hierarchical clustering (e.g. Ward/Single/Complete linkage methods), the topologies and allocations remain mathematically identical. We updated the code to use $0.5$ to align with the literature and keep $D \in [0, 1]$.

#### B. Euclidean Distance Column Vectorization
Replaced $O(N^2)$ Python nested loops for column distance computations with a vectorized version:
```python
dist_cols = squareform(pdist(dist, metric='euclidean'))
```
This saves execution time and improves numerical stability.

#### C. Tree-Based Bisection vs Index-Based Bisection
- **Index-Based (Original)**: Splits the sorted list strictly in half by index (`mid = len(current)//2`). If the tree is highly unbalanced (e.g., 1 low-volatility bond merging with 11 equities at the top), index bisection puts 5 equities with the bond, allocating almost all weight to the bond and starving the remaining equities.
- **Tree-Based (Updated)**: Follows the actual branches of linkage matrix $Z$ (e.g., splitting into the actual left and right child clusters). This prevents weight starvation and stabilizes turnover, since small ordering shifts do not cross an arbitrary index midpoint.

#### Denoising Trace Preservation Proof
- Empirical eigenvalues $\lambda_1 \ge \dots \ge \lambda_N$ below Marchenko-Pastur cutoff $\lambda_{\text{max}}$ are set to their mean:
  $$\bar{\lambda}_{\text{noise}} = \frac{1}{n_{\text{noise}}} \sum_{i \in \text{noise}} \lambda_i$$
- The sum of eigenvalues remains unchanged:
  $$\sum_{i=1}^N \lambda^{\text{denoised}}_i = \sum_{j \in \text{signal}} \lambda_j + n_{\text{noise}} \cdot \bar{\lambda}_{\text{noise}} = \sum_{i=1}^N \lambda_i = N$$
- After diagonal rescaling ($C_{\text{denoised}} = D^{-1/2} C_{\text{raw}} D^{-1/2}$ where $D = \text{diag}(C_{\text{raw}})$), the diagonal elements are exactly $1.0$, conserving the trace $N$ and ensuring Positive Semi-Definiteness (PSD) by Sylvester's Law of Inertia.
