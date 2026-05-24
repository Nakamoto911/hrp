# HRP: Hierarchical Risk Parity Portfolio Engine

This repository provides a fast, interactive, and research-grade engine for Hierarchical Risk Parity (HRP) portfolio construction and walk-forward backtesting. It is designed to support robust empirical research in portfolio construction, covariance denoising, tax/friction modeling, and real-world portfolio simulation.

---

## Features

- **Interactive Streamlit Dashboard:** A web dashboard for configuring, analyzing, and visualizing HRP strategies, including tax drag modeling, asset pools, and parameter sweeps.
- **Walk-Forward Backtester:** Python API and command-line interface to run multi-decade HRP backtests with tax, transaction fee, and drift-band simulation.
- **Covariance Denoising:** Employs Random Matrix Theory (Marchenko-Pastur theorem) to reduce estimation error in high-dimensional regimes while conserving trace properties and PSD characteristics.
- **Recursive Bisection Selection:** Choice between Lopez de Prado's original index-based bisection and a tree-based bisection to prevent asset starvation in unbalanced linkage trees.
- **Data Cleaning Pipeline:** Automated price cleaning including flat startup segment trimming and a 3-day rolling median filter for bad tick suppression.
- **Asset Pools & Regional Benchmarks:** Support for three asset pools (US ETFs, US Mutual Funds, and EUR Xetra European Investable Pool) matched with appropriate local index benchmarks.
- **Tax and Friction Modeling:** Real-world transaction costs (in bps) and flat-rate capital gains taxes (e.g., French PFU) tracked with a pro-rata cost basis and tax loss carryforward.
- **Drift-Band Execution:** Restricts execution to trades where the active weight drifts beyond a user-specified threshold, reducing turnover and tax drag.
- **Parameter Sensitivity Analysis:** Support for sweeps across lookback windows, rebalance frequencies, clustering linkages, and drift-band thresholds.
- **Unit Tests:** Extensive unit testing verifying the core mathematical modules, backtest engine, and data pipelines.

---

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Nakamoto911/hrp.git
   cd hrp
   ```

2. **Set up a virtual environment (Recommended):**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Quick Start

### 1. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
If using a virtual environment directly:
```bash
.venv/bin/streamlit run app.py
```
Use the sidebar to choose between ETF, Mutual Fund, or European asset pools, adjust parameters, and examine portfolio metrics and denoising heatmaps.

### 2. Run Command-Line Backtests
```bash
python run_backtest.py --pool etf --lookback-years 4 --frequency quarterly --linkage single --drift-threshold 0.015 --transaction-cost 5.0 --pfu-rate 0.314 --bisection tree --limit-history
```
Or via the virtual environment:
```bash
.venv/bin/python run_backtest.py --pool etf --bisection tree --limit-history
```

### 3. Run Unit Tests
```bash
python -m unittest test_hrp.py
```
Or:
```bash
.venv/bin/python -m unittest test_hrp.py
```

---

## Command-Line Reference

The backtest runner `run_backtest.py` exposes the following command-line flags:

| Parameter | Type | Default | Choices | Description |
| :--- | :--- | :--- | :--- | :--- |
| `--pool` | `str` | `etf` | `etf`, `mutual_fund`, `european` | Asset pool dataset to backtest. |
| `--lookback-years` | `int` | `4` | Any positive integer | rolling window size in years for covariance matrix estimation. |
| `--frequency` | `str` | `quarterly` | `daily`, `weekly`, `monthly`, `quarterly`, `semi-annually`, `yearly` | Rebalancing frequency of the portfolio. |
| `--linkage` | `str` | `single` | `single`, `complete`, `ward` | Hierarchical clustering linkage method. |
| `--drift-threshold` | `float` | `0.015` | Any non-negative float | Minimum target weight drift to trigger active trade execution. |
| `--transaction-cost` | `float` | `5.0` | Any non-negative float | Execution fee in basis points (1 bps = 0.01%) on buys/sells. |
| `--pfu-rate` | `float` | `0.314` | Any non-negative float | Flat tax rate applied to net realized capital gains. |
| `--bisection` | `str` | `tree` | `index`, `tree` | Bisection method: Lopez de Prado's original (`index`) or tree-based (`tree`). |
| `--limit-history` | `flag` | `False` | N/A | Aligns simulation start date to common inception date of the asset with least history. |
| `--force-refresh` | `flag` | `False` | N/A | Bypasses local CSV cache to force-redownload price data from yfinance. |
| `--output-json` | `str` | `None` | File Path | File path to save the structured JSON chart and diagnostics output. |

---

## File Structure

- `app.py` – Streamlit dashboard web application.
- `run_backtest.py` – Command-line interface runner for HRP backtests.
- `test_hrp.py` – Suite of unit tests for data, optimization, denoising, and backtesting.
- `requirements.txt` – Strict version requirements for all dependencies.
- `experiment_details.md` – Mathematical details, parameters sensitivity tables, and grid search analysis.
- `memory.md` – Chronological experiment log recording optimizations, outcomes, and findings.
- `hrp_engine/` – Modular core package containing:
  - `config.py` – Parameter configuration schemas and defaults.
  - `data.py` – Data loading, data cleaning, and rolling lookback slicing.
  - `denoiser.py` – Random Matrix Theory (Marchenko-Pastur) covariance denoising.
  - `hrp.py` – Hierarchical clustering, leaf ordering, and bisection weight calculators.
  - `backtest.py` – Walk-forward simulation loop, transaction cost, and tax accounting.
  - `reporting.py` – Metrics computation (Sharpe, Sortino, turnover) and report formatting.

---

## Advanced Core Concepts

### 1. Correlation-to-Distance Mapping
To perform hierarchical clustering, correlation coefficients $\rho_{i,j}$ are converted into distance metrics. This engine uses the scale-invariant mapping:
$$d_{i,j} = \sqrt{0.5 \cdot (1 - \rho_{i,j})}$$
This maps correlation space to bounded distance space $d_{i,j} \in [0, 1]$, matching standard literature formulations.

### 2. Covariance Denoising via Random Matrix Theory
Empirical covariance matrices calculated over short lookback windows contain significant estimation error (noise). Using the Marchenko-Pastur theorem, we identify eigenvalues dominated by noise:
$$\lambda_{\text{max}} = \sigma^2 (1 + \sqrt{q})^2 \quad \text{where} \quad q = \frac{N}{T}$$
Eigenvalues below the analytical noise boundary $\lambda_{\text{max}}$ are set to their mean, while preserving the trace sum of the correlation matrix (sum of eigenvalues $= N$). Diagonal elements are subsequently rescaled to $1.0$, conserving positive semi-definite (PSD) properties.

### 3. Tree-Based Recursive Bisection
Lopez de Prado's original index-based bisection splits the leaf-ordered asset list strictly in half at each step. In unbalanced linkage trees (where assets form highly asymmetric groups), index splits can place low-variance assets with highly diversified clusters, resulting in severe weight starvation for individual assets.
The **Tree-based Bisection** instead splits elements along the branches of the hierarchical tree matrix $Z$. This stabilizes weight allocations, preserves cluster boundaries, and prevents asset starvation.

### 4. Capital Gains Tax Simulation
We track daily portfolio cash, stock holdings, and cost bases using a pro-rata cost-basis accounting structure. Realized capital gains on partial sales are taxed at a flat rate (such as the French 31.4% PFU). Net realized capital losses are added to a carryforward balance that offsets future capital gains, replicating actual tax rules.

---

## License

This project is licensed under the MIT License.
