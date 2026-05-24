# HRP: Hierarchical Risk Parity Portfolio Engine

This repository provides a fast, interactive, and research-grade engine for Hierarchical Risk Parity (HRP) portfolio construction and walk-forward backtesting. It is designed to support robust empirical research in portfolio construction, covariance denoising, tax/friction modeling, and real-world portfolio simulation.

## Features

- **Streamlit Dashboard:** An interactive web dashboard for configuring, analyzing, and visualizing HRP strategies, including tax drag modeling, asset pools, and sensitivity sweeps.
- **Backtester:** Command-line and Python API to run large universe, multi-decade HRP backtests with tax and transaction fee simulation.
- **Covariance Denoising:** Uses Random Matrix Theory (Marchenko-Pastur) for robust estimation with small lookback windows.
- **Comprehensive Benchmarks:** Compare to S&P 500 Buy & Hold and classic 60/40 portfolios.
- **Sensitivity Analysis:** Built-in analysis of parameter impacts—rebalance frequency, lookback window, clustering linkage, drift bands.
- **Unit-Tests:** Extensive unit-testing for core mathematical and data modules.

## Installation

1. Clone the repo:
    ```bash
    git clone https://github.com/Nakamoto911/hrp.git
    cd hrp
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Quick Start

### 1. Launch the Dashboard

```bash
streamlit run app.py
```
- Choose between ETF, Mutual Fund, or European asset pools.
- Configure parameters (rebalance frequency, tax rate, transaction costs) and interactively visualize strategy results.

### 2. Run Command-Line Backtests

```bash
python run_backtest.py --pool etf --lookback-years 4 --frequency quarterly --linkage single --drift-threshold 0.015 --transaction-cost 5.0 --pfu-rate 0.314
```
See `python run_backtest.py --help` for all options.

### 3. Run Tests

```bash
python -m unittest test_hrp.py
```

## Requirements

See [`requirements.txt`](https://github.com/Nakamoto911/hrp/blob/main/requirements.txt) for specific versions.

- numpy
- pandas
- scipy
- streamlit
- plotly
- yfinance
- tabulate

## File Structure

- `app.py` – Main Streamlit dashboard.
- `run_backtest.py` – CLI runner for HRP backtests.
- `test_hrp.py` – Unit tests for engine, denoiser, and backtest.
- `hrp_engine/` – Core package: data fetch, strategy engine, optimization, denoising, reporting utils.
- `experiment_details.md` and `memory.md` – Supporting documentation.

## Core Concepts

- **HRP Algorithm** – Clusters assets based on correlations and allocates weights to reduce concentration risk.
- **Covariance Denoising** – Reduces estimation error in high-dimensional regimes using RMT.
- **Tax Modeling** – Simulates capital gains taxes with user-specified flat rates (e.g., French PFU).
- **Drift-Band Execution** – Avoids unnecessary small trades to minimize friction and tax drag.

## Example Usage

Visualize and analyze the impact of different strategy choices:
- How does annualized return, volatility, and drawdown change with rebalance frequency?
- What is the real drag from transaction costs and capital gains taxes?
- How do alternative clustering linkages (single, complete, ward, etc.) affect allocation and diversification?

## Citing

If you use this code/interactivity in research, presentations, or educational materials, please cite this repository.

---

**Repository:** [Nakamoto911/hrp](https://github.com/Nakamoto911/hrp)

**License:** MIT (add LICENSE file if missing)
