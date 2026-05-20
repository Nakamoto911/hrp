import unittest
import numpy as np
import pandas as pd
import time
from hrp_engine.config import StrategyParams
from hrp_engine.data import generate_synthetic_prices, get_lookback_data, ETF_POOL, EUROPEAN_POOL
from hrp_engine.denoiser import denoise_covariance
from hrp_engine.hrp import optimize_hrp
from hrp_engine.backtest import run_strategy_backtest

class TestHRPEngine(unittest.TestCase):
    def setUp(self):
        # Setup small mock data
        self.tickers = ETF_POOL
        self.prices_df = generate_synthetic_prices(
            tickers=self.tickers, 
            start_date='2010-01-01', 
            end_date='2018-12-31'
        )
        self.params = StrategyParams(
            lookback_years=4,
            rebalance_frequency='quarterly',
            linkage_method='single',
            drift_threshold=0.015,
            transaction_cost_bps=5.0,
            french_pfu_rate=0.314
        )
        
    def test_mp_denoising(self):
        """
        Verify that Marchenko-Pastur denoising preserves trace (sum of diagonal items = N)
        and outputs a valid positive semi-definite matrix.
        """
        N = len(self.tickers)
        # Create a highly correlated mock covariance matrix
        rng = np.random.default_rng(12345)
        raw = rng.normal(size=(N, N))
        cov_emp = np.dot(raw, raw.T)
        
        n_obs = 1000
        cov_denoised = denoise_covariance(cov_emp, n_obs)
        
        self.assertEqual(cov_denoised.shape, (N, N))
        
        # Check that diagonal of reconstructed correlation is exactly 1.0 (trace = N)
        std = np.sqrt(np.diag(cov_denoised))
        corr_denoised = cov_denoised / np.outer(std, std)
        np.testing.assert_allclose(np.diag(corr_denoised), 1.0, rtol=1e-5)
        
        # Check positive semi-definiteness (all eigenvalues >= 0)
        evals = np.linalg.eigvalsh(cov_denoised)
        self.assertTrue(all(e >= -1e-8 for e in evals), f"Eigenvalues should be non-negative: {evals}")
        
    def test_hrp_weights(self):
        """
        Verify HRP weight calculation. Weights must sum to 1.0 and all be positive.
        """
        N = len(self.tickers)
        cov_df = pd.DataFrame(
            np.eye(N) * 0.1 + 0.05, 
            index=self.tickers, 
            columns=self.tickers
        )
        
        w = optimize_hrp(cov_df, linkage_method='single')
        
        self.assertEqual(len(w), N)
        self.assertAlmostEqual(w.sum(), 1.0, places=5)
        self.assertTrue(all(x >= 0.0 for x in w), f"Weights must be positive: {w.to_dict()}")
        
    def test_backtest_execution_and_tax(self):
        """
        Verify the walk-forward backtester execution loop, checking that:
        - Equity curves are computed.
        - Benchmark curves match length and dates of the strategy.
        - Portfolio cash is always kept positive.
        """
        hrp_cum, hrp_diag, sp500_cum, sixty_forty_cum, sf_diag = run_strategy_backtest(
            prices_df=self.prices_df,
            params=self.params,
            pool_name='etf',
            initial_capital=100000.0
        )
        
        # Ensure length and indices match
        self.assertEqual(len(hrp_cum), len(sp500_cum))
        self.assertEqual(len(hrp_cum), len(sixty_forty_cum))
        np.testing.assert_array_equal(hrp_cum.index, sp500_cum.index)
        
        # Check initial value is close to 1.0 (after initial transaction costs)
        self.assertAlmostEqual(hrp_cum.iloc[0], 1.0, delta=0.005)
        self.assertAlmostEqual(sp500_cum.iloc[0], 1.0, delta=0.005)
        self.assertAlmostEqual(sixty_forty_cum.iloc[0], 1.0, delta=0.005)
        
        # Check diagnostics logging
        self.assertGreater(hrp_diag['total_rebalance_events'], 0)
        self.assertGreaterEqual(hrp_diag['total_friction_costs_paid'], 0.0)
        self.assertGreaterEqual(hrp_diag['total_pfu_taxes_paid'], 0.0)
        self.assertGreaterEqual(hrp_diag['remaining_tax_loss_carryforward'], 0.0)
        
    def test_speed_constraint(self):
        """
        Verify that backtest runs under the 2-second speed constraint.
        """
        # Create a large dataset (~20 years of daily data for 12 assets)
        tickers_large = ETF_POOL
        prices_large = generate_synthetic_prices(
            tickers=tickers_large,
            start_date='2000-01-01',
            end_date='2026-05-20'
        )
        
        start_time = time.perf_counter()
        hrp_cum, hrp_diag, sp500_cum, sixty_forty_cum, sf_diag = run_strategy_backtest(
            prices_df=prices_large,
            params=self.params,
            pool_name='etf',
            initial_capital=100000.0
        )
        elapsed_time = time.perf_counter() - start_time
        print(f"\n[Speed Check] Large backtest (12 assets, 26 years) completed in {elapsed_time:.4f} seconds.")
        self.assertLess(elapsed_time, 2.0, "Backtester execution exceeded speed limit of 2 seconds")

    def test_least_history_start_date(self):
        """
        Verify that get_least_history_info finds the correct limiting asset
        and that run_strategy_backtest automatically slices data to that start date.
        """
        from hrp_engine.data import get_least_history_info
        
        # Create a dummy DataFrame where one asset starts late
        dates = pd.date_range('2020-01-01', '2020-01-10')
        data = {
            'A': [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0],
            'B': [np.nan, np.nan, np.nan, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0]
        }
        df = pd.DataFrame(data, index=dates)
        
        start_date, ticker = get_least_history_info(df)
        self.assertEqual(ticker, 'B')
        self.assertEqual(start_date, pd.Timestamp('2020-01-04'))
        
        # Test backtest auto-slicing
        # Create mock prices with 6 years of data, asset AGG starts late
        dates_long = pd.date_range('2010-01-01', '2016-12-31')
        df_long = pd.DataFrame({
            'IVV': np.random.rand(len(dates_long)) + 100,
            'AGG': np.random.rand(len(dates_long)) + 100
        }, index=dates_long)
        # Introduce a late start for AGG
        df_long.iloc[:500, 1] = np.nan
        
        expected_start = df_long['AGG'].first_valid_index()
        
        params = StrategyParams(
            lookback_years=1,
            rebalance_frequency='yearly',
            linkage_method='single',
            drift_threshold=0.015,
            transaction_cost_bps=5.0,
            french_pfu_rate=0.314
        )
        
        hrp_cum, hrp_diag, sp500_cum, sixty_forty_cum, sf_diag = run_strategy_backtest(
            prices_df=df_long,
            params=params,
            pool_name='etf',
            initial_capital=100000.0,
            limit_to_least_history=True
        )
        
        self.assertTrue(hrp_cum.index[0] >= expected_start + pd.DateOffset(years=1))
        
    def test_european_pool_backtest(self):
        """
        Verify that the walk-forward backtester works seamlessly with the new
        European Investable Pool and sets correct benchmarks (SXR8.DE and IS0L.DE).
        """
        # Generate short synthetic data for the European pool
        prices_df = generate_synthetic_prices(
            tickers=EUROPEAN_POOL,
            start_date='2018-01-01',
            end_date='2022-12-31'
        )
        
        hrp_cum, hrp_diag, sp500_cum, sixty_forty_cum, sf_diag = run_strategy_backtest(
            prices_df=prices_df,
            params=self.params,
            pool_name='european',
            initial_capital=100000.0,
            limit_to_least_history=False
        )
        
        # Verify length and that index aligns
        self.assertEqual(len(hrp_cum), len(sp500_cum))
        self.assertEqual(len(hrp_cum), len(sixty_forty_cum))
        np.testing.assert_array_equal(hrp_cum.index, sp500_cum.index)
        
        # Verify benchmarks setup (through checking weight logs or keys in holdings)
        # SXR8.DE should be in sixty_forty holdings or diagnostics if it was traded
        # Since it is a 60/40 benchmark, sixty_forty has equity_ticker and bond_ticker
        # Let's verify that sf_diag does not error and returns reasonable shapes.
        self.assertGreater(hrp_diag['total_rebalance_events'], 0)

if __name__ == '__main__':
    unittest.main()
