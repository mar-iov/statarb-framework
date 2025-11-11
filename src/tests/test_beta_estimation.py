"""
Unit tests for beta estimation methods.

Tests static OLS, rolling window, and Kalman filter implementations.
"""

import unittest
import numpy as np
import pandas as pd
from statarb.beta_estimation import (
    estimate_static_beta,
    estimate_rolling_beta,
    estimate_kalman_beta,
    get_beta_series
)


class TestStaticBeta(unittest.TestCase):
    """Test static OLS beta estimation."""

    def setUp(self):
        """Create synthetic data with known beta."""
        np.random.seed(42)
        n = 1000
        self.S2 = pd.Series(np.cumsum(np.random.randn(n)), name='S2')
        self.true_beta = 1.5
        self.S1 = self.true_beta * self.S2 + np.random.randn(n) * 0.1
        self.S1.name = 'S1'

    def test_beta_recovery(self):
        """Beta estimate should be close to true value."""
        beta = estimate_static_beta(self.S1, self.S2)
        self.assertAlmostEqual(beta, self.true_beta, delta=0.1)

    def test_empty_data_raises(self):
        """Empty data should raise ValueError."""
        empty = pd.Series([], dtype=float)
        with self.assertRaises(ValueError):
            estimate_static_beta(empty, empty)

    def test_return_type(self):
        """Should return Python float, not numpy."""
        beta = estimate_static_beta(self.S1, self.S2)
        self.assertIsInstance(beta, float)


class TestRollingBeta(unittest.TestCase):
    """Test rolling window beta estimation."""

    def setUp(self):
        np.random.seed(42)
        n = 200
        self.S2 = pd.Series(np.cumsum(np.random.randn(n)))
        self.S1 = 1.0 * self.S2 + np.random.randn(n) * 0.1

    def test_output_length(self):
        """Output should match input length."""
        beta_series = estimate_rolling_beta(self.S1, self.S2, window=20)
        self.assertEqual(len(beta_series), len(self.S1))

    def test_initial_nans(self):
        """First window-1 values should be NaN."""
        window = 20
        beta_series = estimate_rolling_beta(self.S1, self.S2, window=window)
        self.assertTrue(beta_series.iloc[:window - 1].isna().all())

    def test_min_periods(self):
        """Min periods should control when estimates start."""
        beta_series = estimate_rolling_beta(self.S1, self.S2, window=50, min_periods=10)
        # Should have estimates before bar 50
        self.assertFalse(beta_series.iloc[10:20].isna().all())


class TestKalmanBeta(unittest.TestCase):
    """Test Kalman filter beta estimation."""

    def setUp(self):
        np.random.seed(42)
        n = 500
        # Time-varying beta
        beta_true = 1.0 + 0.5 * np.sin(np.linspace(0, 4 * np.pi, n))
        self.S2 = pd.Series(np.cumsum(np.random.randn(n)))
        self.S1 = pd.Series(beta_true * self.S2.values + np.random.randn(n) * 0.5)

    def test_output_length(self):
        """Output should match input length."""
        beta_series = estimate_kalman_beta(self.S1, self.S2)
        self.assertEqual(len(beta_series), len(self.S1))

    def test_adapts_to_changes(self):
        """Beta should track time-varying relationship."""
        beta_series = estimate_kalman_beta(self.S1, self.S2, delta=1e-4)
        # Should have variation (not constant)
        self.assertGreater(beta_series.std(), 0.1)

    def test_delta_effect(self):
        """Smaller delta should give smoother beta."""
        beta_smooth = estimate_kalman_beta(self.S1, self.S2, delta=1e-6)
        beta_adaptive = estimate_kalman_beta(self.S1, self.S2, delta=1e-3)
        # Adaptive should have higher variance
        self.assertGreater(beta_adaptive.std(), beta_smooth.std())


class TestGetBetaSeries(unittest.TestCase):
    """Test unified beta interface."""

    def setUp(self):
        np.random.seed(42)
        n = 100
        self.S2 = pd.Series(np.cumsum(np.random.randn(n)))
        self.S1 = 1.5 * self.S2 + np.random.randn(n) * 0.1

    def test_scalar_beta(self):
        """Scalar beta should be returned as-is."""
        beta = get_beta_series(self.S1, self.S2, beta=1.5)
        self.assertEqual(beta, 1.5)

    def test_series_beta(self):
        """Series beta should be aligned and returned."""
        beta_input = pd.Series(np.random.rand(len(self.S1)), index=self.S1.index)
        beta_output = get_beta_series(self.S1, self.S2, beta=beta_input)
        pd.testing.assert_series_equal(beta_output, beta_input.astype(float))

    def test_static_method(self):
        """Static method should return scalar."""
        beta = get_beta_series(self.S1, self.S2, beta_method='static')
        self.assertIsInstance(beta, float)

    def test_rolling_method(self):
        """Rolling method should return Series."""
        beta = get_beta_series(self.S1, self.S2, beta_method='rolling',
                               beta_kwargs={'window': 20})
        self.assertIsInstance(beta, pd.Series)

    def test_kalman_method(self):
        """Kalman method should return Series."""
        beta = get_beta_series(self.S1, self.S2, beta_method='kalman')
        self.assertIsInstance(beta, pd.Series)

    def test_invalid_method_raises(self):
        """Invalid method should raise ValueError."""
        with self.assertRaises(ValueError):
            get_beta_series(self.S1, self.S2, beta_method='invalid')


if __name__ == '__main__':
    unittest.main()