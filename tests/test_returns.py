"""Tests for return transformation functions.

Tests verify:
1. Numerical correctness
2. No lookahead bias
3. Edge case handling (NaN, inf, zero volatility)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.returns import (
    compute_log_returns,
    compute_arithmetic_returns,
    compute_rolling_vol,
    compute_zscore,
)


class TestLogReturns:
    """Test log return calculations."""
    
    def test_log_returns_basic(self):
        """Test basic log return calculation."""
        prices = pd.Series([100.0, 101.0, 102.0, 101.5])
        returns = compute_log_returns(prices)
        
        # First value should be NaN
        assert pd.isna(returns.iloc[0])
        
        # Check known values
        expected_r1 = np.log(101.0 / 100.0)
        assert np.isclose(returns.iloc[1], expected_r1)
        
        expected_r2 = np.log(102.0 / 101.0)
        assert np.isclose(returns.iloc[2], expected_r2)
    
    def test_log_returns_no_lookahead(self):
        """Verify no lookahead bias: return at t depends only on t and t-1."""
        prices = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0])
        returns = compute_log_returns(prices)
        
        # Changing future prices shouldn't affect past returns
        prices_modified = prices.copy()
        prices_modified.iloc[-1] = 999.0  # Change last price
        returns_modified = compute_log_returns(prices_modified)
        
        # All returns except the last should be identical
        assert np.allclose(returns.iloc[:-1].dropna(), returns_modified.iloc[:-1].dropna())
    
    def test_log_returns_additivity(self):
        """Test additive property of log returns."""
        prices = pd.Series([100.0, 102.0, 105.0, 103.0])
        returns = compute_log_returns(prices)
        
        # Multi-period return should equal sum of single-period returns
        multi_period = np.log(prices.iloc[3] / prices.iloc[0])
        sum_returns = returns.iloc[1:4].sum()
        
        assert np.isclose(multi_period, sum_returns)


class TestArithmeticReturns:
    """Test arithmetic return calculations."""
    
    def test_arithmetic_returns_basic(self):
        """Test basic arithmetic return calculation."""
        prices = pd.Series([100.0, 110.0, 105.0])
        returns = compute_arithmetic_returns(prices)
        
        # First value should be NaN
        assert pd.isna(returns.iloc[0])
        
        # Check known values
        assert np.isclose(returns.iloc[1], 0.10)  # 10% gain
        assert np.isclose(returns.iloc[2], -0.045454545)  # ~4.5% loss


class TestRollingVolatility:
    """Test rolling volatility calculations."""
    
    def test_rolling_vol_window_nan(self):
        """First (window-1) values should be NaN."""
        returns = pd.Series(np.random.randn(100) * 0.01)
        window = 20
        vol = compute_rolling_vol(returns, window=window)
        
        # First (window-1) values should be NaN
        assert vol.iloc[:window-1].isna().all()
        
        # From window onwards, should have values
        assert not vol.iloc[window:].isna().any()
    
    def test_rolling_vol_values(self):
        """Test volatility calculation against manual computation."""
        returns = pd.Series([0.01, 0.02, -0.01, 0.03, -0.02])
        window = 3
        vol = compute_rolling_vol(returns, window=window)
        
        # Check third value (first non-NaN)
        expected = returns.iloc[:3].std()
        assert np.isclose(vol.iloc[2], expected)
        
        # Check fourth value
        expected = returns.iloc[1:4].std()
        assert np.isclose(vol.iloc[3], expected)
    
    def test_rolling_vol_no_lookahead(self):
        """Verify no lookahead bias in volatility calculation."""
        returns = pd.Series(np.random.randn(50) * 0.01)
        vol = compute_rolling_vol(returns, window=20)
        
        # Changing future returns shouldn't affect past volatility
        returns_modified = returns.copy()
        returns_modified.iloc[-1] = 999.0
        vol_modified = compute_rolling_vol(returns_modified, window=20)
        
        # All volatility values except the last should be identical
        assert np.allclose(vol.iloc[:-1].dropna(), vol_modified.iloc[:-1].dropna())


class TestZScore:
    """Test z-score normalization."""
    
    def test_zscore_basic(self):
        """Test z-score calculation."""
        series = pd.Series([10.0, 12.0, 14.0, 11.0, 13.0])
        window = 3
        z = compute_zscore(series, window=window)
        
        # First (window-1) should be NaN
        assert z.iloc[:window-1].isna().all()
        
        # Check third value (first non-NaN)
        mu = series.iloc[:3].mean()
        sigma = series.iloc[:3].std()
        expected_z = (series.iloc[2] - mu) / sigma
        assert np.isclose(z.iloc[2], expected_z)
    
    def test_zscore_inf_handling(self):
        """Test that infinities are replaced with NaN."""
        # Create series with constant values (zero volatility)
        series = pd.Series([10.0, 10.0, 10.0, 10.0, 10.0])
        z = compute_zscore(series, window=3)
        
        # Should produce NaN, not inf
        assert z.iloc[2:].isna().all()
        assert not np.isinf(z.dropna()).any()
    
    def test_zscore_interpretation(self):
        """Test z-score gives expected values."""
        # Create series: mean=10, std=2
        np.random.seed(42)
        series = pd.Series(np.random.randn(100) * 2 + 10)
        z = compute_zscore(series, window=20)
        
        # Z-scores should be roughly centered around 0
        assert np.abs(z.dropna().mean()) < 0.5
        
        # Most z-scores should be within ±3
        assert (np.abs(z.dropna()) < 3).sum() / len(z.dropna()) > 0.95
    
    def test_zscore_no_lookahead(self):
        """Verify no lookahead bias in z-score calculation."""
        series = pd.Series(np.random.randn(50))
        z = compute_zscore(series, window=20)
        
        # Changing future values shouldn't affect past z-scores
        series_modified = series.copy()
        series_modified.iloc[-1] = 999.0
        z_modified = compute_zscore(series_modified, window=20)
        
        # All z-scores except the last should be identical
        assert np.allclose(z.iloc[:-1].dropna(), z_modified.iloc[:-1].dropna())


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_series(self):
        """Test with empty series."""
        empty = pd.Series([], dtype=float)
        
        assert compute_log_returns(empty).empty
        assert compute_arithmetic_returns(empty).empty
        assert compute_rolling_vol(empty).empty
        assert compute_zscore(empty).empty
    
    def test_single_value(self):
        """Test with single value."""
        single = pd.Series([100.0])
        
        returns = compute_log_returns(single)
        assert len(returns) == 1
        assert pd.isna(returns.iloc[0])
    
    def test_all_nan(self):
        """Test with all NaN values."""
        nan_series = pd.Series([np.nan, np.nan, np.nan])
        
        assert compute_log_returns(nan_series).isna().all()
        assert compute_rolling_vol(nan_series).isna().all()
        assert compute_zscore(nan_series).isna().all()
