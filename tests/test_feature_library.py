"""Tests for FeatureLibrary feature generation methods.

Tests verify:
1. Numerical correctness
2. Boundary conditions (RSI, close_position)
3. No lookahead bias
4. Edge case handling
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.library import FeatureLibrary


class TestMomentum:
    """Test momentum feature."""
    
    def test_momentum_basic(self):
        """Test basic momentum calculation."""
        prices = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0])
        mom = FeatureLibrary.momentum(prices, period=2)
        
        # First (period) values should be NaN
        assert mom.iloc[:2].isna().all()
        
        # Check known value
        expected = np.log(102.0 / 100.0)
        assert np.isclose(mom.iloc[2], expected)
    
    def test_momentum_no_lookahead(self):
        """Verify no lookahead bias."""
        prices = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0])
        mom = FeatureLibrary.momentum(prices, period=2)
        
        # Changing future price shouldn't affect past momentum
        prices_modified = prices.copy()
        prices_modified.iloc[-1] = 999.0
        mom_modified = FeatureLibrary.momentum(prices_modified, period=2)
        
        assert np.allclose(mom.iloc[:-1].dropna(), mom_modified.iloc[:-1].dropna())


class TestVolatility:
    """Test volatility feature."""
    
    def test_volatility_basic(self):
        """Test basic volatility calculation."""
        returns = pd.Series([0.01, 0.02, -0.01, 0.03, -0.02])
        vol = FeatureLibrary.volatility(returns, window=3)
        
        # First (window-1) should be NaN
        assert vol.iloc[:2].isna().all()
        
        # Check third value
        expected = returns.iloc[:3].std()
        assert np.isclose(vol.iloc[2], expected)


class TestRSI:
    """Test Relative Strength Index."""
    
    def test_rsi_bounds(self):
        """Test that RSI is bounded [0, 100]."""
        np.random.seed(42)
        prices = pd.Series(100 + np.cumsum(np.random.randn(100)))
        rsi = FeatureLibrary.rsi(prices, period=14)
        
        # RSI should be bounded [0, 100]
        rsi_clean = rsi.dropna()
        assert (rsi_clean >= 0).all()
        assert (rsi_clean <= 100).all()
    
    def test_rsi_trending_up(self):
        """RSI should be high for uptrending prices."""
        prices = pd.Series(np.arange(100, 120))  # Strong uptrend
        rsi = FeatureLibrary.rsi(prices, period=14)
        
        # Late RSI values should be high (> 70)
        assert rsi.iloc[-1] > 70
    
    def test_rsi_trending_down(self):
        """RSI should be low for downtrending prices."""
        prices = pd.Series(np.arange(120, 100, -1))  # Strong downtrend
        rsi = FeatureLibrary.rsi(prices, period=14)
        
        # Late RSI values should be low (< 30)
        assert rsi.iloc[-1] < 30
    
    def test_rsi_no_lookahead(self):
        """Verify no lookahead bias."""
        np.random.seed(42)
        prices = pd.Series(100 + np.cumsum(np.random.randn(50)))
        rsi = FeatureLibrary.rsi(prices, period=14)
        
        # Changing future price shouldn't affect past RSI
        prices_modified = prices.copy()
        prices_modified.iloc[-1] = 999.0
        rsi_modified = FeatureLibrary.rsi(prices_modified, period=14)
        
        assert np.allclose(rsi.iloc[:-1].dropna(), rsi_modified.iloc[:-1].dropna())


class TestRangeFeature:
    """Test range feature."""
    
    def test_range_basic(self):
        """Test basic range calculation."""
        high = pd.Series([102.0, 103.5, 104.0])
        low = pd.Series([100.0, 101.0, 102.5])
        
        range_feat = FeatureLibrary.range_feature(high, low)
        
        assert np.isclose(range_feat.iloc[0], 2.0)
        assert np.isclose(range_feat.iloc[1], 2.5)
        assert np.isclose(range_feat.iloc[2], 1.5)
    
    def test_range_non_negative(self):
        """Range should always be non-negative."""
        high = pd.Series([102.0, 103.5, 104.0, 105.0])
        low = pd.Series([100.0, 101.0, 102.5, 103.0])
        
        range_feat = FeatureLibrary.range_feature(high, low)
        
        assert (range_feat >= 0).all()


class TestTrueRange:
    """Test True Range calculation."""
    
    def test_true_range_no_gap(self):
        """Test true range when no gaps."""
        high = pd.Series([102.0, 103.0, 104.0])
        low = pd.Series([100.0, 101.0, 102.0])
        close = pd.Series([101.0, 102.0, 103.0])
        
        tr = FeatureLibrary.true_range(high, low, close)
        
        # First value should be NaN (no prior close)
        assert pd.isna(tr.iloc[0])
        
        # When no gap, TR = high - low
        assert np.isclose(tr.iloc[1], 2.0)  # 103 - 101
        assert np.isclose(tr.iloc[2], 2.0)  # 104 - 102
    
    def test_true_range_with_gap_up(self):
        """Test true range with gap up."""
        high = pd.Series([102.0, 110.0])  # Gap up
        low = pd.Series([100.0, 108.0])
        close = pd.Series([101.0, 109.0])
        
        tr = FeatureLibrary.true_range(high, low, close)
        
        # Second TR should account for gap
        # TR = max(high-low, |high-prev_close|, |low-prev_close|)
        # = max(2, 9, 7) = 9
        assert np.isclose(tr.iloc[1], 9.0)
    
    def test_true_range_non_negative(self):
        """True range should always be non-negative."""
        np.random.seed(42)
        n = 50
        close = pd.Series(100 + np.cumsum(np.random.randn(n) * 0.5))
        high = close + np.abs(np.random.randn(n) * 0.3)
        low = close - np.abs(np.random.randn(n) * 0.3)
        
        tr = FeatureLibrary.true_range(high, low, close)
        
        assert (tr.dropna() >= 0).all()


class TestClosePosition:
    """Test close position within bar."""
    
    def test_close_position_bounds(self):
        """Close position should be bounded [0, 1]."""
        high = pd.Series([103.0, 105.0, 104.0])
        low = pd.Series([100.0, 102.0, 101.0])
        close = pd.Series([102.0, 103.5, 102.5])
        
        cp = FeatureLibrary.close_position(high, low, close)
        
        # Should be bounded [0, 1]
        cp_clean = cp.dropna()
        assert (cp_clean >= 0).all()
        assert (cp_clean <= 1).all()
    
    def test_close_position_at_high(self):
        """Close at high should give 1.0."""
        high = pd.Series([105.0])
        low = pd.Series([100.0])
        close = pd.Series([105.0])  # Close at high
        
        cp = FeatureLibrary.close_position(high, low, close)
        
        assert np.isclose(cp.iloc[0], 1.0)
    
    def test_close_position_at_low(self):
        """Close at low should give 0.0."""
        high = pd.Series([105.0])
        low = pd.Series([100.0])
        close = pd.Series([100.0])  # Close at low
        
        cp = FeatureLibrary.close_position(high, low, close)
        
        assert np.isclose(cp.iloc[0], 0.0)
    
    def test_close_position_at_midpoint(self):
        """Close at midpoint should give ~0.5."""
        high = pd.Series([105.0])
        low = pd.Series([100.0])
        close = pd.Series([102.5])  # Close at midpoint
        
        cp = FeatureLibrary.close_position(high, low, close)
        
        assert np.isclose(cp.iloc[0], 0.5)
    
    def test_close_position_zero_range(self):
        """Zero range should produce NaN."""
        high = pd.Series([100.0])
        low = pd.Series([100.0])
        close = pd.Series([100.0])
        
        cp = FeatureLibrary.close_position(high, low, close)
        
        # Should be NaN, not inf
        assert pd.isna(cp.iloc[0])


class TestDistanceFromMA:
    """Test distance from moving average."""
    
    def test_distance_from_ma_basic(self):
        """Test basic distance calculation."""
        prices = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0, 105.0])
        dist = FeatureLibrary.distance_from_ma(prices, window=3)
        
        # First (window-1) should be NaN
        assert dist.iloc[:2].isna().all()
    
    def test_distance_from_ma_interpretation(self):
        """Test interpretation: above MA = positive, below MA = negative."""
        # Create series with clear trend
        prices = pd.Series([100.0, 100.1, 100.2, 100.3, 100.4, 105.0])  # Jump at end
        dist = FeatureLibrary.distance_from_ma(prices, window=3)
        
        # Last value should be strongly positive (above MA)
        assert dist.iloc[-1] > 1.0
    
    def test_distance_from_ma_no_lookahead(self):
        """Verify no lookahead bias."""
        prices = pd.Series(100 + np.cumsum(np.random.randn(50) * 0.5))
        dist = FeatureLibrary.distance_from_ma(prices, window=20)
        
        # Changing future price shouldn't affect past distance
        prices_modified = prices.copy()
        prices_modified.iloc[-1] = 999.0
        dist_modified = FeatureLibrary.distance_from_ma(prices_modified, window=20)
        
        assert np.allclose(dist.iloc[:-1].dropna(), dist_modified.iloc[:-1].dropna())


class TestStatelessDesign:
    """Test that FeatureLibrary is truly stateless."""
    
    def test_no_internal_state(self):
        """Verify calling methods doesn't create internal state."""
        lib = FeatureLibrary()
        
        # Library should have no instance attributes
        assert len(vars(lib)) == 0
    
    def test_static_methods_work_without_instance(self):
        """Verify static methods work without instantiation."""
        prices = pd.Series([100.0, 101.0, 102.0])
        
        # Should work without creating an instance
        mom = FeatureLibrary.momentum(prices, period=1)
        
        assert len(mom) == 3
        assert pd.isna(mom.iloc[0])
