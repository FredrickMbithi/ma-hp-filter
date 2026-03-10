"""
Tests for FXDataLoader and data validation pipeline.

Tests ensure:
- Timestamp normalization to UTC
- Duplicate/monotonic index enforcement
- OHLC structural integrity
- Gap detection (excluding weekends)
- Spike detection (>5σ)
- No lookahead bias in features
- Gap filling policy
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import (
    FXDataLoader,
    DataLoadError,
    DataValidationError,
    create_target_variable,
    safe_train_test_split,
)
from src.data.validator import (
    ValidationResult,
    check_duplicates,
    check_monotonic,
    check_ohlc_integrity,
    check_missing_bars,
    check_extreme_spikes,
    check_timezone,
    run_all_checks,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def valid_ohlc_df():
    """Create valid OHLC DataFrame with UTC index."""
    dates = pd.date_range(
        start='2025-01-06 00:00',  # Monday
        periods=100,
        freq='1h',
        tz='UTC'
    )
    np.random.seed(42)
    
    close = 150.0 + np.cumsum(np.random.randn(100) * 0.1)
    high = close + np.abs(np.random.randn(100) * 0.05)
    low = close - np.abs(np.random.randn(100) * 0.05)
    open_ = low + (high - low) * np.random.rand(100)
    
    return pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': np.random.randint(1000, 10000, 100)
    }, index=dates)


@pytest.fixture
def df_with_duplicates(valid_ohlc_df):
    """DataFrame with duplicate timestamps."""
    df = valid_ohlc_df.copy()
    # Insert duplicate row
    dup_row = df.iloc[[5]].copy()
    return pd.concat([df, dup_row]).sort_index()


@pytest.fixture
def df_with_ohlc_violations():
    """DataFrame with OHLC constraint violations."""
    dates = pd.date_range('2025-01-06', periods=5, freq='1h', tz='UTC')
    return pd.DataFrame({
        'open': [100, 102, 103, 104, 105],
        'high': [101, 103, 104, 105, 106],
        'low': [99, 103, 102, 103, 104],  # Row 1: low > open (103 > 102)
        'close': [100.5, 102.5, 103.5, 104.5, 105.5],
        'volume': [1000] * 5
    }, index=dates)


@pytest.fixture
def df_with_gaps():
    """DataFrame with missing bars (not weekends)."""
    dates = pd.date_range('2025-01-06', periods=10, freq='1h', tz='UTC')
    # Remove bars 3 and 7 (still weekday)
    dates = dates.delete([3, 7])
    
    return pd.DataFrame({
        'open': [100] * 8,
        'high': [101] * 8,
        'low': [99] * 8,
        'close': [100.5] * 8,
        'volume': [1000] * 8
    }, index=dates)


@pytest.fixture
def df_with_spikes(valid_ohlc_df):
    """DataFrame with extreme price spikes."""
    df = valid_ohlc_df.copy()
    # Insert 10σ spike
    df.iloc[50, df.columns.get_loc('close')] = df['close'].iloc[49] * 1.10  # 10% jump
    return df


# ============================================================================
# Timestamp Normalization Tests
# ============================================================================

class TestTimestampNormalization:
    """Tests for UTC enforcement and index validation."""
    
    def test_utc_localized(self, valid_ohlc_df):
        """Verify index is UTC-localized."""
        result = check_timezone(valid_ohlc_df)
        assert result.is_valid
        assert valid_ohlc_df.index.tz is not None
        assert str(valid_ohlc_df.index.tz) == 'UTC'
    
    def test_naive_timezone_fails(self, valid_ohlc_df):
        """Naive timezone should fail validation."""
        df = valid_ohlc_df.copy()
        df.index = df.index.tz_localize(None)
        
        result = check_timezone(df)
        assert not result.is_valid
        assert 'timezone-naive' in result.message
    
    def test_monotonic_valid(self, valid_ohlc_df):
        """Valid sorted index passes monotonic check."""
        result = check_monotonic(valid_ohlc_df)
        assert result.is_valid
    
    def test_monotonic_invalid(self, valid_ohlc_df):
        """Unsorted index fails monotonic check."""
        df = valid_ohlc_df.iloc[::-1]  # Reverse order
        result = check_monotonic(df)
        assert not result.is_valid


# ============================================================================
# Duplicate Detection Tests
# ============================================================================

class TestDuplicateDetection:
    """Tests for duplicate timestamp handling."""
    
    def test_no_duplicates(self, valid_ohlc_df):
        """Clean data has no duplicates."""
        result = check_duplicates(valid_ohlc_df)
        assert result.is_valid
    
    def test_duplicates_detected(self, df_with_duplicates):
        """Duplicates are detected."""
        result = check_duplicates(df_with_duplicates)
        assert not result.is_valid
        assert 'duplicate' in result.message.lower()
        assert result.details is not None


# ============================================================================
# OHLC Integrity Tests
# ============================================================================

class TestOHLCIntegrity:
    """Tests for OHLC structural constraints."""
    
    def test_valid_ohlc(self, valid_ohlc_df):
        """Valid OHLC data passes integrity check."""
        result = check_ohlc_integrity(valid_ohlc_df)
        assert result.is_valid
    
    def test_low_greater_than_open_fails(self, df_with_ohlc_violations):
        """low > open violation is caught."""
        result = check_ohlc_integrity(df_with_ohlc_violations)
        assert not result.is_valid
        assert 'low > open' in str(result.warnings)
    
    def test_missing_columns_fails(self, valid_ohlc_df):
        """Missing OHLC columns fail validation."""
        df = valid_ohlc_df.drop(columns=['high'])
        result = check_ohlc_integrity(df)
        assert not result.is_valid
        assert 'Missing columns' in result.message


# ============================================================================
# Gap Detection Tests
# ============================================================================

class TestGapDetection:
    """Tests for missing bar detection."""
    
    def test_no_gaps(self, valid_ohlc_df):
        """Continuous data has no gaps."""
        result = check_missing_bars(valid_ohlc_df, freq='1h')
        assert result.is_valid
    
    def test_weekday_gaps_detected(self, df_with_gaps):
        """Weekday gaps are detected."""
        result = check_missing_bars(df_with_gaps, freq='1h')
        assert not result.is_valid
        assert result.details is not None
        assert len(result.details) == 2  # Two missing bars
    
    def test_weekend_gaps_ignored(self):
        """Weekend gaps are not reported as missing."""
        # Create data that spans a weekend
        # Friday 2025-01-10 to Monday 2025-01-13
        friday = pd.date_range('2025-01-10 20:00', periods=4, freq='1h', tz='UTC')
        monday = pd.date_range('2025-01-13 00:00', periods=4, freq='1h', tz='UTC')
        dates = friday.append(monday)
        
        df = pd.DataFrame({
            'open': [100] * 8,
            'high': [101] * 8,
            'low': [99] * 8,
            'close': [100.5] * 8,
            'volume': [1000] * 8
        }, index=dates)
        
        result = check_missing_bars(df, freq='1h', exclude_weekends=True)
        assert result.is_valid  # Weekend gaps are OK


# ============================================================================
# Spike Detection Tests
# ============================================================================

class TestSpikeDetection:
    """Tests for extreme return detection."""
    
    def test_no_spikes(self, valid_ohlc_df):
        """Normal data has no extreme spikes."""
        result = check_extreme_spikes(valid_ohlc_df, threshold=5.0)
        assert result.is_valid
        assert result.details is None or len(result.details) == 0
    
    def test_spikes_detected(self, df_with_spikes):
        """Extreme returns are flagged."""
        result = check_extreme_spikes(df_with_spikes, threshold=5.0)
        # Result is still valid (not a critical failure) but has details
        assert result.details is not None
        assert len(result.details) > 0
        assert 'z_score' in result.details.columns


# ============================================================================
# Lookahead Bias Prevention Tests
# ============================================================================

class TestNoLookahead:
    """Critical tests to ensure no future data leakage."""
    
    def test_log_returns_use_past_only(self, valid_ohlc_df):
        """Log returns only use past prices (shift 1)."""
        df = valid_ohlc_df.copy()
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        
        # First return should be NaN (no past data)
        assert pd.isna(df['log_return'].iloc[0])
        
        # Each return uses only data up to that point
        for i in range(1, len(df)):
            expected = np.log(df['close'].iloc[i] / df['close'].iloc[i-1])
            actual = df['log_return'].iloc[i]
            assert np.isclose(actual, expected)
    
    def test_target_uses_future_data(self, valid_ohlc_df):
        """Target variable explicitly uses future data."""
        df = valid_ohlc_df.copy()
        target = create_target_variable(df, horizon=1)
        
        # Last target should be NaN (no future data)
        assert pd.isna(target.iloc[-1])
        
        # Target looks ahead
        for i in range(len(df) - 1):
            expected = np.log(df['close'].iloc[i+1] / df['close'].iloc[i])
            actual = target.iloc[i]
            assert np.isclose(actual, expected)
    
    def test_rolling_features_no_leakage(self, valid_ohlc_df):
        """Rolling statistics don't leak future data."""
        df = valid_ohlc_df.copy()
        window = 5
        
        df['ma5'] = df['close'].rolling(window).mean()
        
        # First (window-1) values should be NaN
        assert df['ma5'].iloc[:window-1].isna().all()
        
        # Each MA uses only past data
        for i in range(window, len(df)):
            expected = df['close'].iloc[i-window+1:i+1].mean()
            actual = df['ma5'].iloc[i]
            assert np.isclose(actual, expected)
    
    def test_train_test_split_temporal(self, valid_ohlc_df):
        """Train/test split is chronological, not random."""
        train, test = safe_train_test_split(valid_ohlc_df, test_ratio=0.2)
        
        # Train should be earlier than test
        assert train.index.max() < test.index.min()
        
        # No overlap
        assert len(train.index.intersection(test.index)) == 0
    
    def test_train_test_split_with_gap(self, valid_ohlc_df):
        """Gap bars between train and test are excluded."""
        train, test = safe_train_test_split(
            valid_ohlc_df, 
            test_ratio=0.2, 
            gap_bars=5
        )
        
        # Gap should exist between train end and test start
        train_end_idx = valid_ohlc_df.index.get_loc(train.index[-1])
        test_start_idx = valid_ohlc_df.index.get_loc(test.index[0])
        
        assert test_start_idx - train_end_idx > 5


# ============================================================================
# Gap Filling Tests
# ============================================================================

class TestGapFilling:
    """Tests for missing bar fill policy."""
    
    def test_forward_fill_close(self, df_with_gaps):
        """Close prices forward-filled correctly."""
        loader = FXDataLoader('data/raw')
        filled = loader.fill_gaps(df_with_gaps, method='ffill')
        
        # Should have no NaN in close after fill
        assert not filled['close'].isna().any()
    
    def test_filled_bar_uses_prev_close(self):
        """Filled bars use previous close for OHLC."""
        dates = pd.date_range('2025-01-06', periods=3, freq='1h', tz='UTC')
        df = pd.DataFrame({
            'open': [100, np.nan, 102],
            'high': [101, np.nan, 103],
            'low': [99, np.nan, 101],
            'close': [100.5, np.nan, 102.5],
            'volume': [1000, np.nan, 1000]
        }, index=dates)
        
        loader = FXDataLoader('data/raw')
        filled = loader.fill_gaps(df, method='ffill')
        
        # Middle bar should use prev close (100.5) for all prices
        assert filled['close'].iloc[1] == 100.5  # ffill
        assert filled['open'].iloc[1] == 100.5   # prev close
        assert filled['high'].iloc[1] == 100.5   # prev close
        assert filled['low'].iloc[1] == 100.5    # prev close
        assert filled['volume'].iloc[1] == 0     # zero volume


# ============================================================================
# Integration Tests with Real Data
# ============================================================================

class TestRealDataIntegration:
    """Integration tests with actual data files."""
    
    @pytest.fixture
    def data_path(self):
        return Path(__file__).parent.parent / 'data' / 'raw'
    
    def test_load_usdjpy(self, data_path):
        """Load real USDJPY data and validate."""
        if not (data_path / 'USDJPY60.csv').exists():
            pytest.skip("USDJPY60.csv not found")
        
        loader = FXDataLoader(data_path)
        df = loader.load('USDJPY60')
        
        # Basic checks
        assert len(df) > 0
        assert df.index.tz is not None
        assert str(df.index.tz) == 'UTC'
        assert df.index.is_monotonic_increasing
        assert not df.index.has_duplicates
        
        # OHLC structure
        assert (df['low'] <= df['open']).all()
        assert (df['low'] <= df['close']).all()
        assert (df['high'] >= df['open']).all()
        assert (df['high'] >= df['close']).all()
    
    def test_full_validation_report(self, data_path):
        """Get full validation report on real data."""
        if not (data_path / 'USDJPY60.csv').exists():
            pytest.skip("USDJPY60.csv not found")
        
        loader = FXDataLoader(data_path)
        df, results = loader.load('USDJPY60', return_report=True)
        
        # Check all validation checks ran
        check_names = {r.check_name for r in results}
        expected = {'timezone', 'duplicates', 'monotonic', 'ohlc_integrity', 
                   'missing_bars', 'extreme_spikes'}
        assert expected.issubset(check_names)


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Tests for error conditions."""
    
    def test_file_not_found(self, tmp_path):
        """DataLoadError raised for missing file."""
        loader = FXDataLoader(tmp_path)
        with pytest.raises(DataLoadError):
            loader.load('NONEXISTENT')
    
    def test_validation_error_on_critical_failure(self, tmp_path):
        """DataValidationError raised for OHLC violations."""
        # Create file with bad data
        csv_path = tmp_path / 'BAD.csv'
        csv_path.write_text(
            "2025.01.06,00:00,100,99,101,100,1000\n"  # high < low (99 < 101)
        )
        
        loader = FXDataLoader(tmp_path)
        with pytest.raises(DataValidationError):
            loader.load('BAD')
