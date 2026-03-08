"""
Unit tests for data forensics module.
"""
import pytest
import pandas as pd
import numpy as np
from src.data.forensics import (
    load_csv_data,
    validate_data_quality,
    detect_missing_bars,
    calculate_data_quality_score
)


def test_perfect_data():
    """Test with perfect OHLC data."""
    dates = pd.date_range('2024-01-01', periods=100, freq='H')
    df = pd.DataFrame({
        'open': 100.0,
        'high': 100.5,
        'low': 99.5,
        'close': 100.0,
        'volume': 1000
    }, index=dates)
    
    report = validate_data_quality(df, "test_perfect.csv")
    
    assert report.quality_score >= 95.0
    assert report.ohlc_violations == 0
    assert report.negative_spreads == 0
    assert len(report.warnings) == 0


def test_ohlc_violations():
    """Test detection of OHLC relationship violations."""
    dates = pd.date_range('2024-01-01', periods=10, freq='H')
    df = pd.DataFrame({
        'open': 100.0,
        'high': 99.0,  # High < Open (violation!)
        'low': 99.5,
        'close': 100.0,
        'volume': 1000
    }, index=dates)
    
    report = validate_data_quality(df, "test_violations.csv")
    
    assert report.ohlc_violations > 0
    assert report.quality_score < 100
    assert any('OHLC' in w for w in report.warnings)


def test_negative_spreads():
    """Test detection of negative spreads (high < low)."""
    dates = pd.date_range('2024-01-01', periods=10, freq='H')
    df = pd.DataFrame({
        'open': 100.0,
        'high': 99.0,  # High < Low (impossible!)
        'low': 100.0,
        'close': 99.5,
        'volume': 1000
    }, index=dates)
    
    report = validate_data_quality(df, "test_negative_spread.csv")
    
    assert report.negative_spreads > 0
    assert report.quality_score < 50  # Critical error
    assert any('Negative spread' in w for w in report.warnings)


def test_missing_bars_detection():
    """Test gap detection in time series."""
    # Create data with a gap
    dates1 = pd.date_range('2024-01-01', periods=50, freq='H')
    dates2 = pd.date_range('2024-01-05', periods=50, freq='H')
    dates = dates1.union(dates2)
    
    df = pd.DataFrame({
        'open': 100.0,
        'high': 100.5,
        'low': 99.5,
        'close': 100.0,
        'volume': 1000
    }, index=dates)
    
    report = validate_data_quality(df, "test_gaps.csv")
    
    assert report.missing_bars > 0
    assert any('Missing bars' in w for w in report.warnings)


def test_outlier_detection():
    """Test detection of statistical outliers."""
    dates = pd.date_range('2024-01-01', periods=100, freq='H')
    prices = 100.0 + np.random.randn(100) * 0.1
    prices[50] = 150.0  # Huge outlier (50% jump!)
    
    df = pd.DataFrame({
        'open': prices,
        'high': prices + 0.5,
        'low': prices - 0.5,
        'close': prices,
        'volume': 1000
    }, index=dates)
    
    report = validate_data_quality(df, "test_outliers.csv")
    
    assert report.outliers > 0
    assert any('Outliers' in w for w in report.warnings)


def test_quality_score_calculation():
    """Test quality score calculation logic."""
    # Perfect data
    score = calculate_data_quality_score(
        total_bars=1000,
        ohlc_violations=0,
        missing_bars=0,
        outliers=0,
        negative_spreads=0,
        zero_volume_bars=0
    )
    assert score == 100.0
    
    # Data with some issues
    score = calculate_data_quality_score(
        total_bars=1000,
        ohlc_violations=5,
        missing_bars=10,
        outliers=2,
        negative_spreads=0,
        zero_volume_bars=5
    )
    assert 0 < score < 100


def test_load_real_data():
    """Test loading actual CSV file from data/raw."""
    try:
        df = load_csv_data("data/raw/USDJPY60.csv", has_header=False)
        
        assert len(df) > 0
        assert 'open' in df.columns
        assert 'high' in df.columns
        assert 'low' in df.columns
        assert 'close' in df.columns
        assert isinstance(df.index, pd.DatetimeIndex)
        
        print(f"\n✓ Loaded {len(df)} bars from USDJPY60.csv")
        print(f"  Period: {df.index.min()} to {df.index.max()}")
        
    except FileNotFoundError:
        pytest.skip("USDJPY60.csv not found in data/raw")


def test_validate_real_usdjpy_data():
    """Run full validation on actual USD/JPY data."""
    try:
        df = load_csv_data("data/raw/USDJPY60.csv", has_header=False)
        report = validate_data_quality(df, "data/raw/USDJPY60.csv", expected_freq='H')
        
        print(f"\n{'='*60}")
        print(f"USD/JPY Data Quality: {report.quality_score:.2f}/100")
        print(f"Total Bars: {report.total_bars:,}")
        print(f"Missing Bars: {report.missing_bars}")
        print(f"OHLC Violations: {report.ohlc_violations}")
        print(f"Outliers: {report.outliers}")
        print(f"Negative Spreads: {report.negative_spreads}")
        print(f"{'='*60}")
        
        # Basic quality assertions
        assert report.total_bars > 0
        assert report.quality_score > 0
        assert report.negative_spreads == 0, "Data has negative spreads (critical error)"
        
        # Quality should be at least acceptable
        if report.quality_score < 70:
            print(f"\nWARNINGS:")
            for w in report.warnings:
                print(f"  - {w}")
        
    except FileNotFoundError:
        pytest.skip("USDJPY60.csv not found in data/raw")


def test_empty_dataframe():
    """Test handling of empty DataFrame."""
    df = pd.DataFrame()
    report = validate_data_quality(df, "empty.csv")
    
    assert report.total_bars == 0
    assert report.quality_score == 0.0
    assert len(report.warnings) > 0


def test_missing_columns():
    """Test handling of missing OHLC columns."""
    dates = pd.date_range('2024-01-01', periods=10, freq='H')
    df = pd.DataFrame({
        'price': 100.0  # Missing OHLC columns
    }, index=dates)
    
    report = validate_data_quality(df, "missing_cols.csv")
    
    assert report.quality_score == 0.0
    assert any('Missing columns' in w for w in report.warnings)


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
