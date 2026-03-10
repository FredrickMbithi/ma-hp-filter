"""
Data forensics: Quality validation for raw CSV data files.

Validates individual data sources without requiring comparison to identify:
- Missing bars and gaps
- OHLC relationship violations
- Statistical outliers
- Data quality issues
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataQualityReport:
    """Container for data quality assessment results."""
    file_path: str
    total_bars: int
    date_range: Tuple[str, str]
    missing_bars: int
    ohlc_violations: int
    outliers: int
    negative_spreads: int
    zero_volume_bars: int
    quality_score: float
    warnings: List[str]
    statistics: Dict[str, float]


def load_csv_data(file_path: str, has_header: bool = None) -> pd.DataFrame:
    """
    Load OHLC data from CSV file.

    Supports two layouts:
    - Headerless (MT5/MetaTrader): ``Date, Time, Open, High, Low, Close, Volume``
    - Headered (yfinance): ``Datetime, Open, High, Low, Close, Volume[, ...]``

    Header detection is automatic when ``has_header`` is left as ``None``:
    if the first cell looks like a dot-separated date (``YYYY.MM.DD``) the
    file is treated as headerless; otherwise a header row is assumed.
    Pass ``has_header=True/False`` explicitly to override.

    Args:
        file_path: Path to CSV file
        has_header: ``True`` = file has a header row, ``False`` = no header,
                    ``None`` (default) = auto-detect

    Returns:
        DataFrame with UTC-aware DatetimeIndex and OHLC columns
    """
    path = _resolve_file_path(file_path)

    # ----- auto-detect header -----
    if has_header is None:
        probe = pd.read_csv(path, header=None, nrows=1)
        first_cell = str(probe.iloc[0, 0])
        # Dot-separated date like "2025.10.27" → headerless MT5 format
        has_header = not (first_cell.count('.') == 2)

    if has_header:
        df = pd.read_csv(path)
        df.columns = df.columns.str.lower().str.strip()

        # Identify the datetime column (may be 'datetime', 'timestamp', 'date')
        dt_col = next(
            (c for c in df.columns if c in ('datetime', 'timestamp', 'date')),
            None
        )
        if dt_col is None:
            raise ValueError(
                f"Cannot find a datetime column in {path}. "
                f"Columns found: {list(df.columns)}"
            )

        df[dt_col] = pd.to_datetime(df[dt_col], utc=True, errors='coerce')
        df = df.set_index(dt_col)
        df.index.name = 'datetime'

        # Drop metadata-only columns that carry no trading signal
        drop_cols = [c for c in df.columns if c in ('dividends', 'stock splits', 'stock_splits')]
        if drop_cols:
            df = df.drop(columns=drop_cols)
    else:
        # Headerless MT5 format: Date, Time, Open, High, Low, Close, Volume
        df = pd.read_csv(
            path,
            names=['date', 'time', 'open', 'high', 'low', 'close', 'volume']
        )
        df['datetime'] = pd.to_datetime(
            df['date'] + ' ' + df['time'],
            format='%Y.%m.%d %H:%M',
            errors='coerce'
        )
        df = df.set_index('datetime')
        df = df.drop(columns=['date', 'time'])
        df.index = df.index.tz_localize('UTC')

    # Ensure UTC-aware index regardless of path taken
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    else:
        df.index = df.index.tz_convert('UTC')

    # Sort chronologically and drop any duplicate timestamps
    df = df[~df.index.duplicated(keep='first')]
    df = df.sort_index()

    return df


def _resolve_file_path(file_path: str) -> Path:
    """Return an existing path for the requested CSV.

    The helper first checks the provided path relative to the current
    working directory. If that is missing and the path is relative, it
    also checks the project root (two levels up from this file). This
    makes notebook execution robust whether it is started from the repo
    root or the `notebooks/` directory.
    """

    path = Path(file_path)
    candidates = [path]

    if not path.is_absolute():
        project_root = Path(__file__).resolve().parents[2]
        candidates.append(project_root / path)

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    tried = ", ".join(str(c.resolve()) for c in candidates)
    raise FileNotFoundError(
        f"CSV file not found. Tried: {tried}. Current working dir: {Path.cwd()}"
    )


def validate_data_quality(df: pd.DataFrame, file_path: str = "data.csv",
                         expected_freq: str = 'h') -> DataQualityReport:
    """
    Comprehensive data quality validation for a single data source.
    
    Checks:
    - OHLC relationship consistency (high >= open/close/low, low <= all)
    - Missing bars in time series
    - Statistical outliers (>5 sigma moves)
    - Negative spreads
    - Zero volume bars
    
    Args:
        df: DataFrame with OHLC data
        file_path: Path to file (for reporting)
        expected_freq: Expected frequency ('h'=hourly, 'D'=daily, '15min'=15min)
    
    Returns:
        DataQualityReport with detailed validation results
    """
    warnings_list = []
    
    # Basic validation
    if len(df) == 0:
        return DataQualityReport(
            file_path=file_path,
            total_bars=0,
            date_range=("N/A", "N/A"),
            missing_bars=0,
            ohlc_violations=0,
            outliers=0,
            negative_spreads=0,
            zero_volume_bars=0,
            quality_score=0.0,
            warnings=["Empty DataFrame"],
            statistics={}
        )
    
    # Check required columns
    required_cols = ['open', 'high', 'low', 'close']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        warnings_list.append(f"Missing columns: {missing_cols}")
        return DataQualityReport(
            file_path=file_path,
            total_bars=len(df),
            date_range=(str(df.index.min()), str(df.index.max())),
            missing_bars=0,
            ohlc_violations=0,
            outliers=0,
            negative_spreads=0,
            zero_volume_bars=0,
            quality_score=0.0,
            warnings=warnings_list,
            statistics={}
        )
    
    # 1. Check OHLC relationships
    ohlc_violations = 0
    
    # High should be >= all other prices
    high_violations = (
        (df['high'] < df['open']) |
        (df['high'] < df['close']) |
        (df['high'] < df['low'])
    ).sum()
    
    # Low should be <= all other prices
    low_violations = (
        (df['low'] > df['open']) |
        (df['low'] > df['close']) |
        (df['low'] > df['high'])
    ).sum()
    
    ohlc_violations = high_violations + low_violations
    
    if ohlc_violations > 0:
        warnings_list.append(
            f"OHLC violations: {ohlc_violations} bars have inconsistent high/low"
        )
    
    # 2. Detect missing bars
    missing_bars_list = detect_missing_bars(df, expected_freq)
    missing_bars_count = len(missing_bars_list)
    
    if missing_bars_count > 0:
        gap_pct = missing_bars_count / (len(df) + missing_bars_count) * 100
        warnings_list.append(
            f"Missing bars: {missing_bars_count} gaps ({gap_pct:.2f}% of expected)"
        )
    
    # 3. Check for negative spreads (data errors)
    negative_spreads = (df['high'] < df['low']).sum()
    if negative_spreads > 0:
        warnings_list.append(f"Negative spreads: {negative_spreads} bars (DATA ERROR)")
    
    # 4. Check zero volume bars (if volume column exists)
    zero_volume_bars = 0
    if 'volume' in df.columns:
        zero_volume_bars = (df['volume'] == 0).sum()
        if zero_volume_bars > len(df) * 0.01:  # >1% zero volume
            warnings_list.append(
                f"Zero volume: {zero_volume_bars} bars ({zero_volume_bars/len(df)*100:.1f}%)"
            )
    
    # 5. Detect price outliers
    returns = df['close'].pct_change()
    mean_return = returns.mean()
    std_return = returns.std()
    
    outlier_threshold = 5  # 5 sigma
    outliers = (np.abs(returns - mean_return) > outlier_threshold * std_return).sum()
    
    if outliers > 0:
        max_move = returns.abs().max() * 100
        warnings_list.append(
            f"Outliers detected: {outliers} bars with >{outlier_threshold}σ moves "
            f"(max: {max_move:.2f}%)"
        )
    
    # 6. Calculate statistics
    statistics = {
        'mean_close': df['close'].mean(),
        'std_close': df['close'].std(),
        'min_close': df['close'].min(),
        'max_close': df['close'].max(),
        'mean_spread': (df['high'] - df['low']).mean(),
        'mean_return': mean_return * 100,
        'std_return': std_return * 100,
        'max_return': returns.max() * 100 if len(returns) > 0 else 0,
        'min_return': returns.min() * 100 if len(returns) > 0 else 0,
    }
    
    if 'volume' in df.columns:
        statistics['mean_volume'] = df['volume'].mean()
        statistics['total_volume'] = df['volume'].sum()
    
    # Calculate quality score (0-100)
    quality_score = calculate_data_quality_score(
        total_bars=len(df),
        ohlc_violations=ohlc_violations,
        missing_bars=missing_bars_count,
        outliers=outliers,
        negative_spreads=negative_spreads,
        zero_volume_bars=zero_volume_bars
    )
    
    # Date range
    date_range = (
        df.index.min().strftime('%Y-%m-%d %H:%M'),
        df.index.max().strftime('%Y-%m-%d %H:%M')
    )
    
    return DataQualityReport(
        file_path=file_path,
        total_bars=len(df),
        date_range=date_range,
        missing_bars=missing_bars_count,
        ohlc_violations=ohlc_violations,
        outliers=outliers,
        negative_spreads=negative_spreads,
        zero_volume_bars=zero_volume_bars,
        quality_score=quality_score,
        warnings=warnings_list,
        statistics=statistics
    )


def detect_missing_bars(df: pd.DataFrame, expected_freq: str = 'h') -> List[pd.Timestamp]:
    """
    Detect gaps in time series data.
    
    Args:
        df: DataFrame with DatetimeIndex
        expected_freq: Expected frequency ('h'=hourly, 'D'=daily, '15min'=15min)
    
    Returns:
        List of timestamps where bars are missing
    """
    if len(df) < 2:
        return []
    
    # Generate expected date range
    expected_range = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq=expected_freq
    )
    
    # For FX hourly/intraday data, exclude weekends
    if expected_freq in ['h', 'H', '15min', '15T', '5min', '5T', '1min', '1T']:
        # Exclude Saturday and most of Sunday (FX market closed)
        # Keep Friday until 22:00 UTC, reopen Sunday 22:00 UTC
        mask = (
            (expected_range.dayofweek < 5) |  # Mon-Fri
            ((expected_range.dayofweek == 4) & (expected_range.hour < 22)) |  # Fri before 22:00
            ((expected_range.dayofweek == 6) & (expected_range.hour >= 22))  # Sun after 22:00
        )
        expected_range = expected_range[mask]
    
    # Find missing timestamps
    missing = expected_range.difference(df.index)
    
    return missing.tolist()


def calculate_data_quality_score(total_bars: int, ohlc_violations: int,
                                missing_bars: int, outliers: int,
                                negative_spreads: int, zero_volume_bars: int) -> float:
    """
    Calculate overall data quality score (0-100).
    
    Scoring:
    - Start at 100 points
    - Deduct points for each issue type
    """
    score = 100.0
    
    # OHLC violations are critical errors (-5 points each, max -30)
    score -= min(ohlc_violations * 5, 30)
    
    # Negative spreads are critical data errors (-10 points each, max -30)
    score -= min(negative_spreads * 10, 30)
    
    # Missing bars (-0.5 points per 1% missing, max -20)
    if total_bars > 0:
        missing_pct = (missing_bars / (total_bars + missing_bars)) * 100
        score -= min(missing_pct * 0.5, 20)
    
    # Outliers (-0.1 point each, max -10)
    score -= min(outliers * 0.1, 10)
    
    # Zero volume bars (-0.05 points each, max -10)
    score -= min(zero_volume_bars * 0.05, 10)
    
    return max(0.0, round(score, 2))


def print_quality_report(report: DataQualityReport):
    """Print formatted data quality report."""
    print("=" * 70)
    print("DATA QUALITY REPORT")
    print("=" * 70)
    print(f"\nFile: {report.file_path}")
    print(f"Period: {report.date_range[0]} to {report.date_range[1]}")
    print(f"Total Bars: {report.total_bars:,}")
    
    print(f"\n{'QUALITY SCORE:':<30} {report.quality_score:.2f}/100")
    
    # Grade
    if report.quality_score >= 95:
        grade = "EXCELLENT ✓✓✓"
    elif report.quality_score >= 85:
        grade = "GOOD ✓✓"
    elif report.quality_score >= 70:
        grade = "ACCEPTABLE ✓"
    elif report.quality_score >= 50:
        grade = "POOR ⚠"
    else:
        grade = "CRITICAL - DO NOT USE ✗"
    print(f"{'Grade:':<30} {grade}")
    
    print(f"\n{'DATA ISSUES:':}")
    print(f"  {'Missing Bars:':<28} {report.missing_bars}")
    print(f"  {'OHLC Violations:':<28} {report.ohlc_violations}")
    print(f"  {'Statistical Outliers:':<28} {report.outliers}")
    print(f"  {'Negative Spreads:':<28} {report.negative_spreads}")
    print(f"  {'Zero Volume Bars:':<28} {report.zero_volume_bars}")
    
    print(f"\n{'STATISTICS:':}")
    for key, value in report.statistics.items():
        formatted_key = key.replace('_', ' ').title() + ':'
        if 'return' in key or 'pct' in key:
            print(f"  {formatted_key:<28} {value:.4f}%")
        elif 'volume' in key and value > 1000:
            print(f"  {formatted_key:<28} {value:,.0f}")
        else:
            print(f"  {formatted_key:<28} {value:.4f}")
    
    if report.warnings:
        print(f"\n{'WARNINGS:':}")
        for i, warning in enumerate(report.warnings, 1):
            print(f"  {i}. {warning}")
    else:
        print(f"\n{'WARNINGS:':<30} None - Data is clean ✓")
    
    print("=" * 70)


def scan_data_directory(data_dir: str = "data/raw", pattern: str = "*.csv") -> Dict[str, DataQualityReport]:
    """
    Scan directory and run quality checks on all CSV files.
    
    Args:
        data_dir: Directory to scan
        pattern: File pattern to match
    
    Returns:
        Dictionary mapping filename to quality report
    """
    data_path = Path(data_dir)
    csv_files = list(data_path.glob(pattern))
    
    if not csv_files:
        print(f"No CSV files found in {data_dir}")
        return {}
    
    reports = {}
    
    print(f"\nScanning {len(csv_files)} files in {data_dir}...\n")
    
    for csv_file in csv_files:
        try:
            print(f"Processing: {csv_file.name}...")
            df = load_csv_data(str(csv_file), has_header=False)
            report = validate_data_quality(df, str(csv_file), expected_freq='h')
            reports[csv_file.name] = report
            print(f"  ✓ Quality Score: {report.quality_score:.2f}/100")
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    return reports


if __name__ == '__main__':
    import sys
    
    print("Data Forensics - Quality Validation Tool\n")
    
    # Check if specific file provided
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        print(f"Validating: {file_path}\n")
        df = load_csv_data(file_path, has_header=False)
        report = validate_data_quality(df, file_path, expected_freq='h')
        print_quality_report(report)
    else:
        # Scan data/raw directory
        reports = scan_data_directory("data/raw")
        
        if reports:
            print("\n" + "=" * 70)
            print("SUMMARY")
            print("=" * 70)
            for filename, report in reports.items():
                print(f"{filename:<30} Quality: {report.quality_score:.2f}/100  Bars: {report.total_bars:,}")
            
            # Print detailed report for each file
            print("\n")
            for filename, report in reports.items():
                print_quality_report(report)
                print("\n")
