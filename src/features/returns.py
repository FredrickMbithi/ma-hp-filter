"""Return transformations and normalization utilities.

This module provides core return calculations and rolling statistical transforms
for financial time series. All functions are stateless and designed to prevent
lookahead bias.
"""

import numpy as np
import pandas as pd


def compute_log_returns(prices: pd.Series) -> pd.Series:
    """Compute logarithmic returns.
    
    Log returns have the additivity property:
    r(t1→t3) = r(t1→t2) + r(t2→t3)
    
    This makes them preferable for:
    - Multi-period analysis
    - Statistical modeling (closer to normal distribution)
    - Avoiding compounding errors
    
    Args:
        prices: Price series (close, open, etc.)
        
    Returns:
        Log returns series: ln(P_t / P_{t-1})
        First value will be NaN (no prior price).
    """
    prices = prices.astype(float)
    return np.log(prices).diff()


def compute_arithmetic_returns(prices: pd.Series) -> pd.Series:
    """Compute simple arithmetic returns.
    
    Simple returns have the portfolio additivity property:
    r_portfolio = w1*r1 + w2*r2
    
    Args:
        prices: Price series (close, open, etc.)
        
    Returns:
        Simple returns series: (P_t - P_{t-1}) / P_{t-1}
        First value will be NaN (no prior price).
    """
    prices = prices.astype(float)
    return prices.pct_change()


def compute_rolling_vol(returns: pd.Series, window: int = 20) -> pd.Series:
    """Compute rolling realized volatility.
    
    Uses standard deviation of returns over a rolling window.
    Setting min_periods=window prevents partial statistics from
    misleading early-window values.
    
    Args:
        returns: Return series (log or arithmetic)
        window: Lookback period for volatility calculation (default: 20)
        
    Returns:
        Rolling volatility series.
        First (window-1) values will be NaN.
    """
    return returns.rolling(window=window, min_periods=window).std()


def compute_zscore(series: pd.Series, window: int = 20) -> pd.Series:
    """Compute rolling z-score normalization.
    
    Z-score interpretation:
    - |z| < 1: Within 1 standard deviation (68% of data)
    - |z| < 2: Within 2 standard deviations (95% of data)
    - |z| > 3: Extreme outlier (0.3% of data)
    
    Normalization by volatility makes signals regime-invariant.
    In high-vol periods, same absolute move has lower z-score.
    
    Args:
        series: Any numeric series (returns, prices, features)
        window: Lookback period for mean/std calculation (default: 20)
        
    Returns:
        Z-score normalized series: (x - μ) / σ
        Infinities are replaced with NaN (zero volatility case).
        First (window-1) values will be NaN.
    """
    mu = series.rolling(window, min_periods=window).mean()
    sigma = series.rolling(window, min_periods=window).std()
    
    z = (series - mu) / sigma
    z = z.replace([np.inf, -np.inf], np.nan)
    
    return z
