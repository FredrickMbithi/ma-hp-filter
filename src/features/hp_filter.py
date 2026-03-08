"""
Hodrick-Prescott filter for trend-cycle decomposition.
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.filters.hp_filter import hpfilter


def apply_hp_filter(prices: pd.Series, lamb: float = 14400) -> tuple[pd.Series, pd.Series]:
    """
    Apply HP filter to extract trend and cycle components.
    
    Args:
        prices: Price series (close prices)
        lamb: Smoothness parameter (higher = smoother trend)
              - Daily: ~1600
              - Hourly: ~14400
              - 15min: ~129600
    
    Returns:
        trend: Smooth trend component
        cycle: Cyclical component (prices - trend)
    """
    cycle, trend = hpfilter(prices, lamb=lamb)
    return pd.Series(trend, index=prices.index), pd.Series(cycle, index=prices.index)


def cycle_zscore(cycle: pd.Series, window: int = 50) -> pd.Series:
    """
    Compute rolling z-score of cycle component.
    
    Args:
        cycle: HP cycle series
        window: Lookback for mean/std
    
    Returns:
        Z-score normalized cycle
    """
    rolling_mean = cycle.rolling(window).mean()
    rolling_std = cycle.rolling(window).std()
    return (cycle - rolling_mean) / rolling_std
