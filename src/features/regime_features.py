"""Regime feature generators — volatility, trend strength, and liquidity.

All generators are causal (no lookahead): the value at bar t uses only
data from bars ≤ t.  All return pd.Series with the same DatetimeIndex as
the input.

New features in this module
---------------------------
trend_strength
    Rolling percentile rank of |HP slope| / ATR.  Measures how strong the
    current HP trend is relative to its own history.  High values indicate
    a strong trend; low values indicate a choppy or ranging market.

trend_consistency
    Fraction of bars in a rolling window where the HP slope has the same
    sign as the current slope.  High values (close to 1.0) indicate a
    sustained directional trend; low values indicate choppy conditions.

hl_spread_ratio
    (High − Low) / ATR.  An intrabar liquidity proxy: a ratio > 1 means
    the current bar's range exceeds the average volatility, indicating
    wider-than-normal spreads or a news-driven move.

vol_zscore
    Z-score of realized volatility relative to its own rolling history.
    Positive = currently elevated vol; negative = calm regime.

Re-exported from generators.py (unified regime API)
----------------------------------------------------
vol_regime
    Categorical regime: 0 = low, 1 = medium, 2 = high.

vol_rolling_percentile
    Rolling percentile rank [0, 100] of realized volatility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Re-export the existing generators so callers can import everything
# regime-related from one place.
from .generators import vol_regime, vol_rolling_percentile  # noqa: F401

__all__ = [
    # New generators
    "trend_strength",
    "trend_consistency",
    "hl_spread_ratio",
    "vol_zscore",
    # Re-exports for unified API
    "vol_regime",
    "vol_rolling_percentile",
]


def trend_strength(
    trend: pd.Series,
    atr_series: pd.Series,
    window: int = 20,
    lookback: int = 252 * 24,
    min_periods: int = 30,
) -> pd.Series:
    """Rolling percentile rank of |HP trend slope| / ATR.

    Quantifies how strong the current HP trend is relative to its own
    historical distribution.  A value of 80 means the current trend slope
    is stronger than 80% of observed slopes in the lookback window.

    Args:
        trend: Causal HP trend series (from ``causal_hp_trend``).
        atr_series: ATR normalizer series (same index).
        window: Slope window: slope = trend[t] − trend[t-window+1].
            Default 20 bars gives a short-term slope estimate.
        lookback: Percentile-rank lookback in bars (default 1 year H1).
        min_periods: Minimum observations before producing a value.

    Returns:
        Trend strength as a rolling percentile in [0, 100].
        First (window + min_periods − 2) bars will be NaN.
    """
    slope = trend.diff(window)
    norm_slope = slope.abs() / atr_series.replace(0, np.nan)

    def _rank(arr: np.ndarray) -> float:
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            return np.nan
        return float(np.sum(valid <= valid[-1]) / len(valid)) * 100.0

    result = norm_slope.rolling(window=lookback, min_periods=min_periods).apply(
        _rank, raw=True
    )
    result.name = "trend_strength"
    return result


def trend_consistency(
    slope: pd.Series,
    window: int = 48,
    min_periods: int = 10,
) -> pd.Series:
    """Fraction of recent bars where HP slope sign matches the current sign.

    A value of 1.0 means every bar in the window has the same slope
    direction as the current bar.  A value of 0.5 means the trend has
    been choppy (equally split).

    This is a trend conviction / persistence measure.  It is most useful
    as a filter: only trade when trend_consistency > 0.7 (strong trend).

    Args:
        slope: HP trend slope series (hp_trend_slope).
        window: Rolling window in bars (default 48 bars ≈ 2 days H1).
        min_periods: Minimum observations per window.

    Returns:
        Consistency fraction in [0, 1].
        First (min_periods − 1) bars are NaN.
    """
    signs = np.sign(slope)

    def _consistency(arr: np.ndarray) -> float:
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            return np.nan
        current_sign = valid[-1]
        if current_sign == 0:
            return np.nan
        return float(np.sum(valid == current_sign) / len(valid))

    result = signs.rolling(window=window, min_periods=min_periods).apply(
        _consistency, raw=True
    )
    result.name = "trend_consistency"
    return result


def hl_spread_ratio(
    high: pd.Series,
    low: pd.Series,
    atr_series: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Intrabar high-low range normalized by ATR.

    Measures whether the current bar's range is wider or narrower than
    the recent average range:
    - Ratio > 1.0: abnormally wide bar (news event, liquidity crunch)
    - Ratio < 0.5: very narrow bar (low-information period)

    Can be used as a liquidity proxy or entry-quality filter.

    Args:
        high: Bar high prices.
        low: Bar low prices.
        atr_series: ATR series used as normalizer (same index).
        window: ATR averaging window (used only for min_periods alignment;
            the atr_series is already computed over its own window).

    Returns:
        Rolling H−L / ATR ratio.  Bars where ATR = 0 output NaN.
    """
    hl_range = high - low
    ratio = hl_range / atr_series.replace(0, np.nan)
    ratio = ratio.rolling(window=window, min_periods=1).mean()
    ratio.name = "hl_spread_ratio"
    return ratio


def vol_zscore(
    returns: pd.Series,
    vol_window: int = 20,
    lookback: int = 252 * 24,
    min_periods: int = 30,
) -> pd.Series:
    """Z-score of realized volatility relative to its rolling history.

    Steps:
    1. Compute realized vol = rolling_std(returns, vol_window).
    2. Compute rolling mean and std of that vol series over ``lookback``.
    3. Return (vol − vol_mean) / vol_std.

    Positive values indicate currently elevated volatility.
    Negative values indicate a calm / compressed regime.

    Args:
        returns: Log or arithmetic return series.
        vol_window: Window for realized volatility calculation (default 20).
        lookback: Window for the vol's own mean/std (default 1 year H1).
        min_periods: Minimum observations before outputting a value.

    Returns:
        Z-score of realized vol.  Bars with zero vol_std output NaN.
    """
    realized_vol = returns.rolling(window=vol_window, min_periods=vol_window).std()
    vol_mean = realized_vol.rolling(window=lookback, min_periods=min_periods).mean()
    vol_std = realized_vol.rolling(window=lookback, min_periods=min_periods).std()
    z = (realized_vol - vol_mean) / vol_std.replace(0, np.nan)
    z = z.replace([np.inf, -np.inf], np.nan)
    z.name = "vol_zscore"
    return z
