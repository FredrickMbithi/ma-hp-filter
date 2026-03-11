"""Feature normalization and scaling utilities.

All features must be on a comparable scale before being combined into a
multi-factor signal.  Raw features have incompatible units (RSI is 0–100,
ATR-normalised spreads are typically −3 to +3, etc.).

Two methods are provided:

Z-score normalization
    ``(x − μ) / σ`` computed over a rolling lookback window.  The result
    is in standard-deviation units, roughly −3 to +3 for well-behaved
    distributions.  Sensitive to outliers (extreme values inflate σ but
    can also produce large z-scores themselves).

Rank normalization
    Convert each bar's value to its rolling percentile rank in [0, 1].
    Completely robust to outliers since only the rank matters, not the
    magnitude.  Produces a uniform distribution which is ideal for equal-
    weighting features that have different tail behaviours.

Rolling window convention
    Both methods use a ``window``-bar rolling lookback ending at bar t
    (no lookahead).  The first ``window - 1`` bars will be NaN.  A
    ``min_periods`` guard is applied; bars with fewer observations in the
    current window output NaN rather than noisy estimates.

Choosing ``window``
    The default is 252 × 24 = 6 048 bars (approximately one year of H1
    FX data).  This is long enough to capture full volatility cycles but
    still allows the normalizer to adapt as market regimes shift.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

__all__ = [
    "zscore_normalize",
    "rank_normalize",
    "normalize_features",
]


def zscore_normalize(
    feature: pd.Series,
    window: int = 252 * 24,
    min_periods: int = 30,
) -> pd.Series:
    """Rolling z-score normalization.

    Computes ``(x − μ_t) / σ_t`` where μ and σ are the rolling mean and
    standard deviation over the past ``window`` bars.

    Args:
        feature: Raw feature series (any units).
        window: Lookback in bars (default 6 048 ≈ 1 year H1).
        min_periods: Minimum observations before producing a value.

    Returns:
        Z-score normalized series.  Bars with zero variance output NaN.
        First ``min_periods − 1`` bars are NaN.
    """
    mu = feature.rolling(window=window, min_periods=min_periods).mean()
    sigma = feature.rolling(window=window, min_periods=min_periods).std()
    z = (feature - mu) / sigma
    z = z.replace([np.inf, -np.inf], np.nan)
    z.name = feature.name
    return z


def rank_normalize(
    feature: pd.Series,
    window: int = 252 * 24,
    min_periods: int = 30,
) -> pd.Series:
    """Rolling percentile rank normalization to [0, 1].

    At each bar t, counts how many of the previous ``window`` values are
    ≤ feature[t] and divides by the count of non-NaN values.  The result
    is a uniform distribution in [0, 1] regardless of the feature's
    original distribution.

    For signals that point in both directions (e.g., z-scored MA spread)
    you may want to centre by subtracting 0.5 after calling this function
    so that 0.0 = short and 1.0 = long becomes −0.5 = short and +0.5 =
    long.

    Args:
        feature: Raw feature series (any units).
        window: Lookback in bars (default 6 048 ≈ 1 year H1).
        min_periods: Minimum observations before producing a value.

    Returns:
        Percentile rank in [0, 1].  First ``min_periods − 1`` bars are NaN.
    """
    def _rolling_rank(arr: np.ndarray) -> float:
        """Percentile rank of the last element among all elements."""
        valid = arr[~np.isnan(arr)]
        if len(valid) == 0:
            return np.nan
        return float(np.sum(valid <= valid[-1]) / len(valid))

    result = feature.rolling(window=window, min_periods=min_periods).apply(
        _rolling_rank, raw=True
    )
    result.name = feature.name
    return result


def normalize_features(
    features: Dict[str, pd.Series],
    method: str = "zscore",
    window: int = 252 * 24,
    min_periods: int = 30,
) -> Dict[str, pd.Series]:
    """Apply normalization to a dictionary of features.

    All features must already be aligned to the same DatetimeIndex.

    Args:
        features: Mapping of feature name → raw pd.Series.
        method: "zscore" (default) or "rank".
        window: Lookback window in bars.
        min_periods: Minimum observations per window.

    Returns:
        New dictionary with the same keys; values are normalized series.

    Raises:
        ValueError: If ``method`` is not "zscore" or "rank".
    """
    if method not in ("zscore", "rank"):
        raise ValueError(f"method must be 'zscore' or 'rank'; got {method!r}")

    normalize_fn = zscore_normalize if method == "zscore" else rank_normalize
    return {
        name: normalize_fn(series, window=window, min_periods=min_periods)
        for name, series in features.items()
    }
