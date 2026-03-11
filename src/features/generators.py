"""Causal feature generators for Strategy 8.1 — MA + HP Filter (USDJPY H1).

All functions are pure and stateless.  No lookahead bias: every generator
only reads data available at bar t when computing the value at bar t.

Design notes
------------
Two-sided vs causal HP filter
    ``statsmodels.tsa.filters.hp_filter.hpfilter`` solves a global
    least-squares problem that uses *all* data points simultaneously.
    The trend at bar t depends on future prices, making it unsuitable for
    live signal generation.

    ``causal_hp_trend`` wraps the two-sided filter in a per-bar rolling
    window: at each bar t, only prices[t-window : t+1] are passed to
    hpfilter, and the *last* element of the resulting trend vector — which
    is the endpoint of a short series — is taken as S*(t).

    This is computationally expensive (O(N × window²)) but correct.
    For N=30 000 bars and window=500 this takes ~60 s on a modern CPU.
    Precompute and persist results to disk for iterative development.

    Endpoint instability: the final element of the HP trend vector has the
    highest estimation variance regardless of window length.  This is a
    known limitation of the HP filter.  A window ≥ 500 bars significantly
    reduces (but does not eliminate) the instability.  For very large
    lambda the endpoint instability worsens because high smoothness means
    the filter must extrapolate the trend from a small number of effective
    observations.

Lambda calibration
    The standard rule of thumb for scaling lambda to data frequency is:
        lambda_h = lambda_q × (observations_per_quarter / 12) ** 4
    For H1 FX data (≈504 bars per quarter) and lambda_q = 1600:
        lambda_h ≈ 1600 × (504 / 12) ** 4 ≈ 3.9e9
    Values far above this (e.g. 1e11) produce near-constant trends and
    are effectively equivalent to long-window moving averages.

Normalization convention
    ATR (average true range, 20-bar default) is used as the normalizer
    throughout rather than rolling standard deviation.  ATR is more robust
    to volatility collapses (low-vol periods where rolling std → 0 can
    produce division-by-zero or explosive z-scores).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.filters.hp_filter import hpfilter

from .moving_average import sma
from .library import FeatureLibrary

__all__ = [
    "atr",
    "causal_hp_trend",
    "hp_trend_slope",
    "hp_trend_curvature",
    "ma_spread_on_trend",
    "ma_crossover_signal",
    "ma_crossover_age",
    "trend_deviation_from_ma",
    "lambda_sensitivity_score",
    "vol_regime",
    "threshold_time_stop_signal",
    "vol_rolling_percentile",
    "crossing_threshold_signal",
]


# ---------------------------------------------------------------------------
# Shared utility
# ---------------------------------------------------------------------------

def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 20,
) -> pd.Series:
    """Rolling Average True Range.

    ATR = rolling_mean(true_range, window).

    Used as a normalizer in other generators.  Exposing it explicitly
    allows callers to precompute once and reuse across features.

    Args:
        high: High price series.
        low: Low price series.
        close: Close price series.
        window: Smoothing window (default 20).

    Returns:
        ATR series.  First (window) values will be NaN.
    """
    tr = FeatureLibrary.true_range(high, low, close)
    return tr.rolling(window=window, min_periods=window).mean()


# ---------------------------------------------------------------------------
# HP-filter based features
# ---------------------------------------------------------------------------

def causal_hp_trend(
    prices: pd.Series,
    lamb: float = 3.9e9,
    window: int = 500,
) -> pd.Series:
    """Causal (one-sided) HP-filtered trend S*(t).

    Applies ``statsmodels.hpfilter`` on a rolling window of length
    ``window`` ending at each bar t, and takes the last element of the
    resulting trend vector as S*(t).  This ensures that only data up to
    and including bar t is used at each step.

    Why not use ``apply_hp_filter`` from hp_filter.py?
        ``apply_hp_filter`` passes the *entire* price series to hpfilter,
        which is a two-sided filter.  The trend value at bar t then depends
        on future prices — a lookahead bias that would inflate all derived
        signals.  This function eliminates that bias.

    Args:
        prices: Close price series.
        lamb: HP smoothness penalty (lambda).
              Rule of thumb for H1 FX: ≈3.9e9 (see module docstring).
        window: Number of bars per rolling window.
                Must be ≥ 10.  Recommended ≥ 500 for endpoint stability.

    Returns:
        Causal HP trend S*(t).
        First (window − 1) values are NaN. Leading NaNs in ``prices`` are
        preserved until a full clean window is available.
    """
    if window < 10:
        raise ValueError(f"window must be ≥ 10, got {window}")

    filled_prices = prices.ffill()

    n = len(filled_prices)
    values = filled_prices.values.astype(float)
    trend_values = np.full(n, np.nan)

    for t in range(window - 1, n):
        window_prices = values[t - window + 1 : t + 1]
        if np.isnan(window_prices).any():
            continue
        _, trend_window = hpfilter(window_prices, lamb=lamb)
        trend_values[t] = trend_window[-1]

    return pd.Series(trend_values, index=prices.index, name="hp_trend")


def hp_trend_slope(trend: pd.Series) -> pd.Series:
    """First difference of the causal HP trend: ΔS*(t) = S*(t) − S*(t−1).

    Measures the current momentum of the clean trend.
    Expected to be stationary (I(0)) since it is the first difference
    of an I(1) series.

    Args:
        trend: Causal HP trend series from ``causal_hp_trend``.

    Returns:
        Slope series.  One additional NaN at the start beyond trend's NaNs.
    """
    return trend.diff(1).rename("hp_trend_slope")


def hp_trend_curvature(trend: pd.Series) -> pd.Series:
    """Second difference of the HP trend: Δ²S*(t) = ΔS*(t) − ΔS*(t−1).

    Measures trend acceleration.  A large negative curvature during an
    uptrend signals deceleration — the mathematical signature of exhaustion.

    WARNING: For large lambda, the HP filter minimises this quantity by
    construction, so it carries no independent information in that regime.
    Always test corr(hp_trend_slope, hp_trend_curvature) before treating
    this as an independent signal.

    Args:
        trend: Causal HP trend series from ``causal_hp_trend``.

    Returns:
        Curvature series.  Two additional NaNs at the start beyond trend's NaNs.
    """
    return trend.diff(1).diff(1).rename("hp_trend_curvature")


# ---------------------------------------------------------------------------
# MA-on-trend features
# ---------------------------------------------------------------------------

def ma_spread_on_trend(
    trend: pd.Series,
    t1: int,
    t2: int,
    atr_series: pd.Series | None = None,
) -> pd.Series:
    """Spread between the short and long SMA of the HP trend.

    spread = sma(S*, t1) − sma(S*, t2)

    If ``atr_series`` is provided, returns the ATR-normalised spread:
    spread_norm = spread / ATR, which is comparable across vol regimes.

    Stationarity is NOT guaranteed even with normalisation — test with ADF.

    Args:
        trend: Causal HP trend series S*(t).
        t1: Short MA period (bars).
        t2: Long MA period (bars), must be > t1.
        atr_series: Optional ATR series for normalisation.

    Returns:
        Spread series.  First (t2 − 1) values are NaN.
    """
    if t2 <= t1:
        raise ValueError(f"t2 ({t2}) must be greater than t1 ({t1})")

    short_ma = sma(trend, t1)
    long_ma = sma(trend, t2)
    spread = short_ma - long_ma

    if atr_series is not None:
        spread = spread / atr_series

    return spread.rename(f"ma_spread_{t1}_{t2}")


def ma_crossover_signal(
    trend: pd.Series,
    t1: int,
    t2: int,
) -> pd.Series:
    """Binary trend direction signal from MA crossover on S*(t).

    Returns +1.0 when sma(S*, t1) > sma(S*, t2) (uptrend),
    and -1.0 when sma(S*, t1) ≤ sma(S*, t2) (downtrend).

    ADF is not applicable to this bounded discrete series.
    Use as a regime indicator, not a continuous model feature.

    Args:
        trend: Causal HP trend series S*(t).
        t1: Short MA period (bars).
        t2: Long MA period (bars), must be > t1.

    Returns:
        Signal series of {+1.0, -1.0, NaN}.
        NaN for the first (t2 − 1) bars.
    """
    if t2 <= t1:
        raise ValueError(f"t2 ({t2}) must be greater than t1 ({t1})")

    spread = sma(trend, t1) - sma(trend, t2)
    signal = pd.Series(
        np.where(spread.notna(), np.where(spread > 0, 1.0, -1.0), np.nan),
        index=trend.index,
        name=f"crossover_{t1}_{t2}",
    )
    return signal


def ma_crossover_age(
    trend: pd.Series,
    t1: int,
    t2: int,
) -> pd.Series:
    """Bars elapsed since the last MA crossover on S*(t).

    A crossover is defined as a sign change in (sma(t1) − sma(t2)).
    The counter resets to 0 at each crossover and increments each bar.

    This feature is non-stationary (unbounded counter).  Normalise by
    the average crossover interval before using in a model.

    Args:
        trend: Causal HP trend series S*(t).
        t1: Short MA period (bars).
        t2: Long MA period (bars), must be > t1.

    Returns:
        Age series (integer counts).  NaN until the first crossover occurs
        and for the first (t2 − 1) bars.
    """
    if t2 <= t1:
        raise ValueError(f"t2 ({t2}) must be greater than t1 ({t1})")

    spread = sma(trend, t1) - sma(trend, t2)
    sign = np.sign(spread)

    n = len(trend)
    age_values = np.full(n, np.nan)
    current_age = np.nan
    prev_sign = np.nan

    for i in range(n):
        s = sign.iloc[i]
        if np.isnan(s):
            continue
        if np.isnan(prev_sign):
            prev_sign = s
            current_age = 0.0
        elif s != prev_sign:
            prev_sign = s
            current_age = 0.0
        else:
            current_age += 1.0
        age_values[i] = current_age

    return pd.Series(age_values, index=trend.index, name=f"crossover_age_{t1}_{t2}")


# ---------------------------------------------------------------------------
# Mean-reversion features
# ---------------------------------------------------------------------------

def trend_deviation_from_ma(
    trend: pd.Series,
    window: int,
    atr_series: pd.Series,
) -> pd.Series:
    """ATR-normalised deviation of S*(t) from its own moving average.

    deviation = (S*(t) − sma(S*, window)) / ATR

    Measures how far the trend has extended from its own average.
    Extreme positive values → overextension → potential reversal zone.

    ATR normalisation is preferred over rolling-std normalisation because
    rolling std can collapse to near-zero in quiet periods, producing
    explosive z-scores that are not economically meaningful.

    Args:
        trend: Causal HP trend series S*(t).
        window: MA lookback period (bars).
        atr_series: ATR series from ``atr()``.

    Returns:
        Normalised deviation series.  NaN for first (window − 1) bars
        and wherever ATR is zero or NaN.
    """
    ma = sma(trend, window)
    deviation = trend - ma
    normalised = deviation / atr_series
    normalised = normalised.replace([np.inf, -np.inf], np.nan)
    return normalised.rename(f"trend_dev_from_ma_{window}")


# ---------------------------------------------------------------------------
# Diagnostic / meta-features
# ---------------------------------------------------------------------------

def lambda_sensitivity_score(
    prices: pd.Series,
    lambdas: list[float],
    hp_window: int = 500,
    atr_series: pd.Series | None = None,
) -> pd.Series:
    """Model disagreement across lambda values, measured on ΔS*(t).

    Computes the causal HP trend for each lambda in ``lambdas``, takes
    the first difference (slope) of each, then returns the cross-lambda
    standard deviation of those slopes at each bar.

    High score → lambda values disagree on trend direction → low confidence.
    Low score → all lambda values agree → high confidence in trend estimate.

    WHY ΔS* and not S*?
        Two lambda values can produce very different *level* estimates while
        agreeing on *direction*.  Using level std would flag false ambiguity.
        Slope std directly measures directional disagreement.

    Args:
        prices: Close price series.
        lambdas: List of HP lambda values to compare.
        hp_window: Rolling window for causal HP (default 500).
        atr_series: Optional ATR series for normalisation.

    Returns:
        Cross-lambda slope std series, optionally ATR-normalised.
        NaN for first (hp_window − 1) bars.
    """
    if len(lambdas) < 2:
        raise ValueError("Need at least 2 lambda values to compute disagreement.")

    slopes: list[pd.Series] = []
    for lam in lambdas:
        trend = causal_hp_trend(prices, lamb=lam, window=hp_window)
        slope = hp_trend_slope(trend)
        slopes.append(slope)

    slopes_df = pd.concat(slopes, axis=1)
    score = slopes_df.std(axis=1)

    if atr_series is not None:
        score = score / atr_series
        score = score.replace([np.inf, -np.inf], np.nan)

    return score.rename("lambda_sensitivity_score")


def vol_regime(
    returns: pd.Series,
    window: int = 20,
    classification_window: int = 252,
) -> pd.Series:
    """Rolling volatility regime classifier: 0 = low, 1 = medium, 2 = high.

    Computes rolling realized volatility, then classifies each bar into
    one of three regimes based on rolling quantile thresholds computed
    over the past ``classification_window`` bars.

    Used as a regime filter — run IC analysis and backtests separately
    per regime to detect regime-conditional predictive power.  Averaging
    over regimes risks finding features that work in only one state.

    Args:
        returns: Log return series.
        window: Lookback for realized volatility (default 20).
        classification_window: Lookback for quantile thresholds (default 252).

    Returns:
        Integer series: 0 (low), 1 (medium), 2 (high).
        NaN for the warm-up period.
    """
    realized_vol = returns.rolling(window=window, min_periods=window).std()

    low_thresh = realized_vol.rolling(
        window=classification_window, min_periods=classification_window
    ).quantile(0.33)
    high_thresh = realized_vol.rolling(
        window=classification_window, min_periods=classification_window
    ).quantile(0.67)

    conditions = [
        realized_vol < low_thresh,
        realized_vol >= high_thresh,
    ]
    choices = [0.0, 2.0]
    regime = pd.Series(
        np.select(conditions, choices, default=1.0),
        index=returns.index,
        name="vol_regime",
    ).astype(float)

    # Mask warm-up period
    warm_up_mask = realized_vol.isna() | low_thresh.isna()
    regime[warm_up_mask] = np.nan

    return regime


# ---------------------------------------------------------------------------
# Threshold / time-stop signal generators
# ---------------------------------------------------------------------------

def threshold_time_stop_signal(
    feature: pd.Series,
    threshold_sigma: float,
    hold_bars: int,
    confirmation_bars: int = 0,
    lookback: int = 252 * 24,
) -> pd.Series:
    """Entry on feature threshold crossing; exit on time-stop or opposite threshold.

    Long entry:  feature < −threshold_sigma × rolling_std (confirmed for
                 confirmation_bars+1 consecutive bars).
    Short entry: feature > +threshold_sigma × rolling_std (confirmed for
                 confirmation_bars+1 consecutive bars).
    Exit:        hold_bars bars after entry, OR immediately when feature
                 crosses the opposite threshold (early reversal exit).

    Args:
        feature: Signal series (e.g. hp_trend_curvature or ATR-norm MA spread).
        threshold_sigma: Entry/exit threshold in rolling-std units.
        hold_bars: Maximum bars to hold the position (time-stop).
        confirmation_bars: Extra consecutive bars the feature must stay beyond
            the threshold before entry triggers.  0 = immediate entry.
        lookback: Rolling window for std normalisation (default 1 yr of H1 bars).

    Returns:
        Signal series of {−1.0, 0.0, +1.0}.  NaN-free; warm-up bars are 0.0.
    """
    n = len(feature)
    values = feature.values.astype(float)
    rolling_std = feature.rolling(window=lookback, min_periods=lookback).std().values

    signal_out = np.zeros(n)
    position = 0        # current position: -1, 0, +1
    hold_count = 0      # bars held in current position
    confirm_count = 0   # consecutive bars beyond threshold
    confirm_dir = 0     # direction being confirmed: +1 (long setup) or -1 (short setup)

    for i in range(n):
        if np.isnan(values[i]) or np.isnan(rolling_std[i]) or rolling_std[i] == 0.0:
            signal_out[i] = float(position)
            if position != 0:
                hold_count += 1
            else:
                confirm_count = 0
                confirm_dir = 0
            continue

        thresh = threshold_sigma * rolling_std[i]
        exited_this_bar = False

        # --- Manage open position ---
        if position != 0:
            hold_count += 1
            if position == 1 and values[i] > thresh:
                # Opposite threshold crossed → early exit long
                position = 0
                hold_count = 0
                confirm_count = 0
                confirm_dir = 0
                exited_this_bar = True
            elif position == -1 and values[i] < -thresh:
                # Opposite threshold crossed → early exit short
                position = 0
                hold_count = 0
                confirm_count = 0
                confirm_dir = 0
                exited_this_bar = True
            elif hold_count >= hold_bars:
                # Time-stop
                position = 0
                hold_count = 0
                confirm_count = 0
                confirm_dir = 0
                exited_this_bar = True

        # --- Look for new entry when flat ---
        if position == 0 and not exited_this_bar:
            if values[i] < -thresh:
                if confirm_dir == 1:
                    confirm_count += 1
                else:
                    confirm_dir = 1
                    confirm_count = 1
                if confirm_count > confirmation_bars:
                    position = 1
                    hold_count = 0
                    confirm_count = 0
                    confirm_dir = 0
            elif values[i] > thresh:
                if confirm_dir == -1:
                    confirm_count += 1
                else:
                    confirm_dir = -1
                    confirm_count = 1
                if confirm_count > confirmation_bars:
                    position = -1
                    hold_count = 0
                    confirm_count = 0
                    confirm_dir = 0
            else:
                confirm_count = 0
                confirm_dir = 0

        signal_out[i] = float(position)

    return pd.Series(signal_out, index=feature.index,
                     name=f"tts_{threshold_sigma}s_{hold_bars}h")


def vol_rolling_percentile(
    returns: pd.Series,
    vol_window: int = 20,
    lookback: int = 252 * 24,
) -> pd.Series:
    """Rolling percentile rank of realized volatility, expressed as 0–100.

    Used as a continuous vol filter: only trade when vol_rolling_percentile
    exceeds a threshold (e.g. > 50 trades only in above-median-vol regimes).

    Args:
        returns: Log return series.
        vol_window: Window for realized volatility (default 20 bars).
        lookback: Lookback for percentile rank (default 1 yr of H1 bars).

    Returns:
        Series in [0, 100].  NaN during warm-up period.
    """
    realized_vol = returns.rolling(window=vol_window, min_periods=vol_window).std()
    pct_rank = (
        realized_vol
        .rolling(window=lookback, min_periods=lookback)
        .rank(pct=True)
        * 100.0
    )
    return pct_rank.rename("vol_pct_rank")


def crossing_threshold_signal(
    feature: pd.Series,
    threshold_sigma: float,
    hold_bars: int,
    lookback: int = 252 * 24,
) -> pd.Series:
    """Mean-reversion signal using threshold CROSSING entry (not residency).

    Entry fires on the first bar where |feature| exceeds the threshold coming
    from neutral territory (previous bar was NOT extreme in the same direction).
    After a position exits via time-stop, the feature must first return below the
    threshold before a new entry is allowed.  This eliminates serial re-entries
    while the feature remains continuously extreme.

    Compared to threshold_time_stop_signal (confirmation_bars=0), the only
    behavioural difference is the post-exit lockout: this function requires the
    feature to cross BACK through the threshold level before re-entering.

    Long entry:  feature crosses below −threshold_sigma × rolling_std
                 (previous bar was NOT in the negative extreme zone)
    Short entry: feature crosses above +threshold_sigma × rolling_std
                 (previous bar was NOT in the positive extreme zone)
    Exit:        hold_bars time-stop OR feature crosses opposite threshold.

    Args:
        feature: Signal series (e.g. ATR-normalised MA spread).
        threshold_sigma: Entry/exit threshold in rolling-std units.
        hold_bars: Maximum bars to hold a position (time-stop).
        lookback: Rolling window for std normalisation (default 1 yr H1 bars).

    Returns:
        Signal series of {−1.0, 0.0, +1.0}.  NaN-free; warm-up bars are 0.0.
    """
    n = len(feature)
    values = feature.values.astype(float)
    rolling_std = feature.rolling(window=lookback, min_periods=lookback).std().values

    signal_out = np.zeros(n)
    position = 0        # current position: -1, 0, +1
    hold_count = 0      # bars held in current position
    prev_extreme = 0    # extreme state at end of previous bar: +1, -1, or 0

    for i in range(n):
        if np.isnan(values[i]) or np.isnan(rolling_std[i]) or rolling_std[i] == 0.0:
            signal_out[i] = float(position)
            if position != 0:
                hold_count += 1
            continue

        thresh = threshold_sigma * rolling_std[i]
        curr_extreme = (1 if values[i] > thresh else (-1 if values[i] < -thresh else 0))
        exited_this_bar = False

        # --- Manage open position ---
        if position != 0:
            hold_count += 1
            if position == 1 and curr_extreme == 1:
                # Feature crossed into positive extreme → early exit long
                position = 0
                hold_count = 0
                exited_this_bar = True
            elif position == -1 and curr_extreme == -1:
                # Feature crossed into negative extreme → early exit short
                position = 0
                hold_count = 0
                exited_this_bar = True
            elif hold_count >= hold_bars:
                # Time-stop
                position = 0
                hold_count = 0
                exited_this_bar = True

        # --- Look for crossing entry when flat ---
        # Entry fires only when feature JUST crossed into extreme territory
        # (prev_extreme was NOT the same direction → this is a genuine crossing)
        if position == 0 and not exited_this_bar:
            if curr_extreme == -1 and prev_extreme != -1:
                # LONG: fresh cross below −threshold
                position = 1
                hold_count = 0
            elif curr_extreme == 1 and prev_extreme != 1:
                # SHORT: fresh cross above +threshold
                position = -1
                hold_count = 0

        signal_out[i] = float(position)
        prev_extreme = curr_extreme

    return pd.Series(signal_out, index=feature.index,
                     name=f"cross_{threshold_sigma}s_{hold_bars}h")
