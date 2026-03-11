"""Multi-factor strategy — signal construction and backtest integration.

Combines the four strongest USDJPY H1 features identified in research
into a single trading signal:

1. ``ma_spread_on_trend``  — ATR-normalised MA spread on HP trend (IC −0.318)
2. ``trend_deviation_from_ma`` — HP trend deviation from its own MA (IC −0.344)
3. ``ma_crossover_signal``     — Binary HP-trend MA crossover (inverted, IC −0.397)
4. ``trend_strength``          — Regime: how strong the current trend is

Both (1) and (2) are mean-reversion features: a high positive value means
price is stretched and due to snap back (hence negative IC — the feature
is a leading indicator of a *negative* return).  The signal is reversed
internally so that positive combined signal → long position.

Signal pipeline
---------------
1. Compute causal HP trend from raw prices.
2. Derive ``ma_spread_on_trend``, ``trend_deviation_from_ma``,
   ``ma_crossover_signal``, ``trend_strength``. 
3. Negate mean-reversion features so the sign convention is consistent:
   "positive feature value → expect positive return".
4. Normalize all features (z-score or rank).
5. Combine via ``signals.combine_features()`` with the chosen mode.
6. Translate to discrete positions via ``signals.signal_to_position()``.

Usage::

    from src.strategies.multi_factor import run_multi_factor_backtest, MultiFactorConfig

    df = loader.load("USDJPY_10yr_1h_dukascopy")
    results = run_multi_factor_backtest(df)

    # Or with custom parameters:
    config = MultiFactorConfig(signal_mode="proportional", threshold=0.2)
    results = run_multi_factor_backtest(df, config)
    print(results["sharpe"])
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from math import sqrt
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.features.generators import (
    atr,
    causal_hp_trend,
    hp_trend_slope,
    ma_spread_on_trend,
    ma_crossover_signal,
    trend_deviation_from_ma,
)
from src.features.regime_features import trend_strength, vol_zscore
from src.features.returns import compute_log_returns
from src.features.normalization import normalize_features
from src.features.signals import SignalConfig, combine_features, signal_to_position
from src.backtest.engine import VectorizedBacktest
from src.backtest.cost_model import FXCostModel

__all__ = [
    "MultiFactorConfig",
    "build_multi_factor_signal",
    "run_multi_factor_backtest",
]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MultiFactorConfig:
    """Parameters for the multi-factor signal and backtest.

    Feature construction
    --------------------
    hp_lambda : HP smoothing parameter (default 3.9e9 for H1 FX).
    hp_window : Causal HP rolling window in bars (default 500).
    ma_short  : Short MA window applied to HP trend (default 72 bars = 3 days H1).
    ma_long   : Long MA window applied to HP trend (default 240 bars = 10 days H1).
    atr_window : ATR averaging window (default 20 bars).

    Normalization
    -------------
    norm_method : "zscore" or "rank" (default "zscore").
    norm_window : Rolling lookback for normalization (default 1 year H1).

    Feature weights
    ---------------
    feature_weights : Dict mapping feature name to weight.  None = equal weights.

    Signal construction
    -------------------
    signal_mode : "binary_threshold", "proportional", or "ensemble_vote".
    threshold   : Used by "binary_threshold" mode (default 0.3).

    Cost model
    ----------
    spread_bps    : Bid-ask spread in basis points (default 0.7).
    slippage_bps  : Slippage per trade in basis points (default 0.2).
    """

    # HP filter
    hp_lambda: float = 3.9e9
    hp_window: int = 500

    # MA windows (applied to HP trend)
    ma_short: int = 72
    ma_long: int = 240

    # ATR window
    atr_window: int = 20

    # Normalization
    norm_method: str = "zscore"
    norm_window: int = 252 * 24

    # Feature weights (None = equal)
    feature_weights: Optional[Dict[str, float]] = None

    # Signal
    signal_mode: str = "binary_threshold"
    threshold: float = 0.3

    # Cost
    spread_bps: float = 0.7
    slippage_bps: float = 0.2


# ---------------------------------------------------------------------------
# Signal builder
# ---------------------------------------------------------------------------

def build_multi_factor_signal(
    df: pd.DataFrame,
    config: MultiFactorConfig = None,
) -> pd.Series:
    """Compute the combined multi-factor position signal.

    All computations are causal (no lookahead).

    Args:
        df: OHLC DataFrame with DatetimeIndex and columns
            ['open', 'high', 'low', 'close'].
        config: MultiFactorConfig.  Uses defaults when None.

    Returns:
        Position signal pd.Series with the same index as ``df``.
        Values are {−1.0, 0.0, +1.0} for binary_threshold and
        ensemble_vote modes, or float in [−1, 1] for proportional.
    """
    if config is None:
        config = MultiFactorConfig()

    close = df["close"]
    high = df["high"]
    low = df["low"]
    log_ret = compute_log_returns(close)
    atr_series = atr(high, low, close, window=config.atr_window)

    # --- Step 1: Compute causal HP trend ---
    trend = causal_hp_trend(close, lamb=config.hp_lambda, window=config.hp_window)

    # --- Step 2: Derive features ---
    # Mean-reversion features: positive value → price stretched → expect reversal
    # Sign is INVERTED so positive → expect positive return (long bias)
    ma_spread = ma_spread_on_trend(
        trend=trend,
        t1=config.ma_short,
        t2=config.ma_long,
        atr_series=atr_series,
    )
    trend_dev = trend_deviation_from_ma(
        trend=trend,
        window=config.ma_long,
        atr_series=atr_series,
    )

    # Crossover signal: already {+1, −1}; research shows it must be inverted
    # (reversed strategy: cross UP → go SHORT, cross DOWN → go LONG)
    crossover = ma_crossover_signal(trend=trend, t1=config.ma_short, t2=config.ma_long)

    # Trend strength (regime feature): high → strong trend; keep positive = positive
    t_strength = trend_strength(trend=trend, atr_series=atr_series, lookback=config.norm_window)

    # Vol z-score (filter only, not combined into signal directly)
    # — used later for position masking in advanced configurations

    # --- Step 3: Negate mean-reversion features so sign convention is unified ---
    # (positive raw value = stretched = expect reversal = we want to go SHORT)
    # After inversion: positive = expect positive return = go LONG
    features_raw: Dict[str, pd.Series] = {
        "ma_spread_inv": -ma_spread,
        "trend_dev_inv": -trend_dev,
        "crossover": -crossover,      # research: inverted crossover is predictive
        "trend_strength": t_strength,
    }

    # Override with any user-specified weights
    if config.feature_weights is not None:
        features_raw = {
            k: v for k, v in features_raw.items() if k in config.feature_weights
        }

    # --- Step 4: Normalize ---
    features_norm = normalize_features(
        features_raw,
        method=config.norm_method,
        window=config.norm_window,
    )

    # --- Step 5: Combine ---
    signal_cfg = SignalConfig(
        mode=config.signal_mode,
        threshold=config.threshold,
        weights=config.feature_weights,
    )
    combined = combine_features(features_norm, signal_cfg)

    # --- Step 6: Translate to position ---
    positions = signal_to_position(
        combined,
        mode=config.signal_mode,
        threshold=config.threshold,
    )
    positions = positions.reindex(df.index).fillna(0.0)
    return positions


# ---------------------------------------------------------------------------
# Backtest runner
# ---------------------------------------------------------------------------

def run_multi_factor_backtest(
    df: pd.DataFrame,
    config: MultiFactorConfig = None,
) -> Dict[str, Any]:
    """Build and backtest the multi-factor signal on the provided OHLC data.

    Returns standard backtest metrics alongside the VectorizedBacktest output.

    Args:
        df: OHLC DataFrame (DatetimeIndex, 'open'/'high'/'low'/'close').
        config: MultiFactorConfig.  Uses defaults when None.

    Returns:
        Dictionary with keys:
        - All keys from VectorizedBacktest.run():
          "equity", "strategy_returns", "net_returns", "costs",
          "positions", "trades"
        - "signal": the raw position signal before backtest
        - "sharpe": annualized Sharpe ratio (252 × 24 bars)
        - "annual_return": annualized net return
        - "max_drawdown": maximum drawdown fraction
        - "n_trades": total number of position changes
        - "config": the MultiFactorConfig used
    """
    if config is None:
        config = MultiFactorConfig()

    signal = build_multi_factor_signal(df, config)
    cost_model = FXCostModel(
        spread_bps=config.spread_bps,
        slippage_bps=config.slippage_bps,
    )
    bt = VectorizedBacktest(df, signal, cost_model)
    raw = bt.run()

    net_ret = raw["net_returns"]
    bars_per_year = 252 * 24

    mean_r = net_ret.mean()
    std_r = net_ret.std()
    sharpe = float((mean_r / std_r) * sqrt(bars_per_year)) if std_r > 0 else np.nan

    annual_return = float((1 + mean_r) ** bars_per_year - 1)

    equity = raw["equity"]
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    max_dd = float(drawdown.min())

    n_trades = int(raw["trades"].abs().gt(0).sum())

    return {
        **raw,
        "signal": signal,
        "sharpe": sharpe,
        "annual_return": annual_return,
        "max_drawdown": max_dd,
        "n_trades": n_trades,
        "config": config,
    }
