"""Feature engineering modules."""

# Return transformations
from .returns import (
    compute_log_returns,
    compute_arithmetic_returns,
    compute_rolling_vol,
    compute_zscore,
)

# Feature library
from .library import FeatureLibrary

# Technical indicators
from .hp_filter import apply_hp_filter, cycle_zscore
from .moving_average import sma, ema, ma_envelope

# Strategy 8.1 generators (causal, no lookahead)
from .generators import (
    atr,
    causal_hp_trend,
    hp_trend_slope,
    hp_trend_curvature,
    ma_spread_on_trend,
    ma_crossover_signal,
    ma_crossover_age,
    trend_deviation_from_ma,
    lambda_sensitivity_score,
    vol_regime,
)

# Feature taxonomy and catalog
from .taxonomy import FeatureSpec, FeatureCatalog, FeatureCategory, FeatureOutcome, CATALOG

__all__ = [
    # Returns
    "compute_log_returns",
    "compute_arithmetic_returns",
    "compute_rolling_vol",
    "compute_zscore",
    # Library
    "FeatureLibrary",
    # HP filter (two-sided, visualization only)
    "apply_hp_filter",
    "cycle_zscore",
    # Moving averages
    "sma",
    "ema",
    "ma_envelope",
    # Strategy 8.1 generators (causal)
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
    # Taxonomy
    "FeatureSpec",
    "FeatureCatalog",
    "FeatureCategory",
    "FeatureOutcome",
    "CATALOG",
]
