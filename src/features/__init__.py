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

# Univariate testing framework
from .testing import (
    StationarityResult,
    ICResult,
    ICDecayResult,
    RollingICResult,
    HitRateResult,
    MonotonicityResult,
    FeatureTestSummary,
    compute_forward_returns,
    check_stationarity,
    compute_ic,
    compute_ic_decay,
    compute_rolling_ic,
    compute_hit_rate,
    compute_quantile_ic,
    UnivariateFeatureTester,
)

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
    # Univariate testing
    "StationarityResult",
    "ICResult",
    "ICDecayResult",
    "RollingICResult",
    "HitRateResult",
    "MonotonicityResult",
    "FeatureTestSummary",
    "compute_forward_returns",
    "check_stationarity",
    "compute_ic",
    "compute_ic_decay",
    "compute_rolling_ic",
    "compute_hit_rate",
    "compute_quantile_ic",
    "UnivariateFeatureTester",
    # Correlation analysis
    "RedundantPair",
    "CorrelationResult",
    "compute_correlation_matrix",
    "find_redundant_pairs",
    "select_non_redundant_features",
    # Normalization
    "zscore_normalize",
    "rank_normalize",
    "normalize_features",
    # Regime features
    "trend_strength",
    "trend_consistency",
    "hl_spread_ratio",
    "vol_zscore",
    "vol_rolling_percentile",
    # Stability
    "PeriodICResult",
    "StabilityResult",
    "compute_walk_forward_stability",
    "FeatureStabilityTester",
    # Selection
    "FeatureSelectionResult",
    "FeatureSelector",
    # Signals
    "SignalConfig",
    "combine_features",
    "signal_to_position",
]


# Correlation analysis
from .correlation_analysis import (
    RedundantPair,
    CorrelationResult,
    compute_correlation_matrix,
    find_redundant_pairs,
    select_non_redundant_features,
)

# Normalization
from .normalization import (
    zscore_normalize,
    rank_normalize,
    normalize_features,
)

# Regime features (new + re-exports from generators)
from .regime_features import (
    trend_strength,
    trend_consistency,
    hl_spread_ratio,
    vol_zscore,
    vol_rolling_percentile,
)

# Walk-forward stability
from .stability import (
    PeriodICResult,
    StabilityResult,
    compute_walk_forward_stability,
    FeatureStabilityTester,
)

# Feature selection
from .selection import (
    FeatureSelectionResult,
    FeatureSelector,
)

# Signal construction
from .signals import (
    SignalConfig,
    combine_features,
    signal_to_position,
)
