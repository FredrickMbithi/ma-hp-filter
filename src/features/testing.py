"""Univariate feature testing framework for FX strategy research.

Provides statistical tools to evaluate the predictive power of individual
features against forward returns in isolation, before combining features
into multi-factor models.

Key metrics
-----------
Information Coefficient (IC)
    Spearman rank correlation between a feature observed at time t and the
    log return from t to t+H.  IC > 0.05 is considered meaningful in FX.

IC Decay
    How quickly predictive power decays as the prediction horizon H grows.
    Features with rapid IC decay are suited to higher-frequency strategies.

Rolling IC (stability)
    IC computed over a rolling window to detect non-stationarity in the
    feature–return relationship.  A stable signal has consistent IC sign
    and magnitude across the full sample.

IC Information Ratio (IC-IR)
    mean(rolling_IC) / std(rolling_IC).  Values > 0.5 indicate a reliable
    signal; values < 0.3 suggest the relationship is regime-dependent.

Hit Rate
    Fraction of bars where sign(feature) == sign(fwd_return).  Expected
    value is 0.5 under no predictability.

Monotonicity
    Whether the relationship between feature quantile and mean forward
    return is monotonic.  Spearman(bin_rank, mean_fwd_return) of ±1
    confirms a stable linear ordering.

Reject criteria
---------------
1. Non-stationary (ADF p ≥ 0.05) — spurious relationship risk.
2. Peak |IC| < ic_threshold across all tested horizons.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller

__all__ = [
    # Dataclasses
    "StationarityResult",
    "ICResult",
    "ICDecayResult",
    "RollingICResult",
    "HitRateResult",
    "MonotonicityResult",
    "FeatureTestSummary",
    # Functions
    "compute_forward_returns",
    "check_stationarity",
    "compute_ic",
    "compute_ic_decay",
    "compute_rolling_ic",
    "compute_hit_rate",
    "compute_quantile_ic",
    # Class
    "UnivariateFeatureTester",
]

IC_THRESHOLD = 0.05  # Minimum meaningful IC for FX (Spearman rho)
MIN_OBS = 30         # Minimum valid observations for any correlation estimate


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class StationarityResult:
    """Augmented Dickey-Fuller unit root test outcome."""

    feature_name: str
    adf_stat: float
    p_value: float
    n_lags: int
    n_obs: int
    is_stationary: bool  # True when p_value < 0.05

    def __str__(self) -> str:
        status = "STATIONARY" if self.is_stationary else "NON-STATIONARY"
        return (
            f"{self.feature_name}: ADF={self.adf_stat:.4f}, "
            f"p={self.p_value:.6f}, n={self.n_obs}  [{status}]"
        )


@dataclass
class ICResult:
    """Spearman rank IC at a single horizon."""

    ic: float
    p_value: float
    n_obs: int
    significant: bool  # True when p_value < 0.05

    @property
    def abs_ic(self) -> float:
        return abs(self.ic)

    @property
    def stars(self) -> str:
        if np.isnan(self.p_value):
            return "   "
        if self.p_value < 0.001:
            return "***"
        if self.p_value < 0.01:
            return "** "
        if self.p_value < 0.05:
            return "*  "
        return "   "


@dataclass
class ICDecayResult:
    """IC values across multiple prediction horizons."""

    feature_name: str
    horizons: List[int]
    ic_values: List[float]
    p_values: List[float]
    n_obs: List[int]
    peak_horizon: int
    peak_ic: float      # signed — negative for mean-reversion features
    peak_abs_ic: float

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "horizon": self.horizons,
                "ic": self.ic_values,
                "p_value": self.p_values,
                "n_obs": self.n_obs,
            }
        )


@dataclass
class RollingICResult:
    """Rolling IC time-series for temporal stability analysis."""

    feature_name: str
    window: int
    series: pd.Series       # IC value at each bar (NaN until window fills)
    mean_ic: float
    std_ic: float
    ic_ir: float            # IC Information Ratio = mean / std
    pct_positive: float     # fraction of windows where IC > 0
    pct_above_threshold: float  # fraction of windows where |IC| > IC_THRESHOLD


@dataclass
class HitRateResult:
    """Directional accuracy of a feature at a specific horizon."""

    feature_name: str
    horizon: int
    hit_rate: float
    n_calls: int
    p_value: float  # two-tailed binomial test vs p=0.5

    @property
    def is_significant(self) -> bool:
        return self.p_value < 0.05


@dataclass
class MonotonicityResult:
    """Feature quantile analysis — relationship to mean forward return."""

    feature_name: str
    n_quantiles: int
    quantile_labels: List[str]
    mean_fwd_returns_bps: List[float]  # mean forward return per bin, in bps
    spearman_rho: float                # ±1 = perfectly monotonic
    p_value: float
    is_monotonic: bool                 # |rho| > 0.8 and p < 0.05


@dataclass
class FeatureTestSummary:
    """Consolidated test results and pass/fail verdict for one feature."""

    name: str
    is_stationary: bool
    adf_p: float
    peak_ic: float        # signed
    peak_abs_ic: float
    peak_horizon: int
    rolling_ic_mean: float
    rolling_ic_std: float
    ic_ir: float
    hit_rate: float       # at peak-IC horizon
    monotonicity_rho: float
    passed: bool
    reject_reason: str    # empty string when passed


# ---------------------------------------------------------------------------
# Forward returns (IC target — not a feature)
# ---------------------------------------------------------------------------

def compute_forward_returns(prices: pd.Series, horizon: int) -> pd.Series:
    """Log return from t to t+horizon.

    At bar t the value is log(price[t+horizon] / price[t]).
    This is the IC *target*, computed for rank-correlation analysis only.
    The last ``horizon`` bars will be NaN (no future data available).

    No lookahead bias is introduced because feature values at time t are
    correlated with future returns at t+H — this is the correct procedure
    for measuring prospective predictive power.

    Args:
        prices: Close price series.
        horizon: Number of bars forward.

    Returns:
        Series of forward log returns; final ``horizon`` bars are NaN.
    """
    fwd = np.log(prices.shift(-horizon) / prices)
    fwd.name = f"fwd_return_H{horizon}"
    return fwd


# ---------------------------------------------------------------------------
# Stationarity test
# ---------------------------------------------------------------------------

def check_stationarity(
    series: pd.Series,
    name: str = "",
    max_lags: int = 24,
) -> StationarityResult:
    """Augmented Dickey-Fuller test for unit roots.

    H0: series has a unit root (non-stationary).
    Reject H0 (p < 0.05) → series is stationary.

    Uses AIC automatic lag selection bounded by ``max_lags``.
    Missing values are dropped before the test.

    Args:
        series: Feature series to test.
        name: Display name.
        max_lags: Maximum lags for AIC selection (default 24 for H1 data).

    Returns:
        StationarityResult with ADF statistic, p-value, and verdict.
    """
    display_name = name or series.name or "unnamed"
    clean = series.dropna()
    adf_stat, p_value, n_lags, n_obs, *_ = adfuller(
        clean, maxlag=max_lags, autolag="AIC"
    )
    return StationarityResult(
        feature_name=display_name,
        adf_stat=float(adf_stat),
        p_value=float(p_value),
        n_lags=int(n_lags),
        n_obs=int(n_obs),
        is_stationary=float(p_value) < 0.05,
    )


# ---------------------------------------------------------------------------
# IC computation
# ---------------------------------------------------------------------------

def compute_ic(
    feature: pd.Series,
    fwd_returns: pd.Series,
) -> ICResult:
    """Spearman rank IC between a feature and forward returns.

    Drops bars where either series is NaN before computing correlation.
    Returns NaN IC when fewer than MIN_OBS valid pairs are available.

    Args:
        feature: Feature values at time t.
        fwd_returns: Forward return at time t, horizon H.

    Returns:
        ICResult with Spearman rho, p-value, n_obs, and significance flag.
    """
    aligned = pd.concat([feature, fwd_returns], axis=1).dropna()
    n = len(aligned)
    if n < MIN_OBS:
        return ICResult(ic=np.nan, p_value=np.nan, n_obs=n, significant=False)
    rho, p_value = stats.spearmanr(aligned.iloc[:, 0], aligned.iloc[:, 1])
    return ICResult(
        ic=float(rho),
        p_value=float(p_value),
        n_obs=n,
        significant=float(p_value) < 0.05,
    )


def compute_ic_decay(
    feature: pd.Series,
    prices: pd.Series,
    horizons: Sequence[int],
    feature_name: str = "",
) -> ICDecayResult:
    """IC at each horizon in ``horizons``, revealing the predictive lifespan.

    Args:
        feature: Feature values at each bar.
        prices: Close price series (used to derive forward returns).
        horizons: Forward horizons to test, in bars.
        feature_name: Display name.

    Returns:
        ICDecayResult with per-horizon IC values and peak-IC information.
    """
    ic_values: List[float] = []
    p_values: List[float] = []
    n_obs_list: List[int] = []

    for h in horizons:
        fwd = compute_forward_returns(prices, h)
        result = compute_ic(feature, fwd)
        ic_values.append(result.ic)
        p_values.append(result.p_value)
        n_obs_list.append(result.n_obs)

    abs_ics = [abs(v) if not np.isnan(v) else 0.0 for v in ic_values]
    peak_idx = int(np.argmax(abs_ics))

    return ICDecayResult(
        feature_name=feature_name or (feature.name or "unnamed"),
        horizons=list(horizons),
        ic_values=ic_values,
        p_values=p_values,
        n_obs=n_obs_list,
        peak_horizon=horizons[peak_idx],
        peak_ic=ic_values[peak_idx],
        peak_abs_ic=abs_ics[peak_idx],
    )


# ---------------------------------------------------------------------------
# Rolling IC (temporal stability)
# ---------------------------------------------------------------------------

def compute_rolling_ic(
    feature: pd.Series,
    fwd_returns: pd.Series,
    window: int = 1000,
    feature_name: str = "",
) -> RollingICResult:
    """Rolling Spearman IC over a sliding window.

    Computes exact Spearman rank correlation within each window by
    converting each per-window slice to ranks and computing Pearson of
    those ranks (equivalent to Spearman).  Uses numpy for speed:
    ~1 second per feature on 62,000-bar series with window=1000.

    IC-IR = mean_IC / std_IC:
    * IC-IR > 0.5  → reliable signal
    * IC-IR < 0.3  → regime-dependent or noise

    Args:
        feature: Feature series.
        fwd_returns: Forward return series at a fixed horizon.
        window: Rolling window in bars (default 1000 ≈ 6 months H1).
        feature_name: Display name.

    Returns:
        RollingICResult with time-series IC and aggregate statistics.
    """
    display_name = feature_name or (feature.name or "unnamed")
    aligned = pd.concat([feature, fwd_returns], axis=1).dropna()

    if len(aligned) < window:
        empty = pd.Series(np.nan, index=feature.index, name="rolling_ic")
        return RollingICResult(
            feature_name=display_name,
            window=window,
            series=empty,
            mean_ic=np.nan,
            std_ic=np.nan,
            ic_ir=np.nan,
            pct_positive=np.nan,
            pct_above_threshold=np.nan,
        )

    f_arr = aligned.iloc[:, 0].to_numpy(dtype=np.float64)
    r_arr = aligned.iloc[:, 1].to_numpy(dtype=np.float64)
    n = len(f_arr)
    ic_vals = np.full(n, np.nan)

    for i in range(window - 1, n):
        fw = f_arr[i - window + 1 : i + 1]
        rw = r_arr[i - window + 1 : i + 1]
        # Convert to ranks (argsort of argsort gives 0-indexed rank)
        f_rank = np.argsort(np.argsort(fw)).astype(np.float64)
        r_rank = np.argsort(np.argsort(rw)).astype(np.float64)
        # Pearson of ranks == Spearman
        f_dev = f_rank - f_rank.mean()
        r_dev = r_rank - r_rank.mean()
        std_f = np.sqrt(np.dot(f_dev, f_dev))
        std_r = np.sqrt(np.dot(r_dev, r_dev))
        if std_f > 0 and std_r > 0:
            ic_vals[i] = np.dot(f_dev, r_dev) / (std_f * std_r)

    ic_series = pd.Series(ic_vals, index=aligned.index, name="rolling_ic")
    valid = ic_series.dropna()

    mean_ic = float(valid.mean()) if len(valid) > 0 else np.nan
    std_ic = float(valid.std()) if len(valid) > 1 else np.nan
    ic_ir = (mean_ic / std_ic) if (std_ic and std_ic > 0) else np.nan
    pct_pos = float((valid > 0).mean()) if len(valid) > 0 else np.nan
    pct_above = float((valid.abs() > IC_THRESHOLD).mean()) if len(valid) > 0 else np.nan

    # Re-index to the original feature's index for alignment in plots
    ic_series_reindexed = ic_series.reindex(feature.index)

    return RollingICResult(
        feature_name=display_name,
        window=window,
        series=ic_series_reindexed,
        mean_ic=mean_ic,
        std_ic=std_ic,
        ic_ir=ic_ir,
        pct_positive=pct_pos,
        pct_above_threshold=pct_above,
    )


# ---------------------------------------------------------------------------
# Hit rate
# ---------------------------------------------------------------------------

def compute_hit_rate(
    feature: pd.Series,
    fwd_returns: pd.Series,
    feature_name: str = "",
    horizon: int = 0,
) -> HitRateResult:
    """Fraction of bars where sign(feature) == sign(fwd_return).

    Bars where either series is zero or NaN are excluded.
    Uses a two-tailed binomial test against H0: hit_rate = 0.5.

    Args:
        feature: Feature values.
        fwd_returns: Forward returns at a specific horizon.
        feature_name: Display name.
        horizon: Horizon label attached to result (informational).

    Returns:
        HitRateResult with hit_rate, n_calls, and binomial p-value.
    """
    display_name = feature_name or (feature.name or "unnamed")
    aligned = pd.concat([feature, fwd_returns], axis=1).dropna()
    nonzero = aligned[(aligned.iloc[:, 0] != 0) & (aligned.iloc[:, 1] != 0)]
    n = len(nonzero)

    if n < MIN_OBS:
        return HitRateResult(
            feature_name=display_name,
            horizon=horizon,
            hit_rate=np.nan,
            n_calls=n,
            p_value=np.nan,
        )

    hits = int((np.sign(nonzero.iloc[:, 0]) == np.sign(nonzero.iloc[:, 1])).sum())
    hit_rate = hits / n
    binom_result = stats.binomtest(hits, n, p=0.5, alternative="two-sided")
    return HitRateResult(
        feature_name=display_name,
        horizon=horizon,
        hit_rate=hit_rate,
        n_calls=n,
        p_value=float(binom_result.pvalue),
    )


# ---------------------------------------------------------------------------
# Quantile / monotonicity analysis
# ---------------------------------------------------------------------------

def compute_quantile_ic(
    feature: pd.Series,
    fwd_returns: pd.Series,
    n_quantiles: int = 5,
    feature_name: str = "",
) -> MonotonicityResult:
    """Mean forward return per feature quantile, plus a monotonicity test.

    Bins the feature into ``n_quantiles`` equal-frequency bins, computes
    the mean forward return in each bin (in basis points), then tests
    whether the bin-rank → mean-return relationship is monotonic via
    Spearman correlation.

    A Spearman rho of +1 / −1 with p < 0.05 and |rho| > 0.8 confirms a
    stable, monotonic feature–return relationship.

    Args:
        feature: Feature series.
        fwd_returns: Forward returns at a specific horizon.
        n_quantiles: Equal-frequency bins (default 5 = quintiles).
        feature_name: Display name.

    Returns:
        MonotonicityResult with per-bin mean returns and monotonicity verdict.
    """
    display_name = feature_name or (feature.name or "unnamed")
    aligned = pd.concat([feature, fwd_returns], axis=1).dropna()
    aligned.columns = ["feature", "fwd_return"]  # type: ignore[assignment]

    if len(aligned) < n_quantiles * MIN_OBS:
        return MonotonicityResult(
            feature_name=display_name,
            n_quantiles=n_quantiles,
            quantile_labels=[],
            mean_fwd_returns_bps=[],
            spearman_rho=np.nan,
            p_value=np.nan,
            is_monotonic=False,
        )

    aligned["bin"] = pd.qcut(
        aligned["feature"], q=n_quantiles, labels=False, duplicates="drop"
    )
    quantile_means = (
        aligned.groupby("bin")["fwd_return"].mean() * 10_000  # convert to bps
    )
    labels = [f"Q{int(q) + 1}" for q in quantile_means.index]
    mean_fwd = list(quantile_means.values)
    bin_ranks = list(range(len(mean_fwd)))

    if len(mean_fwd) < 2:
        rho, p_val = np.nan, np.nan
    else:
        rho, p_val = stats.spearmanr(bin_ranks, mean_fwd)

    return MonotonicityResult(
        feature_name=display_name,
        n_quantiles=n_quantiles,
        quantile_labels=labels,
        mean_fwd_returns_bps=mean_fwd,
        spearman_rho=float(rho) if not np.isnan(rho) else np.nan,
        p_value=float(p_val) if not np.isnan(p_val) else np.nan,
        is_monotonic=(
            not np.isnan(rho)
            and not np.isnan(p_val)
            and abs(rho) > 0.8
            and p_val < 0.05
        ),
    )


# ---------------------------------------------------------------------------
# UnivariateFeatureTester class
# ---------------------------------------------------------------------------

class UnivariateFeatureTester:
    """Orchestrates the full univariate predictive-power test suite.

    Each feature is evaluated independently against forward log returns
    at multiple horizons.

    Reject criteria (applied in order):
    1. Non-stationary (ADF p ≥ 0.05) — spurious relationship risk.
    2. Peak |IC| < ``ic_threshold`` across all tested horizons.

    Parameters
    ----------
    prices : pd.Series
        Close price series aligned with all feature series.
    ic_threshold : float
        Minimum |IC| for a feature to be considered meaningful (default 0.05).
    horizons : sequence of int
        Forward horizons (bars) to test IC decay over.
    rolling_window : int
        Rolling window for IC stability analysis in bars.
    """

    def __init__(
        self,
        prices: pd.Series,
        ic_threshold: float = IC_THRESHOLD,
        horizons: Sequence[int] = (1, 4, 12, 24, 48),
        rolling_window: int = 1000,
    ) -> None:
        self.prices = prices
        self.ic_threshold = ic_threshold
        self.horizons = list(horizons)
        self.rolling_window = rolling_window

        # Pre-compute forward returns at each horizon once
        self._fwd_cache: Dict[int, pd.Series] = {
            h: compute_forward_returns(prices, h) for h in self.horizons
        }

    # ------------------------------------------------------------------
    # Single-feature test
    # ------------------------------------------------------------------

    def test(
        self,
        feature: pd.Series,
        name: str = "",
    ) -> FeatureTestSummary:
        """Run the full univariate test suite on one feature.

        Steps:
          1. ADF stationarity test.
          2. IC decay across all horizons.
          3. Rolling IC at the peak-IC horizon.
          4. Hit rate at the peak-IC horizon.
          5. Quantile monotonicity at the peak-IC horizon.
          6. Apply reject criteria → pass/fail verdict.

        Args:
            feature: Feature series aligned to ``self.prices``.
            name: Display name (falls back to series.name or 'unnamed').

        Returns:
            FeatureTestSummary with all results and pass/fail verdict.
        """
        display_name = name or (feature.name or "unnamed")

        # 1. Stationarity
        stat_result = check_stationarity(feature, display_name)

        # 2. IC decay
        ic_decay = compute_ic_decay(
            feature, self.prices, self.horizons, display_name
        )

        # 3. Rolling IC at peak horizon
        peak_fwd = self._fwd_cache[ic_decay.peak_horizon]
        rolling = compute_rolling_ic(
            feature, peak_fwd, self.rolling_window, display_name
        )

        # 4. Hit rate at peak horizon
        hit = compute_hit_rate(
            feature, peak_fwd, display_name, ic_decay.peak_horizon
        )

        # 5. Quantile monotonicity at peak horizon
        mono = compute_quantile_ic(feature, peak_fwd, 5, display_name)

        # 6. Reject criteria (checked in priority order)
        reject_reason = ""
        if not stat_result.is_stationary:
            reject_reason = (
                f"Non-stationary (ADF p={stat_result.p_value:.4f})"
            )
        elif ic_decay.peak_abs_ic < self.ic_threshold:
            reject_reason = (
                f"Peak |IC|={ic_decay.peak_abs_ic:.4f} "
                f"< threshold {self.ic_threshold}"
            )

        return FeatureTestSummary(
            name=display_name,
            is_stationary=stat_result.is_stationary,
            adf_p=stat_result.p_value,
            peak_ic=ic_decay.peak_ic,
            peak_abs_ic=ic_decay.peak_abs_ic,
            peak_horizon=ic_decay.peak_horizon,
            rolling_ic_mean=rolling.mean_ic,
            rolling_ic_std=rolling.std_ic,
            ic_ir=rolling.ic_ir,
            hit_rate=hit.hit_rate,
            monotonicity_rho=mono.spearman_rho,
            passed=reject_reason == "",
            reject_reason=reject_reason,
        )

    # ------------------------------------------------------------------
    # Batch test
    # ------------------------------------------------------------------

    def test_all(
        self,
        features: Dict[str, pd.Series],
    ) -> pd.DataFrame:
        """Run the full test suite on all features, return a summary DataFrame.

        Args:
            features: Mapping of feature name → Series.

        Returns:
            DataFrame with one row per feature and columns:
            name, is_stationary, adf_p, peak_ic, peak_abs_ic, peak_horizon,
            rolling_ic_mean, rolling_ic_std, ic_ir, hit_rate,
            monotonicity_rho, passed, reject_reason.
        """
        rows = []
        for feat_name, feat_series in features.items():
            summary = self.test(feat_series, feat_name)
            rows.append(vars(summary))
        return pd.DataFrame(rows)
