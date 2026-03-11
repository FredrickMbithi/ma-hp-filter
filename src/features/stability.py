"""Walk-forward IC stability analysis.

A feature might have strong IC in one market regime but weak or reversed
IC in another.  Walk-forward stability testing slices the full sample into
non-overlapping periods and computes the IC in each period independently.

Key outputs
-----------
sign_consistency
    Fraction of periods where the IC has the same sign as the overall mean
    IC.  A value of 1.0 means the feature always points in the same
    direction.  A value of 0.5 means it's no better than a coin flip.

is_stable
    True when ``sign_consistency ≥ 0.7`` — the feature points in the
    correct direction in at least 70% of out-of-sample periods.

Stability does NOT guarantee profitability, but instability (sign flipping)
is a hard disqualifier for inclusion in a production strategy.

Walk-forward mechanics
-----------------------
Period length is configurable (default 252 × 24 = 6 048 bars ≈ 1 year
of H1 FX data).  Each period is fully non-overlapping.  The final partial
period (if len(data) mod period_bars != 0) is included only if it contains
at least ``min_periods`` valid observations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .testing import compute_ic, compute_forward_returns

__all__ = [
    "PeriodICResult",
    "StabilityResult",
    "compute_walk_forward_stability",
    "FeatureStabilityTester",
]

_MIN_PERIOD_OBS: int = 30   # minimum IC observations per period


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PeriodICResult:
    """IC for a single walk-forward period."""

    period_start: pd.Timestamp
    period_end: pd.Timestamp
    ic: float
    p_value: float
    n_obs: int

    @property
    def is_significant(self) -> bool:
        return self.p_value < 0.05 if not np.isnan(self.p_value) else False


@dataclass
class StabilityResult:
    """Walk-forward IC stability for one feature."""

    feature_name: str
    horizon: int
    period_results: List[PeriodICResult]

    # Aggregate statistics (computed post-init)
    period_ics: pd.Series = field(default_factory=pd.Series)  # indexed by period_start
    mean_ic: float = np.nan
    std_ic: float = np.nan
    sign_consistency: float = np.nan   # fraction of periods matching overall sign
    pct_significant: float = np.nan    # fraction of periods with p < 0.05
    is_stable: bool = False
    stability_notes: str = ""

    def __post_init__(self) -> None:
        if not self.period_results:
            self.stability_notes = "No valid periods"
            return

        starts = [p.period_start for p in self.period_results]
        ics = [p.ic for p in self.period_results]

        self.period_ics = pd.Series(ics, index=starts, name="period_ic")
        valid_ics = [v for v in ics if not np.isnan(v)]

        if not valid_ics:
            self.stability_notes = "All period ICs are NaN"
            return

        self.mean_ic = float(np.mean(valid_ics))
        self.std_ic = float(np.std(valid_ics, ddof=1)) if len(valid_ics) > 1 else np.nan

        # sign_consistency: fraction matching the overall direction
        mean_sign = np.sign(self.mean_ic)
        if mean_sign == 0:
            self.sign_consistency = 0.5
        else:
            matching = sum(1 for v in valid_ics if np.sign(v) == mean_sign)
            self.sign_consistency = float(matching / len(valid_ics))

        # pct_significant
        sig_count = sum(1 for p in self.period_results if p.is_significant)
        self.pct_significant = float(sig_count / len(self.period_results))

        self.is_stable = self.sign_consistency >= 0.7

        if self.is_stable:
            self.stability_notes = (
                f"STABLE: sign_consistency={self.sign_consistency:.2f}, "
                f"mean_ic={self.mean_ic:.4f}"
            )
        else:
            self.stability_notes = (
                f"UNSTABLE: sign flips in {1 - self.sign_consistency:.0%} of periods, "
                f"mean_ic={self.mean_ic:.4f}"
            )


# ---------------------------------------------------------------------------
# Core function
# ---------------------------------------------------------------------------

def compute_walk_forward_stability(
    feature: pd.Series,
    prices: pd.Series,
    horizon: int = 24,
    period_bars: int = 252 * 24,
    feature_name: str = "",
) -> StabilityResult:
    """Compute walk-forward IC stability for a single feature.

    Slices the data into non-overlapping ``period_bars``-length windows and
    computes Spearman IC(feature, fwd_return_H) in each window.

    Args:
        feature: Feature series (must be aligned with ``prices``).
        prices: Close prices used to compute forward returns.
        horizon: Forward return horizon in bars (default 24).
        period_bars: Bars per period (default 252 × 24 ≈ 1 year H1).
        feature_name: Display name for the feature.

    Returns:
        StabilityResult with per-period IC values and stability verdict.
    """
    display_name = feature_name or (feature.name or "unnamed")

    # Align on common index
    aligned = pd.DataFrame({"feature": feature, "price": prices}).dropna(
        subset=["price"]
    )
    feature_aligned = aligned["feature"]
    price_aligned = aligned["price"]

    fwd_returns = compute_forward_returns(price_aligned, horizon)

    n_bars = len(aligned)
    period_results: List[PeriodICResult] = []

    for start_idx in range(0, n_bars, period_bars):
        end_idx = min(start_idx + period_bars, n_bars)
        feat_slice = feature_aligned.iloc[start_idx:end_idx]
        fwd_slice = fwd_returns.iloc[start_idx:end_idx]

        valid = pd.concat([feat_slice, fwd_slice], axis=1).dropna()
        if len(valid) < _MIN_PERIOD_OBS:
            continue

        ic_result = compute_ic(valid.iloc[:, 0], valid.iloc[:, 1])

        period_results.append(
            PeriodICResult(
                period_start=feat_slice.index[0],
                period_end=feat_slice.index[-1],
                ic=ic_result.ic,
                p_value=ic_result.p_value,
                n_obs=ic_result.n_obs,
            )
        )

    return StabilityResult(
        feature_name=display_name,
        horizon=horizon,
        period_results=period_results,
    )


# ---------------------------------------------------------------------------
# Tester class
# ---------------------------------------------------------------------------

class FeatureStabilityTester:
    """Walk-forward IC stability tester following the UnivariateFeatureTester API.

    Usage::

        tester = FeatureStabilityTester(prices, horizon=24, period_bars=252*24)
        result = tester.test(feature_series, name="ma_spread")
        summary_df = tester.test_all({"ma_spread": ma_s, "trend_dev": td_s})

    Args:
        prices: Close price series.
        horizon: Forward return horizon for IC computation (default 24 bars).
        period_bars: Bars per walk-forward period (default 6 048 ≈ 1 year H1).
    """

    def __init__(
        self,
        prices: pd.Series,
        horizon: int = 24,
        period_bars: int = 252 * 24,
    ) -> None:
        self.prices = prices
        self.horizon = horizon
        self.period_bars = period_bars

    def test(self, feature: pd.Series, name: str = "") -> StabilityResult:
        """Run walk-forward stability analysis for a single feature.

        Args:
            feature: Feature series aligned with self.prices.
            name: Display name (falls back to series.name or "unnamed").

        Returns:
            StabilityResult with period IC breakdown and stability verdict.
        """
        return compute_walk_forward_stability(
            feature=feature,
            prices=self.prices,
            horizon=self.horizon,
            period_bars=self.period_bars,
            feature_name=name or (feature.name or ""),
        )

    def test_all(self, features: Dict[str, pd.Series]) -> pd.DataFrame:
        """Run stability tests for all features and return a summary DataFrame.

        Columns: feature_name, n_periods, mean_ic, std_ic, sign_consistency,
        pct_significant, is_stable, stability_notes.

        Args:
            features: Mapping of feature name → pd.Series.

        Returns:
            pd.DataFrame with one row per feature, sorted by sign_consistency
            descending.
        """
        rows = []
        for name, series in features.items():
            result = self.test(series, name)
            rows.append(
                {
                    "feature_name": result.feature_name,
                    "horizon": result.horizon,
                    "n_periods": len(result.period_results),
                    "mean_ic": result.mean_ic,
                    "std_ic": result.std_ic,
                    "sign_consistency": result.sign_consistency,
                    "pct_significant": result.pct_significant,
                    "is_stable": result.is_stable,
                    "stability_notes": result.stability_notes,
                }
            )

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        df = df.sort_values("sign_consistency", ascending=False).reset_index(drop=True)
        return df
