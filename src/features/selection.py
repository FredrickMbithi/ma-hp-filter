"""Feature selection with formal pass/fail criteria.

A feature is accepted into the multi-factor model only when it passes all
of the following criteria:

1. IC > 0.03        — minimum predictive signal (Spearman rank correlation)
2. t-stat > 2.0     — IC is statistically distinguishable from noise
3. ADF p < 0.05     — feature is stationary (no unit root)
4. sign_consistency ≥ 0.70 — IC doesn't flip sign across time periods
5. max_pair_corr < 0.70  — not redundant with an already-selected feature

Note on cross-pair generalization (criterion not yet implemented)
-----------------------------------------------------------------
The research plan also requires that a feature "generalizes across 3+ FX
pairs".  This depends on ``src/features/cross_validation.py`` which is not
yet built.  ``FeatureSelector`` will emit a warning when this criterion
cannot be evaluated and will skip it, not penalizing the feature for the
missing check.

Usage::

    selector = FeatureSelector()
    results = selector.select(
        features={"ma_spread": ..., "trend_dev": ...},
        prices=close,
        univariate_summaries=tester.test_all(features),   # optional
        stability_summary=stability_tester.test_all(features),  # optional
    )
    df = FeatureSelector.summary_df(results)
    passed = [r.feature_name for r in results if r.passed]
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from .testing import (
    UnivariateFeatureTester,
    compute_ic,
    compute_forward_returns,
    check_stationarity,
)
from .stability import FeatureStabilityTester
from .correlation_analysis import select_non_redundant_features

__all__ = [
    "FeatureSelectionResult",
    "FeatureSelector",
]

_CROSS_PAIR_WARNING = (
    "Cross-pair generalization criterion requires cross_validation.py "
    "(not yet implemented).  Skipping for all features."
)


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class FeatureSelectionResult:
    """Pass/fail verdict and underlying metrics for a single feature."""

    feature_name: str

    # Raw metrics
    ic: float           # |IC| at peak horizon
    t_stat: float       # IC t-statistic: ic * sqrt(n-2) / sqrt(1-ic²)
    adf_p: float        # ADF p-value (lower = more stationary)
    sign_consistency: float   # from walk-forward stability
    max_pair_correlation: float  # highest |corr| with any other feature in set

    # Per-criterion flags
    passes_ic: bool
    passes_t_stat: bool
    passes_stationarity: bool
    passes_stability: bool
    passes_redundancy: bool

    # Summary
    passed: bool
    reject_reasons: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Selector class
# ---------------------------------------------------------------------------

class FeatureSelector:
    """Formal feature selection with configurable pass/fail thresholds.

    Args:
        ic_threshold: Minimum |IC| for acceptance (default 0.03).
        t_stat_threshold: Minimum IC t-statistic (default 2.0).
        adf_threshold: Maximum ADF p-value for stationarity (default 0.05).
        sign_consistency_threshold: Minimum sign_consistency from walk-
            forward stability (default 0.70).
        correlation_threshold: Maximum pairwise |correlation| before a
            feature is considered redundant (default 0.70).
        horizon: Forward return horizon for IC computation (default 24 bars).
        stability_period_bars: Walk-forward window size (default 1 year H1).
    """

    def __init__(
        self,
        ic_threshold: float = 0.03,
        t_stat_threshold: float = 2.0,
        adf_threshold: float = 0.05,
        sign_consistency_threshold: float = 0.70,
        correlation_threshold: float = 0.70,
        horizon: int = 24,
        stability_period_bars: int = 252 * 24,
    ) -> None:
        self.ic_threshold = ic_threshold
        self.t_stat_threshold = t_stat_threshold
        self.adf_threshold = adf_threshold
        self.sign_consistency_threshold = sign_consistency_threshold
        self.correlation_threshold = correlation_threshold
        self.horizon = horizon
        self.stability_period_bars = stability_period_bars

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select(
        self,
        features: Dict[str, pd.Series],
        prices: pd.Series,
        univariate_summaries: Optional[pd.DataFrame] = None,
        stability_summary: Optional[pd.DataFrame] = None,
        ic_scores: Optional[Dict[str, float]] = None,
    ) -> List[FeatureSelectionResult]:
        """Run the full selection pipeline.

        Accepts pre-computed summaries to avoid redundant computation, but
        will compute them internally if not provided.

        Args:
            features: Feature dictionary (name → pd.Series).
            prices: Close prices for computing forward returns.
            univariate_summaries: Output of UnivariateFeatureTester.test_all()
                as a DataFrame with columns matching FeatureTestSummary fields.
                If None, computed internally.
            stability_summary: Output of FeatureStabilityTester.test_all()
                as a DataFrame with columns including 'feature_name' and
                'sign_consistency'.  If None, computed internally.
            ic_scores: Optional {feature_name: abs_IC} for ranking in the
                redundancy step.  If None, derived from univariate_summaries.

        Returns:
            List of FeatureSelectionResult, one per input feature, ordered
            by |IC| descending.
        """
        if not features:
            return []

        warnings.warn(_CROSS_PAIR_WARNING, stacklevel=2)

        # --- Step 1: Gather univariate metrics ---
        ic_map, t_stat_map, adf_p_map = self._gather_univariate_metrics(
            features, prices, univariate_summaries
        )

        # --- Step 2: Gather stability metrics ---
        sign_consistency_map = self._gather_stability_metrics(
            features, prices, stability_summary
        )

        # --- Step 3: Redundancy check ---
        _ic_for_redundancy = ic_scores or {
            n: abs(ic_map.get(n, 0.0)) for n in features
        }
        corr_result = select_non_redundant_features(
            features,
            ic_scores=_ic_for_redundancy,
            threshold=self.correlation_threshold,
        )
        max_corr_map: Dict[str, float] = {}
        for name in features:
            if len(corr_result.spearman_matrix) == 0:
                max_corr_map[name] = 0.0
                continue
            row = corr_result.spearman_matrix.loc[name].drop(name)
            max_corr_map[name] = float(row.abs().max()) if len(row) > 0 else 0.0

        # dropped features failed redundancy WITH a better feature
        dropped_set = set(corr_result.features_to_drop)

        # --- Step 4: Build per-feature results ---
        results: List[FeatureSelectionResult] = []
        for name in features:
            ic_val = abs(ic_map.get(name, np.nan))
            t_stat = t_stat_map.get(name, np.nan)
            adf_p = adf_p_map.get(name, np.nan)
            sc = sign_consistency_map.get(name, np.nan)
            max_corr = max_corr_map.get(name, np.nan)

            passes_ic = not np.isnan(ic_val) and ic_val > self.ic_threshold
            passes_t = not np.isnan(t_stat) and abs(t_stat) > self.t_stat_threshold
            passes_adf = not np.isnan(adf_p) and adf_p < self.adf_threshold
            passes_stab = not np.isnan(sc) and sc >= self.sign_consistency_threshold
            passes_redund = name not in dropped_set

            reject_reasons: List[str] = []
            if not passes_ic:
                reject_reasons.append(
                    f"|IC|={ic_val:.4f} < {self.ic_threshold} (threshold)"
                )
            if not passes_t:
                reject_reasons.append(
                    f"t-stat={t_stat:.2f} < {self.t_stat_threshold} (threshold)"
                )
            if not passes_adf:
                reject_reasons.append(
                    f"ADF p={adf_p:.4f} ≥ {self.adf_threshold} (non-stationary)"
                )
            if not passes_stab:
                reject_reasons.append(
                    f"sign_consistency={sc:.2f} < {self.sign_consistency_threshold}"
                )
            if not passes_redund:
                # Find which feature it's redundant with
                partner = next(
                    (
                        p.keep
                        for p in corr_result.redundant_pairs
                        if p.feature_a == name
                    ),
                    "another feature",
                )
                reject_reasons.append(
                    f"Redundant with '{partner}' (corr > {self.correlation_threshold})"
                )

            passed = len(reject_reasons) == 0
            results.append(
                FeatureSelectionResult(
                    feature_name=name,
                    ic=ic_val,
                    t_stat=t_stat,
                    adf_p=adf_p,
                    sign_consistency=sc,
                    max_pair_correlation=max_corr,
                    passes_ic=passes_ic,
                    passes_t_stat=passes_t,
                    passes_stationarity=passes_adf,
                    passes_stability=passes_stab,
                    passes_redundancy=passes_redund,
                    passed=passed,
                    reject_reasons=reject_reasons,
                )
            )

        results.sort(key=lambda r: -r.ic if not np.isnan(r.ic) else 0.0)
        return results

    @staticmethod
    def summary_df(results: Sequence[FeatureSelectionResult]) -> pd.DataFrame:
        """Convert a list of FeatureSelectionResult to a summary DataFrame.

        Args:
            results: Output of FeatureSelector.select().

        Returns:
            DataFrame with one row per feature; columns: feature_name, passed,
            ic, t_stat, adf_p, sign_consistency, max_pair_correlation,
            passes_ic, passes_t_stat, passes_stationarity, passes_stability,
            passes_redundancy, reject_reasons.
        """
        rows = []
        for r in results:
            rows.append(
                {
                    "feature_name": r.feature_name,
                    "passed": r.passed,
                    "ic": r.ic,
                    "t_stat": r.t_stat,
                    "adf_p": r.adf_p,
                    "sign_consistency": r.sign_consistency,
                    "max_pair_correlation": r.max_pair_correlation,
                    "passes_ic": r.passes_ic,
                    "passes_t_stat": r.passes_t_stat,
                    "passes_stationarity": r.passes_stationarity,
                    "passes_stability": r.passes_stability,
                    "passes_redundancy": r.passes_redundancy,
                    "reject_reasons": "; ".join(r.reject_reasons),
                }
            )
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _gather_univariate_metrics(
        self,
        features: Dict[str, pd.Series],
        prices: pd.Series,
        pre_computed: Optional[pd.DataFrame],
    ) -> tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
        """Return ic_map, t_stat_map, adf_p_map from pre-computed or fresh."""
        ic_map: Dict[str, float] = {}
        t_stat_map: Dict[str, float] = {}
        adf_p_map: Dict[str, float] = {}

        if pre_computed is not None and len(pre_computed) > 0:
            for _, row in pre_computed.iterrows():
                name = row.get("name", row.get("feature_name", ""))
                if not name:
                    continue
                ic_map[name] = float(row.get("peak_abs_ic", row.get("ic", np.nan)))
                # t-stat: ic * sqrt(n-2) / sqrt(1-ic²)
                ic_raw = float(row.get("peak_ic", row.get("ic", np.nan)))
                n = float(row.get("n_obs", row.get("n", 1000)))
                t_stat_map[name] = _compute_t_stat(ic_raw, n)
                adf_p_map[name] = float(row.get("adf_p", np.nan))
            return ic_map, t_stat_map, adf_p_map

        # Compute fresh
        fwd = compute_forward_returns(prices, self.horizon)
        for name, series in features.items():
            ic_result = compute_ic(series, fwd)
            ic_map[name] = ic_result.abs_ic
            t_stat_map[name] = _compute_t_stat(ic_result.ic, ic_result.n_obs)

            adf_result = check_stationarity(series, name=name)
            adf_p_map[name] = adf_result.p_value

        return ic_map, t_stat_map, adf_p_map

    def _gather_stability_metrics(
        self,
        features: Dict[str, pd.Series],
        prices: pd.Series,
        pre_computed: Optional[pd.DataFrame],
    ) -> Dict[str, float]:
        """Return sign_consistency_map from pre-computed or fresh."""
        sc_map: Dict[str, float] = {}

        if pre_computed is not None and len(pre_computed) > 0:
            for _, row in pre_computed.iterrows():
                name = row.get("feature_name", "")
                if not name:
                    continue
                sc_map[name] = float(row.get("sign_consistency", np.nan))
            return sc_map

        tester = FeatureStabilityTester(
            prices=prices,
            horizon=self.horizon,
            period_bars=self.stability_period_bars,
        )
        for name, series in features.items():
            result = tester.test(series, name)
            sc_map[name] = result.sign_consistency

        return sc_map


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _compute_t_stat(ic: float, n_obs: int) -> float:
    """IC t-statistic: t = ic * sqrt(n-2) / sqrt(1 - ic²).

    Args:
        ic: Signed Spearman IC value.
        n_obs: Number of valid observations.

    Returns:
        t-statistic.  Returns NaN when ic is NaN or n_obs < 3.
    """
    if np.isnan(ic) or n_obs < 3:
        return np.nan
    ic2 = ic * ic
    if ic2 >= 1.0:
        return np.nan
    return float(ic * np.sqrt(n_obs - 2) / np.sqrt(1.0 - ic2))
