"""Feature correlation and redundancy analysis.

Identifies pairs of features that are highly correlated and therefore
redundant when combined in a multi-factor model.  Provides a greedy
algorithm to select a maximally non-redundant subset, keeping the
higher-IC feature from each correlated pair.

Correlation methods
-------------------
Spearman (default)
    Rank-based; robust to non-Gaussian features and outliers.  Consistent
    with IC testing (which also uses Spearman rank correlation).

Pearson
    Linear; appropriate when features are approximately normally
    distributed.  More sensitive to extreme values.

Redundancy threshold
    Two features are considered redundant when |corr| > threshold (default
    0.7).  The threshold can be adjusted to be more or less aggressive.

Greedy deduplication
    1. Sort features by |IC| descending so higher-value features are
       retained first.
    2. Iterate: add a feature to the kept set only if its maximum
       correlation with any already-kept feature is below the threshold.
    3. Features that fail the check are dropped and the highest-correlated
       kept feature is recorded in ``RedundantPair.keep``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "RedundantPair",
    "CorrelationResult",
    "compute_correlation_matrix",
    "find_redundant_pairs",
    "select_non_redundant_features",
]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RedundantPair:
    """A pair of features whose correlation exceeds the redundancy threshold."""

    feature_a: str       # The feature that is dropped
    feature_b: str       # The feature that is *kept* (higher |IC|)
    correlation: float   # signed correlation value
    method: str          # "spearman" or "pearson"
    keep: str            # name of the feature to retain; equals feature_b


@dataclass
class CorrelationResult:
    """Full output of the correlation/redundancy analysis."""

    spearman_matrix: pd.DataFrame        # n×n pairwise Spearman correlations
    pearson_matrix: pd.DataFrame         # n×n pairwise Pearson correlations
    redundant_pairs: List[RedundantPair] # pairs with |corr| > threshold
    features_to_keep: List[str]          # after greedy deduplication
    features_to_drop: List[str]          # redundant with a kept feature
    threshold: float                     # the |corr| threshold used


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def compute_correlation_matrix(
    features: Dict[str, pd.Series],
    method: str = "spearman",
    min_periods: int = 30,
) -> pd.DataFrame | Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute pairwise feature correlation matrix.

    Args:
        features: Mapping of feature name → aligned pd.Series.
        method: "spearman", "pearson", or "both".  When "both", returns a
            tuple (spearman_matrix, pearson_matrix).
        min_periods: Minimum overlapping observations required to compute a
            correlation; pairs with fewer are set to NaN.

    Returns:
        A single DataFrame (method="spearman" or "pearson") or a tuple of
        two DataFrames (method="both"), each with features as index and
        columns.
    """
    if method not in ("spearman", "pearson", "both"):
        raise ValueError(f"method must be 'spearman', 'pearson', or 'both'; got {method!r}")

    # Align all series into a single DataFrame, dropping NaN rows pairwise
    # is handled by pd.DataFrame.corr with min_periods.
    df = pd.DataFrame(features)

    if method == "both":
        spearman = _spearman_matrix(df, min_periods)
        pearson = df.corr(method="pearson", min_periods=min_periods)
        return spearman, pearson

    if method == "spearman":
        return _spearman_matrix(df, min_periods)

    # method == "pearson"
    return df.corr(method="pearson", min_periods=min_periods)


def _spearman_matrix(df: pd.DataFrame, min_periods: int) -> pd.DataFrame:
    """Compute pairwise Spearman correlation matrix respecting min_periods."""
    n = len(df.columns)
    names = list(df.columns)
    mat = np.full((n, n), np.nan)

    for i in range(n):
        mat[i, i] = 1.0
        for j in range(i + 1, n):
            pair = df[[names[i], names[j]]].dropna()
            if len(pair) < min_periods:
                continue
            rho, _ = stats.spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])
            mat[i, j] = mat[j, i] = float(rho)

    return pd.DataFrame(mat, index=names, columns=names)


def find_redundant_pairs(
    corr_matrix: pd.DataFrame,
    threshold: float = 0.7,
) -> List[RedundantPair]:
    """Find all feature pairs with |correlation| above the threshold.

    Only the upper triangle of the matrix is scanned to avoid duplicate
    pairs (feature_a, feature_b) and (feature_b, feature_a).

    Args:
        corr_matrix: Square DataFrame of pairwise correlations.
        threshold: Absolute correlation threshold (default 0.7).

    Returns:
        List of RedundantPair, one per correlated pair.  The ``keep``
        field is empty string — use ``select_non_redundant_features`` if
        you want IC-guided selection.
    """
    names = list(corr_matrix.columns)
    pairs: List[RedundantPair] = []

    for i, a in enumerate(names):
        for j in range(i + 1, len(names)):
            b = names[j]
            val = corr_matrix.iloc[i, j]
            if np.isnan(val):
                continue
            if abs(val) > threshold:
                pairs.append(
                    RedundantPair(
                        feature_a=a,
                        feature_b=b,
                        correlation=float(val),
                        method="",   # caller knows which matrix they passed
                        keep="",
                    )
                )

    return pairs


def select_non_redundant_features(
    features: Dict[str, pd.Series],
    ic_scores: Optional[Dict[str, float]] = None,
    threshold: float = 0.7,
    method: str = "spearman",
    min_periods: int = 30,
) -> CorrelationResult:
    """Select a maximally non-redundant subset of features.

    Algorithm
    ---------
    1. Compute both Spearman and Pearson correlation matrices.
    2. Sort features by |IC| descending so more predictive features are
       retained first.  Features without IC scores are appended last in
       original order.
    3. Greedy inclusion: add a feature to the ``kept`` set if its maximum
       |correlation| (using the chosen ``method`` matrix) with any already-
       kept feature is ≤ ``threshold``.
    4. Features that fail step 3 are added to ``dropped``.  The most-
       correlated already-kept feature is recorded as the winner.

    Args:
        features: Mapping of feature name → pd.Series (pre-computed,
            aligned to the same DatetimeIndex).
        ic_scores: Optional mapping of feature name → |IC| value.  Used to
            break ties: the feature with higher |IC| is kept.  When None,
            features are processed in insertion order.
        threshold: Absolute correlation threshold for redundancy (0.7).
        method: Which correlation matrix to use for the greedy step
            ("spearman" or "pearson").  Both matrices are always computed
            and returned regardless of this setting.
        min_periods: Minimum overlapping observations for pairwise corr.

    Returns:
        CorrelationResult with both matrices, the redundant pairs
        (annotated with the ``keep`` field), and the keep/drop lists.
    """
    if len(features) == 0:
        empty = pd.DataFrame()
        return CorrelationResult(
            spearman_matrix=empty,
            pearson_matrix=empty,
            redundant_pairs=[],
            features_to_keep=[],
            features_to_drop=[],
            threshold=threshold,
        )

    spearman_mat, pearson_mat = compute_correlation_matrix(
        features, method="both", min_periods=min_periods
    )

    active_matrix = spearman_mat if method == "spearman" else pearson_mat

    # Sort by |IC| descending; unknown IC → 0 so they appear last
    _ic = ic_scores or {}
    sorted_names = sorted(features.keys(), key=lambda n: -abs(_ic.get(n, 0.0)))

    kept: List[str] = []
    dropped: List[str] = []
    redundant_pairs: List[RedundantPair] = []

    for name in sorted_names:
        if not kept:
            kept.append(name)
            continue

        # Check correlation against all already-kept features
        max_corr_val = 0.0
        max_corr_partner = kept[0]
        for k in kept:
            val = active_matrix.loc[name, k]
            if np.isnan(val):
                continue
            if abs(val) > abs(max_corr_val):
                max_corr_val = float(val)
                max_corr_partner = k

        if abs(max_corr_val) > threshold:
            dropped.append(name)
            redundant_pairs.append(
                RedundantPair(
                    feature_a=name,
                    feature_b=max_corr_partner,
                    correlation=max_corr_val,
                    method=method,
                    keep=max_corr_partner,
                )
            )
        else:
            kept.append(name)

    return CorrelationResult(
        spearman_matrix=spearman_mat,
        pearson_matrix=pearson_mat,
        redundant_pairs=redundant_pairs,
        features_to_keep=kept,
        features_to_drop=dropped,
        threshold=threshold,
    )
