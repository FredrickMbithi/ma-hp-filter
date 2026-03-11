"""Multi-feature signal construction.

Combines multiple normalized features into a single tradeable signal.
Features must be pre-normalized (e.g., via normalization.py) so they are
on a comparable scale before combination.

Three signal combination modes
-------------------------------
binary_threshold
    ``combined_signal > +threshold → +1`` (long)
    ``combined_signal < −threshold → −1`` (short)
    ``|combined_signal| ≤ threshold → 0`` (flat)

    This is the simplest and most transparent approach.  The threshold
    controls trade frequency: higher threshold → fewer but higher-
    conviction trades.

proportional
    ``position = clip(combined_signal, −1, 1)``

    Position size scales continuously with signal strength.  A combined
    signal of 0.6 → 60% of maximum position size.  Suitable for
    execution systems that support fractional sizing.

ensemble_vote
    Each feature casts a directional vote (+1 if positive, −1 if negative,
    0 if zero or NaN).  The position is the sign of the majority vote.
    Tie → 0 (flat).

    Robust to outliers because extreme feature values have no more
    influence than mild ones — only direction counts.

Workflow::

    features_norm = normalize_features(features, method="zscore")
    config = SignalConfig(mode="binary_threshold", threshold=0.3)
    signal = combine_features(features_norm, config)
    positions = signal_to_position(signal, mode="binary_threshold", threshold=0.3)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Literal, Optional

import numpy as np
import pandas as pd

__all__ = [
    "SignalConfig",
    "combine_features",
    "signal_to_position",
]

SignalMode = Literal["binary_threshold", "proportional", "ensemble_vote"]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SignalConfig:
    """Configuration for multi-feature signal construction.

    Attributes:
        mode: Combination and position-translation method.
            "binary_threshold" — trade when |combined| exceeds threshold.
            "proportional" — fractional sizing proportional to signal.
            "ensemble_vote" — directional vote from each feature.
        threshold: Used by "binary_threshold" mode.  Determines the minimum
            combined signal strength required to open a position (default 0.3,
            meaning 30% of one standard deviation when features are z-scored).
        weights: Optional per-feature weights for linear combination.  When
            None, equal weights are applied.  The weights are normalized to
            sum to 1.0 internally so the combined signal stays in a
            comparable range to the individual feature scales.
    """

    mode: SignalMode = "binary_threshold"
    threshold: float = 0.3
    weights: Optional[Dict[str, float]] = None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def combine_features(
    features: Dict[str, pd.Series],
    config: SignalConfig = None,
) -> pd.Series:
    """Combine a dictionary of pre-normalized features into a single signal.

    All features should share the same DatetimeIndex and be on a comparable
    scale (use normalization.py first).  Bars where a feature is NaN
    contribute 0.0 to the linear combination or are treated as abstentions
    in ensemble_vote mode.

    Args:
        features: Mapping of feature name → normalized pd.Series.
        config: SignalConfig specifying mode, threshold, and weights.
            If None, defaults are used (binary_threshold, threshold=0.3,
            equal weights).

    Returns:
        Combined signal pd.Series on the union of all feature indices.
        - "binary_threshold" or "proportional" mode: continuous float.
        - "ensemble_vote" mode: integer {−1, 0, +1}.
    """
    if config is None:
        config = SignalConfig()

    if not features:
        return pd.Series(dtype=float)

    if config.mode == "ensemble_vote":
        return _ensemble_vote(features)

    return _linear_combine(features, config.weights)


def signal_to_position(
    signal: pd.Series,
    mode: SignalMode = "binary_threshold",
    threshold: float = 0.3,
) -> pd.Series:
    """Translate a continuous combined signal to discrete positions.

    Args:
        signal: Output of combine_features().
        mode: Position translation mode.
        threshold: Threshold for "binary_threshold" mode.

    Returns:
        Position series:
        - "binary_threshold": {−1, 0, +1}
        - "proportional": float in [−1, 1] (clipped)
        - "ensemble_vote": {−1, 0, +1} (signal already discrete)
    """
    if mode == "binary_threshold":
        positions = pd.Series(0.0, index=signal.index)
        positions[signal > threshold] = 1.0
        positions[signal < -threshold] = -1.0
        positions.name = "position"
        return positions

    if mode == "proportional":
        positions = signal.clip(-1.0, 1.0)
        positions = positions.fillna(0.0)
        positions.name = "position"
        return positions

    if mode == "ensemble_vote":
        positions = np.sign(signal).fillna(0.0)
        positions.name = "position"
        return positions

    raise ValueError(
        f"mode must be 'binary_threshold', 'proportional', or 'ensemble_vote'; "
        f"got {mode!r}"
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _linear_combine(
    features: Dict[str, pd.Series],
    weights: Optional[Dict[str, float]],
) -> pd.Series:
    """Equal or custom weighted linear combination of features.

    NaN values in any feature at a given bar contribute 0.0 to the
    weighted sum for that bar (treated as absent, not as missing data).
    The denominator is adjusted to only count non-NaN features, preserving
    the intended scale.

    Args:
        features: Feature dictionary (pre-normalized).
        weights: Per-feature weights.  If None, equal weights are used.
            Weights are normalized to sum to 1.

    Returns:
        Combined signal pd.Series.
    """
    names = list(features.keys())
    n = len(names)

    if weights is not None:
        w = {k: float(weights.get(k, 0.0)) for k in names}
        total = sum(abs(v) for v in w.values())
        if total == 0:
            raise ValueError("Sum of absolute weights is zero — cannot normalize.")
        w_norm = {k: v / total for k, v in w.items()}
    else:
        w_norm = {k: 1.0 / n for k in names}

    # Align all features to a common index
    df = pd.DataFrame(features)

    # Weighted sum; NaN → 0; denominator = sum of weights for non-NaN features
    weighted_sum = pd.Series(0.0, index=df.index)
    weight_sum = pd.Series(0.0, index=df.index)
    for name in names:
        col = df[name].fillna(0.0)
        not_nan_mask = df[name].notna().astype(float)
        weighted_sum += col * w_norm[name]
        weight_sum += not_nan_mask * w_norm[name]

    # Re-scale so that bars with all features present have expected scale
    combined = weighted_sum / weight_sum.replace(0, np.nan)
    combined = combined.replace([np.inf, -np.inf], np.nan)
    combined.name = "combined_signal"
    return combined


def _ensemble_vote(features: Dict[str, pd.Series]) -> pd.Series:
    """Directional vote from each feature; majority wins.

    Each feature contributes a vote of sign(feature[t]):
    - +1 (long),  −1 (short), or 0 (abstain if NaN or exactly 0).

    The combined signal is the sign of the vote total.
    If votes are exactly tied across +1 and −1, the result is 0.

    Args:
        features: Feature dictionary.

    Returns:
        Integer signal {−1, 0, +1}.
    """
    df = pd.DataFrame(features)
    votes = df.apply(np.sign).fillna(0.0)
    vote_total = votes.sum(axis=1)
    result = np.sign(vote_total)
    result.name = "combined_signal"
    return result
