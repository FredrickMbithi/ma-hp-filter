"""Feature taxonomy and hypothesis registry for Strategy 8.1 (MA + HP Filter).

Provides a structured catalog of every candidate feature with its research
hypothesis, stationarity assumptions, parameter grid, and outcome tracking.

Usage:
    from src.features.taxonomy import CATALOG, FeatureCategory, FeatureOutcome

    # List all trend features
    trend_features = CATALOG.by_category(FeatureCategory.TREND)

    # Print summary table
    print(CATALOG.summary_df().to_string())

    # Retrieve a specific spec
    spec = CATALOG.get("hp_trend_slope")
    print(spec.hypothesis)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class FeatureCategory(str, Enum):
    TREND = "Trend"
    MEAN_REVERSION = "Mean Reversion"
    MICROSTRUCTURE = "Microstructure"
    DIAGNOSTIC = "Diagnostic"


class FeatureOutcome(str, Enum):
    PENDING = "Pending"
    PREDICTIVE = "Predictive"
    NOT_PREDICTIVE = "Not Predictive"
    INCONCLUSIVE = "Inconclusive"


# ---------------------------------------------------------------------------
# FeatureSpec
# ---------------------------------------------------------------------------

@dataclass
class FeatureSpec:
    """Complete research specification for a single feature.

    Attributes:
        name: Unique identifier, snake_case. Used as registry key.
        category: Broad functional family.
        hypothesis: Why this feature might predict forward returns.
        lookback: Human-readable description of lookback window(s).
        parameters: Free parameters and their test values, e.g.
                    {"lambda": [1e6, 1e9, 3.9e9], "t1": [24, 48, 72]}.
        expected_stationary: True = stationary expected by theory.
                             False = non-stationary expected by theory.
                             None = ambiguous — test empirically with ADF.
        stationarity_notes: Explains the reasoning behind expected_stationary.
                            Required to be precise when expected_stationary is None.
        test_date: ISO 8601 date string (YYYY-MM-DD).
        outcome: Current research outcome.
        notes: Key findings, caveats, or implementation notes.
    """

    name: str
    category: FeatureCategory
    hypothesis: str
    lookback: str
    parameters: dict[str, Any] = field(default_factory=dict)
    expected_stationary: bool | None = None
    stationarity_notes: str = ""
    test_date: str = "2026-03-10"
    outcome: FeatureOutcome = FeatureOutcome.PENDING
    notes: str = ""

    def __repr__(self) -> str:
        stationary_str = (
            "Yes" if self.expected_stationary is True
            else "No" if self.expected_stationary is False
            else "Unknown"
        )
        return (
            f"FeatureSpec(name={self.name!r}, "
            f"category={self.category.value!r}, "
            f"stationary={stationary_str}, "
            f"outcome={self.outcome.value!r})"
        )


# ---------------------------------------------------------------------------
# FeatureCatalog
# ---------------------------------------------------------------------------

class FeatureCatalog:
    """Registry of FeatureSpec objects, keyed by feature name.

    Pre-populated at module import via _build_catalog().  Access the
    singleton via the module-level CATALOG constant.

    Thread safety: The catalog is intended to be read-only after module
    import.  Do not call register() after tests or production code has
    started iterating over entries.
    """

    def __init__(self) -> None:
        self._registry: dict[str, FeatureSpec] = {}

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def register(self, spec: FeatureSpec) -> None:
        """Add a FeatureSpec to the registry.

        Raises:
            ValueError: If a spec with the same name is already registered.
        """
        if spec.name in self._registry:
            raise ValueError(
                f"Feature '{spec.name}' is already registered. "
                "Use a unique name or update the existing spec directly."
            )
        self._registry[spec.name] = spec

    def update_outcome(
        self,
        name: str,
        outcome: FeatureOutcome,
        notes: str = "",
    ) -> None:
        """Update the outcome (and optionally notes) of a registered feature."""
        spec = self.get(name)
        spec.outcome = outcome
        if notes:
            spec.notes = notes

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def get(self, name: str) -> FeatureSpec:
        """Retrieve a spec by name.

        Raises:
            KeyError: If name is not in the registry.
        """
        if name not in self._registry:
            raise KeyError(
                f"Feature '{name}' not found in catalog. "
                f"Registered features: {sorted(self._registry.keys())}"
            )
        return self._registry[name]

    def all(self) -> list[FeatureSpec]:
        """Return all registered specs in insertion order."""
        return list(self._registry.values())

    def by_category(self, category: FeatureCategory) -> list[FeatureSpec]:
        """Return all specs belonging to a given category."""
        return [s for s in self._registry.values() if s.category == category]

    def pending(self) -> list[FeatureSpec]:
        """Return all specs not yet tested."""
        return [s for s in self._registry.values() if s.outcome == FeatureOutcome.PENDING]

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary_df(self) -> pd.DataFrame:
        """Return a DataFrame summarising the catalog.

        Columns: Name, Category, Stationary, Parameters, Outcome
        """
        rows = []
        for spec in self._registry.values():
            stationary_str = (
                "Yes" if spec.expected_stationary is True
                else "No" if spec.expected_stationary is False
                else "Test required"
            )
            rows.append({
                "Name": spec.name,
                "Category": spec.category.value,
                "Stationary": stationary_str,
                "Parameters": ", ".join(
                    f"{k}={v}" for k, v in spec.parameters.items()
                ),
                "Outcome": spec.outcome.value,
            })
        return pd.DataFrame(rows)

    def __len__(self) -> int:
        return len(self._registry)

    def __contains__(self, name: str) -> bool:
        return name in self._registry

    def __repr__(self) -> str:
        return f"FeatureCatalog({len(self._registry)} features registered)"


# ---------------------------------------------------------------------------
# Catalog population
# ---------------------------------------------------------------------------

def _build_catalog() -> FeatureCatalog:
    """Construct and return the pre-populated strategy catalog.

    Called once at module import to produce the CATALOG singleton.
    """
    cat = FeatureCatalog()

    # ------------------------------------------------------------------
    # HP-filter derived features
    # ------------------------------------------------------------------

    cat.register(FeatureSpec(
        name="hp_trend_level",
        category=FeatureCategory.TREND,
        hypothesis=(
            "The HP-filtered price S*(t), computed causally on a rolling window, "
            "represents the smoothed underlying directional bias in USDJPY. Trading "
            "in the direction of S*(t)'s slope should capture institutional carry and "
            "momentum flows rather than microstructure noise."
        ),
        lookback="Full causal window (min 50 bars; recommended ≥500 for stability)",
        parameters={
            "lambda": [1e6, 1e7, 1e8, 1e9, 3.9e9, 1e11],
            "window": [500],
        },
        expected_stationary=False,
        stationarity_notes=(
            "S*(t) is a smoothed version of price. Price is I(1) (random walk). "
            "The HP filter does not change the integration order of the level — S*(t) "
            "is also non-stationary."
        ),
        outcome=FeatureOutcome.PREDICTIVE,
        notes=(
            "EMPIRICAL RESULT (2026-03-10): ADF stationary (t=−5.78, p<0.000001) "
            "despite I(1) theoretical expectation — causal rolling window bounds "
            "variation within a moving reference frame at N=2,104 bars. "
            "IC = −0.397 at H=24, strongest in the feature set. "
            "DIRECTION INVERSION: negative IC means high trend level predicts lower "
            "future returns — mean reversion, not trend continuation. "
            "λ ≥ 1B functionally identical (slope ρ ≥ 0.9995); effective grid: {1M, 100M, 1B}."
        ),
    ))

    cat.register(FeatureSpec(
        name="hp_trend_slope",
        category=FeatureCategory.TREND,
        hypothesis=(
            "The first difference ΔS*(t) = S*(t) − S*(t−1) measures the current "
            "momentum of the clean trend. A positive slope means the filtered trend "
            "is rising, capturing genuine directional conviction. This should predict "
            "short-term continuation."
        ),
        lookback="1 bar (first difference of S*)",
        parameters={"lambda": [1e6, 1e7, 1e8, 1e9, 3.9e9, 1e11]},
        expected_stationary=True,
        stationarity_notes=(
            "First difference of an I(1) series is I(0) by definition. "
            "ΔS* should be stationary regardless of lambda."
        ),
        outcome=FeatureOutcome.INCONCLUSIVE,
        notes=(
            "EMPIRICAL RESULT (2026-03-10): ADF stationary across all λ (worst p=0.016). "
            "IC = −0.052 at H=24 (marginal significance). H=1 and H=4 not significant. "
            "Med Vol regime IC = −0.204 — signal concentrated entirely in medium volatility. "
            "Trend continuation hypothesis rejected. Too weak to trade standalone; "
            "regime-conditional use only as secondary confirmation."
        ),
    ))

    cat.register(FeatureSpec(
        name="hp_trend_curvature",
        category=FeatureCategory.TREND,
        hypothesis=(
            "The second difference Δ²S*(t) = ΔS*(t) − ΔS*(t−1) measures trend "
            "acceleration. A large negative curvature during an uptrend signals "
            "deceleration — the mathematical signature of exhaustion. This should "
            "predict mean-reversion entries when curvature turns sharply against "
            "the trend direction."
        ),
        lookback="3 bars (second difference window)",
        parameters={"lambda": [1e6, 1e7, 1e8, 1e9, 3.9e9, 1e11]},
        expected_stationary=True,
        stationarity_notes=(
            "Second difference of an I(1) series is I(-1) — strongly stationary. "
            "May exhibit over-differencing artifacts (negative autocorrelation at lag 1)."
        ),
        outcome=FeatureOutcome.INCONCLUSIVE,
        notes=(
            "EMPIRICAL RESULT (2026-03-10): ADF strongly stationary (t≈−42, p≈0). "
            "IC = +0.059 at H=24 (p<0.011). Med Vol IC = +0.137. "
            "Positive sign is opposite to other HP features — potential orthogonal "
            "exhaustion dimension. λ ≥ 100M produce near-zero curvature by construction. "
            "Too weak standalone; candidate as a compound confirmation signal only."
        ),
    ))

    # ------------------------------------------------------------------
    # MA-on-trend features
    # ------------------------------------------------------------------

    cat.register(FeatureSpec(
        name="ma_spread_on_trend",
        category=FeatureCategory.TREND,
        hypothesis=(
            "The spread sma(S*, T1) − sma(S*, T2) measures trend strength and direction. "
            "A large positive spread indicates a strong uptrend from persistent macro "
            "flows. A spread near zero indicates trend exhaustion or regime transition — "
            "higher probability zone for counter-trend entries."
        ),
        lookback="T1 ∈ {24, 48, 72}, T2 ∈ {120, 168, 240, 480} bars",
        parameters={
            "T1": [24, 48, 72],
            "T2": [120, 168, 240, 480],
            "normalize_by_atr": [True, False],
        },
        expected_stationary=None,
        stationarity_notes=(
            "Not guaranteed. Moving averages are linear filters of a non-stationary "
            "series, so the spread may itself be non-stationary in finite samples. "
            "Some MA pairs exhibit mean-reversion empirically, but this must be "
            "confirmed with ADF before assuming stationarity in any model."
        ),
        outcome=FeatureOutcome.PREDICTIVE,
        notes=(
            "EMPIRICAL RESULT (2026-03-10): ADF stationary when ATR-normalised "
            "(best: t=−4.86 at T1=24, T2=120). DIRECTION INVERSION: positive spread "
            "predicts lower future returns — overextension, not confirmation. "
            "Best config: T1=72, T2=240, IC=−0.318 at H=24. "
            "Consistent across all vol regimes; Med Vol strongest (IC=−0.417). "
            "ATR normalisation required — raw spread does not pass ADF."
        ),
    ))

    cat.register(FeatureSpec(
        name="ma_crossover_signal",
        category=FeatureCategory.TREND,
        hypothesis=(
            "When sma(S*, T1) crosses above sma(S*, T2), a persistent uptrend is "
            "beginning from genuine macro directional flows rather than noise. "
            "Trading long at crossover until the next crossover should generate "
            "positive returns because the HP filter ensures the crossover reflects "
            "a real trend shift."
        ),
        lookback="T1 ∈ {24, 48, 72}, T2 ∈ {120, 168, 240, 480} bars",
        parameters={
            "lambda": [1e6, 1e7, 1e8, 1e9, 3.9e9, 1e11],
            "T1": [24, 48, 72],
            "T2": [120, 168, 240, 480],
        },
        expected_stationary=None,
        stationarity_notes=(
            "Binary signal (+1 / −1). ADF is not applicable to a bounded discrete "
            "series. Treat as a regime indicator, not a continuous feature."
        ),
        outcome=FeatureOutcome.PREDICTIVE,
        notes=(
            "EMPIRICAL RESULT (2026-03-10) — original direction: All 72 combinations fail. "
            "DSR=0.000 for every combination. Best Sharpe=+0.258 (λ=1B, T1=72, T2=240). "
            "Root cause: wrong direction — IC is negative (mean reversion), signal was long. "
            "REVERSED SIGNAL RESULT (2026-03-10): 72/72 positive Sharpe with full filter. "
            "Variant C (reversed + vol_regime==1 + directional trend_dev_240>1.5σ): "
            "67/72 combinations pass DSR>0.95. Best: λ=1M T1=48 T2=480 Sharpe=+3.71 "
            "Ann Ret=+12.1% Max DD=−1.3% n_trades=59 DSR=1.000. "
            "Regime-gated only (Variant B): 14/72 pass DSR>0.95, trade counts 73–92. "
            "PRODUCTION SIGNAL: short when sma(S*,T1) > sma(S*,T2) AND vol_regime==1 "
            "AND trend_dev_240 > +1.5σ; long when reversed conditions. "
            "Trade count shortfall: best combos reach 59–92; target is 100+. Expand dataset."
        ),
    ))

    cat.register(FeatureSpec(
        name="ma_crossover_age",
        category=FeatureCategory.TREND,
        hypothesis=(
            "Bars elapsed since the last MA crossover on S*(t) measures trend maturity. "
            "Counter-trend signals occurring in old, mature trends (many bars since "
            "crossover) should have higher win rates than those in young trends."
        ),
        lookback="Dynamic (count from last crossover event)",
        parameters={"T1": [24, 48, 72], "T2": [120, 168, 240, 480]},
        expected_stationary=False,
        stationarity_notes=(
            "Unbounded counter — grows without limit until reset. "
            "Non-stationary by construction. Normalise by average crossover interval "
            "before using in a model: maturity = age / mean_crossover_interval."
        ),
        outcome=FeatureOutcome.INCONCLUSIVE,
        notes=(
            "EMPIRICAL RESULT (2026-03-10): ADF stationary empirically (t=−4.45, p<0.0003) "
            "due to bounded crossover count. IC = +0.062 at H=24 (p<0.011). "
            "Positive IC: older trend → higher future returns — reassertion, not exhaustion. "
            "Primary signal in Low Vol regime (IC=+0.086). Only ~10 crossover events "
            "in 2,104 bars makes statistics noisy. Normalise before production use."
        ),
    ))

    # ------------------------------------------------------------------
    # Mean-reversion features
    # ------------------------------------------------------------------

    cat.register(FeatureSpec(
        name="trend_deviation_from_ma",
        category=FeatureCategory.MEAN_REVERSION,
        hypothesis=(
            "The ATR-normalised distance of S*(t) from its long MA measures how far "
            "the trend has extended from its own average. Extreme positive values "
            "indicate overextension — a condition that historically precedes either "
            "a pause or a reversal, predicting counter-trend entry quality."
        ),
        lookback="T2 ∈ {120, 168, 240, 480} bars",
        parameters={"T2": [120, 168, 240, 480], "atr_window": [20]},
        expected_stationary=None,
        stationarity_notes=(
            "ATR normalisation is expected to produce a stationary series in stable "
            "regimes, but ATR itself can trend during volatility regime shifts. "
            "Test with ADF. Values beyond ±2σ of rolling history mark exhaustion zones."
        ),
        outcome=FeatureOutcome.PREDICTIVE,
        notes=(
            "EMPIRICAL RESULT (2026-03-10): ADF stationary when ATR-normalised "
            "(T2=240: t=−3.89, p<0.002). IC = −0.344 at H=24 (T2=240), "
            "second strongest feature overall. Med Vol IC = −0.535 — strongest "
            "signal in the entire set in that regime. Overextension hypothesis confirmed. "
            "Longer T2 (240 vs 120) yields materially stronger IC. "
            "Primary entry quality filter: deviation > 1.5σ marks exhaustion zone."
        ),
    ))

    cat.register(FeatureSpec(
        name="distance_from_ma_20_raw",
        category=FeatureCategory.MEAN_REVERSION,
        hypothesis=(
            "Extreme distance from the 20-bar SMA on raw USDJPY H1 price indicates "
            "temporary mispricing — price should revert. Baseline mean-reversion "
            "feature, no HP filtering."
        ),
        lookback="20 bars",
        parameters={"window": [20]},
        expected_stationary=True,
        stationarity_notes=(
            "Z-scored deviation from a rolling MA is stationary by construction "
            "when computed as (price − sma) / rolling_std(price − sma)."
        ),
        outcome=FeatureOutcome.INCONCLUSIVE,
        notes=(
            "EMPIRICAL RESULT (2026-03-10): ADF strongly stationary (t=−9.29, p≈0). "
            "IC = +0.069 at H=24 (unexpected positive sign); H=4 IC ≈ 0. "
            "Mean reversion already captured at shorter horizons — only residual "
            "positive autocorrelation visible at H=24. Strictly inferior to "
            "trend_deviation_from_ma on all metrics. Baseline only; not for production."
        ),
    ))

    # ------------------------------------------------------------------
    # Benchmark features (raw price, no HP filtering)
    # ------------------------------------------------------------------

    cat.register(FeatureSpec(
        name="ma_spread_raw_5020",
        category=FeatureCategory.TREND,
        hypothesis=(
            "Positive spread (sma(50) > sma(200)) on raw USDJPY H1 price indicates "
            "a persistent uptrend from slow-moving institutional capital. Provides a "
            "baseline trend filter independent of HP filtering — useful for comparing "
            "with HP-based features."
        ),
        lookback="50, 200 bars",
        parameters={"T1": [50], "T2": [200]},
        expected_stationary=None,
        stationarity_notes=(
            "The spread between two SMA filters of a non-stationary price series "
            "is not guaranteed to be stationary. It may appear mean-reverting in "
            "small samples while actually drifting. Test empirically with ADF."
        ),
        outcome=FeatureOutcome.PREDICTIVE,
        notes=(
            "EMPIRICAL RESULT (2026-03-10): ADF stationary when ATR-normalised "
            "(t=−4.65, p<0.0002). IC = −0.228 at H=24. Med Vol IC = −0.472. "
            "BENCHMARK CONCLUSION: Comparable IC to HP-filtered MA spread at equivalent "
            "lookback on N=2,104 bars — HP filter adds marginal value at this sample size. "
            "Revisit with ≥10,000 bars to isolate HP filter contribution from sample noise."
        ),
    ))

    # ------------------------------------------------------------------
    # Diagnostic / meta-features
    # ------------------------------------------------------------------

    cat.register(FeatureSpec(
        name="lambda_sensitivity_score",
        category=FeatureCategory.DIAGNOSTIC,
        hypothesis=(
            "When multiple lambda values produce consistent S*(t) slopes, the trend "
            "estimate is robust and signals should be trusted. When lambdas disagree "
            "on slope direction, the trend is ambiguous and signal confidence should "
            "be reduced or trades filtered out."
        ),
        lookback="N/A — computed across lambda values at each bar",
        parameters={
            "lambdas": [1e6, 1e7, 1e8, 1e9, 3.9e9, 1e11],
            "atr_window": [20],
        },
        expected_stationary=True,
        stationarity_notes=(
            "std(ΔS*(t) across lambdas) measures slope disagreement, not level "
            "disagreement. Since each ΔS* is I(0), the std across them is also "
            "bounded and stationary. Normalise by ATR to adjust for vol regimes."
        ),
        outcome=FeatureOutcome.INCONCLUSIVE,
        notes=(
            "EMPIRICAL RESULT (2026-03-10): Lambda collapse — λ ≥ 1B are functionally "
            "identical on N=2,104 bars (slope ρ ≥ 0.9995). Sensitivity score between "
            "collapsed lambdas is near zero, partially invalidating the disagreement "
            "hypothesis. Effective distinct lambdas: {1M, 100M, 1B}. Feature may regain "
            "value with N > 10,000 bars where high-lambda regimes diverge. "
            "Standalone IC not yet tested."
        ),
    ))

    cat.register(FeatureSpec(
        name="vol_regime",
        category=FeatureCategory.DIAGNOSTIC,
        hypothesis=(
            "USDJPY H1 exhibits distinct volatility regimes (carry unwind, "
            "range-bound, trending). Classifying the current regime before evaluating "
            "any directional feature improves signal quality by preventing "
            "regime-averaging that dilutes predictive power."
        ),
        lookback="Rolling 20-bar realized volatility; quantile thresholds computed over 252-bar window",
        parameters={"vol_window": [20], "classification_window": [252]},
        expected_stationary=True,
        stationarity_notes=(
            "Categorical output (0=low, 1=medium, 2=high vol). Bounded by construction."
        ),
        outcome=FeatureOutcome.PREDICTIVE,
        notes=(
            "EMPIRICAL RESULT (2026-03-10): Confirmed essential conditioning variable. "
            "Med Vol is the signal regime for all mean-reversion features. "
            "trend_deviation_from_ma IC = −0.535 in Med Vol vs −0.266/−0.228 in Low/High. "
            "Without regime conditioning, signal is averaged down to −0.344. "
            "Regime filter is not optional — it is the primary modulator of "
            "all feature predictive power in this strategy."
        ),
    ))

    return cat


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

CATALOG: FeatureCatalog = _build_catalog()
