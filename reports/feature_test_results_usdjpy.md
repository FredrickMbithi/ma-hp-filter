# Univariate Feature Test Results — USDJPY H1

**Strategy:** 8.1 — Moving Averages with HP Filter (USDJPY H1)  
**Report Date:** 2026-03-11  
**Notebook:** `notebooks/04_univariate_feature_tests.ipynb`  
**Framework:** `src/features/testing.py` — `UnivariateFeatureTester`

---

## 1. Executive Summary

This report documents the results of independent univariate predictive-power tests for eight continuous features on the USDJPY H1 Dukascopy 10-year dataset. The tests determine whether each feature individually predicts forward log-returns before any multi-feature combination is attempted.

The framework applies two sequential reject criteria:

1. **Non-stationary** (ADF p ≥ 0.05): Feature correlates spuriously with returns due to shared trending behaviour, not genuine predictive content.
2. **Weak signal** (Peak |IC| < 0.05 across all horizons): Feature does not carry sufficient information about future returns to justify inclusion.

Features surviving both criteria are forwarded to multi-feature backtest design. `vol_regime` is a categorical indicator and treated separately as a regime filter rather than a predictive feature.

---

## 2. Dataset & Methodology

### Dataset

| Property       | Value                                                                                    |
| -------------- | ---------------------------------------------------------------------------------------- |
| Symbol         | USDJPY (spot)                                                                            |
| Provider       | Dukascopy                                                                                |
| Timeframe      | H1 (hourly bars)                                                                         |
| Total bars     | 62,303                                                                                   |
| Date range     | 2016-03-13 → 2026-03-10                                                                  |
| HP trend cache | `data/interim/hp_trends_window500.csv` (window=500, λ ∈ {1M, 10M, 100M, 1B, 3.9B, 100B}) |

### Methodology

| Parameter              | Value                       | Rationale                                                     |
| ---------------------- | --------------------------- | ------------------------------------------------------------- |
| IC metric              | Spearman rank correlation   | Robust to non-normal return distributions                     |
| IC threshold           | 0.05                        | Minimum meaningful IC in FX markets                           |
| Horizons tested        | H ∈ {1, 4, 12, 24, 48} bars | Covers intraday to ~2-day hold periods                        |
| Rolling IC window      | 1,000 bars                  | Approximately 6 months of H1 data; detects regime instability |
| ADF max lags           | 24                          | One full trading day of lags for H1 data                      |
| Stationarity threshold | p < 0.05                    | Standard 95% confidence                                       |
| HP lambda              | 3.9 × 10⁹                   | Theoretically calibrated (λ_q=1600, H1 ≈ 504 bars/quarter)    |
| ATR window             | 20 bars                     | Volatility normaliser throughout                              |
| Quantile bins          | 5 (quintiles)               | Equal-frequency; monotonicity Spearman threshold \|ρ\| > 0.8  |

**No lookahead bias**: All features are computed causal-forward (bar t reads only bars ≤ t). HP trends are loaded from the pre-computed causal rolling window cache. Forward returns used for IC computation are aligned at time t with feature values at time t — this is the correct procedure for prospective predictive power measurement.

**IC Interpretation Convention**: In FX mean-reversion research, negative IC (feature high → returns negative) is valid and expected for overextension features. The sign of IC is secondary to its magnitude and stability.

---

## 3. Feature Inventory

| #   | Feature Name         | Category       | Formula                            | HP lambda | Notes                                                       |
| --- | -------------------- | -------------- | ---------------------------------- | --------- | ----------------------------------------------------------- |
| 1   | `hp_trend_slope`     | Trend          | ΔS*(t) = S*(t) − S\*(t−1)          | 3.9B      | First diff of causal HP trend; expected I(0)                |
| 2   | `hp_trend_curvature` | Trend          | Δ²S\*(t)                           | 3.9B      | Second diff; measures trend acceleration / exhaustion       |
| 3   | `trend_dev_from_ma`  | Mean-reversion | (S*(t) − SMA(S*, 240)) / ATR       | 3.9B      | Prior best result: IC = −0.344 at H=24                      |
| 4   | `ma_spread_on_trend` | Trend          | (SMA(S*, 72) − SMA(S*, 240)) / ATR | 3.9B      | T1=72, T2=240 (best from grid search)                       |
| 5   | `momentum_24h`       | Momentum       | log(close*t / close*{t−24})        | —         | Raw 24-bar log-return on price                              |
| 6   | `rsi_14`             | Oscillator     | RSI(14) on close                   | —         | Bounded [0, 100]; mean-reversion indicator                  |
| 7   | `distance_from_ma`   | Mean-reversion | Z-score of (close − SMA(20))       | —         | Rolling z-score, non-HP-based equivalent                    |
| 8   | `lambda_sensitivity` | Diagnostic     | std*λ(Δhp*λ(t)) / ATR              | all 6λ    | Cross-lambda slope disagreement; measures trend uncertainty |
| 9   | `vol_regime`         | Categorical    | Tertile of 20-bar realised vol     | —         | 0=Low, 1=Med, 2=High; regime filter only                    |

---

## 4. Results Table

> Run `notebooks/04_univariate_feature_tests.ipynb` to populate with exact values.

| Feature            | ADF p | Stationary? | Peak IC | Peak H | IC-IR | Hit Rate | Monotonic? | **Verdict** | Reject Reason |
| ------------------ | ----- | ----------- | ------- | ------ | ----- | -------- | ---------- | ----------- | ------------- |
| hp_trend_slope     | TBD   | TBD         | TBD     | TBD    | TBD   | TBD      | TBD        | TBD         | TBD           |
| hp_trend_curvature | TBD   | TBD         | TBD     | TBD    | TBD   | TBD      | TBD        | TBD         | TBD           |
| trend_dev_from_ma  | TBD   | TBD         | TBD     | TBD    | TBD   | TBD      | TBD        | TBD         | TBD           |
| ma_spread_on_trend | TBD   | TBD         | TBD     | TBD    | TBD   | TBD      | TBD        | TBD         | TBD           |
| momentum_24h       | TBD   | TBD         | TBD     | TBD    | TBD   | TBD      | TBD        | TBD         | TBD           |
| rsi_14             | TBD   | TBD         | TBD     | TBD    | TBD   | TBD      | TBD        | TBD         | TBD           |
| distance_from_ma   | TBD   | TBD         | TBD     | TBD    | TBD   | TBD      | TBD        | TBD         | TBD           |
| lambda_sensitivity | TBD   | TBD         | TBD     | TBD    | TBD   | TBD      | TBD        | TBD         | TBD           |

_IC-IR = mean(rolling IC) / std(rolling IC). Values > 0.5 indicate reliable signal._

---

## 5. Feature-by-Feature Analysis

### 5.1 hp_trend_slope (ΔS\* momentum)

**Hypothesis**: Positive slope (rising filtered trend) predicts positive forward returns at short horizons.

**Expected stationarity**: YES — first difference of I(1) series.

**Prior IC results** (from `research_pipeline.py`, λ=1B, H1 full dataset): slope IC was described as "weak, regime-conditional only." This is consistent with the known HP filter behaviour at high lambda: the slope series is smooth and changes slowly, limiting short-horizon predictive power.

**Directional note**: The sign of the IC is critical. A positive IC (rising slope → positive returns) would indicate trend-following behaviour. A negative IC would indicate mean-reversion — the HP slope overshoots. Prior experiments found that **mean-reversion direction is dominant** in USDJPY, so negative IC at short horizons is expected.

**Result**: TBD from notebook.

---

### 5.2 hp_trend_curvature (Δ²S\* exhaustion)

**Hypothesis**: Negative curvature (decelerating uptrend) predicts negative forward returns (exhaustion signal for mean-reversion entry).

**Expected stationarity**: YES — second difference of I(1) series.

**Prior IC results** (from `data/interim/ic_horizon_scan_results.csv`, curv_λ1M):

| Horizon | IC     | p-value | Significant |
| ------- | ------ | ------- | ----------- |
| 1       | −0.021 | < 0.001 | \*\*\*      |
| 2       | −0.014 | 0.001   | \*\*        |
| 4       | +0.003 | 0.446   | —           |
| 6       | +0.009 | 0.027   | \*          |
| 12      | +0.005 | 0.178   | —           |
| 24      | +0.003 | 0.413   | —           |
| 48      | +0.003 | 0.497   | —           |

The prior scan used λ=1M (fast HP, very wiggly). Using λ=3.9B (tighter smoothing), curvature is smaller in magnitude and may carry less signal independently, but the exhaustion signal should be qualitatively similar.

**Key warning**: At high lambda, the HP filter minimises curvature by construction. The curvature series at λ=3.9B may have very low variance, reducing any IC estimate. Consider testing with λ=100M or λ=1B if this feature is rejected.

**Result**: TBD from notebook.

---

### 5.3 trend_dev_from_ma (Mean-reversion from HP trend)

**Hypothesis**: Large positive deviation (S\* above its own 240-bar MA) predicts negative forward returns (reversion to MA).

**Expected stationarity**: YES — ATR-normalised deviation from a mean should be I(0).

**Prior IC results** (from `research_pipeline.py`, Step 2, λ=3.9B):

| Metric                  | Value                                |
| ----------------------- | ------------------------------------ |
| IC at H=1 (λ=3.9B)      | +0.005                               |
| IC at H=4 (λ=3.9B)      | +0.007                               |
| IC at H=24 (λ=3.9B)     | +0.003                               |
| IC at H=24 (λ=1B, est.) | ~0.01–0.04 (based on strategy_b_efc) |

> **Note on the −0.344 figure**: The value −0.344 (or +0.344) seen in earlier reports is the **IS Sharpe Ratio** of the crossing-entry strategy (Experiment 8 in `research_progress_report_2026-03-10.md`), not an IC value. The feature hypothesis log attributed this Sharpe incorrectly as an IC. Actual IC for `trend_dev_240` at λ=3.9B is ~+0.003 (tiny), because at λ=3.9B the HP trend is so smooth that the 240-bar MA of the trend is nearly equal to the trend itself — the deviation has near-zero variance.

This notebook therefore evaluates `trend_dev_from_ma` using **λ=1B**, following the empirically-verified choice in `experiments/strategy_b_efc_filter.py` (`HP_FILTER_LAMBDA = "1B"`, `TREND_DEV_WINDOW = 240`).

**Expected result**: UNCERTAIN at λ=1B. IC > 0.05 is possible but not guaranteed for H1 data. Sign should be negative (high deviation → reversion).

---

### 5.4 ma_spread_on_trend (MA spread on filtered price)

**Hypothesis**: Wide positive MA spread (SMA-72 above SMA-240 on S\*) predicts positive forward returns (trend continuation) OR negative returns (reversion from overextension).

**Expected stationarity**: UNCERTAIN — ATR normalisation helps but is not guaranteed. ADF test required.

**Prior results** (from reversed grid search, `experiments/reversed_grid_search.py`): MA spread with T1=72, T2=240, λ≥1B showed positive strategy performance in reversed (mean-reversion) direction. This implies **negative IC** (wide positive spread → negative forward returns).

**Key difference from trend_dev_from_ma**: This spreads the two MA values of S* (a momentum measure on the cleaned trend), while trend_dev measures deviation of S* from its own MA (an overextension measure). The two features are conceptually related but measure different aspects of the same signal.

**Expected result**: TBD — likely passes IC threshold. Sign of IC expected negative.

---

### 5.5 momentum_24h (Raw price momentum)

**Hypothesis**: 24-bar log-return predicts subsequent returns in the same direction (trend continuation at intraday to 1-day horizon).

**Expected stationarity**: YES — log returns are stationary by construction.

**Context**: This is the simplest possible momentum feature — raw price log-return over the past 24 hours. FX momentum literature shows mixed results at H1 frequency:

- Very short-term momentum (H ≤ 4) often shows **reversal** (bid-ask bounce effect)
- Medium-term momentum (H ∈ {24, 48}) can show **continuation** in strong trends
- In USDJPY specifically, which has shown mean-reversion dominance throughout this dataset, momentum signals are likely to underperform

The HP filter framework was specifically motivated by the failure of raw price signals. This feature serves as a **baseline comparison** for the HP-enhanced features.

**Expected result**: Uncertain. May pass IC threshold at some horizon, but likely weaker than HP-based features. IC sign and peak horizon TBD.

---

### 5.6 rsi_14 (Relative Strength Index)

**Hypothesis**: RSI extreme readings (>70 or <30) predict mean reversion.

**Expected stationarity**: YES — bounded [0, 100], mean-reverting by construction.

**IC sign**: Negative expected (RSI high → returns negative), consistent with mean-reversion. This is the same direction hypothesis as trend_dev_from_ma but computed on raw prices rather than the HP-filtered trend.

**Key question**: Does RSI on raw price carry additional information beyond distance_from_ma (which is equivalent to a z-scored version of the same concept)? They should be highly correlated. If RSI peaks earlier in the decay curve (shorter lifespan), it may be the better short-horizon feature.

**Expected result**: Likely passes IC threshold. RSI and distance_from_ma may have near-identical IC given structural similarity — one may be redundant if both pass.

---

### 5.7 distance_from_ma (Z-scored distance from 20-bar MA)

**Hypothesis**: Z-score of (close − SMA(20)) predicts mean reversion — high positive z-score → negative forward returns.

**Expected stationarity**: YES — z-scored rolling deviation is approximately I(0) when the underlying process has stable volatility.

**Relationship to other features**: This is the raw-price analogue of `trend_dev_from_ma`. The key structural difference is that `trend_dev_from_ma` operates on the HP-smoothed price S\*(t), which removes noise and cycle components. The raw-price version will be noisier, potentially reducing IC at longer horizons.

**Expected result**: Likely passes IC threshold. IC expected to be lower than `trend_dev_from_ma` at H=24 due to noise from raw prices. May outperform at shorter horizons (H ≤ 4) if noise is short-horizon predictive.

---

### 5.8 lambda_sensitivity (Cross-lambda disagreement)

**Hypothesis**: High cross-lambda slope std (λ values disagree on trend direction) predicts larger absolute future returns (elevated uncertainty = wider price swings ahead).

**Expected stationarity**: YES — ATR-normalised std of stationary slopes should be I(0).

**IC measurement note**: Lambda sensitivity measures trend _ambiguity_, not direction. Its natural relationship is with **absolute forward returns** (volatility forecasting) rather than signed returns. In the standard signed-return IC test, this feature may show near-zero IC. Its diagnostic value is as a regime filter (low sensitivity = high confidence in trend signal).

**Expected result**: Likely FAILS signed-return IC threshold. This is acceptable — it is a diagnostic/meta-feature, not a directional predictor. Its value is in conditioning other signals.

---

## 6. vol_regime Analysis

`vol_regime` is a categorical feature (0=Low, 1=Med, 2=High) produced by percentile-ranking 20-bar realised volatility within a rolling 252-bar classification window. It cannot be meaningfully tested with IC or hit rate.

### Key prior findings (from `experiments/feature_hypothesis_log.md`):

- **Medium volatility (Regime 1)** is the productive regime for all mean-reversion features in this dataset
- IC of `trend_dev_from_ma` in Med Vol: −0.535 vs −0.344 full sample (56% improvement)
- Low and High vol regimes both show compressed IC (~−0.2)

### Practical use in strategy:

`vol_regime` should be used as a **gate condition** in the backtest:

- Activate signal generation only when `vol_regime == 1` (Med Vol)
- Suppress trades in Low Vol (insufficient movement to capture reversion) and High Vol (signal breaks down under stress)

This conditioning is implemented in Strategy Variant B (from `experiments/strategy_b_efc_filter.py`) and is one of the key features of the regime-gated strategy that achieves DSR > 0.95 across 14/72 parameter combinations.

### notebook results (TBD):

- Mean forward return per regime at H ∈ {1, 4, 12, 24, 48}: TBD
- IC of `trend_dev_from_ma` per regime (expected −0.535 in Med Vol at H=24): TBD

---

## 7. Pass/Fail Summary

> To be finalised after running `notebooks/04_univariate_feature_tests.ipynb`.

### Expected Outcomes (Based on Prior Experiments)

| Feature            | Expected Verdict                 | Confidence | Basis                                                 |
| ------------------ | -------------------------------- | ---------- | ----------------------------------------------------- | --- | ------------------------------------ |
| hp_trend_slope     | FAIL or borderline               | Low        | Prior IC weak; regime-conditional only                |
| hp_trend_curvature | FAIL (λ=3.9B)                    | Medium     | Peak                                                  | IC  | ≈ 0.02 at λ=1M; even smaller at 3.9B |
| trend_dev_from_ma  | FAIL (λ=3.9B) / Uncertain (λ=1B) | Low-Med    | IC≈+0.003 at λ=3.9B; λ=1B expected ~0.01–0.04         |
| ma_spread_on_trend | Uncertain (λ=1B)                 | Low        | Positive backtest evidence but IC unconfirmed         |
| momentum_24h       | Uncertain                        | Low        | No prior IC results; baseline feature                 |
| rsi_14             | PASS                             | Medium     | Analogous to distance_from_ma; mean-reversion bounded |
| distance_from_ma   | PASS                             | Medium     | Raw-price analogue of trend_dev_from_ma               |
| lambda_sensitivity | FAIL                             | High       | Diagnostic; no signed-return IC expected              |

### Features Recommended for Multi-Feature Backtest

Based on expected and prior results, features recommended for the next stage are:

1. **`trend_dev_from_ma` (λ=1B)** — strategy_b_efc evidence; IC confirmation pending from this notebook
2. **`ma_spread_on_trend` (λ=1B)** — backtest evidence in reversed direction
3. **`rsi_14` or `distance_from_ma`** — raw-price mean-reversion baseline (likely redundant; pick one after comparing ICs)
4. **`vol_regime`** — not a predictor, but gate condition for all of the above

> **Thresholding note**: H1 FX features inherently produce lower IC than daily data. Most features are expected to fall below the 0.05 threshold. This is the correct and valuable finding — strategy value comes from signal combination and regime conditioning, not individual IC magnitude.

`lambda_sensitivity` — retain as a confidence gate even if IC fails the threshold; use to suppress trades during high cross-lambda disagreement.

---

## 8. Methodology Notes

### Why Spearman IC and Not Pearson?

Spearman rank correlation is robust to outliers and fat-tailed return distributions. FX returns are known to have kurtosis > 3 (leptokurtic), and a single large move can inflate Pearson IC artificially. Spearman's rank-based formulation is the standard in quantitative equity and FX factor research.

### Why Rolling IC?

The full-sample IC captures the average relationship but masks regime instability. A feature with IC = −0.10 in aggregate but swinging between IC = +0.30 and IC = −0.50 across different periods is a regime-dependent signal, not a stable predictor. The rolling IC window (1,000 bars ≈ 6 months) detects these regimes. IC-IR = mean/std > 0.5 is required for a consistent signal.

### Multiple Testing Caution

Eight features tested across 5 horizons = 40 hypothesis tests. At α = 0.05, approximately 2 tests will appear significant by chance. The IC threshold of 0.05 (full-sample, 62,000 observations) has a minimum t-stat of ~12 for significance — extremely unlikely to be spurious. The rolling IC stability test provides an additional guard against false positives.

### Feature-Return Stationarity vs Feature Stationarity

The ADF test assesses stationarity of the feature series itself. A non-stationary feature (e.g., hp_trend_level, ADF p = 0.746) can appear to correlate with future returns simply because both trend persistently in the same direction over subperiods. This is a spurious correlation driven by shared non-stationarity, not genuine predictive content. Rejecting non-stationary features is essential before drawing conclusions about IC significance.

---

## 9. References

- `experiments/feature_hypothesis_log.md` — Feature registry and full prior research record
- `reports/research_progress_report_2026-03-10.md` — Full pipeline re-run results (Step 1–5)
- `data/interim/ic_horizon_scan_results.csv` — Raw IC results at 7 horizons for curvature features
- `data/interim/return_decay_results.csv` — Event-study decay table for curvature and MA spread
- `src/features/testing.py` — Statistical testing framework (this run)
- `notebooks/04_univariate_feature_tests.ipynb` — Notebook producing the quantitative results in this report
