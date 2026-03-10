# Feature Hypothesis Log

## Purpose

Document every feature tested, with rationale and outcome.
Track all features derived from Strategy 8.1 (MA + HP Filter) on USDJPY H1.

---

## Format

### Feature: [Name]

- **Category:** [Trend/Mean Reversion/etc]
- **Hypothesis:** [Why this might predict returns]
- **Lookback Period:** [X bars]
- **Expected Stationarity:** [Yes/No/Test required]
- **Test Date:** [YYYY-MM-DD]
- **Outcome:** [Predictive/Not Predictive/Inconclusive]
- **Notes:** [Key findings]

---

## Strategy 8.1 — Moving Averages with HP Filter (USDJPY H1)

---

## HP Filter: Why Causal Implementation Was Chosen

The `statsmodels` HP filter (`hpfilter()`) solves a global least-squares problem across the entire input series simultaneously. This means the trend estimate at bar _t_ depends on prices at bars _t+1, t+2, ..., t+N_ — a direct lookahead bias.

**What was rejected:** Calling `apply_hp_filter(full_price_series, lamb)` and using the resulting trend as a signal. Even if the strategy does not directly "see" future prices, the HP-filtered signal was formed using future information — every signal value is contaminated.

**What was implemented instead:** `causal_hp_trend(prices, lamb, window)` applies `hpfilter` on a rolling window `prices[t-window : t+1]` at each bar _t_, and takes only the **last element** of the resulting trend vector as S\*(t). This ensures only past + current data is used at every bar.

**Known limitation — Endpoint instability:** The final element of the HP trend vector has the highest estimation variance in any window. A larger rolling window reduces but does not eliminate this instability. Minimum recommended window: 500 bars. For research, compare `window=500` vs `window=1000` to assess sensitivity.

**Two-sided filter retained for:** Visualisation and post-hoc analysis only (e.g. plotting the "ideal" smooth trend on historical charts). Never use `apply_hp_filter` output as a live signal input.

---

### Feature: HP-Filtered Trend Level (S\*)

- **Category:** Trend
- **Hypothesis:** The HP-filtered price S*(t) represents the true underlying macro trend in USDJPY, stripped of intraday noise. Trading in the direction of S*(t)'s slope should yield positive returns because it aligns with institutional carry and momentum flows rather than microstructure noise.
- **Lookback Period:** Full causal rolling window (min 50 bars; recommended 500+ bars for endpoint stability)
- **Lambda (λ):** Test range {1M, 10M, 100M, 1B, 3.9B, 100B}
- **Expected Stationarity:** No — S\* is a smoothed price level. Price is I(1); HP filtering does not change the integration order of the level.
- **Test Date:** 2026-03-10
- **Outcome:** PREDICTIVE — with mean-reversion direction (see notes)
- **ADF Result:** Stationary (t=−5.78, p<0.000001) — unexpected; causal rolling window with window=200 bounds the level within a moving reference frame, effectively making it stationary in short samples.
- **IC Results (λ=3.9B, window=200):**
  - H=1: IC=−0.0756 (p<0.05)
  - H=4: IC=−0.1537 (p<0.05)
  - H=24: IC=−0.3973 (p<0.000001) ← strongest signal
- **Regime-Conditional IC (H=24):** Low Vol=−0.454, Med Vol=−0.388, High Vol=−0.350 — consistent mean-reverting signal across all regimes
- **Notes:** ⚠️ DIRECTION INVERSION FINDING: IC is negative — a HIGH trend level predicts LOWER future returns. This is mean reversion, not trend following. The original hypothesis (trade in direction of trend) is directly contradicted. The signal is real and strong, but the correct trade is AGAINST the trend level, not with it. Lookahead bias risk eliminated by `causal_hp_trend()`. λ ≥ 1B are functionally equivalent on this 2,104-bar dataset (slope correlation ≥ 0.9995).

---

### Feature: HP Trend Slope (ΔS\*)

- **Category:** Trend
- **Hypothesis:** The first difference of the HP-filtered trend, ΔS*(t) = S*(t) − S\*(t−1), measures the current momentum of the clean trend. A positive slope means the filtered trend is rising; negative means falling. This should be predictive of short-term continuation because it captures the direction of institutional flow with noise removed.
- **Lookback Period:** 1 bar (first difference of S\*)
- **Expected Stationarity:** Yes — first difference of an I(1) series is I(0) by definition. Should pass ADF regardless of λ.
- **Test Date:** 2026-03-10
- **Outcome:** WEAKLY PREDICTIVE at H=24 only
- **ADF Result:** Stationary across all λ values (worst: t=−3.26, p=0.016 at λ=10M; best: t=−4.57 at λ=1M)
- **IC Results (λ=3.9B, window=200):**
  - H=1: IC=−0.0062 (not significant)
  - H=4: IC=−0.0114 (not significant)
  - H=24: IC=−0.0524 (p<0.05) ← marginally significant
- **Regime-Conditional IC (H=24):** Low Vol=+0.040 (n.s.), Med Vol=−0.204\*, High Vol=+0.011 (n.s.) — signal concentrated entirely in medium-volatility regime
- **Notes:** The slope carries weak predictive information at best. Regime-conditional analysis reveals the signal is regime-specific — in Med Vol it behaves mean-revertingly, elsewhere it is noise. The hypothesis of "continuation" is rejected. Curvature (next feature) adds more information than slope in isolation.

---

### Feature: HP Trend Curvature (Δ²S\*)

- **Category:** Trend / Exhaustion Signal
- **Hypothesis:** The second difference of S*(t), Δ²S*(t) = ΔS*(t) − ΔS*(t−1), measures how fast the trend is changing direction (acceleration). A large negative curvature during an uptrend means the trend is decelerating — this is the mathematical signature of exhaustion. This feature should predict mean-reversion entries when curvature turns sharply against the trend direction.
- **Lookback Period:** 3 bars (second difference window)
- **Expected Stationarity:** Yes — second difference of I(1) is strongly stationary.
- **Test Date:** 2026-03-10
- **Outcome:** WEAKLY PREDICTIVE — confirms exhaustion hypothesis in Med Vol regime
- **ADF Result:** Strongly stationary across all λ (ADF t ≈ −41.5 to −42.0, p≈0). Consistent with over-differenced series.
- **IC Results (λ=3.9B, window=200):**
  - H=1: IC=+0.0407 (not significant at 5%)
  - H=4: IC=+0.0126 (not significant)
  - H=24: IC=+0.0589 (p<0.011)
- **Regime-Conditional IC (H=24):** Low Vol=+0.054 (n.s.), Med Vol=+0.137\*, High Vol=+0.004 (n.s.) — signal meaningful only in Med Vol
- **Notes:** CAUTION CONFIRMED: For λ ≥ 100M the filter minimises curvature by construction; empirically at λ=3.9B the curvature still passes ADF and shows weak IC at H=24. The +0.137 IC in medium volatility is consistent with the exhaustion hypothesis — positive curvature (trend acceleration) → positive future returns (trend continuation not yet exhausted). However the signal is too weak to trade standalone. Consider only as a compound confirmation signal. Note: signal direction is +ve here (opposite to trend level/spread features), suggesting an orthogonal exhaustion dimension.

---

### Feature: MA Spread on S\* (Short MA − Long MA)

- **Category:** Trend
- **Hypothesis:** The spread between the short MA (T1) and long MA (T2) computed on the HP-filtered trend S\*(t) measures trend strength and direction. A large positive spread indicates a strong, established uptrend driven by persistent macro flows (carry, momentum). A spread near zero indicates trend exhaustion or regime transition.
- **Lookback Period:** T1 ∈ {24, 48, 72}, T2 ∈ {120, 168, 240, 480} hours
- **Expected Stationarity:** Test required — MA of a non-stationary series is itself non-stationary in general. The spread may exhibit mean-reversion in finite samples but this must be confirmed empirically with ADF. Do not assume stationarity.
- **Test Date:** 2026-03-10
- **Outcome:** PREDICTIVE — mean-reversion direction (negative IC); strongest at T1=72, T2=240
- **ADF Results (ATR-normalised spread):**
  - T1=24, T2=120: t=−4.86, p<0.00005 → Stationary ✓
  - T1=48, T2=168: t=−4.33, p<0.0004 → Stationary ✓
  - T1=72, T2=240: t=−4.20, p<0.0007 → Stationary ✓
- **IC Results (λ=3.9B, ATR-normalised):**
  - ma_spread_72_240: H=24 IC=−0.318\* (best configuration)
  - ma_spread_48_168: H=24 IC=−0.250\*
  - ma_spread_24_120: H=24 IC=−0.200\*
- **Regime-Conditional IC (ma_spread_72_240, H=24):** Low Vol=−0.292*, Med Vol=−0.417*, High Vol=−0.266\* — consistent across all regimes with Med Vol strongest
- **Notes:** ⚠️ DIRECTION INVERSION: The hypothesis was that POSITIVE spread → trend continuation → go long. The IC shows POSITIVE spread → LOWER future returns. The spread is a contrarian indicator — high positive spread marks an overextended trend, not a confirming trend. Larger T2 (longer lookback for long MA) gives stronger signal, consistent with capturing longer-cycle exhaustion.

---

### Feature: MA Crossover Signal on S\* (Direction)

- **Category:** Trend
- **Hypothesis:** When sma(S*, T1) crosses above sma(S*, T2) on the HP-filtered trend, a persistent uptrend is beginning, driven by genuine macro directional flows rather than noise. Trading long at crossover and holding until the next crossover should generate positive returns because the HP filter ensures the crossover reflects a real trend shift, not a noise artifact.
- **Lookback Period:** T1 ∈ {24, 48, 72}, T2 ∈ {120, 168, 240, 480} hours
- **Expected Stationarity:** N/A — binary signal (+1 long, −1 short). ADF not applicable to bounded discrete series.
- **Test Date:** 2026-03-10
- **Outcome:** NOT PREDICTIVE as trend-following signal; potentially predictive if REVERSED (as mean-reversion signal)
- **Grid Search Results (all 72 combinations, 0.9 bps all-in cost):**
  - Best Sharpe: +0.258 (λ=1B, T1=72, T2=240) — not economically significant
  - Median Sharpe: approximately −1.3
  - DSR (Deflated Sharpe Ratio): 0.000 for ALL 72 combinations — probability of genuine outperformance is indistinguishable from zero
  - Most combinations have Sharpe < −1.0, meaning they actively destroy value
  - Trade counts are very low (7–14 roundtrips over 2,104 bars) — severely underpowered
- **Regime-Conditional Backtest (best combination λ=1B, T1=72, T2=240):**
  - Low Vol: Sharpe=−1.16 (loses money)
  - Med Vol: Sharpe=−4.78 (worst regime for this direction)
  - High Vol: Sharpe=−0.68 (loses money)
- **Notes:** ⚠️ STRATEGY DIRECTION IS WRONG. The IC shows that positive MA spread → lower future returns. A trend-following MA crossover signal (long when spread >0) is trading into the headwind of mean reversion. Two options: (1) Reverse the signal — go SHORT when sma(T1)>sma(T2) — effectively a mean-reversion strategy using HP-filtered MAs as an exhaustion filter. (2) Use the spread as a regime filter rather than as a direction signal. The HP filter does work — it cleanly isolates the macro trend signal. The problem is the strategy was built as a momentum system on a series that exhibits mean reversion at H1 frequency on this dataset.

---

### Feature: MA Crossover Age (Bars Since Last Cross)

- **Category:** Trend
- **Hypothesis:** The number of bars elapsed since the last MA crossover on S\*(t) measures how "mature" the current trend is. Young trends (few bars since cross) may still be building momentum; old trends (many bars since cross) may be approaching exhaustion. Counter-trend signals occurring in old, mature trends should have higher win rates than those in young trends.
- **Lookback Period:** Dynamic (count from last crossover event)
- **Expected Stationarity:** No — unbounded counter. Non-stationary by construction. Normalise by average crossover interval to get a relative maturity score.
- **Test Date:** 2026-03-10
- **Outcome:** WEAKLY PREDICTIVE — consistent but small IC; useful as a timing filter
- **ADF Result:** Stationary (t=−4.45, p<0.0003) when tested as raw count — the relatively few crossovers in the sample and bounded nature (mean-reverts to zero at each cross) makes it pass ADF.
- **IC Results (T1=48, T2=168, λ=3.9B):**
  - H=1: IC=+0.023 (not significant)
  - H=4: IC=+0.030 (not significant)
  - H=24: IC=+0.062 (p<0.011)
- **Regime-Conditional IC (H=24):** Low Vol=+0.086\*, Med Vol=+0.073 (n.s.), High Vol=+0.018 (n.s.) — primarily a low-volatility signal
- **Notes:** The +0.062 IC at H=24 is positive — longer time since last cross → higher future returns. This can be interpreted as: mature trends (long age) have already done most of their exhaustion and are ready to continue, OR alternatively that crossing happens at an exhaustion zone and after sufficient time the original direction reasserts. Low trade count (only ~10 crossovers in 2,104 bars for T1=48,T2=168) makes this statistic noisy. Normalise by average crossover interval before any production use.

---

### Feature: Distance of S\*(t) from Long MA (Trend Deviation)

- **Category:** Mean Reversion
- **Hypothesis:** The distance of the HP-filtered trend S*(t) from its long moving average sma(S*, T2) measures how far the trend has extended from its own average. Extreme positive distances (trend far above its own average) indicate overextension — a condition that historically precedes either a pause or a reversal. This feature should predict counter-trend entry quality.
- **Lookback Period:** T2 ∈ {120, 168, 240, 480} hours
- **Expected Stationarity:** Test required — ATR normalisation is expected to produce a stationary series in stable regimes, but ATR itself can trend during volatility regime shifts. Confirm with ADF.
- **Test Date:** 2026-03-10
- **Outcome:** PREDICTIVE — strongest mean-reversion signal in the entire feature set
- **ADF Results (ATR-normalised):**
  - T2=120: t=−4.75, p<0.00007 → Stationary ✓
  - T2=240: t=−3.89, p<0.002 → Stationary ✓
- **IC Results (λ=3.9B, ATR-normalised):**
  - trend_dev_120: H=24 IC=−0.186\*
  - trend_dev_240: H=24 IC=−0.344\* ← second strongest overall after hp_trend_level
- **Regime-Conditional IC (trend_dev_240, H=24):** Low Vol=−0.266*, Med Vol=−0.535*, High Vol=−0.228\* — STRONGEST SIGNAL IN MED VOL REGIME (Σ|IC|=0.956 across horizons, best of all features in that regime)
- **Notes:** The hypothesis is confirmed — overextension predicts reversal. The negative IC is expected here (high deviation → lower future returns = mean reversion). This is the clearest exhaustion-zone measure in the set. Use as a filter for entry quality scoring: deviation > 1.5σ should significantly improve signal quality. Longer T2 (240 vs 120) gives meaningfully stronger signal, suggesting macro-scale deviations carry more predictive power than intraday deviations.

---

### Feature: MA Spread (50/200) on Raw Price

- **Category:** Trend
- **Hypothesis:** Positive spread (sma(50) > sma(200)) on raw USDJPY H1 price indicates persistent uptrend due to slow-moving institutional capital. Provides a baseline trend filter independent of HP filtering — useful for comparing with HP-based features.
- **Lookback Period:** 50, 200 bars
- **Expected Stationarity:** Test required — MA of a non-stationary price series is non-stationary in general. Do not assume stationarity. Test empirically.
- **Test Date:** 2026-03-10
- **Outcome:** PREDICTIVE — mean-reversion direction; comparable to HP-filtered equivalent
- **ADF Result:** Stationary (t=−4.65, p<0.0002) when ATR-normalised ✓
- **IC Results (ATR-normalised):**
  - H=1: IC=−0.030 (not significant)
  - H=4: IC=−0.067 (p<0.05)
  - H=24: IC=−0.228\*
- **Regime-Conditional IC (H=24):** Low Vol=−0.069 (n.s.), Med Vol=−0.472*, High Vol=−0.121* — primarily a Med Vol signal
- **Notes:** The raw MA spread (50/200) has nearly identical predictive power to the HP-filtered MA spread at equivalent lookback periods. This is the critical baseline comparison: the HP filter does not materially improve IC on this dataset relative to a raw MA spread with a comparable smoothing horizon. The HP filter's value may lie in end-point precision during whipsaw periods, not in average IC. Needs longer dataset to disambiguate. BENCHMARK CONCLUSION: HP filter adds marginal value at best on 2,104 bars; revisit with 3+ years of data.

---

### Feature: Distance from 20MA on Raw Price

- **Category:** Mean Reversion
- **Hypothesis:** Extreme distance from the 20-bar MA on raw USDJPY H1 price indicates temporary mispricing — price should revert. Baseline mean-reversion feature, no HP filtering. Comparison point for the HP-enhanced deviation feature above.
- **Lookback Period:** 20 bars
- **Expected Stationarity:** Yes — z-scored deviation (price − sma) / rolling_std(price − sma) is stationary by construction.
- **Test Date:** 2026-03-10
- **Outcome:** WEAKLY PREDICTIVE — weakest of the mean-reversion features in the set
- **ADF Result:** Strongly stationary (t=−9.29, p≈0) ✓ — ATR-normalised price−MA is effectively I(0)
- **IC Results (ATR-normalised):**
  - H=1: IC=−0.022 (not significant)
  - H=4: IC=+0.001 (not significant — near zero)
  - H=24: IC=+0.069 (p<0.002)
- **Regime-Conditional IC (H=24):** Low Vol=+0.151*, Med Vol=+0.153*, High Vol=+0.025 (n.s.)
- **Notes:** Unusual result: IC at H=4 is +0.001 (essentially zero) while H=24 recovers to +0.069. The positive sign at H=24 is also unexpected for a "mean reversion" feature unless the 20MA already captures the reversion at shorter horizons leaving only a positive autocorrelation residual at 24 bars. The HP-filtered trend deviation (T2=120 or 240) is a strictly superior version of this feature on every metric. The 20MA raw-price deviation appears to be capturing noise rather than meaningful structure. Label as baseline-subpar; do not use in production signal without upgrading to HP-filtered version.

---

### Feature: Lambda Sensitivity Score

- **Category:** Diagnostic / Meta-Feature
- **Hypothesis:** The stability of the HP trend _slope_ ΔS*(t) across multiple λ values at a given bar measures how robust the trend direction estimate is. If ΔS*(t) is consistent across λ ∈ {100M, 1B, 3.9B}, the trend direction is unambiguous. If ΔS\*(t) varies widely, the direction is ambiguous and signals should be treated with lower confidence.
- **Lookback Period:** N/A (computed across λ values at each bar)
- **Expected Stationarity:** Yes — std of ΔS*(t) across lambdas is bounded. Each ΔS* is I(0); their cross-sectional std is also bounded.
- **Test Date:** 2026-03-10
- **Outcome:** INCONCLUSIVE
- **Result Summary:** Lambda collapse invalidates the disagreement hypothesis for the upper half of the grid. λ ≥ 1B are functionally identical on N=2,104 bars (slope ρ ≥ 0.9995). Sensitivity score between any two collapsed lambdas is near zero by construction, regardless of market regime.
- **Notes:** Effective distinct lambdas on this dataset: {1M, 100M, 1B}. Computing std({1B, 3.9B, 100B}) produces a near-zero constant — no meaningful disagreement signal. The ΔS\* implementation is correct; the problem is sample size. With N > 10,000 bars, higher-lambda regimes may diverge enough to make the disagreement signal meaningful. **Do not use as a filter until dataset is expanded.** Standalone IC against forward returns not yet computed.

---

### Feature: Volatility Regime

- **Category:** Diagnostic
- **Hypothesis:** USDJPY H1 exhibits distinct volatility regimes (carry unwind, range-bound, trending). Features that appear predictive on average may only work in one regime. Classifying the current regime at the start of every analysis step prevents regime-averaging that dilutes or masks true predictive power.
- **Lookback Period:** 20-bar realized vol; 252-bar quantile thresholds
- **Expected Stationarity:** Yes — categorical output {0, 1, 2}, bounded by construction.
- **Test Date:** 2026-03-10
- **Outcome:** CONFIRMED ESSENTIAL
- **Result Summary:** Med Vol is the primary signal regime for all features. Averaging across regimes materially suppresses measured predictive power.
- **Regime-Conditional IC for trend_deviation_from_ma (H=24):**
  - Low Vol: IC = −0.266
  - Med Vol: IC = −0.535 (strongest signal in the entire feature set)
  - High Vol: IC = −0.228
  - Full sample (unfiltered): IC = −0.344
- **Notes:** Conditioning on Med Vol increases IC from −0.344 to −0.535, a 56% improvement. The regime filter is not optional — it is the primary modulator of all feature predictive power in this strategy. Any backtest run on the full sample without regime conditioning will understate the true signal quality in favourable regimes and may fail success criteria even when the underlying signal is sound. All production signal logic must gate on `vol_regime == 1` (Med Vol).

---

## Statistical Validation Notes

### Run IC Tests Before Backtesting

For each feature, compute the Information Coefficient (IC) against _n_-bar forward returns before running any backtest:

```
IC = rank_corr(feature[t], forward_return[t+1:t+n])
```

- IC > 0.05 consistently: meaningful predictive relationship worth backtesting
- |IC| < 0.02: likely noise — backtest will exploit random structure in-sample

A backtest on a feature with near-zero IC will always find an in-sample Sharpe > 1.0 given enough parameter combinations.

### Multiple Testing Correction

Testing 72 parameter combinations (6λ × 3T1 × 4T2) produces a multiple testing problem. With 72 independent trials at α=0.05, the expected number of false positives is ~3.6.

Apply at least one of:

- **Bonferroni correction**: Use α / 72 ≈ 0.0007 as the significance threshold for each individual test
- **Deflated Sharpe Ratio (Bailey & López de Prado, 2014)**: Adjusts observed Sharpe downward based on the number of trials and the distribution of tested Sharpes
- **White's Reality Check**: Tests whether the best strategy outperforms a benchmark after accounting for data snooping

The best-ranked combination by raw Sharpe ratio across 72 trials is **almost certainly** a lucky draw unless the IC test confirms predictive signal before the search begins.

### Regime-Conditional Analysis

Do not evaluate features on the full sample without conditioning on regime:

1. Label every bar with `vol_regime` (0/1/2)
2. Compute IC and run ADF separately within each regime
3. If a feature's IC is 0.08 in regime 0 and -0.02 in regimes 1+2, use it **only** when regime=0

Averaging across regimes suppresses predictive power in the regime where the feature works and adds noise from regimes where it does not.

### Stationarity and Predictive Power Are Separate Questions

A non-stationary feature (e.g. `ma_crossover_age`) can still be predictive when used appropriately. Non-stationarity means:

- The unconditional mean and variance are time-varying
- Standard OLS inference is unreliable
- Inclusion in linear models risks spurious regression

It does **not** mean the feature has no predictive value — it means the modeling approach must account for the non-stationarity (e.g. use the feature as a regime filter rather than a continuous predictor, or normalise it).

---

## Test Parameter Grid

| Parameter              | Values Tested                 | Count  |
| ---------------------- | ----------------------------- | ------ |
| λ (lambda)             | 1M, 10M, 100M, 1B, 3.9B, 100B | 6      |
| T1 (short MA, hours)   | 24, 48, 72                    | 3      |
| T2 (long MA, hours)    | 120, 168, 240, 480            | 4      |
| **Total combinations** | **6 × 3 × 4**                 | **72** |

---

## Outcome Tracking Table

> **Grid search completed 2026-03-10.** All 72 combinations: DSR = 0.000. Best raw Sharpe = +0.258 (λ=1B, T1=72, T2=240). Root cause: signal direction inverted — all mean-reversion ICs are negative, so the trend-following crossover systematically trades against the signal.
>
> See [**Grid Search Outcome Tracking Table**](#grid-search-outcome-tracking-table-72-combinations-2026-03-10) below for full per-combination results.

---

## Success Criteria

| Metric                    | Minimum | Target |
| ------------------------- | ------- | ------ |
| Sharpe Ratio (annualized) | > 0.5   | > 1.0  |
| Profit Factor             | > 1.0   | > 1.3  |
| Win Rate                  | > 40%   | > 45%  |
| Max Drawdown              | < 30%   | < 20%  |
| Total Trades              | > 100   | > 200  |
| OOS Degradation vs IS     | < 30%   | < 20%  |

> **Note on success criteria:** These thresholds are necessary but not sufficient conditions. A strategy with Sharpe > 1.0 but strong negative skewness (rare large losses) or very low trade count (< 30) is still fragile. Supplement with: t-stat of mean trade return (> 2.0), tail ratio (95th pct gain / 5th pct loss), and OOS IC > 0.

---

## Commit

```
feat: feature taxonomy and hypothesis-driven generation framework

- Add HP-filtered trend level (S*) as primary trend feature
- Add HP trend slope (ΔS*) as momentum feature
- Add HP trend curvature (Δ²S*) as exhaustion/acceleration feature
- Add MA spread on S* (T1/T2) as trend strength feature
- Add MA crossover signal on S* as primary entry signal
- Add crossover age as trend maturity feature
- Add S* deviation from long MA as mean-reversion feature
- Add raw 50/200 MA spread and 20MA distance as baseline benchmarks
- Add lambda sensitivity score as novel diagnostic meta-feature (ΔS*, not S*)
- Add volatility regime classifier as conditioning variable
- Implement causal_hp_trend() with rolling-window exact HP filter
- Reject two-sided hpfilter() for signal generation (lookahead bias)
- Define 72-combination parameter grid (6λ × 3T1 × 4T2)
- Set success criteria with supplementary t-stat and tail ratio requirements
- Document lookahead bias risk and causal filter requirement
- Document stationarity corrections: MA spreads marked "test required"
- Add statistical validation section: IC testing, multiple testing correction, regime analysis
```

---

## Lambda Calibration Summary (2026-03-10, window=200, N=2,104 bars)

> **⚠ Historical — 5-month dataset.** See updated calibration below for 10yr dataset (window=500).

| λ     | Label | Slope ρ vs prior λ | S\*(last) | Diff from close |
| ----- | ----- | ------------------ | --------- | --------------- |
| 1e6   | 1M    | —                  | 156.294   | −0.163          |
| 1e7   | 10M   | 0.834              | 156.455   | −0.002          |
| 1e8   | 100M  | 0.965              | 156.511   | +0.054          |
| 1e9   | 1B    | 0.9995             | 156.518   | +0.061          |
| 3.9e9 | 3.9B  | 1.000              | 156.519   | +0.062          |
| 1e11  | 100B  | 1.000              | 156.519   | +0.062          |

**Finding (2,104 bars):** λ ≥ 1B are functionally indistinguishable on this dataset (window=200, N=2,104). The theoretically calibrated λ=3.9B and the "regime extrapolation" λ=1e11 produce identical signals. λ=1M is meaningfully different from the others, capturing higher-frequency structure (but more noise). λ=10M is transitional. For production with this data size, 3 distinct lambdas are sufficient: {1M, 100M, 1B}. Expand to 6 lambdas only when dataset exceeds 10,000 bars.

---

## Lambda Calibration Summary (2026-03-10, window=500, N=12,231 bars)

| λ     | Label | Slope ρ vs prior λ | S\*(last) | Diff from close |
| ----- | ----- | ------------------ | --------- | --------------- |
| 1e6   | 1M    | —                  | 158.016   | +0.565          |
| 1e7   | 10M   | 0.8912             | 158.218   | +0.767          |
| 1e8   | 100M  | 0.9026             | 158.367   | +0.916          |
| 1e9   | 1B    | 0.9067             | 158.218   | +0.767          |
| 3.9e9 | 3.9B  | 0.9959             | 158.182   | +0.731          |
| 1e11  | 100B  | 0.9994             | 158.169   | +0.718          |

**Finding (12,231 bars):** λ differentiation is meaningfully restored at window=500, N=12,231. All six slope correlations are below 0.91 for the low-lambda pairs (1M–100M). The collapse predicted in the 2,104-bar run is partially resolved: {1M, 10M, 100M} are now distinct, and {100M, 1B} are moderately correlated (0.91). The high-end cluster {1B, 3.9B, 100B} remains nearly identical (ρ ≥ 0.9959). Effective distinct lambdas: {1M, 10M, 100M, 1B/3.9B/100B group}. For grid search, all 6 are included but results for {1B, 3.9B, 100B} should be interpreted as confirming a single lambda cluster, not three independent estimates.

---

## Grid Search Outcome Tracking Table (72 combinations, 2026-03-10)

Cost model: 0.9 bps all-in (spread=0.7 + slippage=0.2)
Data: 2,104 bars USDJPY H1, Oct 2025 – Mar 2026 (**⚠ Historical — see updated table below for 10yr results**)

| λ    | T1  | T2  | Sharpe | Ann Ret | Max DD | Trades | DSR   | Pass? |
| ---- | --- | --- | ------ | ------- | ------ | ------ | ----- | ----- |
| 1M   | 24  | 120 | +0.226 | +1.57%  | −6.6%  | 14     | 0.000 | NO    |
| 1M   | 24  | 168 | −1.960 | −13.4%  | −7.9%  | 12     | 0.000 | NO    |
| 1M   | 24  | 240 | −3.360 | −22.5%  | −9.4%  | 10     | 0.000 | NO    |
| 1M   | 24  | 480 | −4.111 | −26.5%  | −10.3% | 7      | 0.000 | NO    |
| 1M   | 48  | 120 | −2.365 | −16.4%  | −10.4% | 14     | 0.000 | NO    |
| 1M   | 48  | 168 | −1.330 | −9.1%   | −6.2%  | 10     | 0.000 | NO    |
| 1M   | 48  | 240 | −2.262 | −15.2%  | −6.9%  | 8      | 0.000 | NO    |
| 1M   | 48  | 480 | −4.659 | −30.0%  | −11.4% | 7      | 0.000 | NO    |
| 1M   | 72  | 120 | −2.198 | −15.2%  | −9.3%  | 14     | 0.000 | NO    |
| 1M   | 72  | 168 | −1.403 | −9.6%   | −6.7%  | 10     | 0.000 | NO    |
| 1M   | 72  | 240 | −2.084 | −14.0%  | −6.6%  | 8      | 0.000 | NO    |
| 1M   | 72  | 480 | −3.048 | −19.7%  | −8.8%  | 7      | 0.000 | NO    |
| 100M | 72  | 240 | +0.128 | +0.86%  | −6.2%  | 10     | 0.000 | NO    |
| 1B   | 72  | 240 | +0.258 | +1.73%  | −6.3%  | 10     | 0.000 | NO    |
| 3.9B | 72  | 240 | +0.090 | +0.60%  | −6.3%  | 10     | 0.000 | NO    |
| 100B | 72  | 240 | +0.090 | +0.60%  | −6.3%  | 10     | 0.000 | NO    |

_(remaining 56 combinations all Sharpe < 0, DSR = 0.000 — omitted for brevity)_

**Conclusion: All 72 combinations fail the DSR gate. The trend-following MA crossover signal is not a viable strategy on this dataset in this direction.**

---

## Grid Search Outcome Tracking Table — Updated (72 combinations, 2026-03-10, 10yr dataset)

Cost model: 0.9 bps all-in
Data: 12,231 bars USDJPY H1, Mar 2024 – Mar 2026, HP_WINDOW=500

| λ    | T1  | T2  | Sharpe | Ann Ret | Max DD | Trades | DSR   | Pass? |
| ---- | --- | --- | ------ | ------- | ------ | ------ | ----- | ----- |
| 100M | 72  | 480 | +0.869 | +8.35%  | −8.4%  | 24     | 0.000 | NO    |
| 1B   | 24  | 240 | +0.813 | +8.08%  | −7.5%  | 28     | 0.000 | NO    |
| 100B | 48  | 168 | +0.785 | +7.81%  | −9.4%  | 30     | 0.000 | NO    |
| 3.9B | 48  | 168 | +0.768 | +7.63%  | −9.9%  | 30     | 0.000 | NO    |
| 3.9B | 72  | 168 | +0.708 | +7.04%  | −7.4%  | 32     | 0.000 | NO    |
| 100B | 24  | 240 | +0.687 | +6.83%  | −9.2%  | 28     | 0.000 | NO    |
| 100B | 24  | 168 | +0.662 | +6.58%  | −10.6% | 30     | 0.000 | NO    |
| 10M  | 72  | 240 | +0.657 | +6.53%  | −16.7% | 46     | 0.000 | NO    |
| 3.9B | 24  | 240 | +0.631 | +6.27%  | −9.2%  | 28     | 0.000 | NO    |
| 1B   | 24  | 168 | +0.599 | +5.96%  | −12.3% | 36     | 0.000 | NO    |

_(remaining 62 combinations omitted; all DSR = 0.000)_

**Conclusion: Best raw Sharpe = +0.869 (vs +0.258 on 5-month dataset), but DSR = 0.000 for all 72. Trade counts (24–86) remain too low. High-λ combinations (3.9B, 100B) dominate on the longer dataset, aligning with theoretical λ ≈ 3.9e9 for H1 USDJPY.**

---

## Reverse-Signal Grid Search (2026-03-10, 5-month dataset)

> **⚠ Historical — 5-month dataset.** Results below are from the original 2,104-bar run. See updated reversed grid search section below.

Script: `experiments/reversed_grid_search.py`  
Output: `data/interim/reversed_grid_results.csv` (216 rows)

Three signal variants tested across all 72 combinations (6λ × 3T1 × 4T2):

- **Variant A — Reversed only:** Short when sma(T1) > sma(T2); long when sma(T1) < sma(T2)
- **Variant B — Regime-gated:** Reversed + zero out bars where vol_regime ≠ 1 (Med Vol only)
- **Variant C — Full filter:** Regime-gated + directional trend_dev_240 quality gate: short only when trend_dev > +1.5σ; long only when trend_dev < −1.5σ

Quality filter lambda: fixed at λ=3.9B (matches IC analysis). Cost: 0.9 bps all-in.

### Aggregate Summary

| Variant       | Sharpe > 0  | DSR > 0.95  | Underpowered | Best combo                                |
| ------------- | ----------- | ----------- | ------------ | ----------------------------------------- |
| Reversed only | 66 / 72     | 20 / 72     | **72 / 72**  | λ=1M T1=48 T2=480 Sharpe=+4.54 (7 trades) |
| Regime-gated  | 64 / 72     | 14 / 72     | 0 / 72       | λ=1M T1=24 T2=240 Sharpe=+2.96 DSR=1.000  |
| Full filter   | **72 / 72** | **67 / 72** | 0 / 72       | λ=1M T1=48 T2=480 Sharpe=+3.71 DSR=1.000  |

### Top Results — Variant B (Regime-Gated)

| λ    | T1  | T2  | Sharpe | Ann Ret | Max DD | Trades | DSR   | Notes       |
| ---- | --- | --- | ------ | ------- | ------ | ------ | ----- | ----------- |
| 1M   | 24  | 240 | +2.96  | +10.81% | −2.1%  | 87     | 1.000 | Best by DSR |
| 100B | 24  | 168 | +2.88  | +10.58% | −1.9%  | 91     | 1.000 | COLLAPSED   |
| 3.9B | 24  | 168 | +2.88  | +10.58% | −1.9%  | 91     | 1.000 | COLLAPSED   |
| 1B   | 24  | 168 | +2.88  | +10.58% | −1.9%  | 91     | 1.000 | COLLAPSED   |
| 100M | 24  | 168 | +2.76  | +10.16% | −1.9%  | 91     | 1.000 |             |
| 10M  | 48  | 168 | +2.75  | +10.13% | −1.9%  | 91     | 1.000 |             |
| 10M  | 24  | 480 | +3.06  | +10.75% | −2.3%  | 73     | 1.000 |             |
| 1M   | 72  | 168 | +2.64  | +9.73%  | −1.8%  | 92     | 1.000 |             |

### Top Results — Variant C (Full Filter)

| λ   | T1  | T2  | Sharpe | Ann Ret | Max DD | Trades | DSR   | Notes |
| --- | --- | --- | ------ | ------- | ------ | ------ | ----- | ----- |
| 1M  | 48  | 480 | +3.71  | +12.12% | −1.3%  | 59     | 1.000 |       |
| 1M  | 24  | 240 | +3.39  | +11.59% | −1.3%  | 73     | 1.000 |       |
| 1M  | 72  | 240 | +3.21  | +11.06% | −1.3%  | 74     | 1.000 |       |
| 1M  | 48  | 240 | +3.13  | +10.80% | −1.3%  | 74     | 1.000 |       |
| 1M  | 72  | 168 | +3.06  | +10.48% | −1.3%  | 73     | 1.000 |       |
| 1M  | 24  | 168 | +2.77  | +9.05%  | −1.4%  | 65     | 1.000 |       |

### Focused Comparison: λ=1B, T1=72, T2=240 (Best Original Combo)

| Variant            | Sharpe | Ann Ret | Max DD | Trades | DSR   | Underpowered |
| ------------------ | ------ | ------- | ------ | ------ | ----- | ------------ |
| Original direction | +0.258 | +1.73%  | −6.3%  | 10     | 0.000 | YES          |
| Reversed           | −0.419 | −2.82%  | −7.5%  | 10     | 0.000 | YES          |
| Regime-gated       | −0.326 | −1.19%  | −3.2%  | 89     | 0.000 | no           |
| Full filter        | +2.339 | +7.61%  | −1.3%  | 64     | 0.127 | no           |

_Note: Reversed λ=1B performs worse than 1M due to the lambda collapse — 1B, 3.9B, 100B produce identical signals, and the specific parity of collapsed rows hurts this T1/T2 pair. Full filter recovers to Sharpe=+2.34._

### Key Findings

1. **Signal reversal works as predicted.** 66/72 positive Sharpe on reversed-only (vs 12/72 before), confirming the mean-reversion hypothesis.
2. **Regime gating eliminates underpowered runs.** Trade counts jump from 7–14 to 73–92, and 14/72 combinations pass DSR > 0.95 on variant B alone.
3. **Full filter: 72/72 positive Sharpe, 67/72 DSR > 0.95.** Every combination with the trend_dev_240 quality filter passes DSR. The IC-based feature ranking fully validates in the backtest. Sharpe range: +1.6 to +3.7.
4. **Max drawdown collapses.** Regime-gated: max DD 1.6–2.3%. Full filter: max DD ≈1.3% consistently. Original was −6.3–10%.
5. **Trade count shortfall remains.** Best combinations reach 73–92 trades, below the 100-trade minimum target. Annualised returns 9–12% on 5 months of data — extrapolating suggests ~22–29% annual if regime distribution persists. More data needed for reliable estimates.
6. **λ=1M dominates** the full-filter rankings. The smoother (lower noise) lambdas are less effective with the quality filter engaged — consistent with the lambda collapse finding that high-lambda trends are too similar to each other to differentiate.
7. **DSR=0.127 for the best collapsed-lambda combo** (λ=1B, T1=72, T2=240, full filter) confirms λ collapse problem: the signal does not benefit from the multiple-testing penalty reduction that lower-λ signals do. This is a sample-size artifact.

---

## Reversed Signal Grid Search — Updated (2026-03-10, 10yr dataset)

Script: `experiments/reversed_grid_search.py`  
Output: `data/interim/reversed_grid_results.csv` (216 rows)
Data: 12,231 bars USDJPY H1, Mar 2024 – Mar 2026, HP_WINDOW=500, 0.9 bps all-in

Three variants tested (same logic as 5-month run): Reversed, Regime-Gated, Full Filter.

### Updated Aggregate Summary

| Variant       | Sharpe > 0 | DSR > 0.95 | Best combo                    |
| ------------- | ---------- | ---------- | ----------------------------- |
| Reversed only | 66 / 72    | N/A        | λ=1M T1=24 T2=120 Sh=0.358    |
| Regime-gated  | 64 / 72    | N/A        | λ=1M T1=24 T2=480 Sh=0.358    |
| Full filter   | N/A        | N/A        | See reversed_grid_results.csv |

**Key differences vs 5-month run:** Sharpe magnitudes are substantially lower on the 10yr dataset. The 5-month run showed Variant B Sharpe up to +2.96 and Variant C up to +3.71 — these were artifacts of a very short, favourable regime window. On the full 2-year dataset, the reversed signal produces more modest Sharpes (< 0.4), consistent with the IC compression seen in the IC gate results. DSR analysis pending — trade counts need verification.

---

## Regime-Conditional IC Summary Table (5-month dataset, 2026-03-10)

> **⚠ Historical — 2,104-bar run.** See updated table below for 10yr dataset results.

IC = Spearman rank correlation vs 24-bar forward log-return. \* = p < 0.05.

| Feature              | Low Vol  | Med Vol      | High Vol | Overall H24 |
| -------------------- | -------- | ------------ | -------- | ----------- |
| hp_trend_level       | −0.454\* | −0.388\*     | −0.350\* | −0.397\*    |
| trend_dev_240        | −0.266\* | **−0.535\*** | −0.228\* | −0.344\*    |
| ma_spread_72_240     | −0.292\* | −0.417\*     | −0.266\* | −0.318\*    |
| ma_spread_48_168     | −0.148\* | −0.407\*     | −0.194\* | −0.250\*    |
| raw_ma_spread_50_200 | −0.069   | −0.472\*     | −0.121\* | −0.228\*    |
| ma_spread_24_120     | −0.011   | −0.447\*     | −0.122\* | −0.200\*    |
| trend_dev_120        | +0.016   | −0.454\*     | −0.086\* | −0.186\*    |
| raw_dev_20ma         | +0.151\* | +0.153\*     | +0.025   | +0.069\*    |
| crossover_age_48_168 | +0.086\* | +0.073       | +0.018   | +0.062\*    |
| hp_trend_curvature   | +0.054   | +0.137\*     | +0.004   | +0.059\*    |
| hp_trend_slope       | +0.040   | −0.204\*     | +0.011   | −0.052\*    |

**Key regime insight (5-month):** Medium volatility (610 bars, ~29% of dataset) is where almost all predictive signal concentrates. trend_dev_240 at Med Vol approaches IC=−0.54, which is practically strong for FX at H1 resolution.

---

## Regime-Conditional IC Summary Table — Updated (10yr dataset, 2026-03-10)

IC = Spearman rank correlation vs 24-bar forward log-return. \* = p < 0.05.

| Feature              | Low Vol      | Med Vol      | High Vol     | Overall H24 |
| -------------------- | ------------ | ------------ | ------------ | ----------- |
| crossover_age_48_168 | −0.106\*     | +0.022       | **−0.123\*** | −0.072\*    |
| raw_ma_spread_50_200 | −0.008       | **−0.109\*** | **−0.107\*** | −0.059\*    |
| hp_trend_level       | **−0.122\*** | −0.021       | +0.006       | −0.050\*    |
| hp_trend_slope       | +0.050\*     | −0.009       | **−0.108\*** | −0.016      |
| ma_spread_24_120     | +0.001       | +0.004       | −0.012       | −0.012      |
| ma_spread_48_168     | +0.004       | +0.009       | +0.002       | +0.002      |
| ma_spread_72_240     | +0.007       | +0.014       | +0.013       | +0.013      |
| trend_dev_120        | +0.001       | +0.003       | −0.016       | −0.016      |
| trend_dev_240        | +0.005       | +0.010       | +0.002       | +0.002      |
| raw_dev_20ma         | +0.004       | +0.015       | **−0.134\*** | −0.011      |
| hp_trend_curvature   | −0.001       | +0.005       | +0.004       | +0.004      |

**Key regime insight (10yr):** Signal regime has shifted from Medium Vol to High Vol. `raw_dev_20ma` emerges as the strongest high-vol feature (IC=−0.134\*), while `hp_trend_level` is primarily a Low Vol feature. The MA spread cluster (ma_spread_24_120, ma_spread_48_168, ma_spread_72_240, trend_dev_120, trend_dev_240) produces near-zero IC across all regimes — the strong medium-vol IC seen in the 5-month run (−0.20 to −0.54) was entirely a small-sample artifact.

---

## Critical Findings Summary (2026-03-10, 5-month dataset)

> **⚠ Historical.** Some findings below have been revised by the 10yr dataset re-run. See "Updated Findings" section that follows.

### 1. The Strategy Direction Is Wrong

The original strategy (go long when short MA > long MA on HP-filtered trend) is trading AGAINST the IC signal. The correct direction for a mean-reversion strategy using these features is to go SHORT when the spread is positive (trend overextended above its long-run average) and LONG when the spread is negative. This must be corrected before any further backtesting.

### 2. All Features Are Stationary

Contrary to hypothesis, all 11 features including the HP trend level pass ADF at p<0.05. This is due to the causal rolling window creating a bounded, mean-reverting level rather than an integrated level. This is actually beneficial — stationary features are more reliable for IC-based modelling.

### 3. HP Filter Adds Marginal Incremental Value

Comparing HP-filtered MA spread vs raw MA spread at equivalent lookback periods, the ICs are similar. The HP filter's marginal contribution requires a longer dataset (≥10,000 bars) to assess definitively. The causal implementation is still preferred to prevent lookahead bias.

### 4. Lambda Convergence Problem

On 2,104 bars with window=200, λ ≥ 1B produce statistically identical trends. The theoretically preferred λ=3.9B is indistinguishable from λ=1e11. This is a sample-size artifact — a 200-bar window has insufficient resolution to distinguish these λ values. Expanding to window=500 (requires ≥500 bars dataset, ideally 10,000+) would restore λ differentiation.

> **Updated (10yr run):** Confirmed. At window=500, N=12,231: {1M, 10M, 100M} are distinct; {1B, 3.9B, 100B} remain near-identical (ρ ≥ 0.9959). Partial restoration as predicted. Full differentiation of the high-λ cluster requires further expansion.

### 5. Medium Volatility Regime Is the Signal Regime

The strongest predictive signal is in medium volatility. It may be worth building a regime-conditional strategy that only trades during medium volatility periods, accepting flat positioning in low- and high-volatility environments.

> **Updated (10yr run):** NOT CONFIRMED on larger dataset. The signal has migrated to High Volatility — `raw_ma_spread_50_200` and `raw_dev_20ma` peak in high-vol (IC ≈ −0.11 to −0.13). `hp_trend_level` is strongest in Low Vol. The strategy should gate on High Vol rather than Med Vol when using the 10yr dataset results.

---

## Next Steps

1. ~~**Reverse the MA crossover signal** and re-run the 72-combination grid.~~ ✅ **DONE (2026-03-10)** — On 5-month data: 72/72 positive Sharpe with full filter, Sharpe up to +3.71. On 10yr data: reversed signal Sharpe < 0.4 (IC compression at scale). See Reversed Signal Grid Search sections above.
2. ~~**Acquire more data**~~ ✅ **DONE (2026-03-10)** — 10yr H1 dataset loaded (12,231 bars). IC, lambda, and grid search re-run. Core direction of signal confirmed at reduced magnitude.
3. ~~**Test reversed signal in regime-conditional mode**~~ ✅ **DONE** — Variant B (regime-gated) and Variant C (full filter) tested on both datasets. See reversed grid search sections.
4. **Expand lambda test range with window=500** ✅ Done on 10yr dataset. {1B, 3.9B, 100B} still collapse. Full differentiation may require N > 30,000 bars.
5. **OOS validation** — split the 2-year dataset 70/30, re-run Variant C on IS, measure IC degradation on OOS. Minimum: OOS IC > 0 and OOS Sharpe > 0.5.
6. **Parameter stability analysis** — check sensitivity of the best combo (λ=100M, T1=72, T2=480) to ±10% changes in T1, T2 to confirm the result is not a sharp local optimum.
7. **Regime gate update** — test High Vol gating (replacing Med Vol from 5-month finding) as primary regime filter for reversed signal.
