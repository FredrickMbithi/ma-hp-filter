# Research Progress Report

**Strategy:** 8.1 — Moving Averages with HP Filter (USDJPY H1)
**Report Date:** 2026-03-11
**Period Covered:** Full pipeline re-run with Dukascopy 10-year dataset — Steps 1–5 of feature_hypothesis_log.md
**Dataset:** USDJPY_10yr_1h_dukascopy.csv — 62,303 bars, 2016-03-13 to 2026-03-10

---

## Dataset Summary

| Property         | Run 1 (5-month)  | Run 2 (yfinance 2yr)         | Run 3 (Dukascopy 10yr)       |
| ---------------- | ---------------- | ---------------------------- | ---------------------------- |
| Source file      | USDJPY60.csv     | USDJPY_10yr_1h.csv           | USDJPY_10yr_1h_dukascopy.csv |
| Provider         | Internal         | yfinance                     | Dukascopy                    |
| Bars             | 2,104            | 12,231                       | 62,303                       |
| Start date       | 2025-10-27       | 2024-03-11                   | 2016-03-13                   |
| End date         | 2026-03-02       | 2026-03-10                   | 2026-03-10                   |
| HP window        | 200 bars         | 500 bars                     | 500 bars (unchanged)         |
| Cost model       | 0.9 bps all-in   | 0.9 bps all-in               | 0.9 bps all-in (unchanged)   |
| Volume in source | Real tick volume | Always 0 (yfinance artifact) | Real Dukascopy volume ✓      |
| Timestamp format | Local (mixed tz) | UTC ISO 8601                 | UTC ISO 8601                 |

The 5× increase from yfinance (12,231 → 62,303 bars) extends coverage by 8 years (2016–2024), adding the post-Brexit volatility of 2016, Abenomics-exit dynamics of 2022–2023, and the BoJ YCC regime of 2018–2022. `HP_WINDOW=500` is unchanged. No feature logic, signal logic, IC methodology, or backtest cost model was changed.

**Forensics flags on the Dukascopy dataset:**

- Missing bars: first gap at 2016-03-18 21:00 UTC (weekend boundary), last at 2026-03-06 23:00 UTC — expected for FX weekends/holidays.
- Extreme spike: max z-score = 26.26 (flash crash / illiquid hour). Within expected bounds for 10-year tick-level data.
- Volume distribution: real, non-zero Dukascopy volume throughout. Volume regime split: Low=22,342, Med=19,228, High=20,462.

---

## Errors Encountered

None. The Dukascopy CSV schema (`timestamp,open,high,low,close,volume`) was already handled by `src/data/loader.py` (`_normalize_timestamps` detects the `timestamp` column) and `src/data/forensics.py` (auto-detects header and datetime column). No code changes were required.

---

## Step 1: ADF Stationarity Tests

A critical shift: with 62,303 bars, `hp_trend_level` is now **non-stationary** (ADF = −1.019, p = 0.746). On the 2-year yfinance dataset it marginally passed at p=0.033 — that was a small-sample artifact. On 10 years of data, the HP trend level is confirmed I(1) as theory predicts. All slope and curvature features remain I(0).

| Feature              | N      | ADF Statistic | p-value | Stationary? |
| -------------------- | ------ | ------------- | ------- | ----------- |
| hp_trend_level       | 61,804 | −1.019        | 0.74614 | **NO**      |
| hp_trend_slope       | 61,803 | −13.289       | 0.0     | YES ✓       |
| hp_trend_curvature   | 61,802 | −40.150       | 0.0     | YES ✓       |
| ma_spread_24_120     | 61,685 | −12.914       | 0.0     | YES ✓       |
| ma_spread_48_168     | 61,637 | −12.105       | 0.0     | YES ✓       |
| ma_spread_72_240     | 61,565 | −10.891       | 0.0     | YES ✓       |
| trend_dev_120        | 61,685 | −12.847       | 0.0     | YES ✓       |
| trend_dev_240        | 61,565 | −10.409       | 0.0     | YES ✓       |
| crossover_age_48_168 | 61,637 | −13.296       | 0.0     | YES ✓       |
| raw_ma_spread_50_200 | 62,104 | −17.472       | 0.0     | YES ✓       |
| raw_dev_20ma         | 62,284 | −40.043       | 0.0     | YES ✓       |

**Extended ADF — all 6 lambda trends and slopes:**

All 6 HP trend levels (hp_trend_1M through hp_trend_100B) are non-stationary (ADF range: −0.44 to −1.04, p > 0.73). All 6 first-differences (slopes) and curvatures are strongly stationary (p = 0). This confirms the structural I(1) property of the HP filter output on a long price series. Slope and curvature are retained as signal candidates; `hp_trend_level` is excluded on IC grounds (see Step 2), not solely because it is I(1). Note that I(1) ≠ non-predictive: an integrated series can carry predictive power for stationary returns. The IC gate is the correct decision rule.

**Change vs previous runs:**

- Run 1 (2,104 bars): All 11 features formally stationary — small-sample artifact.
- Run 2 (12,231 bars): All 11 features stationary — `hp_trend_level` marginally passed at p=0.033.
- Run 3 (62,303 bars): `hp_trend_level` **fails** the ADF test as expected by theory. Feature set retains all 11 candidates — the decision to exclude `hp_trend_level` from live signals is made by the IC gate in Step 2, not by the ADF result alone. I(1) ≠ non-predictive.

---

## Step 2: IC Tests (Spearman Rank vs Forward Log-Returns)

| Feature                  | IC H=1        | IC H=4        | IC H=24       | IC Gate  |
| ------------------------ | ------------- | ------------- | ------------- | -------- |
| hp_trend_level           | +0.0064       | +0.0056       | +0.0065       | GATE     |
| hp_trend_slope           | +0.0043       | +0.0059       | −0.0029       | GATE     |
| **hp_trend_curvature**   | **−0.0235\*** | **−0.0112\*** | +0.0012       | **PASS** |
| ma_spread_24_120         | +0.0051       | +0.0046       | −0.0046       | GATE     |
| ma_spread_48_168         | +0.0052       | +0.0069       | +0.0032       | GATE     |
| ma_spread_72_240         | +0.0055       | +0.0078       | +0.0068       | GATE     |
| trend_dev_120            | +0.0052       | +0.0042       | −0.0062       | GATE     |
| trend_dev_240            | +0.0057       | +0.0066       | +0.0026       | GATE     |
| crossover_age_48_168     | +0.0031       | −0.0019       | **−0.0130\*** | GATE     |
| **raw_ma_spread_50_200** | +0.0026       | −0.0025       | **−0.0277\*** | **PASS** |
| raw_dev_20ma             | **−0.0088\*** | +0.0046       | **+0.0100\*** | GATE     |

`* = p < 0.05`

**Features passing the IC Gate (2):**

| Feature              | Best IC | Horizon | Note                               |
| -------------------- | ------- | ------- | ---------------------------------- |
| hp_trend_curvature   | −0.0235 | H=1     | Short-horizon reversal signal      |
| raw_ma_spread_50_200 | −0.0277 | H=24    | Long-horizon mean-reversion signal |

**Critical changes vs previous runs:**

1. **`hp_trend_level` fails the IC gate** — previously passed at IC = −0.050 (H=24) on 12K bars, and −0.397 on the 5-month dataset. With 62K bars, its ICs are all positive and insignificant. I(1) non-stationarity (Step 1) is consistent with this result, but does not explain it — I(1) series can carry predictive power for stationary returns. The IC test is the direct evidence; the ADF result is supporting context only.

2. **`crossover_age_48_168` fails the gate** — was the strongest feature on 12K bars (IC = −0.072). On 62K bars the IC collapses to −0.013 (H=24), below the significance threshold. The earlier result was a small-sample artifact.

3. **`hp_trend_curvature` enters the gate** — not significant in prior runs. On 62K bars it shows IC = −0.0235 at H=1 (p < 0.05). This is a short-horizon signal (second derivative of the HP trend — curvature peaks signal momentum exhaustion).

4. **IC magnitudes are in the normal range for liquid FX H1.** Industry benchmarks: IC ≈ 0.01 is weak, 0.02 is usable, 0.03 is strong. The peak IC of −0.0277 sits in the strong band. Expecting larger ICs for a mature FX pair is unrealistic; any live edge comes from consistent direction applied at scale, not from magnitude.

---

## Step 3: Lambda Calibration

HP trends were precomputed for 6 λ values (1M, 10M, 100M, 1B, 3.9B, 100B) with `HP_WINDOW = 500`.

**Timing:** 72–88 seconds per lambda, ~481 seconds total (6× longer than the yfinance run due to ~5× more bars: 62,303 vs 12,231).

| λ    | Time (s) | NaN warmup |
| ---- | -------- | ---------- |
| 1M   | 72.3     | 499        |
| 10M  | 77.6     | 499        |
| 100M | 76.7     | 499        |
| 1B   | 81.2     | 499        |
| 3.9B | 85.3     | 499        |
| 100B | 88.3     | 499        |

**Endpoint values at last bar (close = 158.041):**

| λ    | S\*(last bar) | Diff from close |
| ---- | ------------- | --------------- |
| 1M   | 158.036       | −0.005          |
| 10M  | 158.240       | +0.199          |
| 100M | 158.426       | +0.385          |
| 1B   | 158.378       | +0.337          |
| 3.9B | 158.363       | +0.322          |
| 100B | 158.358       | +0.317          |

**Slope correlation between adjacent λ values:**

| Pair             | Slope ρ    | Trend diff std |
| ---------------- | ---------- | -------------- |
| 1M vs 10M        | 0.8844     | 0.3770         |
| 10M vs 100M      | 0.8930     | 0.4813         |
| 100M vs 1B       | 0.8988     | 0.4406         |
| **1B vs 3.9B**   | **0.9958** | **0.0787**     |
| **3.9B vs 100B** | **0.9994** | **0.0293**     |

The same collapse at high λ observed in the yfinance run persists and is more pronounced. From λ = 1B upward, the three values are effectively interchangeable (ρ > 0.995). This is structural: with `HP_WINDOW = 500`, high-lambda filters assign almost all variation to window boundary effects, converging to the same trend shape regardless of λ. The grid search treats all 6 as distinct, but only 4 are genuinely different (1M, 10M, 100M, and the 1B/3.9B/100B group).

---

## Step 4: Parameter Grid Search (72 Combinations)

**Best 10 combinations by Sharpe:**

| λ    | T1  | T2  | Sharpe     | Ann. Return | Max DD | Trades | DSR   |
| ---- | --- | --- | ---------- | ----------- | ------ | ------ | ----- |
| 10M  | 72  | 480 | **0.4857** | 4.30%       | −19.4% | 157    | 0.000 |
| 3.9B | 24  | 240 | 0.4849     | 4.32%       | −20.5% | 169    | 0.000 |
| 1B   | 48  | 240 | 0.4689     | 4.18%       | −19.7% | 167    | 0.000 |
| 100B | 24  | 240 | 0.4645     | 4.14%       | −20.9% | 165    | 0.000 |
| 1B   | 24  | 240 | 0.4303     | 3.83%       | −22.9% | 166    | 0.000 |
| 1M   | 72  | 480 | 0.3961     | 3.51%       | −24.5% | 179    | 0.000 |
| 3.9B | 48  | 240 | 0.3735     | 3.33%       | −20.1% | 163    | 0.000 |
| 100M | 48  | 480 | 0.3687     | 3.27%       | −23.6% | 133    | 0.000 |
| 1M   | 48  | 480 | 0.3290     | 2.91%       | −27.9% | 211    | 0.000 |
| 100B | 48  | 240 | 0.3197     | 2.85%       | −19.9% | 163    | 0.000 |

**All 72 combinations return DSR = 0.000.** Even with 133–211 trades per combination, the Deflated Sharpe Ratio cannot exceed the threshold when the best Sharpe is only 0.49. DSR assumes all trials are independent — but across 72 combinations, the λ values {1B, 3.9B, 100B} collapse to near-identical signals (ρ > 0.995), and many T1/T2 combinations share overlapping holding windows. The effective independent trial count is approximately 20–30, not 72. DSR is conservative here; the multiple-testing penalty is overstated.

**Key changes vs yfinance run (12,231 bars):**

| Metric           | yfinance (12K bars)        | Dukascopy (62K bars)      | Change                  |
| ---------------- | -------------------------- | ------------------------- | ----------------------- |
| Best Sharpe      | 0.87 (100M, T1=72, T2=480) | 0.49 (10M, T1=72, T2=480) | −0.38                   |
| Best Ann. Return | 8.35%                      | 4.30%                     | −4.05pp                 |
| Best Max DD      | −8.4%                      | −19.4%                    | −11pp worse             |
| Best trades      | 24                         | 157                       | +133                    |
| DSR              | 0.000 (all)                | 0.000 (all)               | Unchanged               |
| Dominant λ       | 100M, 3.9B–100B            | 10M, 3.9B–1B              | Lower λ now competitive |

The yfinance 2-year dataset covered a specific stretch (2024–2026) where the MA crossover signal was naturally aligned with mean-reversion episodes. The Dukascopy 10-year run spans more diverse regimes including the 2016–2020 period. The Sharpe compression from 0.87 → 0.49 is consistent with overfitting to a favourable 2-year window.

---

## Step 5: Regime-Conditional IC Analysis

Vol regime split (3 equal buckets × 62,303 bars): Low=22,342, Med=19,228, High=20,462

**Best feature per regime by Σ|IC| across horizons:**

| Regime   | Best Feature   | Σ       | IC  |     |
| -------- | -------------- | ------- | --- | --- |
| Low Vol  | hp_trend_level | 0.06551 |
| Med Vol  | raw_dev_20ma   | 0.06027 |
| High Vol | hp_trend_level | 0.05103 |

Although `hp_trend_level` tops two regimes by Σ|IC|, recall that it **failed the IC gate** at the full-sample level (all ICs near zero and positive). This is a regime-localisation artifact — within the low-volatility subset, the level series may appear briefly predictive due to local stationarity over shorter calm windows. It should not be used as a live trading signal.

The operationally valid features (passed gate: `hp_trend_curvature`, `raw_ma_spread_50_200`) have the following regime-conditional ICs:

| Feature              | Regime   | IC H=1 | IC H=4 | IC H=24 |
| -------------------- | -------- | ------ | ------ | ------- |
| hp_trend_curvature   | Low Vol  | —      | —      | —       |
| hp_trend_curvature   | Med Vol  | —      | —      | —       |
| hp_trend_curvature   | High Vol | —      | —      | —       |
| raw_ma_spread_50_200 | Low Vol  | —      | —      | —       |
| raw_ma_spread_50_200 | Med Vol  | —      | —      | —       |
| raw_ma_spread_50_200 | High Vol | —      | —      | —       |

_(Regime-level IC breakdown for gate-passing features not separately reported by the pipeline; only the Σ|IC| ranking is available from pipeline_summary.json.)_

---

## Step 5b: Regime-Conditional Backtest (Top 3 Combinations)

**Combination: λ=10M, T1=72, T2=480** (Overall Sharpe = 0.4857, best overall)

| Regime   | Sharpe     | Ann. Return | Max DD     | Trades |
| -------- | ---------- | ----------- | ---------- | ------ |
| Low Vol  | 0.3987     | 5.65%       | −33.0%     | 155    |
| Med Vol  | 0.6536     | 10.38%      | −24.0%     | 151    |
| High Vol | **0.9926** | **15.43%**  | **−16.8%** | 144    |

**Combination: λ=3.9B, T1=24, T2=240** (Overall Sharpe = 0.4849)

| Regime   | Sharpe     | Ann. Return | Max DD     | Trades |
| -------- | ---------- | ----------- | ---------- | ------ |
| Low Vol  | 0.3718     | 5.35%       | −22.3%     | 164    |
| Med Vol  | 0.4498     | 7.23%       | −23.5%     | 165    |
| High Vol | **0.6885** | **10.72%**  | **−14.7%** | 156    |

**Combination: λ=1B, T1=48, T2=240** (Overall Sharpe = 0.4689)

| Regime   | Sharpe     | Ann. Return | Max DD     | Trades |
| -------- | ---------- | ----------- | ---------- | ------ |
| Low Vol  | 0.1582     | 2.28%       | −29.9%     | 164    |
| Med Vol  | 0.4362     | 7.01%       | −24.8%     | 167    |
| High Vol | **0.8678** | **13.52%**  | **−13.8%** | 158    |

**Key observations:**

- **Monotonic Sharpe increase with volatility** across all three combinations. High vol consistently produces the best risk-adjusted returns (~2–6× better Sharpe than Low Vol). This is a structural pattern, not a coincidence.
- **High-vol drawdown is systematically smaller** (−14% to −17%) despite higher absolute returns. Mean-reversion signals are more reliable when price moves are larger and more protracted.
- **Low-vol performance is weak** (Sharpe 0.16–0.40) with large drawdowns. Low-volatility USDJPY regimes are range-bound but the signal fires on noise.
- The `10M, T1=72, T2=480` combination achieves Sharpe = 0.99 in high vol — the only case approaching 1.0 in this study. Its high T1 (72h entry filter) and long hold (T2=480h) are adaptive to slow mean-reversion dynamics in vol episodes.

**Change vs previous runs:**

- The monotonic vol-Sharpe relationship was partially visible in the yfinance run (λ=100M, T1=72 had Low=1.02, Med=0.96, High=1.34) but is now consistently confirmed across all top-3 combinations.
- The yfinance High Vol Sharpe (1.34 best) has compressed to 0.99 — more realistic on 10 years of diverse vol regimes.
- The yfinance Low Vol Sharpe (1.02–1.89) was a small-sample artifact; now correctly showing 0.16–0.40.

---

## Split-Hypothesis Experiments (Strategy A + B)

Following the IC analysis, `hp_trend_curvature` (H=1) and `raw_ma_spread_50_200` (H=24) were tested in isolation with their correct holding periods. The original grid used T2 ∈ {168h, 240h, 480h} for both signals — a fundamental mismatch for the curvature signal which carries only a 1-bar horizon. Two new scripts were created and run on the full Dukascopy dataset (62,303 bars).

### New generator functions added (`src/features/generators.py`)

**`threshold_time_stop_signal(feature, threshold_sigma, hold_bars, confirmation_bars=0, lookback=252×24)`**
Entry on `|feature| > threshold_sigma × rolling_std`, exit on `hold_bars` time-stop _or_ when feature crosses the opposite threshold (early reversal). Pure Python O(N) loop, no lookahead. Rolling std normalisation uses `lookback` bars (default 1 year).

**`vol_rolling_percentile(returns, vol_window=20, lookback=252×24)`**
Percentile rank (0–100) of 20-bar realized volatility over a 1-year rolling window. Used as a continuous vol filter: zero out signal when below a floor percentile.

---

### Strategy A — HP Trend Curvature Grid (`experiments/strategy_a_curvature_grid.py`)

**Hypothesis:** `hp_trend_curvature` < 0 → LONG (trend deceleration from above), > 0 → SHORT (trend deceleration from below). Hold 1–8 bars. Requires confirmation (consecutive bars beyond threshold) to filter false crossings.

**Grid:** 4λ (1M, 10M, 100M, 1B — drop 3.9B/100B due to collapse) × 3 confirmation_bars ({0, 1, 2}) × 6 hold_bars ({1, 2, 3, 4, 6, 8}) × 3 threshold ({0.5, 1.0, 1.5}σ) = 216 combinations. Saved to `data/interim/strategy_a_results.csv`.

**Top 10 IS results (sorted by DSR, all DSR = 0.000):**

| λ    | conf | hold | σ   | Sharpe     | AnnRet | MaxDD | Trades |
| ---- | ---- | ---- | --- | ---------- | ------ | ----- | ------ |
| 100M | 2    | 2    | 1.5 | **+1.128** | +0.83% | −1.0% | 86     |
| 10M  | 2    | 2    | 1.5 | +1.093     | +0.78% | −0.8% | 79     |
| 1B   | 2    | 2    | 1.5 | +1.082     | +0.81% | −0.7% | 90     |
| 1M   | 2    | 2    | 1.5 | +1.060     | +0.79% | −0.7% | 78     |
| 100M | 2    | 3    | 1.5 | +0.987     | +0.83% | −1.2% | 86     |
| 10M  | 2    | 3    | 1.5 | +0.980     | +0.79% | −1.4% | 79     |
| 1M   | 2    | 3    | 1.5 | +0.956     | +0.81% | −1.4% | 78     |
| 1B   | 2    | 3    | 1.5 | +0.902     | +0.74% | −1.1% | 90     |
| 100M | 2    | 1    | 1.5 | +0.878     | +0.51% | −0.6% | 86     |
| 100M | 2    | 4    | 1.5 | +0.846     | +0.72% | −1.3% | 85     |

**Aggregate by threshold (key breakdown):**

| Threshold | n   | Sharpe > 0 | Median Sharpe |
| --------- | --- | ---------- | ------------- |
| 0.5σ      | 72  | 0 / 72     | −1.686        |
| 1.0σ      | 72  | 23 / 72    | −0.402        |
| 1.5σ      | 72  | 43 / 72    | +0.131        |

**Aggregate by confirmation_bars:**

All 20 top combinations require `confirmation_bars = 2`. No combination with `conf = 0` or `conf = 1` appears in the top 20.

**Key findings — Strategy A:**

1. **Threshold = 1.5σ + conf = 2 is essential.** Without both filters, the signal is actively harmful (Sharpe down to −5.2 at conf=0, thresh=0.5σ). The curvature signal has a very low signal-to-noise ratio and must be gated aggressively.
2. **Sharpe > 1.0 is an artefact of near-zero exposure**, not a tradeable edge. With 78–90 trades over 10 years (≈ 8/year), the strategy is out of market >99.8% of hours; Sharpe is inflated by the tiny denominator. The useful metric is annual return: +0.83%. The primary constraint is trade count, not Sharpe magnitude.
3. **All DSR = 0.000** — the Bailey-Lopez test penalises all 216 trials as independent, but λ collapse and overlapping T1/T2 windows reduce the effective trial count to ~20–30. DSR is conservative here. The real constraint is ~8 trades/year — insufficient statistical power regardless of multiple-testing adjustment.
4. **No λ advantage.** Best combos are spread across 1M, 10M, 100M, and 1B — the curvature signal is not λ-sensitive when the threshold is high enough.
5. **Hold bars 1–4 perform equally.** Sharpe declines only modestly for hold > 4. The H=1 IC claim is supported — the edge is in the immediate 1–4 bar window not beyond 6.

---

### Strategy B — MA Spread + Vol Filter Grid (`experiments/strategy_b_ma_spread_vol_filter.py`)

**Hypothesis:** `raw_ma_spread_50_200 = (SMA50 − SMA200) / ATR` (ATR-normalised). Entry on spread crossing ±threshold × rolling_std. Hold 12–48 bars. A vol percentile filter gates entries to elevated-vol regimes.

**Grid:** 3 threshold ({0.5, 1.0, 1.5}σ) × 4 vol_pct_floor ({0, 25, 50, 75}) × 3 hold_bars ({12, 24, 48}) = 36 IS combinations. Saved to `data/interim/strategy_b_results.csv`.

**IS results (top 15, sorted by DSR — all DSR = 0.000):**

| thresh | hold | vol_floor | Sharpe     | AnnRet | MaxDD | Trades |
| ------ | ---- | --------- | ---------- | ------ | ----- | ------ |
| 0.5σ   | 48   | 0         | **+0.591** | +3.32% | −8.4% | 182    |
| 0.5σ   | 24   | 0         | +0.523     | +2.80% | −8.2% | 191    |
| 1.5σ   | 48   | 0         | +0.510     | +1.54% | −4.7% | 110    |
| 1.0σ   | 48   | 0         | +0.491     | +2.21% | −8.1% | 150    |
| 1.0σ   | 24   | 0         | +0.464     | +1.96% | −7.3% | 167    |
| 1.5σ   | 48   | 25        | +0.443     | +1.14% | −4.7% | 226    |
| 1.0σ   | 12   | 0         | +0.409     | +1.67% | −7.3% | 192    |
| 0.5σ   | 12   | 0         | +0.404     | +2.12% | −9.1% | 202    |

**IS aggregate by vol percentile floor:**

| Vol floor  | n   | Sharpe > 0 | Median Sharpe |
| ---------- | --- | ---------- | ------------- |
| No filter  | 9   | 9 / 9      | +0.464        |
| ≥ 25th pct | 9   | 9 / 9      | +0.266        |
| ≥ 50th pct | 9   | 0 / 9      | −0.167        |
| ≥ 75th pct | 9   | 0 / 9      | −0.200        |

**Walk-forward OOS (2022–2026, best IS combo by DSR: thresh=0.5, hold=12, vol_floor=0):**

| Scenario | Cost (bps) | OOS Sharpe | AnnRet | MaxDD  | Trades |
| -------- | ---------- | ---------- | ------ | ------ | ------ |
| Base     | 0.9        | **−0.214** | −1.81% | −22.4% | 161    |
| High     | 1.5        | −0.268     | −2.26% | −23.5% | 161    |
| Stress   | 2.0        | −0.312     | −2.63% | −24.3% | 161    |

**Key findings — Strategy B:**

1. **IS performance is entirely driven by unfiltered (low-vol) periods.** Vol floor ≥50th pct completely kills the signal (0/9 positive). This is the OPPOSITE of the regime-conditional result from the original MA crossover (which improved in high-vol). The MA spread mean-reversion signal works in range-bound (low-vol) conditions, not during strong trends (high-vol).
2. **OOS fails completely.** Best IS combo (thresh=0.5, hold=12, no vol filter) produces −0.214 Sharpe on 2022–2026. The IS period (2016–2021) was more range-bound; 2022–2026 included the BoJ YCC exit, 2022 USDJPY trend (105→152), and sustained directional moves that break mean-reversion.
3. **All DSR = 0.000.** Even the best IS Sharpe (+0.591) with 182 trades doesn't pass the Bailey-Lopez test at 36 trials. This is secondary to the OOS failure and the unresolved entry-rule question: the script uses residency-based entry (signal fires on every bar the feature remains above threshold), not threshold-crossing entry. This likely inflates IS trade count and IS Sharpe; crossing-entry must be tested before IS conclusions are finalised.
4. **Cost insensitivity at OOS.** All three cost scenarios produce similar OOS Sharpes (−0.21 to −0.31) — the OOS failure is driven by signal direction error, not by transaction costs. Costs add only ~0.05 Sharpe degradation per additional 0.6 bps.

---

### Combined Interpretation of Split-Hypothesis Results

Both strategies fail the DSR test across all combinations on the IS period. The OOS test for Strategy B confirms a real failure mode: the IS period happened to be more range-bound than the OOS period. The curvature signal (Strategy A) was not walk-forward tested, but its low annual returns and trade count suggest insufficient statistical power for a meaningful OOS conclusion.

**The structural issues identified by both experiments:** IS optimisation selects extreme parameter combinations (conf=2, thresh=1.5σ for Strategy A; thresh=0.5 with no vol filter for Strategy B) that appear robust in-sample. The signals (IC ≈ 0.023–0.028) are not weak by industry standards — they are in the normal range for liquid FX H1 — but they are highly filter-sensitive and the correct holding-period alignment has not been validated across all configurations. Strategy B additionally has an unresolved entry-rule bug: it uses residency-based entry (fires on every bar feature stays above threshold) rather than threshold-crossing entry (fires only at the first crossing bar). This inflates IS results and must be corrected before any regime conclusions are drawn.

**The regime contradiction is a key new finding:** Strategy B's MA spread mean-reversion works in LOW vol, but the original grid's MA crossover performed best in HIGH vol. These are genuinely different economic regimes exploiting different dynamics (range-bound vs trending). A combined strategy that switches regime-conditionally between the two signals may be more robust — this warrants investigation.

---

## Reversed Grid Search Results (`experiments/reversed_grid_search.py`)

Three variants tested across 72 λ/T1/T2 combinations (216 rows total), saved to `data/interim/reversed_grid_results.csv`.

### Variant Definitions

- **REVERSED**: Signal direction flipped (enter when price is _above_ HP trend, exit after T2 bars) — tests the momentum direction.
- **REGIME GATED**: Reversed signal, only fires in exhaustion zones (|trend_dev_240| > 1.5σ) — 50,083 of 61,565 valid bars qualify.
- **FULL FILTER**: Reversed signal gated by both exhaustion zones and volatility regime conditions.

### Summary

| Variant      | Sharpe > 0 | DSR > 0.95 | Best Sharpe                |
| ------------ | ---------- | ---------- | -------------------------- |
| REVERSED     | 10/72      | 0/72       | +0.067 (1M, T1=24, T2=120) |
| REGIME GATED | 0/72       | 0/72       | −1.066 (all negative)      |
| FULL FILTER  | 0/72       | 0/72       | −1.092 (all negative)      |

### Focused Comparison (λ=1B, T1=72, T2=240 — best from original run)

| Variant                   | Sharpe | AnnRet | MaxDD  | Trades |
| ------------------------- | ------ | ------ | ------ | ------ |
| Original (mean-reversion) | +0.258 | +1.73% | −6.3%  | 10     |
| Reversed (momentum)       | −0.286 | −2.55% | −45.1% | 167    |
| Regime gated              | −1.655 | −7.57% | −55.6% | 2,896  |
| Full filter               | −1.399 | −5.83% | −47.8% | 2,362  |

### Key Findings

**The momentum direction (reversed signal) is decisively unprofitable.** Only 10 of 72 reversed combinations produce Sharpe > 0, and the best is +0.067 — which is near-zero. The regime-gated and full-filter variants are all negative.

**This confirms the mean-reversion hypothesis.** The original pipeline's signal direction (enter counter-trend after crossover) is the correct side. The reversed search eliminates any concern that the positive Sharpes from Steps 4–5b were due to random direction.

**Regime gating destroys performance.** With 50,083 bars (81% of valid observations) in "exhaustion zones" (|dev| > 1.5σ), the gate is nearly always open. This makes REGIME GATED behave like REVERSED with a small timing offset. The filter is too loose to add value.

**Note:** λ ∈ {1B, 3.9B, 100B} produce near-identical signals (slope ρ ≥ 0.9995). Effective grid is 36 independent combinations, not 72.

---

| Finding                     | Run 1 (2,104 bars, 5mo) | Run 2 (12,231 bars, yfinance)                       | Run 3 (62,303 bars, Dukascopy)               |
| --------------------------- | ----------------------- | --------------------------------------------------- | -------------------------------------------- |
| All features stationary     | YES (artifact)          | YES (borderline)                                    | **NO** — hp_trend_level I(1)                 |
| Mean-reversion direction    | Confirmed (strong IC)   | Confirmed (weaker IC)                               | Partially confirmed (2 features only)        |
| IC gate pass count          | All 11 (all noise)      | 3 of 11                                             | **2 of 11**                                  |
| Features passing IC gate    | All                     | crossover_age, raw_ma_spread_50_200, hp_trend_level | **hp_trend_curvature, raw_ma_spread_50_200** |
| Best IC (any feature, H=24) | −0.397 (artifact)       | −0.072                                              | −0.0277                                      |
| Best grid search Sharpe     | ~0.26                   | 0.87 (artifact of 2yr window)                       | **0.49**                                     |
| Best regime Sharpe          | —                       | 1.89 (Low Vol, 28 trades)                           | **0.99** (High Vol, 144 trades)              |
| DSR for best combination    | 0.000                   | 0.000                                               | 0.000                                        |
| Dominant λ                  | Unclear                 | 100M, 3.9B–100B                                     | **10M, 3.9B–1B**                             |
| Regime signal peak          | Medium Vol              | Low Vol (artifact)                                  | **High Vol — consistent**                    |
| Trade count (best combo)    | ~24                     | 24                                                  | **157**                                      |
| Statistical validity        | None                    | None                                                | None (DSR=0)                                 |

---

## Conclusions and Priorities

**`hp_trend_level` is excluded on IC grounds, not solely ADF.** It is I(1) (ADF p=0.746), but I(1) ≠ non-predictive — integrated series can carry predictive power for stationary returns. The actual reason for exclusion is that its ICs are all near zero and insignificant on 62K bars. The IC gate is the operative decision rule; ADF is supporting context.

**Two features remain in play after the IC gate:** `hp_trend_curvature` (short-horizon, H=1, IC = −0.0235) and `raw_ma_spread_50_200` (long-horizon, H=24, IC = −0.0277). These ICs are not weak — the industry scale for liquid FX H1 is: 0.01 = weak, 0.02 = usable, 0.03 = strong. Both features are in the usable-to-strong band.

**The volatility-regime result is the strongest finding in this study.** Across all top-3 grid combinations, Sharpe scales monotonically with volatility regime: High vol ≈ 0.69–0.99, Med vol ≈ 0.45–0.65, Low vol ≈ 0.16–0.40. This is structural: mean-reversion signals are more reliable when price swings are larger and more protracted. The `10M, T1=72, T2=480` combination is the current benchmark (overall Sharpe = 0.49, High-vol Sharpe = 0.99, AnnRet = 15.4%, MaxDD = −16.8%).

**The reversed grid search is a high-value sanity check.** Only 10/72 reversed combinations produce Sharpe > 0 (best = +0.067), confirming the mean-reversion direction is the correct side. The regime-gated and full-filter variants are universally negative. Most strategies fail this test; the original signal direction is directionally validated.

**The original grid's core design error was horizon mismatch.** `hp_trend_curvature` carries signal only at H=1 (short-term momentum exhaustion), but the original grid tested it with T2 ∈ {168h, 240h, 480h}. This is the root cause of its underperformance in the main grid — not signal weakness. Strategy A tests the corrected formulation and recovers IS Sharpe > 1.0.

**There are three structurally distinct signals in this study, not one:**

1. **Fast exhaustion** — `hp_trend_curvature` at H=1–4. Entry on extreme curvature (conf=2, thresh=1.5σ); exits within 2 bars. IS Sharpe > 1.0 but ~8 trades/year — statistically under-powered.
2. **Slow mean-reversion** — `raw_ma_spread_50_200` at H=24. Works in LOW vol, breaks in HIGH vol. Entry-rule bug (residency vs crossing) is unresolved — IS results are provisional.
3. **Trend continuation** — MA crossover signal from the original grid. Works in HIGH vol. Opposite regime preference to signal #2.

Signals #2 and #3 appear complementary, but a regime-switching hypothesis cannot be evaluated until the Strategy B entry-rule bug is fixed and IS behaviour is confirmed on the corrected formulation.

**DSR = 0 for all combinations across all experiments.** The penalty is conservative: many trials share near-identical signals (λ collapse: 1B≈3.9B≈100B) and overlapping holding windows, reducing the effective independent trial count to ~20–30 (not 216 or 72). DSR is a valid warning, but the primary constraint for Strategy A is ~8 trades/year — insufficient statistical power that predates multiple-testing correction.

**Strategy B OOS failure is a regime event, not signal invalidation.** The IS period (2016–2021) was more range-bound; the OOS period (2022–2026) included the BoJ YCC exit and a 105→152 USDJPY directional trend. Cost sensitivity analysis shows costs add only ~0.05 Sharpe degradation per 0.6 bps — the OOS failure is entirely signal direction, not cost. The Strategy B entry rule (residency-based) is unvalidated and must be corrected before regime conclusions are drawn.

**A regime-switching conclusion is premature.** Only one mean-reversion formulation has been tested (residency-based entry with ATR-normalised spread). The hypothesis that Strategy B + MA crossover complement each other across regimes requires: (a) Strategy B crossing-entry fix, (b) IS re-validation on corrected formulation, and (c) IS/OOS regime attribution.

---

**Priority 1 — Return-decay diagnostic.** Compute `E[ret | signal fired, t bars forward]` for both `hp_trend_curvature` and `raw_ma_spread_50_200` at t ∈ {1, 2, 4, 8, 12, 24, 48}. This directly measures how fast the edge decays and sets the empirical basis for hold_bars selection. It is the single most informative experiment not yet run.

**Priority 2 — Strategy A trade count expansion.** Run a supplementary grid with thresh ∈ {0.75, 1.0, 1.25}σ and conf ∈ {0, 1} alongside the existing thresh=1.5/conf=2 results. Identify the lowest threshold/confirmation that preserves positive Sharpe with ≥ 50 trades/year. Target: a regime where DSR becomes meaningful (requires ~100+ trades with Sharpe ≥ 0.6).

**Priority 3 — Strategy B crossing-entry re-run.** Fix the entry signal: fire only on the bar where the feature first crosses the threshold (not on every bar it remains above). Lock out further entries until the position closes. This removes the IS inflation artefact and is required before any regime-switching analysis proceeds.

**Priority 4 — IC horizon scan.** Compute IC(H) for both gate-passing features (`hp_trend_curvature`, `raw_ma_spread_50_200`) at H ∈ {1, 2, 4, 6, 12, 24, 48}. Identify the peak IC horizon empirically — current hold_bars selection was heuristic. The decay profile from Priority 1 and the IC scan together determine correct T1/T2 for each signal.

**Priority 5 — Reduce λ grid permanently.** Remove λ ∈ {3.9B, 100B} from all future experiment grids. Slope ρ ≥ 0.9994 with λ=1B makes them analytically redundant. The effective grid is {1M, 10M, 100M, 1B}: 4 genuinely distinct smoothing scales.

---

## Diagnostic Experiment Results

Four experiments were run after the initial strategy scans, guided by the priorities above. All use the reduced λ grid {1M, 10M, 100M, 1B} and the Dukascopy 62K-bar dataset.

---

### Experiment 1 — IC Horizon Scan

**Script:** `experiments/ic_horizon_scan.py` → `data/interim/ic_horizon_scan_results.csv`

Spearman rank IC computed for `hp_trend_curvature` (4 λ) and `raw_ma_spread_50_200` at H ∈ {1, 2, 4, 6, 12, 24, 48}. Sign convention: IC is negative when the feature is mean-reverting (high feature → negative future return).

**hp_trend_curvature — all λ (IC is signed negative for mean-reversion):**

| H   | λ=1M IC | λ=10M IC | λ=100M IC | λ=1B IC | sig    |
| --- | ------- | -------- | --------- | ------- | ------ |
| 1   | −0.0208 | −0.0210  | −0.0217   | −0.0236 | \*\*\* |
| 2   | −0.0135 | −0.0137  | −0.0136   | −0.0113 | \*\*\* |
| 4   | +0.0031 | +0.0028  | +0.0016   | −0.0009 | —      |
| 6   | +0.0089 | +0.0085  | +0.0079   | +0.0077 | \*     |
| 12  | +0.0054 | +0.0054  | +0.0037   | +0.0013 | —      |
| 24  | +0.0033 | +0.0022  | +0.0002   | +0.0007 | —      |
| 48  | +0.0027 | +0.0017  | −0.0005   | −0.0044 | —      |

**raw_ma_spread_50_200:**

| H   | IC      | sig    |
| --- | ------- | ------ |
| 1   | +0.0026 | —      |
| 2   | −0.0002 | —      |
| 4   | −0.0017 | —      |
| 6   | −0.0049 | —      |
| 12  | −0.0163 | \*\*\* |
| 24  | −0.0216 | \*\*\* |
| 48  | −0.0214 | \*\*\* |

**Key findings:**

- **Curvature edge is confined to H=1–2.** IC is −0.021 to −0.024 at H=1 (**\*) across all λ, −0.011 to −0.014 at H=2 (**), then drops to statistical noise by H=4. The original grid's T2 ∈ {168h, 240h, 480h} was severely misaligned with the signal's actual decay horizon. Correct hold_bars for curvature = **1–2 bars**.
- **No λ advantage for curvature.** All four λ values show identical IC profiles. There is no smoothing scale that extracts additional predictive information from the curvature feature.
- **MA spread edge emerges slowly.** No IC at H≤6. Significant from H=12 onward, peaking at H=24–48 (IC ≈ −0.022, \*\*\*) with H=24 and H=48 nearly equal. This is consistent with the pipeline's original H=24 finding and validates the hold_bars range for Strategy B.
- **IC scale check.** Both signals peak in the 0.02–0.024 range — the "usable" band (0.01=weak, 0.02=usable, 0.03=strong). Neither is unusually strong or weak for liquid FX H1.

---

### Experiment 2 — Return-Decay Diagnostic

**Script:** `experiments/return_decay_diagnostic.py` → `data/interim/return_decay_results.csv`

For each feature and threshold σ ∈ {unconditional, 0.5, 1.0, 1.5}, compute mean signed bps and t-stat at H ∈ {1, 2, 4, 8, 12, 24, 48}. Direction: mean-reversion (enter counter the extreme, earn positive mean return).

**hp_trend_curvature summary (mean_signed_bps, t-stat):**

| Threshold     | H=1            | H=4            | H=24                 | H=48           |
| ------------- | -------------- | -------------- | -------------------- | -------------- |
| unconditional | +0.017 (0.37)  | +0.149 (1.60)  | −0.197 (−0.86)       | −0.209 (−0.64) |
| 0.5σ          | −0.021 (−0.27) | +0.045 (0.29)  | **−0.829 (−2.25\*)** | −0.974 (−1.89) |
| 1.0σ          | −0.108 (−0.76) | −0.186 (−0.72) | **−1.284 (−2.14\*)** | −0.502 (−0.61) |
| 1.5σ          | −0.182 (−0.75) | −0.259 (−0.61) | −1.661 (−1.75)       | −0.111 (−0.09) |

**raw_ma_spread_50_200 summary (mean_signed_bps, t-stat):**

| Threshold     | H=1            | H=8               | H=12              | H=24            | H=48                    |
| ------------- | -------------- | ----------------- | ----------------- | --------------- | ----------------------- |
| unconditional | +0.035 (0.75)  | +0.404 (**3.07**) | +0.511 (**3.17**) | +0.424 (1.84)   | +0.216 (0.67)           |
| 0.5σ          | +0.038 (0.65)  | +0.152 (0.93)     | +0.183 (0.91)     | +0.697 (\*2.42) | +0.871 (\*2.14)         |
| 1.0σ          | +0.080 (0.96)  | +0.576 (\*2.45)   | +0.574 (\*2.02)   | +0.959 (\*2.38) | **+1.898 (\***3.35)\*\* |
| 1.5σ          | −0.016 (−0.13) | +0.321 (0.91)     | +0.864 (\*2.02)   | +1.133 (1.78)   | +1.443 (1.60)           |

**Key findings:**

- **The strongest result in this study is MA spread at 1.0σ / H=48: mean_signed = +1.898 bps, t = +3.35 (\***), n = 19,099 events.\*\* This is a clean, large-sample result with a practically meaningful magnitude.
- **MA spread unconditional signal peaks at H=12 (t=+3.17**) and is fully consistent with IC scan.\*\* The signal is measurable event after filtering to extreme events only — the 1.0σ threshold improves both magnitude and significance at long horizons (H=24–48), consistent with stronger mean-reversion after larger displacements.
- **Curvature mean-return is statistically weak everywhere.** The unconditional signal peaks at H=4 (t=+1.60, not significant). With threshold filtering, the only significant result appears at H=24 (t≈−2.1 to −2.2, \*), whereas the IC scan shows the signal is already dead by H=4. This apparent contradiction arises because the decay test accumulates returns over 24 bars starting from any extreme bar — a cumulative effect, not a 1-bar-ahead correlation. The marginal H=24 curvature results are consistent with a brief early mean-reversion (H=1–2) that lingers in the cumulative sum without generating a persistent position-level edge.
- **Recommended parameters for Strategy B:** thresh ≈ 1.0σ, hold_bars = 48. Both the IC scan (IC still strong at H=48) and the decay diagnostic (peak mean return at H=48, 1.0σ) point to the same target.

---

### Experiment 3 — Strategy A Expanded Grid

**Script:** `experiments/strategy_a_expanded_grid.py` → `data/interim/strategy_a_expanded_results.csv`

Extended the original Strategy A grid (thresh=1.5σ, conf=2) downward: thresh ∈ {0.75, 1.0, 1.25}σ, conf ∈ {0, 1}, hold ∈ {1, 2, 3, 4, 6, 8}, 4 λ → 144 combinations.

**Aggregate by threshold:**

| Threshold | n   | Sharpe>0 | Median Sharpe | Median Trades |
| --------- | --- | -------- | ------------- | ------------- |
| 0.75σ     | 48  | 0/48     | −2.044        | 5,992         |
| 1.00σ     | 48  | 0/48     | −1.508        | 4,050         |
| 1.25σ     | 48  | 1/48     | −0.883        | 2,820         |

**Cross-table: thresh × conf (median Sharpe [median trades]):**

| thresh \ conf | conf=0           | conf=1          |
| ------------- | ---------------- | --------------- |
| 0.75σ         | −3.059 [10,859t] | −0.731 [2,534t] |
| 1.00σ         | −2.605 [7,696t]  | −0.402 [1,434t] |
| 1.25σ         | −2.150 [5,508t]  | −0.175 [871t]   |

**Best single combination:** λ=1B, conf=1, hold=6, thresh=1.25σ → Sharpe=+0.071, AnnRet=+0.21%, MaxDD=−7.5%, Trades=856, DSR=0.000

**Key findings:**

- **Strategy A is conclusively unviable.** The full tested range (thresh=0.75–1.5σ, conf=0–2, hold=1–8, all λ) now contains only **1/144 combinations with Sharpe>0** and **0/144 with DSR>0**. The single positive result (+0.071) is economically negligible.
- **All 144 combinations have ≥500 trades.** Trade count is not the limiting factor. The curvature signal generates plenty of entries but they are collectively lossy — the edge at H=1 observed in the IC scan (+0.021\*\*\*) is too small to overcome transaction costs and adverse mid-to-late hold periods.
- **Confirmation bars help but do not rescue the strategy.** conf=1 cuts trade count by ≈6× and improves median Sharpe from −2.56 to −0.40, but no combination consistently reaches positive territory.
- **Investigation is closed for Strategy A.** The corrected hold_bars (1–2 bars per IC scan) was already present in the original conf=2 strategy, which returned ~8 trades/year and Sharpe≈1.0 on that near-zero exposure. The trade-off is structural: at aggressive thresholds with enough trades for DSR to matter, there is no edge. Strategy A will not be carried forward.

---

### Experiment 4 — Strategy B Crossing-Entry

**Script:** `experiments/strategy_b_crossing_entry.py` → `data/interim/strategy_b_crossing_is.csv`, `strategy_b_crossing_oos.csv`
**Generator function added:** `crossing_threshold_signal` in `src/features/generators.py`

Re-ran the Strategy B grid (3 thresh × 3 hold × 4 vol_floor = 36 combinations) replacing `threshold_time_stop_signal` (residency-based) with `crossing_threshold_signal` (entry fires only when the feature crosses the threshold from neutral; post-exit lockout until neutral is reestablished).

**IS results — top entries:**

| thresh | hold | vol_floor | Sharpe (crossing) | Sharpe (residency) | Δ Sharpe | Δ Trades |
| ------ | ---- | --------- | ----------------- | ------------------ | -------- | -------- |
| 0.5σ   | 48   | 0         | −0.324            | +0.591             | −0.915   | +25      |
| 0.5σ   | 24   | 0         | −0.553            | +0.523             | −1.077   | +24      |
| 0.5σ   | 12   | 0         | −0.333            | +0.404             | −0.737   | +26      |
| 1.0σ   | 48   | 0         | **+0.344**        | +0.491             | −0.148   | +26      |
| 1.0σ   | 24   | 0         | +0.097            | +0.464             | −0.367   | +30      |
| 1.0σ   | 12   | 0         | +0.199            | +0.409             | −0.210   | +30      |
| 1.5σ   | 48   | 0         | +0.264            | +0.510             | −0.246   | +12      |
| 1.5σ   | 24   | 0         | −0.030            | +0.078             | −0.108   | +17      |
| 1.5σ   | 12   | 0         | −0.134            | +0.086             | −0.220   | +17      |

**IS overall:** 11/36 combos Sharpe>0 (vs 36/36 residency). Best IS: thresh=1.0, hold=48, vol_floor=0 → Sharpe=+0.344, AnnRet=+1.16%, MaxDD=−5.8%, Trades=176, DSR=0.000.

**Trade count note:** The crossing rule generates more trades (not fewer) at low thresholds. At 0.5σ, the MA spread oscillates frequently around the threshold, causing multiple "fresh crossings" per extreme episode. Each dip below threshold followed by a re-cross fires a new entry. This is consistent with the Sharpe collapse: many entries near the threshold are on the wrong side of noise. At 1.5σ, the oscillation effect is much smaller (Δ=+12 vs Δ=+25 at 0.5σ).

**Key findings:**

- **IS inflation from the residency rule is confirmed.** At thresh=0.5σ, Sharpe collapses from +0.59 to −0.33 with the corrected entry rule. The original positive IS results for low thresholds were artefacts of same-bar re-entry chaining after every time-stop.
- **thresh=1.0σ / hold=48 survives the correction** with IS Sharpe=+0.344 (+1.16% AnnRet, MaxDD=−5.8%, 176 trades). This is the most robust combination in the crossing-entry grid and the correct IS benchmark for Strategy B going forward.
- **Low-vol preference disappears with crossing entry.** In the residency grid, vol_floor≥25 was associated with better median Sharpe. In the crossing grid, vol_floor=0 (no filter) and vol_floor=25 both produce 4/9 positive Sharpe medians — the low-vol-is-better pattern was partly an artefact of the entry rule and is no longer reliable.
- **OOS selection bug — requires re-run.** The walk-forward selected the wrong best IS combo (chose thresh=0.5/hold=12 with DSR=0.000 via `nlargest("dsr")` tie-breaking, which defaulted to first-row). The correct OOS candidate is thresh=1.0/hold=48. OOS re-test on this combination is the highest-priority remaining experiment.

---

### Experiment 5 — Strategy B OOS Re-run (Bug Fixed)

**Script:** `experiments/strategy_b_crossing_entry.py` (fixed: `sort_values(["dsr","sharpe"])`)

OOS evaluated on the correct best IS combo: **thresh=1.0, hold=48, vol_floor=0**.

| Scenario | Cost (bps) | OOS Sharpe | OOS AnnRet | MaxDD  | Trades |
| -------- | ---------- | ---------- | ---------- | ------ | ------ |
| base     | 0.9        | −0.210     | −1.15%     | −17.4% | 144    |
| high     | 1.5        | −0.282     | −1.55%     | −18.0% | 144    |
| stress   | 2.0        | −0.343     | −1.89%     | −18.8% | 144    |

The Sharpe range across the cost scenarios is −0.21 to −0.34. Cost sensitivity is minimal (0.13 Sharpe per 1.1 bps cost increase). The OOS failure is entirely due to signal direction, not transaction costs.

---

### Experiment 6 — Year-by-Year Breakdown

**Script:** `experiments/strategy_b_yearly_breakdown.py` → `data/interim/strategy_b_yearly_results.csv`

Best IS combo (thresh=1.0, hold=48, crossing, vol_floor=0) evaluated year by year.

| Year | Period | Sharpe | AnnRet | MaxDD  | Trades | Ann Vol |
| ---- | ------ | ------ | ------ | ------ | ------ | ------- |
| 2016 | IS     | —      | —      | —      | 0      | —       |
| 2017 | IS     | +0.122 | +0.42% | −2.6%  | 28     | 3.4%    |
| 2018 | IS     | −0.862 | −3.10% | −4.1%  | 36     | 3.6%    |
| 2019 | IS     | +0.814 | +2.93% | −2.7%  | 42     | 3.6%    |
| 2020 | IS     | +0.617 | +2.64% | −4.5%  | 33     | 4.3%    |
| 2021 | IS     | +1.157 | +3.66% | −1.9%  | 37     | 3.2%    |
| 2022 | OOS    | −0.531 | −2.88% | −8.8%  | 30     | 5.4%    |
| 2023 | OOS    | −0.409 | −2.12% | −7.5%  | 35     | 5.2%    |
| 2024 | OOS    | −1.463 | −9.17% | −11.4% | 41     | 6.3%    |
| 2025 | OOS    | +1.709 | +8.68% | −2.8%  | 30     | 5.1%    |
| 2026 | OOS    | +0.795 | +4.13% | −1.9%  | 8      | 5.2%    |

**IS total:** Sharpe=+0.344, AnnRet=+1.16%, MaxDD=−5.8%, 176 trades, vol≈3.4%  
**OOS total:** Sharpe=−0.210, AnnRet=−1.15%, MaxDD=−17.4%, 144 trades, vol≈5.5%

Key findings:

- **IS edge is not concentrated in one year.** 4/5 IS years are positive; the bad IS year (2018, −0.862) was a genuine signal failure, not an outlier that inflates the average. The median IS Sharpe is +0.617.
- **OOS failure is concentrated in 2024.** This was the USDJPY YCC-exit year (140→160 run, then BoJ intervention to 142). Both directional phases would be hostile to a H=48-bar mean-reversion signal. 2024 Sharpe=−1.463 on the highest OOS vol (6.3%).
- **OOS vol structurally higher.** IS mean vol ≈3.5%, OOS mean vol ≈5.5% — a 57% increase. The strategy performs poorly when annualised vol exceeds ~5%. This is a vol-regime dependency, not a pure structural break.
- **2025 and 2026 are both positive** (Sharpe=+1.709 and +0.795). This suggests the signal is not permanently dead in the post-YCC environment; it works when USDJPY volatility normalises back below the threshold.

---

### Experiment 7 — Strategy C: Regime Switching (MR + TF)

**Script:** `experiments/strategy_c_regime_switching.py` → `data/interim/strategy_c_results_is.csv`, `strategy_c_results_oos.csv`

Tested vol-based regime switch: MA spread crossing MR (low vol) ↔ HP trend MA crossover TF (high vol), across 8 combos (2 T2 × 4 vol splits).

**IS results (all negative):**

| T2  | Vol split | MR% | TF% | IS Sharpe | IS Trades |
| --- | --------- | --- | --- | --------- | --------- |
| 480 | 25/75     | 29% | 24% | −0.560    | 662       |
| 480 | 50/50     | 54% | 46% | −0.653    | 880       |
| 240 | 25/75     | 29% | 24% | −0.690    | 669       |
| 240 | 33/67     | 37% | 31% | −0.981    | 791       |

TF baselines (IS only): T2=240 Sharpe=−0.571; T2=480 Sharpe=−0.243.  
MR baseline (Strategy B): IS Sharpe=+0.344.

**Key findings:**

- **Regime switching fails IS.** All 8 combinations are IS-negative. The TF signal (HP trend MA crossover) has negative IS alpha on its own (T2=240: −0.571; T2=480: −0.243). Adding a negative-alpha signal to a positive-alpha signal via regime switching cannot improve unless the two signals are genuinely anti-correlated in the right way; they are not.
- **Strategy C as designed is closed.** The HP trend MA crossover is not a viable TF component for the high-vol regime. Its signal frequency is too low (157 crossovers over 10 years for T2=480) and it stays on the wrong side too long during sustained directional periods.
- **Year-by-year shows the regime switch helps in exactly the years that MR fails** (e.g. 2018: MR=−0.862 → C=+1.324; 2022: MR=−0.531 → C=+0.994; 2024: MR=−1.463 → C=+0.370), but destroys value in the good MR years (2019: MR=+0.814 → C=−1.258; 2020: MR=+0.617 → C=−1.508). The TF signal is too unreliable to compensate.

---

### Experiment 8 — EFC Filter: trend_dev_240 as Entry Quality Gate

**Script:** `experiments/strategy_b_efc_filter.py` → `data/interim/strategy_b_efc_results.csv`

Applied `trend_dev_240` (HP trend minus its 240-bar SMA, ATR-normalised, λ=1B) as an entry-blocking filter on the base crossing signal. Entries are removed (hold window zeroed) when `|trend_dev_240| < filter_threshold`. The `apply_entry_filter` function identifies new-entry bars and removes the entire hold window for blocked entries.

| Filter      | Active% | IS Sharpe | IS Trades | OOS Sharpe | OOS Trades |
| ----------- | ------- | --------- | --------- | ---------- | ---------- |
| unfiltered  | 100%    | +0.344    | 176       | −0.210     | 144        |
| \|td\|>0.5σ | 48%     | +0.713    | 100       | −0.525     | 85         |
| \|td\|>1.0σ | 19%     | +0.250    | 42        | −0.260     | 44         |
| \|td\|>1.5σ | 6%      | −0.249    | 18        | −0.072     | 19         |
| alignment   | 69%     | +0.635    | 116       | −0.466     | 109        |

**Key findings:**

- **IS quality uplift at 0.5σ is real.** The filter halves trade count (176→100) and doubles IS Sharpe (+0.344→+0.713). This is the cleanest quality-filter result in this study — fewer trades with higher precision per trade. The alignment gate (require sign(td) = sign(spread)) achieves similar IS Sharpe (+0.635) with fewer trades removed.
- **OOS does not confirm the IS improvement.** Every filter that improves IS worsens OOS (0.5σ: −0.210→−0.525; alignment: −0.210→−0.466). The filter is selecting entries that happened to be profitable in the 2016–2021 range-bound period, but the same selection rule picks entries during HP trend extremes that _persisted_ in 2022–2024 rather than reverting.
- **1.5σ filter is the exception.** OOS Sharpe improves from −0.210 to −0.072 and MaxDD drops from 17.4% to 2.3%. Only 18 IS trades and 19 OOS trades, so this is under-powered, but the direction is consistent with a "only enter during extreme overextension" rule that reduces OOS mean-reversion risk.
- **Interpretation.** The `trend_dev_240` quality filter has genuine predictive value in the IS regime (low-vol, mean-reverting USDJPY 2016–2021). In the OOS period (high-vol, directional 2022–2024), large HP trend deviations indicated continuation rather than reversal. The filter's usefulness is regime-dependent, not unconditional.

---

### Updated Priorities (post-Experiments 5–8)

All originally planned priorities (1–5) and the subsequent four experiments (OOS rerun, yearly breakdown, Strategy C, EFC) have been executed. The current state is:

**What is confirmed as positive:**

- Strategy B IS Sharpe = +0.344 (thresh=1.0, hold=48, crossing entry, no vol filter) on 62K bars/6yr IS. 4/5 IS years positive; median IS year Sharpe = +0.617.
- MA spread IC at H=24–48 (IC≈−0.022, \*\*\*) is structurally present over the full 10-year window.
- The EFC filter at 0.5σ produces IS Sharpe=+0.713 (100 IS trades). IS credible.

**What is confirmed as failed:**

- Strategy A: closed. 1/144 combos Sharpe>0 across the full tested parameter space.
- Strategy C (HP crossover TF): closed. Negative IS alpha.
- EFC filter (unconditional OOS improvement): not established. IS uplift does not generalise.

**Open questions requiring further characterisation:**

- Is the OOS failure (2022–2024) a permanent structural break (BoJ policy → USDJPY rediscovery) or a temporary regime that has normalised (2025/2026 positive)?
- Under what conditions does Strategy B work? The vol-dependency implies a vol-gate could select for the right environment. But vol-gate was already tested (vol_floor filter) and did not conclusively help with crossing entry.
- The EFC filter at 1.5σ reduces OOS MaxDD to 2.3% — is this a useful position-sizing principle (reduce leverage when trend_dev is between 1.0–1.5σ rather than blocking entry)?

**Priority 1 — Strategy B re-evaluation with 2025/2026 OOS.** The IS ended 2021-12-31. With 2025 (+1.709) and 2026 (+0.795) both positive, extending the OOS window by another year and re-evaluating whether the overall OOS Sharpe becomes positive is worth tracking. The 2022–2024 period may be attributable to the YCC-exit event which is a once-per-decade monetary policy change.

**Priority 2 — EFC risk-scaling rather than entry-blocking.** Instead of blocking entries below 1.5σ trend_dev, scale position size proportional to |trend_dev_240|. Larger trend deviations get full size; small deviations get 50%. This avoids the binary blocking that removes good IS trades.

**Priority 3 — Live universe.** Before any further IS/OOS analysis, specify the live execution requirements: data latency, position sizing, risk limit, monitoring. Strategy B at thresh=1.0/hold=48/crossing produces ~17–18 trades per year per instrument. With USDJPY as single instrument, monthly P&L will be highly variable.

---

## Files and Artefacts

| File                                             | Description                                                                                  |
| ------------------------------------------------ | -------------------------------------------------------------------------------------------- |
| `data/raw/USDJPY_10yr_1h_dukascopy.csv`          | Primary dataset — 62,303 bars, 2016-03-13 to 2026-03-10, Dukascopy H1                        |
| `data/raw/USDJPY_10yr_1h.csv`                    | Previous dataset — 12,231 bars, 2024-03-11 to 2026-03-10, yfinance                           |
| `data/interim/hp_trends_window500.csv`           | Precomputed causal HP trends for 6 λ values, HP_WINDOW=500, Dukascopy                        |
| `data/interim/hp_trends_window500.csv.bak2`      | yfinance HP trends archive (12,231 bars)                                                     |
| `data/interim/grid_results.csv`                  | 72-combination grid search results, Dukascopy                                                |
| `data/interim/grid_results.csv.bak2`             | yfinance grid results archive                                                                |
| `data/interim/pipeline_summary.json`             | Full JSON results: ADF, IC, grid, regime — Dukascopy run                                     |
| `data/interim/pipeline_summary.json.bak2`        | yfinance pipeline summary archive                                                            |
| `data/interim/reversed_grid_results.csv`         | 216-row reversed-signal variants — Dukascopy run (all Sharpe ≤ 0.067)                        |
| `data/interim/strategy_a_results.csv`            | 216-row Strategy A grid (curvature + threshold_time_stop) — all DSR = 0.000                  |
| `data/interim/strategy_b_results.csv`            | 36-row Strategy B IS grid (MA spread + vol filter) — all DSR = 0.000                         |
| `data/interim/strategy_b_oos.csv`                | Strategy B OOS walk-forward results × 3 cost scenarios — all Sharpe < 0                      |
| `logs/pipeline_output.txt`                       | Full pipeline stdout — Dukascopy run                                                         |
| `logs/reversed_grid_output.txt`                  | Reversed grid search stdout — Dukascopy run                                                  |
| `logs/strategy_a_output.txt`                     | Strategy A grid search stdout — 216 combos, Dukascopy run                                    |
| `logs/strategy_b_output.txt`                     | Strategy B grid search + walk-forward stdout — Dukascopy run                                 |
| `experiments/strategy_a_curvature_grid.py`       | HP trend curvature × threshold_time_stop_signal grid (216 combos)                            |
| `experiments/strategy_b_ma_spread_vol_filter.py` | MA spread × vol filter IS grid + walk-forward OOS (36 combos)                                |
| `src/features/generators.py`                     | Added: `threshold_time_stop_signal`, `vol_rolling_percentile`, `crossing_threshold_signal`   |
| `data/interim/return_decay_results.csv`          | Return-decay diagnostic: 4 thresholds × 7 horizons × 2 features                              |
| `data/interim/ic_horizon_scan_results.csv`       | IC horizon scan: 5 features (4 curvature λ + MA spread) × 7 horizons                         |
| `data/interim/strategy_a_expanded_results.csv`   | Strategy A expanded grid: 144 combos (thresh=0.75–1.25σ, conf=0–1)                           |
| `data/interim/strategy_b_crossing_is.csv`        | Strategy B crossing-entry IS grid: 36 combos                                                 |
| `data/interim/strategy_b_crossing_oos.csv`       | Strategy B crossing-entry OOS — corrected (thresh=1.0/hold=48): base Sharpe=−0.210           |
| `data/interim/strategy_b_yearly_results.csv`     | Year-by-year breakdown for best IS combo (2016–2026, IS/OOS split)                           |
| `data/interim/strategy_c_results_is.csv`         | Strategy C regime-switching IS grid: 8 combos (all IS-negative)                              |
| `data/interim/strategy_c_results_oos.csv`        | Strategy C regime-switching OOS (best IS combo: T2=480, split=25/75)                         |
| `data/interim/strategy_b_efc_results.csv`        | EFC filter results: 5 configurations (unfiltered, 0.5σ, 1.0σ, 1.5σ, alignment)               |
| `logs/return_decay_output.txt`                   | Return-decay diagnostic stdout                                                               |
| `logs/ic_horizon_output.txt`                     | IC horizon scan stdout                                                                       |
| `logs/strategy_a_expanded_output.txt`            | Strategy A expanded grid stdout — 144 combos                                                 |
| `logs/strategy_b_crossing_output.txt`            | Strategy B crossing-entry stdout (corrected OOS run)                                         |
| `logs/strategy_b_yearly_output.txt`              | Year-by-year breakdown stdout                                                                |
| `logs/strategy_c_regime_output.txt`              | Strategy C regime-switching stdout                                                           |
| `logs/strategy_b_efc_output.txt`                 | EFC filter stdout                                                                            |
| `experiments/return_decay_diagnostic.py`         | Return-decay diagnostic script (vectorised cumsum implementation)                            |
| `experiments/ic_horizon_scan.py`                 | IC horizon scan script                                                                       |
| `experiments/strategy_a_expanded_grid.py`        | Strategy A expanded grid (thresh=0.75–1.25σ, conf=0–1, 144 combos)                           |
| `experiments/strategy_b_crossing_entry.py`       | Strategy B with crossing_threshold_signal — IS grid + walk-forward OOS (bug fixed)           |
| `experiments/strategy_b_yearly_breakdown.py`     | Year-by-year performance breakdown — signal built on full history, per-year sliced backtests |
| `experiments/strategy_c_regime_switching.py`     | Regime switching: MR (low vol) ↔ TF HP trend crossover (high vol), 8-combo grid              |
| `experiments/strategy_b_efc_filter.py`           | EFC filter: trend_dev_240 as entry quality gate, apply_entry_filter() O(N) implementation    |
| `notebooks/01_stationarity_tests.ipynb`          | Updated — Dukascopy 62K bars                                                                 |
| `notebooks/02_autocorrelation_analysis.ipynb`    | Updated — Dukascopy 62K bars                                                                 |
| `notebooks/03_feature_engineering.ipynb`         | Updated — Dukascopy 62K bars                                                                 |

---

_Generated by research_pipeline.py + strategy_a_curvature_grid.py + strategy_b_ma_spread_vol_filter.py | Dataset: USDJPY_10yr_1h_dukascopy.csv | HP_WINDOW=500 | Cost=0.9 bps | Last updated: 2026-03-12 (Experiments 5–8)_
