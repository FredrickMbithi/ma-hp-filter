# Final Feature Selection — USDJPY H1

**Date:** 2026-03-11  
**Horizon:** H24 (24-bar forward IC)  
**Universe:** USDJPY H1, 2016-03-13 → 2026-03-10 (≈62,303 bars)

---

## Selection Criteria

A feature passes selection if **all** of the following hold:

| Criterion              | Threshold                 | Rationale                                                |
| ---------------------- | ------------------------- | -------------------------------------------------------- |
| IC                     | `> 0.03`                  | Minimum meaningful predictive signal in FX               |
| t-statistic            | `> 2.0`                   | IC is statistically distinguishable from noise           |
| Stationarity (ADF)     | `p < 0.05`                | Non-stationary features produce spurious correlations    |
| Walk-forward stability | `sign_consistency ≥ 0.70` | IC sign must be consistent across 70%+ of yearly periods |
| Redundancy             | `max pair corr < 0.70`    | Not redundant with a higher-IC feature already selected  |

> **Note:** Cross-pair generalization (IC across 3+ FX pairs) is not yet tested — requires `cross_validation.py` (planned).

---

## Feature Scores

_Results from notebooks 06–08, 2026-03-11. IC measured at H=24 (24-bar forward log-return). CATALOG peak IC shown for reference (best horizon across 1/4/12/24/48 bars)._

| Feature                   | \|IC\| H=24 | t-stat | Sign Consistency | Max Pair Corr | CATALOG peak IC | Status                                       |
| ------------------------- | ----------- | ------ | ---------------- | ------------- | --------------- | -------------------------------------------- |
| `ma_crossover_signal`     | 0.011       | 2.7    | 64%              | 0.862         | 0.397           | ❌ FAIL (IC<0.03, stability<0.70, corr>0.70) |
| `trend_deviation_from_ma` | 0.000       | 0.0    | 55%              | 0.971         | 0.344           | ❌ FAIL (IC≈0, all criteria)                 |
| `ma_spread_on_trend`      | 0.004       | 1.0    | 64%              | 0.971         | 0.318           | ❌ FAIL (IC<0.03, stability<0.70, corr>0.70) |
| `trend_strength`          | 0.001       | 0.2    | 36%              | 0.169         | —               | ❌ FAIL (IC≈0, stability<0.70)               |
| `vol_zscore`              | —           | —      | —                | 0.692         | —               | ⚠️ NOT TESTED (no IC, below corr threshold)  |
| `hp_trend_level`          | —           | —      | —                | 0.692         | 0.397           | ⚠️ NOT TESTED directly                       |

---

## Passed Features

**0 out of 4 tested features pass all criteria at H=24.**

No feature passes the combined IC > 0.03 + sign_consistency ≥ 0.70 bar at this horizon.

---

## Failed Features

| Feature                   | Failing Criteria                                                       |
| ------------------------- | ---------------------------------------------------------------------- |
| `ma_crossover_signal`     | IC < 0.03 (0.011), sign_consistency=64% (<0.70), max corr=0.86 (>0.70) |
| `trend_deviation_from_ma` | IC ≈ 0.000, sign_consistency=55%, max corr=0.97                        |
| `ma_spread_on_trend`      | IC < 0.03 (0.004), sign_consistency=64%, max corr=0.97                 |
| `trend_strength`          | IC ≈ 0.001, sign_consistency=36% (worst of all)                        |

---

## Redundancy Summary

**3 redundant pairs found (Spearman |ρ| > 0.70) — 2026-03-11:**

| Pair                                              | ρ      | Dropped                   |
| ------------------------------------------------- | ------ | ------------------------- |
| `ma_spread_on_trend` ↔ `trend_deviation_from_ma`  | +0.971 | `ma_spread_on_trend`      |
| `ma_spread_on_trend` ↔ `ma_crossover_signal`      | +0.862 | `ma_spread_on_trend`      |
| `trend_deviation_from_ma` ↔ `ma_crossover_signal` | +0.835 | `trend_deviation_from_ma` |

**Greedy selection result** (non-redundant subset, ranked by CATALOG peak IC):

- KEPT: `ma_crossover_signal`, `hp_trend_level`, `trend_strength`, `vol_zscore`
- DROPPED: `trend_deviation_from_ma`, `ma_spread_on_trend`

Key observation: all three core mean-reversion features (`ma_spread`, `trend_deviation`, `ma_crossover`) measure essentially the same information — distance of price from HP trend. The correlations (0.83–0.97) leave no diversification value from combining them.

`hp_trend_level ↔ vol_zscore` correlation = −0.692, just below the 0.70 threshold.

---

## Recommendations

All four tested features fail at H=24. The root cause is **sign instability across years** (IC flips from negative to positive in some years). This is the primary risk for live trading.

**Immediate next steps:**

1. **Re-test at the peak horizon.** The CATALOG peak ICs (0.3–0.4) were at a different horizon than H=24. Run `notebooks/04_univariate_feature_tests.ipynb` and inspect `peak_horizon` per feature. The actual horizon may be H=1, H=4, or even H=48.
2. **Investigate the IS Sharpe discrepancy.** Prior research claimed IS Sharpe ≈ 3.7 for `ma_crossover_signal` (inverted). The causal HP filter run gives IS Sharpe = 0.54. Check whether prior research used the two-sided `apply_hp_filter` (lookahead bias) or different HP parameters.
3. **Use `vol_regime` as a filter,** not a feature — restrict to Medium-Vol regime (regime=1) where prior research showed peak IC.
4. **Do not combine `ma_spread`, `trend_deviation`, and `ma_crossover`** — they are 83–97% correlated. Use only `ma_crossover_signal` as the single representative.
5. **`trend_strength` should be removed** — sign_consistency=36% is the worst of all features and indicates a noisy regime indicator with no directional edge.
