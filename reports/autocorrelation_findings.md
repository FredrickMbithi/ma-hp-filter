# Autocorrelation and Volatility Clustering Analysis

**Date:** March 8, 2026  
**Analyst:** MA-HP Filter Development Team

---

## Executive Summary

This report analyzes the autocorrelation structure and volatility dynamics of USD/JPY hourly log returns.

The analysis evaluates two questions:

1. Whether returns exhibit statistically significant serial correlation.
2. Whether volatility shows persistence or clustering.

All conclusions in this report are derived from the empirical tests presented below.

### Key Findings

| Property                          | Finding         | p-value | Implication                       |
| --------------------------------- | --------------- | ------- | --------------------------------- |
| **Returns Serial Correlation**    | Not significant | 0.1735  | Returns approximately white noise |
| **Volatility Clustering (r²)**    | Significant     | 0.0000  | Strong volatility persistence     |
| **Volatility Clustering (\|r\|)** | Significant     | 0.0000  | Confirms clustering               |
| **ARCH Effects**                  | Present         | 0.0000  | Time-varying volatility           |
| **Regime Persistence**            | High (89-93%)   | N/A     | Regimes are sticky                |

---

## 1. Dataset Description

| Property         | Value                            |
| ---------------- | -------------------------------- |
| **Instrument**   | USD/JPY                          |
| **Frequency**    | Hourly (H1)                      |
| **Price Type**   | Close (OHLC)                     |
| **Observations** | 2,103                            |
| **Period**       | October 27, 2025 - March 2, 2026 |

### Data Preprocessing

Returns are computed from cleaned price data with missing timestamps removed before statistical tests are applied.

Log returns: $r_t = \ln(P_t / P_{t-1})$

---

## 2. Methodology

### 2.1 Ljung-Box Q-Test

The Ljung-Box test evaluates whether autocorrelation exists up to lag $m$:

$$Q(m) = n(n+2) \sum_{k=1}^{m} \frac{\hat{\rho}_k^2}{n-k}$$

- **H0:** No autocorrelation up to lag $m$ (white noise)
- **H1:** At least one $\rho_k \neq 0$
- **Decision:** Reject H0 if p-value < 0.05

**Interpretation:**

- If Ljung-Box p-value < 0.05 → Reject white noise → Serial correlation exists
- If Ljung-Box p-value ≥ 0.05 → Cannot reject white noise

### 2.2 ACF Significance Threshold

For sample size $n$, ACF values are considered significant if they exceed the approximate 95% confidence interval:

$$CI = \pm \frac{1.96}{\sqrt{n}}$$

For n = 2,103: **CI = ±0.0427**

### 2.3 ARCH LM Test

Tests whether volatility is time-varying:

- **H0:** No ARCH effects (constant volatility)
- **H1:** ARCH effects present (time-varying volatility)
- **Decision:** Reject H0 if p-value < 0.05

### 2.4 Volatility Regime Classification

Regime thresholds are calculated using rolling historical volatility to avoid lookahead bias:

```
q25 = vol.rolling(window).quantile(0.25)
q75 = vol.rolling(window).quantile(0.75)
```

| Regime | Definition                         |
| ------ | ---------------------------------- |
| Low    | $\sigma_t < Q_{25}$                |
| Medium | $Q_{25} \leq \sigma_t \leq Q_{75}$ |
| High   | $\sigma_t > Q_{75}$                |

### 2.5 Transition Probability Calculation

$$P(i \to j) = \frac{\text{count(transitions from } i \text{ to } j)}{\text{total transitions from } i}$$

---

## 3. Empirical Results Summary

| Test               | Series           | Statistic   | p-value | Conclusion            |
| ------------------ | ---------------- | ----------- | ------- | --------------------- |
| Ljung-Box (lag 20) | Log Returns      | Q = joint   | 0.1735  | No serial correlation |
| Ljung-Box (lag 20) | Squared Returns  | Q = joint   | 0.0000  | Volatility clustering |
| Ljung-Box (lag 20) | Absolute Returns | Q = joint   | 0.0000  | Volatility clustering |
| ARCH LM (lag 10)   | Log Returns      | LM = 121.59 | 0.0000  | ARCH effects present  |

---

## 4. Results: Returns Autocorrelation

### 4.1 Ljung-Box Test on Returns

| Metric        | Value                      |
| ------------- | -------------------------- |
| Series        | USD/JPY Hourly Log Returns |
| Lags Tested   | 20                         |
| Joint p-value | 0.1735                     |

**Interpretation:**

- p-value = 0.1735 ≥ 0.05 → Fail to reject H0
- **Conclusion:** Returns are approximately white noise (no significant serial correlation)

### 4.2 ACF Values (First 10 Lags)

**95% Confidence Interval:** ±0.0427

| Lag | ACF Value | Significant? |
| --- | --------- | ------------ |
| 1   | 0.0398    | No           |
| 2   | -0.0185   | No           |
| 3   | -0.0181   | No           |
| 4   | 0.0164    | No           |
| 5   | 0.0392    | No           |
| 6   | -0.0046   | No           |
| 7   | 0.0436    | Yes          |
| 8   | 0.0220    | No           |
| 9   | 0.0080    | No           |
| 10  | 0.0077    | No           |

**Interpretation:**

- Only 1 of 10 lags exceeds the confidence interval (Lag 7)
- Lag 7 significance is marginal (0.0436 vs threshold 0.0427)
- No consistent pattern suggesting momentum or mean-reversion
- **Conclusion:** Returns show no exploitable serial correlation

---

## 5. Results: Volatility Clustering

### 5.1 Ljung-Box Test on Squared Returns

| Metric           | Value                    |
| ---------------- | ------------------------ |
| Series           | Squared Log Returns (r²) |
| Lags Tested      | 20                       |
| Joint p-value    | 0.0000                   |
| Significant Lags | 20/20                    |

**Interpretation:**

- p-value < 0.05 → Reject H0
- All 20 lags show significant autocorrelation
- **Conclusion:** Strong volatility clustering in squared returns

### 5.2 Ljung-Box Test on Absolute Returns

| Metric           | Value                        |
| ---------------- | ---------------------------- |
| Series           | Absolute Log Returns (\|r\|) |
| Lags Tested      | 20                           |
| Joint p-value    | 0.0000                       |
| Significant Lags | 20/20                        |

**Interpretation:**

- Confirms volatility clustering finding
- Absolute returns often show stronger persistence than squared returns
- **Conclusion:** Robust evidence of volatility clustering

### 5.3 ARCH LM Test

| Metric       | Value    |
| ------------ | -------- |
| LM Statistic | 121.5910 |
| LM p-value   | 0.0000   |
| F Statistic  | 12.8412  |
| F p-value    | 0.0000   |
| Lags         | 10       |

**Interpretation:**

- p-value < 0.05 → Reject H0 (constant volatility)
- **Conclusion:** ARCH effects are present; volatility is time-varying and predictable

---

## 6. Volatility Regime Analysis

### 6.1 Regime Thresholds

Calculated from 20-period rolling standard deviation:

| Threshold      | Value      |
| -------------- | ---------- |
| Low Vol (Q25)  | < 0.000671 |
| High Vol (Q75) | > 0.001012 |

### 6.2 Regime Distribution

| Regime | Count | Percentage |
| ------ | ----- | ---------- |
| Low    | 521   | 25.0%      |
| Medium | 1,042 | 50.0%      |
| High   | 521   | 25.0%      |

### 6.3 Returns by Regime

**Note:** Statistics for non-stationary series are reported for descriptive purposes only.

| Regime | Count | Mean      | Std Dev  | Min | Max      |
| ------ | ----- | --------- | -------- | --- | -------- |
| Low    | 521   | 0.000008  | 0.000577 | -   | 0.001718 |
| Medium | 1,042 | 0.000032  | 0.000843 | -   | 0.002956 |
| High   | 521   | -0.000021 | 0.001385 | -   | 0.007279 |

**Observations:**

- Standard deviation increases with regime (as expected)
- High-vol regime shows slightly negative mean return
- Extreme values (max) concentrated in high-vol periods

### 6.4 Transition Matrix

Probability of transitioning from row regime to column regime (in %):

| From / To  | High  | Low   | Medium |
| ---------- | ----- | ----- | ------ |
| **High**   | 93.09 | 0.19  | 6.72   |
| **Low**    | 0.38  | 89.44 | 10.17  |
| **Medium** | 3.27  | 5.19  | 91.55  |

**Interpretation:**

- Diagonal values (89-93%) indicate strong regime persistence
- Regimes are "sticky" - once in a regime, likely to stay
- Direct high↔low transitions are rare (<0.5%)
- Confirms volatility clustering at regime level

### 6.5 Volatility Persistence (GARCH)

**Status:** Data Not Available - GARCH model estimation pending.

Half-life of volatility shocks will be computed as:

$$\text{half-life} = \frac{\ln(0.5)}{\ln(\alpha + \beta)}$$

Where $\alpha$ and $\beta$ are GARCH(1,1) parameters.

---

## 7. Trading Implications

### 7.1 Returns Predictability

**Finding:** Returns show no significant serial correlation (Ljung-Box p=0.1735)

**Implications:**

- Market is approximately weak-form efficient at hourly frequency
- Pure momentum or reversal strategies unlikely to be profitable
- Focus on other sources of edge (volatility, carry, fundamentals)

### 7.2 Volatility Management

**Finding:** Strong ARCH effects present (p=0.0000)

**Implications:**

- Volatility is predictable
- Dynamic position sizing is justified
- Regime-based filters may improve risk-adjusted returns

### 7.3 Position Sizing

Volatility-targeted position sizing:

$$\text{position} = \text{signal} \times \frac{\text{target\_vol}}{\text{current\_vol}}$$

With position limits applied afterward:

```python
position = signal * (target_vol / current_vol)
position = np.clip(position, -max_position, max_position)
```

**Note:** Regime-based multipliers will be determined during strategy calibration.

### 7.4 For MA-HP Filter Strategy

Based on findings:

1. **HP filter cycle extraction** - Justified (creates stationary signal from non-stationary prices)
2. **Returns-based backtesting** - Safe (returns are stationary, approximately white noise)
3. **Volatility-adjusted sizing** - Recommended (ARCH effects confirmed)
4. **Regime filtering** - Consider (high regime persistence observed)

---

## 8. Conclusions

### 8.1 Returns Characteristics

| Test                 | Result          | Conclusion                |
| -------------------- | --------------- | ------------------------- |
| Ljung-Box (returns)  | p = 0.1735      | No serial correlation     |
| Significant ACF lags | 1/10 (marginal) | Approximately white noise |

**Overall:** USD/JPY hourly returns are unpredictable (weak-form efficient).

### 8.2 Volatility Characteristics

| Test               | Result                 | Conclusion           |
| ------------------ | ---------------------- | -------------------- |
| Ljung-Box (r²)     | p = 0.0000, 20/20 lags | Strong clustering    |
| Ljung-Box (\|r\|)  | p = 0.0000, 20/20 lags | Confirms clustering  |
| ARCH LM            | p = 0.0000             | ARCH effects present |
| Regime persistence | 89-93% diagonal        | Regimes are sticky   |

**Overall:** USD/JPY volatility shows strong clustering and regime persistence.

### 8.3 Recommendations

1. **Implement volatility-adjusted position sizing** (ARCH effects confirmed)
2. **Use rolling windows for regime thresholds** (avoid lookahead bias)
3. **Consider regime-based filters** (high persistence supports regime strategies)

---

## 9. Next Steps

1. ~~Run autocorrelation tests~~ (Completed)
2. Estimate GARCH(1,1) model for volatility persistence half-life
3. Backtest MA-HP filter with fixed vs. vol-adjusted sizing
4. Calibrate regime multipliers through optimization
5. Implement volatility targeting in position sizing module

---

## Appendix A: Test Output Details

### Ljung-Box Test: Log Returns

```
Series: USD/JPY Hourly Log Returns
Lags: 20
Joint p-value: 0.173467
Conclusion: Fail to reject H0 (no serial correlation)
```

### Ljung-Box Test: Squared Returns

```
Series: Squared Log Returns
Lags: 20
Joint p-value: 0.000000
Significant lags: 20/20
Conclusion: Reject H0 (volatility clustering present)
```

### Ljung-Box Test: Absolute Returns

```
Series: Absolute Log Returns
Lags: 20
Joint p-value: 0.000000
Significant lags: 20/20
Conclusion: Reject H0 (volatility clustering present)
```

### ARCH LM Test

```
Series: USD/JPY Hourly Log Returns
Lags: 10
LM Statistic: 121.590954
LM P-value: 0.000000
F Statistic: 12.841189
F P-value: 0.000000
Conclusion: ARCH effects present (reject H0)
```

---

## Appendix B: Code Reference

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch

class VolatilityAnalyzer:
    """
    Analyze autocorrelation and volatility clustering.
    """

    def __init__(self, returns):
        self.returns = returns
        self.squared_returns = returns ** 2
        self.abs_returns = returns.abs()
        self.n = len(returns)

    def compute_acf(self, lags=20):
        """Compute ACF values."""
        return acf(self.returns, nlags=lags)

    def acf_confidence_interval(self):
        """Compute 95% CI for ACF."""
        return 1.96 / np.sqrt(self.n)

    def ljung_box_test(self, series, lags=20):
        """Run Ljung-Box test."""
        result = acorr_ljungbox(series, lags=lags, return_df=True)
        return {
            'joint_pvalue': result['lb_pvalue'].iloc[-1],
            'significant_lags': (result['lb_pvalue'] < 0.05).sum(),
            'total_lags': lags
        }

    def arch_test(self, lags=10):
        """Run ARCH LM test."""
        lm_stat, lm_pvalue, f_stat, f_pvalue = het_arch(self.returns, nlags=lags)
        return {
            'lm_statistic': lm_stat,
            'lm_pvalue': lm_pvalue,
            'f_statistic': f_stat,
            'f_pvalue': f_pvalue,
            'arch_present': lm_pvalue < 0.05
        }

    def calculate_rolling_vol(self, window=20):
        """Calculate rolling volatility."""
        return self.returns.rolling(window=window).std()

    def identify_regimes(self, vol):
        """
        Classify volatility into regimes using rolling thresholds.
        """
        q25 = vol.rolling(500, min_periods=20).quantile(0.25)
        q75 = vol.rolling(500, min_periods=20).quantile(0.75)

        regimes = pd.Series('medium', index=vol.index)
        regimes[vol < q25] = 'low'
        regimes[vol > q75] = 'high'

        return regimes

    def compute_transition_matrix(self, regimes):
        """
        Compute transition probabilities.
        P(i -> j) = count(i to j) / total from i
        """
        return pd.crosstab(
            regimes[:-1].values,
            regimes[1:].values,
            normalize='index'
        )
```

---

**Report Generated:** March 8, 2026  
**Version:** 2.0 (Data-Driven)  
**Status:** Complete - Test results populated from analysis
