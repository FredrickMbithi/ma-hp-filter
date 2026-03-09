# Stationarity Analysis Report

**Date:** March 8, 2026  
**Analyst:** MA-HP Filter Development Team

---

## Executive Summary

This document analyzes the stationarity properties of the USD/JPY hourly dataset using Augmented Dickey-Fuller (ADF) and KPSS tests.

Conclusions in this report are derived strictly from the statistical test results shown in the sections below.

### Key Findings

Based on empirical test results:

1. **Prices are non-stationary** (ADF p=0.1172, KPSS p=0.01)
2. **Log returns are stationary** (ADF p=0.0000, KPSS p=0.10)

See Sections 4 and 5 for detailed results.

---

## 1. Dataset Description

| Property         | Value                            |
| ---------------- | -------------------------------- |
| **Instrument**   | USD/JPY                          |
| **Frequency**    | Hourly (H1)                      |
| **Price Type**   | Close (OHLC)                     |
| **Observations** | 2,104                            |
| **Period**       | October 27, 2025 - March 2, 2026 |

The dataset is used to evaluate whether FX prices and returns satisfy the stationarity assumptions required for time-series modeling.

### Note on Sample Size

The dataset contains 2,104 hourly observations. Unit root tests may have limited statistical power in moderate sample sizes and can be sensitive to structural breaks in the time series. Results should therefore be interpreted as indicative rather than definitive.

---

## 2. Methodology

### 2.1 Tests Applied

We use two complementary tests with opposite null hypotheses:

| Test                                         | Null Hypothesis                   | Alternative Hypothesis    | Reject H0 Means          |
| -------------------------------------------- | --------------------------------- | ------------------------- | ------------------------ |
| **ADF** (Augmented Dickey-Fuller)            | Unit root exists (non-stationary) | No unit root (stationary) | Series is stationary     |
| **KPSS** (Kwiatkowski-Phillips-Schmidt-Shin) | Stationary                        | Unit root exists          | Series is non-stationary |

### 2.2 Data Preprocessing

Missing observations are removed and returns are computed using log differences after aligning timestamps.

### 2.3 Interpretation Rules

**Significance level:** $\alpha = 0.05$ (95% confidence)

**Decision rules:**

- **ADF:** Reject H0 (unit root) if p-value < 0.05
- **KPSS:** Reject H0 (stationarity) if p-value < 0.05

**Combined interpretation:**

| ADF Result                    | KPSS Result                   | Conclusion              |
| ----------------------------- | ----------------------------- | ----------------------- |
| Reject H0 (p < 0.05)          | Fail to reject H0 (p >= 0.05) | **Stationary**          |
| Fail to reject H0 (p >= 0.05) | Reject H0 (p < 0.05)          | **Non-stationary**      |
| Both reject                   | Both reject                   | Ambiguous - investigate |
| Both fail to reject           | Both fail to reject           | Ambiguous - investigate |

---

## 3. Empirical Stationarity Results

**Instructions:** Fill in values from notebook output. Do not enter values until tests are executed.

| Series      | ADF Statistic | ADF p-value | Lags | KPSS Statistic | KPSS p-value | Conclusion     |
| ----------- | ------------- | ----------- | ---- | -------------- | ------------ | -------------- |
| Prices      | -2.4930       | 0.1172      | 1    | 1.0165         | 0.01         | Non-stationary |
| Log Returns | -44.0346      | 0.0000      | 0    | 0.0798         | 0.10         | Stationary     |

---

## 4. Results: USD/JPY Prices

### 4.1 ADF Test Results

**Instructions:** Copy exact values from notebook execution.

| Metric        | Value   |
| ------------- | ------- |
| ADF Statistic | -2.4930 |
| P-value       | 0.1172  |
| Lags Used     | 1       |

**Critical Values (reference):**

- 1%: -3.43
- 5%: -2.86
- 10%: -2.57

### 4.2 KPSS Test Results

| Metric         | Value  |
| -------------- | ------ |
| KPSS Statistic | 1.0165 |
| P-value        | 0.01   |

**Critical Values (reference):**

- 1%: 0.739
- 5%: 0.463
- 10%: 0.347

### 4.3 Interpretation (Prices)

**Apply interpretation rules using actual p-values:**

```
ADF p-value = 0.1172 → Fail to reject unit root (p >= 0.05)
KPSS p-value = 0.01 → Reject stationarity (p < 0.05)

Combined conclusion: NON-STATIONARY (both tests agree)
```

---

## 5. Results: USD/JPY Log Returns

### 5.1 ADF Test Results

**Instructions:** Copy exact values from notebook execution.

| Metric        | Value    |
| ------------- | -------- |
| ADF Statistic | -44.0346 |
| P-value       | 0.0000   |
| Lags Used     | 0        |

### 5.2 KPSS Test Results

| Metric         | Value  |
| -------------- | ------ |
| KPSS Statistic | 0.0798 |
| P-value        | 0.10   |

### 5.3 Interpretation (Log Returns)

**Apply interpretation rules using actual p-values:**

```
ADF p-value = 0.0000 → Reject unit root (p < 0.05)
KPSS p-value = 0.10 → Fail to reject stationarity (p >= 0.05)

Combined conclusion: STATIONARY (both tests agree)
```

---

## 6. HP Filter Cycle Stationarity (Next Step)

Because the MA-HP strategy uses the HP cycle component as a trading signal, the stationarity of the cycle component must also be verified.

**Procedure:**

1. Apply HP filter to log prices. Because λ depends on sampling frequency, multiple λ values should be tested (e.g., 100k, 500k, 1M) to evaluate stability of the extracted cycle component.
2. Extract cycle component: $c_t = P_t - \tau_t$
3. Run ADF and KPSS tests on the cycle

**Status:** Data Not Available - Analysis will be performed in the next research notebook.

| Metric                | Value              |
| --------------------- | ------------------ |
| HP Cycle ADF p-value  | Data Not Available |
| HP Cycle KPSS p-value | Data Not Available |
| HP Cycle Stationarity | Data Not Available |

---

## 7. Descriptive Statistics

**Instructions:** Fill from notebook output after execution.

**Note:** Statistics for non-stationary series (e.g., prices) are reported for descriptive purposes only and do not imply stable statistical properties over time.

| Metric   | Prices   | Log Returns |
| -------- | -------- | ----------- |
| Mean     | 155.5698 | 0.000011    |
| Std Dev  | 1.6267   | 0.000957    |
| Min      | 151.6170 | -0.007690   |
| Max      | 159.3270 | 0.007279    |
| Skewness | -0.0618  | -0.5213     |
| Kurtosis | -0.5975  | 7.8385      |

---

## 8. Autocorrelation Diagnostics

Autocorrelation functions (ACF) are examined to evaluate persistence in the time series.

**Expected patterns:**

- **Prices:** Slow ACF decay (indicates non-stationarity)
- **Returns:** Rapid ACF decay toward zero (indicates stationarity)

**Status:** Data Not Available - ACF plots to be generated in notebook execution.

---

## 9. Implications for Modeling

### 9.1 General Guidance

Returns are typically used because they are more likely to satisfy stationarity assumptions required by time-series models.

| Series Type | Stationarity   | Modeling Suitability             | Transformation     |
| ----------- | -------------- | -------------------------------- | ------------------ |
| Raw Prices  | Non-stationary | Not suitable for direct modeling | Convert to returns |
| Log Returns | Stationary     | Suitable for time-series models  | None required      |
| HP Cycle    | To be verified | Verify before use                | None if stationary |

### 9.2 Strategy Considerations

The MA-HP Filter strategy depends on:

1. **HP filter cycle component** - must verify stationarity before use
2. **Returns for backtesting metrics** - verified stationary, safe to use

**Verification status:** Prices and returns tested. HP cycle pending.

---

## 10. Conclusions

Based on ADF and KPSS test results:

1. **Price levels contain a unit root** - Non-stationary (ADF p=0.1172, KPSS p=0.01)
2. **Returns satisfy stationarity assumptions** - Stationary (ADF p=0.0000, KPSS p=0.10)
3. **HP filter cycle stationarity** - To be verified in next notebook

Log returns are suitable for time-series modeling. Raw prices should not be used directly.

---

## 11. Next Steps

1. ~~Execute stationarity notebook and fill in test results~~ (Completed)
2. Verify stationarity of the HP filter cycle component
3. Analyze autocorrelation structure of returns
4. Test for volatility clustering (ARCH effects)
5. Evaluate mean-reversion properties of the cycle component
6. Construct trading signals for the MA-HP strategy

---

## Appendix A: Test Output Details

### Full ADF Test Output (Prices)

```
ADF Statistic: -2.492983
P-value: 0.117191
Lags Used: 1
Observations: 2101
Critical Values:
  1%: -3.43
  5%: -2.86
  10%: -2.57
Conclusion: Fail to reject H0 (unit root present)
```

### Full ADF Test Output (Returns)

```
ADF Statistic: -44.034593
P-value: 0.000000
Lags Used: 0
Observations: 2102
Conclusion: Reject H0 (no unit root, stationary)
```

### Full KPSS Test Output (Prices)

```
KPSS Statistic: 1.016511
P-value: 0.010000
Lags Used: 28
Conclusion: Reject H0 (non-stationary)
```

### Full KPSS Test Output (Returns)

```
KPSS Statistic: 0.079766
P-value: 0.100000
Lags Used: 7
Conclusion: Fail to reject H0 (stationary)
```

---

## Appendix B: Code Reference

```python
from statsmodels.tsa.stattools import adfuller, kpss

def run_stationarity_tests(series, name='Series', alpha=0.05):
    """
    Run ADF and KPSS tests and return results.
    """
    # ADF Test (H0: unit root)
    # maxlag=24 constrains lag search to avoid overfitting
    adf_result = adfuller(series, maxlag=24, autolag='AIC')
    adf_stat, adf_p, adf_lags = adf_result[0], adf_result[1], adf_result[2]

    # KPSS Test (H0: stationary)
    kpss_result = kpss(series, regression='c', nlags='auto')
    kpss_stat, kpss_p = kpss_result[0], kpss_result[1]

    # Apply decision rules
    adf_rejects = adf_p < alpha  # Reject unit root
    kpss_rejects = kpss_p < alpha  # Reject stationarity

    print(f"\n{name}")
    print(f"  ADF Statistic: {adf_stat:.4f}")
    print(f"  ADF p-value: {adf_p:.4f} → {'Reject' if adf_rejects else 'Fail to reject'} unit root")
    print(f"  Lags Used: {adf_lags}")
    print(f"  KPSS Statistic: {kpss_stat:.4f}")
    print(f"  KPSS p-value: {kpss_p:.4f} → {'Reject' if kpss_rejects else 'Fail to reject'} stationarity")

    # Combined conclusion
    if adf_rejects and not kpss_rejects:
        conclusion = "STATIONARY"
    elif not adf_rejects and kpss_rejects:
        conclusion = "NON-STATIONARY"
    else:
        conclusion = "AMBIGUOUS"

    print(f"  Conclusion: {conclusion}")
    return {
        'adf_stat': adf_stat, 'adf_p': adf_p, 'adf_lags': adf_lags,
        'kpss_stat': kpss_stat, 'kpss_p': kpss_p, 'conclusion': conclusion
    }
```

---

**Report Generated:** March 8, 2026  
**Version:** 2.0 (Data-Driven)  
**Status:** Complete - Test results populated from notebook execution

---

**Report Generated:** March 8, 2026  
**Version:** 2.0 (Data-Driven Refactor)  
**Status:** Awaiting notebook execution to populate test results
