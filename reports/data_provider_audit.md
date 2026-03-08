# Data Provider Audit Report

## Executive Summary

This document details the data sources, validation methodology, and quality assurance procedures for the MA-HP Filter trading system.

**Current Data:** USD/JPY Hourly (USDJPY60.csv)  
**Last Audit Date:** March 8, 2026  
**Data Quality Status:** Under Review

---

## Data Sources

### Primary Data: USD/JPY Hourly

**File:** `data/raw/USDJPY60.csv`  
**Format:** CSV (Date, Time, Open, High, Low, Close, Volume)  
**Timeframe:** October 27, 2025 - March 2, 2026  
**Bars:** 2,105 hourly candles

#### Data Structure

```csv
Date       , Time , Open   , High   , Low    , Close  , Volume
2025.10.27 , 10:00, 153.018, 153.066, 152.735, 152.813, 7367
2025.10.27 , 11:00, 152.814, 152.814, 152.689, 152.744, 5860
...
```

---

## Provider Selection Criteria

### Why This Data Source

1. **Historical Coverage**
   - 4+ months of continuous hourly data
   - Covers multiple market regimes (trending, ranging, volatile)
   - Includes major economic events (Fed meetings, BOJ policy decisions)

2. **Data Quality Requirements**
   - Clean OHLC relationships (high ≥ all, low ≤ all)
   - Minimal gaps (accounting for weekend FX market closures)
   - Realistic spreads and volume patterns
   - No obvious data manipulation or errors

3. **Strategy Suitability**
   - Hourly timeframe matches MA-HP Filter strategy requirements
   - USD/JPY is liquid G7 pair with tight spreads
   - Sufficient volatility for mean-reversion opportunities

---

## Known Issues & Limitations

### 1. Weekend Gaps (Expected)

**Issue:** FX market closed from Friday 22:00 UTC to Sunday 22:00 UTC

**Impact:**

- Natural gaps in hourly time series
- ~48 hours of missing data per week
- **Status:** Normal behavior, handled by forensics module

**Mitigation:**

- Gap detection excludes weekend periods
- Strategy logic does not trade during market close
- Weekend gap filter applied in `detect_missing_bars()`

### 2. Historical Data Completeness

**Issue:** Dataset begins October 27, 2025 (limited history)

**Impact:**

- Only ~4 months of data available
- May not capture full market cycle
- Limited for long-term backtesting (e.g., multi-year)

**Mitigation:**

- Sufficient for proof-of-concept and strategy validation
- Plan to acquire additional historical data if strategy proves viable
- Focus on recent market conditions (more relevant)

### 3. Volume Data Interpretation

**Issue:** Volume in FX is not standardized (varies by aggregator)

**Impact:**

- Volume represents tick count, not actual traded notional
- Cannot compare volume across different data providers
- Less reliable than equity/futures volume

**Mitigation:**

- Use volume as relative indicator only (high/low periods)
- Primary strategy signals based on price, not volume
- Document that volume is indicative, not absolute

### 4. Potential Data Quality Issues (To Verify)

**To Check:**

- OHLC relationship violations (high < low, etc.)
- Missing bars beyond expected weekend gaps
- Statistical outliers (>5 sigma price moves)
- Zero-volume bars (may indicate data gaps)
- Timestamp consistency

**Action:** Run forensics validation to identify actual issues

---

## Validation Methodology

### Automated Quality Checks

The forensics module (`src/data/forensics.py`) performs:

1. **OHLC Relationship Validation**
   - Verify: High ≥ Open, Close, Low
   - Verify: Low ≤ Open, Close, High
   - Flag violations as data errors

2. **Gap Detection**
   - Generate expected hourly timestamp series
   - Exclude weekend market closures
   - Identify unexpected missing bars

3. **Outlier Detection**
   - Calculate return distribution (mean, std)
   - Flag returns >5 standard deviations
   - Review for data errors vs real market events

4. **Spread Analysis**
   - Check for negative spreads (High < Low) → data error
   - Analyze spread distribution for anomalies

5. **Volume Validation**
   - Identify zero-volume bars (may indicate gaps)
   - Check for volume spikes (>10x mean)

### Quality Scoring (0-100)

**Deductions:**

- **OHLC Violations:** -5 points each (max -30)
- **Negative Spreads:** -10 points each (max -30) [Critical]
- **Missing Bars:** -0.5 per 1% missing (max -20)
- **Outliers:** -0.1 each (max -10)
- **Zero Volume:** -0.05 each (max -10)

**Grade Scale:**

- **95-100:** Excellent ✓✓✓ (Production ready)
- **85-94:** Good ✓✓ (Minor issues, usable)
- **70-84:** Acceptable ✓ (Review warnings)
- **50-69:** Poor ⚠ (Use with caution)
- **0-49:** Critical ✗ (Do not use)

---

## Validation Results

### Latest Audit: March 8, 2026

**Status:** Pending validation run

Run validation with:

```bash
python -m src.data.forensics data/raw/USDJPY60.csv
```

Or run tests:

```bash
pytest tests/test_data_forensics.py::test_validate_real_usdjpy_data -v -s
```

#### Expected Results

Based on visual inspection of data:

- **Total Bars:** 2,105
- **Date Range:** 2025-10-27 to 2026-03-02
- **Expected Missing Bars:** ~100-150 (weekends)
- **OHLC Violations:** Expected 0
- **Negative Spreads:** Expected 0

**Action Required:** Run forensics and document actual results here.

---

## Data Provider Comparison (Future)

### Potential Alternative Sources

| Provider                | Pros                               | Cons               | Priority        |
| ----------------------- | ---------------------------------- | ------------------ | --------------- |
| **OANDA API**           | Free, reliable, tradeable spreads  | Requires account   | High            |
| **Dukascopy**           | Swiss ECN data, high quality       | Limited to FX      | Medium          |
| **Interactive Brokers** | Broad coverage, actual broker data | Complex setup      | Medium          |
| **Alpha Vantage**       | Free API                           | Rate limits        | Low             |
| **Yahoo Finance**       | Free                               | FX data unreliable | Not recommended |

### Validation Strategy

If multiple sources available:

1. Download same period from 2+ providers
2. Use `compare_providers()` function (future enhancement)
3. Check correlation, price differences, gap patterns
4. Document any systematic differences

---

## Quality Assurance Procedures

### Pre-Production Checklist

- [ ] Run `python -m src.data.forensics` on all raw data files
- [ ] Review quality score (must be ≥70 for production)
- [ ] Investigate all warnings flagged by forensics module
- [ ] Verify date range covers intended backtest period
- [ ] Check for outliers and cross-reference with news/events
- [ ] Confirm no data leakage (future data in historical file)

### Ongoing Monitoring

**Weekly:**

- Validate any new data downloads
- Check for data staleness (last update date)

**Monthly:**

- Re-run forensics on production data
- Review any quality score degradation
- Document any data source changes

**After Market Events:**

- Validate data quality post-flash crash, circuit breaker
- Check for provider-specific data gaps or errors

---

## Incident Log

### Data Quality Issues

| Date       | Issue                      | Severity | Resolution               |
| ---------- | -------------------------- | -------- | ------------------------ |
| 2026-03-08 | Initial data audit pending | Low      | Run forensics validation |
| -          | -                          | -        | -                        |

_No incidents recorded yet. Update after first validation run._

---

## Recommendations

### Immediate Actions

1. **Run Forensics Validation**

   ```bash
   python -m src.data.forensics data/raw/USDJPY60.csv
   ```

   Document results in this file

2. **Review Test Results**

   ```bash
   pytest tests/test_data_forensics.py -v -s
   ```

   Ensure all tests pass

3. **Address Any Issues**
   - If quality score <70: investigate warnings
   - If OHLC violations found: clean or re-download data
   - If major gaps found: document expected vs unexpected

### Short Term (Next Month)

1. **Expand Data Coverage**
   - Download 1-2 years of USD/JPY hourly data
   - Validate additional pairs (EUR/USD, GBP/USD)

2. **Implement Data Downloads**
   - Create `src/data/download.py` for OANDA API integration
   - Automate daily/weekly data updates

3. **Cross-Validation**
   - Download same period from second source
   - Compare and document discrepancies

### Long Term (Next Quarter)

1. **Real-Time Data Integration**
   - Set up live data feed for paper trading
   - Validate live vs historical data consistency

2. **Multi-Asset Support**
   - Expand to other FX pairs
   - Add indices, commodities if strategy adapts well

3. **Data Versioning**
   - Implement data version control
   - Track when data was downloaded and from where

---

## Appendix A: Running Forensics

### Basic Usage

**Validate single file:**

```bash
python -m src.data.forensics data/raw/USDJPY60.csv
```

**Scan entire directory:**

```bash
python -m src.data.forensics
# Scans data/raw/*.csv by default
```

### Python API

```python
from src.data.forensics import load_csv_data, validate_data_quality, print_quality_report

# Load and validate
df = load_csv_data("data/raw/USDJPY60.csv", has_header=False)
report = validate_data_quality(df, "USDJPY60.csv", expected_freq='H')

# Display results
print_quality_report(report)

# Access specific metrics
print(f"Quality Score: {report.quality_score}/100")
print(f"Missing Bars: {report.missing_bars}")
print(f"OHLC Violations: {report.ohlc_violations}")
```

### Automated Alerts

```python
if report.quality_score < 70:
    print("⚠ WARNING: Data quality below threshold!")
    for warning in report.warnings:
        print(f"  - {warning}")

if report.negative_spreads > 0:
    print("✗ CRITICAL: Negative spreads detected (data corruption)")
```

---

## Appendix B: Data Format Specification

### Expected CSV Format

**Without Header:**

```csv
YYYY.MM.DD,HH:MM,Open,High,Low,Close,Volume
2025.10.27,10:00,153.018,153.066,152.735,152.813,7367
```

**Column Definitions:**

1. **Date:** YYYY.MM.DD format
2. **Time:** HH:MM in 24-hour UTC
3. **Open:** Opening price for the hour
4. **High:** Highest price during the hour
5. **Low:** Lowest price during the hour
6. **Close:** Closing price for the hour
7. **Volume:** Tick volume (number of price changes)

**Data Types:**

- Prices: Float (2-5 decimal places typical for FX)
- Volume: Integer
- Timestamps: Parsed to pandas DatetimeIndex

---

## Sign-Off

**Initial Review:** March 8, 2026  
**Validation Status:** Pending forensics run  
**Approved For Use:** TBD (after quality score ≥70)  
**Next Review Date:** April 8, 2026

---

**Document Version:** 1.0  
**Last Updated:** March 8, 2026
