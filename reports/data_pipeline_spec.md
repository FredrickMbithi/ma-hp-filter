# Data Pipeline Specification

## Overview

Production-grade FX H1 data pipeline for the MA + HP Filter trading strategy. Ensures data integrity, prevents lookahead bias, and provides consistent validation across all components.

---

## Data Flow

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌─────────────┐
│  data/raw/  │───▶│  FXDataLoader │───▶│ Validation  │───▶│ data/       │
│  *.csv      │    │  load()       │    │ Suite       │    │ processed/  │
└─────────────┘    └──────────────┘    └─────────────┘    └─────────────┘
                          │
                          ▼
                   ┌──────────────┐
                   │ data/interim/│
                   │ (cache)      │
                   └──────────────┘
```

### Stages

| Stage         | Location          | Purpose                                |
| ------------- | ----------------- | -------------------------------------- |
| **Raw**       | `data/raw/`       | Unmodified broker exports (CSV)        |
| **Interim**   | `data/interim/`   | Validated, normalized (optional cache) |
| **Processed** | `data/processed/` | Final datasets with features           |

---

## Dataset Contract

### H1 FX Data Requirements

| Property           | Requirement                                  |
| ------------------ | -------------------------------------------- |
| **Frequency**      | 1 hour (H1)                                  |
| **Timezone**       | UTC (enforced)                               |
| **Index**          | DatetimeIndex, monotonically increasing      |
| **Duplicates**     | None (first kept if found)                   |
| **OHLC Integrity** | `low ≤ open ≤ high` and `low ≤ close ≤ high` |

### Column Schema

| Column   | Type          | Description   |
| -------- | ------------- | ------------- |
| `open`   | float64       | Opening price |
| `high`   | float64       | Period high   |
| `low`    | float64       | Period low    |
| `close`  | float64       | Closing price |
| `volume` | int64/float64 | Tick volume   |

---

## Validation Rules

### Critical Checks (Fail on Violation)

| Check              | Function                 | Pass Criteria                     |
| ------------------ | ------------------------ | --------------------------------- |
| **Timezone**       | `check_timezone()`       | Index is UTC-localized            |
| **Duplicates**     | `check_duplicates()`     | No duplicate timestamps           |
| **Monotonic**      | `check_monotonic()`      | Index strictly increasing         |
| **OHLC Integrity** | `check_ohlc_integrity()` | All bars satisfy OHLC constraints |

### Warning Checks (Log but Continue)

| Check              | Function                 | Pass Criteria                      |
| ------------------ | ------------------------ | ---------------------------------- |
| **Missing Bars**   | `check_missing_bars()`   | No weekday gaps (weekends allowed) |
| **Extreme Spikes** | `check_extreme_spikes()` | No returns > 5σ (flags for review) |

### OHLC Constraint Details

```
For each bar:
  low ≤ open ≤ high    ✓
  low ≤ close ≤ high   ✓
```

Violations indicate:

- Data corruption
- Bad tick aggregation
- Invalid broker feed

---

## Missing Data Handling

### Gap Policy

**Rule**: Forward-fill preserves price continuity without inventing volatility.

| Field    | Fill Method    | Rationale                             |
| -------- | -------------- | ------------------------------------- |
| `close`  | Forward fill   | Maintains last known price            |
| `open`   | Previous close | No gap jump at bar open               |
| `high`   | Previous close | Conservative (no invented volatility) |
| `low`    | Previous close | Conservative (no invented volatility) |
| `volume` | 0              | No activity during gap                |

### Allowed Gaps

| Gap Type          | Handling                    |
| ----------------- | --------------------------- |
| Weekend (Sat/Sun) | Ignored (normal FX closure) |
| US/UK holidays    | Flagged but allowed         |
| Weekday gaps      | Error - investigate source  |

### Implementation

```python
from src.data import FXDataLoader

loader = FXDataLoader('data/raw')
df = loader.load('USDJPY_10yr_1h')
df_filled = loader.fill_gaps(df, method='ffill')
```

---

## Lookahead Bias Prevention

### The Problem

Lookahead bias occurs when future information leaks into training data, causing:

- Overfitted models that fail live
- Unrealistic backtests
- False confidence in strategies

### Prevention Rules

| Component            | Rule                                            |
| -------------------- | ----------------------------------------------- |
| **Features**         | Use only past data: `.shift(1)` or `.rolling()` |
| **Targets**          | Use future data: `.shift(-1)`                   |
| **Train/Test Split** | Split BEFORE computing rolling statistics       |
| **Cross-Validation** | Time-series aware (no random shuffle)           |

### Correct Feature Engineering

```python
# CORRECT: Features use past data
df['log_return'] = np.log(df['close'] / df['close'].shift(1))  # t vs t-1
df['ma20'] = df['close'].rolling(20).mean()                     # t-19 to t

# CORRECT: Target uses future data
df['target'] = df['log_return'].shift(-1)                       # t+1 return
```

### Correct Train/Test Split

```python
from src.data import safe_train_test_split

# Split FIRST
train, test = safe_train_test_split(df, test_ratio=0.2, gap_bars=10)

# THEN compute features (separately for each set)
train['ma20'] = train['close'].rolling(20).mean()
test['ma20'] = test['close'].rolling(20).mean()
```

### Anti-Patterns (AVOID)

```python
# WRONG: Computing features before split leaks test info into rolling windows
df['ma20'] = df['close'].rolling(20).mean()  # Uses entire dataset!
train, test = train_test_split(df, ...)       # Too late

# WRONG: Random split breaks time ordering
train, test = sklearn.train_test_split(df, shuffle=True)  # Never shuffle!
```

---

## Usage Examples

### Basic Loading

```python
from src.data import FXDataLoader

loader = FXDataLoader('data/raw')
df = loader.load('USDJPY_10yr_1h')

# df has:
# - UTC DatetimeIndex
# - Sorted, no duplicates
# - Validated OHLC
```

### With Validation Report

```python
df, results = loader.load('USDJPY_10yr_1h', return_report=True)

for r in results:
    print(f"{r.check_name}: {'✓' if r.is_valid else '✗'} - {r.message}")
    if r.warnings:
        for w in r.warnings:
            print(f"  ⚠ {w}")
```

### Full Pipeline

```python
from src.data import FXDataLoader, safe_train_test_split, create_target_variable
import numpy as np

# 1. Load validated data
loader = FXDataLoader('data/raw')
df = loader.load('USDJPY_10yr_1h')

# 2. Fill any gaps
df = loader.fill_gaps(df, method='ffill')

# 3. Split BEFORE features
train, test = safe_train_test_split(df, test_ratio=0.2, gap_bars=20)

# 4. Create features on train only
train['ma20'] = train['close'].rolling(20).mean()
train['ma50'] = train['close'].rolling(50).mean()
train['log_return'] = np.log(train['close'] / train['close'].shift(1))

# 5. Create target (uses future data - intentional)
train['target'] = create_target_variable(train, horizon=1)

# 6. Drop NaN from feature warmup
train = train.dropna()
```

---

## Configuration

Settings in `config/config.yaml`:

```yaml
data:
  symbol: "USDJPY"
  timeframe: "H1"
  raw_path: "data/raw"
  interim_path: "data/interim"
  processed_path: "data/processed"
  spike_threshold: 5.0 # σ for outlier detection
```

---

## Testing

Run validation tests:

```bash
pytest tests/test_data_loader.py -v
```

Key test categories:

- `TestTimestampNormalization` - UTC enforcement
- `TestDuplicateDetection` - Duplicate handling
- `TestOHLCIntegrity` - Structural validation
- `TestGapDetection` - Missing bar detection
- `TestSpikeDetection` - Outlier flagging
- `TestNoLookahead` - **Critical** bias prevention tests
- `TestGapFilling` - Fill policy verification

---

## Changelog

| Date       | Change                        |
| ---------- | ----------------------------- |
| 2026-03-09 | Initial pipeline spec created |
