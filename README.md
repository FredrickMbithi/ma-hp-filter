# MA + HP Filter FX Research Pipeline

A systematic quantitative research framework for FX trading strategy development, combining Moving Averages (MA) and Hodrick-Prescott (HP) filters for trend detection and mean-reversion opportunities on USDJPY H1 data.

## 🎯 Project Overview

This repository implements **Strategy 8.1**: a 5-step research pipeline that transforms hypothesis → feature engineering → validation → backtesting → production-ready strategy.

**Core Hypothesis:** HP filter-based trend decomposition combined with MA crossovers can identify profitable mean-reversion and trend-following entries in FX markets.

## 📊 Research Pipeline (5 Steps)

1. **Stationarity Testing** — ADF tests on 10 feature generators
2. **Information Coefficient Analysis** — Spearman IC tests at 1-bar, 4-bar, 24-bar horizons
3. **Lambda Calibration** — HP filter smoothing parameter optimization
4. **Grid Search** — 72-combination parameter sweep with Deflated Sharpe Ratio
5. **Regime-Conditional Analysis** — Volatility regime detection + regime-specific IC/backtest metrics

All executed via:
```bash
python research_pipeline.py
```

## 🏗️ Architecture

```
ma-hp-filter/
├── config/
│   └── config.yaml              # Data paths, backtest params, HP lambdas
├── docs/
│   ├── backtest_spec.md         # Execution model, signal conventions
│   └── environment_setup.md     # Python environment instructions
├── experiments/
│   ├── feature_hypothesis_log.md      # Full research log
│   ├── strategy_a_*.py                # Curvature-based strategies
│   ├── strategy_b_*.py                # MA crossing strategies
│   └── strategy_c_regime_switching.py # Adaptive regime strategies
├── notebooks/
│   ├── 01_stationarity_tests.ipynb
│   ├── 02_autocorrelation_analysis.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── ...                      # 9 analysis notebooks
├── reports/
│   ├── feature_test_results_usdjpy.md
│   ├── research_progress_report_2026-03-10.md
│   └── *.png                    # Performance charts
├── src/
│   ├── backtest/                # Strategy-agnostic backtest engine
│   ├── data/                    # FXDataLoader, MetaTrader bridge
│   ├── features/                # 10 feature generators
│   ├── strategies/              # Signal generation logic
│   ├── risk/                    # Position sizing, stop-loss
│   ├── execution/               # MT5 trade execution
│   └── utils/                   # Logging, metrics
└── research_pipeline.py         # Main orchestrator
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Key Dependencies:**
- `pandas`, `numpy` — Data manipulation
- `scipy`, `statsmodels` — Statistical tests (ADF, Spearman IC)
- `MetaTrader5` — Live data feed + execution
- `pyyaml` — Config management

### 2. Configuration

Edit `config/config.yaml`:

```yaml
data:
  symbol: "USDJPY"
  timeframe: "H1"
  start_date: "2010-01-01"
  end_date: "2024-12-31"

backtest:
  initial_capital: 100000
  spread_pips: 1.2
  commission_per_lot: 7.0
```

### 3. Run Full Pipeline

```bash
python research_pipeline.py > output.log 2>&1
```

This executes all 5 steps and saves:
- IC metrics to `data/interim/ic_*.json`
- Optimized HP lambdas to `data/interim/hp_lambdas.pkl`
- Grid search results to console (capture via `> output.log`)

### 4. Explore Results

Notebooks are pre-populated with analysis:

```bash
jupyter notebook notebooks/01_stationarity_tests.ipynb
```

Key reports:
- `reports/feature_test_results_usdjpy.md` — IC values per feature
- `reports/research_progress_report_2026-03-10.md` — Latest findings
- `reports/multi_factor_equity_curves.png` — Visual performance

## 📈 Feature Generators (10 Total)

| Feature | Description | Use Case |
|---------|-------------|----------|
| `causal_hp_trend` | HP filter trend component | Regime detection |
| `hp_trend_slope` | 1st derivative of HP trend | Momentum signal |
| `hp_trend_curvature` | 2nd derivative (concavity) | Mean-reversion entry |
| `ma_crossover_signal` | Fast MA - Slow MA | Trend direction |
| `ma_crossover_age` | Bars since last crossover | Exhaustion timing |
| `trend_deviation_from_ma` | Price distance from MA | Overbought/oversold |
| `ma_spread_on_trend` | MA deviation from HP trend | Divergence filter |
| `vol_regime` | ATR-based volatility state | Risk adjustment |
| `atr` | Average True Range | Stop-loss sizing |
| `sma` | Simple Moving Average | Baseline reference |

All features are **look-ahead bias free** (only use past data).

## 🎓 Research Methodology

### Signal Execution Timing

**Critical Rule:** Signals generated at bar T can only fill at bar T+1.

```python
# In backtest engine
positions = signal.shift(1).fillna(0)  # Enforced lag
```

This prevents lookahead bias — the #1 source of fake alpha.

### IC Testing Framework

Information Coefficient = Spearman rank correlation between feature and forward returns.

```python
ic_1bar = feature.corr(returns_1bar, method='spearman')  # Predictive power
```

**Acceptance Criteria:**
- |IC| > 0.03
- t-stat > 2.0 (95% confidence)
- p-value < 0.05

### Regime Detection

3-state HMM using ATR + HP trend slope:

| Regime | Condition | Strategy |
|--------|-----------|----------|
| Trending | High |slope|, normal vol | Momentum |
| Mean-Rev | Low |slope|, normal vol | Counter-trend |
| High Vol | Elevated ATR | Reduce size |

## 🔬 Experiment Scripts

Standalone experiments in `experiments/`:

```bash
# Test IC sensitivity to different return horizons
python experiments/ic_horizon_scan.py

# Measure IC decay over time
python experiments/return_decay_diagnostic.py

# Strategy B: Crossing-based entry rules
python experiments/strategy_b_crossing_entry.py

# Regime-adaptive strategy
python experiments/strategy_c_regime_switching.py
```

Each script is self-contained with its own grid search + backtest.

## 📊 Backtest Engine

**Design Principle:** Strategy-agnostic. Engine has zero knowledge of indicators.

```python
from src.backtest.engine import BacktestEngine

engine = BacktestEngine(
    data=ohlc_df,
    signal=signal_series,  # +1 long, -1 short, 0 flat
    spread_pips=1.2,
    commission=7.0
)
results = engine.run()
print(results['sharpe'])  # Sharpe ratio
print(results['max_drawdown'])  # Worst peak-to-trough
```

**Outputs:**
- Sharpe Ratio (annualized)
- Max Drawdown
- Win Rate
- Profit Factor
- Trade count

See `docs/backtest_spec.md` for full specification.

## 🛠️ Production Integration

### MT5 Live Trading

```python
from src.execution.mt5_adapter import MT5Executor

executor = MT5Executor(
    symbol="USDJPY",
    magic_number=12345,
    lot_size=0.1
)

# Send signal to MT5
executor.execute_signal(signal=+1)  # Opens long position
```

### Risk Management

```python
from src.risk.position_sizer import VolatilityScaler

sizer = VolatilityScaler(
    target_vol=0.10,  # 10% annual vol target
    lookback=20
)

position_size = sizer.calculate(atr=atr_value, account_balance=100000)
```

## 📝 Key Reports

1. **Stationarity Analysis** (`reports/stationarity_analysis.md`)
   - ADF test results for all features
   - Differencing transformations

2. **Feature Test Results** (`reports/feature_test_results_usdjpy.md`)
   - IC values per feature per horizon
   - Statistical significance

3. **Autocorrelation Findings** (`reports/autocorrelation_findings.md`)
   - Return predictability decay
   - Optimal signal holding periods

4. **Progress Report 2026-03-10** (`reports/research_progress_report_2026-03-10.md`)
   - Latest strategy performance
   - Next steps

## 🔧 Configuration Files

### `config/config.yaml`

```yaml
hp_filter:
  lambda_h1: 129600  # Smoothing for H1 data (= 1600 × 81)
  lambda_d1: 1600    # Daily equivalent

cost_model:
  spread_pips: 1.2
  commission_per_lot: 7.0
  slippage_pips: 0.5

strategy:
  ma_fast: 20
  ma_slow: 50
  hp_lambda: 129600
```

## 🧪 Testing

Run unit tests:

```bash
pytest src/  # (if test files exist)
```

Validate data pipeline:

```python
from src.data.loader import FXDataLoader

loader = FXDataLoader("USDJPY", "H1")
df = loader.load("2020-01-01", "2023-12-31")
loader.validate(df)  # Checks for gaps, duplicates
```

## 📚 Documentation

- [Backtest Specification](docs/backtest_spec.md) — Execution model, signal conventions
- [Environment Setup](docs/environment_setup.md) — Python dependencies, MT5 setup
- [Feature Hypothesis Log](experiments/feature_hypothesis_log.md) — Full research journey

## 🤝 Contributing

This is a research repository. To extend:

1. Add new feature generators in `src/features/generators.py`
2. Register in `src/features/library.py`
3. Run `python research_pipeline.py` to test IC
4. Document findings in `experiments/feature_hypothesis_log.md`

## ⚠️ Disclaimer

This code is for **research purposes only**. No warranty is provided. Live trading involves risk of capital loss. Always paper-trade first.

## 📄 License

MIT License (or specify your license)

## 📧 Contact

For questions: [Your GitHub Profile](https://github.com/FredrickMbithi)

---

**Last Updated:** March 2026  
**Status:** Active Research  
**Primary Pair:** USDJPY H1  
**Framework:** Pandas + NumPy + SciPy + MetaTrader5
