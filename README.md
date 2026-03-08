# MA + HP Filter Quant System

Counter-trend FX trading system combining Moving Averages and Hodrick-Prescott filtering for cycle extraction.

## Architecture

```
├── data/           # Raw, interim, processed datasets
├── src/            # Core engine
│   ├── data/       # Ingestion, cleaning
│   ├── features/   # MA, HP filter, indicators
│   ├── backtest/   # Vectorized backtesting
│   ├── strategies/ # Signal generation logic
│   ├── risk/       # Position sizing, drawdown limits
│   ├── execution/  # Order management
│   ├── portfolio/  # Multi-asset allocation
│   └── utils/      # Helpers, logging
├── notebooks/      # Research, exploratory analysis
├── experiments/    # One-off tests, parameter scans
├── reports/        # Performance reports, tearsheets
├── state/          # Pickled models, calibration
├── logs/           # Runtime logs
├── tests/          # Unit/integration tests
├── services/       # MT5 bridge, cTrader, monitoring
└── config/         # YAML configuration
```

## Strategy

**Core Hypothesis:** Price oscillates around a slowly-moving trend (HP trend). When price deviates significantly from this trend (measured via z-score or percentile), mean reversion occurs.

**Components:**
1. **HP Filter** extracts smooth trend from noisy price series
2. **Cycle Component** = Price - HP Trend → oscillating signal
3. **MA Envelope** for additional confirmation/regime detection
4. **Entry:** Price far from trend (e.g., >2σ)
5. **Exit:** Price returns to trend or hits stop

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
make test

# Backtest default config
make backtest

# Launch research notebook
make notebook
```

## Configuration

Edit `config/config.yaml`:
- HP filter lambda (smoothness parameter)
- MA periods and envelope width
- Entry/exit thresholds
- Risk parameters (max position, stop %, drawdown limit)

## Development

**Add a new feature:**
1. Write transform in `src/features/`
2. Add unit test in `tests/`
3. Integrate in `src/strategies/`

**Run backtest:**
```python
from src.backtest.engine import BacktestEngine
from src.strategies.ma_hp import MAHPStrategy

engine = BacktestEngine(strategy=MAHPStrategy(), config="config/config.yaml")
results = engine.run()
print(results.summary())
```

## Services

- **MT5 Bridge:** Live data feed + order execution
- **cTrader:** Alternative broker integration
- **Monitoring:** Prometheus metrics, alerting

## Notes

- HP filter requires sufficient history (~200+ bars for stability)
- Lambda ≈ 1600 for daily, ~14400 for hourly data
- Backtest on 2+ years, walk-forward validate

---

**Author:** Ghost 👻  
**Created:** 2026-03-08  
**Status:** Research phase
