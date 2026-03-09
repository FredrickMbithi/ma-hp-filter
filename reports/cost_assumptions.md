# Cost Assumptions

This document records the transaction cost assumptions used in backtesting.
These numbers directly determine whether a strategy is viable.
Overstating costs is safer than understating them.

---

## 1. Cost Components

### 1.1 Spread Cost
The broker's bid/ask spread paid on every fill.  For ECN/STP brokers the spread
is variable and tightest during London/New York overlap (13:00–17:00 UTC).
Widen by 2–3× during Asian session and news events.

### 1.2 Slippage
Execution deviation from the quoted mid-price at signal time.  Sources:

- **Latency** – time between signal generation and order receipt at broker.
- **Quote movement** – price moves between signal and fill.
- **Partial fills** – in low-liquidity windows, the full size may not fill
  at one price.

For retail ECN with VPS co-location: ~0.2–0.5 pip slippage on majors.
Without co-location (home network): 0.5–1.5 pips.

### 1.3 Market Impact
Price movement caused by the order itself.  **Negligible at retail/small-prop
sizing** (< 10 lots per trade on majors).  Relevant only above ~50 lots.
Not modelled in the current `FXCostModel`.

### 1.4 Rollover / Swap
Interest rate differential applied when a position is held past 17:00 Eastern
(the FX "end of day").  Can be positive (carry earned) or negative (carry paid)
depending on direction and the pair's interest rate environment.
Wednesday carries a triple roll (covers Saturday and Sunday).

---

## 2. Per-Symbol Cost Assumptions

All costs are quoted as pips (1 pip = the 4th decimal for USD pairs; 2nd
decimal for JPY pairs) and in basis points.

| Symbol | Spread (pips) | Slippage (pips) | Total (pips) | Total (bps) | Notes |
|--------|---------------|-----------------|--------------|-------------|-------|
| EURUSD | 0.6           | 0.2             | **0.8**      | 0.8         | Benchmark liquid major |
| GBPUSD | 0.7           | 0.3             | **1.0**      | 1.0         | Wider spread, higher volatility |
| USDJPY | 0.6           | 0.3             | **0.9**      | 0.9         | JPY pip = 0.01; quoted in JPY |
| AUDUSD | 0.8           | 0.4             | **1.2**      | 1.2         | Commodity-linked, can widen in risk-off |
| EURUSD (news) | 2.0  | 1.0             | **3.0**      | 3.0         | NFP / ECB press conferences |

**Config mapping** (`config/config.yaml`):
```yaml
execution:
  slippage_bps: 2      # per-side slippage
  commission_bps: 1    # per-side spread proxy
```
`FXCostModel` is constructed as `FXCostModel(spread_bps=1, slippage_bps=2)` → **3 bps total**.
This is deliberately conservative (worse than EURUSD best-case) to stress-test strategies.

---

## 3. Swap / Rollover Rates

### 3.1 Current Defaults

The following rates are baked into `SWAP_DEFAULTS` in `src/backtest/swap_calculator.py`.
These are rough estimates for mid-2020s retail ECN conditions.
**Replace with broker historical data before any serious use.**

| Symbol  | Long (bps/day) | Short (bps/day) | Notes |
|---------|---------------|-----------------|-------|
| USDJPY  | +0.50         | −1.20           | High carry; long side earns |
| EURUSD  | −0.35         | +0.15           | EUR/USD rate differential |
| GBPUSD  | −0.20         | +0.05           | Low carry spread |
| AUDUSD  | +0.10         | −0.80           | AUD rate compression since 2020 |
| USDCHF  | +0.40         | −1.10           | CHF negative rates legacy premium |

### 3.2 Data Acquisition from MT5

When `data/swap_rates/` is populated, `compute_swap_cost` automatically uses
the CSV data instead of SWAP_DEFAULTS.

**Expected file format:** `data/swap_rates/{SYMBOL}_swaps.csv`
```
date,long_bps,short_bps
2024-01-02,-0.12,0.08
2024-01-03,-0.12,0.08
...
```

**To download from MT5:**
1. Open MT5 → Market Watch → right-click a symbol → Specification.
2. Note the "Swap Long" and "Swap Short" fields (in pips or percent).
3. Convert to bps: multiply pip swaps by 10 for USD majors.
4. Alternatively use `pymt5` or write an EA that logs daily swap rates.
5. Historical swap rates can also be obtained from the broker's FIX API
   or from third-party aggregators (e.g. OANDA rates archive).

### 3.3 Why Swap Kills Carry-Unaware Strategies

A mean-reversion strategy holding USDJPY shorts for an average of 5 days:
```
swap_cost = 1.20 bps/day × 5 days = 6.0 bps
```
If the strategy's gross edge is only 4 bps per trade, swap alone makes it
unprofitable *before* spread and slippage.

---

## 4. Break-Even Analysis

### 4.1 How Costs Kill Most Backtests

For a strategy with:
- Gross return per trade: $r_g$ bps
- All-in cost per trade: $c$ bps
- Win rate: $w$, Average Win: $W$, Average Loss: $L$

The **post-cost profit factor** must satisfy:

$$PF_{net} = \frac{w \cdot (W - c)}{(1-w) \cdot (L + c)} > 1$$

At 0.8 bps all-in cost (EURUSD best-case):

| Gross edge / trade | Trades / day | Annual cost drag | Required gross Sharpe to achieve net 1.0 |
|--------------------|-------------|-----------------|------------------------------------------|
| 1.0 bps            | 2           | ~96 bps         | > 3.5                                    |
| 2.0 bps            | 2           | ~96 bps         | > 2.0                                    |
| 5.0 bps            | 2           | ~96 bps         | > 1.2                                    |
| 5.0 bps            | 10          | ~480 bps        | Not viable                               |

**Rule of thumb:** if transaction costs halve your Sharpe ratio, the strategy
is unlikely to be robust to real-world execution.

### 4.2 Turnover Sensitivity

Use `results["trades"].abs().sum()` to count total trade units executed.
Multiply by cost per unit to get total cost drag.

```python
total_cost_bps = results["costs"].sum() * 10_000
daily_cost = total_cost_bps / len(results["equity"]) * 24  # hourly bars → daily
```

If `daily_cost > 1 bps`, reconsider the strategy's signal frequency.

---

## 5. References

- Avellaneda & Stoikov (2008) – *High-frequency trading in a limit order book*
- Kissell (2013) – *The Science of Algorithmic Trading and Portfolio Management*
- Aldridge (2013) – *High-Frequency Trading* (Chapter 7: Transaction Costs)
- MT5 documentation – Swap rates and overnight financing
