"""
Strategy B — MA Spread + Vol Filter (Crossing-Entry, USDJPY H1)
===============================================================
Corrected re-run of strategy_b_ma_spread_vol_filter.py using
threshold CROSSING entry instead of residency-based entry.

Bug fixed:
  The original script used threshold_time_stop_signal (confirmation_bars=0)
  which re-enters on the SAME bar a position exits if the feature is still
  extreme.  This causes serial re-entries during prolonged extreme periods,
  inflating IS trade count and IS Sharpe.

Fix:
  Uses crossing_threshold_signal which requires the feature to return below
  the threshold before a new entry fires.  Entry fires only at the first
  bar after the feature crosses from neutral territory into extreme territory.

Everything else is identical to strategy_b_ma_spread_vol_filter.py:
  - Same dataset (USDJPY_10yr_1h_dukascopy, 62,303 bars)
  - Same grid: 3 thresh × 4 vol_pct_floor × 3 hold_bars = 36 IS combinations
  - Same IS/OOS split: IS ≤ 2021-12-31, OOS ≥ 2022-01-01
  - Same cost scenarios: base=0.9, high=1.5, stress=2.0 bps

Outputs:
  data/interim/strategy_b_crossing_is.csv   — 36-row IS grid
  data/interim/strategy_b_crossing_oos.csv  — OOS × 3 cost scenarios
  stdout                                     — formatted summary

Usage:
  python experiments/strategy_b_crossing_entry.py
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)

from src.data.loader import FXDataLoader
from src.features.generators import (
    atr,
    crossing_threshold_signal,
    vol_rolling_percentile,
)
from src.features.moving_average import sma
from src.features.returns import compute_log_returns
from src.backtest.engine import VectorizedBacktest
from src.backtest.cost_model import FXCostModel

# ---------------------------------------------------------------------------
# Constants (identical to strategy_b_ma_spread_vol_filter.py)
# ---------------------------------------------------------------------------
MA_SHORT    = 50
MA_LONG     = 200
ATR_WINDOW  = 20
SIGNAL_LOOKBACK = 252 * 24
VOL_WINDOW      = 20
VOL_LOOKBACK    = 252 * 24

THRESHOLD_SIGMA_VALUES = [0.5, 1.0, 1.5]
VOL_PCT_FLOORS         = [0, 25, 50, 75]
HOLD_BARS_VALUES       = [12, 24, 48]

N_BARS_PER_YEAR = 252 * 24
MIN_TRADES      = 30
N_TRIALS        = 36

COST_BPS_SCENARIOS = {
    "base":   0.9,
    "high":   1.5,
    "stress": 2.0,
}

IS_END    = "2021-12-31 23:59:59"
OOS_START = "2022-01-01 00:00:00"

INTERIM_DIR    = os.path.join(PROJ_ROOT, "data", "interim")
OUTPUT_IS_CSV  = os.path.join(INTERIM_DIR, "strategy_b_crossing_is.csv")
OUTPUT_OOS_CSV = os.path.join(INTERIM_DIR, "strategy_b_crossing_oos.csv")
os.makedirs(INTERIM_DIR, exist_ok=True)

SEP  = "=" * 72
SEP2 = "-" * 72


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def deflated_sharpe(
    observed_sr: float,
    n_obs: int,
    n_trials: int,
    skew: float = 0.0,
    excess_kurt: float = 0.0,
) -> float:
    euler_gamma = 0.5772156649
    if n_trials <= 1:
        sr_bench = 0.0
    else:
        sr_bench = (
            (1 - euler_gamma) * norm.ppf(1 - 1.0 / n_trials)
            + euler_gamma * norm.ppf(1 - 1.0 / (n_trials * np.e))
        )
    var_sr = (
        1.0 - skew * observed_sr + (excess_kurt - 1.0) / 4.0 * observed_sr ** 2
    ) / (n_obs - 1)
    if var_sr <= 0:
        var_sr = 1.0 / max(n_obs - 1, 1)
    z = (observed_sr - sr_bench) / np.sqrt(var_sr)
    return float(norm.cdf(z))


def compute_metrics(net_returns: pd.Series, equity: pd.Series,
                    trades: pd.Series, n_trials: int) -> dict:
    net = net_returns.dropna()
    if len(net) < 10:
        return {"sharpe": None, "annual_return": None, "max_dd": None,
                "n_trades": 0, "n_obs": len(net), "dsr": 0.0,
                "skew": None, "kurt": None}

    mean_ret = net.mean()
    std_ret  = net.std()
    sharpe   = (mean_ret / std_ret * np.sqrt(N_BARS_PER_YEAR)) if std_ret > 0 else 0.0
    annual_return = mean_ret * N_BARS_PER_YEAR

    rolling_max = equity.cummax()
    max_dd      = float(((equity - rolling_max) / rolling_max).min())
    n_trades    = int(trades.abs().sum() / 2)

    active = net[net != 0]
    skew   = float(active.skew())     if len(active) > 3 else 0.0
    kurt   = float(active.kurtosis()) if len(active) > 3 else 0.0
    dsr    = deflated_sharpe(sharpe, len(net), n_trials, skew, kurt)

    return {
        "sharpe":        round(sharpe, 4),
        "annual_return": round(annual_return, 6),
        "max_dd":        round(max_dd, 6),
        "n_trades":      n_trades,
        "n_obs":         len(net),
        "dsr":           round(dsr, 4),
        "skew":          round(skew, 3),
        "kurt":          round(kurt, 3),
    }


def run_backtest_slice(df_slice: pd.DataFrame, signal_slice: pd.Series,
                       cost_model: FXCostModel, n_trials: int) -> dict:
    try:
        bt = VectorizedBacktest(data=df_slice, signal=signal_slice, cost_model=cost_model)
        results = bt.run()
    except Exception as exc:
        return {"error": str(exc)}
    return compute_metrics(
        results["net_returns"], results["equity"], results["trades"], n_trials
    )


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print(SEP)
print("STRATEGY B CROSSING-ENTRY — MA SPREAD + VOL FILTER (USDJPY H1)")
print("Entry rule: threshold CROSSING only (post-exit lockout until neutral)")
print(SEP)

loader = FXDataLoader(os.path.join(PROJ_ROOT, "data", "raw"))
df     = loader.load("USDJPY_10yr_1h_dukascopy")
print(f"Loaded: {len(df)} bars  |  {df.index[0]} → {df.index[-1]}")

close   = df["close"]
log_ret = compute_log_returns(close)

# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------
print("\nPrecomputing features on full history...")

atr_series = atr(df["high"], df["low"], df["close"], window=ATR_WINDOW)
ma_sh      = sma(close, MA_SHORT)
ma_lng     = sma(close, MA_LONG)
raw_spread = (ma_sh - ma_lng) / atr_series.replace(0, np.nan)
raw_spread = raw_spread.replace([np.inf, -np.inf], np.nan)
print(f"  raw_ma_spread_{MA_SHORT}_{MA_LONG}: {raw_spread.dropna().shape[0]} valid bars")

vol_pct = vol_rolling_percentile(log_ret, vol_window=VOL_WINDOW, lookback=VOL_LOOKBACK)
print(f"  vol_pct_rank:  {vol_pct.dropna().shape[0]} valid bars")

is_mask  = df.index <= IS_END
oos_mask = df.index >= OOS_START
print(f"\n  IS  slice: {int(is_mask.sum())} bars  "
      f"({df.index[is_mask][0].date()} → {df.index[is_mask][-1].date()})")
print(f"  OOS slice: {int(oos_mask.sum())} bars  "
      f"({df.index[oos_mask][0].date()} → {df.index[oos_mask][-1].date()})")

base_cost_model = FXCostModel(spread_bps=0.7, slippage_bps=0.2)

# ---------------------------------------------------------------------------
# Grid search (IS only)
# ---------------------------------------------------------------------------
n_combos = len(THRESHOLD_SIGMA_VALUES) * len(VOL_PCT_FLOORS) * len(HOLD_BARS_VALUES)
print(f"\nRunning IS grid: {n_combos} combinations (crossing-entry)...")

results_is = []
combo_idx  = 0

for thresh in THRESHOLD_SIGMA_VALUES:
    for hold in HOLD_BARS_VALUES:
        # Build crossing signal for full history once per (thresh, hold) pair
        signal_full = crossing_threshold_signal(
            raw_spread,
            threshold_sigma=thresh,
            hold_bars=hold,
            lookback=SIGNAL_LOOKBACK,
        )

        for vol_floor in VOL_PCT_FLOORS:
            combo_idx += 1
            if combo_idx % 12 == 0:
                print(f"  [{combo_idx}/{n_combos}] thresh={thresh} hold={hold} vol_floor={vol_floor}")

            if vol_floor > 0:
                signal_filtered = signal_full.copy()
                signal_filtered[vol_pct < vol_floor] = 0.0
                signal_filtered[vol_pct.isna()] = 0.0
            else:
                signal_filtered = signal_full.copy()

            row_base = {
                "threshold_sigma": thresh,
                "hold_bars":       hold,
                "vol_pct_floor":   vol_floor,
                "active_bars_full": int((signal_filtered != 0).sum()),
                "active_bars_is":   int((signal_filtered[is_mask] != 0).sum()),
            }

            metrics_is = run_backtest_slice(
                df[is_mask], signal_filtered[is_mask], base_cost_model, N_TRIALS
            )
            if "error" in metrics_is:
                results_is.append({**row_base, "sharpe": None, "annual_return": None,
                                    "max_dd": None, "n_trades": 0, "n_obs": 0,
                                    "dsr": 0.0, "underpowered": True,
                                    "error": metrics_is["error"]})
                continue

            underpowered = (metrics_is.get("n_trades", 0) or 0) < MIN_TRADES
            results_is.append({**row_base, **metrics_is, "underpowered": underpowered})

print(f"  [{n_combos}/{n_combos}] IS grid done")

# ---------------------------------------------------------------------------
# Save IS
# ---------------------------------------------------------------------------
is_df = pd.DataFrame(results_is)
is_df.to_csv(OUTPUT_IS_CSV, index=False)
print(f"\nSaved IS grid ({len(is_df)} rows) → {OUTPUT_IS_CSV}")

valid_is = is_df[is_df["sharpe"].notna()].copy()

# ---------------------------------------------------------------------------
# IS summary
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("  IN-SAMPLE RESULTS (crossing-entry)  — top 15 by DSR")
print(SEP)
top15 = valid_is.sort_values(["dsr", "sharpe"], ascending=False).head(15)
print(f"  {'thresh':>6}  {'hold':>4}  {'vpct':>5}  "
      f"{'Sharpe':>8}  {'AnnRet':>8}  {'MaxDD':>7}  {'Trades':>7}  {'DSR':>7}  {'Flag'}")
print(f"  {'─'*6}  {'─'*4}  {'─'*5}  "
      f"{'─'*8}  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}")
for _, r in top15.iterrows():
    flag = "LOW_CNT" if r.get("underpowered") else ""
    print(f"  {r['threshold_sigma']:>6.1f}  {int(r['hold_bars']):>4}  "
          f"{int(r['vol_pct_floor']):>5}  "
          f"{r['sharpe']:>+8.3f}  {r['annual_return']*100:>+7.2f}%  "
          f"{r['max_dd']*100:>6.1f}%  {int(r['n_trades']):>7}  "
          f"{r['dsr']:>7.3f}  {flag}")

# --- Breakdown by vol_floor ---
print(f"\n{SEP2}")
print("  IS AGGREGATE BY VOL_PCT_FLOOR")
print(SEP2)
for vf in VOL_PCT_FLOORS:
    sub   = valid_is[valid_is["vol_pct_floor"] == vf]
    n_pos = int((sub["sharpe"] > 0).sum())
    med_sh = sub["sharpe"].median()
    med_tr = sub["n_trades"].median()
    label = f"≥{vf}th pct" if vf > 0 else "no filter"
    print(f"  vol_floor={vf:>3} ({label:<10})  n={len(sub):>2}  "
          f"Sharpe>0: {n_pos:>2}/{len(sub)}  "
          f"median_Sharpe={med_sh:>+7.3f}  median_Trades={med_tr:>5.0f}")

# --- Compare trade counts: residency vs crossing ---
print(f"\n{SEP2}")
print("  TRADE COUNT COMPARISON: residency (original) vs crossing (this run)")
print(f"  (Cross-checking: crossing should have FEWER trades when feature is sticky)")
print(SEP2)
residency_csv = os.path.join(INTERIM_DIR, "strategy_b_results.csv")
if os.path.exists(residency_csv):
    res_df = pd.read_csv(residency_csv)
    for thresh in THRESHOLD_SIGMA_VALUES:
        for hold in HOLD_BARS_VALUES:
            r_row = res_df[(res_df["threshold_sigma"] == thresh) &
                           (res_df["hold_bars"] == hold) &
                           (res_df["vol_pct_floor"] == 0)]
            c_row = valid_is[(valid_is["threshold_sigma"] == thresh) &
                             (valid_is["hold_bars"] == hold) &
                             (valid_is["vol_pct_floor"] == 0)]
            if len(r_row) > 0 and len(c_row) > 0:
                r_t = int(r_row.iloc[0]["n_trades"])
                c_t = int(c_row.iloc[0]["n_trades"])
                r_s = float(r_row.iloc[0]["sharpe"])
                c_s = float(c_row.iloc[0]["sharpe"])
                delta_t = c_t - r_t
                delta_s = c_s - r_s
                print(f"  thresh={thresh:.1f} hold={hold:>2}  "
                      f"residency: {r_t:>4}t {r_s:>+7.3f}  "
                      f"crossing: {c_t:>4}t {c_s:>+7.3f}  "
                      f"Δtrades={delta_t:>+5}  ΔSharpe={delta_s:>+7.3f}")
else:
    print("  (strategy_b_results.csv not found — skipping comparison)")

# ---------------------------------------------------------------------------
# Walk-forward OOS
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("  WALK-FORWARD: PICKING BEST IS COMBO → EVALUATING ON OOS")
print(SEP)

powered_is = valid_is[~valid_is.get("underpowered", pd.Series(False, index=valid_is.index))]
if len(powered_is) == 0:
    powered_is = valid_is

best_row    = powered_is.nlargest(1, "dsr").iloc[0]
best_thresh = best_row["threshold_sigma"]
best_hold   = int(best_row["hold_bars"])
best_vfloor = int(best_row["vol_pct_floor"])

print(f"  Best IS combo:  thresh={best_thresh:.1f}  hold={best_hold}  "
      f"vol_floor={best_vfloor}")
print(f"  IS metrics:     Sharpe={best_row['sharpe']:+.3f}  "
      f"AnnRet={best_row['annual_return']*100:+.2f}%  "
      f"MaxDD={best_row['max_dd']*100:.1f}%  "
      f"Trades={int(best_row['n_trades'])}  DSR={best_row['dsr']:.3f}")

best_signal_full = crossing_threshold_signal(
    raw_spread,
    threshold_sigma=best_thresh,
    hold_bars=best_hold,
    lookback=SIGNAL_LOOKBACK,
)
if best_vfloor > 0:
    best_signal_full[vol_pct < best_vfloor] = 0.0
    best_signal_full[vol_pct.isna()] = 0.0

print(f"\n  OOS slice: {int(oos_mask.sum())} bars  "
      f"({df.index[oos_mask][0].date()} → {df.index[oos_mask][-1].date()})")
print(f"\n  {'Scenario':<10}  {'Bps':>5}  "
      f"{'Sharpe':>8}  {'AnnRet':>8}  {'MaxDD':>7}  {'Trades':>7}  {'DSR':>7}")
print(f"  {'─'*10}  {'─'*5}  {'─'*8}  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*7}")

oos_records = []
for scenario, bps in COST_BPS_SCENARIOS.items():
    cm = FXCostModel(spread_bps=bps * 0.78, slippage_bps=bps * 0.22)
    m  = run_backtest_slice(df[oos_mask], best_signal_full[oos_mask], cm, n_trials=1)
    if "error" in m:
        print(f"  {scenario:<10}  {bps:>5.1f}  ERROR: {m['error']}")
        continue
    print(f"  {scenario:<10}  {bps:>5.1f}  "
          f"{m['sharpe']:>+8.3f}  {m['annual_return']*100:>+7.2f}%  "
          f"{m['max_dd']*100:>6.1f}%  {int(m['n_trades']):>7}  "
          f"{m['dsr']:>7.3f}")
    oos_records.append({
        "scenario": scenario, "cost_bps": bps,
        "thresh": best_thresh, "hold_bars": best_hold,
        "vol_pct_floor": best_vfloor,
        "entry_rule": "crossing",
        **m,
    })

oos_df = pd.DataFrame(oos_records)
oos_df.to_csv(OUTPUT_OOS_CSV, index=False)
print(f"\nSaved OOS results → {OUTPUT_OOS_CSV}")

n_pos_total = int((valid_is["sharpe"] > 0).sum())
n_dsr_total = int((valid_is["dsr"] > 0.95).sum())
print(f"\n{SEP}")
print(f"  IS OVERALL: {n_pos_total}/{len(valid_is)} combos Sharpe>0  |  "
      f"{n_dsr_total}/{len(valid_is)} DSR>0.95")
print(f"\nDone.")
print(f"  IS results  → {OUTPUT_IS_CSV}")
print(f"  OOS results → {OUTPUT_OOS_CSV}")
print(SEP)
