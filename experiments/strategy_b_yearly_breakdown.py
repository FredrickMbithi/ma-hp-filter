"""
Strategy B — Year-by-Year Performance Breakdown (Crossing Entry, USDJPY H1)
============================================================================
Runs the best IS combo (thresh=1.0σ, hold=48, vol_floor=0, crossing entry)
year-by-year to determine whether the IS edge is concentrated in one quiet
period or is structurally consistent across the 2016–2021 range.

Also runs OOS years (2022–2026) on the same parameters to quantify how and
when the signal breaks down.

Outputs:
  data/interim/strategy_b_yearly_results.csv  — per-year metrics
  stdout                                        — formatted table + analysis
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm

warnings.filterwarnings("ignore")

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
# Best IS parameters (from corrected strategy_b_crossing_entry.py)
# ---------------------------------------------------------------------------
THRESH_SIGMA    = 1.0
HOLD_BARS       = 48
VOL_FLOOR       = 0          # no vol filter

MA_SHORT        = 50
MA_LONG         = 200
ATR_WINDOW      = 20
SIGNAL_LOOKBACK = 252 * 24
VOL_WINDOW      = 20
VOL_LOOKBACK    = 252 * 24

N_BARS_PER_YEAR = 252 * 24
COST_BPS        = 0.9        # base case only for yearly breakdown

INTERIM_DIR = os.path.join(PROJ_ROOT, "data", "interim")
OUTPUT_CSV  = os.path.join(INTERIM_DIR, "strategy_b_yearly_results.csv")
os.makedirs(INTERIM_DIR, exist_ok=True)

SEP  = "=" * 72
SEP2 = "-" * 72


def compute_metrics(net_returns: pd.Series, equity: pd.Series,
                    trades: pd.Series) -> dict:
    net = net_returns.dropna()
    if len(net) < 5:
        return {"sharpe": None, "annual_return": None, "max_dd": None,
                "n_trades": 0, "n_obs": len(net), "pct_pos": None,
                "vol_ann": None}

    mean_ret = net.mean()
    std_ret  = net.std()
    # Annualise Sharpe with actual bars in the year (not full-year constant)
    # so partial years (2016, 2026) show comparable numbers
    n_obs      = len(net)
    ann_factor = N_BARS_PER_YEAR  # consistent with full-year denominator
    sharpe     = (mean_ret / std_ret * np.sqrt(ann_factor)) if std_ret > 0 else 0.0
    annual_return = mean_ret * ann_factor
    vol_ann       = std_ret * np.sqrt(ann_factor)

    rolling_max = equity.cummax()
    max_dd      = float(((equity - rolling_max) / rolling_max).min())
    n_trades    = int(trades.abs().sum() / 2)
    active      = net[net != 0]
    pct_pos     = float((active > 0).mean()) if len(active) > 0 else None

    return {
        "sharpe":        round(sharpe, 3),
        "annual_return": round(annual_return, 4),
        "max_dd":        round(max_dd, 4),
        "n_trades":      n_trades,
        "n_obs":         n_obs,
        "pct_pos":       round(pct_pos, 3) if pct_pos is not None else None,
        "vol_ann":       round(vol_ann * 100, 2),
    }


def run_slice(df_slice: pd.DataFrame, signal_slice: pd.Series,
              cost_model: FXCostModel) -> dict:
    if len(df_slice) < 10 or signal_slice.abs().sum() == 0:
        return {"sharpe": None, "annual_return": None, "max_dd": None,
                "n_trades": 0, "n_obs": len(df_slice), "pct_pos": None,
                "vol_ann": None}
    try:
        bt = VectorizedBacktest(data=df_slice, signal=signal_slice,
                                cost_model=cost_model)
        r = bt.run()
    except Exception as exc:
        return {"error": str(exc)}
    return compute_metrics(r["net_returns"], r["equity"], r["trades"])


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print(SEP)
print("STRATEGY B — YEAR-BY-YEAR BREAKDOWN (CROSSING ENTRY, USDJPY H1)")
print(f"Parameters: thresh={THRESH_SIGMA}σ  hold={HOLD_BARS}  vol_floor={VOL_FLOOR}")
print(SEP)

loader = FXDataLoader(os.path.join(PROJ_ROOT, "data", "raw"))
df     = loader.load("USDJPY_10yr_1h_dukascopy")
print(f"Loaded: {len(df)} bars  |  {df.index[0]} → {df.index[-1]}")

close   = df["close"]
log_ret = compute_log_returns(close)

# ---------------------------------------------------------------------------
# Build signal on full history (required for correct rolling-std normalisation)
# ---------------------------------------------------------------------------
print("\nBuilding features on full history...")
atr_series = atr(df["high"], df["low"], df["close"], window=ATR_WINDOW)
ma_sh      = sma(close, MA_SHORT)
ma_lng     = sma(close, MA_LONG)
raw_spread = (ma_sh - ma_lng) / atr_series.replace(0, np.nan)
raw_spread = raw_spread.replace([np.inf, -np.inf], np.nan)

vol_pct = vol_rolling_percentile(log_ret, vol_window=VOL_WINDOW, lookback=VOL_LOOKBACK)

signal = crossing_threshold_signal(
    raw_spread,
    threshold_sigma=THRESH_SIGMA,
    hold_bars=HOLD_BARS,
    lookback=SIGNAL_LOOKBACK,
)
if VOL_FLOOR > 0:
    signal[vol_pct < VOL_FLOOR] = 0.0
    signal[vol_pct.isna()] = 0.0

print(f"  Active bars: {int((signal != 0).sum())} of {len(signal)}")
print(f"  Long bars:   {int((signal > 0).sum())}")
print(f"  Short bars:  {int((signal < 0).sum())}")

cost_model = FXCostModel(spread_bps=COST_BPS * 0.78, slippage_bps=COST_BPS * 0.22)

# ---------------------------------------------------------------------------
# Year-by-year breakdown
# ---------------------------------------------------------------------------
years  = sorted(df.index.year.unique())
rows   = []
for yr in years:
    mask = df.index.year == yr
    m    = run_slice(df[mask], signal[mask], cost_model)
    period = "IS " if yr <= 2021 else "OOS"
    rows.append({
        "year":    yr,
        "period":  period.strip(),
        **m,
        "n_bars":  int(mask.sum()),
    })

results_df = pd.DataFrame(rows)
results_df.to_csv(OUTPUT_CSV, index=False)

# ---------------------------------------------------------------------------
# Full IS / full OOS totals
# ---------------------------------------------------------------------------
IS_END    = "2021-12-31 23:59:59"
OOS_START = "2022-01-01"
is_mask   = df.index <= IS_END
oos_mask  = df.index >= OOS_START

full_is  = run_slice(df[is_mask],  signal[is_mask],  cost_model)
full_oos = run_slice(df[oos_mask], signal[oos_mask], cost_model)

# ---------------------------------------------------------------------------
# Print
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("  YEAR-BY-YEAR RESULTS")
print(SEP)
print(f"  {'Year':>4}  {'Period':>3}  {'Bars':>5}  "
      f"{'Sharpe':>7}  {'AnnRet':>8}  {'MaxDD':>7}  {'Trades':>6}  "
      f"{'Pct+':>5}  {'Vol%':>5}")
print(f"  {'─'*4}  {'─'*3}  {'─'*5}  "
      f"{'─'*7}  {'─'*8}  {'─'*7}  {'─'*6}  "
      f"{'─'*5}  {'─'*5}")

for _, r in results_df.iterrows():
    if r.get("error"):
        print(f"  {int(r['year']):>4}  {r['period']:>3}  {int(r['n_bars']):>5}  ERROR")
        continue
    sharpe_str = f"{r['sharpe']:>+7.3f}" if r['sharpe'] is not None else "    N/A"
    ret_str    = f"{r['annual_return']*100:>+7.2f}%" if r['annual_return'] is not None else "     N/A"
    dd_str     = f"{r['max_dd']*100:>6.1f}%" if r['max_dd'] is not None else "    N/A"
    pct_str    = f"{r['pct_pos']*100:>4.0f}%" if r['pct_pos'] is not None else "  N/A"
    vol_str    = f"{r['vol_ann']:>4.1f}%" if r['vol_ann'] is not None else " N/A"
    print(f"  {int(r['year']):>4}  {r['period']:>3}  {int(r['n_bars']):>5}  "
          f"{sharpe_str}  {ret_str}  {dd_str}  {int(r['n_trades']):>6}  "
          f"{pct_str}  {vol_str}")

print(SEP2)
# Full period rows
for label, m in [("IS total (2016–2021)", full_is), ("OOS total (2022–2026)", full_oos)]:
    sharpe_str = f"{m['sharpe']:>+7.3f}" if m.get('sharpe') is not None else "    N/A"
    ret_str    = f"{m['annual_return']*100:>+7.2f}%" if m.get('annual_return') is not None else "     N/A"
    dd_str     = f"{m['max_dd']*100:>6.1f}%" if m.get('max_dd') is not None else "    N/A"
    vol_str    = f"{m['vol_ann']:>4.1f}%" if m.get('vol_ann') is not None else " N/A"
    print(f"  {label:<22}  "
          f"{sharpe_str}  {ret_str}  {dd_str}  {int(m.get('n_trades',0)):>6}  "
          f"{'':>5}  {vol_str}")

# ---------------------------------------------------------------------------
# Per-period stats
# ---------------------------------------------------------------------------
is_rows  = results_df[results_df["period"] == "IS"].copy()
oos_rows = results_df[results_df["period"] == "OOS"].copy()

print(f"\n{SEP2}")
print("  IS PERIOD SUMMARY (2016–2021)")
print(SEP2)
is_valid = is_rows[is_rows["sharpe"].notna()]
print(f"  Years positive: {int((is_valid['sharpe'] > 0).sum())}/{len(is_valid)}")
print(f"  Median Sharpe:  {is_valid['sharpe'].median():>+.3f}")
print(f"  Worst year:     {int(is_valid.loc[is_valid['sharpe'].idxmin(), 'year'])}  "
      f"Sharpe={is_valid['sharpe'].min():+.3f}")
print(f"  Best year:      {int(is_valid.loc[is_valid['sharpe'].idxmax(), 'year'])}  "
      f"Sharpe={is_valid['sharpe'].max():+.3f}")

print(f"\n{SEP2}")
print("  OOS PERIOD SUMMARY (2022–2026)")
print(SEP2)
oos_valid = oos_rows[oos_rows["sharpe"].notna()]
print(f"  Years positive: {int((oos_valid['sharpe'] > 0).sum())}/{len(oos_valid)}")
print(f"  Median Sharpe:  {oos_valid['sharpe'].median():>+.3f}")
print(f"  Worst year:     {int(oos_valid.loc[oos_valid['sharpe'].idxmin(), 'year'])}  "
      f"Sharpe={oos_valid['sharpe'].min():+.3f}")
if (oos_valid['sharpe'] > 0).any():
    print(f"  Best year:      {int(oos_valid.loc[oos_valid['sharpe'].idxmax(), 'year'])}  "
          f"Sharpe={oos_valid['sharpe'].max():+.3f}")

print(f"\n{SEP}")
print(f"Saved → {OUTPUT_CSV}")
print(SEP)
