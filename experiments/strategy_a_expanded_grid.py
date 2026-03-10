"""
Strategy A — Expanded Grid Search: Lower Thresholds, Lower Confirmation
========================================================================
Supplementary grid targeting the region between the confirmed winners
(thresh=1.5σ, conf=2) and the confirmed losers (thresh=0.5σ) from the
original 216-combo Strategy A grid.

Goal: identify the lowest threshold + confirmation combination that
preserves positive Sharpe with ≥ 50 trades/year (≥ 500 total on 10yr
dataset). That is the trade count floor below which DSR becomes meaningful.

Grid:
  4  lambda values        (1M, 10M, 100M, 1B — drop 3.9B/100B as before)
  2  confirmation_bars    ({0, 1} — conf=2 already tested and confirmed)
  6  hold_bars            ({1, 2, 3, 4, 6, 8} — same as original)
  3  threshold_sigma      ({0.75, 1.0, 1.25} — fills in original gap)

Total: 4 × 2 × 6 × 3 = 144 combinations

Output includes a dedicated "high-count" table showing only combinations
with ≥ 500 trades (≈ 50/year), sorted by Sharpe.

Outputs:
  data/interim/strategy_a_expanded_results.csv   — full 144-row results
  stdout                                          — summary tables

Usage:
  python experiments/strategy_a_expanded_grid.py
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
    causal_hp_trend,
    hp_trend_curvature,
    threshold_time_stop_signal,
)
from src.features.returns import compute_log_returns
from src.backtest.engine import VectorizedBacktest
from src.backtest.cost_model import FXCostModel

# ---------------------------------------------------------------------------
# Grid definition — fills gap between original 0.5/1.0/1.5σ grid
# ---------------------------------------------------------------------------
LAMBDAS       = [1e6, 1e7, 1e8, 1e9]
LAMBDA_LABELS = ["1M", "10M", "100M", "1B"]

# Lower conf values (conf=2 is already confirmed; test {0,1} here)
CONFIRMATION_BARS_VALUES = [0, 1]
HOLD_BARS_VALUES         = [1, 2, 3, 4, 6, 8]
THRESHOLD_SIGMA_VALUES   = [0.75, 1.0, 1.25]     # fills original gap

HP_WINDOW        = 500
ATR_WINDOW       = 20
SIGNAL_LOOKBACK  = 252 * 24      # 1 yr of H1 bars
N_BARS_PER_YEAR  = 252 * 24
MIN_TRADES       = 30            # below this → underpowered flag

# Target for "high-count" analysis
HIGH_COUNT_THRESHOLD = 500       # ≈ 50 trades/year on 10yr dataset

N_TRIALS = 144                   # DSR correction: total expanded grid size

COST_MODEL = FXCostModel(spread_bps=0.7, slippage_bps=0.2)

INTERIM_DIR = os.path.join(PROJ_ROOT, "data", "interim")
CACHE_FILE  = os.path.join(INTERIM_DIR, f"hp_trends_window{HP_WINDOW}.csv")
OUTPUT_CSV  = os.path.join(INTERIM_DIR, "strategy_a_expanded_results.csv")
os.makedirs(INTERIM_DIR, exist_ok=True)

SEP  = "=" * 72
SEP2 = "-" * 72


# ---------------------------------------------------------------------------
# Metric helpers (identical to strategy_a_curvature_grid.py)
# ---------------------------------------------------------------------------

def deflated_sharpe(
    observed_sr: float,
    n_obs: int,
    n_trials: int,
    skew: float = 0.0,
    excess_kurt: float = 0.0,
) -> float:
    """Probability that observed SR exceeds the expected maximum under null."""
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


def run_backtest(df: pd.DataFrame, signal: pd.Series) -> dict:
    """Run VectorizedBacktest and return summary metrics dict."""
    try:
        bt = VectorizedBacktest(data=df, signal=signal, cost_model=COST_MODEL)
        results = bt.run()
    except Exception as exc:
        return {"error": str(exc)}

    net = results["net_returns"].dropna()
    if len(net) < 10:
        return {"sharpe": None, "annual_return": None, "max_dd": None,
                "n_trades": 0, "n_obs": len(net), "dsr": 0.0}

    mean_ret = net.mean()
    std_ret  = net.std()
    sharpe   = (mean_ret / std_ret * np.sqrt(N_BARS_PER_YEAR)) if std_ret > 0 else 0.0
    annual_return = mean_ret * N_BARS_PER_YEAR

    equity      = results["equity"]
    rolling_max = equity.cummax()
    max_dd      = float(((equity - rolling_max) / rolling_max).min())

    n_trades = int(results["trades"].abs().sum() / 2)

    active    = net[net != 0]
    skew      = float(active.skew())     if len(active) > 3 else 0.0
    kurt      = float(active.kurtosis()) if len(active) > 3 else 0.0
    dsr       = deflated_sharpe(sharpe, len(net), N_TRIALS, skew, kurt)

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


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print(SEP)
print("STRATEGY A — EXPANDED GRID (LOWER THRESH / LOWER CONF, USDJPY H1)")
print(f"Grid: {len(LAMBDAS)}λ × {len(CONFIRMATION_BARS_VALUES)} conf × "
      f"{len(HOLD_BARS_VALUES)} hold × {len(THRESHOLD_SIGMA_VALUES)} thresh "
      f"= {len(LAMBDAS)*len(CONFIRMATION_BARS_VALUES)*len(HOLD_BARS_VALUES)*len(THRESHOLD_SIGMA_VALUES)} combos")
print(SEP)

loader = FXDataLoader(os.path.join(PROJ_ROOT, "data", "raw"))
df     = loader.load("USDJPY_10yr_1h_dukascopy")
print(f"Loaded: {len(df)} bars  |  {df.index[0]} → {df.index[-1]}")

close   = df["close"]
log_ret = compute_log_returns(close)

# ---------------------------------------------------------------------------
# Precompute HP trends and curvatures
# ---------------------------------------------------------------------------
print("\nPrecomputing HP trends / curvatures...")
if os.path.exists(CACHE_FILE):
    hp_trends_all = pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)
    print(f"  Loaded cached HP trends: {hp_trends_all.shape}")
else:
    print(f"  Cache not found — computing HP trends (window={HP_WINDOW})...")
    trends_dict = {}
    for lam, label in zip(LAMBDAS, LAMBDA_LABELS):
        trends_dict[f"hp_trend_{label}"] = causal_hp_trend(close, lamb=lam, window=HP_WINDOW)
        print(f"    λ={label} done")
    hp_trends_all = pd.DataFrame(trends_dict, index=close.index)
    hp_trends_all.to_csv(CACHE_FILE)

available_cols = [f"hp_trend_{lbl}" for lbl in LAMBDA_LABELS]
missing = [c for c in available_cols if c not in hp_trends_all.columns]
if missing:
    print(f"  Recomputing missing columns: {missing}")
    for lam, label in zip(LAMBDAS, LAMBDA_LABELS):
        col = f"hp_trend_{label}"
        if col not in hp_trends_all.columns:
            hp_trends_all[col] = causal_hp_trend(close, lamb=lam, window=HP_WINDOW)

curvatures = {}
for label in LAMBDA_LABELS:
    col = f"hp_trend_{label}"
    curvatures[label] = hp_trend_curvature(hp_trends_all[col])
    n_valid = curvatures[label].dropna().shape[0]
    print(f"  curvature λ={label}: {n_valid} valid bars")

# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------
n_combos  = (len(LAMBDAS) * len(CONFIRMATION_BARS_VALUES) *
             len(HOLD_BARS_VALUES) * len(THRESHOLD_SIGMA_VALUES))
results   = []
combo_idx = 0

print(f"\nRunning {n_combos} combinations...")

for label in LAMBDA_LABELS:
    curv = curvatures[label]
    for conf in CONFIRMATION_BARS_VALUES:
        for hold in HOLD_BARS_VALUES:
            for thresh in THRESHOLD_SIGMA_VALUES:
                combo_idx += 1
                if combo_idx % 30 == 0:
                    pct = combo_idx / n_combos * 100
                    print(f"  [{combo_idx:>3}/{n_combos}] {pct:5.1f}%  "
                          f"λ={label} conf={conf} hold={hold} thresh={thresh:.2f}")

                signal = threshold_time_stop_signal(
                    curv,
                    threshold_sigma=thresh,
                    hold_bars=hold,
                    confirmation_bars=conf,
                    lookback=SIGNAL_LOOKBACK,
                )

                n_nonzero = int((signal != 0).sum())
                row_base  = {
                    "lambda":             label,
                    "confirmation_bars":  conf,
                    "hold_bars":          hold,
                    "threshold_sigma":    thresh,
                    "active_bars":        n_nonzero,
                }

                if n_nonzero < 2:
                    results.append({**row_base,
                                    "sharpe": None, "annual_return": None,
                                    "max_dd": None, "n_trades": 0,
                                    "n_obs": 0, "dsr": 0.0,
                                    "underpowered": True})
                    continue

                metrics = run_backtest(df, signal)
                if "error" in metrics:
                    results.append({**row_base,
                                    "sharpe": None, "annual_return": None,
                                    "max_dd": None, "n_trades": 0,
                                    "n_obs": 0, "dsr": 0.0,
                                    "underpowered": True,
                                    "error": metrics["error"]})
                    continue

                underpowered = (metrics.get("n_trades", 0) or 0) < MIN_TRADES
                results.append({**row_base, **metrics, "underpowered": underpowered})

print(f"  [{n_combos}/{n_combos}] done")

# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved {len(results_df)} rows → {OUTPUT_CSV}")

valid = results_df[results_df["sharpe"].notna()].copy()

# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("  TOP 20 COMBINATIONS (sorted by DSR, then Sharpe)")
print(SEP)
top20 = valid.sort_values(["dsr", "sharpe"], ascending=False).head(20)
print(f"  {'λ':>5}  {'conf':>4}  {'hold':>4}  {'σ':>5}  "
      f"{'Sharpe':>8}  {'AnnRet':>8}  {'MaxDD':>7}  {'Trades':>7}  {'DSR':>7}  {'Flag'}")
print(f"  {'─'*5}  {'─'*4}  {'─'*4}  {'─'*5}  "
      f"{'─'*8}  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}")
for _, r in top20.iterrows():
    flag = "LOW_CNT" if r.get("underpowered") else ""
    print(f"  {r['lambda']:>5}  {int(r['confirmation_bars']):>4}  "
          f"{int(r['hold_bars']):>4}  {r['threshold_sigma']:>5.2f}  "
          f"{r['sharpe']:>+8.3f}  {r['annual_return']*100:>+7.2f}%  "
          f"{r['max_dd']*100:>6.1f}%  {int(r['n_trades']):>7}  "
          f"{r['dsr']:>7.3f}  {flag}")

# --- High-count filter: ≥ 500 trades ---
print(f"\n{SEP}")
print(f"  HIGH-COUNT FILTER: combinations with ≥ {HIGH_COUNT_THRESHOLD} trades (≈ 50/year)")
print(f"  (This is the primary target — enough trades for DSR to become meaningful)")
print(SEP)
high_count = valid[valid["n_trades"] >= HIGH_COUNT_THRESHOLD].sort_values(
    "sharpe", ascending=False
)
if len(high_count) == 0:
    print(f"  *** NO combinations reached ≥ {HIGH_COUNT_THRESHOLD} trades ***")
    print(f"  Closest (top 10 by trade count):")
    closest = valid.nlargest(10, "n_trades")
    for _, r in closest.iterrows():
        print(f"    λ={r['lambda']}  conf={int(r['confirmation_bars'])}  "
              f"hold={int(r['hold_bars'])}  thresh={r['threshold_sigma']:.2f}  "
              f"Trades={int(r['n_trades'])}  Sharpe={r['sharpe']:+.3f}")
else:
    print(f"  {'λ':>5}  {'conf':>4}  {'hold':>4}  {'σ':>5}  "
          f"{'Sharpe':>8}  {'AnnRet':>8}  {'MaxDD':>7}  {'Trades':>7}  {'DSR':>7}")
    print(f"  {'─'*5}  {'─'*4}  {'─'*4}  {'─'*5}  "
          f"{'─'*8}  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*7}")
    for _, r in high_count.iterrows():
        print(f"  {r['lambda']:>5}  {int(r['confirmation_bars']):>4}  "
              f"{int(r['hold_bars']):>4}  {r['threshold_sigma']:>5.2f}  "
              f"{r['sharpe']:>+8.3f}  {r['annual_return']*100:>+7.2f}%  "
              f"{r['max_dd']*100:>6.1f}%  {int(r['n_trades']):>7}  "
              f"{r['dsr']:>7.3f}")

# --- Breakdown by threshold ---
print(f"\n{SEP2}")
print("  AGGREGATE BY THRESHOLD_SIGMA")
print(SEP2)
for thresh in THRESHOLD_SIGMA_VALUES:
    sub    = valid[valid["threshold_sigma"] == thresh]
    n_pos  = int((sub["sharpe"] > 0).sum())
    med_sh = sub["sharpe"].median()
    med_tr = sub["n_trades"].median()
    n_500  = int((sub["n_trades"] >= HIGH_COUNT_THRESHOLD).sum())
    print(f"  thresh={thresh:.2f}σ  n={len(sub):>3}  "
          f"Sharpe>0: {n_pos:>3}/{len(sub)}  "
          f"median_Sharpe={med_sh:>+7.3f}  "
          f"median_Trades={med_tr:>6.0f}  "
          f"≥{HIGH_COUNT_THRESHOLD}t: {n_500:>3}")

# --- Breakdown by confirmation ---
print(f"\n{SEP2}")
print("  AGGREGATE BY CONFIRMATION_BARS")
print(SEP2)
for conf in CONFIRMATION_BARS_VALUES:
    sub    = valid[valid["confirmation_bars"] == conf]
    n_pos  = int((sub["sharpe"] > 0).sum())
    med_sh = sub["sharpe"].median()
    med_tr = sub["n_trades"].median()
    print(f"  conf={conf}  n={len(sub):>3}  "
          f"Sharpe>0: {n_pos:>3}/{len(sub)}  "
          f"median_Sharpe={med_sh:>+7.3f}  "
          f"median_Trades={med_tr:>6.0f}")

# --- Cross-tab: threshold × confirmation (median Sharpe and median trade count) ---
print(f"\n{SEP2}")
print("  CROSS-TABLE: thresh × conf — median_Sharpe  [median_Trades]")
print(SEP2)
header = "  thresh \\ conf"
for conf in CONFIRMATION_BARS_VALUES:
    header += f"  {'conf='+str(conf):>20}"
print(header)
for thresh in THRESHOLD_SIGMA_VALUES:
    row_str = f"  {thresh:.2f}σ         "
    for conf in CONFIRMATION_BARS_VALUES:
        cell = valid[(valid["threshold_sigma"] == thresh) &
                     (valid["confirmation_bars"] == conf)]
        if len(cell) == 0:
            row_str += f"  {'N/A':>20}"
        else:
            ms = cell["sharpe"].median()
            mt = cell["n_trades"].median()
            row_str += f"  {ms:>+8.3f} [{mt:>6.0f}t]"
    print(row_str)

# --- Overall ---
n_pos_total = int((valid["sharpe"] > 0).sum())
n_dsr_total = int((valid["dsr"] > 0.95).sum())
n_hc_total  = int((valid["n_trades"] >= HIGH_COUNT_THRESHOLD).sum())
print(f"\n{SEP}")
print(f"  OVERALL: {n_pos_total}/{len(valid)} combos Sharpe>0  |  "
      f"{n_dsr_total}/{len(valid)} DSR>0.95  |  "
      f"{n_hc_total}/{len(valid)} trades≥{HIGH_COUNT_THRESHOLD}")
print(f"\nDone. Results → {OUTPUT_CSV}")
print(SEP)
