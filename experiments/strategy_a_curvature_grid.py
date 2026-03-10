"""
Strategy A — HP Trend Curvature Grid Search (USDJPY H1)
========================================================
Tests the hypothesis that hp_trend_curvature (Δ²S*(t)) carries a short-horizon
(H=1) mean-reversion signal.  Uses threshold_time_stop_signal with:

  - 4 lambda values (1M, 10M, 100M, 1B)  — drop 3.9B / 100B (correlated)
  - 3 confirmation_bars values            — {0, 1, 2}
  - 6 hold_bars values                    — {1, 2, 3, 4, 6, 8}
  - 3 entry threshold values              — {0.5, 1.0, 1.5}σ

Total: 4 × 3 × 6 × 3 = 216 combinations.

Signal logic (threshold_time_stop_signal):
  Long  when curvature < −threshold × rolling_std  (exhaustion of uptrend)
  Short when curvature > +threshold × rolling_std  (exhaustion of downtrend)
  Exit  after hold_bars bars OR when feature crosses opposite threshold.

IC finding (Dukascopy 10yr, 62K bars):
  hp_trend_curvature  H=1  IC = −0.0235  (mean-reversion, p ≈ 0.000)

Outputs:
  data/interim/strategy_a_results.csv   — full 216-row results table
  stdout                                 — top-10 by DSR + aggregate summary

Usage:
  python experiments/strategy_a_curvature_grid.py
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
# Constants
# ---------------------------------------------------------------------------
# Drop 3.9B and 100B — highly correlated with 1B on this dataset (slope ρ ≥ 0.9995)
LAMBDAS = [1e6, 1e7, 1e8, 1e9]
LAMBDA_LABELS = ["1M", "10M", "100M", "1B"]

CONFIRMATION_BARS_VALUES = [0, 1, 2]
HOLD_BARS_VALUES = [1, 2, 3, 4, 6, 8]
THRESHOLD_SIGMA_VALUES = [0.5, 1.0, 1.5]

HP_WINDOW = 500
ATR_WINDOW = 20
SIGNAL_LOOKBACK = 252 * 24      # 1 yr of H1 bars for rolling-std normalisation
N_BARS_PER_YEAR = 252 * 24
MIN_TRADES = 30
N_TRIALS = 216                  # DSR correction: total grid size

COST_MODEL = FXCostModel(spread_bps=0.7, slippage_bps=0.2)

INTERIM_DIR = os.path.join(PROJ_ROOT, "data", "interim")
OUTPUT_CSV = os.path.join(INTERIM_DIR, "strategy_a_results.csv")
os.makedirs(INTERIM_DIR, exist_ok=True)

SEP = "=" * 72
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


def run_backtest(df: pd.DataFrame, signal: pd.Series, cost_model=None) -> dict:
    """Run VectorizedBacktest and return summary metrics dict."""
    cm = cost_model if cost_model is not None else COST_MODEL
    try:
        bt = VectorizedBacktest(data=df, signal=signal, cost_model=cm)
        results = bt.run()
    except Exception as exc:
        return {"error": str(exc)}

    net = results["net_returns"].dropna()
    if len(net) < 10:
        return {"sharpe": None, "annual_return": None, "max_dd": None,
                "n_trades": 0, "n_obs": len(net), "dsr": 0.0}

    mean_ret = net.mean()
    std_ret = net.std()
    sharpe = (mean_ret / std_ret * np.sqrt(N_BARS_PER_YEAR)) if std_ret > 0 else 0.0
    annual_return = mean_ret * N_BARS_PER_YEAR

    equity = results["equity"]
    rolling_max = equity.cummax()
    max_dd = float(((equity - rolling_max) / rolling_max).min())

    n_trades = int(results["trades"].abs().sum() / 2)

    active = net[net != 0]
    skew = float(active.skew()) if len(active) > 3 else 0.0
    kurt = float(active.kurtosis()) if len(active) > 3 else 0.0
    dsr = deflated_sharpe(sharpe, len(net), N_TRIALS, skew, kurt)

    return {
        "sharpe": round(sharpe, 4),
        "annual_return": round(annual_return, 6),
        "max_dd": round(max_dd, 6),
        "n_trades": n_trades,
        "n_obs": len(net),
        "dsr": round(dsr, 4),
        "skew": round(skew, 3),
        "kurt": round(kurt, 3),
    }


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print(SEP)
print("STRATEGY A — HP TREND CURVATURE GRID SEARCH (USDJPY H1)")
print(SEP)

loader = FXDataLoader(os.path.join(PROJ_ROOT, "data", "raw"))
df = loader.load("USDJPY_10yr_1h_dukascopy")
print(f"Loaded: {len(df)} bars  |  {df.index[0]} → {df.index[-1]}")

close = df["close"]
log_ret = compute_log_returns(close)

# ---------------------------------------------------------------------------
# Shared precomputation
# ---------------------------------------------------------------------------
print("\nPrecomputing shared series...")

# HP trends — load from cache (or recompute)
CACHE_FILE = os.path.join(INTERIM_DIR, f"hp_trends_window{HP_WINDOW}.csv")
if os.path.exists(CACHE_FILE):
    hp_trends_all = pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)
    print(f"  Loaded cached HP trends: {hp_trends_all.shape}")
else:
    print(f"  Computing HP trends (window={HP_WINDOW}, {len(LAMBDAS)} lambdas)...")
    trends_dict = {}
    for lam, label in zip(LAMBDAS, LAMBDA_LABELS):
        trends_dict[f"hp_trend_{label}"] = causal_hp_trend(close, lamb=lam, window=HP_WINDOW)
        print(f"    λ={label} done")
    hp_trends_all = pd.DataFrame(trends_dict, index=close.index)
    hp_trends_all.to_csv(CACHE_FILE)

# Select only the 4 lambda columns we need
available_cols = [f"hp_trend_{lbl}" for lbl in LAMBDA_LABELS]
missing = [c for c in available_cols if c not in hp_trends_all.columns]
if missing:
    print(f"  Cache missing columns {missing} — recomputing...")
    for lam, label in zip(LAMBDAS, LAMBDA_LABELS):
        col = f"hp_trend_{label}"
        if col not in hp_trends_all.columns:
            hp_trends_all[col] = causal_hp_trend(close, lamb=lam, window=HP_WINDOW)

hp_trends = hp_trends_all[available_cols]

# Curvatures (per lambda)
print("  Computing curvatures...")
curvatures = {}
for label in LAMBDA_LABELS:
    col = f"hp_trend_{label}"
    curvatures[label] = hp_trend_curvature(hp_trends[col])
    n_valid = curvatures[label].dropna().shape[0]
    print(f"    curvature λ={label}: {n_valid} valid bars")

n_combos = len(LAMBDAS) * len(CONFIRMATION_BARS_VALUES) * len(HOLD_BARS_VALUES) * len(THRESHOLD_SIGMA_VALUES)
print(f"\nRunning {n_combos} combinations...")
print(f"  Grid: {len(LAMBDAS)}λ × "
      f"{len(CONFIRMATION_BARS_VALUES)} conf × "
      f"{len(HOLD_BARS_VALUES)} hold × "
      f"{len(THRESHOLD_SIGMA_VALUES)} thresh")

# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------
results = []
combo_idx = 0

for label in LAMBDA_LABELS:
    curv = curvatures[label]

    for conf in CONFIRMATION_BARS_VALUES:
        for hold in HOLD_BARS_VALUES:
            for thresh in THRESHOLD_SIGMA_VALUES:
                combo_idx += 1
                if combo_idx % 50 == 0:
                    print(f"  [{combo_idx}/{n_combos}] λ={label} conf={conf} hold={hold} thresh={thresh}")

                signal = threshold_time_stop_signal(
                    curv,
                    threshold_sigma=thresh,
                    hold_bars=hold,
                    confirmation_bars=conf,
                    lookback=SIGNAL_LOOKBACK,
                )

                n_nonzero = int((signal != 0).sum())
                row_base = {
                    "lambda": label,
                    "confirmation_bars": conf,
                    "hold_bars": hold,
                    "threshold_sigma": thresh,
                    "active_bars": n_nonzero,
                }

                if n_nonzero < 2:
                    results.append({**row_base, "sharpe": None, "annual_return": None,
                                    "max_dd": None, "n_trades": 0, "n_obs": 0,
                                    "dsr": 0.0, "underpowered": True})
                    continue

                metrics = run_backtest(df, signal)
                if "error" in metrics:
                    results.append({**row_base, "sharpe": None, "annual_return": None,
                                    "max_dd": None, "n_trades": 0, "n_obs": 0,
                                    "dsr": 0.0, "underpowered": True,
                                    "error": metrics["error"]})
                    continue

                underpowered = (metrics.get("n_trades", 0) or 0) < MIN_TRADES
                results.append({**row_base, **metrics, "underpowered": underpowered})

print(f"  [{n_combos}/{n_combos}] done")

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved {len(results_df)} rows → {OUTPUT_CSV}")

# ---------------------------------------------------------------------------
# Summary output
# ---------------------------------------------------------------------------
valid = results_df[results_df["sharpe"].notna()].copy()

print(f"\n{SEP}")
print("  TOP 20 COMBINATIONS  (sorted by DSR, then Sharpe)")
print(SEP)
top20 = valid.sort_values(["dsr", "sharpe"], ascending=False).head(20)
print(f"  {'λ':>5}  {'conf':>4}  {'hold':>4}  {'σ':>4}  "
      f"{'Sharpe':>8}  {'AnnRet':>8}  {'MaxDD':>7}  {'Trades':>7}  {'DSR':>7}  {'Flag'}")
print(f"  {'─'*5}  {'─'*4}  {'─'*4}  {'─'*4}  "
      f"{'─'*8}  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*10}")
for _, r in top20.iterrows():
    flag = "LOW_CNT" if r.get("underpowered") else ""
    print(f"  {r['lambda']:>5}  {int(r['confirmation_bars']):>4}  {int(r['hold_bars']):>4}  "
          f"{r['threshold_sigma']:>4.1f}  "
          f"{r['sharpe']:>+8.3f}  {r['annual_return']*100:>+7.2f}%  "
          f"{r['max_dd']*100:>6.1f}%  {int(r['n_trades']):>7}  "
          f"{r['dsr']:>7.3f}  {flag}")

# --- Breakdown by lambda ---
print(f"\n{SEP2}")
print("  AGGREGATE BY LAMBDA")
print(SEP2)
for lbl in LAMBDA_LABELS:
    sub = valid[valid["lambda"] == lbl]
    n_pos = int((sub["sharpe"] > 0).sum())
    n_dsr = int((sub["dsr"] > 0.95).sum())
    best = sub.nlargest(1, "dsr")
    best_str = ""
    if not best.empty:
        r = best.iloc[0]
        best_str = (f"conf={int(r['confirmation_bars'])} hold={int(r['hold_bars'])} "
                    f"thresh={r['threshold_sigma']:.1f}  "
                    f"Sharpe={r['sharpe']:+.3f}  DSR={r['dsr']:.3f}")
    print(f"  λ={lbl:<6}  Sharpe>0: {n_pos:>3}/{len(sub)}  "
          f"DSR>0.95: {n_dsr:>2}/{len(sub)}  |  {best_str}")

# --- Breakdown by holding period ---
print(f"\n{SEP2}")
print("  AGGREGATE BY HOLD_BARS")
print(SEP2)
for hold in HOLD_BARS_VALUES:
    sub = valid[valid["hold_bars"] == hold]
    n_pos = int((sub["sharpe"] > 0).sum())
    median_sh = sub["sharpe"].median()
    print(f"  hold={hold:>2}  n={len(sub):>3}  "
          f"Sharpe>0: {n_pos:>3}/{len(sub)}  "
          f"median_Sharpe={median_sh:>+7.3f}")

# --- Breakdown by threshold ---
print(f"\n{SEP2}")
print("  AGGREGATE BY THRESHOLD_SIGMA")
print(SEP2)
for thresh in THRESHOLD_SIGMA_VALUES:
    sub = valid[valid["threshold_sigma"] == thresh]
    n_pos = int((sub["sharpe"] > 0).sum())
    median_sh = sub["sharpe"].median()
    print(f"  thresh={thresh:.1f}σ  n={len(sub):>3}  "
          f"Sharpe>0: {n_pos:>3}/{len(sub)}  "
          f"median_Sharpe={median_sh:>+7.3f}")

# --- Overall pass rate ---
n_pos_total = int((valid["sharpe"] > 0).sum())
n_dsr_total = int((valid["dsr"] > 0.95).sum())
n_underpwd = int(valid["underpowered"].sum())
print(f"\n{SEP}")
print(f"  OVERALL: {n_pos_total}/{len(valid)} combos Sharpe>0  |  "
      f"{n_dsr_total}/{len(valid)} DSR>0.95  |  "
      f"{n_underpwd} underpowered")
print(f"\nDone. Results → {OUTPUT_CSV}")
print(SEP)
