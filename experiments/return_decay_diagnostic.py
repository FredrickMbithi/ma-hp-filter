"""
Return-Decay Diagnostic (USDJPY H1)
=====================================
Computes E[ret | signal fired, t bars forward] for both IC gate-passing features:
  - hp_trend_curvature     (lambda=100M, mean-reversion direction)
  - raw_ma_spread_50_200   (ATR-normalised, mean-reversion direction)

Horizons tested: t ∈ {1, 2, 4, 8, 12, 24, 48}
Threshold levels: unconditional (all bars) + 0.5σ, 1.0σ, 1.5σ

For each (feature, threshold, horizon) combination reports:
  n_events        : number of qualifying signal event bars
  mean_signed_bps : E[direction × cumulative_fwd_log_return] × 10000 (basis points)
  t_stat          : mean / (std / sqrt(n))  — two-tailed, H0: mean = 0
  pct_pos         : fraction of events where signed return > 0

Signal convention (mean-reversion):
  feature > +threshold × rolling_std  →  SHORT direction (direction = -1)
  feature < −threshold × rolling_std  →  LONG  direction (direction = +1)
  direction = -sign(feature) at event bar

WARNING: events may overlap (a single future bar can appear in multiple event
windows). This is standard practice for decay diagnostics — we are measuring
information content, not simulating a strategy.

Outputs:
  data/interim/return_decay_results.csv   — machine-readable full results
  stdout                                   — formatted summary table

Usage:
  python experiments/return_decay_diagnostic.py
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJ_ROOT)

from src.data.loader import FXDataLoader
from src.features.generators import atr, hp_trend_curvature
from src.features.moving_average import sma
from src.features.returns import compute_log_returns

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LAMBDA_CURV = 1e8          # 100M — best Strategy A lambda (no lambda advantage, use as reference)
HP_WINDOW   = 500

MA_SHORT    = 50
MA_LONG     = 200
ATR_WINDOW  = 20

SIGNAL_LOOKBACK = 252 * 24      # 1-year rolling std window for normalisation
HORIZONS        = [1, 2, 4, 8, 12, 24, 48]
THRESHOLDS      = [0.0, 0.5, 1.0, 1.5]   # 0.0 = unconditional (all bars)
N_BARS_PER_YEAR = 252 * 24

INTERIM_DIR = os.path.join(PROJ_ROOT, "data", "interim")
HP_CACHE    = os.path.join(INTERIM_DIR, "hp_trends_window500.csv")
OUTPUT_CSV  = os.path.join(INTERIM_DIR, "return_decay_results.csv")

SEP  = "=" * 72
SEP2 = "-" * 72


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def rolling_std_normalise(series: pd.Series, lookback: int) -> pd.Series:
    """Return series divided by its rolling std over `lookback` bars."""
    std = series.rolling(lookback, min_periods=lookback // 2).std()
    return series / std.replace(0, np.nan)


def compute_decay_table(
    feature: pd.Series,
    log_ret: pd.Series,
    label: str,
    thresholds: list[float],
    horizons: list[int],
) -> pd.DataFrame:
    """
    Compute forward-return decay statistics for a mean-reversion feature.

    Vectorized: precomputes cumulative forward returns via cumsum, then
    selects event-bar rows.  O(N × H) rather than O(N² × H).
    """
    # Align on common index
    z = rolling_std_normalise(feature, SIGNAL_LOOKBACK)
    combined = pd.concat([z, log_ret], axis=1).dropna()
    combined.columns = ["z", "lr"]

    z_arr  = combined["z"].values.astype(float)
    lr_arr = combined["lr"].values.astype(float)
    n = len(z_arr)

    # Build forward-return lookup using cumulative sum
    # fwd_cum[i] = sum(lr_arr[0 : i])  (0-indexed)
    # sum(lr_arr[i+1 : i+1+h]) = fwd_cum[i+1+h] - fwd_cum[i+1]
    fwd_cum = np.concatenate([[0.0], np.nancumsum(lr_arr)])  # length n+1

    # direction for mean-reversion: feature high → short (-1); low → long (+1)
    direction_arr = -np.sign(z_arr)  # +1 when z<0 (long), -1 when z>0 (short)

    rows = []

    for thresh in thresholds:
        if thresh == 0.0:
            event_mask = direction_arr != 0   # all valid bars (sign ≠ 0)
        else:
            event_mask = np.abs(z_arr) > thresh

        event_idx = np.where(event_mask)[0]
        d_arr     = direction_arr[event_idx]

        for h in horizons:
            # Vectorized forward return: sum(lr_arr[i+1 : i+1+h])
            end_idx   = event_idx + h + 1         # cumsum index for end of window
            start_idx = event_idx + 1             # cumsum index for start of window

            # Filter bars where the forward window fits in the array
            within = end_idx <= n
            ei  = event_idx[within]
            di  = d_arr[within]
            end = end_idx[within]
            st  = start_idx[within]

            # Filter bars where any forward return is NaN (via cumsum check)
            # We use the fact that nancumsum silently skips NaN, so verify directly
            fwd_rets = fwd_cum[end] - fwd_cum[st]

            # Remove zero-direction bars (z_arr was exactly 0.0)
            nonzero = di != 0
            fwd_rets = fwd_rets[nonzero]
            di       = di[nonzero]

            signed_rets = di * fwd_rets

            if len(signed_rets) < 10:
                rows.append({
                    "feature": label, "threshold_sigma": thresh,
                    "horizon_h": h, "n_events": len(signed_rets),
                    "mean_signed_bps": np.nan, "t_stat": np.nan, "pct_pos": np.nan,
                })
                continue

            mean_val = signed_rets.mean()
            std_val  = signed_rets.std(ddof=1)
            t_stat   = mean_val / (std_val / np.sqrt(len(signed_rets))) if std_val > 0 else 0.0
            pct_pos  = (signed_rets > 0).mean()

            rows.append({
                "feature":         label,
                "threshold_sigma": thresh,
                "horizon_h":       h,
                "n_events":        len(signed_rets),
                "mean_signed_bps": round(mean_val * 10_000, 4),
                "t_stat":          round(t_stat, 3),
                "pct_pos":         round(pct_pos, 4),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print(SEP)
print("RETURN-DECAY DIAGNOSTIC (USDJPY H1, DUKASCOPY 10yr)")
print(SEP)

loader = FXDataLoader(os.path.join(PROJ_ROOT, "data", "raw"))
df     = loader.load("USDJPY_10yr_1h_dukascopy")
print(f"Loaded: {len(df)} bars  |  {df.index[0]} → {df.index[-1]}")

close   = df["close"]
log_ret = compute_log_returns(close)

# ---------------------------------------------------------------------------
# Feature 1: hp_trend_curvature (lambda=100M, from cache)
# ---------------------------------------------------------------------------
print(f"\nLoading HP trends cache: {HP_CACHE}")
hp_cache = pd.read_csv(HP_CACHE, index_col=0, parse_dates=True)
print(f"  Cache: {len(hp_cache)} rows × {hp_cache.shape[1]} columns")
print(f"  Columns: {list(hp_cache.columns)}")

# Find the 100M column
col_map = {
    "hp_trend_1M":   1e6,
    "hp_trend_10M":  1e7,
    "hp_trend_100M": 1e8,
    "hp_trend_1B":   1e9,
    "hp_trend_3.9B": 3.9e9,
    "hp_trend_100B": 1e11,
}
# Identify the 100M column name from the cache
trend_col = None
for col in hp_cache.columns:
    if "100M" in col or col == "hp_trend_100M":
        trend_col = col
        break

if trend_col is None:
    # fallback: pick third column
    trend_col = hp_cache.columns[2]
    print(f"  WARNING: Could not find 100M column; using '{trend_col}'")
else:
    print(f"  Using column: '{trend_col}'")

trend_100M = hp_cache[trend_col].reindex(df.index)

# Curvature: second difference of HP trend
curv = hp_trend_curvature(trend_100M)
n_curv = curv.dropna().shape[0]
print(f"\nhp_trend_curvature (λ=100M): {n_curv} valid bars")

# ---------------------------------------------------------------------------
# Feature 2: raw_ma_spread_50_200 (ATR-normalised)
# ---------------------------------------------------------------------------
print("\nComputing raw_ma_spread_50_200 ...")
atr_series = atr(df["high"], df["low"], df["close"], window=ATR_WINDOW)
ma_sh  = sma(close, MA_SHORT)
ma_lng = sma(close, MA_LONG)
raw_spread = (ma_sh - ma_lng) / atr_series.replace(0, np.nan)
raw_spread = raw_spread.replace([np.inf, -np.inf], np.nan)
n_spread = raw_spread.dropna().shape[0]
print(f"raw_ma_spread_{MA_SHORT}_{MA_LONG}: {n_spread} valid bars")

# ---------------------------------------------------------------------------
# Compute decay tables
# ---------------------------------------------------------------------------
print(f"\nComputing decay tables for {len(THRESHOLDS)} thresholds × {len(HORIZONS)} horizons ...")
print("  (events may overlap — measuring information content, not strategy P&L)")

results_curv   = compute_decay_table(curv,       log_ret, "hp_trend_curvature",   THRESHOLDS, HORIZONS)
results_spread = compute_decay_table(raw_spread, log_ret, "raw_ma_spread_50_200", THRESHOLDS, HORIZONS)

all_results = pd.concat([results_curv, results_spread], ignore_index=True)
all_results.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved: {OUTPUT_CSV}")

# ---------------------------------------------------------------------------
# Print results
# ---------------------------------------------------------------------------
THRESH_LABELS = {0.0: "unconditional", 0.5: "0.5σ", 1.0: "1.0σ", 1.5: "1.5σ"}

for feature_label in ["hp_trend_curvature", "raw_ma_spread_50_200"]:
    print(f"\n{SEP}")
    print(f"FEATURE: {feature_label}")
    print(SEP)

    sub = all_results[all_results["feature"] == feature_label]

    for thresh in THRESHOLDS:
        t_label = THRESH_LABELS[thresh]
        row = sub[sub["threshold_sigma"] == thresh]
        n_sample = int(row["n_events"].iloc[0]) if len(row) > 0 else 0
        print(f"\n  Threshold: {t_label}  (n_events ≈ {n_sample} per horizon)")
        print(f"  {'H':>5}  {'mean_bps':>10}  {'t_stat':>8}  {'pct_pos':>8}  {'sig':>4}")
        print(f"  {'-'*5}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*4}")

        for _, r in row.iterrows():
            h    = int(r["horizon_h"])
            mbps = r["mean_signed_bps"]
            tst  = r["t_stat"]
            pp   = r["pct_pos"]

            if np.isnan(mbps):
                print(f"  {h:>5}  {'NaN':>10}  {'NaN':>8}  {'NaN':>8}  {'':>4}")
                continue

            # Significance marker: two-tailed p < 0.05 ↔ |t| > 1.96
            sig = "***" if abs(tst) > 3.29 else ("** " if abs(tst) > 2.58 else
                  ("*  " if abs(tst) > 1.96 else "   "))

            print(f"  {h:>5}  {mbps:>+10.3f}  {tst:>+8.3f}  {pp:>7.1%}  {sig}")

# ---------------------------------------------------------------------------
# Cross-feature summary: peak horizon and decay rate
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("SUMMARY — PEAK HORIZON (highest |t_stat| per feature × threshold)")
print(SEP)
print(f"  {'Feature':<26}  {'Thresh':>7}  {'PeakH':>6}  {'mean_bps':>10}  {'t_stat':>8}  {'pct_pos':>8}")
print(f"  {'-'*26}  {'-'*7}  {'-'*6}  {'-'*10}  {'-'*8}  {'-'*8}")

for feature_label in ["hp_trend_curvature", "raw_ma_spread_50_200"]:
    sub = all_results[all_results["feature"] == feature_label].dropna(subset=["t_stat"])
    for thresh in [0.0, 0.5, 1.0, 1.5]:
        t_sub = sub[sub["threshold_sigma"] == thresh]
        if len(t_sub) == 0:
            continue
        best = t_sub.loc[t_sub["t_stat"].abs().idxmax()]
        t_label = THRESH_LABELS[thresh]
        print(f"  {feature_label:<26}  {t_label:>7}  {int(best['horizon_h']):>6}  "
              f"{best['mean_signed_bps']:>+10.3f}  {best['t_stat']:>+8.3f}  "
              f"{best['pct_pos']:>7.1%}")

print(f"\n{SEP}")
print("DONE")
print(SEP)
