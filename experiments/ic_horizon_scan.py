"""
IC Horizon Scan (USDJPY H1)
===========================
Computes Spearman rank IC between both IC gate-passing features and
forward log-returns at fine-grained horizons H ∈ {1, 2, 4, 6, 12, 24, 48}.

Features scanned:
  - hp_trend_curvature     (all 4 lambda values: 1M, 10M, 100M, 1B)
  - raw_ma_spread_50_200   (ATR-normalised; single feature)

For each (feature, lambda, horizon) combination reports:
  IC      : Spearman rank correlation (negative = mean-reversion signal)
  p-value : two-tailed H0: IC = 0
  n       : number of valid bar pairs
  sig     : * p<0.05, ** p<0.01, *** p<0.001

Also reports the peak-IC horizon per feature and the full IC decay table.

Outputs:
  data/interim/ic_horizon_scan_results.csv   — machine-readable full results
  stdout                                      — formatted summary table

Usage:
  python experiments/ic_horizon_scan.py
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
LAMBDAS       = [1e6, 1e7, 1e8, 1e9]
LAMBDA_LABELS = ["1M", "10M", "100M", "1B"]

MA_SHORT   = 50
MA_LONG    = 200
ATR_WINDOW = 20
HP_WINDOW  = 500

HORIZONS = [1, 2, 4, 6, 12, 24, 48]

INTERIM_DIR = os.path.join(PROJ_ROOT, "data", "interim")
HP_CACHE    = os.path.join(INTERIM_DIR, "hp_trends_window500.csv")
OUTPUT_CSV  = os.path.join(INTERIM_DIR, "ic_horizon_scan_results.csv")

SEP  = "=" * 72
SEP2 = "-" * 72


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ic_spearman(feature: pd.Series, fwd_returns: pd.Series) -> tuple[float, float, int]:
    """
    Compute Spearman rank IC between feature and fwd_returns.
    Returns (rho, p_value, n_valid).
    """
    valid = pd.concat([feature, fwd_returns], axis=1).dropna()
    n = len(valid)
    if n < 30:
        return (np.nan, np.nan, n)
    rho, pval = stats.spearmanr(valid.iloc[:, 0], valid.iloc[:, 1])
    return (float(rho), float(pval), n)


def sig_stars(pval: float) -> str:
    if np.isnan(pval):
        return "   "
    if pval < 0.001:
        return "***"
    if pval < 0.01:
        return "** "
    if pval < 0.05:
        return "*  "
    return "   "


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print(SEP)
print("IC HORIZON SCAN (USDJPY H1, DUKASCOPY 10yr)")
print(SEP)

loader = FXDataLoader(os.path.join(PROJ_ROOT, "data", "raw"))
df     = loader.load("USDJPY_10yr_1h_dukascopy")
print(f"Loaded: {len(df)} bars  |  {df.index[0]} → {df.index[-1]}")

close   = df["close"]
log_ret = compute_log_returns(close)

# ---------------------------------------------------------------------------
# Load HP trends cache
# ---------------------------------------------------------------------------
print(f"\nLoading HP trends cache: {HP_CACHE}")
hp_cache = pd.read_csv(HP_CACHE, index_col=0, parse_dates=True)
print(f"  {len(hp_cache)} rows × {hp_cache.shape[1]} columns")
print(f"  Columns: {list(hp_cache.columns)}")

# Map column names to lambdas
lambda_col_map = {}
for col in hp_cache.columns:
    if "1M"   in col and "10" not in col and "100" not in col: lambda_col_map["1M"]   = col
    if "10M"  in col and "100" not in col:                      lambda_col_map["10M"]  = col
    if "100M" in col:                                            lambda_col_map["100M"] = col
    if "1B"   in col and "3.9" not in col and "100" not in col: lambda_col_map["1B"]   = col

print(f"  Lambda → column mapping: {lambda_col_map}")

# ---------------------------------------------------------------------------
# Compute curvature features for each lambda
# ---------------------------------------------------------------------------
curvature_features: dict[str, pd.Series] = {}
for lbl in LAMBDA_LABELS:
    if lbl not in lambda_col_map:
        print(f"  WARNING: column for λ={lbl} not found in cache, skipping")
        continue
    trend = hp_cache[lambda_col_map[lbl]].reindex(df.index)
    curv  = hp_trend_curvature(trend)
    n_val = curv.dropna().shape[0]
    print(f"  hp_trend_curvature λ={lbl}: {n_val} valid bars")
    curvature_features[f"curv_λ{lbl}"] = curv

# ---------------------------------------------------------------------------
# Compute ATR-normalised MA spread
# ---------------------------------------------------------------------------
print("\nComputing raw_ma_spread_50_200 ...")
atr_series = atr(df["high"], df["low"], df["close"], window=ATR_WINDOW)
ma_sh  = sma(close, MA_SHORT)
ma_lng = sma(close, MA_LONG)
raw_spread = (ma_sh - ma_lng) / atr_series.replace(0, np.nan)
raw_spread = raw_spread.replace([np.inf, -np.inf], np.nan)
print(f"  {raw_spread.dropna().shape[0]} valid bars")

# ---------------------------------------------------------------------------
# Run IC scan for each feature × horizon
# ---------------------------------------------------------------------------
print(f"\nRunning IC scan: {len(curvature_features) + 1} features × {len(HORIZONS)} horizons ...")

records = []

all_features = dict(curvature_features)
all_features["raw_ma_spread_50_200"] = raw_spread

for feat_label, feat_series in all_features.items():
    for h in HORIZONS:
        # Forward log return = rolling cumulative sum of h bars, shifted back
        fwd = log_ret.shift(-h).rolling(h).sum() if h > 1 else log_ret.shift(-1)
        if h == 1:
            fwd = log_ret.shift(-1)
        else:
            # sum of log_ret[t+1 : t+1+h] — align to bar t
            fwd = log_ret[::-1].rolling(h).sum()[::-1].shift(-(h))

        rho, pval, n = ic_spearman(feat_series, fwd)
        records.append({
            "feature": feat_label,
            "horizon_h": h,
            "IC": round(rho, 5) if not np.isnan(rho) else np.nan,
            "p_value": round(pval, 6) if not np.isnan(pval) else np.nan,
            "n_valid": n,
            "significant": (pval < 0.05) if not np.isnan(pval) else False,
        })

results_df = pd.DataFrame(records)
results_df.to_csv(OUTPUT_CSV, index=False)
print(f"Saved: {OUTPUT_CSV}")

# ---------------------------------------------------------------------------
# Print formatted tables
# ---------------------------------------------------------------------------

# Table 1: Full IC by horizon for each feature
for feat_label in all_features.keys():
    print(f"\n{SEP}")
    print(f"FEATURE: {feat_label}")
    print(SEP)
    sub = results_df[results_df["feature"] == feat_label]
    print(f"  {'H':>4}  {'IC':>8}  {'p-value':>10}  {'n':>7}  {'sig':>4}")
    print(f"  {'-'*4}  {'-'*8}  {'-'*10}  {'-'*7}  {'-'*4}")
    for _, row in sub.iterrows():
        h    = int(row["horizon_h"])
        ic   = row["IC"]
        pv   = row["p_value"]
        n    = int(row["n_valid"])
        sig  = sig_stars(pv)
        if np.isnan(ic):
            print(f"  {h:>4}  {'NaN':>8}  {'NaN':>10}  {n:>7}  {sig}")
        else:
            print(f"  {h:>4}  {ic:>+8.5f}  {pv:>10.6f}  {n:>7}  {sig}")

# Table 2: Peak IC summary
print(f"\n{SEP}")
print("SUMMARY — PEAK IC HORIZON PER FEATURE (max |IC| among significant horizons)")
print(SEP)
print(f"  {'Feature':<26}  {'PeakH':>6}  {'IC':>8}  {'p-value':>10}  {'sig':>4}")
print(f"  {'-'*26}  {'-'*6}  {'-'*8}  {'-'*10}  {'-'*4}")

for feat_label in all_features.keys():
    sub = results_df[results_df["feature"] == feat_label].dropna(subset=["IC"])
    if len(sub) == 0:
        print(f"  {feat_label:<26}  {'N/A':>6}  {'N/A':>8}  {'N/A':>10}")
        continue
    # Prefer significant horizons; fall back to highest |IC|
    sig_sub = sub[sub["significant"] == True]
    pool = sig_sub if len(sig_sub) > 0 else sub
    best = pool.loc[pool["IC"].abs().idxmax()]
    print(f"  {feat_label:<26}  {int(best['horizon_h']):>6}  {best['IC']:>+8.5f}  "
          f"{best['p_value']:>10.6f}  {sig_stars(best['p_value'])}")

# Table 3: Cross-lambda comparison for curvature at H=1 and H=24
print(f"\n{SEP}")
print("CURVATURE: CROSS-LAMBDA IC COMPARISON")
print(SEP)
curv_rows = results_df[results_df["feature"].str.startswith("curv_λ")]
for h in [1, 2, 4, 12, 24]:
    h_rows = curv_rows[curv_rows["horizon_h"] == h]
    if len(h_rows) == 0:
        continue
    vals = []
    for lbl in LAMBDA_LABELS:
        feat_key = f"curv_λ{lbl}"
        r = h_rows[h_rows["feature"] == feat_key]
        if len(r) == 0:
            vals.append(f"{'N/A':>12}")
        else:
            ic  = r.iloc[0]["IC"]
            sig = sig_stars(r.iloc[0]["p_value"])
            vals.append(f"{ic:>+8.5f}{sig}" if not np.isnan(ic) else f"{'NaN':>11}")
    print(f"  H={h:>2}:  " + "  |  ".join(
        f"λ={lbl} {v}" for lbl, v in zip(LAMBDA_LABELS, vals)
    ))

print(f"\n{SEP}")
print("DONE")
print(SEP)
