"""
Strategy C — Regime-Switching: MA Spread MR (low vol) + HP Trend TF (high vol)
================================================================================
Tests the hypothesis that two complementary signals can be combined via a
volatility-regime switch:

  LOW VOL  (vol_pct < vol_low):   MA spread crossing mean-reversion (Strategy B)
  HIGH VOL (vol_pct > vol_high):  HP trend MA crossover trend-following (original grid)
  MED VOL  (between thresholds):  Flat (no position)

Rationale from earlier diagnostics:
  - MA spread MR (IC peak H=24–48, decay peak H=48 at 1.0σ) works when price
    oscillates around a mean — i.e. low/medium volatility.
  - HP trend MA crossover follows sustained directional moves — works in HIGH vol
    (confirmed in original pipeline regime breakdown: High-vol Sharpe ≈ 0.99).
  - Year-by-year OOS breakdown shows 2024 (highest vol, 6.3%) accounts for the
    majority of OOS losses (-1.463 Sharpe). Replacing the MR signal with a TF
    signal in that regime is the direct fix.

Grid tested:
  MR side:  thresh_mr ∈ {1.0}  hold_mr ∈ {48}  (best IS Strategy B combo)
  TF side:  T1=72, T2 ∈ {240, 480}  (λ=10M HP trend; original grid benchmark)
  Vol thresholds: (vol_low, vol_high) ∈ {(25,75), (33,67), (40,60), (50,50)}

  (50,50) means switch at median vol — always active in one signal or the other.
  (33,67) means bottom-third/top-third active; middle-third flat.

Full IS/OOS split plus year-by-year breakdown for the best combo.

Outputs:
  data/interim/strategy_c_results_is.csv
  data/interim/strategy_c_results_oos.csv
  stdout
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
    causal_hp_trend,
    crossing_threshold_signal,
    ma_crossover_signal,
    vol_rolling_percentile,
)
from src.features.moving_average import sma
from src.features.returns import compute_log_returns
from src.backtest.engine import VectorizedBacktest
from src.backtest.cost_model import FXCostModel

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
# MR side (Strategy B best combo)
MA_SHORT        = 50
MA_LONG         = 200
ATR_WINDOW      = 20
SIGNAL_LOOKBACK = 252 * 24
THRESH_MR       = 1.0
HOLD_MR         = 48

# TF side (HP trend crossover, λ=10M)
LAMBDA_TF   = 1e7
HP_WINDOW   = 500
T1_TF       = 72
T2_OPTIONS  = [240, 480]

# Vol regime
VOL_WINDOW   = 20
VOL_LOOKBACK = 252 * 24
VOL_PAIRS    = [(25, 75), (33, 67), (40, 60), (50, 50)]

N_BARS_PER_YEAR = 252 * 24
COST_BPS        = 0.9
IS_END    = "2021-12-31 23:59:59"
OOS_START = "2022-01-01"

N_TRIALS = len(VOL_PAIRS) * len(T2_OPTIONS)  # for DSR

INTERIM_DIR    = os.path.join(PROJ_ROOT, "data", "interim")
OUTPUT_IS_CSV  = os.path.join(INTERIM_DIR, "strategy_c_results_is.csv")
OUTPUT_OOS_CSV = os.path.join(INTERIM_DIR, "strategy_c_results_oos.csv")
os.makedirs(INTERIM_DIR, exist_ok=True)

SEP  = "=" * 72
SEP2 = "-" * 72


def deflated_sharpe(sr, n_obs, n_trials):
    euler_gamma = 0.5772156649
    if n_trials <= 1:
        sr_bench = 0.0
    else:
        sr_bench = (
            (1 - euler_gamma) * norm.ppf(1 - 1.0 / n_trials)
            + euler_gamma * norm.ppf(1 - 1.0 / (n_trials * np.e))
        )
    var_sr = 1.0 / max(n_obs - 1, 1)
    z = (sr - sr_bench) / np.sqrt(var_sr)
    return float(norm.cdf(z))


def compute_metrics(net_returns, equity, trades, n_trials=1):
    net = net_returns.dropna()
    if len(net) < 10:
        return {"sharpe": None, "annual_return": None, "max_dd": None,
                "n_trades": 0, "n_obs": len(net), "dsr": 0.0}
    mean_ret = net.mean()
    std_ret  = net.std()
    sharpe   = (mean_ret / std_ret * np.sqrt(N_BARS_PER_YEAR)) if std_ret > 0 else 0.0
    ann_ret  = mean_ret * N_BARS_PER_YEAR
    rolling_max = equity.cummax()
    max_dd = float(((equity - rolling_max) / rolling_max).min())
    n_trades   = int(trades.abs().sum() / 2)
    dsr        = deflated_sharpe(sharpe, len(net), n_trials)
    return {
        "sharpe":        round(sharpe, 4),
        "annual_return": round(ann_ret, 6),
        "max_dd":        round(max_dd, 6),
        "n_trades":      n_trades,
        "n_obs":         len(net),
        "dsr":           round(dsr, 4),
    }


def run_slice(df_slice, signal_slice, cost_model, n_trials=1):
    if len(df_slice) < 10:
        return {"sharpe": None, "annual_return": None, "max_dd": None,
                "n_trades": 0, "n_obs": len(df_slice), "dsr": 0.0}
    try:
        bt = VectorizedBacktest(data=df_slice, signal=signal_slice,
                                cost_model=cost_model)
        r  = bt.run()
    except Exception as exc:
        return {"error": str(exc)}
    return compute_metrics(r["net_returns"], r["equity"], r["trades"], n_trials)


def yearly_sharpes(df, signal, cost_model):
    """Return dict {year: sharpe}."""
    out = {}
    for yr in sorted(df.index.year.unique()):
        mask = df.index.year == yr
        m    = run_slice(df[mask], signal[mask], cost_model)
        out[yr] = m.get("sharpe")
    return out


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print(SEP)
print("STRATEGY C — REGIME SWITCHING: MR (low vol) + TF (high vol)")
print(f"MR: thresh={THRESH_MR}σ hold={HOLD_MR}  |  TF: HP λ=10M MA(T1={T1_TF}, T2=...)")
print(SEP)

loader = FXDataLoader(os.path.join(PROJ_ROOT, "data", "raw"))
df     = loader.load("USDJPY_10yr_1h_dukascopy")
print(f"Loaded: {len(df)} bars  |  {df.index[0]} → {df.index[-1]}")

close   = df["close"]
log_ret = compute_log_returns(close)

# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
print("\nPrecomputing features...")

# MR signal: MA spread
atr_series = atr(df["high"], df["low"], df["close"], window=ATR_WINDOW)
ma_sh      = sma(close, MA_SHORT)
ma_lng     = sma(close, MA_LONG)
raw_spread = (ma_sh - ma_lng) / atr_series.replace(0, np.nan)
raw_spread = raw_spread.replace([np.inf, -np.inf], np.nan)

signal_mr = crossing_threshold_signal(
    raw_spread,
    threshold_sigma=THRESH_MR,
    hold_bars=HOLD_MR,
    lookback=SIGNAL_LOOKBACK,
)
print(f"  MR signal: {int((signal_mr != 0).sum())} active bars")

# Vol percentile
vol_pct = vol_rolling_percentile(log_ret, vol_window=VOL_WINDOW, lookback=VOL_LOOKBACK)
print(f"  vol_pct_rank: {int(vol_pct.dropna().shape[0])} valid bars")

# TF signal: HP trend MA crossover (λ=10M)
hp_cache_path = os.path.join(PROJ_ROOT, "data", "interim", "hp_trends_window500.csv")
if os.path.exists(hp_cache_path):
    cache_df = pd.read_csv(hp_cache_path, index_col=0, parse_dates=True)
    cache_df.index = cache_df.index.tz_localize("UTC") if cache_df.index.tzinfo is None else cache_df.index
    hp_trend_10m = cache_df["hp_trend_10M"].reindex(df.index)
    print(f"  hp_trend_10M (from cache): {int(hp_trend_10m.dropna().shape[0])} valid bars")
else:
    print("  Cache not found — computing hp_trend_10M (slow)...")
    hp_trend_10m = causal_hp_trend(close, lam=LAMBDA_TF, window=HP_WINDOW)
    print(f"  hp_trend_10M: {int(hp_trend_10m.dropna().shape[0])} valid bars")

tf_signals = {}
for T2 in T2_OPTIONS:
    sig = ma_crossover_signal(hp_trend_10m, T1_TF, T2)
    tf_signals[T2] = sig
    n_cross = int((sig.diff().abs() > 0).sum())
    print(f"  TF signal T2={T2}: {int(sig.dropna().shape[0])} valid bars, ~{n_cross} crossovers")

is_mask  = df.index <= IS_END
oos_mask = df.index >= OOS_START
print(f"\n  IS:  {int(is_mask.sum())} bars  ({df.index[is_mask][0].date()} → {df.index[is_mask][-1].date()})")
print(f"  OOS: {int(oos_mask.sum())} bars  ({df.index[oos_mask][0].date()} → {df.index[oos_mask][-1].date()})")

cost_model = FXCostModel(spread_bps=COST_BPS * 0.78, slippage_bps=COST_BPS * 0.22)

# ---------------------------------------------------------------------------
# Grid: combine MR + TF under vol regime
# ---------------------------------------------------------------------------
print(f"\nRunning {len(T2_OPTIONS) * len(VOL_PAIRS)} combinations...")

results_is  = []
results_oos = []

for T2 in T2_OPTIONS:
    sig_tf = tf_signals[T2]
    for vol_low, vol_high in VOL_PAIRS:
        # Build the combined signal
        # Low vol  → MR signal
        # High vol → TF signal
        # Med vol  → 0 (flat)
        sig_c = pd.Series(0.0, index=df.index)

        low_vol_mask   = (vol_pct < vol_low) & vol_pct.notna()
        high_vol_mask  = (vol_pct >= vol_high) & vol_pct.notna()

        sig_c[low_vol_mask]  = signal_mr[low_vol_mask]
        sig_c[high_vol_mask] = sig_tf[high_vol_mask].fillna(0)

        # Stats on composition
        mr_bars  = int(low_vol_mask.sum())
        tf_bars  = int(high_vol_mask.sum())
        med_bars = int((~low_vol_mask & ~high_vol_mask & vol_pct.notna()).sum())

        label = f"T2={T2} vol_split={vol_low}/{vol_high}"

        # IS
        m_is = run_slice(df[is_mask], sig_c[is_mask], cost_model, N_TRIALS)
        results_is.append({
            "T2": T2, "vol_low": vol_low, "vol_high": vol_high,
            "mr_pct": round(mr_bars / (mr_bars + tf_bars + med_bars) * 100, 1),
            "tf_pct": round(tf_bars / (mr_bars + tf_bars + med_bars) * 100, 1),
            **m_is,
        })

        # OOS
        m_oos = run_slice(df[oos_mask], sig_c[oos_mask], cost_model, n_trials=1)
        results_oos.append({
            "T2": T2, "vol_low": vol_low, "vol_high": vol_high,
            **m_oos,
        })

is_df  = pd.DataFrame(results_is)
oos_df = pd.DataFrame(results_oos)

is_df.to_csv(OUTPUT_IS_CSV,  index=False)
oos_df.to_csv(OUTPUT_OOS_CSV, index=False)

# ---------------------------------------------------------------------------
# Print IS results
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("  IN-SAMPLE RESULTS (sorted by Sharpe)")
print(SEP)
print(f"  {'T2':>4}  {'split':>8}  {'MR%':>4}  {'TF%':>4}  "
      f"{'Sharpe':>8}  {'AnnRet':>8}  {'MaxDD':>7}  {'Trades':>7}  {'DSR':>7}")
print(f"  {'─'*4}  {'─'*8}  {'─'*4}  {'─'*4}  "
      f"{'─'*8}  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*7}")
sorted_is = is_df.sort_values("sharpe", ascending=False)
for _, r in sorted_is.iterrows():
    if r.get("sharpe") is None:
        continue
    split = f"{int(r['vol_low'])}/{int(r['vol_high'])}"
    print(f"  {int(r['T2']):>4}  {split:>8}  {r['mr_pct']:>3.0f}%  {r['tf_pct']:>3.0f}%  "
          f"{r['sharpe']:>+8.3f}  {r['annual_return']*100:>+7.2f}%  "
          f"{r['max_dd']*100:>6.1f}%  {int(r['n_trades']):>7}  "
          f"{r['dsr']:>7.3f}")

# Compare: pure MR (Strategy B baseline)
print(f"\n{SEP2}")
pure_mr_is = run_slice(df[is_mask], signal_mr[is_mask], cost_model, N_TRIALS)
print(f"  Baseline (pure MR, Strategy B):  "
      f"Sharpe={pure_mr_is.get('sharpe', 'N/A'):>+.3f}  "
      f"Trades={pure_mr_is.get('n_trades', 0)}")

# Pure TF baselines
for T2 in T2_OPTIONS:
    m = run_slice(df[is_mask], tf_signals[T2][is_mask], cost_model, N_TRIALS)
    print(f"  Baseline (pure TF T2={T2}):         "
          f"Sharpe={m.get('sharpe', 'N/A'):>+.3f}  "
          f"Trades={m.get('n_trades', 0)}")

# ---------------------------------------------------------------------------
# Print OOS results alongside IS for best combo
# ---------------------------------------------------------------------------
# Pick best IS combo
best_is_idx = is_df["sharpe"].idxmax() if is_df["sharpe"].notna().any() else 0
best        = is_df.iloc[best_is_idx]
best_T2     = int(best["T2"])
best_vl     = int(best["vol_low"])
best_vh     = int(best["vol_high"])

oos_match = oos_df[(oos_df["T2"] == best_T2) &
                   (oos_df["vol_low"] == best_vl) &
                   (oos_df["vol_high"] == best_vh)]

print(f"\n{SEP}")
print(f"  BEST IS COMBO:  T2={best_T2}  split={best_vl}/{best_vh}")
print(f"  IS:  Sharpe={best.get('sharpe', 'N/A'):>+.3f}  "
      f"AnnRet={best.get('annual_return', 0)*100:>+.2f}%  "
      f"MaxDD={best.get('max_dd', 0)*100:>.1f}%  Trades={int(best.get('n_trades', 0))}")
if len(oos_match) > 0:
    oo = oos_match.iloc[0]
    print(f"  OOS: Sharpe={oo.get('sharpe', 'N/A'):>+.3f}  "
          f"AnnRet={oo.get('annual_return', 0)*100:>+.2f}%  "
          f"MaxDD={oo.get('max_dd', 0)*100:>.1f}%  Trades={int(oo.get('n_trades', 0))}")

# ---------------------------------------------------------------------------
# Year-by-year for best combo
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print(f"  YEAR-BY-YEAR BREAKDOWN FOR BEST COMBO (T2={best_T2}, split={best_vl}/{best_vh})")
print(SEP)

# Rebuild the best combined signal
sig_best = pd.Series(0.0, index=df.index)
lv_mask = (vol_pct < best_vl) & vol_pct.notna()
hv_mask = (vol_pct >= best_vh) & vol_pct.notna()
sig_best[lv_mask] = signal_mr[lv_mask]
sig_best[hv_mask] = tf_signals[best_T2][hv_mask].fillna(0)

print(f"  {'Year':>4}  {'Period':>3}  {'Sharpe':>7}  {'AnnRet':>8}  {'MaxDD':>7}  "
      f"{'Trades':>6}  {'Source'}")
print(f"  {'─'*4}  {'─'*3}  {'─'*7}  {'─'*8}  {'─'*7}  {'─'*6}  {'─'*20}")

# Also show what pure MR and pure TF would have done each year, for context
for yr in sorted(df.index.year.unique()):
    mask   = df.index.year == yr
    m_c    = run_slice(df[mask], sig_best[mask], cost_model)
    m_mr   = run_slice(df[mask], signal_mr[mask], cost_model)
    period = "IS " if yr <= 2021 else "OOS"
    s_c  = f"{m_c.get('sharpe', 0):>+.3f}" if m_c.get('sharpe') is not None else "  N/A"
    r_c  = f"{m_c.get('annual_return', 0)*100:>+.2f}%" if m_c.get('annual_return') is not None else "   N/A"
    d_c  = f"{m_c.get('max_dd', 0)*100:>.1f}%" if m_c.get('max_dd') is not None else "  N/A"
    s_mr = f"{m_mr.get('sharpe', 0):>+.3f}" if m_mr.get('sharpe') is not None else "  N/A"
    print(f"  {yr:>4}  {period:>3}  {s_c:>7}  {r_c:>8}  {d_c:>7}  "
          f"{m_c.get('n_trades', 0):>6}  (MR-only: {s_mr})")

print(f"\n{SEP}")
print(f"Saved IS  → {OUTPUT_IS_CSV}")
print(f"Saved OOS → {OUTPUT_OOS_CSV}")
print(SEP)
