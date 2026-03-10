"""
Reversed Signal Grid Search — Strategy 8.1 (MA + HP Filter, USDJPY H1)
=======================================================================
Re-runs the 72-combination parameter grid with three signal variants to
isolate the contribution of each filter layer:

  Variant A  —  Reversed MA crossover only
                (short when sma(T1) > sma(T2))
  Variant B  —  Reversed + vol_regime == 1 gate (Med Vol only)
  Variant C  —  Reversed + vol_regime + directional trend_dev_240 filter
                (short only when trend_dev > +1.5; long only when < −1.5)

Grid: 6λ × 3T1 × 4T2 = 72 combinations per variant → 216 rows total.
Quality filter (trend_dev_240) is fixed at λ=3.9B to match the IC analysis.

Outputs:
  data/interim/reversed_grid_results.csv   — full 216-row results table
  stdout                                    — ranked top-10 per variant +
                                             focused comparison for best combo

Usage:
  python experiments/reversed_grid_search.py
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
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
    ma_crossover_signal,
    trend_deviation_from_ma,
    vol_regime,
)
from src.features.returns import compute_log_returns
from src.backtest.engine import VectorizedBacktest
from src.backtest.cost_model import FXCostModel

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LAMBDAS = [1e6, 1e7, 1e8, 1e9, 3.9e9, 1e11]
LAMBDA_LABELS = ["1M", "10M", "100M", "1B", "3.9B", "100B"]
T1_VALUES = [24, 48, 72]
T2_VALUES = [120, 168, 240, 480]

HP_WINDOW = 500             # Matches updated research_pipeline.py (10yr dataset)
ATR_WINDOW = 20
TREND_DEV_T2 = 240          # Long MA window for quality filter
TREND_DEV_THRESHOLD = 1.5   # σ units; directional — see apply_trend_dev_filter()
QUALITY_FILTER_LAMBDA = 3.9e9  # Fixed λ for trend_dev quality filter (matches IC analysis)
MIN_TRADES = 30             # Below this: flag as UNDERPOWERED in output
N_TRIALS = 72               # For DSR multiple-testing correction (full original grid size)
N_BARS_PER_YEAR = 252 * 24  # H1 bars per trading year

COST_MODEL = FXCostModel(spread_bps=0.7, slippage_bps=0.2)  # 0.9 bps all-in

INTERIM_DIR = os.path.join(PROJ_ROOT, "data", "interim")
OUTPUT_CSV = os.path.join(INTERIM_DIR, "reversed_grid_results.csv")
os.makedirs(INTERIM_DIR, exist_ok=True)

SEP = "=" * 72
SEP2 = "-" * 72

# ---------------------------------------------------------------------------
# Metric helpers (adapted from research_pipeline.py)
# ---------------------------------------------------------------------------

def deflated_sharpe(
    observed_sr: float,
    n_obs: int,
    n_trials: int,
    skew: float = 0.0,
    excess_kurt: float = 0.0,
) -> float:
    """Probability that observed SR exceeds expected max under null (72 trials)."""
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
    }


# ---------------------------------------------------------------------------
# Signal builder functions
# ---------------------------------------------------------------------------

def build_reversed_signal(trend: pd.Series, t1: int, t2: int) -> pd.Series:
    """Invert the MA crossover signal: short when sma(T1) > sma(T2)."""
    raw = ma_crossover_signal(trend, t1, t2)
    return -1.0 * raw


def apply_regime_gate(signal: pd.Series, regime: pd.Series) -> pd.Series:
    """Zero-out all bars not in medium volatility (regime != 1)."""
    out = signal.copy()
    out[regime != 1] = 0.0
    return out


def apply_trend_dev_filter(
    signal: pd.Series, trend_dev: pd.Series, threshold: float = 1.5
) -> pd.Series:
    """
    Directional exhaustion quality filter.

    Short entry (-1) requires trend_dev > +threshold (overextended above own MA).
    Long entry (+1) requires trend_dev < -threshold (overextended below own MA).
    Entries that don't meet the condition are suppressed (set to 0).
    """
    out = signal.copy()
    # Block shorts that are NOT in an exhaustion zone above
    out[(signal == -1.0) & (trend_dev <= threshold)] = 0.0
    # Block longs that are NOT in an exhaustion zone below
    out[(signal == 1.0) & (trend_dev >= -threshold)] = 0.0
    return out


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print(SEP)
print("REVERSED SIGNAL GRID SEARCH — Strategy 8.1 (USDJPY H1)")
print(SEP)

loader = FXDataLoader(os.path.join(PROJ_ROOT, "data", "raw"))
df = loader.load("USDJPY_10yr_1h_dukascopy")
print(f"Loaded: {len(df)} bars  |  {df.index[0]} → {df.index[-1]}")

close = df["close"]
log_ret = compute_log_returns(close)

# ---------------------------------------------------------------------------
# Shared precomputation (once, not per-loop)
# ---------------------------------------------------------------------------
print("\nPrecomputing shared series...")

atr_series = atr(df["high"], df["low"], df["close"], window=ATR_WINDOW)

# Vol regime (for gate)
regime_series = vol_regime(log_ret, window=20, classification_window=252)
regime_counts = regime_series.value_counts().sort_index()
print(f"  vol_regime distribution: Low={regime_counts.get(0,0)}, "
      f"Med={regime_counts.get(1,0)}, High={regime_counts.get(2,0)} bars")

# HP trends — check cache, compute if missing
CACHE_FILE = os.path.join(INTERIM_DIR, f"hp_trends_window{HP_WINDOW}.csv")
if os.path.exists(CACHE_FILE):
    hp_trends = pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)
    print(f"  Loaded cached HP trends: {hp_trends.shape}")
else:
    print(f"  Computing HP trends for {len(LAMBDAS)} lambda values (window={HP_WINDOW})...")
    trends_dict = {}
    for lam, label in zip(LAMBDAS, LAMBDA_LABELS):
        col = f"hp_trend_{label}"
        trends_dict[col] = causal_hp_trend(close, lamb=lam, window=HP_WINDOW)
        print(f"    λ={label} done")
    hp_trends = pd.DataFrame(trends_dict, index=close.index)
    hp_trends.to_csv(CACHE_FILE)
    print(f"  Saved to {CACHE_FILE}")

# Trend deviation quality filter — fixed at λ=3.9B (matches IC analysis)
trend_quality = hp_trends["hp_trend_3.9B"]
trend_dev_quality = trend_deviation_from_ma(
    trend_quality, window=TREND_DEV_T2, atr_series=atr_series
)
exhaustion_bars = int(((trend_dev_quality > TREND_DEV_THRESHOLD) |
                       (trend_dev_quality < -TREND_DEV_THRESHOLD)).sum())
print(f"  trend_dev_240 quality filter: {exhaustion_bars} bars in exhaustion zones "
      f"(|dev| > {TREND_DEV_THRESHOLD}σ) out of {trend_dev_quality.dropna().shape[0]} valid")

# ---------------------------------------------------------------------------
# Grid search loop
# ---------------------------------------------------------------------------
print(f"\nRunning {len(LAMBDAS) * len(T1_VALUES) * len(T2_VALUES)} combinations "
      f"× 3 variants...")

VARIANTS = ["reversed", "regime_gated", "full_filter"]
results = []

for lam, label in zip(LAMBDAS, LAMBDA_LABELS):
    col = f"hp_trend_{label}"
    trend = hp_trends[col]

    for t1 in T1_VALUES:
        for t2 in T2_VALUES:
            # --- Build the three signal variants ---
            sig_rev = build_reversed_signal(trend, t1, t2)

            sig_regime = apply_regime_gate(sig_rev, regime_series)

            sig_full = apply_trend_dev_filter(
                apply_regime_gate(sig_rev, regime_series),
                trend_dev_quality,
                threshold=TREND_DEV_THRESHOLD,
            )

            for variant, signal in [
                ("reversed", sig_rev),
                ("regime_gated", sig_regime),
                ("full_filter", sig_full),
            ]:
                valid_obs = signal.dropna().shape[0]
                n_nonzero = int((signal != 0).sum())
                row_base = {
                    "lambda": label,
                    "t1": t1,
                    "t2": t2,
                    "variant": variant,
                    "valid_obs": valid_obs,
                    "active_bars": n_nonzero,
                }

                if n_nonzero < 2:
                    # No signal at all — record as empty
                    results.append({**row_base, "sharpe": None, "annual_return": None,
                                    "max_dd": None, "n_trades": 0, "n_obs": valid_obs,
                                    "dsr": 0.0, "underpowered": True})
                    continue

                metrics = run_backtest(df, signal)
                if "error" in metrics:
                    results.append({**row_base, **{"sharpe": None, "annual_return": None,
                                    "max_dd": None, "n_trades": 0, "n_obs": valid_obs,
                                    "dsr": 0.0, "underpowered": True,
                                    "error": metrics["error"]}})
                    continue

                underpowered = (metrics.get("n_trades", 0) or 0) < MIN_TRADES
                results.append({**row_base, **metrics, "underpowered": underpowered})

# ---------------------------------------------------------------------------
# Build results DataFrame and save
# ---------------------------------------------------------------------------
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved {len(results_df)} rows → {OUTPUT_CSV}")

# ---------------------------------------------------------------------------
# Summary output
# ---------------------------------------------------------------------------

def print_top10(df: pd.DataFrame, variant_name: str) -> None:
    sub = df[df["variant"] == variant_name].copy()
    sub = sub[sub["sharpe"].notna()].sort_values("dsr", ascending=False)
    print(f"\n{'─'*72}")
    print(f"  VARIANT: {variant_name.upper().replace('_', ' ')}  "
          f"(top 10 by DSR)")
    print(f"{'─'*72}")
    print(f"  {'λ':>5}  {'T1':>3}  {'T2':>4}  {'Sharpe':>8}  "
          f"{'AnnRet':>8}  {'MaxDD':>7}  {'Trades':>7}  {'DSR':>7}  {'Flag'}")
    print(f"  {'─'*5}  {'─'*3}  {'─'*4}  {'─'*8}  "
          f"{'─'*8}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*12}")
    for _, r in sub.head(10).iterrows():
        flag = "UNDERPWRD" if r.get("underpowered") else (
            "COLLAPSED" if r["lambda"] in {"1B", "3.9B", "100B"} else ""
        )
        sharpe_str = f"{r['sharpe']:+.3f}" if pd.notna(r["sharpe"]) else "  N/A "
        ret_str = f"{r['annual_return']*100:+.2f}%" if pd.notna(r["annual_return"]) else "  N/A "
        dd_str = f"{r['max_dd']*100:.1f}%" if pd.notna(r["max_dd"]) else "  N/A "
        print(f"  {r['lambda']:>5}  {r['t1']:>3}  {r['t2']:>4}  "
              f"{sharpe_str:>8}  {ret_str:>8}  {dd_str:>7}  "
              f"{int(r['n_trades']):>7}  {r['dsr']:>7.3f}  {flag}")


for v in VARIANTS:
    print_top10(results_df, v)

# ---------------------------------------------------------------------------
# Focused comparison: best original combo (λ=1B, T1=72, T2=240)
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("  FOCUSED COMPARISON: λ=1B, T1=72, T2=240")
print(f"  (Best combo from original failing grid — Sharpe was +0.258, DSR=0.000)")
print(SEP)
print(f"  {'Variant':<20}  {'Sharpe':>8}  {'AnnRet':>8}  "
      f"{'MaxDD':>7}  {'Trades':>7}  {'DSR':>7}  {'Underpowered'}")
print(f"  {'─'*20}  {'─'*8}  {'─'*8}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*12}")
print(f"  {'[original direction]':<20}  {'+0.258':>8}  {'+1.73%':>8}  "
      f"{'-6.3%':>7}  {'10':>7}  {'0.000':>7}  YES")
for v in VARIANTS:
    row = results_df[
        (results_df["variant"] == v) &
        (results_df["lambda"] == "1B") &
        (results_df["t1"] == 72) &
        (results_df["t2"] == 240)
    ]
    if row.empty:
        continue
    r = row.iloc[0]
    sharpe_str = f"{r['sharpe']:+.3f}" if pd.notna(r["sharpe"]) else "  N/A "
    ret_str = f"{r['annual_return']*100:+.2f}%" if pd.notna(r["annual_return"]) else "  N/A "
    dd_str = f"{r['max_dd']*100:.1f}%" if pd.notna(r["max_dd"]) else "  N/A "
    up = "YES" if r.get("underpowered") else "no"
    print(f"  {v:<20}  {sharpe_str:>8}  {ret_str:>8}  "
          f"{dd_str:>7}  {int(r['n_trades']):>7}  {r['dsr']:>7.3f}  {up}")

# ---------------------------------------------------------------------------
# Lambda collapse note
# ---------------------------------------------------------------------------
print(f"\n{SEP2}")
print("  NOTE: λ ∈ {{1B, 3.9B, 100B}} produce near-identical signals on this")
print("  dataset (slope ρ ≥ 0.9995). Rows marked COLLAPSED are not independent")
print("  tests — effective grid is 36 combinations, not 72.")
print(f"{SEP2}")

# ---------------------------------------------------------------------------
# Aggregate: best per variant (pass/fail DSR > 0.95)
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("  AGGREGATE SUMMARY")
print(SEP)
for v in VARIANTS:
    sub = results_df[(results_df["variant"] == v) & results_df["sharpe"].notna()]
    n_pos_sharpe = int((sub["sharpe"] > 0).sum())
    n_dsr_pass = int((sub["dsr"] > 0.95).sum())
    n_underpowered = int(sub["underpowered"].sum())
    best = sub.sort_values("dsr", ascending=False).head(1)
    best_str = ""
    if not best.empty:
        r = best.iloc[0]
        best_str = (f"best: λ={r['lambda']} T1={r['t1']} T2={r['t2']} "
                    f"Sharpe={r['sharpe']:+.3f} DSR={r['dsr']:.3f}")
    print(f"  {v:<20}  Sharpe>0: {n_pos_sharpe:>3}/72  "
          f"DSR>0.95: {n_dsr_pass:>2}/72  Underpowered: {n_underpowered:>2}  |  {best_str}")

print(f"\n{SEP}")
print(f"Done. Results saved to {OUTPUT_CSV}")
print(SEP)
