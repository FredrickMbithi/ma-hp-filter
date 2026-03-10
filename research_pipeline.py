"""
Full Research Pipeline — Strategy 8.1 (MA + HP Filter, USDJPY H1)
==================================================================
Executes Steps 1–5 of the feature_hypothesis_log.md research plan:

  Step 1 — ADF stationarity tests on all 10 generators
  Step 2 — IC (Spearman rank correlation) tests: 1-bar, 4-bar, 24-bar
  Step 3 — Lambda calibration check; precompute and cache to interim/
  Step 4 — 72-combination parameter grid search with Deflated Sharpe
  Step 5 — Regime-conditional IC + backtest metrics per vol_regime

All output is printed for capture; heavy artefacts saved to data/interim/.
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import norm
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ_ROOT)

from src.data.loader import FXDataLoader
from src.features.generators import (
    atr,
    causal_hp_trend,
    hp_trend_slope,
    hp_trend_curvature,
    ma_crossover_age,
    ma_crossover_signal,
    ma_spread_on_trend,
    trend_deviation_from_ma,
    vol_regime,
)
from src.features.library import FeatureLibrary
from src.features.moving_average import sma
from src.features.returns import compute_log_returns
from src.backtest.engine import VectorizedBacktest, ZeroCost
from src.backtest.cost_model import FXCostModel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LAMBDAS = [1e6, 1e7, 1e8, 1e9, 3.9e9, 1e11]
LAMBDA_LABELS = ["1M", "10M", "100M", "1B", "3.9B", "100B"]
T1_VALUES = [24, 48, 72]
T2_VALUES = [120, 168, 240, 480]
HP_WINDOW = 500          # raised from 200: 10yr dataset (~17 500 bars) supports wider window for better HP endpoint stability
IC_HORIZONS = [1, 4, 24]
ADF_PVALUE_THRESHOLD = 0.05
COST_MODEL = FXCostModel(spread_bps=0.7, slippage_bps=0.2)  # 0.9 bps all-in USDJPY
INTERIM_DIR = os.path.join(PROJ_ROOT, "data", "interim")
os.makedirs(INTERIM_DIR, exist_ok=True)

SEP = "=" * 72

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def adf_summary(series: pd.Series, name: str) -> dict:
    """Run ADF on series and return a result dict."""
    clean = series.dropna()
    if len(clean) < 30:
        return {"feature": name, "n": len(clean), "statistic": None,
                "p_value": None, "stationary": None, "note": "insufficient data"}
    try:
        result = adfuller(clean, autolag="AIC")
        stationary = result[1] < ADF_PVALUE_THRESHOLD
        return {
            "feature": name,
            "n": len(clean),
            "statistic": round(result[0], 4),
            "p_value": round(result[1], 6),
            "critical_1pct": round(result[4]["1%"], 4),
            "critical_5pct": round(result[4]["5%"], 4),
            "stationary": stationary,
            "note": "I(0)" if stationary else "non-stationary (I(1) or higher)",
        }
    except Exception as e:
        return {"feature": name, "n": len(clean), "error": str(e)}


def ic_series(feature: pd.Series, fwd_returns: pd.Series,
              horizon: int, feature_name: str) -> dict:
    """Spearman rank IC of feature vs forward log-return at horizon."""
    valid = pd.concat([feature, fwd_returns], axis=1).dropna()
    if len(valid) < 30:
        return {"feature": feature_name, "horizon": horizon,
                "ic": None, "p_value": None, "n": len(valid)}
    rho, pval = stats.spearmanr(valid.iloc[:, 0], valid.iloc[:, 1])
    return {
        "feature": feature_name,
        "horizon": horizon,
        "ic": round(float(rho), 5),
        "p_value": round(float(pval), 6),
        "n": len(valid),
        "significant": pval < 0.05,
    }


def deflated_sharpe(observed_sr: float, n_obs: int, n_trials: int,
                    skew: float = 0.0, excess_kurt: float = 0.0) -> float:
    """
    Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

    Returns the probability that the observed SR exceeds the expected
    maximum of n_trials independent SR estimates from the null.
    """
    # Expected maximum Sharpe under null (Gaussian approximation)
    euler_gamma = 0.5772156649
    if n_trials <= 1:
        sr_bench = 0.0
    else:
        sr_bench = (
            (1 - euler_gamma) * norm.ppf(1 - 1.0 / n_trials)
            + euler_gamma * norm.ppf(1 - 1.0 / (n_trials * np.e))
        )

    # Variance of SR estimator
    var_sr = (1.0 - skew * observed_sr + (excess_kurt - 1.0) / 4.0 * observed_sr ** 2) / (n_obs - 1)
    if var_sr <= 0:
        var_sr = 1.0 / (n_obs - 1)

    # Deflated SR (probability)
    z = (observed_sr - sr_bench) / np.sqrt(var_sr)
    return float(norm.cdf(z))


def run_backtest(df: pd.DataFrame, signal: pd.Series) -> dict:
    """Run backtest and return summary metrics."""
    bt = VectorizedBacktest(data=df, signal=signal, cost_model=COST_MODEL)
    results = bt.run()
    net = results["net_returns"]
    clean = net.dropna()
    if len(clean) < 10:
        return {"sharpe": None, "annual_return": None, "max_dd": None}

    n_bars_per_year = 252 * 24  # H1 bars per year
    mean_ret = clean.mean()
    std_ret = clean.std()
    sharpe = (mean_ret / std_ret * np.sqrt(n_bars_per_year)) if std_ret > 0 else 0.0

    annual_return = mean_ret * n_bars_per_year

    # Max drawdown
    equity = results["equity"]
    rolling_max = equity.cummax()
    dd = (equity - rolling_max) / rolling_max
    max_dd = dd.min()

    # Trades count
    n_trades = int(results["trades"].abs().sum() / 2)

    # DSR
    net_clean = clean[clean != 0]
    skew = float(net_clean.skew()) if len(net_clean) > 3 else 0.0
    kurt = float(net_clean.kurtosis()) if len(net_clean) > 3 else 0.0
    dsr = deflated_sharpe(sharpe, len(clean), 72, skew, kurt)

    return {
        "sharpe": round(sharpe, 4),
        "annual_return": round(annual_return, 6),
        "max_dd": round(max_dd, 6),
        "n_trades": n_trades,
        "n_obs": len(clean),
        "dsr": round(dsr, 4),
    }


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print(SEP)
print("LOADING DATA")
print(SEP)

loader = FXDataLoader(os.path.join(PROJ_ROOT, "data", "raw"))
df = loader.load("USDJPY_10yr_1h_dukascopy")
print(f"Loaded: {len(df)} bars | {df.index[0]} → {df.index[-1]}")
print(f"Columns: {list(df.columns)}")

close = df["close"]
log_ret = compute_log_returns(close)

atr_20 = atr(df["high"], df["low"], df["close"], window=20)

# Forward log-returns for IC
fwd_1  = log_ret.shift(-1).rename("fwd_1")
fwd_4  = log_ret.rolling(4).sum().shift(-4).rename("fwd_4")
fwd_24 = log_ret.rolling(24).sum().shift(-24).rename("fwd_24")
fwd_returns = {1: fwd_1, 4: fwd_4, 24: fwd_24}

# Vol regime (for regime-conditional analysis)
regime_series = vol_regime(log_ret)
regime_counts = regime_series.value_counts().sort_index()
print(f"\nVol regime distribution:\n{regime_counts.astype(int).to_dict()}")

# ---------------------------------------------------------------------------
# STEP 3 — Lambda calibration: precompute and cache
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print(f"STEP 3: LAMBDA CALIBRATION + PRECOMPUTE (window={HP_WINDOW})")
print(SEP)

CACHE_FILE = os.path.join(INTERIM_DIR, f"hp_trends_window{HP_WINDOW}.csv")

if os.path.exists(CACHE_FILE):
    print(f"Cache found: {CACHE_FILE}")
    hp_trends = pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)
    print(f"Loaded cached trends: {hp_trends.shape}")
else:
    print(f"Computing causal HP trends for {len(LAMBDAS)} lambda values...")
    trends_dict = {}
    for lam, label in zip(LAMBDAS, LAMBDA_LABELS):
        t0 = time.time()
        print(f"  lambda={label:>6s} ({lam:.1e})...", end="", flush=True)
        trend = causal_hp_trend(close, lamb=lam, window=HP_WINDOW)
        elapsed = time.time() - t0
        print(f"  done in {elapsed:.1f}s | NaN count: {trend.isna().sum()}")
        trends_dict[f"hp_trend_{label}"] = trend

    hp_trends = pd.DataFrame(trends_dict)
    hp_trends.to_csv(CACHE_FILE)
    print(f"Saved to {CACHE_FILE}")

# Lambda calibration: correlation between adjacent lambda trends (slopes)
print("\nLambda calibration diagnostics:")
print("  Spearman(slope_i, slope_{i+1}) — should decrease as lambdas diverge:")
for i in range(len(LAMBDAS) - 1):
    col_a = f"hp_trend_{LAMBDA_LABELS[i]}"
    col_b = f"hp_trend_{LAMBDA_LABELS[i+1]}"
    if col_a in hp_trends.columns and col_b in hp_trends.columns:
        slope_a = hp_trends[col_a].diff()
        slope_b = hp_trends[col_b].diff()
        valid = pd.concat([slope_a, slope_b], axis=1).dropna()
        rho, _ = stats.spearmanr(valid.iloc[:, 0], valid.iloc[:, 1])
        # Check how distinct the trend values are
        trend_diff_std = (hp_trends[col_a] - hp_trends[col_b]).std()
        print(f"  {LAMBDA_LABELS[i]:>5s} vs {LAMBDA_LABELS[i+1]:>5s}: "
              f"slope_rho={rho:.4f}  trend_diff_std={trend_diff_std:.6f}")

# Range check: are endpoints reasonable?
print("\nTrend endpoint values at last bar (close=", round(float(close.iloc[-1]), 3), "):")
for label in LAMBDA_LABELS:
    col = f"hp_trend_{label}"
    if col in hp_trends.columns:
        last_val = hp_trends[col].dropna().iloc[-1]
        print(f"  lambda={label:>6s}: S*(last)={last_val:.3f}  "
              f"diff_from_close={last_val - float(close.iloc[-1]):.4f}")

# ---------------------------------------------------------------------------
# Build feature set for ADF + IC (using calibrated lambda 3.9B)
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("BUILDING FEATURES (λ=3.9B reference)")
print(SEP)

trend_ref = hp_trends["hp_trend_3.9B"]
slope_ref  = hp_trend_slope(trend_ref)
curv_ref   = hp_trend_curvature(trend_ref)

# Raw price features
ma50  = sma(close, 50)
ma200 = sma(close, 200)
raw_spread = (ma50 - ma200) / atr_20
raw_dev_20 = (close - sma(close, 20)) / atr_20

# HP-derived features (reference lambda 3.9B)
ma_spread_24_120  = ma_spread_on_trend(trend_ref, 24, 120, atr_20)
ma_spread_48_168  = ma_spread_on_trend(trend_ref, 48, 168, atr_20)
ma_spread_72_240  = ma_spread_on_trend(trend_ref, 72, 240, atr_20)
trend_dev_120     = trend_deviation_from_ma(trend_ref, 120, atr_20)
trend_dev_240     = trend_deviation_from_ma(trend_ref, 240, atr_20)
crossover_age_48_168 = ma_crossover_age(trend_ref, 48, 168)

features = {
    # HP-filter derived
    "hp_trend_level":       trend_ref,
    "hp_trend_slope":       slope_ref,
    "hp_trend_curvature":   curv_ref,
    "ma_spread_24_120":     ma_spread_24_120,
    "ma_spread_48_168":     ma_spread_48_168,
    "ma_spread_72_240":     ma_spread_72_240,
    "trend_dev_120":        trend_dev_120,
    "trend_dev_240":        trend_dev_240,
    "crossover_age_48_168": crossover_age_48_168,
    # Raw price features (baseline)
    "raw_ma_spread_50_200": raw_spread,
    "raw_dev_20ma":         raw_dev_20,
}

for name, feat in features.items():
    n_valid = feat.dropna().shape[0]
    print(f"  {name:<28s}: {n_valid:>5d} valid obs")

# ---------------------------------------------------------------------------
# STEP 1 — ADF Stationarity Tests
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("STEP 1: ADF STATIONARITY TESTS")
print(SEP)
print(f"{'Feature':<28} {'N':>6}  {'ADF Stat':>9}  {'p-value':>9}  {'Stationary?'}")
print("-" * 72)

adf_results = []
# Features where ADF is applicable (skip crossover signal — bounded discrete)
adf_features = {k: v for k, v in features.items()
                if k != "ma_crossover_signal"}
for name, feat in adf_features.items():
    r = adf_summary(feat, name)
    adf_results.append(r)
    stat_str = f"{r.get('statistic', 'N/A'):>9}" if r.get("statistic") is not None else "      N/A"
    p_str    = f"{r.get('p_value', 'N/A'):>9}" if r.get("p_value") is not None else "      N/A"
    flag     = "YES ✓" if r.get("stationary") else ("NO ✗" if r.get("stationary") is False else "N/A")
    print(f"  {name:<28} {r['n']:>6}  {stat_str}  {p_str}  {flag}")

# ---------------------------------------------------------------------------
# STEP 2 — IC Tests
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("STEP 2: IC TESTS (Spearman Rank Correlation vs Forward Returns)")
print(SEP)
print(f"{'Feature':<28} {'H=1':>9} {'H=4':>9} {'H=24':>9}  Signal?")
print("-" * 72)

ic_results = []
ic_table = {}  # name → {1: ic, 4: ic, 24: ic}

for name, feat in features.items():
    row = {}
    for h, fwd in fwd_returns.items():
        r = ic_series(feat, fwd, h, name)
        ic_results.append(r)
        row[h] = r
    ic_table[name] = row

    ic1  = row[1]["ic"]  if row[1]["ic"] is not None else float("nan")
    ic4  = row[4]["ic"]  if row[4]["ic"] is not None else float("nan")
    ic24 = row[24]["ic"] if row[24]["ic"] is not None else float("nan")

    # Gate: any IC with |value| > 0.02 and p < 0.05 counts
    has_signal = any(
        abs(row[h]["ic"] or 0) > 0.02 and row[h].get("significant", False)
        for h in IC_HORIZONS
    )
    flag = "PASS →" if has_signal else "GATE  "
    p1   = "*" if row[1].get("significant") else " "
    p4   = "*" if row[4].get("significant") else " "
    p24  = "*" if row[24].get("significant") else " "

    print(f"  {name:<28} {ic1:>8.4f}{p1} {ic4:>8.4f}{p4} {ic24:>8.4f}{p24}  {flag}")

print("  (* = p < 0.05)")

# ---------------------------------------------------------------------------
# STEP 4 — Parameter Grid Search (72 combinations)
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("STEP 4: PARAMETER GRID SEARCH (72 combinations)")
print(SEP)
print(f"{'λ':>6} {'T1':>4} {'T2':>4} {'Sharpe':>8} {'AnnRet':>8} {'MaxDD':>8} "
      f"{'Trades':>7} {'DSR':>7}  {'Obs':>6}")
print("-" * 72)

grid_results = []

for lam, label in zip(LAMBDAS, LAMBDA_LABELS):
    col = f"hp_trend_{label}"
    trend = hp_trends[col] if col in hp_trends.columns else None
    if trend is None:
        continue

    for t1 in T1_VALUES:
        for t2 in T2_VALUES:
            signal = ma_crossover_signal(trend, t1, t2)
            # Gate: skip if fewer than 100 valid non-nan observations
            valid_obs = signal.dropna().shape[0]
            if valid_obs < 100:
                continue

            metrics = run_backtest(df, signal)
            row = {
                "lambda": label,
                "t1": t1,
                "t2": t2,
                **metrics,
            }
            grid_results.append(row)

            sharpe  = metrics["sharpe"]  or 0.0
            ar      = metrics["annual_return"] or 0.0
            mdd     = metrics["max_dd"] or 0.0
            ntrades = metrics["n_trades"] or 0
            dsr     = metrics["dsr"] or 0.0
            nobs    = metrics["n_obs"] or 0

            flag = " ←" if dsr > 0.95 else ""
            print(f"  {label:>6} {t1:>4} {t2:>4}  "
                  f"{sharpe:>7.4f}  {ar:>7.4f}  {mdd:>8.4f}  "
                  f"{ntrades:>6}  {dsr:>6.4f}  {nobs:>6}{flag}")

grid_df = pd.DataFrame(grid_results)
if not grid_df.empty:
    grid_df.to_csv(os.path.join(INTERIM_DIR, "grid_results.csv"), index=False)
    print(f"\nGrid results saved to data/interim/grid_results.csv")

    # DSR-corrected ranking
    print("\nTop 10 combinations by DSR-corrected Sharpe:")
    top10 = (grid_df.dropna(subset=["sharpe", "dsr"])
             .sort_values(["dsr", "sharpe"], ascending=False)
             .head(10))
    print(top10[["lambda", "t1", "t2", "sharpe", "annual_return",
                 "max_dd", "n_trades", "dsr"]].to_string(index=False))

# ---------------------------------------------------------------------------
# STEP 5 — Regime-Conditional Analysis
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("STEP 5: REGIME-CONDITIONAL IC ANALYSIS")
print(SEP)

REGIME_NAMES = {0.0: "Low Vol", 1.0: "Med Vol", 2.0: "High Vol"}

regime_ic_results = {}

for reg_val, reg_name in REGIME_NAMES.items():
    mask = regime_series == reg_val
    n_bars = int(mask.sum())
    print(f"\n  Regime: {reg_name} ({n_bars} bars)")
    print(f"  {'Feature':<28} {'H=1':>9} {'H=4':>9} {'H=24':>9}")
    print("  " + "-" * 60)

    regime_ic_results[reg_name] = {}

    for name, feat in features.items():
        feat_reg = feat[mask]
        row = {}
        line_parts = []
        for h, fwd in fwd_returns.items():
            fwd_reg = fwd[mask]
            r = ic_series(feat_reg, fwd_reg, h, name)
            row[h] = r
            ic_val = r["ic"] if r["ic"] is not None else float("nan")
            sig_str = "*" if r.get("significant") else " "
            line_parts.append(f"{ic_val:>8.4f}{sig_str}")
        regime_ic_results[reg_name][name] = row
        print(f"  {name:<28} {'  '.join(line_parts)}")

# Best performing feature per regime
print("\n  Regime Best Features (by |IC| sum across horizons):")
for reg_name, feat_dict in regime_ic_results.items():
    scores = {}
    for fname, row in feat_dict.items():
        total_ic = sum(
            abs(row[h]["ic"]) for h in IC_HORIZONS
            if row[h]["ic"] is not None
        )
        scores[fname] = total_ic
    best = max(scores, key=scores.get) if scores else "—"
    best_score = scores.get(best, 0)
    print(f"    {reg_name}: {best} (Σ|IC|={best_score:.5f})")

# Regime-conditional backtest: best overall combination per regime
print(f"\n{SEP}")
print("STEP 5b: REGIME-CONDITIONAL BACKTEST (using best grid combinations)")
print(SEP)

if not grid_df.empty:
    # Use top 3 overall combinations
    top3 = (grid_df.dropna(subset=["sharpe", "dsr"])
            .sort_values(["dsr", "sharpe"], ascending=False)
            .head(3))

    for _, combo in top3.iterrows():
        label_c = combo["lambda"]
        t1_c = int(combo["t1"])
        t2_c = int(combo["t2"])
        col_c = f"hp_trend_{label_c}"
        trend_c = hp_trends[col_c]
        sig_c = ma_crossover_signal(trend_c, t1_c, t2_c)

        print(f"\n  Combination: λ={label_c}, T1={t1_c}, T2={t2_c}  "
              f"(Overall Sharpe={combo['sharpe']:.4f}, DSR={combo['dsr']:.4f})")
        print(f"  {'Regime':<12} {'Sharpe':>8} {'AnnRet':>9} {'MaxDD':>9} "
              f"{'Trades':>8} {'Obs':>6}")
        print("  " + "-" * 60)

        for reg_val, reg_name in REGIME_NAMES.items():
            mask = regime_series == reg_val
            df_reg = df[mask]
            sig_reg = sig_c[mask]
            if len(df_reg) < 50 or sig_reg.dropna().shape[0] < 10:
                print(f"  {reg_name:<12} {'INSUFFICIENT DATA':>40}")
                continue

            # Re-index for backtest (must be contiguous-seeming for engine)
            df_reg_reset = df_reg.copy()
            sig_reg_aligned = sig_reg.copy()
            m = run_backtest(df_reg_reset, sig_reg_aligned)
            print(f"  {reg_name:<12} {m['sharpe']:>8.4f}  "
                  f"{m['annual_return']:>9.5f}  {m['max_dd']:>9.6f}  "
                  f"{m['n_trades']:>8}  {m['n_obs']:>6}")

# ---------------------------------------------------------------------------
# STEP 1 REVISIT — ADF on ALL lambda trends and slopes
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("STEP 1 (EXTENDED): ADF ON ALL LAMBDA TRENDS AND SLOPES")
print(SEP)
print(f"{'Feature':<28} {'ADF Stat':>10}  {'p-value':>9}  {'Stationary?'}")
print("-" * 72)

for label in LAMBDA_LABELS:
    col = f"hp_trend_{label}"
    if col not in hp_trends.columns:
        continue
    trend_l = hp_trends[col]
    slope_l = trend_l.diff()
    curv_l  = trend_l.diff().diff()

    for feat_series, feat_name in [
        (trend_l, f"hp_trend_{label}"),
        (slope_l, f"hp_slope_{label}"),
        (curv_l,  f"hp_curv_{label}"),
    ]:
        r = adf_summary(feat_series, feat_name)
        stat_str = f"{r['statistic']:>10.4f}" if r.get("statistic") is not None else "       N/A"
        p_str    = f"{r['p_value']:>9.6f}" if r.get("p_value") is not None else "      N/A"
        flag     = "YES ✓" if r.get("stationary") else ("NO ✗" if r.get("stationary") is False else "N/A")
        print(f"  {feat_name:<28} {stat_str}  {p_str}  {flag}")

# ---------------------------------------------------------------------------
# Summary: IC gate results
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("SUMMARY: IC GATE RESULTS")
print(SEP)

ic_gate_pass = []
ic_gate_fail = []

for name, row in ic_table.items():
    any_signal = any(
        abs(row[h]["ic"] or 0) > 0.02 and row[h].get("significant", False)
        for h in IC_HORIZONS
    )
    if any_signal:
        best_h = max(IC_HORIZONS, key=lambda h: abs(row[h]["ic"] or 0))
        ic_gate_pass.append((name, row[best_h]["ic"], best_h, row[best_h]["p_value"]))
    else:
        ic_gate_fail.append(name)

print(f"\n  PASSED IC GATE ({len(ic_gate_pass)} features):")
for name, ic, h, p in sorted(ic_gate_pass, key=lambda x: abs(x[1]), reverse=True):
    print(f"    {name:<28} best_IC={ic:>8.5f} at H={h}  p={p:.5f}")

print(f"\n  FAILED IC GATE ({len(ic_gate_fail)} features) — DO NOT BACKTEST:")
for name in ic_gate_fail:
    print(f"    {name}")

# ---------------------------------------------------------------------------
# Save consolidated results
# ---------------------------------------------------------------------------
summary = {
    "run_date": "2026-03-10",
    "n_bars": len(df),
    "date_range": [str(df.index[0]), str(df.index[-1])],
    "hp_window": HP_WINDOW,
    "cost_bps": COST_MODEL.total_bps,
    "adf_results": adf_results,
    "ic_results": ic_results,
    "grid_results": grid_results,
}

with open(os.path.join(INTERIM_DIR, "pipeline_summary.json"), "w") as f:
    json.dump(summary, f, indent=2, default=str)

print(f"\n{SEP}")
print("Pipeline complete. Artefacts saved to data/interim/")
print(SEP)
