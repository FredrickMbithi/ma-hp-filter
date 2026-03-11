"""
Strategy B + EFC Filter — trend_dev_240 as Entry Quality Filter (USDJPY H1)
============================================================================
Tests whether adding a trend_dev_240 quality gate to the Strategy B (crossing
entry) signal improves precision by reducing spurious entries.

EFC (Entry Filter Criterion): Only take MA spread entries when the HP trend
itself is overextended from its own 240-bar moving average — i.e. when the
trend has over-shot far enough in one direction to make mean-reversion more
likely.

  trend_dev_240 = (hp_trend − sma(hp_trend, 240)) / ATR_20

An entry is permitted only when:
  |trend_dev_240| > filter_thresh   (HP trend overextended — more MR fuel)

  OR (alternative gate):
  sign(trend_dev_240) == sign(raw_spread)  (both signals aligned)

Filter implementation: after computing the crossing signal, entry bars where
the filter isn't met are removed along with their entire hold window.

Grid:
  MR params (fixed best IS combo): thresh=1.0σ, hold=48
  HP trend lambda for trend_dev_240: {1B}  (most useful per IC scan)
  filter_thresh (|trend_dev_240|):   {0.5, 1.0, 1.5}σ  + unfiltered baseline
  Alignment gate tested separately

Outputs:
  data/interim/strategy_b_efc_results.csv
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
    crossing_threshold_signal,
    trend_deviation_from_ma,
    vol_rolling_percentile,
)
from src.features.moving_average import sma
from src.features.returns import compute_log_returns
from src.backtest.engine import VectorizedBacktest
from src.backtest.cost_model import FXCostModel

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------
MA_SHORT        = 50
MA_LONG         = 200
ATR_WINDOW      = 20
SIGNAL_LOOKBACK = 252 * 24
THRESH_MR       = 1.0
HOLD_MR         = 48

TREND_DEV_WINDOW = 240      # SMA window for trend_dev_240
HP_FILTER_LAMBDA = "1B"     # use λ=1B (highest |IC| at H=1 in IC scan)
FILTER_THRESHOLDS = [0.5, 1.0, 1.5]   # σ levels for |trend_dev_240|

VOL_WINDOW   = 20
VOL_LOOKBACK = 252 * 24

N_BARS_PER_YEAR = 252 * 24
COST_BPS        = 0.9
IS_END    = "2021-12-31 23:59:59"
OOS_START = "2022-01-01"

# DSR accounts for unfiltered + 3 filter levels + alignment gate = 5 combos
N_TRIALS = 5

INTERIM_DIR = os.path.join(PROJ_ROOT, "data", "interim")
OUTPUT_CSV  = os.path.join(INTERIM_DIR, "strategy_b_efc_results.csv")
os.makedirs(INTERIM_DIR, exist_ok=True)

SEP  = "=" * 72
SEP2 = "-" * 72


def deflated_sharpe(sr, n_obs, n_trials):
    euler_gamma = 0.5772156649
    sr_bench = (
        (1 - euler_gamma) * norm.ppf(1 - 1.0 / n_trials)
        + euler_gamma * norm.ppf(1 - 1.0 / (n_trials * np.e))
    ) if n_trials > 1 else 0.0
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
    if len(df_slice) < 10 or signal_slice.abs().sum() == 0:
        return {"sharpe": None, "annual_return": None, "max_dd": None,
                "n_trades": 0, "n_obs": len(df_slice), "dsr": 0.0}
    try:
        bt = VectorizedBacktest(data=df_slice, signal=signal_slice, cost_model=cost_model)
        r  = bt.run()
    except Exception as exc:
        return {"error": str(exc)}
    return compute_metrics(r["net_returns"], r["equity"], r["trades"], n_trials)


def apply_entry_filter(signal: pd.Series, filter_ok: pd.Series) -> pd.Series:
    """Remove hold windows for entries where filter_ok is False.

    Scans through the signal and identifies new entries (transitions from 0 to
    ±1).  If filter_ok is False at the entry bar, the entire consecutive hold
    window (all bars while the original signal is non-zero for that trade) is
    zeroed out.  Positions that survive the filter are kept intact.

    Args:
        signal:    Original signal series {-1, 0, +1}.
        filter_ok: Boolean Series — True where entry is permitted.

    Returns:
        Filtered signal series.
    """
    sig = signal.values.copy()
    ok  = filter_ok.values
    n   = len(sig)

    in_blocked = False
    i = 0
    while i < n:
        if sig[i] != 0 and not in_blocked:
            # New hold window starts
            if not ok[i]:
                in_blocked = True
                sig[i] = 0.0
            # else: valid entry, keep it
        elif in_blocked:
            # We are zeroing out a blocked window
            sig[i] = 0.0
            if signal.iloc[i] == 0:
                # Original signal returned to flat — block ends
                in_blocked = False
        i += 1

    return pd.Series(sig, index=signal.index)


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
print(SEP)
print("STRATEGY B + EFC FILTER — trend_dev_240 as Entry Quality Gate")
print(f"Base signal: thresh={THRESH_MR}σ  hold={HOLD_MR}  crossing entry")
print(f"Filter feature: trend_dev_from_ma_{TREND_DEV_WINDOW}  (λ={HP_FILTER_LAMBDA})")
print(SEP)

loader = FXDataLoader(os.path.join(PROJ_ROOT, "data", "raw"))
df     = loader.load("USDJPY_10yr_1h_dukascopy")
print(f"Loaded: {len(df)} bars  |  {df.index[0]} → {df.index[-1]}")

close   = df["close"]
log_ret = compute_log_returns(close)

# ---------------------------------------------------------------------------
# Build MR signal on full history
# ---------------------------------------------------------------------------
print("\nBuilding base MR signal...")
atr_series = atr(df["high"], df["low"], df["close"], window=ATR_WINDOW)
ma_sh      = sma(close, MA_SHORT)
ma_lng     = sma(close, MA_LONG)
raw_spread = (ma_sh - ma_lng) / atr_series.replace(0, np.nan)
raw_spread = raw_spread.replace([np.inf, -np.inf], np.nan)

signal_base = crossing_threshold_signal(
    raw_spread,
    threshold_sigma=THRESH_MR,
    hold_bars=HOLD_MR,
    lookback=SIGNAL_LOOKBACK,
)
n_active = int((signal_base != 0).sum())
n_long   = int((signal_base > 0).sum())
n_short  = int((signal_base < 0).sum())
print(f"  Active bars: {n_active}  (Long: {n_long}, Short: {n_short})")

# ---------------------------------------------------------------------------
# Build trend_dev_240 filter feature
# ---------------------------------------------------------------------------
print(f"\nBuilding trend_dev_240 filter (λ={HP_FILTER_LAMBDA})...")
hp_cache_path = os.path.join(PROJ_ROOT, "data", "interim", "hp_trends_window500.csv")
if os.path.exists(hp_cache_path):
    cache_df   = pd.read_csv(hp_cache_path, index_col=0, parse_dates=True)
    cache_df.index = (cache_df.index.tz_localize("UTC")
                      if cache_df.index.tzinfo is None else cache_df.index)
    col_name   = f"hp_trend_{HP_FILTER_LAMBDA}"
    hp_trend   = cache_df[col_name].reindex(df.index)
    print(f"  Loaded from cache: {int(hp_trend.dropna().shape[0])} valid bars")
else:
    raise FileNotFoundError(f"HP trends cache not found: {hp_cache_path}")

trend_dev = trend_deviation_from_ma(hp_trend, window=TREND_DEV_WINDOW,
                                    atr_series=atr_series)
valid_td = trend_dev.dropna().shape[0]
print(f"  trend_dev_{TREND_DEV_WINDOW}: {valid_td} valid bars")
print(f"  Rolling std will be used as σ denominator for filter threshold")

# Compute rolling std of trend_dev for threshold calibration
td_std = trend_dev.rolling(SIGNAL_LOOKBACK, min_periods=SIGNAL_LOOKBACK // 2).std()

# Normalised trend_dev (in σ units, matching the filter threshold scale)
td_norm = trend_dev / td_std.replace(0, np.nan)
print(f"  |td_norm| > 0.5σ: {int((td_norm.abs() > 0.5).sum())} bars  "
      f"({int((td_norm.abs() > 0.5).sum()) / len(td_norm) * 100:.0f}%)")
print(f"  |td_norm| > 1.0σ: {int((td_norm.abs() > 1.0).sum())} bars  "
      f"({int((td_norm.abs() > 1.0).sum()) / len(td_norm) * 100:.0f}%)")
print(f"  |td_norm| > 1.5σ: {int((td_norm.abs() > 1.5).sum())} bars  "
      f"({int((td_norm.abs() > 1.5).sum()) / len(td_norm) * 100:.0f}%)")

is_mask  = df.index <= IS_END
oos_mask = df.index >= OOS_START

cost_model = FXCostModel(spread_bps=COST_BPS * 0.78, slippage_bps=COST_BPS * 0.22)

# ---------------------------------------------------------------------------
# Run experiments
# ---------------------------------------------------------------------------
rows = []

combos = [
    ("unfiltered", None, False),
] + [
    (f"|td|>{f}σ", f, False) for f in FILTER_THRESHOLDS
] + [
    ("alignment  ", None, True),
]

print(f"\nRunning {len(combos)} configurations...")

for label, f_thresh, use_alignment in combos:
    # Build filter series (True = entry permitted)
    if use_alignment:
        # Only enter when sign(td_norm) == sign(raw_spread)
        # i.e. HP trend overextended in same direction as the MA spread
        # (both pointing to mean-reversion in same direction)
        filter_ok = (np.sign(td_norm) == np.sign(raw_spread)) & td_norm.notna() & raw_spread.notna()
    elif f_thresh is not None:
        filter_ok = (td_norm.abs() > f_thresh) & td_norm.notna()
    else:
        filter_ok = pd.Series(True, index=df.index)

    # Apply filter to base signal
    if use_alignment or f_thresh is not None:
        sig_filtered = apply_entry_filter(signal_base, filter_ok)
    else:
        sig_filtered = signal_base.copy()

    n_kept = int((sig_filtered != 0).sum())
    pct_kept = n_kept / max(n_active, 1) * 100

    # IS
    m_is  = run_slice(df[is_mask],  sig_filtered[is_mask],  cost_model, N_TRIALS)
    # OOS
    m_oos = run_slice(df[oos_mask], sig_filtered[oos_mask], cost_model, n_trials=1)

    rows.append({
        "filter": label,
        "pct_active_kept": round(pct_kept, 1),
        "is_sharpe":  m_is.get("sharpe"),
        "is_annret":  m_is.get("annual_return"),
        "is_maxdd":   m_is.get("max_dd"),
        "is_trades":  m_is.get("n_trades", 0),
        "is_dsr":     m_is.get("dsr", 0),
        "oos_sharpe": m_oos.get("sharpe"),
        "oos_annret": m_oos.get("annual_return"),
        "oos_maxdd":  m_oos.get("max_dd"),
        "oos_trades": m_oos.get("n_trades", 0),
    })

results_df = pd.DataFrame(rows)
results_df.to_csv(OUTPUT_CSV, index=False)

# ---------------------------------------------------------------------------
# Print
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("  EFC FILTER RESULTS — IS and OOS comparison")
print(SEP)
print(f"  {'Filter':<14}  {'Kept%':>5}  "
      f"{'IS Sharpe':>9}  {'IS AnnRet':>9}  {'IS MaxDD':>8}  {'IS Trades':>9}  {'IS DSR':>7}  "
      f"{'OOS Sharpe':>10}  {'OOS AnnRet':>10}  {'OOS MaxDD':>9}  {'OOS Trades':>10}")
print(f"  {'─'*14}  {'─'*5}  "
      f"{'─'*9}  {'─'*9}  {'─'*8}  {'─'*9}  {'─'*7}  "
      f"{'─'*10}  {'─'*10}  {'─'*9}  {'─'*10}")

for _, r in results_df.iterrows():
    is_s  = f"{r['is_sharpe']:>+8.3f}" if r['is_sharpe'] is not None else "    N/A  "
    is_r  = f"{r['is_annret']*100:>+8.2f}%" if r['is_annret'] is not None else "    N/A  "
    is_d  = f"{r['is_maxdd']*100:>7.1f}%" if r['is_maxdd'] is not None else "   N/A  "
    oo_s  = f"{r['oos_sharpe']:>+9.3f}" if r['oos_sharpe'] is not None else "     N/A  "
    oo_r  = f"{r['oos_annret']*100:>+9.2f}%" if r['oos_annret'] is not None else "     N/A  "
    oo_d  = f"{r['oos_maxdd']*100:>8.1f}%" if r['oos_maxdd'] is not None else "    N/A  "
    print(f"  {r['filter']:<14}  {r['pct_active_kept']:>4.0f}%  "
          f"{is_s}  {is_r}  {is_d}  {int(r['is_trades']):>9}  {r['is_dsr']:>7.3f}  "
          f"{oo_s}  {oo_r}  {oo_d}  {int(r['oos_trades']):>10}")

print(f"\n{SEP2}")
# Compression stats
print("  Active bar retention when filter applied:")
for _, r in results_df.iterrows():
    print(f"    {r['filter']:<16} → {r['pct_active_kept']:>4.0f}% of unfiltered active bars kept  "
          f"({r['is_trades']:>3} IS trades, {r['oos_trades']:>3} OOS trades)")

print(f"\n{SEP2}")
print("  Interpretation guide:")
print("    A useful filter REDUCES trade count AND IMPROVES Sharpe per trade.")
print("    IS Sharpe improvement with fewer trades = genuine quality uplift.")
print("    IS Sharpe deterioration = filter is removing GOOD trades.")

print(f"\n{SEP}")
print(f"Saved → {OUTPUT_CSV}")
print(SEP)
