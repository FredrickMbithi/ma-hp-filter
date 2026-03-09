"""
FX Rollover / Swap Cost Calculator
====================================
Swap (rollover) cost is the interest rate differential applied when an FX
position is held overnight.  Brokers pass on the interbank rate differential
(e.g. Fed Funds vs short-term JPY rates) minus a markup, and add a spread
on top.

Formula
-------
    swap_pnl = swap_rate_bps_per_day * position_size_lots * hold_days / 10_000

where `swap_rate_bps_per_day` is *signed* – positive means the position
earns, negative means it pays.

Long/short rates differ because the broker quotes two separate rates that
both include the markup but in opposite directions.  In carry trade pairs
(e.g. USDJPY) the long side typically earns while the short side pays more
than symmetry would suggest (the markup gap is the broker's cut).

Data source priority
--------------------
1. Historical swap rates loaded from  ``data/swap_rates/{SYMBOL}_swaps.csv``
   Expected CSV schema:
       date,long_bps,short_bps
       2024-01-02,-0.12,0.08
       ...
   Rates should already include broker markup (as downloaded from MT5).

2. If the file does not exist, SWAP_DEFAULTS is used.  These are rough
   mid-2020s estimates; **do not use in production** without real data.

TODO: Replace SWAP_DEFAULTS with MT5 historical download.
      In MT5: Tools → Options → Charts → tick "Show Trade Levels" to expose
      the swap tab per symbol.  Alternatively use pymt5 to pull historical
      swaps programmatically.

Usage
-----
>>> from src.backtest.swap_calculator import compute_swap_cost
>>> pnl = compute_swap_cost("USDJPY", position_size=1.0, hold_days=3)
>>> print(f"Swap PnL: {pnl:.6f} lots-equivalent")
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

import pandas as pd

# ---------------------------------------------------------------------------
# Default swap rate assumptions (bps per day, sign: + = position earns)
# Source: approximate retail ECN averages, mid-2020s.
# DO NOT use in production without replacing with broker historical data.
# ---------------------------------------------------------------------------
SWAP_DEFAULTS: dict[str, dict[str, float]] = {
    # Format: {symbol: {"long": bps/day, "short": bps/day}}
    # Positive = trader earns, negative = trader pays.
    "USDJPY": {"long":  0.50, "short": -1.20},   # high carry: long earns
    "EURUSD": {"long": -0.35, "short":  0.15},   # EUR usually pays
    "GBPUSD": {"long": -0.20, "short":  0.05},
    "AUDUSD": {"long":  0.10, "short": -0.80},
    "USDCHF": {"long":  0.40, "short": -1.10},
    "NZDUSD": {"long":  0.05, "short": -0.75},
    "USDCAD": {"long":  0.30, "short": -0.90},
    "EURGBP": {"long": -0.15, "short":  0.05},
    "EURJPY": {"long":  0.20, "short": -1.00},
    "GBPJPY": {"long":  0.35, "short": -1.30},
}

# Path relative to project root
_SWAP_RATES_DIR = Path(__file__).parents[2] / "data" / "swap_rates"

# Broker markup in bps applied on top of the interbank rate difference.
# This is already embedded in SWAP_DEFAULTS; adjust when using raw
# interbank data.
BROKER_MARKUP_BPS: float = 0.3


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _swap_file_path(symbol: str) -> Path:
    """Return the expected path for a symbol's historical swap CSV."""
    return _SWAP_RATES_DIR / f"{symbol.upper()}_swaps.csv"


def load_swap_rates(symbol: str) -> dict[str, Union[float, pd.DataFrame]]:
    """
    Load swap rates for a symbol.

    Returns a dict with keys ``'long'`` and ``'short'``.  Values are either
    a scalar float (constant rate from SWAP_DEFAULTS) or a
    ``pd.DataFrame`` with a ``DatetimeIndex`` and columns ``long_bps`` /
    ``short_bps`` (historical CSV).

    Parameters
    ----------
    symbol : str
        FX symbol, e.g. ``'USDJPY'``.

    Returns
    -------
    dict
        ``{'long': float | pd.DataFrame, 'short': float | pd.DataFrame,
           'source': 'csv' | 'default'}``
    """
    symbol = symbol.upper()
    fpath = _swap_file_path(symbol)

    if fpath.exists():
        df = pd.read_csv(fpath, parse_dates=["date"], index_col="date")
        df = df.sort_index()
        return {"long": df["long_bps"], "short": df["short_bps"], "source": "csv"}

    if symbol in SWAP_DEFAULTS:
        rates = SWAP_DEFAULTS[symbol]
        return {"long": rates["long"], "short": rates["short"], "source": "default"}

    raise KeyError(
        f"No swap rates found for '{symbol}'. "
        f"Add data/swap_rates/{symbol}_swaps.csv or extend SWAP_DEFAULTS."
    )


def _scalar_rate(rates: dict, side: str, date: Optional[pd.Timestamp] = None) -> float:
    """
    Resolve a scalar bps/day rate from whatever load_swap_rates returned.

    For CSV-based rates, `date` is used to look up the correct row (or the
    most recent available rate if the exact date has no entry).
    """
    value = rates[side]
    if isinstance(value, pd.Series):
        if date is None:
            # Use the last available rate
            return float(value.iloc[-1])
        # Find the most recent rate on or before `date`
        prior = value[value.index <= date]
        if prior.empty:
            return float(value.iloc[0])
        return float(prior.iloc[-1])
    return float(value)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_swap_cost(
    symbol: str,
    position_size: float,
    hold_days: int,
    side: str = "long",
    swap_rates: Optional[dict] = None,
    reference_date: Optional[pd.Timestamp] = None,
) -> float:
    """
    Compute the cumulative swap (rollover) cost for a held FX position.

    Parameters
    ----------
    symbol : str
        FX pair, e.g. ``'USDJPY'``.
    position_size : float
        Position size in standard lots (1 lot = 100,000 units of base
        currency).  Use fractional lots (e.g. 0.1) for mini lots.
    hold_days : int
        Number of calendar days the position is held overnight.
        Weekends count triple in standard FX (Friday roll covers
        Sat/Sun/Mon); pass the raw calendar days and this function
        applies a weekend multiplier automatically.
    side : str
        ``'long'`` or ``'short'``.  Determines which rate applies.
    swap_rates : dict, optional
        Pre-loaded rates from :func:`load_swap_rates`.  If ``None``,
        rates are loaded automatically (file or default).
    reference_date : pd.Timestamp, optional
        Reference date for historical CSV rate lookup.  Only used when
        swap_rates source is ``'csv'``.

    Returns
    -------
    float
        Swap profit/loss in units of bps * lots * days.
        Positive → position *earns* swap income.
        Negative → position *pays* swap cost.

    Notes
    -----
    Weekend triple-roll: FX swaps rolled on Wednesday cover three days
    (Wed/Thu/Fri settlement, Sat/Sun included).  The function applies a
    3× multiplier to any Wednesday rolls in the hold period.

    Swap = (rate_long_or_short - broker_markup_already_embedded) * size * days

    The broker markup is already embedded in SWAP_DEFAULTS and in the
    broker-downloaded CSV; do not double-subtract it.

    Examples
    --------
    >>> # USDJPY long for 5 days – position earns
    >>> compute_swap_cost("USDJPY", position_size=1.0, hold_days=5, side="long")
    2.5
    >>> # EURUSD long for 1 day – position pays
    >>> compute_swap_cost("EURUSD", position_size=1.0, hold_days=1, side="long")
    -0.35
    """
    side = side.lower()
    if side not in ("long", "short"):
        raise ValueError(f"side must be 'long' or 'short', got '{side}'")
    if position_size < 0:
        raise ValueError(f"position_size must be non-negative, got {position_size}")
    if hold_days < 0:
        raise ValueError(f"hold_days must be non-negative, got {hold_days}")
    if hold_days == 0 or position_size == 0:
        return 0.0

    if swap_rates is None:
        swap_rates = load_swap_rates(symbol)

    rate_bps = _scalar_rate(swap_rates, side, reference_date)

    # Weekend triple-roll approximation: roughly 2/7 of hold days fall on
    # a Wednesday, adding 2 extra days per such roll.
    # Exact: count Wednesdays in the hold period from reference_date.
    extra_days = _count_triple_roll_days(hold_days, reference_date)
    effective_days = hold_days + extra_days

    swap_pnl = rate_bps * position_size * effective_days

    return swap_pnl


def compute_swap_series(
    symbol: str,
    positions: pd.Series,
    position_size: float = 1.0,
    swap_rates: Optional[dict] = None,
) -> pd.Series:
    """
    Compute per-bar swap cost for a position time series.

    Applies the overnight swap rate for each bar where a non-zero position
    is held.  Useful for integrating swap costs directly into the backtest
    equity curve.

    Parameters
    ----------
    symbol : str
        FX pair.
    positions : pd.Series
        Position series with DatetimeIndex.  Values are {-1, 0, 1} or
        fractional lots.  Positive = long, negative = short.
    position_size : float
        Scaling factor (lots per unit of signal).
    swap_rates : dict, optional
        Pre-loaded rates.  Loaded automatically if None.

    Returns
    -------
    pd.Series
        Per-bar swap PnL in bps.  Positive = earned, Negative = paid.
        Zero wherever position = 0.
    """
    if swap_rates is None:
        swap_rates = load_swap_rates(symbol)

    result = pd.Series(0.0, index=positions.index)

    for ts, pos in positions.items():
        if pos == 0:
            continue
        side = "long" if pos > 0 else "short"
        rate = _scalar_rate(swap_rates, side, reference_date=ts)
        result[ts] = rate * abs(pos) * position_size

    return result


# ---------------------------------------------------------------------------
# Weekend triple-roll
# ---------------------------------------------------------------------------

def _count_triple_roll_days(hold_days: int, start_date: Optional[pd.Timestamp]) -> int:
    """
    Count additional days from Wednesday triple rolls in the hold period.

    In FX, Wednesday swap settles T+2 into Friday, so the broker applies a
    3× multiplier for Wednesday rolls.  Each Wednesday in the hold period
    adds 2 extra swap days.

    Returns the number of *extra* days to add to `hold_days`.
    """
    if start_date is None or hold_days <= 0:
        # Conservative estimate: ~1 Wednesday per 7 days → ~2 extra days per week
        return round(hold_days * 2 / 7)

    extra = 0
    current = pd.Timestamp(start_date)
    for _ in range(hold_days):
        if current.dayofweek == 2:  # Wednesday = 2
            extra += 2
        current += pd.Timedelta(days=1)
    return extra
