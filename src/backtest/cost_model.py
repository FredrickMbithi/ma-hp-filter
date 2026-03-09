"""
FX Transaction Cost Model
=========================
Covers the three components of trading friction in FX markets:

    Spread cost   – bid/ask spread paid on every fill.
    Slippage      – execution deviation from the quoted mid-price, driven by
                    latency, order size, and market liquidity.
    Market impact – price movement caused by the order itself (relevant only
                    for large sizes; negligible at retail/small-prop scale).

For liquid majors at retail/prop sizing the dominant costs are spread and
slippage.  Market impact is left as a future extension.

Typical all-in cost benchmarks (mid-2020s retail/ECN):
    EURUSD: ~0.8 bps   (0.6 pip spread + 0.2 pip slippage)
    GBPUSD: ~1.0 bps
    USDJPY: ~0.9 bps
    AUDUSD: ~1.2 bps

Why realistic costs matter
--------------------------
A strategy that returns 2 bps/trade gross needs > 60% win rate just to
break even at 1.5 bps all-in cost.  Backtests that omit or understate costs
systematically overstate edge.  Rule of thumb: if costs halve your Sharpe
the strategy is not tradeable.

Cost interface contract
-----------------------
Any object passed to VectorizedBacktest as `cost_model` must implement:

    compute(trades: pd.Series) -> pd.Series

where `trades` is the position diff series (signed, e.g. long→short = -2)
and the return is a non-negative cost series in the same units as returns
(i.e. already divided by price – expressed as a fraction, not pips).
"""

from __future__ import annotations

import pandas as pd


class FXCostModel:
    """
    Fixed spread + slippage cost model for FX instruments.

    Parameters
    ----------
    spread_bps : float
        Broker bid/ask spread in basis points (per side).
        1 bps = 0.01% = 0.0001.
    slippage_bps : float
        Expected execution slippage in basis points (per side).
        Accounts for latency, partial fills, and price movement between
        signal generation and order fill.

    Examples
    --------
    >>> model = FXCostModel(spread_bps=0.6, slippage_bps=0.2)
    >>> model.total_bps
    0.8
    >>> model.compute_cost(trade_size=100_000, side='buy')
    8.0
    """

    def __init__(self, spread_bps: float, slippage_bps: float) -> None:
        if spread_bps < 0:
            raise ValueError(f"spread_bps must be non-negative, got {spread_bps}")
        if slippage_bps < 0:
            raise ValueError(f"slippage_bps must be non-negative, got {slippage_bps}")

        self.spread = spread_bps
        self.slippage = slippage_bps

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def total_bps(self) -> float:
        """Total all-in cost per trade in basis points."""
        return self.spread + self.slippage

    @property
    def total_rate(self) -> float:
        """Total all-in cost as a decimal fraction (e.g. 0.8 bps → 0.00008)."""
        return self.total_bps / 10_000.0

    # ------------------------------------------------------------------
    # Single-trade scalar interface (matches the FXCostModel skeleton)
    # ------------------------------------------------------------------

    def compute_cost(self, trade_size: float, side: str) -> float:
        """
        Cost in the account's quote currency for one trade.

        Parameters
        ----------
        trade_size : float
            Notional size of the trade (e.g. units of base currency).
            Must be positive; use `side` to express direction.
        side : str
            'buy' or 'sell'.  Direction does not affect cost magnitude
            but is validated and can be used for logging / audit trails.

        Returns
        -------
        float
            Transaction cost in quote-currency terms:
                cost = trade_size * (spread_bps + slippage_bps) / 10_000

        Notes
        -----
        For the vectorized engine use :meth:`compute` instead.
        """
        side = side.lower()
        if side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got '{side}'")
        if trade_size < 0:
            raise ValueError(f"trade_size must be non-negative, got {trade_size}")

        return trade_size * self.total_rate

    # ------------------------------------------------------------------
    # Vectorized engine interface
    # ------------------------------------------------------------------

    def compute(self, trades: pd.Series) -> pd.Series:
        """
        Vectorized cost computation for use inside VectorizedBacktest.

        Parameters
        ----------
        trades : pd.Series
            Position diff series.  A value of +1 means new long unit, -1
            means closing a long (or new short unit), ±2 means a
            long→short or short→long flip.  The *magnitude* determines
            cost; direction does not affect it.

        Returns
        -------
        pd.Series
            Non-negative cost series expressed as a fraction of notional
            (same units as pct_change returns).  Index is identical to
            `trades`.

        Examples
        --------
        >>> import pandas as pd
        >>> model = FXCostModel(spread_bps=0.6, slippage_bps=0.2)
        >>> trades = pd.Series([1, 0, 0, -2, 0, 1])  # open, hold, flip, hold, close
        >>> model.compute(trades)
        0    0.00008
        1    0.00000
        2    0.00000
        3    0.00016
        4    0.00000
        5    0.00008
        dtype: float64
        """
        return trades.abs() * self.total_rate

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"FXCostModel(spread_bps={self.spread}, slippage_bps={self.slippage}, "
            f"total_bps={self.total_bps})"
        )


# ---------------------------------------------------------------------------
# Factory helpers – construct from config dict
# ---------------------------------------------------------------------------

def cost_model_from_config(config: dict) -> FXCostModel:
    """
    Build an FXCostModel from the project's config.yaml `execution` section.

    Expected keys:
        execution.slippage_bps  (float)
        execution.commission_bps  (float, treated as per-side spread proxy)

    Parameters
    ----------
    config : dict
        Full parsed config dict (e.g. from yaml.safe_load).

    Returns
    -------
    FXCostModel

    Example
    -------
    >>> import yaml
    >>> with open("config/config.yaml") as f:
    ...     cfg = yaml.safe_load(f)
    >>> model = cost_model_from_config(cfg)
    """
    execution = config.get("execution", {})
    spread_bps = float(execution.get("commission_bps", 1.0))
    slippage_bps = float(execution.get("slippage_bps", 2.0))
    return FXCostModel(spread_bps=spread_bps, slippage_bps=slippage_bps)
