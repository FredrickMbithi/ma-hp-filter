"""
Vectorized Backtest Engine
==========================
Simulates a single-instrument FX strategy over historical OHLC data using a
vectorized (whole-dataset) approach.

Design principles
-----------------
1. **Strategy-agnostic** – the engine knows nothing about moving averages,
   HP filters, z-scores, or any indicator.  It receives a signal Series and
   executes it.  If your engine references an indicator, the design is broken.

2. **T→T+1 execution** – signals generated at bar close T are filled at the
   next bar open T+1 (modelled as T+1 close for simplicity since we use
   close-to-close returns; see the spec for a more precise open-price model).

3. **Modular costs** – transaction costs are never hardcoded.  Any object
   implementing ``compute(trades: pd.Series) -> pd.Series`` can be passed.

Execution timeline
------------------
    Bar T   →  signal generated at close
    Bar T+1 →  position takes effect; return computed close-to-close

Pipeline
--------
    data (OHLC) → signal → [shift(1)] → positions → [diff()] → trades
                                                          ↓
                                                        costs
                                                          ↓
    market returns → strat_returns → strat_returns - costs → equity

Position / trade relationship
------------------------------
    Signal  :  1  →  position  :  0  1  1  -1  -1   0
    Trade   :                      +1  0  -2   0  -1

Switching from long to short requires two units of trade (close the long,
open the short).  The cost model correctly sees |trade| = 2 at the flip.

Output
------
``run()`` returns a dict with keys:

    equity            : pd.Series  – (1 + net_returns).cumprod()
    strategy_returns  : pd.Series  – gross returns before costs
    net_returns       : pd.Series  – returns after costs
    costs             : pd.Series  – per-bar cost as a fraction
    positions         : pd.Series  – {-1, 0, 1} shifted signal
    trades            : pd.Series  – position diff (entry/exit/flip events)

No metrics (Sharpe, drawdown, etc.) are computed here.  Those belong in a
separate ``metrics.py`` module.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd


# ---------------------------------------------------------------------------
# Cost model protocol (structural subtyping)
# ---------------------------------------------------------------------------

@runtime_checkable
class CostModel(Protocol):
    """
    Structural interface that any cost model passed to VectorizedBacktest
    must satisfy.

    A class satisfies this protocol if it has a ``compute`` method with the
    correct signature – no inheritance required.
    """

    def compute(self, trades: pd.Series) -> pd.Series:
        """
        Parameters
        ----------
        trades : pd.Series
            Position diff series.

        Returns
        -------
        pd.Series
            Non-negative cost per bar expressed as a decimal fraction.
        """
        ...


# ---------------------------------------------------------------------------
# Zero-cost sentinel (useful for gross-return benchmarks)
# ---------------------------------------------------------------------------

class ZeroCost:
    """A no-op cost model.  Returns zero cost for every bar."""

    def compute(self, trades: pd.Series) -> pd.Series:  # noqa: D102
        return pd.Series(0.0, index=trades.index)

    def __repr__(self) -> str:  # noqa: D105
        return "ZeroCost()"


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class VectorizedBacktest:
    """
    Vectorized single-instrument backtest engine.

    Parameters
    ----------
    data : pd.DataFrame
        OHLC data with DatetimeIndex and at minimum a ``close`` column.
        Expected output of ``FXDataLoader.load()``.
    signal : pd.Series
        Strategy signal with values in {-1, 0, 1} (or fractional, e.g.
        from volatility scaling).  Must share the same index as ``data``.
    cost_model : object implementing CostModel protocol
        Any object with a ``compute(trades: pd.Series) -> pd.Series``
        method.  Defaults to ``ZeroCost()`` for gross-return analysis.

    Examples
    --------
    >>> from src.backtest.engine import VectorizedBacktest, ZeroCost
    >>> from src.backtest.cost_model import FXCostModel
    >>> model = FXCostModel(spread_bps=0.6, slippage_bps=0.2)
    >>> bt = VectorizedBacktest(data=df, signal=signal, cost_model=model)
    >>> results = bt.run()
    >>> results["equity"].plot(title="Equity Curve")
    """

    def __init__(
        self,
        data: pd.DataFrame,
        signal: pd.Series,
        cost_model: object | None = None,
    ) -> None:
        self._validate_inputs(data, signal)
        self.data = data
        self.signal = signal
        self.cost_model = cost_model if cost_model is not None else ZeroCost()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_inputs(data: pd.DataFrame, signal: pd.Series) -> None:
        """Fail fast with a clear error rather than silent NaN propagation."""
        if "close" not in data.columns:
            raise ValueError(
                "data must contain a 'close' column.  "
                f"Found columns: {list(data.columns)}"
            )
        if not data.index.equals(signal.index):
            raise ValueError(
                "data and signal must share the same index.  "
                f"data has {len(data)} rows, signal has {len(signal)} rows."
            )
        if data.empty:
            raise ValueError("data is empty.")
        if not isinstance(data.index, pd.DatetimeIndex):
            raise TypeError(
                f"data.index must be a DatetimeIndex, got {type(data.index).__name__}."
            )

    # ------------------------------------------------------------------
    # Core run
    # ------------------------------------------------------------------

    def run(self) -> dict[str, pd.Series]:
        """
        Execute the backtest and return the results dict.

        Returns
        -------
        dict with keys:
            equity, strategy_returns, net_returns, costs, positions, trades
        """
        # Step 1: T → T+1 execution lag (critical: prevents lookahead bias)
        #   A signal generated at bar T's close is filled at bar T+1's open.
        #   With close-to-close returns this is approximated by shift(1).
        positions = self.signal.shift(1).fillna(0)

        # Step 2: Detect trades (position changes)
        #   trade = +1  → entered long (or covered a short)
        #   trade = -1  → closed long (or entered short)
        #   trade = +2  → flipped short→long (close short + open long)
        #   trade = -2  → flipped long→short
        trades = positions.diff().fillna(0)

        # Step 3: Market returns (close-to-close, fractional)
        returns = self.data["close"].pct_change().fillna(0)

        # Step 4: Gross strategy returns
        #   position * market_return = contribution from holding the position
        strategy_returns = positions * returns

        # Step 5: Transaction costs (applied at the bar where the trade occurs)
        costs = self.cost_model.compute(trades)

        # Step 6: Net returns
        net_returns = strategy_returns - costs

        # Step 7: Equity curve starting at 1.0
        equity = (1 + net_returns).cumprod()

        return {
            "equity": equity,
            "strategy_returns": strategy_returns,
            "net_returns": net_returns,
            "costs": costs,
            "positions": positions,
            "trades": trades,
        }

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def n_bars(self) -> int:
        """Total number of bars in the dataset."""
        return len(self.data)

    @property
    def n_trades(self) -> int:
        """Number of bars with a non-zero trade (entry/exit/flip)."""
        trades = self.signal.shift(1).fillna(0).diff().fillna(0)
        return int((trades != 0).sum())

    def __repr__(self) -> str:
        return (
            f"VectorizedBacktest("
            f"bars={self.n_bars}, "
            f"cost_model={self.cost_model!r})"
        )
