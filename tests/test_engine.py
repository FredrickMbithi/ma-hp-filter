"""
Tests for VectorizedBacktest engine.
"""

import pytest
import pandas as pd
import numpy as np

from src.backtest.engine import VectorizedBacktest, ZeroCost
from src.backtest.cost_model import FXCostModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_ohlc(n: int = 100, start_price: float = 1.10, drift: float = 0.0001) -> pd.DataFrame:
    """Synthetic OHLC with a slight upward drift."""
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    close = start_price * np.cumprod(1 + drift + np.random.default_rng(42).normal(0, 0.001, n))
    df = pd.DataFrame({
        "open":  close * 0.9999,
        "high":  close * 1.0005,
        "low":   close * 0.9995,
        "close": close,
        "volume": np.ones(n) * 1000,
    }, index=idx)
    return df


def _make_signal(n: int, value: float = 0.0) -> pd.Series:
    """Constant signal Series aligned with _make_ohlc(n)."""
    idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
    return pd.Series(value, index=idx)


# ---------------------------------------------------------------------------
# ZeroCost
# ---------------------------------------------------------------------------

class TestZeroCost:
    def test_returns_zero_series(self):
        trades = pd.Series([1.0, 0.0, -2.0, 1.0])
        costs = ZeroCost().compute(trades)
        assert (costs == 0.0).all()

    def test_output_index_preserved(self):
        idx = pd.date_range("2024-01-01", periods=4, freq="h")
        trades = pd.Series([1.0, 0.0, -1.0, 0.0], index=idx)
        costs = ZeroCost().compute(trades)
        assert costs.index.equals(idx)


# ---------------------------------------------------------------------------
# VectorizedBacktest – input validation
# ---------------------------------------------------------------------------

class TestEngineValidation:
    def test_missing_close_column_raises(self):
        df = _make_ohlc(10).drop(columns=["close"])
        signal = _make_signal(10)
        with pytest.raises(ValueError, match="close"):
            VectorizedBacktest(df, signal)

    def test_index_mismatch_raises(self):
        df = _make_ohlc(10)
        signal = _make_signal(5)  # wrong length
        with pytest.raises(ValueError, match="index"):
            VectorizedBacktest(df, signal)

    def test_empty_data_raises(self):
        df = _make_ohlc(0)
        signal = _make_signal(0)
        with pytest.raises(ValueError, match="empty"):
            VectorizedBacktest(df, signal)

    def test_non_datetime_index_raises(self):
        df = _make_ohlc(10).reset_index(drop=True)  # integer index
        signal = pd.Series(0.0, index=range(10))
        with pytest.raises(TypeError, match="DatetimeIndex"):
            VectorizedBacktest(df, signal)


# ---------------------------------------------------------------------------
# VectorizedBacktest – zero-signal baseline
# ---------------------------------------------------------------------------

class TestZeroSignal:
    def setup_method(self):
        self.df = _make_ohlc(50)
        signal = _make_signal(50, value=0.0)
        cost = FXCostModel(spread_bps=0.6, slippage_bps=0.2)
        self.bt = VectorizedBacktest(self.df, signal, cost)
        self.results = self.bt.run()

    def test_positions_are_all_zero(self):
        assert (self.results["positions"] == 0).all()

    def test_trades_are_all_zero(self):
        assert (self.results["trades"] == 0).all()

    def test_strategy_returns_are_zero(self):
        assert (self.results["strategy_returns"] == 0).all()

    def test_equity_stays_at_one(self):
        # All net returns should be zero → equity curve stays at 1.0
        # Use np.allclose because pytest.approx doesn't vectorize over Series
        assert np.allclose(self.results["equity"].values, 1.0, atol=1e-9)

    def test_costs_are_zero(self):
        assert (self.results["costs"] == 0).all()


# ---------------------------------------------------------------------------
# VectorizedBacktest – constant long signal
# ---------------------------------------------------------------------------

class TestConstantLongSignal:
    def setup_method(self):
        np.random.seed(99)
        self.df = _make_ohlc(100, drift=0.0002)  # persistent upward drift
        signal = _make_signal(100, value=1.0)
        self.bt = VectorizedBacktest(self.df, signal, ZeroCost())
        self.results = self.bt.run()

    def test_first_position_is_zero_due_to_shift(self):
        """Signals generated at T only take effect at T+1 (shift=1)."""
        assert self.results["positions"].iloc[0] == 0.0

    def test_subsequent_positions_are_long(self):
        assert (self.results["positions"].iloc[1:] == 1.0).all()

    def test_only_one_entry_trade(self):
        """Constant long after first bar: only one +1 trade at bar 1."""
        trades = self.results["trades"]
        non_zero = trades[trades != 0]
        assert len(non_zero) == 1
        assert non_zero.iloc[0] == pytest.approx(1.0)

    def test_equity_above_one_on_rising_prices(self):
        """With zero cost and upward drift, equity must end above 1.0."""
        assert self.results["equity"].iloc[-1] > 1.0


# ---------------------------------------------------------------------------
# VectorizedBacktest – long-to-short flip
# ---------------------------------------------------------------------------

class TestLongToShortFlip:
    """Validate that a flip generates |trade| = 2 and double the cost."""

    def setup_method(self):
        n = 10
        idx = pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC")
        close = pd.Series(np.linspace(1.10, 1.11, n), index=idx)
        df = pd.DataFrame({"open": close, "high": close, "low": close,
                           "close": close}, index=idx)

        # Signal: long for first 5 bars, short for last 5
        signal = pd.Series(
            [0, 0, 1, 1, 1, -1, -1, -1, -1, -1],
            index=idx, dtype=float
        )
        self.cost = FXCostModel(spread_bps=1.0, slippage_bps=0.0)
        self.bt = VectorizedBacktest(df, signal, self.cost)
        self.results = self.bt.run()

    def test_flip_produces_trade_of_minus_two(self):
        """When position changes from +1 to -1, trade = -2."""
        trades = self.results["trades"]
        assert -2.0 in trades.values

    def test_flip_cost_is_double(self):
        """Flip costs 2× a normal entry or exit."""
        costs = self.results["costs"]
        flip_cost = costs[costs > 0].max()
        single_cost = self.cost.total_rate * 1
        assert flip_cost == pytest.approx(2 * single_cost)


# ---------------------------------------------------------------------------
# VectorizedBacktest – output schema
# ---------------------------------------------------------------------------

class TestOutputSchema:
    def setup_method(self):
        df = _make_ohlc(20)
        signal = _make_signal(20, 1.0)
        self.results = VectorizedBacktest(df, signal).run()

    def test_all_required_keys_present(self):
        required = {"equity", "strategy_returns", "net_returns",
                    "costs", "positions", "trades"}
        assert required.issubset(self.results.keys())

    def test_all_series_same_length(self):
        lengths = {k: len(v) for k, v in self.results.items()}
        assert len(set(lengths.values())) == 1, f"Length mismatch: {lengths}"

    def test_equity_starts_near_one(self):
        # First equity value: no position held yet (shift(1)), so = 1.0
        assert self.results["equity"].iloc[0] == pytest.approx(1.0)

    def test_costs_non_negative(self):
        assert (self.results["costs"] >= 0).all()


# ---------------------------------------------------------------------------
# VectorizedBacktest – cost vs gross return comparison
# ---------------------------------------------------------------------------

class TestCostImpact:
    def test_costs_reduce_returns(self):
        df = _make_ohlc(200, drift=0.0001)
        signal = pd.Series(1.0, index=df.index)
        cost_model = FXCostModel(spread_bps=50.0, slippage_bps=50.0)  # deliberately extreme

        gross = VectorizedBacktest(df, signal, ZeroCost()).run()
        net = VectorizedBacktest(df, signal, cost_model).run()

        assert net["equity"].iloc[-1] < gross["equity"].iloc[-1]

    def test_total_costs_match_manual_calculation(self):
        # 4-bar round-trip: enter at bar 1, hold bar 2, exit at bar 3.
        # signal = [1, 1, 1, 0]
        # positions (shift 1) = [0, 1, 1, 1]   ← bar 3 position is still 1
        # To get the exit trade we need a 5th bar so the 0-signal at bar 3
        # becomes position 0 at bar 4.
        # Simpler: use signal = [1, 1, 0, 0] over 4 bars
        #   positions = [0, 1, 1, 0]
        #   trades    = [0, +1, 0, -1]  → 2 cost events
        idx = pd.date_range("2024-01-01", periods=4, freq="h", tz="UTC")
        close = pd.Series([1.10, 1.11, 1.12, 1.13], index=idx)
        df = pd.DataFrame({"open": close, "high": close, "low": close,
                           "close": close}, index=idx)
        signal = pd.Series([1.0, 1.0, 0.0, 0.0], index=idx)
        cost_model = FXCostModel(spread_bps=1.0, slippage_bps=0.0)

        results = VectorizedBacktest(df, signal, cost_model).run()

        # positions = [0, 1, 1, 0]; trades = [0, +1, 0, -1]
        # costs = [0, 1/10000, 0, 1/10000]
        expected_total = 2 * (1.0 / 10_000)
        assert results["costs"].sum() == pytest.approx(expected_total)
