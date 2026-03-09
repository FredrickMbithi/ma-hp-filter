"""
Tests for FXCostModel and cost_model_from_config.
"""

import pytest
import pandas as pd
import numpy as np

from src.backtest.cost_model import FXCostModel, cost_model_from_config


# ---------------------------------------------------------------------------
# FXCostModel – construction
# ---------------------------------------------------------------------------

class TestFXCostModelConstruction:
    def test_stores_spread_and_slippage(self):
        model = FXCostModel(spread_bps=0.6, slippage_bps=0.2)
        assert model.spread == 0.6
        assert model.slippage == 0.2

    def test_total_bps(self):
        model = FXCostModel(spread_bps=0.6, slippage_bps=0.2)
        assert model.total_bps == pytest.approx(0.8)

    def test_total_rate(self):
        model = FXCostModel(spread_bps=0.6, slippage_bps=0.2)
        assert model.total_rate == pytest.approx(0.8 / 10_000)

    def test_zero_costs_allowed(self):
        model = FXCostModel(spread_bps=0.0, slippage_bps=0.0)
        assert model.total_bps == 0.0

    def test_negative_spread_raises(self):
        with pytest.raises(ValueError, match="spread_bps"):
            FXCostModel(spread_bps=-0.1, slippage_bps=0.2)

    def test_negative_slippage_raises(self):
        with pytest.raises(ValueError, match="slippage_bps"):
            FXCostModel(spread_bps=0.6, slippage_bps=-0.1)

    def test_repr_contains_key_info(self):
        model = FXCostModel(spread_bps=0.6, slippage_bps=0.2)
        r = repr(model)
        assert "0.6" in r
        assert "0.2" in r
        assert "0.8" in r


# ---------------------------------------------------------------------------
# FXCostModel.compute_cost – scalar interface
# ---------------------------------------------------------------------------

class TestComputeCostScalar:
    def setup_method(self):
        # EURUSD-like: 0.6 pip spread + 0.2 pip slippage = 0.8 bps total
        self.model = FXCostModel(spread_bps=0.6, slippage_bps=0.2)

    def test_buy_cost(self):
        cost = self.model.compute_cost(trade_size=100_000, side="buy")
        # 100_000 * 0.8 / 10_000 = 8.0
        assert cost == pytest.approx(8.0)

    def test_sell_cost(self):
        cost = self.model.compute_cost(trade_size=100_000, side="sell")
        assert cost == pytest.approx(8.0)

    def test_cost_symmetry_buy_vs_sell(self):
        buy_cost = self.model.compute_cost(100_000, "buy")
        sell_cost = self.model.compute_cost(100_000, "sell")
        assert buy_cost == sell_cost

    def test_zero_size_costs_zero(self):
        assert self.model.compute_cost(0, "buy") == 0.0

    def test_invalid_side_raises(self):
        with pytest.raises(ValueError, match="side"):
            self.model.compute_cost(100_000, "long")

    def test_negative_size_raises(self):
        with pytest.raises(ValueError, match="trade_size"):
            self.model.compute_cost(-100_000, "buy")

    def test_case_insensitive_side(self):
        cost_lower = self.model.compute_cost(1.0, "buy")
        cost_upper = self.model.compute_cost(1.0, "BUY")
        assert cost_lower == cost_upper


# ---------------------------------------------------------------------------
# FXCostModel.compute – vectorized interface
# ---------------------------------------------------------------------------

class TestComputeVectorized:
    def setup_method(self):
        self.model = FXCostModel(spread_bps=0.6, slippage_bps=0.2)
        self.rate = 0.8 / 10_000  # 0.00008

    def test_zero_trades_zero_costs(self):
        trades = pd.Series([0.0, 0.0, 0.0])
        costs = self.model.compute(trades)
        assert (costs == 0.0).all()

    def test_single_long_entry(self):
        # Open a long: trade = +1
        trades = pd.Series([1.0, 0.0, 0.0])
        costs = self.model.compute(trades)
        assert costs.iloc[0] == pytest.approx(self.rate)
        assert costs.iloc[1] == 0.0
        assert costs.iloc[2] == 0.0

    def test_flip_costs_double(self):
        # Long-to-short flip: trade = -2
        trades = pd.Series([0.0, -2.0, 0.0])
        costs = self.model.compute(trades)
        assert costs.iloc[1] == pytest.approx(2 * self.rate)

    def test_round_trip_cost(self):
        # Enter long (+1) and exit (-1): total cost = 2 * rate
        trades = pd.Series([1.0, 0.0, 0.0, -1.0])
        total_cost = self.model.compute(trades).sum()
        assert total_cost == pytest.approx(2 * self.rate)

    def test_output_index_matches_input(self):
        idx = pd.date_range("2024-01-01", periods=5, freq="h")
        trades = pd.Series([1, 0, 0, -2, 1], index=idx)
        costs = self.model.compute(trades)
        assert costs.index.equals(idx)

    def test_costs_always_non_negative(self):
        trades = pd.Series([-1, 2, -3, 0, 1])
        costs = self.model.compute(trades)
        assert (costs >= 0).all()

    def test_known_values(self):
        # |trades| = [1, 0, 2, 0, 1]; each unit costs 0.00008
        trades = pd.Series([1.0, 0.0, -2.0, 0.0, 1.0])
        costs = self.model.compute(trades)
        expected = pd.Series([1 * self.rate, 0.0, 2 * self.rate, 0.0, 1 * self.rate])
        pd.testing.assert_series_equal(costs.reset_index(drop=True),
                                       expected, check_names=False)


# ---------------------------------------------------------------------------
# cost_model_from_config
# ---------------------------------------------------------------------------

class TestCostModelFromConfig:
    def test_builds_from_config_dict(self):
        cfg = {"execution": {"commission_bps": 1.0, "slippage_bps": 2.0}}
        model = cost_model_from_config(cfg)
        assert model.spread == 1.0
        assert model.slippage == 2.0

    def test_missing_execution_section_uses_defaults(self):
        model = cost_model_from_config({})
        assert model.spread == pytest.approx(1.0)
        assert model.slippage == pytest.approx(2.0)

    def test_partial_config_uses_defaults(self):
        cfg = {"execution": {"slippage_bps": 3.5}}
        model = cost_model_from_config(cfg)
        assert model.slippage == 3.5
        assert model.spread == pytest.approx(1.0)  # default commission_bps
