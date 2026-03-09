"""
src.backtest
============
Vectorized backtest engine and transaction cost models for FX strategies.

Public API
----------
    VectorizedBacktest   – main backtest engine
    ZeroCost             – no-op cost model (gross returns)
    FXCostModel          – fixed spread + slippage cost model
    cost_model_from_config – build FXCostModel from config.yaml dict
    compute_swap_cost    – overnight rollover cost for a held position
    compute_swap_series  – per-bar swap costs for a position series
    load_swap_rates      – load swap rates from CSV or defaults

Usage
-----
>>> from src.backtest import VectorizedBacktest, FXCostModel
>>> cost = FXCostModel(spread_bps=0.6, slippage_bps=0.2)
>>> bt = VectorizedBacktest(data=ohlc_df, signal=signal_series, cost_model=cost)
>>> results = bt.run()
>>> equity = results["equity"]
"""

from src.backtest.cost_model import FXCostModel, cost_model_from_config
from src.backtest.engine import VectorizedBacktest, ZeroCost
from src.backtest.swap_calculator import (
    compute_swap_cost,
    compute_swap_series,
    load_swap_rates,
)

__all__ = [
    "VectorizedBacktest",
    "ZeroCost",
    "FXCostModel",
    "cost_model_from_config",
    "compute_swap_cost",
    "compute_swap_series",
    "load_swap_rates",
]
