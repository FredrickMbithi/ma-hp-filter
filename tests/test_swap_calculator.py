"""Tests for FX swap/rollover cost calculator.

Tests verify:
1. Basic swap cost calculations
2. Wednesday triple roll logic
3. Long vs short rate handling
4. Historical CSV and default fallback
5. Validation and edge cases
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtest.swap_calculator import (
    compute_swap_cost,
    compute_swap_series,
    load_swap_rates,
    _count_triple_roll_days,
    SWAP_DEFAULTS,
)


class TestSwapDefaults:
    """Test default swap rate constants."""
    
    def test_swap_defaults_exist(self):
        """Verify SWAP_DEFAULTS has expected pairs."""
        expected_pairs = ['USDJPY', 'EURUSD', 'GBPUSD', 'AUDUSD']
        
        for pair in expected_pairs:
            assert pair in SWAP_DEFAULTS
            assert 'long' in SWAP_DEFAULTS[pair]
            assert 'short' in SWAP_DEFAULTS[pair]
    
    def test_usdjpy_carry_trade_rates(self):
        """USDJPY: long earns (positive), short pays (negative)."""
        rates = SWAP_DEFAULTS['USDJPY']
        
        # Long should be positive (earns)
        assert rates['long'] > 0
        # Short should be negative (pays)
        assert rates['short'] < 0
        # Long earns ~0.5 bps/day
        assert rates['long'] == pytest.approx(0.50, abs=0.1)
        # Short pays ~1.2 bps/day
        assert rates['short'] == pytest.approx(-1.20, abs=0.3)


class TestLoadSwapRates:
    """Test load_swap_rates() function."""
    
    def test_load_from_defaults(self):
        """Test loading rates from SWAP_DEFAULTS."""
        rates = load_swap_rates('USDJPY')
        
        assert rates['source'] == 'default'
        assert isinstance(rates['long'], float)
        assert isinstance(rates['short'], float)
        assert rates['long'] == SWAP_DEFAULTS['USDJPY']['long']
        assert rates['short'] == SWAP_DEFAULTS['USDJPY']['short']
    
    def test_unknown_symbol_raises(self):
        """Test that unknown symbol raises KeyError."""
        with pytest.raises(KeyError, match="No swap rates found"):
            load_swap_rates('GARBAGE')
    
    def test_case_insensitive(self):
        """Test that symbol lookup is case-insensitive."""
        rates_upper = load_swap_rates('USDJPY')
        rates_lower = load_swap_rates('usdjpy')
        
        assert rates_upper['long'] == rates_lower['long']
        assert rates_upper['short'] == rates_lower['short']


class TestComputeSwapCost:
    """Test compute_swap_cost() function."""
    
    def test_usdjpy_long_earns(self):
        """USDJPY long position earns positive swap."""
        # 1 lot, 1 day, long side
        pnl = compute_swap_cost('USDJPY', position_size=1.0, hold_days=1, side='long')
        
        # Should be positive (earns)
        assert pnl > 0
        # Should match default rate
        assert pnl == pytest.approx(SWAP_DEFAULTS['USDJPY']['long'])
    
    def test_usdjpy_short_pays(self):
        """USDJPY short position pays negative swap."""
        # 1 lot, 1 day, short side
        pnl = compute_swap_cost('USDJPY', position_size=1.0, hold_days=1, side='short')
        
        # Should be negative (pays)
        assert pnl < 0
        # Should match default rate
        assert pnl == pytest.approx(SWAP_DEFAULTS['USDJPY']['short'])
    
    def test_scaling_by_position_size(self):
        """Swap scales linearly with position size."""
        swap_1_lot = compute_swap_cost('USDJPY', position_size=1.0, hold_days=1, side='long')
        swap_2_lot = compute_swap_cost('USDJPY', position_size=2.0, hold_days=1, side='long')
        
        assert swap_2_lot == pytest.approx(2 * swap_1_lot)
    
    def test_scaling_by_hold_days(self):
        """Swap scales with hold days (before triple roll adjustment)."""
        # Note: this is approximate due to triple roll logic
        swap_1_day = compute_swap_cost('USDJPY', position_size=1.0, hold_days=1, side='long')
        swap_5_day = compute_swap_cost('USDJPY', position_size=1.0, hold_days=5, side='long')
        
        # Should be roughly 5× (may vary due to triple roll approximation)
        assert swap_5_day > 4 * swap_1_day
        assert swap_5_day < 7 * swap_1_day  # Upper bound accounting for triple roll
    
    def test_zero_position_size(self):
        """Zero position size produces zero cost."""
        pnl = compute_swap_cost('USDJPY', position_size=0.0, hold_days=5, side='long')
        assert pnl == 0.0
    
    def test_zero_hold_days(self):
        """Zero hold days produces zero cost."""
        pnl = compute_swap_cost('USDJPY', position_size=1.0, hold_days=0, side='long')
        assert pnl == 0.0
    
    def test_fractional_lot_size(self):
        """Fractional lot sizes work correctly."""
        # 0.1 lot (mini lot)
        pnl = compute_swap_cost('USDJPY', position_size=0.1, hold_days=1, side='long')
        
        expected = SWAP_DEFAULTS['USDJPY']['long'] * 0.1
        assert pnl == pytest.approx(expected)


class TestSwapValidation:
    """Test parameter validation in compute_swap_cost()."""
    
    def test_invalid_side_rejected(self):
        """Test that invalid side parameter is rejected."""
        with pytest.raises(ValueError, match="side must be 'long' or 'short'"):
            compute_swap_cost('USDJPY', position_size=1.0, hold_days=1, side='buy')
    
    def test_negative_position_size_rejected(self):
        """Test that negative position size is rejected."""
        with pytest.raises(ValueError, match="position_size must be non-negative"):
            compute_swap_cost('USDJPY', position_size=-1.0, hold_days=1, side='long')
    
    def test_negative_hold_days_rejected(self):
        """Test that negative hold days are rejected."""
        with pytest.raises(ValueError, match="hold_days must be non-negative"):
            compute_swap_cost('USDJPY', position_size=1.0, hold_days=-5, side='long')


class TestWednesdayTripleRoll:
    """Test Wednesday triple roll logic."""
    
    def test_count_triple_roll_single_wednesday(self):
        """Test counting one Wednesday in hold period."""
        # Start on Tuesday, hold for 2 days → hits Wednesday
        tuesday = pd.Timestamp('2024-01-02')  # Tuesday
        extra = _count_triple_roll_days(hold_days=2, start_date=tuesday)
        
        # Should count 1 Wednesday = +2 extra days
        assert extra == 2
    
    def test_count_triple_roll_no_wednesday(self):
        """Test period with no Wednesday."""
        # Start on Thursday, hold for 2 days → Thu, Fri (no Wed)
        thursday = pd.Timestamp('2024-01-04')  # Thursday
        extra = _count_triple_roll_days(hold_days=2, start_date=thursday)
        
        # No Wednesday = 0 extra days
        assert extra == 0
    
    def test_count_triple_roll_full_week(self):
        """Test full 7-day period hits exactly 1 Wednesday."""
        monday = pd.Timestamp('2024-01-01')  # Monday
        extra = _count_triple_roll_days(hold_days=7, start_date=monday)
        
        # 7 days starting Monday hits exactly 1 Wednesday = +2 extra days
        assert extra == 2
    
    def test_count_triple_roll_two_weeks(self):
        """Test 14-day period hits 2 Wednesdays."""
        monday = pd.Timestamp('2024-01-01')
        extra = _count_triple_roll_days(hold_days=14, start_date=monday)
        
        # 14 days hits 2 Wednesdays = +4 extra days
        assert extra == 4
    
    def test_count_triple_roll_approximation_no_date(self):
        """Test approximation when no start date provided."""
        # Should use ~2/7 approximation
        extra = _count_triple_roll_days(hold_days=7, start_date=None)
        
        # Expect ~2 extra days for 7 days
        assert extra == pytest.approx(2, abs=1)
    
    def test_swap_cost_with_wednesday(self):
        """Test that swap cost increases with Wednesday in period."""
        # Start on Tuesday, hold 1 day (no Wednesday)
        tuesday = pd.Timestamp('2024-01-02')
        swap_no_wed = compute_swap_cost(
            'USDJPY', position_size=1.0, hold_days=1, side='long',
            reference_date=tuesday
        )
        
        # Start on Tuesday, hold 2 days (includes Wednesday)
        swap_with_wed = compute_swap_cost(
            'USDJPY', position_size=1.0, hold_days=2, side='long',
            reference_date=tuesday
        )
        
        # Wednesday should add ~2 extra days of swap
        # So 2 days with Wednesday ≈ 4 effective days
        base_rate = SWAP_DEFAULTS['USDJPY']['long']
        assert swap_with_wed == pytest.approx(4 * base_rate)


class TestComputeSwapSeries:
    """Test compute_swap_series() for position time series."""
    
    def test_empty_positions(self):
        """Test with all-zero positions."""
        dates = pd.date_range('2024-01-01', periods=5, freq='D')
        positions = pd.Series(0.0, index=dates)
        
        swap_series = compute_swap_series('USDJPY', positions, position_size=1.0)
        
        # All zeros
        assert (swap_series == 0.0).all()
    
    def test_constant_long_position(self):
        """Test with constant long position."""
        dates = pd.date_range('2024-01-01', periods=5, freq='D')
        positions = pd.Series(1.0, index=dates)
        
        swap_series = compute_swap_series('USDJPY', positions, position_size=1.0)
        
        # All should be positive (long earns on USDJPY)
        assert (swap_series > 0).all()
        # Each bar should earn the long rate
        assert swap_series.iloc[0] == pytest.approx(SWAP_DEFAULTS['USDJPY']['long'])
    
    def test_constant_short_position(self):
        """Test with constant short position."""
        dates = pd.date_range('2024-01-01', periods=5, freq='D')
        positions = pd.Series(-1.0, index=dates)
        
        swap_series = compute_swap_series('USDJPY', positions, position_size=1.0)
        
        # All should be negative (short pays on USDJPY)
        assert (swap_series < 0).all()
        # Each bar should pay the short rate
        assert swap_series.iloc[0] == pytest.approx(SWAP_DEFAULTS['USDJPY']['short'])
    
    def test_mixed_positions(self):
        """Test with mixed long/short/flat positions."""
        dates = pd.date_range('2024-01-01', periods=5, freq='D')
        positions = pd.Series([1.0, 1.0, 0.0, -1.0, -1.0], index=dates)
        
        swap_series = compute_swap_series('USDJPY', positions, position_size=1.0)
        
        # First two: positive (long)
        assert swap_series.iloc[0] > 0
        assert swap_series.iloc[1] > 0
        # Third: zero (flat)
        assert swap_series.iloc[2] == 0.0
        # Last two: negative (short)
        assert swap_series.iloc[3] < 0
        assert swap_series.iloc[4] < 0
    
    def test_position_scaling(self):
        """Test position_size scaling parameter."""
        dates = pd.date_range('2024-01-01', periods=3, freq='D')
        positions = pd.Series(1.0, index=dates)
        
        swap_1x = compute_swap_series('USDJPY', positions, position_size=1.0)
        swap_2x = compute_swap_series('USDJPY', positions, position_size=2.0)
        
        # Should scale linearly
        pd.testing.assert_series_equal(swap_2x, 2 * swap_1x, check_names=False)


class TestSwapImpactAnalysis:
    """Test documenting swap impact on strategies."""
    
    def test_carry_kills_unaware_strategies(self):
        """Document: 5-day USDJPY short costs 6 bps (kills 4 bps edge)."""
        # Mean-reversion strategy holding short for 5 days
        swap_cost = compute_swap_cost(
            'USDJPY', position_size=1.0, hold_days=5, side='short'
        )
        
        # Cost is negative (paying), magnitude is ~6 bps
        assert swap_cost < 0
        assert abs(swap_cost) > 5.0  # More than 5 bps
        
        # If strategy has only 4 bps gross edge → net negative
        gross_edge_bps = 4.0
        net_edge = gross_edge_bps + swap_cost  # swap_cost is negative
        assert net_edge < 0, "Swap cost kills the strategy"
    
    def test_eurusd_long_pays(self):
        """EURUSD long typically pays (EUR rate < USD rate)."""
        pnl = compute_swap_cost('EURUSD', position_size=1.0, hold_days=1, side='long')
        
        # Should be negative (pays)
        assert pnl < 0
        assert pnl == pytest.approx(SWAP_DEFAULTS['EURUSD']['long'])


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_very_long_hold_period(self):
        """Test with unrealistically long hold (30 days)."""
        # Should not crash, should scale appropriately
        pnl = compute_swap_cost('USDJPY', position_size=1.0, hold_days=30, side='long')
        
        # Should be positive and roughly 30× daily rate (with triple roll adjustments)
        base_rate = SWAP_DEFAULTS['USDJPY']['long']
        assert pnl > 20 * base_rate  # Lower bound
        assert pnl < 40 * base_rate  # Upper bound accounting for triple rolls
    
    def test_mini_lot(self):
        """Test with 0.01 lot (micro lot)."""
        pnl = compute_swap_cost('USDJPY', position_size=0.01, hold_days=1, side='long')
        
        expected = SWAP_DEFAULTS['USDJPY']['long'] * 0.01
        assert pnl == pytest.approx(expected)
    
    def test_case_insensitive_side(self):
        """Test that side parameter is case-insensitive."""
        pnl_lower = compute_swap_cost('USDJPY', position_size=1.0, hold_days=1, side='long')
        pnl_upper = compute_swap_cost('USDJPY', position_size=1.0, hold_days=1, side='LONG')
        
        assert pnl_lower == pnl_upper


class TestRealisticBenchmarks:
    """Test against realistic swap cost benchmarks."""
    
    def test_usdjpy_5day_short_costly(self):
        """USDJPY short for 5 days should cost ~6 bps."""
        cost = compute_swap_cost('USDJPY', position_size=1.0, hold_days=5, side='short')
        
        # Should be negative, magnitude ~5-7 bps
        assert cost < 0
        assert abs(cost) > 4.0
        assert abs(cost) < 8.0
    
    def test_eurusd_long_pays_small(self):
        """EURUSD long pays but typically small magnitude."""
        cost = compute_swap_cost('EURUSD', position_size=1.0, hold_days=1, side='long')
        
        # Should be negative but small
        assert cost < 0
        assert abs(cost) < 0.5  # Less than 0.5 bps/day
