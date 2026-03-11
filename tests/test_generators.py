import numpy as np
import pandas as pd

from src.features.generators import (
    causal_hp_trend,
    crossing_threshold_signal,
    threshold_time_stop_signal,
)


def _series(values: list[float]) -> pd.Series:
    index = pd.date_range("2024-01-01", periods=len(values), freq="h", tz="UTC")
    return pd.Series(values, index=index, dtype=float)


class TestCausalHPTrend:
    def test_forward_fills_interior_nan_without_poisoning_future_windows(self):
        prices = _series([
            100.0, 100.2, 100.4, 100.6, 100.8,
            101.0, 101.2, 101.4, 101.6, 101.8,
            102.0, 102.2, np.nan, 102.6, 102.8,
            103.0, 103.2, 103.4, 103.6, 103.8,
            104.0, 104.2, 104.4, 104.6, 104.8,
        ])

        trend = causal_hp_trend(prices, lamb=1e6, window=10)
        trend_ffill = causal_hp_trend(prices.ffill(), lamb=1e6, window=10)

        assert trend.iloc[9:].notna().all()
        assert np.allclose(trend.iloc[9:], trend_ffill.iloc[9:])

    def test_leading_nans_remain_uncomputed_until_window_is_clean(self):
        prices = _series([
            np.nan, np.nan, 100.0, 100.1, 100.2,
            100.3, 100.4, 100.5, 100.6, 100.7,
            100.8, 100.9, 101.0, 101.1, 101.2,
        ])

        trend = causal_hp_trend(prices, lamb=1e6, window=10)

        assert trend.iloc[:11].isna().all()
        assert trend.iloc[11:].notna().all()


class TestThresholdTimeStopSignal:
    def test_invalid_bars_preserve_open_position(self):
        feature = _series([0.0, 0.5, 0.0, -3.0, np.nan, 0.0, 0.0])

        signal = threshold_time_stop_signal(
            feature,
            threshold_sigma=1.0,
            hold_bars=10,
            confirmation_bars=0,
            lookback=3,
        )

        assert signal.tolist() == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]

    def test_time_stop_cannot_reenter_on_same_bar(self):
        feature = _series([0.0, 1.0, 0.0, -3.0, -2.5])

        signal = threshold_time_stop_signal(
            feature,
            threshold_sigma=1.0,
            hold_bars=1,
            confirmation_bars=0,
            lookback=3,
        )

        assert signal.tolist() == [0.0, 0.0, 0.0, 1.0, 0.0]


class TestCrossingThresholdSignal:
    def test_invalid_bars_preserve_open_position(self):
        feature = _series([0.0, 0.5, 0.0, -3.0, np.nan, 0.0, 0.0])

        signal = crossing_threshold_signal(
            feature,
            threshold_sigma=1.0,
            hold_bars=10,
            lookback=3,
        )

        assert signal.tolist() == [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]

    def test_time_stop_cannot_reenter_on_same_bar(self):
        feature = _series([0.0, 0.5, 0.0, -3.0, 0.0, -3.0])

        signal = crossing_threshold_signal(
            feature,
            threshold_sigma=1.0,
            hold_bars=2,
            lookback=3,
        )

        assert signal.tolist() == [0.0, 0.0, 0.0, 1.0, 1.0, 0.0]