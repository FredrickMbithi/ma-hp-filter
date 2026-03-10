"""Standardized feature generation library.

This module provides a stateless feature factory for technical indicators
and microstructure features. All methods are static to avoid hidden state
and prevent lookahead bias.
"""

import numpy as np
import pandas as pd


class FeatureLibrary:
    """Standardized feature generation.
    
    Design philosophy:
    - All methods are @staticmethod (stateless)
    - Features are computed on-demand, not cached
    - No hidden dependencies or lookahead bias
    - Pass data explicitly (no internal storage)
    """
    
    @staticmethod
    def momentum(prices: pd.Series, period: int = 10) -> pd.Series:
        """Compute momentum as log return over a horizon.
        
        Momentum measures the total return over N periods.
        Using log returns for additivity property.
        
        Args:
            prices: Price series (typically close)
            period: Lookback horizon (default: 10)
            
        Returns:
            Log return from t-period to t.
            First (period) values will be NaN.
        """
        return np.log(prices).diff(period)
    
    @staticmethod
    def volatility(returns: pd.Series, window: int = 20) -> pd.Series:
        """Compute rolling realized volatility.
        
        Standard deviation of returns over rolling window.
        
        Args:
            returns: Return series (log or arithmetic)
            window: Lookback period (default: 20)
            
        Returns:
            Rolling volatility series.
            First (window-1) values will be NaN.
        """
        return returns.rolling(window=window, min_periods=window).std()
    
    @staticmethod
    def rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Compute Relative Strength Index.
        
        RSI measures momentum by comparing magnitude of recent gains vs losses.
        Bounded between 0-100:
        - RSI > 70: Overbought
        - RSI < 30: Oversold
        
        Args:
            prices: Price series (typically close)
            period: Lookback period for averaging (default: 14)
            
        Returns:
            RSI series bounded [0, 100].
            First (period) values will be NaN.
        """
        delta = prices.diff()
        
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        avg_gain = gain.rolling(period, min_periods=period).mean()
        avg_loss = loss.rolling(period, min_periods=period).mean()
        
        rs = avg_gain / avg_loss
        
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def range_feature(high: pd.Series, low: pd.Series) -> pd.Series:
        """Compute bar range (high - low).
        
        Simple measure of intrabar volatility.
        
        Args:
            high: High price series
            low: Low price series
            
        Returns:
            Range series (high - low).
        """
        return high - low
    
    @staticmethod
    def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Compute True Range (Wilder, 1978).
        
        Accounts for gaps by measuring the greatest of:
        - Current high - current low
        - |Current high - previous close|
        - |Current low - previous close|
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            
        Returns:
            True range series.
            First value will be NaN (no prior close).
        """
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        
        return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    @staticmethod
    def close_position(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """Compute close position within bar range.
        
        Measures where the close occurred within the bar's range:
        - 0.0: Close at low (bearish)
        - 0.5: Close at midpoint (neutral)
        - 1.0: Close at high (bullish)
        
        Args:
            high: High price series
            low: Low price series
            close: Close price series
            
        Returns:
            Close position ratio bounded [0, 1].
            NaN when high == low (zero range).
        """
        range_val = high - low
        position = (close - low) / range_val
        
        # Handle zero-range bars
        position = position.replace([np.inf, -np.inf], np.nan)
        
        return position
    
    @staticmethod
    def distance_from_ma(prices: pd.Series, window: int = 20) -> pd.Series:
        """Compute z-scored distance from moving average.
        
        Mean reversion feature measuring how far price has deviated
        from its moving average, normalized by volatility.
        
        Args:
            prices: Price series (typically close)
            window: MA lookback period (default: 20)
            
        Returns:
            Z-scored deviation from SMA.
            Positive: Above MA, Negative: Below MA
            First (window-1) values will be NaN.
        """
        ma = prices.rolling(window, min_periods=window).mean()
        deviation = prices - ma
        
        # Normalize by rolling std of deviation
        std = deviation.rolling(window, min_periods=window).std()
        z = deviation / std
        
        z = z.replace([np.inf, -np.inf], np.nan)
        
        return z
