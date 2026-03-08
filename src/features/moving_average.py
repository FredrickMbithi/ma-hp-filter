"""
Moving average features and envelopes.
"""
import pandas as pd


def sma(prices: pd.Series, period: int) -> pd.Series:
    """Simple moving average."""
    return prices.rolling(period).mean()


def ema(prices: pd.Series, period: int) -> pd.Series:
    """Exponential moving average."""
    return prices.ewm(span=period, adjust=False).mean()


def ma_envelope(prices: pd.Series, period: int, pct: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MA envelope bands.
    
    Returns:
        ma: Center MA
        upper: MA + pct%
        lower: MA - pct%
    """
    ma = sma(prices, period)
    upper = ma * (1 + pct / 100)
    lower = ma * (1 - pct / 100)
    return ma, upper, lower
