import pandas as pd
import numpy as np

def rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = (delta.where(delta > 0, 0.0)).ewm(alpha=1/length, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0.0)).ewm(alpha=1/length, adjust=False).mean()
    rs = gain / (loss.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0).rename(f"RSI({length})")

def sma(series: pd.Series, length: int) -> pd.Series:
    return series.rolling(length, min_periods=1).mean().rename(f"SMA({length})")

def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean().rename(f"EMA({length})")

def bbands(close: pd.Series, length: int = 20, mult: float = 2.0):
    mid = sma(close, length)
    std = close.rolling(length, min_periods=1).std(ddof=0)
    upper = (mid + mult * std).rename(f"BBU({length},{mult})")
    lower = (mid - mult * std).rename(f"BBL({length},{mult})")
    return lower, mid.rename(f"BBM({length})"), upper

def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal_len: int = 9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal = macd_line.ewm(span=signal_len, adjust=False).mean().rename(f"MACDsig({signal_len})")
    hist = (macd_line - signal).rename("MACDhist")
    return macd_line.rename(f"MACD({fast},{slow})"), signal, hist

def vwap(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    typical = (high + low + close) / 3.0
    cum_vol = volume.cumsum().replace(0, np.nan)
    cum_tpv = (typical * volume).cumsum()
    return (cum_tpv / cum_vol).rename("VWAP")
# --- Stochastic Oscillator (%K, %D) ---
def stoch_kd(high: pd.Series, low: pd.Series, close: pd.Series, k_len: int = 14, d_len: int = 3, smooth_k: int = 3):
    """
    %K = SMA(smooth_k) of raw %K over k_len lookback.
    %D = SMA(d_len) of %K.
    Returns (%K, %D), both 0..100.
    """
    lowest_low = low.rolling(k_len, min_periods=1).min()
    highest_high = high.rolling(k_len, min_periods=1).max()
    raw_k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, pd.NA)
    k = raw_k.rolling(smooth_k, min_periods=1).mean().rename(f"Stoch%K({k_len},{smooth_k})")
    d = k.rolling(d_len, min_periods=1).mean().rename(f"Stoch%D({d_len})")
    return k.clip(0, 100), d.clip(0, 100)
