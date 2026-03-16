import pandas as pd
from ..indicators.ta import bbands

def bbands_breakout(df: pd.DataFrame, length: int = 20, mult: float = 2.0):
    """
    Long when Close crosses above upper band; flat when it falls below middle band.
    Return signal in {0,1}.
    """
    lower, mid, upper = bbands(df["Close"], length, mult)
    close = df["Close"]
    # Entry: first bar where close > upper and previously wasn't
    long_cond = (close > upper) & (close.shift(1) <= upper.shift(1))
    # Maintain position while close >= mid; exit when close < mid
    sig = pd.Series(0, index=df.index, dtype=int)
    in_pos = False
    for i in range(len(df)):
        if not in_pos and bool(long_cond.iloc[i]):
            in_pos = True
        elif in_pos and close.iloc[i] < mid.iloc[i]:
            in_pos = False
        sig.iloc[i] = 1 if in_pos else 0
    return sig.rename("signal")
