from ..indicators.ta import rsi
def rsi_reversal(df):
    r=rsi(df['Close']); sig=(r<30).astype(int); sig[r>70]=0; return sig