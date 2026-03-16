import pandas as pd
import yfinance as yf

_EXPECTED = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        try:
            levels = [df.columns.get_level_values(i) for i in range(df.columns.nlevels)]
            field_level = None
            for i, lvl in enumerate(levels):
                lower = {str(x).lower() for x in lvl}
                if {"open","high","low","close","adj close","volume"} & lower:
                    field_level = i
                    break
            if field_level is None:
                df.columns = df.columns.droplevel(0)
            else:
                df.columns = df.columns.get_level_values(field_level)
        except Exception:
            df.columns = [c[-1] if isinstance(c, tuple) else c for c in df.columns]
    df.columns = [str(c).strip().title() for c in df.columns]
    return df

def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    df = _flatten_columns(df)
    keep = [c for c in _EXPECTED if c in df.columns]
    if not keep:
        raise ValueError(f"No expected OHLCV columns found. Got: {list(df.columns)}")
    df = df[keep].copy()
    if "Close" not in df.columns and "Adj Close" in df.columns:
        df["Close"] = df["Adj Close"]
    for c in _EXPECTED:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()]
    if "Close" not in df.columns or df["Close"].isna().all():
        raise ValueError("No usable numeric 'Close' data returned after normalization.")
    return df.dropna(subset=["Close"])

def get_ohlcv(
    ticker: str,
    start: str | None,
    end: str | None,
    interval: str = "1d",
    period: str | None = None,
) -> pd.DataFrame:
    """
    Flexible downloader:
    - Daily/hourly with start/end, or
    - Intraday with period (recommended). yfinance 1m data is limited (~7 days).
    """
    intraday = interval in {"1h", "30m", "15m", "5m", "1m"}

    if intraday:
        # yfinance works best with "period" for intraday.
        # 1m typically capped at ~7d. Provide a sensible default.
        if period is None:
            period = "7d" if interval == "1m" else "30d"
        df = yf.download(
            tickers=str(ticker),
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            group_by=None,
        )
    else:
        if not start or not end:
            raise ValueError("Start and end dates are required for non-intraday intervals.")
        df = yf.download(
            tickers=str(ticker),
            start=start,
            end=end,
            interval=interval,
            auto_adjust=False,
            progress=False,
            group_by=None,
        )

    if df is None or df.empty:
        raise ValueError(f"No data returned for {ticker} (interval={interval}, period={period}, start={start}, end={end}).")
    return _normalize_ohlcv(df)
# ---------------------------------------------------------------------
# 🧮 Options Chain Downloader (for the Options tab)
# ---------------------------------------------------------------------
import datetime as dt
import yfinance as yf

def get_options_chain(ticker: str, nearest: bool = True):
    """
    Fetches option chain (calls, puts, expirations) for a given ticker.
    If nearest=True, returns the closest upcoming expiration date.
    """
    tk = yf.Ticker(ticker)
    expirations = tk.options
    if not expirations:
        raise ValueError(f"No options data available for {ticker}")

    # Choose the nearest expiration (today or next)
    if nearest:
        today = dt.date.today()
        exp = min(
            expirations,
            key=lambda e: abs((dt.datetime.strptime(e, "%Y-%m-%d").date() - today).days),
        )
        calls = tk.option_chain(exp).calls.copy()
        puts = tk.option_chain(exp).puts.copy()
        calls["expiration"] = exp
        puts["expiration"] = exp
        return calls, puts, expirations
    else:
        all_calls, all_puts = [], []
        for exp in expirations[:3]:  # limit to first few expirations
            chain = tk.option_chain(exp)
            c, p = chain.calls.copy(), chain.puts.copy()
            c["expiration"], p["expiration"] = exp, exp
            all_calls.append(c)
            all_puts.append(p)
        calls = pd.concat(all_calls)
        puts = pd.concat(all_puts)
        return calls, puts, expirations


