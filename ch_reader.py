"""
ClickHouse data reader — drop-in replacement for local parquet reads.

Usage in app18.py:
    from ch_reader import ch_load_daily_df

    # Replace _load_daily_df calls:
    df = ch_load_daily_df("AAPL")

Set CLICKHOUSE_HOST + CLICKHOUSE_PASSWORD in .env to enable.
If not set, returns None so local parquet fallback still works.
"""

import os
from functools import lru_cache

import pandas as pd

try:
    import clickhouse_connect
except ImportError:
    clickhouse_connect = None

# ── Lazy singleton client ─────────────────────────────────

_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client

    host = os.environ.get("CLICKHOUSE_HOST", "").strip()
    if not host or clickhouse_connect is None:
        return None

    _client = clickhouse_connect.get_client(
        host=host,
        port=int(os.environ.get("CLICKHOUSE_PORT", "8443")),
        username=os.environ.get("CLICKHOUSE_USER", "default"),
        password=os.environ.get("CLICKHOUSE_PASSWORD", ""),
        database=os.environ.get("CLICKHOUSE_DATABASE", "default"),
        secure=True,
    )
    return _client


# ── Public API ────────────────────────────────────────────

def ch_load_daily_df(ticker: str) -> pd.DataFrame | None:
    """Load daily OHLCV+ for one ticker from ClickHouse.
    Returns DataFrame with Date/Open/High/Low/Close/Volume + technicals,
    or None if ClickHouse is not configured / ticker not found.
    """
    client = _get_client()
    if client is None:
        return None

    t = (ticker or "").strip().upper()
    if not t:
        return None

    try:
        df = client.query_df(
            "SELECT * FROM daily_ohlcv WHERE Ticker = {t:String} ORDER BY Timestamp",
            parameters={"t": t},
        )
    except Exception:
        return None

    if df is None or df.empty:
        return None

    # Match the column contract that app18 expects
    df["Date"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    df = df.dropna(subset=["Date", "Close"]).sort_values("Date").reset_index(drop=True)
    return df


def ch_list_tickers() -> list[str]:
    """Return all unique tickers in ClickHouse."""
    client = _get_client()
    if client is None:
        return []
    try:
        result = client.query("SELECT DISTINCT Ticker FROM daily_ohlcv ORDER BY Ticker")
        return [row[0] for row in result.result_rows]
    except Exception:
        return []


def ch_available() -> bool:
    """True if ClickHouse is configured and reachable."""
    return _get_client() is not None
