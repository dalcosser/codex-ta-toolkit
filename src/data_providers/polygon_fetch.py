from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import requests


INTERVAL_MAP = {
    "1m": (1, "minute"),
    "2m": (2, "minute"),
    "5m": (5, "minute"),
    "15m": (15, "minute"),
    "30m": (30, "minute"),
    "60m": (60, "minute"),
    "90m": (90, "minute"),
    "1h": (60, "minute"),
    "1d": (1, "day"),
}

PERIOD_DAYS = {
    "5d": 5,
    "7d": 7,
    "14d": 14,
    "30d": 30,
    "60d": 60,
    "90d": 90,
    "1y": 365,
    "2y": 730,
    "5y": 1825,
}


def _to_date_str(val: str) -> str:
    """Convert any date/datetime/ISO string to plain YYYY-MM-DD for Polygon URL paths.

    Polygon's /v2/aggs range endpoint only accepts YYYY-MM-DD or Unix-ms in the
    path — it returns HTTP 400 for ISO strings with timezone offsets like
    '2026-03-04T04:00:00-05:00'.
    """
    try:
        return pd.to_datetime(val).date().isoformat()
    except Exception:
        # Strip everything after the date part as a last resort
        return str(val)[:10]


def _date_range(period: Optional[str], start: Optional[str], end: Optional[str]) -> tuple[str, str]:
    if period:
        days = PERIOD_DAYS.get(str(period).lower(), 365)
        e = (datetime.utcnow().date() + timedelta(days=1)).isoformat()
        s = (datetime.utcnow().date() - timedelta(days=days)).isoformat()
        return s, e
    if start and end:
        # Always extract plain YYYY-MM-DD — Polygon path doesn't accept TZ offsets
        return _to_date_str(start), _to_date_str(end)
    e = (datetime.utcnow().date() + timedelta(days=1)).isoformat()
    s = (datetime.utcnow().date() - timedelta(days=180)).isoformat()
    return s, e


def fetch_polygon_ohlc(
    ticker: str,
    *,
    interval: str,
    period: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    adjusted: bool = True,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch OHLCV from Polygon.io Aggregates API and return Yahoo-like columns.

    Returns empty DataFrame if api_key missing or no results.
    """
    key = api_key or os.getenv("POLYGON_API_KEY")
    if not key:
        return pd.DataFrame()

    mult, span = INTERVAL_MAP.get(interval, (1, "day"))
    s, e = _date_range(period, start, end)

    poly_ticker = _normalize_polygon_ticker(ticker)
    url = f"https://api.polygon.io/v2/aggs/ticker/{poly_ticker}/range/{mult}/{span}/{s}/{e}"
    params = {"adjusted": str(bool(adjusted)).lower(), "sort": "asc", "limit": 50000, "apiKey": key}
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json() or {}
        results = data.get("results") or []
        if not results:
            return pd.DataFrame()
        df = pd.DataFrame(results)
        if df.empty:
            return pd.DataFrame()
        # Polygon columns: t(ms), o,h,l,c,v, vw, n
        if "t" in df.columns:
            df["date"] = pd.to_datetime(df["t"], unit="ms", utc=True)
            df = df.set_index("date").sort_index()
        out = df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
        keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in out.columns]
        return out[keep]
    except Exception:
        return pd.DataFrame()
INDEX_MAP = {
    "^GSPC": "I:SPX",
    "SPX": "I:SPX",
    "^NDX": "I:NDX",
    "NDX": "I:NDX",
    "^DJI": "I:DJI",
    "DJI": "I:DJI",
    "^RUT": "I:RUT",
    "RUT": "I:RUT",
    "^VIX": "I:VIX",
    "VIX": "I:VIX",
}

def _normalize_polygon_ticker(user_ticker: str) -> str:
    t = (user_ticker or "").strip().upper()
    if t in INDEX_MAP:
        return INDEX_MAP[t]
    if t.startswith("^"):
        # Generic caret index -> try I:<symbol without ^>
        return f"I:{t[1:]}"
    return t
