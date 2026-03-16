from __future__ import annotations

import os
from typing import Optional, Tuple, List

import pandas as pd
import requests


def _get_key(explicit: Optional[str] = None) -> Optional[str]:
    if explicit:
        return explicit
    return os.getenv("POLYGON_API_KEY")


def fetch_polygon_expirations(
    ticker: str,
    *,
    api_key: Optional[str] = None,
    max_pages: int = 5,
) -> List[str]:
    key = _get_key(api_key)
    if not key:
        return []
    base = "https://api.polygon.io/v3/reference/options/contracts"
    params = {
        "underlying_ticker": ticker.upper(),
        "expired": "false",
        "limit": 1000,
        "apiKey": key,
    }
    out = set()
    url = base
    pages = 0
    try:
        while url and pages < max_pages:
            r = requests.get(url, params=params if url == base else {"apiKey": key}, timeout=20)
            r.raise_for_status()
            data = r.json() or {}
            results = data.get("results") or []
            for row in results:
                exp = row.get("expiration_date")
                if exp:
                    out.add(str(exp))
            next_url = data.get("next_url")
            url = next_url
            params = None
            pages += 1
    except Exception:
        pass
    return sorted(out)


def fetch_polygon_chain(
    ticker: str,
    expiration: str,
    *,
    api_key: Optional[str] = None,
    limit: int = 5000,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    key = _get_key(api_key)
    if not key:
        return pd.DataFrame(), pd.DataFrame()
    url = f"https://api.polygon.io/v3/snapshot/options/{ticker.upper()}"
    params = {
        "expiration_date": expiration,
        "includeGreeks": "true",
        "limit": limit,
        "apiKey": key,
    }
    try:
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json() or {}
        results = data.get("results") or []
        rows = []
        for itm in results:
            det = itm.get("details") or {}
            day = itm.get("day") or {}
            lq = itm.get("last_quote") or {}
            lt = itm.get("last_trade") or {}
            greeks = itm.get("greeks") or {}
            rows.append({
                "contract_symbol": det.get("ticker") or det.get("symbol"),
                "option_type": det.get("contract_type"),
                "strike": det.get("strike_price"),
                "expiration": det.get("expiration_date"),
                "bid": lq.get("bid"),
                "ask": lq.get("ask"),
                "last_price": lt.get("price") or day.get("close") or day.get("price") or None,
                "volume": day.get("volume"),
                "open_interest": day.get("open_interest"),
                "implied_volatility": greeks.get("iv"),
            })
        if not rows:
            return pd.DataFrame(), pd.DataFrame()
        df = pd.DataFrame(rows)
        # Normalize columns to match app expectations
        df.columns = [str(c).replace(" ", "_").lower() for c in df.columns]
        calls = df[df.get("option_type", "").astype(str).str.lower().str.startswith("c")].copy()
        puts = df[df.get("option_type", "").astype(str).str.lower().str.startswith("p")].copy()
        return calls, puts
    except Exception:
        return pd.DataFrame(), pd.DataFrame()

