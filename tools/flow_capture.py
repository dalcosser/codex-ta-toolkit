"""
Standalone flow capture script — runs without Streamlit.
Scans the Beta Universe for unusual options flow and appends to _flow_history.parquet.
Intended for scheduled execution (2x daily on weekdays).

Usage:  py -3 tools/flow_capture.py
"""
import os, sys, time, json
from pathlib import Path
from datetime import datetime, timedelta

# Load .env from project root
_root = Path(__file__).resolve().parent.parent
_env_path = _root / ".env"
if _env_path.exists():
    for line in _env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

import pandas as pd

POLY_KEY = (os.getenv("POLYGON_API_KEY") or "").strip()
PARQUET_DIR = os.getenv("PER_TICKER_PARQUET_DIR", "")
FLOW_PATH = os.path.join(PARQUET_DIR, "_flow_history.parquet") if PARQUET_DIR else ""

# ── Polygon helper ──────────────────────────────────────────────────────
def _poly_get(path: str, params: dict = None, timeout: int = 12):
    import requests
    if not POLY_KEY:
        return {}
    p = dict(params or {})
    p["apiKey"] = POLY_KEY
    try:
        r = requests.get(f"https://api.polygon.io{path}", params=p, timeout=timeout)
        return r.json() if r.ok else {}
    except Exception as e:
        print(f"  [WARN] Polygon error: {e}")
        return {}


def _get_tickers() -> list:
    """Load ticker list from Beta Universe parquet files."""
    if not PARQUET_DIR or not os.path.isdir(PARQUET_DIR):
        print(f"[ERROR] PER_TICKER_PARQUET_DIR not set or missing: {PARQUET_DIR}")
        return []
    tickers = []
    cutoff = time.time() - 14 * 86400  # 14-day freshness
    for f in os.listdir(PARQUET_DIR):
        if not f.endswith(".parquet") or f.startswith("_"):
            continue
        fpath = os.path.join(PARQUET_DIR, f)
        if os.stat(fpath).st_mtime < cutoff:
            continue
        tickers.append(f.replace(".parquet", "").upper())
    return sorted(tickers)


def _scan_ticker(ticker: str) -> list:
    """Fetch options snapshot for one ticker, return unusual flow records."""
    data = _poly_get(f"/v3/snapshot/options/{ticker}", {"limit": 250})
    results = data.get("results", [])
    if not results:
        return []

    records = []
    for c in results:
        det = c.get("details", {})
        day = c.get("day") or c.get("session") or {}
        lt = c.get("last_trade", {})
        ua = c.get("underlying_asset", {})

        vol = float(day.get("volume", 0) or 0)
        oi = float(c.get("open_interest", 0) or 0)
        mid = float(day.get("vwap", 0) or lt.get("price", 0) or 0)
        spot = float(ua.get("price", 0) or 0)

        if vol < 100 or oi < 1 or mid <= 0:
            continue

        voi = round(vol / oi, 2)
        if voi < 1.0:
            continue

        cp = det.get("contract_type", "?")[0].upper() if det.get("contract_type") else "?"
        strike = float(det.get("strike_price", 0) or 0)
        expiry = det.get("expiration_date", "")
        prem = round(vol * mid * 100)

        records.append({
            "Ticker": ticker,
            "C/P": cp,
            "Strike": strike,
            "Expiry": expiry,
            "Vol": int(vol),
            "OI": int(oi),
            "Vol/OI": voi,
            "Mid": round(mid, 2),
            "Spot": round(spot, 2),
            "Est $Prem": prem,
        })

    return records


def _save_flow(records: list):
    """Append records to _flow_history.parquet with ScanTime. Purge >30 days."""
    if not records or not FLOW_PATH:
        return
    now = pd.Timestamp.now(tz="US/Eastern")
    new_df = pd.DataFrame(records)
    new_df["ScanTime"] = now

    try:
        if os.path.exists(FLOW_PATH):
            old = pd.read_parquet(FLOW_PATH)
            combined = pd.concat([old, new_df], ignore_index=True)
        else:
            combined = new_df

        if "ScanTime" in combined.columns:
            combined["ScanTime"] = pd.to_datetime(combined["ScanTime"], utc=True, errors="coerce")
            cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=30)
            combined = combined[combined["ScanTime"] >= cutoff]

        combined.to_parquet(FLOW_PATH, index=False)
        print(f"  Saved {len(new_df)} records ({len(combined)} total in history)")
    except Exception as e:
        print(f"  [ERROR] Failed to save: {e}")


def main():
    print(f"=== Flow Capture — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")

    if not POLY_KEY:
        print("[ERROR] POLYGON_API_KEY not set in .env")
        sys.exit(1)

    tickers = _get_tickers()
    if not tickers:
        print("[ERROR] No tickers found in parquet directory")
        sys.exit(1)

    print(f"Scanning {len(tickers)} tickers...")
    all_flow = []
    scanned = 0

    for i, tkr in enumerate(tickers):
        try:
            recs = _scan_ticker(tkr)
            if recs:
                all_flow.extend(recs)
            scanned += 1
        except Exception as e:
            print(f"  [WARN] {tkr}: {e}")

        # Rate limit: Polygon free tier = 5 req/min
        if (i + 1) % 5 == 0:
            time.sleep(12.5)

        # Progress every 25 tickers
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(tickers)} scanned, {len(all_flow)} flow records so far")

    print(f"\nDone: {scanned}/{len(tickers)} tickers, {len(all_flow)} unusual flow records")

    if all_flow:
        _save_flow(all_flow)
    else:
        print("No unusual flow detected this scan.")

    print("=== Complete ===")


if __name__ == "__main__":
    main()
