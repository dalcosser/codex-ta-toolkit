"""
Daily refresh: Polygon → ClickHouse.

Fetches the latest trading day's bars from Polygon grouped daily endpoint,
pulls 260 days of history per ticker from ClickHouse, recomputes all 82
indicator columns, and upserts today's row.

Usage:
    py -3 ch_daily_refresh.py              # auto-detects last trading day
    py -3 ch_daily_refresh.py 2026-03-11   # specific date

Env vars (.env):
    POLYGON_API_KEY     (required)
    CLICKHOUSE_HOST     (required)
    CLICKHOUSE_PASSWORD (required)
    CLICKHOUSE_PORT     (default 8443)
    CLICKHOUSE_USER     (default "default")
    CLICKHOUSE_DATABASE (default "default")
"""

import os, sys, time, datetime as _dt
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import requests

load_dotenv()

try:
    import clickhouse_connect
except ImportError:
    sys.exit("pip install clickhouse-connect")

# ── Config ────────────────────────────────────────────────
POLY_KEY = os.environ.get("POLYGON_API_KEY", "")
CH_HOST  = os.environ.get("CLICKHOUSE_HOST", "")
CH_PASS  = os.environ.get("CLICKHOUSE_PASSWORD", "")
CH_PORT  = int(os.environ.get("CLICKHOUSE_PORT", "8443"))
CH_USER  = os.environ.get("CLICKHOUSE_USER", "default")
CH_DB    = os.environ.get("CLICKHOUSE_DATABASE", "default")

if not POLY_KEY:
    sys.exit("POLYGON_API_KEY not set")
if not CH_HOST:
    sys.exit("CLICKHOUSE_HOST not set")

client = clickhouse_connect.get_client(
    host=CH_HOST, port=CH_PORT,
    username=CH_USER, password=CH_PASS,
    database=CH_DB, secure=True,
)
print(f"Connected to ClickHouse: {CH_HOST}")

# ── Determine target date ─────────────────────────────────
if len(sys.argv) > 1:
    target_date = sys.argv[1]
else:
    # Use previous business day if before 5pm ET, else today
    from datetime import timezone, timedelta
    now_et = _dt.datetime.now(_dt.timezone(_dt.timedelta(hours=-4)))
    d = now_et.date()
    # If weekend, roll back to Friday
    while d.weekday() >= 5:
        d -= _dt.timedelta(days=1)
    target_date = d.isoformat()

print(f"Target date: {target_date}")

# ══════════════════════════════════════════════════════════
# STEP 1: Fetch grouped daily bars from Polygon
# ══════════════════════════════════════════════════════════
print("Fetching grouped daily bars from Polygon...")
url = f"https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/{target_date}"
resp = requests.get(url, params={"adjusted": "true", "apiKey": POLY_KEY}, timeout=30)
resp.raise_for_status()
data = resp.json()

if data.get("resultsCount", 0) == 0:
    print(f"No results for {target_date} — market may have been closed.")
    sys.exit(0)

results = data["results"]
print(f"  Got {len(results)} tickers from Polygon.")

# Build a DataFrame of today's bars
today_bars = pd.DataFrame(results)
today_bars = today_bars.rename(columns={
    "T": "Ticker", "o": "Open", "h": "High", "l": "Low",
    "c": "Close", "v": "Volume", "n": "Transactions",
    "t": "EpochMs",
})
# Filter to common stocks (no warrants, units, etc.)
today_bars = today_bars[today_bars["Ticker"].str.match(r"^[A-Z]{1,5}$", na=False)].copy()
today_bars["Timestamp"] = pd.to_datetime(today_bars["EpochMs"], unit="ms", errors="coerce")
print(f"  Filtered to {len(today_bars)} common-stock tickers.")

# ══════════════════════════════════════════════════════════
# STEP 2: For each ticker, pull history from CH + compute indicators
# ══════════════════════════════════════════════════════════

LOOKBACK = 260  # need ~252 for SMA_200 + buffer

# ── Indicator helpers ─────────────────────────────────────

def _rsi(series, length=14):
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/float(length), adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/float(length), adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    return (100 - (100 / (1 + rs))).fillna(50.0)


def _ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def _macd(close, fast=12, slow=26, signal=9):
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_f - ema_s
    sig_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - sig_line
    return macd_line, sig_line, hist


def _stoch_rsi(close, rsi_period=14, stoch_period=14, k_smooth=3, d_smooth=3):
    r = _rsi(close, rsi_period)
    lo = r.rolling(stoch_period).min()
    hi = r.rolling(stoch_period).max()
    stoch = (r - lo) / (hi - lo).replace(0, np.nan)
    k = stoch.rolling(k_smooth).mean() * 100
    d = k.rolling(d_smooth).mean()
    return k.fillna(50), d.fillna(50)


def _compute_all_indicators(df):
    """Given a DataFrame with Timestamp,Ticker,Open,High,Low,Close,Volume,Transactions
    sorted by Timestamp, compute all 82 columns and return the LAST ROW only."""
    df = df.sort_values("Timestamp").reset_index(drop=True)
    c = df["Close"].astype(float)
    h = df["High"].astype(float)
    l = df["Low"].astype(float)
    o = df["Open"].astype(float)
    v = df["Volume"].astype(float)

    n = len(df)
    last = n - 1

    # -- Moving averages --
    for p in [10, 20, 50, 100, 200]:
        df[f"SMA_{p}"] = c.rolling(p, min_periods=max(1, int(p*0.6))).mean()
    for s in [8, 21, 50]:
        df[f"EMA_{s}"] = _ema(c, s)

    # -- RSI --
    df["RSI_5"] = _rsi(c, 5)
    df["RSI_14"] = _rsi(c, 14)

    # -- Bollinger --
    bb_mid = c.rolling(20, min_periods=20).mean()
    bb_std = c.rolling(20, min_periods=20).std(ddof=0)
    df["BB_Middle_20"] = bb_mid
    df["BB_Upper_20"] = bb_mid + 2 * bb_std
    df["BB_Lower_20"] = bb_mid - 2 * bb_std

    # -- Volume stats --
    df["DollarVolume"] = c * v
    df["AvgVol_5"] = v.rolling(5).mean()
    df["AvgVol_20"] = v.rolling(20).mean()
    txn = df.get("Transactions", pd.Series(np.nan, index=df.index)).astype(float)
    df["AvgTradeSize"] = np.where(txn > 0, v / txn, np.nan)

    # -- Returns & gaps --
    df["PrevClose"] = c.shift(1)
    df["GapPct"] = (o / c.shift(1) - 1)
    df["DayPct"] = c.pct_change()
    df["Ret_1d"] = c.pct_change(1)
    df["Ret_5d"] = c.pct_change(5)
    df["Ret_20d"] = c.pct_change(20)
    df["Ret_60d"] = c.pct_change(60)
    df["Ret_252d"] = c.pct_change(252)

    # Forward returns: NaN for today (we don't know the future)
    df["Fwd1d"] = np.nan
    df["Fwd5d"] = np.nan
    df["Fwd10d"] = np.nan

    # -- Volatility --
    log_ret = np.log(c / c.shift(1))
    df["Vol_20"] = log_ret.rolling(20).std() * np.sqrt(252)
    df["Vol_60"] = log_ret.rolling(60).std() * np.sqrt(252)
    avg_ret_20 = log_ret.rolling(20).mean()
    std_ret_20 = log_ret.rolling(20).std()
    df["SharpeLike_20"] = avg_ret_20 / std_ret_20.replace(0, np.nan)
    avg_ret_60 = log_ret.rolling(60).mean()
    std_ret_60 = log_ret.rolling(60).std()
    df["SharpeLike_60"] = avg_ret_60 / std_ret_60.replace(0, np.nan)

    # -- Drawdown --
    cum_max = c.cummax()
    df["Drawdown"] = (c / cum_max - 1)

    # -- Streaks --
    daily_chg = c.diff()
    up = (daily_chg > 0).astype(int)
    dn = (daily_chg < 0).astype(int)
    # Compute streaks via cumsum trick
    def _streak(mask):
        groups = (~mask).cumsum()
        return mask.groupby(groups).cumsum()
    df["UpStreak"] = _streak(up)
    df["DownStreak"] = _streak(dn)

    vol_up = (v > v.shift(1)).astype(int)
    vol_dn = (v < v.shift(1)).astype(int)
    df["VolUpStreak"] = _streak(vol_up)
    df["VolDownStreak"] = _streak(vol_dn)

    # -- Extremes --
    df["High_20d"] = h.rolling(20).max()
    df["Low_20d"] = l.rolling(20).min()
    df["High_50d"] = h.rolling(50).max()
    df["Low_50d"] = l.rolling(50).min()
    df["High_252d"] = h.rolling(252).max()
    df["Low_252d"] = l.rolling(252).min()
    df["Is_20d_High"] = (c >= df["High_20d"]).astype(float)
    df["Is_20d_Low"] = (c <= df["Low_20d"]).astype(float)

    # -- True Range / ATR --
    pc = c.shift(1)
    tr = pd.concat([(h-l).abs(), (h-pc).abs(), (l-pc).abs()], axis=1).max(axis=1)
    df["TrueRange"] = tr
    df["ATR_14"] = tr.rolling(14).mean()
    df["RangePct"] = (h - l) / c.replace(0, np.nan)
    gap_std = df["GapPct"].rolling(20).std()
    df["GapZ"] = df["GapPct"] / gap_std.replace(0, np.nan)

    # -- MACD --
    macd_line, sig_line, hist = _macd(c)
    df["MACD"] = macd_line
    df["MACD_Signal"] = sig_line
    df["MACD_Hist"] = hist

    # -- StochRSI --
    k, d = _stoch_rsi(c)
    df["StochRSI_K"] = k
    df["StochRSI_D"] = d

    # -- Slopes (1-day change of SMA, normalized by close) --
    for p in [10, 20, 50, 100, 200]:
        sma = df[f"SMA_{p}"]
        df[f"SMA_{p}_Slope1"] = sma.diff() / c.replace(0, np.nan) * 100
    for s in [8, 21, 50]:
        ema = df[f"EMA_{s}"]
        df[f"EMA_{s}_Slope1"] = ema.diff() / c.replace(0, np.nan) * 100

    # -- Relative strength (placeholder — needs SPY/QQQ) --
    # These get filled in a second pass below
    for col in ["RS_SPY", "RS_SPY_Chg_1d", "RS_SPY_Chg_20d",
                 "RS_QQQ", "RS_QQQ_Chg_1d", "RS_QQQ_Chg_20d"]:
        if col not in df.columns:
            df[col] = np.nan

    # -- VolRank --
    df["VolRank_20"] = v.rolling(20).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )
    df["VolRank_60"] = v.rolling(60).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
    )

    # -- Raw prices (copy of adjusted for now) --
    df["RawOpen"] = o
    df["RawHigh"] = h
    df["RawLow"] = l
    df["RawClose"] = c

    return df


# ── Column order must match DDL ───────────────────────────
EXPECTED_COLS = [
    "Timestamp", "Ticker", "Open", "High", "Low", "Close", "Volume",
    "Transactions", "RawOpen", "RawHigh", "RawLow", "RawClose",
    "SMA_10", "SMA_20", "SMA_50", "SMA_100", "SMA_200",
    "EMA_8", "EMA_21", "EMA_50", "RSI_5", "RSI_14",
    "BB_Middle_20", "BB_Upper_20", "BB_Lower_20",
    "DollarVolume", "AvgVol_5", "AvgVol_20", "AvgTradeSize",
    "PrevClose", "GapPct", "DayPct",
    "Ret_1d", "Ret_5d", "Ret_20d", "Ret_60d", "Ret_252d",
    "Fwd1d", "Fwd5d", "Fwd10d",
    "Vol_20", "Vol_60", "SharpeLike_20", "SharpeLike_60",
    "Drawdown", "UpStreak", "DownStreak", "VolUpStreak", "VolDownStreak",
    "High_20d", "Low_20d", "High_50d", "Low_50d", "High_252d", "Low_252d",
    "Is_20d_High", "Is_20d_Low",
    "TrueRange", "ATR_14", "RangePct", "GapZ",
    "MACD", "MACD_Signal", "MACD_Hist", "StochRSI_K", "StochRSI_D",
    "SMA_10_Slope1", "SMA_20_Slope1", "SMA_50_Slope1",
    "SMA_100_Slope1", "SMA_200_Slope1",
    "EMA_8_Slope1", "EMA_21_Slope1", "EMA_50_Slope1",
    "RS_SPY", "RS_SPY_Chg_1d", "RS_SPY_Chg_20d",
    "RS_QQQ", "RS_QQQ_Chg_1d", "RS_QQQ_Chg_20d",
    "VolRank_20", "VolRank_60",
]


# ══════════════════════════════════════════════════════════
# STEP 3: Process tickers in batches
# ══════════════════════════════════════════════════════════

# First, get existing tickers in CH so we only process those + any new ones
existing_tickers = set()
try:
    res = client.query("SELECT DISTINCT Ticker FROM daily_ohlcv")
    existing_tickers = {r[0] for r in res.result_rows}
except Exception:
    pass

print(f"  {len(existing_tickers)} tickers already in ClickHouse.")

# Pre-fetch SPY and QQQ history for relative strength
def _fetch_ch_history(ticker, days=LOOKBACK):
    try:
        df = client.query_df(
            "SELECT * FROM daily_ohlcv WHERE Ticker = {t:String} "
            "ORDER BY Timestamp DESC LIMIT {n:UInt32}",
            parameters={"t": ticker, "n": days},
        )
        return df.sort_values("Timestamp").reset_index(drop=True) if df is not None and len(df) > 0 else None
    except Exception:
        return None

print("Fetching SPY/QQQ history for relative strength...")
spy_hist = _fetch_ch_history("SPY", LOOKBACK)
qqq_hist = _fetch_ch_history("QQQ", LOOKBACK)

ticker_list = today_bars["Ticker"].tolist()
BATCH = 50
errors = []
inserted = 0
t0 = time.time()

# Delete existing rows for target_date to allow re-runs
target_ts = pd.Timestamp(target_date)
try:
    client.command(
        f"ALTER TABLE daily_ohlcv DELETE WHERE toDate(Timestamp) = '{target_date}'"
    )
    print(f"  Cleared existing rows for {target_date}.")
except Exception as e:
    print(f"  Warning: could not clear old rows: {e}")

for batch_start in range(0, len(ticker_list), BATCH):
    batch_tickers = ticker_list[batch_start : batch_start + BATCH]
    rows_to_insert = []

    for ticker in batch_tickers:
        try:
            # Get today's bar
            bar = today_bars[today_bars["Ticker"] == ticker].iloc[0]

            # Pull history from ClickHouse
            hist = _fetch_ch_history(ticker, LOOKBACK)

            # Build combined DataFrame: history + today's bar
            today_row = pd.DataFrame([{
                "Timestamp": bar["Timestamp"],
                "Ticker": ticker,
                "Open": float(bar["Open"]),
                "High": float(bar["High"]),
                "Low": float(bar["Low"]),
                "Close": float(bar["Close"]),
                "Volume": float(bar["Volume"]),
                "Transactions": float(bar.get("Transactions", 0) or 0),
            }])

            if hist is not None and len(hist) > 0:
                # Keep only essential columns from history for concat
                keep = ["Timestamp", "Ticker", "Open", "High", "Low", "Close",
                        "Volume", "Transactions"]
                hist_slim = hist[[c for c in keep if c in hist.columns]].copy()
                # Remove any existing row for target_date
                hist_slim = hist_slim[hist_slim["Timestamp"].dt.date != target_ts.date()]
                combined = pd.concat([hist_slim, today_row], ignore_index=True)
            else:
                combined = today_row

            if len(combined) < 2:
                # Not enough history to compute anything meaningful
                continue

            # Compute all indicators
            full = _compute_all_indicators(combined)

            # Compute relative strength vs SPY/QQQ
            if spy_hist is not None and len(spy_hist) > 0:
                spy_c = spy_hist.set_index("Timestamp")["Close"].reindex(
                    full["Timestamp"], method="ffill"
                )
                rs_spy = full["Close"].values / spy_c.values
                rs_spy_s = pd.Series(rs_spy, index=full.index)
                full["RS_SPY"] = rs_spy_s
                full["RS_SPY_Chg_1d"] = rs_spy_s.pct_change(1)
                full["RS_SPY_Chg_20d"] = rs_spy_s.pct_change(20)

            if qqq_hist is not None and len(qqq_hist) > 0:
                qqq_c = qqq_hist.set_index("Timestamp")["Close"].reindex(
                    full["Timestamp"], method="ffill"
                )
                rs_qqq = full["Close"].values / qqq_c.values
                rs_qqq_s = pd.Series(rs_qqq, index=full.index)
                full["RS_QQQ"] = rs_qqq_s
                full["RS_QQQ_Chg_1d"] = rs_qqq_s.pct_change(1)
                full["RS_QQQ_Chg_20d"] = rs_qqq_s.pct_change(20)

            # Extract ONLY the last row (today)
            last_row = full.iloc[[-1]].copy()

            # Align columns
            for col in EXPECTED_COLS:
                if col not in last_row.columns:
                    last_row[col] = 0.0 if col not in ("Ticker",) else ""
            last_row = last_row[EXPECTED_COLS]

            # Replace NaN/inf
            float_cols = last_row.select_dtypes("float").columns
            last_row[float_cols] = last_row[float_cols].replace(
                [np.inf, -np.inf], np.nan
            ).fillna(0.0)

            rows_to_insert.append(last_row)

        except Exception as e:
            errors.append((ticker, str(e)))

    if rows_to_insert:
        batch_df = pd.concat(rows_to_insert, ignore_index=True)
        try:
            client.insert_df("daily_ohlcv", batch_df)
            inserted += len(batch_df)
        except Exception as e:
            errors.append((f"batch_{batch_start}", str(e)))

    elapsed = time.time() - t0
    done = min(batch_start + BATCH, len(ticker_list))
    print(f"  [{done:>5}/{len(ticker_list)}]  inserted={inserted}  {elapsed:.0f}s", end="\r")

elapsed = time.time() - t0
print(f"\n\nDone in {elapsed:.1f}s.")
print(f"  Inserted: {inserted} rows for {target_date}")
print(f"  Errors:   {len(errors)}")
if errors:
    for name, err in errors[:10]:
        print(f"    {name}: {err}")

# Verify
total = client.command("SELECT count() FROM daily_ohlcv")
tickers = client.command("SELECT uniq(Ticker) FROM daily_ohlcv")
latest = client.command("SELECT max(Timestamp) FROM daily_ohlcv")
print(f"\nClickHouse totals — Rows: {total:,}  Tickers: {tickers}  Latest: {latest}")
