"""
Upload per-ticker parquet files to ClickHouse.

Usage:
    py -3 ch_upload.py daily      # upload daily parquets only
    py -3 ch_upload.py minute     # upload minute parquets only
    py -3 ch_upload.py all        # upload both (default)
"""

import os, sys, time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

try:
    import clickhouse_connect
except ImportError:
    sys.exit("pip install clickhouse-connect")

import pandas as pd

# ── Config ────────────────────────────────────────────────
CH_HOST = os.environ["CLICKHOUSE_HOST"]
CH_PORT = int(os.environ.get("CLICKHOUSE_PORT", "8443"))
CH_USER = os.environ.get("CLICKHOUSE_USER", "default")
CH_PASS = os.environ["CLICKHOUSE_PASSWORD"]
CH_DB   = os.environ.get("CLICKHOUSE_DATABASE", "default")

DAILY_DIR  = os.environ.get("PER_TICKER_PARQUET_DIR", "")
MINUTE_DIR = os.environ.get("PER_TICKER_MINUTE_DIR", "")

# ── Connect ───────────────────────────────────────────────
client = clickhouse_connect.get_client(
    host=CH_HOST, port=CH_PORT,
    username=CH_USER, password=CH_PASS,
    database=CH_DB, secure=True,
)
print(f"Connected to {CH_HOST}:{CH_PORT}/{CH_DB}")

# ── DDL ───────────────────────────────────────────────────

DAILY_DDL = """
CREATE TABLE IF NOT EXISTS daily_ohlcv (
    Timestamp    DateTime64(3),
    Ticker       LowCardinality(String),
    Open         Float64,
    High         Float64,
    Low          Float64,
    Close        Float64,
    Volume       Float64,
    Transactions Float64,
    RawOpen      Float64,
    RawHigh      Float64,
    RawLow       Float64,
    RawClose     Float64,
    SMA_10       Float64,
    SMA_20       Float64,
    SMA_50       Float64,
    SMA_100      Float64,
    SMA_200      Float64,
    EMA_8        Float64,
    EMA_21       Float64,
    EMA_50       Float64,
    RSI_5        Float64,
    RSI_14       Float64,
    BB_Middle_20 Float64,
    BB_Upper_20  Float64,
    BB_Lower_20  Float64,
    DollarVolume Float64,
    AvgVol_5     Float64,
    AvgVol_20    Float64,
    AvgTradeSize Float64,
    PrevClose    Float64,
    GapPct       Float64,
    DayPct       Float64,
    Ret_1d       Float64,
    Ret_5d       Float64,
    Ret_20d      Float64,
    Ret_60d      Float64,
    Ret_252d     Float64,
    Fwd1d        Float64,
    Fwd5d        Float64,
    Fwd10d       Float64,
    Vol_20       Float64,
    Vol_60       Float64,
    SharpeLike_20  Float64,
    SharpeLike_60  Float64,
    Drawdown     Float64,
    UpStreak     Float64,
    DownStreak   Float64,
    VolUpStreak  Float64,
    VolDownStreak Float64,
    High_20d     Float64,
    Low_20d      Float64,
    High_50d     Float64,
    Low_50d      Float64,
    High_252d    Float64,
    Low_252d     Float64,
    Is_20d_High  Float64,
    Is_20d_Low   Float64,
    TrueRange    Float64,
    ATR_14       Float64,
    RangePct     Float64,
    GapZ         Float64,
    MACD         Float64,
    MACD_Signal  Float64,
    MACD_Hist    Float64,
    StochRSI_K   Float64,
    StochRSI_D   Float64,
    SMA_10_Slope1  Float64,
    SMA_20_Slope1  Float64,
    SMA_50_Slope1  Float64,
    SMA_100_Slope1 Float64,
    SMA_200_Slope1 Float64,
    EMA_8_Slope1   Float64,
    EMA_21_Slope1  Float64,
    EMA_50_Slope1  Float64,
    RS_SPY       Float64,
    RS_SPY_Chg_1d  Float64,
    RS_SPY_Chg_20d Float64,
    RS_QQQ       Float64,
    RS_QQQ_Chg_1d  Float64,
    RS_QQQ_Chg_20d Float64,
    VolRank_20   Float64,
    VolRank_60   Float64
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(Timestamp)
ORDER BY (Ticker, Timestamp)
"""

MINUTE_DDL = """
CREATE TABLE IF NOT EXISTS minute_ohlcv (
    Timestamp    DateTime64(3),
    Ticker       LowCardinality(String),
    Open         Float64,
    High         Float64,
    Low          Float64,
    Close        Float64,
    Volume       Float64,
    SMA_20       Float64,
    SMA_50       Float64,
    EMA_8        Float64,
    EMA_21       Float64,
    RSI_5        Float64,
    RSI_14       Float64,
    BB_Middle_20 Float64,
    BB_Upper_20  Float64,
    BB_Lower_20  Float64,
    DollarVolume Float64,
    Transactions Float64,
    AvgTradeSize Float64
)
ENGINE = MergeTree()
PARTITION BY toYYYYMM(Timestamp)
ORDER BY (Ticker, Timestamp)
"""

DAILY_COLS = [
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

MINUTE_COLS = [
    "Timestamp", "Ticker", "Open", "High", "Low", "Close", "Volume",
    "SMA_20", "SMA_50", "EMA_8", "EMA_21",
    "RSI_5", "RSI_14",
    "BB_Middle_20", "BB_Upper_20", "BB_Lower_20",
    "DollarVolume", "Transactions", "AvgTradeSize",
]


# ── Upload helper ─────────────────────────────────────────

def upload_dir(pq_dir, table_name, expected_cols, batch_size=50):
    pq_files = sorted(Path(pq_dir).glob("*.parquet"))
    pq_files = [f for f in pq_files if f.stem[0].isalpha()]
    print(f"\n{'='*60}")
    print(f"Uploading {len(pq_files)} files -> {table_name}")
    print(f"{'='*60}")

    errors = []
    t0 = time.time()

    for i in range(0, len(pq_files), batch_size):
        batch_files = pq_files[i : i + batch_size]
        frames = []
        for fp in batch_files:
            try:
                df = pd.read_parquet(fp)
                # Ensure Timestamp is naive datetime (strip timezone)
                if "Timestamp" in df.columns:
                    ts = df["Timestamp"]
                    if hasattr(ts.dtype, "tz") and ts.dtype.tz is not None:
                        ts = ts.dt.tz_convert("America/New_York").dt.tz_localize(None)
                    else:
                        ts = pd.to_datetime(ts, errors="coerce")
                    df["Timestamp"] = ts
                    df = df.dropna(subset=["Timestamp"])
                # Fill NaN for Float64 columns
                float_cols = df.select_dtypes("float").columns
                df[float_cols] = df[float_cols].fillna(0.0)
                frames.append(df)
            except Exception as e:
                errors.append((fp.name, str(e)))

        if not frames:
            continue

        big = pd.concat(frames, ignore_index=True)

        # Align columns
        for col in expected_cols:
            if col not in big.columns:
                big[col] = 0.0 if col != "Ticker" else ""
        big = big[expected_cols]

        try:
            client.insert_df(table_name, big)
        except Exception as e:
            errors.append((f"batch {i}-{i+len(batch_files)}", str(e)))
            continue

        done = i + len(batch_files)
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        eta = (len(pq_files) - done) / rate if rate > 0 else 0
        print(f"  [{done:>5}/{len(pq_files)}]  {elapsed:.0f}s  (~{eta:.0f}s remaining)", flush=True)

    elapsed = time.time() - t0
    print(f"\n{table_name}: Done in {elapsed:.1f}s.  Errors: {len(errors)}")
    if errors:
        for name, err in errors[:20]:
            print(f"  {name}: {err}")

    count = client.command(f"SELECT count() FROM {table_name}")
    tickers = client.command(f"SELECT uniq(Ticker) FROM {table_name}")
    print(f"  Rows: {count:,}   Unique tickers: {tickers}")


# ── Main ──────────────────────────────────────────────────

mode = (sys.argv[1] if len(sys.argv) > 1 else "all").lower()

if mode in ("daily", "all"):
    if not DAILY_DIR or not Path(DAILY_DIR).is_dir():
        print(f"WARNING: PER_TICKER_PARQUET_DIR not set or missing: {DAILY_DIR!r}")
    else:
        client.command(DAILY_DDL)
        print("Table daily_ohlcv ready.")
        upload_dir(DAILY_DIR, "daily_ohlcv", DAILY_COLS, batch_size=200)

if mode in ("minute", "all"):
    if not MINUTE_DIR or not Path(MINUTE_DIR).is_dir():
        print(f"WARNING: PER_TICKER_MINUTE_DIR not set or missing: {MINUTE_DIR!r}")
    else:
        client.command(MINUTE_DDL)
        print("Table minute_ohlcv ready.")
        # Smaller batches for minute data (much more rows per file)
        upload_dir(MINUTE_DIR, "minute_ohlcv", MINUTE_COLS, batch_size=5)

print("\nAll done!")
