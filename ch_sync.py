"""
Incremental sync: push only changed parquets to ClickHouse.

Checks file mtime vs last sync timestamp. Only re-uploads files
modified since the last run.

Usage:
    py -3 ch_sync.py              # sync both daily + minute
    py -3 ch_sync.py daily        # sync daily only
    py -3 ch_sync.py minute       # sync minute only

Runs after your local parquet builder finishes.
Schedule with Windows Task Scheduler or just run manually.
"""

import os, sys, time, json
from pathlib import Path
from datetime import datetime
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

SYNC_STATE_FILE = Path(__file__).parent / ".ch_sync_state.json"

# ── Column definitions (must match ch_upload.py DDL) ──────

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


# ── Sync state persistence ────────────────────────────────

def load_state():
    if SYNC_STATE_FILE.exists():
        return json.loads(SYNC_STATE_FILE.read_text())
    return {"daily_last_sync": 0, "minute_last_sync": 0}


def save_state(state):
    SYNC_STATE_FILE.write_text(json.dumps(state, indent=2))


# ── Sync logic ────────────────────────────────────────────

def find_changed_files(pq_dir, since_ts):
    """Return list of parquet files modified after since_ts (epoch seconds)."""
    changed = []
    for fp in Path(pq_dir).glob("*.parquet"):
        if not fp.stem[0].isalpha():
            continue
        if fp.stat().st_mtime > since_ts:
            changed.append(fp)
    return sorted(changed)


def sync_dir(client, pq_dir, table_name, expected_cols, since_ts, batch_size):
    changed = find_changed_files(pq_dir, since_ts)
    if not changed:
        print(f"  {table_name}: no files changed since last sync.")
        return since_ts

    print(f"  {table_name}: {len(changed)} files changed, syncing...")

    # For each changed ticker, DELETE old rows then INSERT fresh
    errors = []
    t0 = time.time()

    for i in range(0, len(changed), batch_size):
        batch_files = changed[i : i + batch_size]
        frames = []
        tickers_in_batch = []

        for fp in batch_files:
            try:
                df = pd.read_parquet(fp)
                ticker = fp.stem
                tickers_in_batch.append(ticker)

                if "Timestamp" in df.columns:
                    ts = df["Timestamp"]
                    if hasattr(ts.dtype, "tz") and ts.dtype.tz is not None:
                        ts = ts.dt.tz_convert("America/New_York").dt.tz_localize(None)
                    else:
                        ts = pd.to_datetime(ts, errors="coerce")
                    df["Timestamp"] = ts
                    df = df.dropna(subset=["Timestamp"])

                float_cols = df.select_dtypes("float").columns
                df[float_cols] = df[float_cols].fillna(0.0)
                frames.append(df)
            except Exception as e:
                errors.append((fp.name, str(e)))

        if not frames:
            continue

        # Delete old data for these tickers
        ticker_list = ", ".join(f"'{t}'" for t in tickers_in_batch)
        try:
            client.command(
                f"ALTER TABLE {table_name} DELETE WHERE Ticker IN ({ticker_list})"
            )
        except Exception as e:
            # Table might use lightweight deletes or mutations
            errors.append(("delete", str(e)))

        big = pd.concat(frames, ignore_index=True)

        # Align columns
        for col in expected_cols:
            if col not in big.columns:
                big[col] = 0.0 if col != "Ticker" else ""
        big = big[expected_cols]

        try:
            client.insert_df(table_name, big)
        except Exception as e:
            errors.append((f"batch {i}", str(e)))
            continue

        done = i + len(batch_files)
        elapsed = time.time() - t0
        print(f"    [{done:>5}/{len(changed)}]  {elapsed:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"  {table_name}: synced in {elapsed:.1f}s. Errors: {len(errors)}")
    if errors:
        for name, err in errors[:10]:
            print(f"    {name}: {err}")

    # Verify
    count = client.command(f"SELECT count() FROM {table_name}")
    tickers = client.command(f"SELECT uniq(Ticker) FROM {table_name}")
    print(f"  {table_name}: {count:,} rows, {tickers} tickers total")

    return time.time()


# ── Main ──────────────────────────────────────────────────

client = clickhouse_connect.get_client(
    host=CH_HOST, port=CH_PORT,
    username=CH_USER, password=CH_PASS,
    database=CH_DB, secure=True,
)
print(f"Connected to {CH_HOST}")

state = load_state()
mode = (sys.argv[1] if len(sys.argv) > 1 else "all").lower()
now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

if mode in ("daily", "all") and DAILY_DIR and Path(DAILY_DIR).is_dir():
    print(f"\n[{now_str}] Daily sync (last: {datetime.fromtimestamp(state['daily_last_sync']).strftime('%Y-%m-%d %H:%M') if state['daily_last_sync'] else 'never'})")
    state["daily_last_sync"] = sync_dir(
        client, DAILY_DIR, "daily_ohlcv", DAILY_COLS,
        state["daily_last_sync"], batch_size=200,
    )

if mode in ("minute", "all") and MINUTE_DIR and Path(MINUTE_DIR).is_dir():
    print(f"\n[{now_str}] Minute sync (last: {datetime.fromtimestamp(state['minute_last_sync']).strftime('%Y-%m-%d %H:%M') if state['minute_last_sync'] else 'never'})")
    state["minute_last_sync"] = sync_dir(
        client, MINUTE_DIR, "minute_ohlcv", MINUTE_COLS,
        state["minute_last_sync"], batch_size=5,
    )

save_state(state)
print(f"\nSync complete. State saved to {SYNC_STATE_FILE.name}")
