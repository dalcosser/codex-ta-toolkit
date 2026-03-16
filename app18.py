import streamlit as st
import pandas as pd
import numpy as np
import numpy as _np  # legacy alias kept for backward compat
from typing import Sequence, Optional
import httpx
import csv
import io
import os
import requests

def _parse_ts_to_et(series, assume_naive_ny: bool = False):
    try:
        s = pd.to_numeric(series, errors='coerce')
        if pd.api.types.is_numeric_dtype(s):
            mx = s.dropna().max()
            unit = 'ns' if (mx is not None and mx > 1e18) else ('ms' if (mx is not None and mx > 1e12) else 's')
            ts = pd.to_datetime(s, unit=unit, utc=True)
            return ts.dt.tz_convert('America/New_York')
    except Exception:
        pass
    ts = pd.to_datetime(series, errors='coerce', utc=True)
    try:
        # If resulting dtype is naive via earlier code paths, localize based on preference
        if getattr(ts.dtype, 'tz', None) is None:
            if assume_naive_ny:
                ts = pd.to_datetime(series, errors='coerce').dt.tz_localize('America/New_York')
            else:
                ts = pd.to_datetime(series, errors='coerce').dt.tz_localize('UTC').dt.tz_convert('America/New_York')
            return ts
    except Exception:
        pass
    try:
        return ts.tz_convert('America/New_York')
    except Exception:
        return ts
from pathlib import Path
try:
    from ch_reader import ch_load_daily_df, ch_available
except ImportError:
    ch_load_daily_df = lambda t: None
    ch_available = lambda: False

from offopen_utils import compute_offopen_for_dates, load_minute_dataframe

HAS_MASSIVE_SDK = False
try:
    from massive import RESTClient as MassiveRESTClient  # type: ignore
    HAS_MASSIVE_SDK = True
except Exception:
    MassiveRESTClient = None  # type: ignore
    HAS_MASSIVE_SDK = False

def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Robust OHLCV normalizer: maps common column aliases (o/h/l/c/v, open/high/low/close/volume, t/time) to
    Open/High/Low/Close/Volume and keeps Timestamp if present.
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df
    out = df.copy()
    cols = list(out.columns)
    lower_map = {str(c).lower(): c for c in cols}
    alias_map = {
        "o": "Open", "op": "Open", "open": "Open",
        "h": "High", "high": "High",
        "l": "Low", "low": "Low",
        "c": "Close", "close": "Close",
        "v": "Volume", "volume": "Volume",
        "t": "Timestamp", "time": "Timestamp", "timestamp": "Timestamp", "datetime": "Timestamp",
    }
    ren = {}
    for k, tgt in alias_map.items():
        src = lower_map.get(k)
        if src is not None:
            ren[src] = tgt
    if ren:
        out = out.rename(columns=ren)
    return out


def _show_scan_banner():
    """Show the last cached scan stats banner if available."""
    try:
        banner = st.session_state.get("last_scan_stats_caption")
        if banner:
            st.info(f"Last scan stats: {banner}")
    except Exception:
        pass

def _earnings_flag_for_dates(ticker: str, idx: pd.Index) -> pd.Series:
    """Return boolean series marking if prior calendar day or same day had earnings."""
    try:
        def _aliases(sym: str) -> list[str]:
            s = (sym or "").strip().upper()
            alts = {s}
            if s.endswith("L"):  # GOOGL -> GOOG
                alts.add(s[:-1])
            if s and s[-1].isdigit() is False and s[-1].isalpha() and len(s) > 3:
                alts.add(s[:-1])
            if s == "GOOG":
                alts.add("GOOGL")
            return list(alts)

        ed_list: list[pd.Timestamp] = []
        for sym in _aliases(ticker):
            try:
                ed_list.extend(_get_earnings_dates(sym, limit=200) or [])
            except Exception:
                pass
        ed_dates = {pd.to_datetime(x).normalize().date() for x in ed_list if pd.notna(pd.to_datetime(x))}
        out = []
        for ts in pd.to_datetime(idx):
            try:
                dt = pd.to_datetime(ts).normalize()
                prev_date = (dt - pd.Timedelta(days=1)).date()
                out.append((prev_date in ed_dates) or (dt.date() in ed_dates))
            except Exception:
                out.append(False)
        return pd.Series(out, index=idx)
    except Exception:
        return pd.Series([False] * len(idx), index=idx)

def _prune_scan_columns(df: pd.DataFrame, max_cols: int = 14) -> pd.DataFrame:
    """Keep the most relevant scan columns to avoid dumping full parquet feature sets."""
    if df is None or df.empty:
        return df
    cols = list(df.columns)
    keywords = [
        "Date", "Open", "High", "Low", "Close", "Volume", "AvgVol", "Gap", "Return", "%",
        "Next", "Prev", "RS", "BB", "Vol", "Type", "SMA", "EMA", "LL", "Pull", "Width", "Rank", "Score", "Earnings", "Reason"
    ]
    keep = [c for c in cols if any(k in str(c) for k in keywords)]
    if not keep:
        keep = cols[:max_cols]
    else:
        keep = keep[:max_cols]
    try:
        return df[keep]
    except Exception:
        return df


# Bloomberg terminal integration (requires blpapi + terminal running)
try:
    import xbbg  # noqa: F401
    HAS_XBBG = True
except ImportError:
    HAS_XBBG = False

# Environment loading (uses find_dotenv; diagnostics stored in ENV_DIAG_CAPTION)
DOTENV_FILE = None
ENV_DIAG_CAPTION = None
try:
    from dotenv import load_dotenv, find_dotenv  # type: ignore
    found_path = find_dotenv(usecwd=True)
    if found_path:
        DOTENV_FILE = Path(found_path)
        load_dotenv(found_path, override=True)
    else:
        # Fallback to local .env next to this file
        DOTENV_FILE = Path(__file__).with_name('.env')
        if DOTENV_FILE.exists():
            load_dotenv(DOTENV_FILE, override=True)
except Exception:
    DOTENV_FILE = DOTENV_FILE if DOTENV_FILE is not None else None

# Pin per-ticker Parquet folder next to this app file for deterministic startup
DATA_DIR = Path(__file__).resolve().parent / "per_ticker_daily"
os.environ["PER_TICKER_PARQUET_DIR"] = str(DATA_DIR)

# Optional local minute data directory for scans (set to your folder)
MINUTE_DIR = Path(r"C:\Users\David Alcosser\Documents\Visual Code\codex_ta_toolkit\per_ticker_minute")
os.environ["PER_TICKER_MINUTE_DIR"] = str(MINUTE_DIR)

# Normalize Polygon/Massive API key from .env into process env and session state
try:
    _POLY_ENV = (os.getenv("POLYGON_API_KEY") or os.getenv("MASSIVE_API_KEY") or "").strip()
    os.environ["POLYGON_API_KEY"] = _POLY_ENV
    os.environ["MASSIVE_API_KEY"] = _POLY_ENV
    # Seed session_state so all code paths pick it up without manual typing
    if hasattr(st, "session_state"):
        if ("polygon_api_key" not in st.session_state) or (not st.session_state.get("polygon_api_key")):
            st.session_state["polygon_api_key"] = _POLY_ENV
    # Lightweight diagnostic to confirm .env was seen (masked)
    try:
        env_path = str(DOTENV_FILE) if DOTENV_FILE and Path(DOTENV_FILE).exists() else '(not found)'
        ENV_DIAG_CAPTION = f".env: {env_path} | loaded: {env_path != '(not found)'} | Polygon/Massive key len={len(_POLY_ENV)} last6={_POLY_ENV[-6:] if _POLY_ENV else '-'}"
    except Exception:
        ENV_DIAG_CAPTION = None
except Exception:
    pass

# ============== Scans: robust helpers ==============

def _bbg_quote(sym: str) -> dict | None:
    if not HAS_XBBG or not sym:
        return None
    try:
        tkr = f"{sym.strip().upper()} US Equity" if " " not in sym.strip() else sym.strip()
        dfb = blp.bdp(tickers=[tkr], flds=["PX_LAST","PX_VOLUME","NAME"])
        if dfb is None or dfb.empty:
            return None
        rec = dfb.iloc[0]
        return {
            "Ticker": sym.strip().upper(),
            "Name": rec.get("NAME") if "NAME" in dfb.columns else rec.get("name"),
            "Price": rec.get("PX_LAST") if "PX_LAST" in dfb.columns else rec.get("px_last"),
            "Volume": rec.get("PX_VOLUME") if "PX_VOLUME" in dfb.columns else rec.get("px_volume"),
        }
    except Exception:
        return None

# Quick ThetaData streaming snapshot helper (CSV over HTTP)
def _tt_snapshot_stream(
    ticker: str,
    host: str | None = None,
    path: str | None = None,
    limit_rows: int = 50,
) -> tuple[list[list[str]], str]:
    """
    Streams snapshot CSV rows from ThetaData terminal.
    Returns (rows, url_tried). Host/path defaults to env.
    """
    if not ticker:
        return [], ""
    h = (host or os.getenv("THETADATA_TERMINAL_URL", "http://localhost:25503/v3")).strip().rstrip("/")
    p = (path or os.getenv("THETADATA_SNAPSHOT_PATH", "v3/stock/snapshot/market_value")).strip("/")
    if h.endswith("/v3") and p.startswith("v3/"):
        p = p[len("v3/") :]
    url = f"{h}/{p}"
    rows: list[list[str]] = []
    try:
        with httpx.stream("GET", url, params={"symbol": ticker}, timeout=15) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                for row in csv.reader(io.StringIO(line)):
                    rows.append(row)
                    if len(rows) >= limit_rows:
                        break
                if len(rows) >= limit_rows:
                    break
    except Exception:
        return [], url
    return rows, url

def _tt_snapshot_parse(rows: list[list[str]]) -> dict:
    """Attempt to coerce the first row into useful fields and compute simple anomalies."""
    if not rows:
        return {}
    header = None
    data_rows = rows
    # If first row looks like header (contains non-numeric strings), treat it as header
    if any(c.isalpha() for c in "".join(rows[0])):
        header = rows[0]
        data_rows = rows[1:]
    if not data_rows:
        return {}
    row = data_rows[0]
    # Default column guesses
    default_cols = ["symbol","last","bid","ask","mid","volume","open","high","low","prev_close","market_value","timestamp"]
    cols = header if header else default_cols[:len(row)]
    out = {}
    for c,v in zip(cols,row):
        out[str(c).lower()] = v
    def _to_float(x):
        try:
            return float(x)
        except Exception:
            return None
    bid = _to_float(out.get("bid"))
    ask = _to_float(out.get("ask"))
    last = _to_float(out.get("last"))
    mid = None
    if bid is not None and ask is not None and ask != 0:
        mid = (bid+ask)/2
        out["spread_bps"] = (ask - bid) / ask * 1e4
    if mid is None and last is not None:
        mid = last
    out["mid"] = mid
    out["last_f"] = last
    out["bid_f"] = bid
    out["ask_f"] = ask
    vol = _to_float(out.get("volume"))
    out["volume_f"] = vol
    return out

def _tt_option_eod(
    root: str,
    exp: str,
    strike: str,
    right: str,
    start_date: str,
    end_date: str,
    host: str | None = None,
    path: str | None = None,
) -> tuple[pd.DataFrame, str, str]:
    """
    Fetch ThetaData option history eod (non-streaming).
    Returns (DataFrame, url_tried, status_text).
    """
    h = (host or os.getenv("THETADATA_TERMINAL_URL", "http://localhost:25503/v3")).strip().rstrip("/")
    p = (path or os.getenv("THETADATA_OPTION_EOD_PATH", "v3/option/history/eod")).strip("/")
    if h.endswith("/v3") and p.startswith("v3/"):
        p = p[len("v3/"):]
    url = f"{h}/{p}"
    params = {
        "root": root.upper(),
        "exp": exp,
        "strike": strike,
        "right": right.upper(),
        "start_date": start_date,
        "end_date": end_date,
    }
    try:
        resp = requests.get(url, params=params, timeout=20)
        status = f"{resp.status_code}"
        if not resp.ok:
            return pd.DataFrame(), url, f"HTTP {resp.status_code}"
        try:
            js = resp.json()
        except Exception:
            return pd.DataFrame(), url, "Invalid JSON"
        if isinstance(js, list):
            return pd.DataFrame(js), url, status
        elif isinstance(js, dict):
            return pd.DataFrame([js]), url, status
        else:
            return pd.DataFrame(), url, "Unexpected payload"
    except Exception as e:
        return pd.DataFrame(), url, str(e)


def _resolve_parquet_path(ticker: str) -> Path | None:
    base = os.environ.get("PER_TICKER_PARQUET_DIR") or ""
    if not base:
        return None
    p = Path(base)
    if not p.is_dir():
        return None
    t = (ticker or "").strip().upper()
    if not t:
        return None
    candidates = [
        f"{t}.parquet",
        f"{t.replace('.','_')}.parquet",
        f"{t.replace('.','')}.parquet",
        f"{t.lower()}.parquet",
    ]
    for name in candidates:
        fp = p / name
        if fp.exists():
            return fp
    want = ''.join(ch for ch in t if ch.isalnum())
    for fp in p.glob('*.parquet'):
        stem = fp.stem
        key = ''.join(ch for ch in stem.upper() if ch.isalnum())
        if key == want:
            return fp
    return None

@st.cache_data(show_spinner=False)
def _load_minute_local(ticker: str, start_ts: pd.Timestamp, end_ts: pd.Timestamp, assume_naive_ny: bool = False) -> pd.DataFrame | None:
    base = os.environ.get("PER_TICKER_MINUTE_DIR") or ""
    if not base:
        return None
    basep = Path(base)
    if not basep.exists():
        return None
    t = (ticker or "").strip().upper().replace(".", "_")
    candidates: list[Path] = []
    tdir = basep / t
    if tdir.is_dir():
        candidates += list(tdir.glob("*.parquet"))
        candidates += list(tdir.glob("*.csv"))
    candidates += list(basep.glob(f"{t}*.parquet"))
    candidates += list(basep.glob(f"{t}*.csv"))
    dfs: list[pd.DataFrame] = []
    for fp in candidates:
        try:
            if fp.suffix.lower() == ".parquet":
                df = pd.read_parquet(fp)
            else:
                try:
                    df = pd.read_csv(fp, encoding="utf-8")
                except UnicodeDecodeError:
                    df = pd.read_csv(fp, encoding="latin1")
        except Exception:
            continue
        if not isinstance(df, pd.DataFrame) or df.empty:
            continue
        cols = {str(c).lower(): c for c in df.columns}
        # Prefer numeric epoch columns first; avoid free-form 'time' strings like '09:30-16:00'
        epoch_priority = [
            cols.get("t"), cols.get("timestamp"), cols.get("start"), cols.get("start_timestamp"),
            cols.get("end"), cols.get("end_timestamp")
        ]
        text_priority = [cols.get("datetime"), cols.get("date")]
        dcol = None
        for cand in epoch_priority:
            if cand and cand in df.columns:
                ser = pd.to_numeric(df[cand], errors='coerce')
                if ser.notna().any():
                    dcol = cand
                    break
        if dcol is None:
            for cand in text_priority:
                if cand and cand in df.columns:
                    dcol = cand
                    break
        if dcol:
            idx_ny = _parse_ts_to_et(df[dcol], assume_naive_ny=assume_naive_ny)
        else:
            # If no obvious column, try using index if it looks like datetimes
            if isinstance(df.index, pd.DatetimeIndex):
                idx_ny = _parse_ts_to_et(df.index, assume_naive_ny=assume_naive_ny)
            else:
                # Fallback: compose from 'date' + 'time' string (take left HH:MM)
                d_date = cols.get('date')
                d_time = cols.get('time')
                if d_date and d_time and (d_date in df.columns) and (d_time in df.columns):
                    try:
                        dt_series = pd.to_datetime(df[d_date], errors='coerce')
                        tt = df[d_time].astype(str).str.extract(r'^(\d{1,2}:\d{2})')[0]
                        combo = pd.to_datetime(dt_series.dt.strftime('%Y-%m-%d') + ' ' + tt, errors='coerce')
                        if assume_naive_ny:
                            idx_ny = combo.dt.tz_localize('America/New_York')
                        else:
                            idx_ny = combo.dt.tz_localize('UTC').dt.tz_convert('America/New_York')
                    except Exception:
                        continue
                else:
                    continue
        # Always set parsed NY timezone index to ensure consistent anchoring
        df = df.assign(_dt=idx_ny).set_index("_dt")
        # Normalize columns
        def pick(name: str):
            return cols.get(name)
        for out, src in (("Open","open"),("High","high"),("Low","low"),("Close","close"),("Volume","volume")):
            c = pick(src)
            if c in df.columns:
                df[out] = pd.to_numeric(df[c], errors="coerce")
        for k in ("transactions","n","trades","trade_count"):
            c = cols.get(k)
            if c in df.columns:
                df["Transactions"] = pd.to_numeric(df[c], errors="coerce")
                break
        try:
            ny_start = start_ts.tz_convert("America/New_York")
            ny_end = end_ts.tz_convert("America/New_York")
        except Exception:
            ny_start, ny_end = start_ts, end_ts
        part = df[(df.index >= ny_start) & (df.index <= ny_end)]
        if not part.empty:
            keep = [c for c in ("Open","High","Low","Close","Volume","Transactions") if c in part.columns]
            dfs.append(part[keep])
    if not dfs:
        return None
    out = pd.concat(dfs).sort_index()
    try:
        out = out[~out.index.duplicated(keep="last")]
    except Exception:
        pass
    return out


def _load_full_minute_df(ticker: str) -> pd.DataFrame | None:
    """Load the full minute parquet once and cache it in session_state."""
    base = os.environ.get("PER_TICKER_MINUTE_DIR") or ""
    if not base:
        return None
    minute_dir = Path(base)
    key = f"minute_full_{(ticker or '').strip().upper()}"
    cache = None
    try:
        cache = st.session_state.setdefault('_minute_full_cache', {})
    except Exception:
        cache = {}
    if key and cache and key in cache:
        return cache.get(key)
    try:
        df = load_minute_dataframe(minute_dir, ticker)
    except Exception:
        df = None
    if df is not None and not df.empty:
        try:
            cache[key] = df
            st.session_state['_minute_full_cache'] = cache
        except Exception:
            pass
    return df


def _compute_offopen_table_from_minutes(
    ticker: str,
    dates_idx: Sequence[pd.Timestamp],
    minute_marks: Sequence[int],
    tol_min: int,
    allow_nearest: bool,
) -> pd.DataFrame | None:
    minute_df = _load_full_minute_df(ticker)
    if minute_df is None or minute_df.empty:
        return None
    dates = [pd.Timestamp(d) for d in pd.to_datetime(dates_idx)]
    if not dates:
        return pd.DataFrame()
    try:
        return compute_offopen_for_dates(minute_df, dates, minute_marks, int(tol_min), allow_nearest)
    except Exception:
        return None

# Minute inspector helpers
def _list_minute_files(ticker: str) -> list[Path]:
    base = os.environ.get("PER_TICKER_MINUTE_DIR") or ""
    out: list[Path] = []
    if not base:
        return out
    p = Path(base)
    t = (ticker or "").strip().upper().replace(".", "_")
    if not t:
        return out
    if (p / t).is_dir():
        out += sorted((p / t).glob("*.parquet"))
        out += sorted((p / t).glob("*.csv"))
    out += sorted(p.glob(f"{t}*.parquet"))
    out += sorted(p.glob(f"{t}*.csv"))
    # Dedup while preserving order
    seen = set()
    uniq = []
    for fp in out:
        if fp not in seen:
            seen.add(fp)
            uniq.append(fp)
    return uniq

def _load_minute_file_inspect(fp: Path, assume_naive_ny: bool = False) -> pd.DataFrame | None:
    try:
        if fp.suffix.lower() == ".parquet":
            df = pd.read_parquet(fp)
        else:
            try:
                df = pd.read_csv(fp, encoding="utf-8")
            except UnicodeDecodeError:
                df = pd.read_csv(fp, encoding="latin1")
        if not isinstance(df, pd.DataFrame) or df.empty:
            return None
        cols = {str(c).lower(): c for c in df.columns}
        # Prefer numeric epoch columns first; avoid free-form 'time'
        epoch_priority = [
            cols.get("t"), cols.get("timestamp"), cols.get("start"), cols.get("start_timestamp"),
            cols.get("end"), cols.get("end_timestamp")
        ]
        text_priority = [cols.get("datetime"), cols.get("date")]
        dcol = None
        for cand in epoch_priority:
            if cand and cand in df.columns:
                ser = pd.to_numeric(df[cand], errors='coerce')
                if ser.notna().any():
                    dcol = cand
                    break
        if dcol is None:
            for cand in text_priority:
                if cand and cand in df.columns:
                    dcol = cand
                    break
        assume = bool(st.session_state.get('assume_naive_ny_minutes')) if hasattr(st, 'session_state') else False
        if dcol:
            ts = _parse_ts_to_et(df[dcol], assume_naive_ny=assume)
        elif isinstance(df.index, pd.DatetimeIndex):
            ts = _parse_ts_to_et(df.index, assume_naive_ny=assume)
        else:
            return None
        # Already converted to NY by helper
        df = df.copy(); df.index = ts
        # Normalize columns
        for outc, src in (("Open","open"),("High","high"),("Low","low"),("Close","close"),("Volume","volume")):
            c = cols.get(src)
            if c in df.columns:
                df[outc] = pd.to_numeric(df[c], errors="coerce")
        for k in ("transactions","n","trades","trade_count"):
            c = cols.get(k)
            if c in df.columns:
                df["Transactions"] = pd.to_numeric(df[c], errors="coerce")
                break
        return df
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def _load_daily_df(ticker: str) -> pd.DataFrame | None:
    # ── Try ClickHouse first (works from anywhere) ──
    try:
        chdf = ch_load_daily_df(ticker)
        if chdf is not None and not chdf.empty:
            return chdf
    except Exception:
        pass
    # ── Fall back to local parquet pipeline ──
    try:
        return _load_polygon_daily_for_ticker(
            data_root="",
            ticker=ticker,
            reports_dir=None,
            technicals_script=None,
            auto_generate_report=False,
            excel_override=None,
            excel_path_override=None,
            allow_yahoo_fallback=False,
        )
    except Exception:
        pass
    try:
        fp = _resolve_parquet_path(ticker)
        if not fp:
            return None
        dfp = pd.read_parquet(fp)
        cols = {str(c).lower(): c for c in dfp.columns}
        dcol = cols.get('timestamp') or cols.get('date')
        if not dcol:
            return None
        out = dfp.copy()
        out['Date'] = pd.to_datetime(dfp[dcol], errors='coerce')
        for name in ('Open','High','Low','Close','Volume'):
            c = cols.get(name.lower())
            if c and c != name:
                out.rename(columns={c: name}, inplace=True)
        out = out.dropna(subset=['Date','Close']).sort_values('Date').reset_index(drop=True)
        return out
    except Exception:
        return None

def _load_daily_df_adjusted(ticker: str) -> pd.DataFrame | None:
    """Alias for _load_daily_df. Parquet files are already split-adjusted by
    the data pipeline, so no additional adjustment is needed here."""
    return _load_daily_df(ticker)

# ============ end Scans: robust helpers ============
@st.cache_data(show_spinner=False)
def _load_parquet_daily(ticker: str) -> pd.DataFrame | None:
    try:
        return _load_polygon_daily_for_ticker(
            data_root="",
            ticker=ticker,
            reports_dir=None,
            technicals_script=None,
            auto_generate_report=False,
            excel_override=None,
            excel_path_override=None,
            allow_yahoo_fallback=False,
        )
    except Exception:
        return None

def _gap_events(df: pd.DataFrame, threshold_pct: float) -> pd.DataFrame:
    g = df.sort_values('Date').copy()
    g['PrevClose'] = g['Close'].shift(1)
    g['Gap_%'] = (g['Open'] / g['PrevClose'] - 1.0) * 100.0
    ev = g[_np.sign(threshold_pct) * g['Gap_%'] >= abs(threshold_pct)].copy()
    ev['Next_Close'] = g['Close'].shift(-1)
    ev['Next_Overnight_%'] = (ev['Next_Close'] / ev['Close'] - 1.0) * 100.0
    return ev.dropna(subset=['PrevClose'])

def _bbands(close: pd.Series, period: int = 20, mult: float = 2.0):
    mid = close.rolling(window=period, min_periods=period).mean()
    std = close.rolling(window=period, min_periods=period).std(ddof=0)
    upper = mid + mult * std
    lower = mid - mult * std
    return lower, mid, upper

def _bb_squeeze_breakouts(
    df: pd.DataFrame,
    period: int = 20,
    mult: float = 2.0,
    width_pct_threshold: float = 20.0,
    width_lookback: int = 126,
) -> pd.DataFrame:
    """Rows where BB width is in low percentile AND price breaks a band.
    Aligns arrays to filtered index and guards NaNs to prevent length errors.
    """
    g = df.sort_values('Date').copy()
    lb, mb, ub = _bbands(g['Close'], period, mult)
    valid = ub.notna() & lb.notna()
    width = (ub - lb) / mb.replace(0, _np.nan)
    def pct_rank(x: pd.Series) -> float:
        r = x.rank(pct=True)
        return r.iloc[-1]
    width_pct = width.rolling(width_lookback).apply(
        lambda x: _np.nan if _np.isnan(x).all() else pct_rank(pd.Series(x)), raw=False
    ) * 100.0
    breakout = (g['Close'] > ub) | (g['Close'] < lb)
    mask = breakout & valid & (width_pct <= width_pct_threshold)
    if not mask.any():
        return pd.DataFrame(columns=['Date','Open','High','Low','Close','Volume','BB_Width_%ile','Band_Break'])
    ev_idx = g.index[mask]
    ev = g.loc[ev_idx].copy()
    ev['BB_Width_%ile'] = width_pct.loc[ev_idx].values
    band_break_full = pd.Series(_np.where(g['Close'] > ub, 'Upper', 'Lower'), index=g.index)
    ev['Band_Break'] = band_break_full.loc[ev_idx].values
    return ev

def _breakout_52w(df: pd.DataFrame, lookback: int = 252) -> pd.DataFrame:
    g = df.sort_values('Date').copy()
    roll_max = g['Close'].rolling(lookback, min_periods=lookback).max()
    prev_roll_max = roll_max.shift(1)
    is_break = (g['Close'] >= roll_max) & (g['Close'].shift(1) < prev_roll_max)
    ev = g[is_break].copy()
    ev['52w_High'] = roll_max.loc[ev.index]
    return ev.dropna(subset=['52w_High'])

def _ma_cross_events(df: pd.DataFrame, short: int = 50, long: int = 200) -> pd.DataFrame:
    g = df.sort_values('Date').copy()
    g['SMA_S'] = g['Close'].rolling(short, min_periods=short).mean()
    g['SMA_L'] = g['Close'].rolling(long, min_periods=long).mean()
    prev_diff = (g['SMA_S'] - g['SMA_L']).shift(1)
    curr_diff = (g['SMA_S'] - g['SMA_L'])
    cross_up = (prev_diff <= 0) & (curr_diff > 0)
    cross_dn = (prev_diff >= 0) & (curr_diff < 0)
    ev = g[cross_up | cross_dn].copy()
    ev['CrossType'] = _np.where(cross_up.loc[ev.index], 'Golden', 'Death')
    ev['Next_Close'] = g['Close'].shift(-1)
    ev['Next_Overnight_%'] = (ev['Next_Close'] / ev['Close'] - 1.0) * 100.0
    ev['Close_5'] = g['Close'].shift(-5)
    ev['Next_5d_%'] = (ev['Close_5'] / ev['Close'] - 1.0) * 100.0
    return ev.dropna(subset=['SMA_S','SMA_L'])

def _inside_outside_events(df: pd.DataFrame) -> pd.DataFrame:
    g = df.sort_values('Date').copy()
    ph = g['High'].shift(1)
    pl = g['Low'].shift(1)
    inside = (g['High'] <= ph) & (g['Low'] >= pl)
    outside = (g['High'] >= ph) & (g['Low'] <= pl)
    ev = g[inside | outside].copy()
    ev['Type'] = _np.where(inside.loc[ev.index], 'Inside', 'Outside')
    ev['Next_Close'] = g['Close'].shift(-1)
    ev['Next_Overnight_%'] = (ev['Next_Close'] / ev['Close'] - 1.0) * 100.0
    return ev.dropna(subset=['High','Low'])

def _volume_spike_events(df: pd.DataFrame, mult: float = 2.0, window: int = 20) -> pd.DataFrame:
    g = df.sort_values('Date').copy()
    g['VolAvg'] = g['Volume'].rolling(window, min_periods=window).mean()
    g['SpikeRatio'] = g['Volume'] / g['VolAvg']
    ev = g[g['SpikeRatio'] >= float(mult)].copy()
    ev['Next_Close'] = g['Close'].shift(-1)
    ev['Next_Overnight_%'] = (ev['Next_Close'] / ev['Close'] - 1.0) * 100.0
    return ev.dropna(subset=['VolAvg'])

def _rs_52w_high(df_t: pd.DataFrame, df_spy: pd.DataFrame, lookback: int = 252) -> pd.DataFrame:
    t = df_t.sort_values('Date').copy()
    s = df_spy.sort_values('Date').copy()
    # align on dates
    t = t[['Date','Close']].rename(columns={'Close':'Close_T'})
    s = s[['Date','Close']].rename(columns={'Close':'Close_SPY'})
    m = pd.merge_asof(t.sort_values('Date'), s.sort_values('Date'), on='Date')
    m['RS'] = m['Close_T'] / m['Close_SPY']
    roll_max = m['RS'].rolling(lookback, min_periods=lookback).max()
    prev_roll_max = roll_max.shift(1)
    is_break = (m['RS'] >= roll_max) & (m['RS'].shift(1) < prev_roll_max)
    ev = m[is_break].copy()
    ev['RS_52wHigh'] = roll_max.loc[ev.index]
    ev['Next_Close'] = m['Close_T'].shift(-1)
    ev['Next_Overnight_%'] = (ev['Next_Close'] / ev['Close_T'] - 1.0) * 100.0
    return ev.dropna(subset=['RS_52wHigh'])

# --- Overnight backtest helpers ---
def _true_range(h: pd.Series, l: pd.Series, pc: pd.Series) -> pd.Series:
    a = (h - l).abs()
    b = (h - pc).abs()
    c = (l - pc).abs()
    return pd.concat([a, b, c], axis=1).max(axis=1)

def _atr_percent(df: pd.DataFrame, n: int = 14) -> pd.Series:
    g = df.sort_values('Date').copy()
    pc = g['Close'].shift(1)
    tr = _true_range(g['High'], g['Low'], pc)
    atr = tr.rolling(n, min_periods=n).mean()
    with pd.option_context('mode.use_inf_as_na', True):
        atr_pct = (atr / g['Close']) * 100.0
    return atr_pct.rename(f'ATR%({n})')

def _build_signal_mask(df: pd.DataFrame, strategy: str, thr: float, atr_n: int, atr_change_type: str | None = None) -> pd.Series:
    g = df.sort_values('Date').copy()
    if strategy.startswith('ATR change'):
        atrp = _atr_percent(g, int(atr_n))
        # Decide how to treat ATR% changes
        ch = atrp.diff()
        t = float(thr)
        ctype = (atr_change_type or 'Absolute').lower()
        if ctype.startswith('increase'):
            mask = (ch >= t)
        elif ctype.startswith('decrease'):
            mask = (ch <= -t)
        else:
            mask = (ch.abs() >= t)
        return mask.reindex(g.index).fillna(False)
    if strategy.startswith('Gap'):
        prevc = g['Close'].shift(1)
        gap = (g['Open'] / prevc - 1.0) * 100.0
        s = float(thr)
        return (_np.sign(s) * gap >= abs(s)).reindex(g.index).fillna(False)
    # Close-to-close change
    cc = (g['Close'] / g['Close'].shift(1) - 1.0) * 100.0
    s = float(thr)
    return (_np.sign(s) * cc >= abs(s)).reindex(g.index).fillna(False)

def _overnight_results(df: pd.DataFrame, mask: pd.Series, shares: int = 1, atr_n: int = 14) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Overnight/next-close backtest. Returns (event_df, eq_ov_series, eq_close_series)."""
    g = df.sort_values('Date').copy()
    close = pd.to_numeric(g['Close'], errors='coerce')
    open_ = pd.to_numeric(g['Open'], errors='coerce')
    g['_NextOpen']  = open_.shift(-1)
    g['_NextClose'] = close.shift(-1)
    g['_NextDate']  = g['Date'].shift(-1)
    g['Overnight_%'] = (g['_NextOpen']  / close - 1.0) * 100.0   # close -> next open
    g['NextClose_%'] = (g['_NextClose'] / close - 1.0) * 100.0   # close -> next close
    g['Pct_Chg_%']   = (close / close.shift(1) - 1.0) * 100.0    # same-day cc change
    try:
        atrp = _atr_percent(g, int(atr_n))
        g['ATR_Chg'] = atrp.diff()
    except Exception:
        g['ATR_Chg'] = np.nan
    sig = mask.reindex(g.index).fillna(False)
    ev  = g[sig].copy()
    ev['PL_Open']     = (ev['Overnight_%'] / 100.0 * pd.to_numeric(ev['Close'], errors='coerce')).round(4)
    ev['PL_Close']    = (ev['NextClose_%'] / 100.0 * pd.to_numeric(ev['Close'], errors='coerce')).round(4)
    ev['CumPL_Open']  = (ev['PL_Open']  * int(shares)).cumsum().round(0)
    ev['CumPL_Close'] = (ev['PL_Close'] * int(shares)).cumsum().round(0)
    # Equity series indexed by EXIT date
    try:
        exit_dates_ov    = g.loc[sig, '_NextDate'].values
        exit_dates_cl    = g.loc[sig, '_NextDate'].values
        pnl_ov  = (ev['PL_Open']  * int(shares)).values
        pnl_cl  = (ev['PL_Close'] * int(shares)).values
        eq_ov    = pd.Series(pnl_ov,  index=pd.DatetimeIndex(exit_dates_ov)).groupby(level=0).sum()
        eq_close = pd.Series(pnl_cl,  index=pd.DatetimeIndex(exit_dates_cl)).groupby(level=0).sum()
        all_dates = pd.DatetimeIndex(sorted(pd.to_datetime(g['Date'].values)))
        eq_ov    = eq_ov.reindex(all_dates,   fill_value=0.0).cumsum()
        eq_close = eq_close.reindex(all_dates, fill_value=0.0).cumsum()
    except Exception:
        eq_ov    = pd.Series(dtype=float)
        eq_close = pd.Series(dtype=float)
    eq_ov.name    = 'Equity_OV'
    eq_close.name = 'Equity_CLOSE'
    out_cols = ['Date','_NextDate','Pct_Chg_%','ATR_Chg','Close',
                'Overnight_%','NextClose_%','PL_Open','PL_Close','CumPL_Open','CumPL_Close']
    out_cols = [c for c in out_cols if c in ev.columns]
    ev = ev[out_cols].rename(columns={
        'Date':'Trade Date','_NextDate':'Exit Date','Pct_Chg_%':'Pct Change %',
        'ATR_Chg':'ATR Chg','Overnight_%':'Next Open %','NextClose_%':'Next Close %',
        'PL_Open':'P&L Open','PL_Close':'P&L Close',
        'CumPL_Open':'Cum P&L Open','CumPL_Close':'Cum P&L Close',
    })
    return ev, eq_ov, eq_close

# --- Earnings helpers ---
# Earnings fallbacks: local CSV → Polygon events/financials → Stocktwits → DoltHub → Benzinga → yfinance
def _get_earnings_dates_local_csv(ticker: str, limit: int = 500):
    import pandas as pd, os, glob
    dates = []
    for name in ["earnings_calendar.csv", "earnings calendar.csv"]:
        for path in glob.glob(os.path.join(os.path.dirname(__file__), "..", name)):
            try:
                df = pd.read_csv(path)
                sym_col = next((c for c in ["act_symbol","symbol","ticker"] if c in df.columns), None)
                if not sym_col or "date" not in df.columns:
                    continue
                sub = df[df[sym_col].str.upper() == ticker.upper()]
                dts = pd.to_datetime(sub["date"], errors="coerce").dropna().dt.normalize()
                dates.extend(dts.tolist())
            except Exception:
                pass
    uniq = sorted(set(dates))
    if len(uniq) > limit:
        uniq = uniq[-limit:]  # keep most recent dates
    return uniq

def _get_earnings_dates_stocktwits(ticker: str, years_back=10, years_fwd=2):
    import requests, pandas as pd, datetime as dt
    dates = []
    end = dt.date.today() + dt.timedelta(days=365*years_fwd)
    start = dt.date.today() - dt.timedelta(days=365*years_back)
    try:
        r = requests.get(
            "https://api.stocktwits.com/api/2/discover/earnings_calendar",
            params={"date_from": start.isoformat(), "date_to": end.isoformat()},
            timeout=10,
        )
        if not r.ok:
            return []
        js = r.json()
        for day in js.get("earnings_calendar", []):
            d = pd.to_datetime(day.get("date"), errors="coerce")
            if pd.isna(d):
                continue
            for sym in day.get("symbols", []):
                if sym.get("symbol","").upper() == ticker.upper():
                    dates.append(d.normalize())
        return sorted(set(dates))
    except Exception:
        return []

def _get_earnings_dates_dolthub(ticker: str):
    """Fetch earnings dates from DoltHub earnings_calendar and eps_estimate tables."""
    import requests, pandas as pd
    dates: list[pd.Timestamp] = []
    try:
        endpoints = [
            "https://www.dolthub.com/api/v1alpha1/post-no-preference/earnings/master?q=SELECT%20*%20FROM%20`earnings_calendar`",
            "https://www.dolthub.com/api/v1alpha1/post-no-preference/earnings/master?q=SELECT%20*%20FROM%20`eps_estimate`",
            "https://www.dolthub.com/api/v1alpha1/post-no-preference/earnings/master?q=SELECT%20*%20FROM%20`rank_score`",
        ]
        for url in endpoints:
            r = requests.get(url, timeout=10)
            if not r.ok:
                continue
            js = r.json()
            rows = js.get("rows") or []
            df = pd.DataFrame(rows)
            sym_col = "act_symbol" if "act_symbol" in df.columns else None
            if not sym_col:
                continue
            sub = df[df[sym_col].str.upper() == ticker.upper()]
            for col in ("date", "fiscal_date"):
                if col in sub.columns:
                    dts = pd.to_datetime(sub[col], errors="coerce").dropna().dt.normalize()
                    dates.extend(dts.tolist())
    except Exception:
        return []
    return sorted(set(dates))

@st.cache_data(show_spinner=False, ttl=3600)
def _fetch_dolthub_rank_score(ticker: str) -> pd.DataFrame:
    """Fetch DoltHub rank_score rows for a ticker."""
    import requests
    url = "https://www.dolthub.com/api/v1alpha1/post-no-preference/earnings/master"
    try:
        r = requests.get(url, params={"q": "SELECT * FROM `rank_score`"}, timeout=10)
        if not r.ok:
            return pd.DataFrame()
        js = r.json()
        rows = js.get("rows") or []
        df = pd.DataFrame(rows)
        sym_col = "act_symbol" if "act_symbol" in df.columns else None
        if not sym_col:
            return pd.DataFrame()
        sub = df[df[sym_col].str.upper() == ticker.upper()].copy()
        for c in ("fiscal_date", "date"):
            if c in sub.columns:
                sub[c] = pd.to_datetime(sub[c], errors="coerce")
        # Sort by fiscal_date if present
        if "fiscal_date" in sub.columns:
            sub = sub.sort_values("fiscal_date")
        return sub
    except Exception:
        return pd.DataFrame()

def _get_earnings_dates_polygon(ticker: str, limit: int = 80) -> list[pd.Timestamp]:
    """Attempt to fetch earnings/filing dates from Polygon (multiple endpoints tried)."""
    api_key = (os.getenv("POLYGON_API_KEY") or os.getenv("MASSIVE_API_KEY") or "").strip()
    if not api_key:
        return []
    try:
        import requests as _rq
    except Exception:
        return []
    base_url = os.getenv("POLYGON_API_BASE") or "https://api.polygon.io"
    endpoints = [
        f"{base_url}/v3/reference/financials",
        f"{base_url}/v2/reference/financials/{(ticker or '').strip().upper()}",
    ]
    dates: list[pd.Timestamp] = []
    for url in endpoints:
        try:
            params = {"limit": limit, "order": "desc", "apiKey": api_key}
            if "v3/reference/financials" in url:
                params["ticker"] = (ticker or "").strip().upper()
            resp = _rq.get(url, params=params, timeout=10)
            if not resp.ok:
                continue
            js = resp.json()
            results = js.get("results") or js.get("financials") or []
            for r in results:
                fd = r.get("filing_date") or r.get("filed_date") or r.get("report_period")
                if fd:
                    ts = pd.to_datetime(fd, errors="coerce")
                    if pd.notna(ts):
                        dates.append(ts.normalize())
        except Exception:
            continue
    return sorted(set(dates))

def _get_polygon_events(ticker: str, types=None, limit: int = 200) -> list[pd.Timestamp]:
    """Generic Polygon/Massive events fetcher with type filter."""
    api_key = (os.getenv("POLYGON_API_KEY") or os.getenv("MASSIVE_API_KEY") or "").strip()
    if not api_key:
        return []
    try:
        import requests as _rq
    except Exception:
        return []
    base_url = os.getenv("MASSIVE_API_BASE") or os.getenv("POLYGON_API_BASE") or "https://api.massive.com"
    url = f"{base_url}/vX/reference/tickers/{(ticker or '').strip().upper()}/events"
    params = {
        "limit": limit,
        "order": "desc",
        "apiKey": api_key,
    }
    if types:
        try:
            params["types"] = ",".join(types)
        except Exception:
            params["types"] = types
    try:
        resp = _rq.get(url, params=params, timeout=10)
        if not resp.ok:
            return []
        js = resp.json()
        results = js.get("results") or js.get("data") or []
        dates: list[pd.Timestamp] = []
        for r in results:
            sd = r.get("start_date") or r.get("date") or r.get("fiscal_period_end_date")
            if sd:
                ts = pd.to_datetime(sd, errors="coerce")
                if pd.notna(ts):
                    dates.append(ts.normalize())
        return sorted(set(dates))
    except Exception:
        return []

def _get_earnings_dates_polygon_events(ticker: str, limit: int = 200) -> list[pd.Timestamp]:
    """Fetch earnings events from Polygon/Massive events endpoint."""
    return _get_polygon_events(ticker, types=["earnings_release","earnings_report"], limit=limit)

@st.cache_data(show_spinner=False, ttl=3600)
def _get_earnings_dates_benzinga(ticker: str, limit: int = 200) -> list[pd.Timestamp]:
    """Fallback earnings dates via Massive Benzinga endpoint."""
    api_key = (os.getenv("POLYGON_API_KEY") or os.getenv("MASSIVE_API_KEY") or "").strip()
    if not api_key or not ticker:
        return []
    try:
        import requests as _rq
    except Exception:
        return []
    base = os.getenv("MASSIVE_API_BASE") or os.getenv("POLYGON_API_BASE") or "https://api.massive.com"
    url = f"{base}/benzinga/v1/earnings"
    params = {
        "ticker": ticker.strip().upper(),
        "limit": limit,
        "sort": "last_updated.desc",
        "apiKey": api_key,
    }
    try:
        resp = _rq.get(url, params=params, timeout=10)
        if not resp.ok:
            return []
        js = resp.json()
        results = js.get("results") or []
        dates: list[pd.Timestamp] = []
        for r in results:
            dt = r.get("date")
            if dt:
                ts = pd.to_datetime(dt, errors="coerce")
                if pd.notna(ts):
                    dates.append(ts.normalize())
        return sorted(set(dates))
    except Exception:
        return []

def _get_earnings_dates_yf(ticker: str, limit: int = 80) -> list[pd.Timestamp]:
    """Fallback earnings dates via yfinance."""
    try:
        import yfinance as yf  # type: ignore
    except Exception:
        return []
    try:
        t = yf.Ticker((ticker or '').strip())
        dfed = t.get_earnings_dates(limit=limit)
        if dfed is None or getattr(dfed, 'empty', True):
            return []
        if isinstance(dfed.index, pd.DatetimeIndex):
            return [d.normalize() for d in dfed.index.to_pydatetime()]
        for name in ('Earnings Date','earningsDate','date'):
            if name in getattr(dfed, 'columns', []):
                s = pd.to_datetime(dfed[name], errors='coerce')
                return [d.normalize() for d in s.dropna().to_pydatetime()]
    except Exception:
        return []
    return []

def _get_earnings_dates_financials(ticker: str, limit: int = 80) -> list[pd.Timestamp]:
    return _get_earnings_dates_polygon(ticker, limit=limit)

def _get_earnings_dates(ticker: str, limit: int = 100) -> list[pd.Timestamp]:
    # local CSV -> polygon events -> polygon financials -> stocktwits -> dolthub -> benzinga -> yfinance
    dates: list[pd.Timestamp] = []
    try:
        dates = _get_earnings_dates_local_csv(ticker, limit=limit)
    except Exception:
        dates = []
    if dates:
        uniq = sorted(set(dates))
        if len(uniq) > limit:
            uniq = uniq[-limit:]  # keep most recent
        return uniq
    try:
        events = _get_polygon_events(ticker, types=["earnings","earnings_release","earnings_report"])
        if events:
            dates.extend(events)
    except Exception:
        pass
    try:
        dates.extend(_get_earnings_dates_financials(ticker))
    except Exception:
        pass
    try:
        dates.extend(_get_earnings_dates_stocktwits(ticker))
    except Exception:
        pass
    try:
        dates.extend(_get_earnings_dates_dolthub(ticker))
    except Exception:
        pass
    try:
        dates.extend(_get_earnings_dates_benzinga(ticker, limit=limit))  # may 403 without entitlement
    except Exception:
        pass
    try:
        dates.extend(_get_earnings_dates_yf(ticker, limit=limit))
    except Exception:
        pass
    uniq = sorted(set(dates))
    if len(uniq) > limit:
        uniq = uniq[-limit:]  # keep most recent earnings dates
    return uniq

# --- News helper (Polygon/Massive) ---
@st.cache_data(show_spinner=False, ttl=300)
def _fetch_news(ticker: str, limit: int = 10) -> list[dict]:
    api_key = (os.getenv("POLYGON_API_KEY") or os.getenv("MASSIVE_API_KEY") or "").strip()
    if not api_key or not ticker:
        return []
    base = os.getenv("MASSIVE_API_BASE") or os.getenv("POLYGON_API_BASE") or "https://api.polygon.io"
    url = f"{base}/v2/reference/news"
    params = {
        "ticker": ticker.strip().upper(),
        "limit": limit,
        "order": "desc",
        "sort": "published_utc",
        "apiKey": api_key,
    }
    try:
        import requests as _rq
        resp = _rq.get(url, params=params, timeout=10)
        if not resp.ok:
            return []
        js = resp.json()
        return js.get("results") or []
    except Exception:
        return []

# --- Massive/Polygon movers helper ---
@st.cache_data(show_spinner=False, ttl=120)
def _fetch_movers(direction: str = "gainers", limit: int = 50) -> tuple[pd.DataFrame, list[str]]:
    """Fetch top gainers/losers via Massive/Polygon snapshot API."""
    import requests as _rq
    api_key = (os.getenv("MASSIVE_API_KEY") or os.getenv("POLYGON_API_KEY") or "").strip()
    if not api_key:
        return pd.DataFrame(), ["no API key"]
    base_poly = os.getenv("POLYGON_API_BASE") or "https://api.polygon.io"
    base_mass = os.getenv("MASSIVE_API_BASE") or "https://api.massive.com"
    errs: list[str] = []

    def _rows_from_results(res_list):
        rows = []
        for r in res_list or []:
            if not isinstance(r, dict):
                continue
            # Polygon v2 gainers/losers nests data in sub-objects
            day   = r.get("day") or {}
            prev  = r.get("prevDay") or {}
            trade = r.get("lastTrade") or r.get("last") or {}
            last_price = (trade.get("p") or trade.get("price")
                          or day.get("c") or r.get("last") or r.get("price"))
            pct   = (r.get("todaysChangePerc") or r.get("todays_change_percent")
                     or r.get("change_percent") or r.get("percent_change"))
            chg   = (r.get("todaysChange") or r.get("todays_change")
                     or r.get("change"))
            vol   = (day.get("v") or r.get("volume") or r.get("vol"))
            pc    = (prev.get("c") or r.get("prevDay_close")
                     or r.get("prev_close") or r.get("previous_close"))
            hi    = day.get("h") or r.get("high")
            lo    = day.get("l") or r.get("low")
            op    = day.get("o") or r.get("open")
            vwap  = day.get("vw") or r.get("vwap")
            rows.append({
                "Ticker":     r.get("ticker") or r.get("symbol"),
                "Last":       last_price,
                "% Chg":      pct,
                "Chg":        chg,
                "Volume":     vol,
                "Prev Close": pc,
                "Open":       op,
                "High":       hi,
                "Low":        lo,
                "VWAP":       vwap,
            })
        return rows

    # Prefer Polygon v2 snapshot gainers/losers (more widely available)
    try:
        url = f"{base_poly}/v2/snapshot/locale/us/markets/stocks/{direction}"
        params = {"limit": int(limit), "apiKey": api_key}
        resp = _rq.get(url, params=params, timeout=10)
        if resp.ok:
            js = resp.json()
            rows = _rows_from_results(js.get("tickers") or js.get("results"))
            if rows:
                return pd.DataFrame(rows), errs
        else:
            errs.append(f"poly v2 HTTP {resp.status_code}")
    except Exception as e:
        errs.append(f"poly v2 error {e}")

    # Fallback to Massive v3 direction endpoint
    try:
        url = f"{base_mass}/v3/snapshot/stocks/direction/{direction}"
        params = {"limit": int(limit), "apiKey": api_key}
        resp = _rq.get(url, params=params, timeout=10)
        if resp.ok:
            js = resp.json()
            rows = _rows_from_results(js.get("results"))
            if rows:
                return pd.DataFrame(rows), errs
        else:
            errs.append(f"v3 direction HTTP {resp.status_code}")
    except Exception as e:
        errs.append(f"v3 error {e}")

    return pd.DataFrame(), errs

# --- Premarket helpers (local minute files) ---
def _massive_key() -> str:
    return (os.getenv("MASSIVE_API_KEY") or os.getenv("POLYGON_API_KEY") or "").strip()

def _massive_base() -> str:
    return (os.getenv("MASSIVE_API_BASE") or "https://api.massive.com").rstrip("/")

@st.cache_data(show_spinner=False, ttl=300)
def _massive_client_ready() -> bool:
    return bool(_massive_key()) and HAS_MASSIVE_SDK

def _massive_client():
    if not HAS_MASSIVE_SDK:
        return None
    key = _massive_key()
    if not key:
        return None
    try:
        return MassiveRESTClient(key)
    except Exception:
        return None

@st.cache_data(show_spinner=False, ttl=120)
def _massive_snapshot_all_tickers(limit: int = 1000) -> tuple[pd.DataFrame, list[str]]:
    key = _massive_key()
    if not key:
        return pd.DataFrame(), ["no MASSIVE_API_KEY"]
    errs: list[str] = []
    # Prefer SDK if available
    if _massive_client_ready():
        try:
            client = _massive_client()
            if client is None:
                raise RuntimeError("Massive SDK client init failed")
            items = client.get_snapshot_all("stocks", "all")  # type: ignore
            rows = []
            for t in items[: int(limit)]:
                day = getattr(t, "day", None)
                vol = getattr(t, "volume", None)
                if day is not None:
                    vol = vol or getattr(day, "v", None) or getattr(day, "volume", None)
                rows.append({
                    "Ticker": getattr(t, "ticker", None),
                    "Last": getattr(t, "last", None) or getattr(t, "last_price", None),
                    "% Chg": getattr(t, "todays_change_perc", None) or getattr(t, "todaysChangePerc", None),
                    "Chg": getattr(t, "todays_change", None) or getattr(t, "todaysChange", None),
                    "Volume": vol,
                })
            return pd.DataFrame(rows), errs
        except Exception as e:
            errs.append(f"sdk error: {e}")
    # HTTP fallback
    base = _massive_base()
    url = f"{base}/v2/snapshot/locale/us/markets/stocks/tickers"
    params = {"limit": int(limit), "apiKey": key}
    try:
        resp = requests.get(url, params=params, timeout=20)
        if not resp.ok:
            return pd.DataFrame(), errs + [f"HTTP {resp.status_code}"]
        js = resp.json() or {}
        items = js.get("tickers") or js.get("results") or []
        rows = []
        for r in items:
            day = r.get("day") or {}
            last = r.get("last") or r.get("lastTrade") or {}
            rows.append({
                "Ticker": r.get("ticker") or r.get("symbol"),
                "Last": r.get("last") or last.get("p") or day.get("c"),
                "% Chg": r.get("todays_change_percent") or r.get("todaysChangePerc") or r.get("todays_change_pct"),
                "Chg": r.get("todays_change") or r.get("todaysChange") or r.get("change"),
                "Volume": r.get("volume") or day.get("v") or day.get("volume"),
            })
        return pd.DataFrame(rows), errs
    except Exception as e:
        return pd.DataFrame(), errs + [str(e)]

@st.cache_data(show_spinner=False, ttl=60)
def _massive_premarket_movers(min_volume: float = 100000, limit: int = 50) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Premarket movers using Massive snapshot_all. Requires Massive SDK."""
    if not _massive_client_ready():
        return pd.DataFrame(), pd.DataFrame(), ["Massive SDK not available or API key missing"]
    try:
        client = _massive_client()
        if client is None:
            return pd.DataFrame(), pd.DataFrame(), ["Massive SDK client init failed"]
        items = client.get_snapshot_all("stocks")  # type: ignore
        rows = []
        for t in items:
            try:
                day = getattr(t, "day", None)
                vol = getattr(day, "volume", None) if day is not None else None
                change = getattr(t, "todays_change_perc", None)
                min_bar = getattr(t, "min", None)
                price = getattr(min_bar, "close", None) if min_bar is not None else None
                if vol is None:
                    continue
                if float(vol) < float(min_volume):
                    continue
                rows.append({
                    "Ticker": getattr(t, "ticker", None),
                    "Price": price,
                    "Premarket %": round(float(change), 2) if change is not None else None,
                    "Volume": vol,
                })
            except Exception:
                continue
        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(), pd.DataFrame(), []
        df["Volume"] = pd.to_numeric(df.get("Volume"), errors="coerce")
        df["Premarket %"] = pd.to_numeric(df.get("Premarket %"), errors="coerce")
        gainers = df.sort_values("Premarket %", ascending=False, na_position="last").head(int(limit))
        losers = df.sort_values("Premarket %", ascending=True, na_position="last").head(int(limit))
        return gainers, losers, []
    except Exception as e:
        return pd.DataFrame(), pd.DataFrame(), [str(e)]

def _prev_close_from_daily(ticker: str, day: pd.Timestamp) -> float | None:
    df = _load_daily_df(ticker)
    if df is None or df.empty or "Date" not in df.columns:
        return None
    d = df.sort_values("Date")
    dts = pd.to_datetime(d["Date"], errors="coerce")
    mask = dts.dt.date < pd.Timestamp(day).date()
    if not mask.any():
        return None
    prev_row = d.loc[mask].iloc[-1]
    try:
        return float(prev_row["Close"])
    except Exception:
        return None

def _premarket_from_minutes(ticker: str, day: pd.Timestamp) -> tuple[float | None, float | None]:
    """Return (premarket_last_price, premarket_volume) from local minute data (04:00–09:30 ET)."""
    intr = _load_full_minute_df(ticker)
    if intr is None or intr.empty:
        return None, None
    tz = "America/New_York"
    day_et = pd.Timestamp(day).tz_localize(tz) if pd.Timestamp(day).tzinfo is None else pd.Timestamp(day).tz_convert(tz)
    start = day_et.normalize() + pd.Timedelta(hours=4)
    end = day_et.normalize() + pd.Timedelta(hours=9, minutes=30)
    seg = intr[(intr.index >= start) & (intr.index < end)]
    if seg.empty:
        return None, None
    pre_price = None
    if "Close" in seg.columns:
        try:
            pre_price = float(seg["Close"].iloc[-1])
        except Exception:
            pre_price = None
    pre_vol = None
    if "Volume" in seg.columns:
        try:
            pre_vol = float(pd.to_numeric(seg["Volume"], errors="coerce").sum())
        except Exception:
            pre_vol = None
    return pre_price, pre_vol

def _scan_premarket_local_minutes(
    scan_date: pd.Timestamp,
    tickers: list[str],
    min_vol: float = 0.0,
    max_requests: int = 200,
) -> pd.DataFrame:
    rows = []
    total = min(len(tickers), int(max_requests))
    prog = st.progress(0.0)
    for i, sym in enumerate(tickers, 1):
        if i > int(max_requests):
            break
        if i % 25 == 0:
            prog.progress(min(i / total, 1.0))
        pre_price, pre_vol = _premarket_from_minutes(sym, scan_date)
        if pre_price is None:
            continue
        if pre_vol is not None and float(min_vol) > 0 and float(pre_vol) < float(min_vol):
            continue
        prev_close = _prev_close_from_daily(sym, scan_date)
        if prev_close is None or prev_close == 0:
            continue
        pre_pct = (float(pre_price) / float(prev_close) - 1.0) * 100.0
        rows.append({
            "Ticker": sym,
            "Premarket %": pre_pct,
            "Premarket Price": pre_price,
            "Premarket Volume": pre_vol,
            "Prev Close": prev_close,
        })
    prog.progress(1.0)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)

@st.cache_data(show_spinner=False, ttl=300)
def _massive_snapshot_option_chain(ticker: str):
    if not ticker:
        return []
    if _massive_client_ready():
        try:
            client = _massive_client()
            if client is None:
                return []
            return client.get_snapshot_option_chain(ticker)  # type: ignore
        except Exception:
            return []
    # No HTTP fallback implemented for options chain yet
    return []

@st.cache_data(show_spinner=False, ttl=300)
def _massive_aggs_1d(ticker: str, start: str, end: str) -> pd.DataFrame:
    key = _massive_key()
    if not key or not ticker:
        return pd.DataFrame()
    base = _massive_base()
    url = f"{base}/v2/aggs/ticker/{ticker}/range/1/day/{start}/{end}"
    params = {"adjusted": "true", "sort": "asc", "limit": 50000, "apiKey": key}
    try:
        resp = requests.get(url, params=params, timeout=20)
        if not resp.ok:
            return pd.DataFrame()
        js = resp.json() or {}
        results = js.get("results") or []
        if not results:
            return pd.DataFrame()
        df = pd.DataFrame(results)
        if "t" in df.columns:
            df["Date"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.date
        df = df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume"})
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False, ttl=120)
def _massive_news_count(ticker: str, hours: int = 6) -> int:
    key = _massive_key()
    if not key or not ticker:
        return 0
    base = _massive_base()
    url = f"{base}/v2/reference/news"
    params = {"ticker": ticker, "limit": 50, "order": "desc", "sort": "published_utc", "apiKey": key}
    try:
        resp = requests.get(url, params=params, timeout=10)
        if not resp.ok:
            return 0
        js = resp.json() or {}
        items = js.get("results") or []
        if not items:
            return 0
        cutoff = pd.Timestamp.utcnow() - pd.Timedelta(hours=int(hours))
        cnt = 0
        for it in items:
            ts = it.get("published_utc")
            if not ts:
                continue
            try:
                if pd.to_datetime(ts, utc=True) >= cutoff:
                    cnt += 1
            except Exception:
                continue
        return cnt
    except Exception:
        return 0

@st.cache_data(show_spinner=False, ttl=60)
def _fetch_massive_snapshot_direction(direction: str = "gainers", limit: int = 200) -> tuple[pd.DataFrame, list[str]]:
    key = _massive_key()
    if not key:
        return pd.DataFrame(), ["no MASSIVE_API_KEY"]
    # Prefer Massive SDK if available
    if _massive_client_ready():
        try:
            client = _massive_client()
            if client is None:
                raise RuntimeError("Massive SDK client init failed")
            items = client.get_snapshot_all("stocks", direction)
            rows = []
            for t in items[: int(limit)]:
                day = getattr(t, "day", None)
                prev = getattr(t, "prev_day", None) or getattr(t, "prevDay", None)
                vol = getattr(t, "volume", None)
                if day is not None:
                    vol = vol or getattr(day, "v", None) or getattr(day, "volume", None)
                rows.append({
                    "Ticker": getattr(t, "ticker", None),
                    "Last": getattr(t, "last", None) or getattr(t, "last_price", None),
                    "% Chg": getattr(t, "todays_change_perc", None) or getattr(t, "todaysChangePerc", None),
                    "Chg": getattr(t, "todays_change", None) or getattr(t, "todaysChange", None),
                    "Volume": vol,
                    "Prev Close": getattr(prev, "c", None) if prev is not None else None,
                    "High": getattr(day, "h", None) if day is not None else None,
                    "Low": getattr(day, "l", None) if day is not None else None,
                    "Open": getattr(day, "o", None) if day is not None else None,
                })
            return pd.DataFrame(rows), []
        except Exception as e:
            return pd.DataFrame(), [f"sdk error: {e}"]

    base = _massive_base()
    url = f"{base}/v2/snapshot/locale/us/markets/stocks/{direction}"
    params = {"limit": int(limit), "apiKey": key}
    errs: list[str] = []
    try:
        resp = requests.get(url, params=params, timeout=15)
        if not resp.ok:
            return pd.DataFrame(), [f"HTTP {resp.status_code}"]
        js = resp.json() or {}
        items = js.get("tickers") or js.get("results") or []
        rows = []
        for r in items:
            day = r.get("day") or {}
            prev = r.get("prevDay") or {}
            last = r.get("last") or r.get("last_trade") or r.get("lastTrade") or {}
            vol = r.get("volume") or day.get("v") or day.get("volume")
            last_px = r.get("last") or last.get("p") or day.get("c")
            prev_close = r.get("prev_close") or prev.get("c")
            pct = r.get("todays_change_percent") or r.get("todaysChangePerc") or r.get("todays_change_pct")
            chg = r.get("todays_change") or r.get("todaysChange") or r.get("change")
            rows.append({
                "Ticker": r.get("ticker") or r.get("symbol"),
                "Last": last_px,
                "% Chg": pct,
                "Chg": chg,
                "Volume": vol,
                "Prev Close": prev_close,
                "High": day.get("h"),
                "Low": day.get("l"),
                "Open": day.get("o"),
            })
        return pd.DataFrame(rows), errs
    except Exception as e:
        return pd.DataFrame(), [str(e)]

@st.cache_data(show_spinner=False, ttl=3600)
def _fetch_massive_tickers(active: bool = True, market: str = "stocks", locale: str = "us", refresh_token: int = 0) -> list[str]:
    key = _massive_key()
    if not key:
        return []
    base = _massive_base()
    url = f"{base}/v3/reference/tickers"
    params = {
        "active": "true" if active else "false",
        "market": market,
        "locale": locale,
        "limit": 1000,
        "order": "asc",
        "sort": "ticker",
        "apiKey": key,
    }
    tickers: list[str] = []
    try:
        while url:
            resp = requests.get(url, params=params, timeout=20)
            if not resp.ok:
                break
            js = resp.json() or {}
            for r in js.get("results") or []:
                t = (r.get("ticker") or "").strip().upper()
                if t:
                    tickers.append(t)
            url = js.get("next_url")
            params = {"apiKey": key} if url else {}
    except Exception:
        return tickers
    return tickers

def _massive_open_close(ticker: str, date_str: str, adjusted: bool = True) -> tuple[dict | None, int | None, str | None]:
    key = _massive_key()
    if not key or not ticker or not date_str:
        return None, None, None
    base = _massive_base()
    url = f"{base}/v1/open-close/{ticker}/{date_str}"
    params = {"adjusted": "true" if adjusted else "false", "apiKey": key}
    try:
        resp = requests.get(url, params=params, timeout=15)
        if not resp.ok:
            return None, resp.status_code, resp.text[:200] if resp.text else None
        return (resp.json() or None), resp.status_code, None
    except Exception:
        return None, None, None

def _prev_trading_date(d: pd.Timestamp) -> pd.Timestamp:
    dt = pd.Timestamp(d).normalize()
    dt -= pd.Timedelta(days=1)
    while dt.weekday() >= 5:
        dt -= pd.Timedelta(days=1)
    return dt

def _scan_premarket_massive(
    scan_date: pd.Timestamp,
    tickers: list[str],
    max_requests: int = 100,
    delay_ms: int = 0,
    max_retries: int = 0,
) -> tuple[pd.DataFrame, dict]:
    if not tickers:
        return pd.DataFrame(), {"total": 0, "ok": 0, "no_premarket": 0, "no_close": 0, "errors": 0}
    date_str = pd.Timestamp(scan_date).strftime("%Y-%m-%d")
    rows = []
    stats = {"total": 0, "ok": 0, "no_premarket": 0, "no_close": 0, "errors": 0, "fallback_prev_close": 0, "status_counts": {}}
    samples: list[dict] = []
    prog = st.progress(0.0)
    total = min(len(tickers), int(max_requests))
    for i, sym in enumerate(tickers, 1):
        if i > int(max_requests):
            break
        if i % 50 == 0:
            prog.progress(min(i / total, 1.0))
        stats["total"] += 1
        js, status, err_text = _massive_open_close(sym, date_str)
        # Retry on 429 with backoff
        retry = 0
        while (js is None) and (status == 429) and (retry < int(max_retries)):
            try:
                time.sleep(0.5 * (2 ** retry))
            except Exception:
                pass
            js, status, err_text = _massive_open_close(sym, date_str)
            retry += 1
        if not js:
            stats["errors"] += 1
            if status is not None:
                stats["status_counts"][str(status)] = stats["status_counts"].get(str(status), 0) + 1
            if len(samples) < 10:
                samples.append({"ticker": sym, "status": status, "error": err_text, "premarket": None, "close": None})
            if delay_ms and delay_ms > 0:
                try:
                    time.sleep(float(delay_ms) / 1000.0)
                except Exception:
                    pass
            continue
        pre = js.get("preMarket")
        # Always base premarket % on previous trading day's close
        prev_dt = _prev_trading_date(pd.Timestamp(scan_date))
        js_prev, _, _ = _massive_open_close(sym, prev_dt.strftime("%Y-%m-%d"))
        close = js_prev.get("close") if js_prev else None
        if pre is None:
            stats["no_premarket"] += 1
            if len(samples) < 10:
                samples.append({"ticker": sym, "status": status, "premarket": pre, "close": close})
            continue
        if close is None or close == 0:
            stats["no_close"] += 1
            if len(samples) < 10:
                samples.append({"ticker": sym, "status": status, "premarket": pre, "close": close})
            continue
        try:
            pre_f = float(pre)
            close_f = float(close)
        except Exception:
            stats["errors"] += 1
            continue
        pre_pct = (pre_f / close_f - 1.0) * 100.0
        rows.append({
            "Ticker": sym,
            "Date": date_str,
            "Premarket %": pre_pct,
            "Premarket": pre_f,
            "Prev Close": close_f,
            "After Hours": js.get("afterHours"),
            "Open": js.get("open"),
            "High": js.get("high"),
            "Low": js.get("low"),
            "Volume": js.get("volume"),
        })
        stats["ok"] += 1
        if delay_ms and delay_ms > 0:
            try:
                time.sleep(float(delay_ms) / 1000.0)
            except Exception:
                pass
    prog.progress(1.0)
    if not rows:
        stats["samples"] = samples
        return pd.DataFrame(), stats
    df = pd.DataFrame(rows)
    df = df.sort_values("Premarket %", ascending=False)
    stats["samples"] = samples
    return df, stats

def _minute_dir() -> Path | None:
    base = os.environ.get("PER_TICKER_MINUTE_DIR") or ""
    if not base:
        return None
    p = Path(base)
    return p if p.exists() else None

def _list_minute_tickers(limit: int | None = None) -> list[str]:
    p = _minute_dir()
    if p is None:
        return []
    tickers: list[str] = []
    # Direct files
    for fp in sorted(p.glob("*.parquet")):
        tickers.append(fp.stem.upper())
        if limit and len(tickers) >= limit:
            return tickers
    # Subfolders
    for sub in sorted([d for d in p.iterdir() if d.is_dir()]):
        tickers.append(sub.name.upper())
        if limit and len(tickers) >= limit:
            return tickers
    return tickers

def _premarket_stats_for_day(intr_all: pd.DataFrame, day: pd.Timestamp) -> dict:
    tz = "America/New_York"
    day_et = pd.Timestamp(day).tz_localize(tz) if pd.Timestamp(day).tzinfo is None else pd.Timestamp(day).tz_convert(tz)
    day_start = day_et.normalize()
    pre_start = day_start + pd.Timedelta(hours=4)
    open_time = day_start + pd.Timedelta(hours=9, minutes=30)
    pre_seg = intr_all[(intr_all.index >= pre_start) & (intr_all.index < open_time)]
    if pre_seg.empty:
        return {"Date": day_start.tz_localize(None), "Premarket %": np.nan, "Premarket Vol": np.nan, "Prev Close": np.nan, "Premarket Last": np.nan}
    prev_seg = intr_all[intr_all.index < day_start]
    prev_close = float(prev_seg["Close"].iloc[-1]) if (not prev_seg.empty and "Close" in prev_seg.columns) else np.nan
    pre_last = float(pre_seg["Close"].iloc[-1]) if "Close" in pre_seg.columns else np.nan
    pre_vol = float(pre_seg["Volume"].sum()) if "Volume" in pre_seg.columns else np.nan
    pre_pct = (pre_last / prev_close - 1.0) * 100.0 if (prev_close and prev_close == prev_close and pre_last == pre_last) else np.nan
    return {
        "Date": day_start.tz_localize(None),
        "Premarket %": pre_pct,
        "Premarket Vol": pre_vol,
        "Prev Close": prev_close,
        "Premarket Last": pre_last,
    }

@st.cache_data(show_spinner=False, ttl=300)
def _scan_premarket_local(scan_date: pd.Timestamp, limit: int = 30, max_syms: int = 200) -> pd.DataFrame:
    tickers = _list_minute_tickers(limit=max_syms)
    rows = []
    for sym in tickers:
        intr_all = _load_full_minute_df(sym)
        if intr_all is None or intr_all.empty:
            continue
        try:
            stats = _premarket_stats_for_day(intr_all, scan_date)
            if stats and stats.get("Premarket %") == stats.get("Premarket %"):
                rows.append({"Ticker": sym, **stats})
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.sort_values("Premarket %", ascending=False)
    return df

def _premarket_history_for_ticker(ticker: str, lookback_days: int = 120) -> pd.DataFrame:
    intr_all = _load_full_minute_df(ticker)
    if intr_all is None or intr_all.empty:
        return pd.DataFrame()
    tz = "America/New_York"
    idx = intr_all.index.tz_convert(tz) if intr_all.index.tz is not None else intr_all.index.tz_localize("UTC").tz_convert(tz)
    intr_all = intr_all.copy()
    intr_all.index = idx
    unique_days = pd.to_datetime(pd.Series(intr_all.index.normalize().unique())).sort_values()
    if unique_days.empty:
        return pd.DataFrame()
    cutoff = unique_days.max() - pd.Timedelta(days=int(lookback_days))
    days = [d for d in unique_days if d >= cutoff]
    rows = []
    for d in days:
        try:
            rows.append(_premarket_stats_for_day(intr_all, d))
        except Exception:
            continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).sort_values("Date", ascending=False).set_index("Date")
    return df

def _show_intraday_chart(ticker: str, day: pd.Timestamp):
    intr_all = _load_full_minute_df(ticker)
    if intr_all is None or intr_all.empty:
        st.info("No minute data found for chart.")
        return
    tz = "America/New_York"
    day_et = pd.Timestamp(day).tz_localize(tz) if pd.Timestamp(day).tzinfo is None else pd.Timestamp(day).tz_convert(tz)
    day_start = day_et.normalize()
    day_end = day_start + pd.Timedelta(days=1)
    seg = intr_all[(intr_all.index >= day_start) & (intr_all.index < day_end)]
    if seg.empty:
        st.info("No minute bars for that day.")
        return
    try:
        import plotly.graph_objects as _go
        fig = _go.Figure()
        if {"Open","High","Low","Close"}.issubset(set(seg.columns)):
            fig.add_trace(_go.Candlestick(
                x=seg.index,
                open=seg["Open"],
                high=seg["High"],
                low=seg["Low"],
                close=seg["Close"],
                name="OHLC",
                increasing_line_color="#2ecc40",
                decreasing_line_color="#ff4d4f",
                showlegend=False,
            ))
        elif "Close" in seg.columns:
            fig.add_trace(_go.Scatter(x=seg.index, y=seg["Close"], mode="lines", name="Close"))
        fig.update_layout(title=f"{ticker} intraday {day_start.date()}", xaxis_rangeslider_visible=False, height=500, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        if "Close" in seg.columns:
            st.line_chart(seg["Close"], height=300)

def _earnings_prior_dayof_scan(ticker: str, df: pd.DataFrame) -> pd.DataFrame:
    """Compute prior/day-of stats around earnings.
    Returns rows for each earnings event with:
      - EarningsDate, TradeDate (the trading day used for the event)
      - Prior_% (Close[ED-1]/Close[ED-2] - 1)
      - Gap_% (Open[ED]/Close[ED-1] - 1)
      - Intraday_% (Close[ED]/Open[ED] - 1)
      - Day_% (Close[ED]/Close[ED-1] - 1)
      - NextDay_% (Close[ED+1]/Close[ED] - 1)
    """
    g = df.sort_values('Date').reset_index(drop=True).copy()
    if 'Date' not in g.columns or 'Close' not in g.columns:
        return pd.DataFrame()
    # Normalize types
    g['Date'] = pd.to_datetime(g['Date'], errors='coerce')
    for c in ('Open','High','Low','Close'):
        if c in g.columns:
            g[c] = pd.to_numeric(g[c], errors='coerce')

    e_dates = _get_earnings_dates(ticker, limit=60)
    if not e_dates:
        return pd.DataFrame(columns=['EarningsDate','TradeDate','Prior_%','Gap_%','Intraday_%','Day_%','NextDay_%'])

    out = []
    dts = g['Date']
    for ed in e_dates:
        # find first trading day on/after earnings calendar date
        mask = dts.dt.date >= ed
        if not mask.any():
            continue
        i = int(mask.idxmax())
        # prior and next indices
        i_prev = i - 1
        i_prev2 = i - 2
        i_next = i + 1
        # require previous two days for prior% and day-of calc
        if i_prev2 < 0 or i_prev < 0:
            continue
        try:
            prior = (g.at[i_prev, 'Close'] / g.at[i_prev2, 'Close'] - 1.0) * 100.0 if pd.notna(g.at[i_prev,'Close']) and pd.notna(g.at[i_prev2,'Close']) else _np.nan
            gap = _np.nan
            intr = _np.nan
            day = _np.nan
            if 'Open' in g.columns and pd.notna(g.at[i,'Open']) and pd.notna(g.at[i_prev,'Close']):
                gap = (g.at[i, 'Open'] / g.at[i_prev, 'Close'] - 1.0) * 100.0
            if 'Open' in g.columns and pd.notna(g.at[i,'Open']) and pd.notna(g.at[i,'Close']):
                intr = (g.at[i, 'Close'] / g.at[i, 'Open'] - 1.0) * 100.0
            if pd.notna(g.at[i,'Close']) and pd.notna(g.at[i_prev,'Close']):
                day = (g.at[i, 'Close'] / g.at[i_prev, 'Close'] - 1.0) * 100.0
            nextd = _np.nan
            if i_next < len(g) and pd.notna(g.at[i_next,'Close']) and pd.notna(g.at[i,'Close']):
                nextd = (g.at[i_next, 'Close'] / g.at[i, 'Close'] - 1.0) * 100.0
            out.append({
                'EarningsDate': pd.to_datetime(ed),
                'TradeDate': g.at[i,'Date'],
                'Prior_%': prior,
                'Gap_%': gap,
                'Intraday_%': intr,
                'Day_%': day,
                'NextDay_%': nextd,
            })
        except Exception:
            continue
    res = pd.DataFrame(out)
    return res.sort_values('TradeDate').reset_index(drop=True)
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date, datetime, timedelta, timezone
import math
try:
    import yfinance as yf
    HAS_YFINANCE = True
except Exception:
    yf = None  # type: ignore
    HAS_YFINANCE = False
import time
import os
import json
from typing import Optional
import os as _os_for_parquet_default
from pathlib import Path as _P_DEFAULT

# Force a sane default for per-ticker Parquet directory so the app "just works".
# Prefer an existing directory WITH parquet files, in this priority:
#  1) Next to app.py: per_ticker_daily, per_ticker_daily_ohlcv
#  2) Visual Code under Documents
#  3) Standard Documents
try:
    _app_dir = _P_DEFAULT(__file__).resolve().parent
    _home = _P_DEFAULT.home()
    _cands = [
        _app_dir / 'per_ticker_daily',
        _app_dir / 'per_ticker_daily_ohlcv',
        _home / 'Documents' / 'Visual Code' / 'Polygon Data' / 'per_ticker_daily',
        _home / 'Documents' / 'Visual Code' / 'Polygon Data' / 'per_ticker_daily_ohlcv',
        _home / 'Documents' / 'Polygon Data' / 'per_ticker_daily',
        _home / 'Documents' / 'Polygon Data' / 'per_ticker_daily_ohlcv',
    ]
    _pick = None
    for c in _cands:
        try:
            if c.is_dir() and any(c.glob('*.parquet')):
                _pick = str(c)
                break
        except Exception:
            continue
    # If not found yet, recursively search common roots for any per_ticker_daily* with parquet
    if _pick is None:
        _roots = [
            _home / 'Documents',
            _home / 'Documents' / 'Visual Code',
        ]
        for r in _roots:
            try:
                if not r.exists():
                    continue
                for d in r.rglob('per_ticker_daily*'):
                    try:
                        if d.is_dir() and any(d.glob('*.parquet')):
                            _pick = str(d)
                            break
                    except Exception:
                        continue
                if _pick:
                    break
            except Exception:
                continue
    if _pick:
        # Unconditionally set so we always use a working directory
        _os_for_parquet_default.environ['PER_TICKER_PARQUET_DIR'] = _pick
except Exception:
    pass
try:
    import pandas_ta as pta  # for candlestick pattern detection
    HAS_PANDAS_TA = True
except Exception:
    pta = None
    HAS_PANDAS_TA = False

st.set_page_config(page_title="CODEX TA", layout="wide")

# --- Data dir banner (hidden unless TA_DEBUG_UI=1) ---
if os.getenv('TA_DEBUG_UI','') == '1':
    try:
        _base = os.environ.get('PER_TICKER_PARQUET_DIR') or ''
        _cnt_txt = "0"
        if _base and Path(_base).exists():
            try:
                # Fast capped count to avoid scanning very large folders on startup
                cap = 5000
                n = 0
                for _ in Path(_base).glob('*.parquet'):
                    n += 1
                    if n >= cap:
                        _cnt_txt = f"{cap}+"
                        break
                else:
                    _cnt_txt = str(n)
            except Exception:
                _cnt_txt = "?"
        st.caption("Parquet dir: " + (_base or "(unset)") + " | Files: " + _cnt_txt)
        # Show which .env was loaded (to diagnose missing keys)
        try:
            _env_here = Path(__file__).with_name('.env')
            st.caption(f".env (app folder): {_env_here} | exists: {_env_here.exists()}")
        except Exception:
            pass
    except Exception:
        pass

# ---------------- Theme / Appearance ----------------
def apply_theme(choice: str) -> str:
    """
    Return plotly template and inject CSS.
    Ensures all white-background surfaces (inputs, expanders, tables, buttons)
    show black text, even in Dark theme.
    """
    css = """
    <style>
      /* ---- Global white-surface readability (sidebar + main) ---- */
      input, select, textarea { color:#000 !important; background:#fff !important; }
      div[role="combobox"] { color:#000 !important; background:#fff !important; border-radius:6px; }
      div[role="combobox"] * { color:#000 !important; background:#fff !important; }
      div[role="listbox"] * { color:#000 !important; background:#fff !important; }

      /* Sidebar specificity */
      [data-testid="stSidebar"] input,
      [data-testid="stSidebar"] select,
      [data-testid="stSidebar"] textarea { color:#000 !important; background:#fff !important; }
      [data-testid="stSidebar"] div[role="combobox"],
      [data-testid="stSidebar"] div[role="combobox"] * { color:#000 !important; background:#fff !important; }

      /* Number input buttons (+/-) */
      [data-testid="stNumberInput"] button { background:#fff !important; color:#000 !important; border:1px solid #aaa !important; }
      [data-testid="stNumberInput"] svg { fill:#000 !important; color:#000 !important; }

      /* Expanders */
      [data-testid="stExpander"] details,
      [data-testid="stExpander"] summary { background:#fff !important; color:#000 !important; border-radius:6px; }
      [data-testid="stExpander"] * { color:#000 !important; }

      /* Dataframes/tables: black on white to remain readable in Dark */
      .stDataFrame, .stTable { background:#fff !important; color:#000 !important; }
      .stDataFrame [data-testid="stVerticalBlock"], .stTable [data-testid="stVerticalBlock"] { background:#fff !important; color:#000 !important; }
      .stDataFrame th, .stTable th { background:#fff !important; color:#000 !important; }

      /* Tabs header text */
      .stTabs [role="tab"] { color: inherit !important; }

      /* Buttons: white background, black text */
      .stButton > button {
        background: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #aaa !important;
      }
      [data-testid="stSidebar"] .stButton > button {
        background: #ffffff !important;
        color: #000000 !important;
        border: 1px solid #aaa !important;
      }
      .stButton > button:hover,
      .stButton > button:active {
        background: #f2f2f2 !important;
        color: #000000 !important;
        border: 1px solid #888 !important;
      }
      .stButton > button:disabled {
        background: #f7f7f7 !important;
        color: #6a6a6a !important;
        border: 1px solid #d0d0d0 !important;
      }

      /* Ensure TradingView embed uses full width */
      .tradingview-widget-container, .tradingview-widget-container__widget {
        width: 100% !important;
      }
      /* Professional tab styling */
      .stTabs [data-baseweb="tab-list"] {
          gap: 8px;
          background-color: transparent;
          border-bottom: 2px solid #e0e0e0;
          padding-bottom: 0;
      }
      .stTabs [data-baseweb="tab"] {
          height: 40px;
          padding: 0 20px;
          border-radius: 6px 6px 0 0;
          font-weight: 500;
          font-size: 14px;
          background-color: #f5f5f5;
          border: 1px solid #e0e0e0;
          border-bottom: none;
      }
      .stTabs [aria-selected="true"] {
          background-color: #ffffff !important;
          border-top: 2px solid #1f77b4 !important;
          color: #1f77b4 !important;
      }
      /* Cleaner sidebar */
      [data-testid="stSidebar"] .stMarkdown h3 {
          font-size: 13px;
          font-weight: 700;
          text-transform: uppercase;
          letter-spacing: 0.05em;
          color: #555;
          margin-top: 16px;
          margin-bottom: 4px;
          padding-bottom: 4px;
          border-bottom: 1px solid #ddd;
      }
      [data-testid="stSidebar"] {
          background: #fafafa;
      }
      /* Navigation radio as pill buttons */
      div[data-testid="stHorizontalBlock"] .stRadio > div {
          flex-wrap: wrap;
          gap: 4px;
      }
      div[data-testid="stHorizontalBlock"] .stRadio label {
          background: #f0f2f6;
          border-radius: 20px;
          padding: 4px 14px;
          font-size: 13px;
          cursor: pointer;
          border: 1px solid transparent;
      }
      div[data-testid="stHorizontalBlock"] .stRadio label:hover {
          border-color: #1f77b4;
          color: #1f77b4;
      }
    /* ══════════════════════════════════════════════════════════════════
   TWO-LEVEL NAVIGATION
   Category bar  →  .st-key-_nav_cat
   Sub-nav row   →  [class*="st-key-_sub_"]
   ══════════════════════════════════════════════════════════════════ */

/* Hide the auto-generated "category" / "page" label text */
.st-key-_nav_cat > label:first-child,
[class*="st-key-_sub_"] > label:first-child { display: none !important; }

/* ── Category bar: underline-tab style ────────────────────────── */
.st-key-_nav_cat [data-testid="stRadio"] > div {
    gap: 0 !important;
    border-bottom: 2px solid #e0e0e0;
    padding-bottom: 0;
    margin-bottom: 2px;
    flex-wrap: nowrap !important;
}
.st-key-_nav_cat [data-testid="stRadio"] label {
    background: transparent !important;
    border: none !important;
    border-bottom: 3px solid transparent !important;
    border-radius: 0 !important;
    padding: 10px 24px 9px !important;
    font-size: 14px !important;
    font-weight: 600 !important;
    color: #555 !important;
    cursor: pointer;
    margin-bottom: -2px;
    transition: color 0.15s, border-color 0.15s;
}
.st-key-_nav_cat [data-testid="stRadio"] label:hover {
    color: #1f77b4 !important;
    background: rgba(31,119,180,0.04) !important;
}
.st-key-_nav_cat [data-testid="stRadio"] label:has(input:checked) {
    color: #1f77b4 !important;
    border-bottom-color: #1f77b4 !important;
    background: transparent !important;
}
/* Hide the radio circle dot in category bar */
.st-key-_nav_cat [data-testid="stRadio"] [data-testid="stWidgetLabel"],
.st-key-_nav_cat [data-testid="stRadio"] div[data-baseweb="radio"] > div:first-child {
    display: none !important;
}

/* ── Sub-nav: filled-pill style ───────────────────────────────── */
[class*="st-key-_sub_"] [data-testid="stRadio"] > div {
    gap: 6px !important;
    padding: 6px 0 10px !important;
    flex-wrap: wrap;
}
[class*="st-key-_sub_"] [data-testid="stRadio"] label {
    background: #f0f2f6 !important;
    border: 1px solid #d4d8e0 !important;
    border-radius: 20px !important;
    padding: 4px 18px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    color: #444 !important;
    cursor: pointer;
    transition: all 0.12s;
}
[class*="st-key-_sub_"] [data-testid="stRadio"] label:hover {
    border-color: #1f77b4 !important;
    color: #1f77b4 !important;
    background: rgba(31,119,180,0.06) !important;
}
[class*="st-key-_sub_"] [data-testid="stRadio"] label:has(input:checked) {
    background: #1f77b4 !important;
    color: #fff !important;
    border-color: #1f77b4 !important;
    font-weight: 600 !important;
}
/* Hide radio dot in sub-nav pills too */
[class*="st-key-_sub_"] [data-testid="stRadio"] div[data-baseweb="radio"] > div:first-child {
    display: none !important;
}
      </style>
    """
    st.markdown(css, unsafe_allow_html=True)

    # Theme background + font color rules
    if choice == "System":
        st.markdown("""
        <style>
        @media (prefers-color-scheme: dark) {
          html, body, [data-testid="stAppViewContainer"] { background:#0e1117; color:#e6e6e6; }
          [data-testid="stSidebar"] { background:#161a22; }
          [data-testid="stSidebar"] * { color:#e6e6e6; }
        }
        @media (prefers-color-scheme: light) {
          html, body, [data-testid="stAppViewContainer"] { background:#fff; color:#111; }
          [data-testid="stSidebar"] { background:#f6f6f6; }
          [data-testid="stSidebar"] * { color:#111; }
        }
        </style>
        """, unsafe_allow_html=True)
        return "plotly_dark"
    elif choice == "Dark":
        st.markdown("""
        <style>
        /* ===================================================================
           BLOOMBERG TERMINAL DARK THEME  -  app18.py / Streamlit
           Adapted from arb_dashboard/assets/style.css (read-only reference)
           =================================================================== */

        /* -- Global base --------------------------------------------------- */
        html, body,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        section.main > div {
            background: #0d1117 !important;
            color: #c9d1d9 !important;
            font-family: Consolas, "SF Mono", "Fira Code", monospace !important;
            -webkit-font-smoothing: antialiased;
        }

        /* -- Tighter page container ---------------------------------------- */
        .block-container {
            padding-top: 0.75rem !important;
            padding-bottom: 1rem !important;
        }

        /* -- Kill Streamlit grey overrides ---------------------------------- */
        p, span, div, li, label, td, th, h1, h2, h3, h4, h5, h6,
        .stMarkdown, .stMarkdown p, .stMarkdown span,
        [data-testid="stMarkdownContainer"],
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] span,
        [class*="css-"] { color: #c9d1d9 !important; }

        /* -- Section headings: Bloomberg blue uppercase --------------------- */
        h1, [data-testid="stMarkdownContainer"] h1 {
            color: #58a6ff !important;
            font-family: Consolas, "SF Mono", monospace !important;
            font-size: 15px !important; font-weight: 700 !important;
            text-transform: uppercase !important; letter-spacing: 0.5px !important;
            border-bottom: 1px solid #30363d !important;
            padding-bottom: 6px !important; margin-bottom: 10px !important;
        }
        h2, [data-testid="stMarkdownContainer"] h2 {
            color: #58a6ff !important;
            font-family: Consolas, "SF Mono", monospace !important;
            font-size: 13px !important; font-weight: 700 !important;
            text-transform: uppercase !important; letter-spacing: 0.5px !important;
            border-bottom: 1px solid #30363d !important;
            padding-bottom: 5px !important; margin-bottom: 8px !important;
        }
        h3, [data-testid="stMarkdownContainer"] h3 {
            color: #58a6ff !important;
            font-family: Consolas, "SF Mono", monospace !important;
            font-size: 12px !important; font-weight: 700 !important;
            text-transform: uppercase !important; letter-spacing: 0.5px !important;
            border-bottom: 1px solid #21262d !important;
            padding-bottom: 4px !important; margin-bottom: 6px !important;
        }
        h4, [data-testid="stMarkdownContainer"] h4 {
            color: #79c0ff !important;
            font-family: Consolas, "SF Mono", monospace !important;
            font-size: 12px !important; font-weight: 600 !important;
            text-transform: uppercase !important; letter-spacing: 0.3px !important;
            margin-bottom: 4px !important;
        }

        /* -- Sidebar ------------------------------------------------------- */
        [data-testid="stSidebar"] {
            background: #161b22 !important;
            border-right: 1px solid #30363d !important;
        }
        [data-testid="stSidebar"],
        [data-testid="stSidebar"] * { color: #c9d1d9 !important; }
        [data-testid="stSidebar"] .stMarkdown h3 {
            color: #58a6ff !important; border-bottom-color: #30363d !important;
            font-size: 10px !important; text-transform: uppercase !important;
            letter-spacing: 0.06em !important;
            font-family: Consolas, monospace !important;
            padding-bottom: 3px !important; margin-top: 12px !important;
        }
        [data-testid="stSidebar"] [data-testid="stExpander"] details,
        [data-testid="stSidebar"] [data-testid="stExpander"] summary {
            background: #1c2128 !important; border: 1px solid #30363d !important;
            color: #c9d1d9 !important;
        }

        /* -- Inputs / selects ---------------------------------------------- */
        input, select, textarea, [data-testid="stTextInput"] input {
            background: #161b22 !important; color: #e6edf3 !important;
            border-color: #30363d !important;
            font-family: Consolas, "SF Mono", monospace !important;
            font-size: 12px !important;
        }
        div[role="combobox"], div[role="combobox"] * {
            background: #161b22 !important; color: #e6edf3 !important;
        }
        div[role="listbox"], div[role="listbox"] * {
            background: #1c2128 !important; color: #e6edf3 !important;
        }
        [data-testid="stNumberInput"] button {
            background: #1c2128 !important; color: #e6edf3 !important;
            border-color: #30363d !important;
        }
        [data-testid="stNumberInput"] svg { fill: #e6edf3 !important; }

        /* -- Widget labels: dim uppercase small ---------------------------- */
        [data-testid="stWidgetLabel"],
        [data-testid="stWidgetLabel"] p,
        [data-testid="stWidgetLabel"] span {
            color: #8b949e !important;
            font-size: 10px !important; text-transform: uppercase !important;
            letter-spacing: 0.5px !important;
            font-family: Consolas, "SF Mono", monospace !important;
            font-weight: 600 !important;
        }

        /* -- Checkboxes + radio -------------------------------------------- */
        [data-testid="stCheckbox"] label, [data-testid="stCheckbox"] span,
        [data-testid="stRadio"] label, [data-testid="stRadio"] span {
            color: #c9d1d9 !important;
        }

        /* -- Selectbox / slider labels -------------------------------------- */
        [data-testid="stSelectbox"] label,
        [data-testid="stSlider"] label { color: #8b949e !important; }
        [data-testid="stSlider"] [data-testid="stTickBarMin"],
        [data-testid="stSlider"] [data-testid="stTickBarMax"] { color: #484f58 !important; }

        /* -- Metric tiles: terminal stat blocks ----------------------------- */
        [data-testid="stMetric"] {
            background: #161b22 !important;
            border: 1px solid #30363d !important;
            border-radius: 6px !important;
            padding: 8px 12px !important;
        }
        [data-testid="stMetric"] label,
        [data-testid="stMetricLabel"],
        [data-testid="stMetricLabel"] p {
            color: #8b949e !important;
            font-size: 9px !important; text-transform: uppercase !important;
            letter-spacing: 0.5px !important;
            font-family: Consolas, "SF Mono", monospace !important;
            font-weight: 600 !important;
        }
        [data-testid="stMetricValue"],
        [data-testid="stMetricValue"] div {
            color: #e6edf3 !important;
            font-size: 1.3rem !important; font-weight: 700 !important;
            font-family: Consolas, "SF Mono", monospace !important;
            letter-spacing: -0.5px !important;
            font-variant-numeric: tabular-nums !important;
        }
        [data-testid="stMetricDelta"],
        [data-testid="stMetricDelta"] div {
            font-weight: 600 !important;
            font-family: Consolas, "SF Mono", monospace !important;
            font-size: 11px !important;
        }

        /* -- Expanders: card style ----------------------------------------- */
        [data-testid="stExpander"] details,
        [data-testid="stExpander"] summary {
            background: #161b22 !important; color: #c9d1d9 !important;
            border: 1px solid #30363d !important; border-radius: 6px !important;
        }
        [data-testid="stExpander"] summary {
            font-size: 11px !important; font-weight: 600 !important;
            text-transform: uppercase !important; letter-spacing: 0.3px !important;
            color: #58a6ff !important;
            font-family: Consolas, "SF Mono", monospace !important;
        }
        [data-testid="stExpander"] * { color: #c9d1d9 !important; }

        /* -- Buttons ------------------------------------------------------- */
        .stButton > button {
            background: #21262d !important; color: #c9d1d9 !important;
            border: 1px solid #30363d !important;
            font-family: Consolas, "SF Mono", monospace !important;
            font-size: 12px !important; font-weight: 500 !important;
            padding: 4px 16px !important; border-radius: 4px !important;
            transition: all 0.15s ease !important;
        }
        .stButton > button:hover {
            background: #2d333b !important; color: #e6edf3 !important;
            border-color: #58a6ff !important;
        }
        .stButton > button[kind="primary"] {
            background: #1f6feb !important; color: #fff !important;
            border-color: #1f6feb !important;
        }
        .stButton > button[kind="primary"]:hover {
            background: #388bfd !important; border-color: #388bfd !important;
        }

        /* -- Download button ----------------------------------------------- */
        [data-testid="stDownloadButton"] button {
            background: #21262d !important; color: #c9d1d9 !important;
            border: 1px solid #30363d !important;
            font-family: Consolas, "SF Mono", monospace !important;
            font-size: 12px !important;
        }
        [data-testid="stDownloadButton"] button:hover {
            background: #2d333b !important; border-color: #58a6ff !important;
            color: #e6edf3 !important;
        }

        /* -- Dataframe container wrapper ------------------------------------ */
        .stDataFrame {
            background: #0d1117 !important;
            border: 1px solid #30363d !important;
            border-radius: 6px !important;
            overflow: hidden !important;
        }
        .stDataFrame > div { background: #0d1117 !important; }
        .stDataFrame td { color: #c9d1d9 !important; }
        .stDataFrame th {
            background: #161b22 !important; color: #58a6ff !important;
            font-family: Consolas, "SF Mono", monospace !important;
            font-size: 11px !important; font-weight: 600 !important;
            text-transform: uppercase !important; letter-spacing: 0.3px !important;
            padding: 5px 8px !important; border-bottom: 2px solid #30363d !important;
            white-space: nowrap !important;
        }

        /* -- Webkit scrollbars: dark terminal -------------------------------- */
        ::-webkit-scrollbar { width: 8px; height: 8px; }
        ::-webkit-scrollbar-track { background: #0d1117; }
        ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #484f58; }

        /* -- stTabs --------------------------------------------------------- */
        .stTabs [data-baseweb="tab-list"] {
            background-color: transparent !important;
            border-bottom: 2px solid #30363d !important;
            gap: 0 !important;
        }
        .stTabs [data-baseweb="tab"] {
            background-color: #161b22 !important;
            border: 1px solid #30363d !important; border-bottom: none !important;
            border-radius: 6px 6px 0 0 !important;
            color: #8b949e !important;
            font-family: Consolas, "SF Mono", monospace !important;
            font-size: 12px !important; font-weight: 500 !important;
            padding: 6px 18px !important;
        }
        .stTabs [data-baseweb="tab"]:hover {
            background-color: #1c2128 !important; color: #c9d1d9 !important;
        }
        .stTabs [aria-selected="true"] {
            background-color: #0d1117 !important;
            border-top: 2px solid #58a6ff !important;
            color: #58a6ff !important; font-weight: 700 !important;
        }

        /* -- Nav category bar ---------------------------------------------- */
        .st-key-_nav_cat [data-testid="stRadio"] > div {
            border-bottom: 2px solid #30363d !important;
            gap: 0 !important; flex-wrap: nowrap !important;
        }
        .st-key-_nav_cat [data-testid="stRadio"] label {
            background: transparent !important;
            color: #8b949e !important;
            border: none !important;
            border-bottom: 2px solid transparent !important;
            font-weight: 600 !important; font-size: 13px !important;
            font-family: Consolas, "SF Mono", monospace !important;
            padding: 8px 20px 7px !important; letter-spacing: 0.3px !important;
            transition: color 0.15s, border-color 0.15s !important;
            margin-bottom: -2px !important;
        }
        .st-key-_nav_cat [data-testid="stRadio"] label:hover {
            color: #79c0ff !important;
            background: rgba(88,166,255,0.06) !important;
        }
        .st-key-_nav_cat [data-testid="stRadio"] label:has(input:checked) {
            color: #58a6ff !important;
            border-bottom: 2px solid #58a6ff !important;
            background: transparent !important; font-weight: 700 !important;
        }
        .st-key-_nav_cat [data-testid="stRadio"] [data-testid="stWidgetLabel"],
        .st-key-_nav_cat [data-testid="stRadio"] div[data-baseweb="radio"] > div:first-child {
            display: none !important;
        }

        /* -- Sub-nav pills ------------------------------------------------- */
        [class*="st-key-_sub_"] [data-testid="stRadio"] > div {
            gap: 4px !important; padding: 5px 0 8px !important;
        }
        [class*="st-key-_sub_"] [data-testid="stRadio"] label {
            background: #161b22 !important; border: 1px solid #30363d !important;
            border-radius: 4px !important; padding: 3px 14px !important;
            font-size: 11px !important; font-weight: 500 !important;
            font-family: Consolas, "SF Mono", monospace !important;
            color: #8b949e !important; transition: all 0.12s !important;
            letter-spacing: 0.2px !important;
        }
        [class*="st-key-_sub_"] [data-testid="stRadio"] label:hover {
            border-color: #58a6ff !important; color: #58a6ff !important;
            background: rgba(88,166,255,0.08) !important;
        }
        [class*="st-key-_sub_"] [data-testid="stRadio"] label:has(input:checked) {
            background: #0d419d !important; color: #ffffff !important;
            border-color: #58a6ff !important; font-weight: 700 !important;
        }
        [class*="st-key-_sub_"] [data-testid="stRadio"] div[data-baseweb="radio"] > div:first-child {
            display: none !important;
        }

        /* -- Dividers ------------------------------------------------------ */
        hr, [data-testid="stDivider"] { border-color: #30363d !important; }

        /* -- Captions / small text ----------------------------------------- */
        .stCaption, small,
        [data-testid="stCaptionContainer"],
        [data-testid="stCaptionContainer"] p {
            color: #8b949e !important;
            font-family: Consolas, "SF Mono", monospace !important;
            font-size: 11px !important;
        }

        /* -- Alert boxes --------------------------------------------------- */
        [data-testid="stAlert"] {
            background: #161b22 !important; border-color: #30363d !important;
            border-radius: 6px !important;
        }
        [data-testid="stAlert"] p,
        [data-testid="stAlert"] span {
            color: #c9d1d9 !important;
            font-family: Consolas, "SF Mono", monospace !important;
            font-size: 12px !important;
        }

        /* -- Progress bar -------------------------------------------------- */
        [data-testid="stProgress"] > div { background: #21262d !important; }
        [data-testid="stProgress"] > div > div { background: #1f6feb !important; }

        /* -- Markdown HTML tables ------------------------------------------ */
        [data-testid="stMarkdownContainer"] table {
            border-collapse: collapse !important; width: 100% !important;
            font-family: Consolas, "SF Mono", monospace !important;
            font-size: 12px !important; border: 1px solid #30363d !important;
        }
        [data-testid="stMarkdownContainer"] th {
            background: #161b22 !important; color: #58a6ff !important;
            font-size: 11px !important; font-weight: 600 !important;
            text-transform: uppercase !important; letter-spacing: 0.3px !important;
            padding: 5px 10px !important; border: 1px solid #30363d !important;
            white-space: nowrap !important;
        }
        [data-testid="stMarkdownContainer"] td {
            background: #0d1117 !important; color: #c9d1d9 !important;
            padding: 3px 10px !important; border: 1px solid #21262d !important;
            font-variant-numeric: tabular-nums !important;
        }
        [data-testid="stMarkdownContainer"] tr:nth-child(even) td {
            background: #0f1419 !important;
        }
        [data-testid="stMarkdownContainer"] tr:hover td {
            background: #1c2128 !important;
        }

        /* -- Code blocks --------------------------------------------------- */
        code, pre {
            background: #161b22 !important; color: #e6edf3 !important;
            border: 1px solid #30363d !important;
            font-family: Consolas, "SF Mono", monospace !important;
            font-size: 12px !important;
        }

        /* -- Columns gap --------------------------------------------------- */
        [data-testid="stHorizontalBlock"] { gap: 0.5rem !important; }

        /* -- Tooltip icon -------------------------------------------------- */
        [data-testid="stTooltipIcon"] { color: #484f58 !important; }
        </style>
        """, unsafe_allow_html=True)
        return "plotly_dark"
    else:
        st.markdown("""
        <style>
        html, body, [data-testid="stAppViewContainer"] { background:#fff; color:#111; }
        [data-testid="stSidebar"] { background:#f6f6f6; }
        [data-testid="stSidebar"] * { color:#111; }
        </style>
        """, unsafe_allow_html=True)
        return "plotly_white"


def style_axes(fig: go.Figure, dark: bool, rows: int, extra_rangebreaks: list | None = None, minimalist: bool = False, nticks: int | None = None):
    grid = ("#2a2a2a" if dark else "#eaeaea") if minimalist else ("#333333" if dark else "#cccccc")
    fig.update_layout(
        template="plotly_dark" if dark else "plotly_white",
        plot_bgcolor="#000000" if dark else "#ffffff",
        paper_bgcolor="#000000" if dark else "#ffffff",
        font=dict(color="#e6e6e6" if dark else "#111111"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(color="#e6e6e6" if dark else "#111111")
        )
    )
    # Hide weekend gaps on all x-axes for a continuous look
    for r in range(1, rows + 1):
        rb = [dict(bounds=["sat", "mon"])]
        if extra_rangebreaks:
            rb.extend(extra_rangebreaks)
        fig.update_xaxes(
            row=r,
            col=1,
            showgrid=True,
            gridcolor=grid,
            gridwidth=(0.5 if minimalist else 1),
            zerolinecolor=grid,
            tickfont_color="#e6e6e6" if dark else "#111111",
            rangebreaks=rb,
            showspikes=False,
            nticks=(nticks if nticks else None),
        )
        fig.update_yaxes(row=r, col=1, showgrid=True, gridcolor=grid, gridwidth=(0.5 if minimalist else 1), zerolinecolor=grid,
                         tickfont_color="#e6e6e6" if dark else "#111111",
                         nticks=(nticks if nticks else None))

# ---------------- Sidebar defaults ------------------------------------------------
# Sidebar controls (ticker/interval/dates) only render for Chart/TradingView/Options.
# Defaults here ensure variables exist for all code paths.
ticker = "AAPL"; interval = "1d"; intraday = False; period = None
start = date.today() - timedelta(days=365); end = date.today()

# Theme lives in session_state; apply globally before any tab content
template = apply_theme(st.session_state.get("_app_theme", "Dark"))

# Top navigation (replaces tabs)
# ── Two-level grouped navigation ─────────────────────────────────────────────
_NAV_GROUPS = {
    "📈 Charts":      ["Chart", "TradingView"],
    "🔍 Scans":       ["Scans", "Signal Scanner", "Scanners", "Overnight"],
    "📊 Market":      ["Options", "Premarket", "Movers", "Calendar"],
    "🤖 AI Research": ["GPT-5 Agent", "MS Analysis", "JPM Earnings", "GS Fundamental", "Sentiment"],
    "🎯 Trade Finder": ["Trade Finder", "AI Scanner", "Smart Money"],
    "🌐 Global": ["Index Breadth", "10-Day Screen", "ADR Parity", "Macro Drivers"],
    "🔮 Predictions": ["Polymarket"],
}

_NAV_CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".nav_cache")

# ── Nav state restore: URL query params → session_state ────────────────────
# Only page name stored in URL (no emoji) — avoids URL-encoding issues
_default_cat = "📈 Charts"
try:
    _qp_page = st.query_params.get("page", "")
    if not _qp_page:
        try:
            with open(_NAV_CACHE_FILE) as _ncf: _qp_page = _ncf.read().strip()
        except Exception: pass
    if "_nav_cat" not in st.session_state and _qp_page:
        for _k, _v in _NAV_GROUPS.items():
            if _qp_page in _v:
                _default_cat = _k  # local only — avoids widget/session_state conflict
                _ssk = f"_sub_{_k}"
                if _ssk not in st.session_state:
                    st.session_state[_ssk] = _qp_page
                break
except Exception:
    pass
# If already set by widget from prior render, honour it
if "_nav_cat" in st.session_state:
    _default_cat = st.session_state["_nav_cat"]
if _default_cat not in _NAV_GROUPS:
    _default_cat = "📈 Charts"

st.markdown("---")

_cat = st.radio(
    "category",
    list(_NAV_GROUPS.keys()),
    index=list(_NAV_GROUPS.keys()).index(_default_cat),
    horizontal=True,
    key="_nav_cat",
    label_visibility="collapsed",
)

# Sub-nav for the selected category
_sub_key = f"_sub_{_cat}"
_sub_opts = _NAV_GROUPS[_cat]
_sub_default = st.session_state.get(_sub_key, _sub_opts[0])
if _sub_default not in _sub_opts:
    _sub_default = _sub_opts[0]

nav = st.radio(
    "page",
    _sub_opts,
    index=_sub_opts.index(_sub_default),
    horizontal=True,
    key=_sub_key,
    label_visibility="collapsed",
)

try:
    st.session_state["nav"] = nav
except Exception:
    pass

# ── Write current page to URL (page name only, no emoji) ────────────────────────
try:
    with open(_NAV_CACHE_FILE, "w") as _ncf: _ncf.write(nav)
except Exception: pass
try:
    st.query_params["page"] = nav
except Exception:
    pass

# ── Sidebar: only for Chart / TradingView / Options ──────────────────────────
if nav in ('Chart', 'TradingView', 'Options'):
    with st.sidebar:
        st.header("Controls")
        ticker = st.text_input("Ticker", value="AAPL").strip().upper()
        interval = st.selectbox("Interval", ["1d", "1h", "30m", "15m", "5m", "1m"], index=0)
        intraday = interval in {"1h", "30m", "15m", "5m", "1m"}

        if intraday:
            period = st.selectbox("Period (intraday)", ["1d", "5d", "7d", "14d", "30d"],
                                  index=2 if interval == "1m" else 1)
            start = date.today() - timedelta(days=7)
            end = date.today()
        else:
            period = None
            c1, c2 = st.columns(2)
            with c1:
                start = st.date_input("Start", value=date.today() - timedelta(days=365))
            with c2:
                end = st.date_input("End", value=date.today())

# ── Hide sidebar completely on non-chart pages ───────────────────────────────
if nav not in ('Chart', 'TradingView', 'Options'):
    st.markdown(
        "<style>[data-testid='stSidebar']{display:none}</style>",
        unsafe_allow_html=True,
    )

# ── Theme toggle in nav bar (all pages) ──────────────────────────────────────
_tc1, _tc2 = st.columns([8, 1])
with _tc2:
    st.radio(
        "Theme",
        ["Dark", "Light", "System"],
        index=["Dark", "Light", "System"].index(
            st.session_state.get("_app_theme", "Dark")
        ),
        horizontal=True,
        key="_app_theme",
        label_visibility="collapsed",
    )
# Re-apply theme after widget renders (in case it changed)
template = apply_theme(st.session_state.get("_app_theme", "Dark"))

# Lightweight RSI helper placed before Scans so it is defined for visualizers
def rsi(series: pd.Series, length: int = 14) -> pd.Series:
    series = pd.to_numeric(series, errors='coerce')
    delta = series.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/float(length), adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/float(length), adjust=False).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0).rename(f"RSI({length})")

# ═══════════════════════════════════════════════════════════════════════════════
# TRADE FINDER — helpers
# ═══════════════════════════════════════════════════════════════════════════════

# ── Static universe lists ────────────────────────────────────────────────────
_TF_NDX100 = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","TSLA","AVGO","COST",
    "NFLX","AMD","ADBE","QCOM","PEP","CSCO","INTC","INTU","TXN","AMGN",
    "HON","AMAT","LRCX","MU","PANW","ADI","MRVL","REGN","KLAC","MDLZ",
    "GILD","ADP","SNPS","CDNS","ASML","CRWD","MELI","FTNT","ORLY","CTAS",
    "MNST","WDAY","PCAR","ABNB","TEAM","NXPI","PAYX","AEP","DXCM","ROST",
    "FAST","IDXX","ODFL","KDP","EA","GEHC","EXC","VRSK","CTSH","BKR",
    "ON","XEL","DLTR","ZS","ANSS","ILMN","CPRT","BIIB","PYPL",
    "LULU","SMCI","APP","PLTR","MSTR","SNDK","WDC",
]

_TF_SPX_LIQUID = [
    "AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA","JPM","V","XOM",
    "JNJ","UNH","MA","AVGO","HD","CVX","PG","LLY","ABBV","MRK","COST",
    "KO","BAC","PEP","WMT","AMD","MCD","TMO","CSCO","CRM","ABT","ACN",
    "NFLX","GE","DIS","ADBE","AMGN","TXN","PM","VZ","MS","GS","BA",
    "HON","INTC","IBM","BX","SPGI","CAT","UPS","NEE","LMT","RTX","T",
    "USB","PFE","MO","BMY","GILD","SBUX","SYK","ZTS","ISRG","ADP",
    "REGN","CVS","TGT","MDLZ","BSX","APH","SHW","ADI","PANW","CRWD",
    "SNOW","UBER","ABNB","MU","SNDK","WDC","AMAT","LRCX","KLAC","ASML",
    "MRVL","QCOM","ON","SMCI","APP","PLTR","MSTR","RBLX",
    "F","GM","XOM","CVX","COP","SLB","HAL","OXY",
    "GLD","SLV","USO","GDX","CPER","FCX","NUE","CLF",
    "SPY","QQQ","IWM","XLK","XLF","XLE","XLV","XLI",
    "JPM","BAC","WFC","C","GS","MS","BLK","BX",
    "AMZN","SHOP","SQ","PYPL","COIN","HOOD","SOFI",
]

_TF_MACRO_TICKERS = ["SPY","QQQ","IWM","TLT","GLD","SLV","USO","CPER","UUP","FXY","FXE"]
_TF_MACRO_LABELS  = {
    "SPY":"S&P 500","QQQ":"NDX 100","IWM":"Russell 2k",
    "TLT":"Bonds(TLT)","GLD":"Gold","SLV":"Silver","USO":"Crude Oil","CPER":"Copper",
}


# ── FinBERT sentiment helpers ─────────────────────────────────────────────
_HAS_FINBERT = False
try:
    import torch as _torch  # noqa: F401
    _HAS_FINBERT = True
except ImportError:
    pass

if _HAS_FINBERT:
    @st.cache_resource
    def _finbert_pipeline():
        """Load ProsusAI/finbert once per server lifetime."""
        from transformers import pipeline as _hf_pipeline
        return _hf_pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            device=-1,
            top_k=None,
        )

@st.cache_data(show_spinner=False, ttl=900)
def _tf_news_sentiment(ticker: str, limit: int = 10) -> dict:
    """Fetch Polygon headlines and score with FinBERT.

    Returns dict with keys:
        score   : float (-1 to +1, negative=bearish, positive=bullish)
        label   : str ("positive" / "negative" / "neutral")
        count   : int (number of headlines scored)
        headlines: list[dict] with title, sentiment, confidence
    """
    _neutral = {"score": 0.0, "label": "neutral", "count": 0, "headlines": []}
    if not _HAS_FINBERT:
        return _neutral
    articles = _fetch_news(ticker, limit=limit)
    if not articles:
        return _neutral
    titles = [a.get("title", "") for a in articles if a.get("title")]
    if not titles:
        return _neutral
    try:
        pipe = _finbert_pipeline()
        results = pipe(titles, batch_size=16, truncation=True, max_length=512)
    except Exception:
        return _neutral
    scored = []
    total = 0.0
    for title, res_list in zip(titles, results):
        # res_list is a list of dicts: [{"label": "positive", "score": 0.9}, ...]
        best = max(res_list, key=lambda x: x["score"])
        lbl = best["label"].lower()
        conf = best["score"]
        # Convert to signed score: positive=+1, negative=-1, neutral=0
        if lbl == "positive":
            s = conf
        elif lbl == "negative":
            s = -conf
        else:
            s = 0.0
        total += s
        scored.append({"title": title, "sentiment": lbl, "confidence": round(conf, 3)})
    avg = total / len(scored) if scored else 0.0
    overall = "positive" if avg > 0.15 else ("negative" if avg < -0.15 else "neutral")
    return {
        "score": round(avg, 4),
        "label": overall,
        "count": len(scored),
        "headlines": scored,
    }



def _scan_calendar_emails(lookback_days: int = 7,
                          keywords: list[str] | None = None) -> dict:
    """Scan Outlook Inbox for calendar-relevant emails and extract tickers.

    Parameters
    ----------
    lookback_days : int
        How many days back to scan.
    keywords : list[str] | None
        Subject-line keywords to match (case-insensitive).
        Defaults to ["TMT", "earnings", "catalyst", "calendar events",
                      "analyst day", "investor day", "eco data"].

    Returns
    -------
    dict with keys:
        tickers  : sorted list of extracted ticker symbols
        events   : list of dicts {subject, date, tickers, type} for display
        stats    : {emails_scanned, emails_matched, tickers_found}
    """
    import re as _re
    try:
        import pythoncom
        import win32com.client
    except ImportError:
        return {"tickers": [], "events": [], "stats": {}}

    if keywords is None:
        keywords = ["TMT", "earnings", "catalyst", "calendar events",
                    "analyst day", "investor day", "eco data", "macro",
                    "economic calendar"]
    _kw_lower = [k.lower() for k in keywords]

    tickers: set[str] = set()
    events: list[dict] = []
    emails_scanned = 0
    emails_matched = 0

    _tk_bbg = _re.compile(r"\b([A-Z]{1,5})\s+(?:US|UN|UW|UQ|UA)\b")
    _tk_paren = _re.compile(r"\(([A-Z]{1,5})\)")
    _tk_colon = _re.compile(r"(?:ticker|symbol|stock)[:\s]+([A-Z]{1,5})\b", _re.IGNORECASE)

    def _classify_subject(subj_lower):
        if any(w in subj_lower for w in ("earning", "eps", "report")):
            return "earnings"
        if any(w in subj_lower for w in ("catalyst", "analyst day", "investor day", "capital markets")):
            return "catalyst"
        if any(w in subj_lower for w in ("eco", "macro", "economic", "calendar event")):
            return "eco_data"
        if "tmt" in subj_lower:
            return "tmt"
        return "other"

    try:
        pythoncom.CoInitialize()
        outlook = win32com.client.Dispatch("Outlook.Application")
        ns = outlook.GetNamespace("MAPI")
        inbox = ns.GetDefaultFolder(6)

        cutoff = (datetime.now() - timedelta(days=lookback_days)).strftime(
            "%m/%d/%Y 12:00 AM")
        items = inbox.Items.Restrict(f"[ReceivedTime] >= '{cutoff}'")

        for msg in items:
            try:
                emails_scanned += 1
                subj = getattr(msg, "Subject", "") or ""
                subj_lower = subj.lower()
                if not any(kw in subj_lower for kw in _kw_lower):
                    continue
                emails_matched += 1

                body = getattr(msg, "Body", "") or ""
                msg_tickers = set()
                for m in _tk_bbg.finditer(body):
                    msg_tickers.add(m.group(1))
                for m in _tk_paren.finditer(body):
                    msg_tickers.add(m.group(1))
                for m in _tk_colon.finditer(body):
                    msg_tickers.add(m.group(1))

                recv = getattr(msg, "ReceivedTime", None)
                recv_str = str(recv)[:19] if recv else ""

                tickers |= msg_tickers
                if msg_tickers:
                    events.append({
                        "subject": subj[:80],
                        "date": recv_str,
                        "tickers": sorted(msg_tickers),
                        "type": _classify_subject(subj_lower),
                    })
            except Exception:
                continue
    except Exception:
        pass

    _noise = {"THE", "FOR", "AND", "NOT", "ARE", "BUT", "ALL", "HAS", "HAD",
              "HIS", "HER", "HIM", "ITS", "OUR", "OUT", "NEW", "NOW", "OLD",
              "ONE", "TWO", "USE", "WAY", "WHO", "MAY", "DAY", "GET", "GOT",
              "LET", "SAY", "SHE", "TOO", "BUY", "ANY", "FEW", "TMT", "USD",
              "EST", "PST", "CST", "MST", "CEO", "CFO", "COO", "IPO", "GDP",
              "YOY", "QOQ", "MOM", "DIV", "EPS", "REV", "FCF", "M&A", "NET",
              "SEC", "FED", "IMF", "ECB", "BOJ", "CPI", "PPI", "PMI", "ISM",
              "DOW", "SPX", "NDX", "RTY", "SOX", "DAX", "MESA", "GXA",
              "NASDAQ", "TICKER", "FREE", "NONE", "NOTE", "CALL", "PUT",
              "HIGH", "LOW", "OPEN", "LAST", "BEST", "RISK", "FUND", "CASH"}
    clean = sorted(tickers - _noise)
    return {
        "tickers": clean,
        "events": events,
        "stats": {"emails_scanned": emails_scanned,
                  "emails_matched": emails_matched,
                  "tickers_found": len(clean)},
    }


@st.cache_data(show_spinner=False, ttl=3600)
def _tf_load_beta_universe() -> list:
    '''Read BETA Universe.xlsx, return clean US equity tickers.'''
    import re as _re
    _path = "C:/Users/David Alcosser/Documents/Work/BETA Universe.xlsx"
    try:
        import openpyxl as _opxl
        wb = _opxl.load_workbook(_path, read_only=True, data_only=True)
        ws = wb.active
        seen, out = set(), []
        for row in ws.iter_rows(values_only=True):
            val = row[0]
            if not val or not isinstance(val, str): continue
            t = val.strip().upper()
            for suf in (" US", " NA"):
                if t.endswith(suf): t = t[:-len(suf)].strip(); break
            if _re.match(r"^[A-Z]{2,6}$", t) and t not in seen:
                seen.add(t); out.append(t)
        return out
    except Exception:
        return []

def _tf_poly_get(path: str, params: dict = None, timeout: int = 10) -> dict:
    """Thin Polygon REST wrapper — returns parsed JSON or {}."""
    try:
        import requests as _rq
    except ImportError:
        return {}
    api_key = (os.getenv("POLYGON_API_KEY") or os.getenv("MASSIVE_API_KEY") or "").strip()
    base    = os.getenv("POLYGON_API_BASE") or "https://api.polygon.io"
    if not api_key:
        return {}
    p = {"apiKey": api_key, **(params or {})}
    try:
        r = _rq.get(f"{base}{path}", params=p, timeout=timeout)
        return r.json() if r.ok else {}
    except Exception:
        return {}

@st.cache_data(show_spinner=False, ttl=120)
def _tf_snapshot_batch(tickers: tuple) -> dict:
    """Polygon snapshot for up to 250 tickers — {ticker: {price, chg_pct, volume, ...}}."""
    if not tickers:
        return {}
    chunk = ",".join(str(t) for t in tickers[:250])
    js = _tf_poly_get("/v2/snapshot/locale/us/markets/stocks/tickers",
                      {"tickers": chunk, "include_otc": "false"})
    out = {}
    for r in (js.get("tickers") or []):
        tkr   = r.get("ticker", "")
        day   = r.get("day") or {}
        prev  = r.get("prevDay") or {}
        out[tkr] = {
            "price":   day.get("c") or prev.get("c"),
            "chg_pct": r.get("todaysChangePerc") or 0.0,
            "volume":  day.get("v") or 0,
            "vwap":    day.get("vw"),
            "open":    day.get("o"),
            "high":    day.get("h"),
            "low":     day.get("l"),
            "prev_c":  prev.get("c"),
        }
    return out

@st.cache_data(show_spinner=False, ttl=600)
def _tf_vix() -> float | None:
    """Fetch latest VIX level from Polygon index aggregates."""
    today_str = pd.Timestamp.today().strftime("%Y-%m-%d")
    week_ago  = (pd.Timestamp.today() - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    js = _tf_poly_get(f"/v2/aggs/ticker/I:VIX/range/1/day/{week_ago}/{today_str}",
                      {"limit": 5, "sort": "desc"})
    results = js.get("results") or []
    return float(results[0]["c"]) if results else None

@st.cache_data(show_spinner=False, ttl=120)
def _tf_index_prices() -> dict:
    """Fetch latest levels for equity indices (I:SPX/NDX/RUT), rates (I:TNX/TYX), DXY via Polygon daily aggs."""
    today    = pd.Timestamp.today().strftime("%Y-%m-%d")
    week_ago = (pd.Timestamp.today() - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
    out = {}
    for ix in ("I:SPX", "I:NDX", "I:RUT", "I:TNX", "I:TYX", "I:DXY"):
        js   = _tf_poly_get(f"/v2/aggs/ticker/{ix}/range/1/day/{week_ago}/{today}",
                            {"limit": 5, "sort": "desc"})
        bars = js.get("results") or []
        if bars:
            c  = float(bars[0]["c"])
            pc = float(bars[1]["c"]) if len(bars) > 1 else float(bars[0]["o"])
            out[ix] = {"price": c, "chg_pct": round((c / pc - 1) * 100, 2) if pc else 0.0}
    return out

@st.cache_data(show_spinner=False, ttl=120)
def _tf_commodity_spot() -> dict:
    """Fetch spot/forex: gold ($/oz), silver ($/oz), WTI crude ($/bbl), copper ($/lb),
    USD/JPY, EUR/USD via Polygon forex/commodity aggs."""
    today    = pd.Timestamp.today().strftime("%Y-%m-%d")
    week_ago = (pd.Timestamp.today() - pd.Timedelta(days=14)).strftime("%Y-%m-%d")
    pairs = {
        "GOLD":   "C:XAUUSD",    # Gold spot $/oz
        "SILVER": "C:XAGUSD",    # Silver spot $/oz
        "OIL":    "C:WTICOUSD",  # WTI crude $/bbl
        "COPPER": "C:XCUUSD",    # Copper spot $/lb
        "USDJPY": "C:USDJPY",    # USD per JPY (e.g. 148.50)
        "EURUSD": "C:EURUSD",    # EUR/USD (e.g. 1.0850)
    }
    out = {}
    for key, ticker in pairs.items():
        js   = _tf_poly_get(f"/v2/aggs/ticker/{ticker}/range/1/day/{week_ago}/{today}",
                            {"limit": 5, "sort": "desc"})
        bars = js.get("results") or []
        if bars:
            c  = float(bars[0]["c"])
            pc = float(bars[1]["c"]) if len(bars) > 1 else float(bars[0]["o"])
            out[key] = {"price": c, "chg_pct": round((c / pc - 1) * 100, 2) if pc else 0.0}
    return out

# ── Polymarket helpers ────────────────────────────────────────────────────────

def _pm_get(path: str, params: dict = None, timeout: int = 12):
    """Thin Polymarket Gamma-API wrapper — returns parsed JSON or []."""
    try:
        import requests as _rq
    except ImportError:
        return []
    try:
        r = _rq.get(f"https://gamma-api.polymarket.com{path}",
                     params=params or {}, timeout=timeout)
        return r.json() if r.ok else []
    except Exception:
        return []

def _pm_clob_get(path: str, params: dict = None, timeout: int = 10):
    """Polymarket CLOB API wrapper — returns parsed JSON or {}."""
    try:
        import requests as _rq
    except ImportError:
        return {}
    try:
        r = _rq.get(f"https://clob.polymarket.com{path}",
                     params=params or {}, timeout=timeout)
        return r.json() if r.ok else {}
    except Exception:
        return {}

# ── Category keyword buckets for client-side filtering ──
_PM_CATEGORIES = {
    "Finance & Rates": ["fed", "rate", "interest", "inflation", "cpi", "gdp",
                        "recession", "treasury", "yield", "mortgage", "ecb",
                        "fomc", "monetary", "tariff", "trade war", "jobs",
                        "unemployment", "payroll", "housing"],
    "Equities & Indices": ["s&p", "spx", "sp500", "sp 500", "nasdaq", "dow",
                           "stock", "ipo", "etf", "market cap", "earnings",
                           "tesla", "apple", "nvidia", "amazon", "microsoft",
                           "spacex", "microstrategy"],
    "Crypto": ["bitcoin", "btc", "ethereum", "eth", "crypto", "solana",
               "altcoin", "defi", "nft", "stablecoin", "dogecoin", "xrp"],
    "Commodities & FX": ["gold", "silver", "oil", "crude", "copper", "natural gas",
                          "commodity", "forex", "dollar", "euro", "yen", "yuan"],
    "Geopolitics": ["war", "ceasefire", "iran", "ukraine", "russia", "china",
                    "taiwan", "nato", "strike", "military", "troops", "hormuz",
                    "sanction", "north korea", "regime"],
    "US Politics": ["trump", "biden", "republican", "democrat", "congress",
                    "senate", "president", "election", "nominee", "cabinet",
                    "impeach", "musk", "doge", "executive order"],
    "World Politics": ["macron", "starmer", "orban", "modi", "netanyahu",
                       "zelensky", "bolsonaro", "eu ", "uk ", "canada",
                       "germany", "france", "brazil", "india", "mexico"],
    "Sports": ["nba", "nfl", "mlb", "nhl", "soccer", "football", "cricket",
               "tennis", "ufc", "boxing", "f1", "formula", "premier league",
               "champions league", "world cup", "olympics", "ncaa", "college",
               "nuggets", "lakers", "celtics", "warriors", "playoffs",
               "march madness", "super bowl", "grand prix"],
    "Esports & Gaming": ["dota", "counter-strike", "cs2", "lol", "league of legends",
                          "valorant", "esports", "pgl", "esl"],
    "Soccer (Intl)": ["uel-", "ucl-", "col-", "spl-", "celta", "stuttgart",
                       "braga", "porto", "freiburg", "roma", "bologna",
                       "sporting", "fc ", "sc ", "genk", "lyon", "lille",
                       "villa", "forest", "mainz", "betis"],
    "Science & Tech": ["ai ", "ai model", "spacex", "launch", "mars",
                        "openai", "google", "anthropic", "gpt", "agi",
                        "quantum", "climate", "earthquake", "hurricane"],
    "Entertainment": ["oscar", "grammy", "emmy", "movie", "album", "spotify",
                       "youtube", "tiktok", "twitch", "celebrity", "kanye",
                       "taylor swift", "drake"],
}

@st.cache_data(show_spinner=False, ttl=180)
def _pm_fetch_markets(limit: int = 500, offset: int = 0) -> list:
    """Fetch active Polymarket markets (contract-level) sorted by 24h volume."""
    p = {"active": "true", "closed": "false",
         "order": "volume24hr", "ascending": "false",
         "limit": str(limit), "offset": str(offset)}
    data = _pm_get("/markets", p, timeout=15)
    if isinstance(data, list):
        return data
    return []

@st.cache_data(show_spinner=False, ttl=300)
def _pm_fetch_events(tag: str = "", query: str = "", limit: int = 100) -> list:
    """Fetch active events sorted by 24h volume with optional client-side search."""
    p = {"active": "true", "closed": "false",
         "order": "volume24hr", "ascending": "false",
         "limit": str(limit)}
    if tag:
        p["tag"] = tag
    data = _pm_get("/events", p)
    if query and isinstance(data, list):
        q_low = query.lower()
        data = [e for e in data if q_low in (e.get("title") or "").lower()
                or q_low in (e.get("slug") or "").lower()]
    if isinstance(data, list):
        return data
    return []

@st.cache_data(show_spinner=False, ttl=120)
def _pm_fetch_prices(token_ids: tuple) -> dict:
    """Fetch midpoint prices from CLOB for a tuple of condition-token IDs."""
    out = {}
    for tid in token_ids:
        js = _pm_clob_get("/midpoint", {"token_id": tid})
        if isinstance(js, dict) and "mid" in js:
            out[tid] = float(js["mid"])
    return out

@st.cache_data(show_spinner=False, ttl=300)
def _pm_fetch_event_by_slug(slug: str) -> dict:
    """Fetch a single event by its URL slug."""
    data = _pm_get(f"/events/slug/{slug}")
    if isinstance(data, dict):
        return data
    if isinstance(data, list) and data:
        return data[0]
    return {}

def _pm_watchlist_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "polymarket_watchlist.json")

def _pm_load_watchlist() -> list:
    import json as _js
    p = _pm_watchlist_path()
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as fh:
                return _js.load(fh)
        except Exception:
            pass
    return []

def _pm_save_watchlist(data: list):
    import json as _js
    with open(_pm_watchlist_path(), "w", encoding="utf-8") as fh:
        _js.dump(data, fh, indent=2)

def _pm_extract_best_outcome(event: dict) -> tuple:
    """Return (outcome_label, probability, token_id) for the leading outcome."""
    markets = event.get("markets") or []
    best_label, best_prob, best_tid = "", 0.0, ""
    for mkt in markets:
        tokens = mkt.get("clobTokenIds") or ""
        outcomes = mkt.get("outcomes") or ""
        if isinstance(tokens, str):
            try:
                import json as _js; tokens = _js.loads(tokens)
            except Exception:
                tokens = []
        if isinstance(outcomes, str):
            try:
                import json as _js; outcomes = _js.loads(outcomes)
            except Exception:
                outcomes = []
        prices = mkt.get("outcomePrices") or ""
        if isinstance(prices, str):
            try:
                import json as _js; prices = _js.loads(prices)
            except Exception:
                prices = []
        for i, oc in enumerate(outcomes):
            pr = float(prices[i]) if i < len(prices) else 0.0
            tid = tokens[i] if i < len(tokens) else ""
            if pr > best_prob:
                best_prob = pr
                best_label = oc
                best_tid = tid
    return best_label, best_prob, best_tid

def _pm_categorize(title: str, slug: str) -> str:
    """Assign a market to the best-matching category, or 'Other'."""
    combined = (title + " " + slug).lower()
    for cat, keywords in _PM_CATEGORIES.items():
        if any(kw in combined for kw in keywords):
            return cat
    return "Other"

def _pm_safe_float(val, default=0.0) -> float:
    try:
        return float(val) if val is not None else default
    except (TypeError, ValueError):
        return default

def _pm_market_to_row(m: dict) -> dict:
    """Convert a raw /markets dict to a display row."""
    price = _pm_safe_float(m.get("lastTradePrice"))
    v24 = _pm_safe_float(m.get("volume24hr"))
    v1w = _pm_safe_float(m.get("volume1wk"))
    liq = _pm_safe_float(m.get("liquidityNum") or m.get("liquidity"))
    chg_1w = _pm_safe_float(m.get("oneWeekPriceChange"))
    chg_1m = _pm_safe_float(m.get("oneMonthPriceChange"))
    spread = _pm_safe_float(m.get("spread"))
    title = (m.get("question") or m.get("groupItemTitle") or m.get("slug") or "")
    slug = m.get("slug") or ""
    return {
        "Contract": title[:90],
        "Price": price,
        "1W Chg": chg_1w,
        "1M Chg": chg_1m,
        "Vol 24h": v24,
        "Vol 1W": v1w,
        "Liquidity": liq,
        "Spread": spread,
        "Category": _pm_categorize(title, slug),
        "_slug": slug,
        "_id": m.get("id", ""),
    }

@st.cache_data(show_spinner=False, ttl=900)
def _tf_options_summary(ticker: str) -> dict:
    """ATM IV, P/C ratio, 25d skew from Polygon options chain (next 45 days)."""
    try:
        import requests as _rq
        from datetime import date, timedelta
    except ImportError:
        return {}
    api_key = (os.getenv("POLYGON_API_KEY") or os.getenv("MASSIVE_API_KEY") or "").strip()
    base    = os.getenv("POLYGON_API_BASE") or "https://api.polygon.io"
    if not api_key:
        return {}
    today  = date.today()
    exp_hi = (today + timedelta(days=45)).isoformat()
    try:
        r = _rq.get(f"{base}/v3/snapshot/options/{ticker.upper()}",
                    params={"apiKey": api_key,
                            "expiration_date.gte": today.isoformat(),
                            "expiration_date.lte": exp_hi,
                            "limit": 250},
                    timeout=12)
        results = r.json().get("results") or [] if r.ok else []
    except Exception:
        return {}
    if not results:
        return {}
    # Get spot from underlying_asset in first result, fallback to snapshot batch
    _ua0 = (results[0].get("underlying_asset") or {})
    spot  = _ua0.get("price")
    if not spot:
        snap  = _tf_snapshot_batch((ticker.upper(),))
        spot  = (snap.get(ticker.upper()) or {}).get("price")
    if not spot:
        return {}
    spot = float(spot)
    calls, puts = [], []
    all_contracts = []                      # for unusual-activity / premium scan
    for c in results:
        det    = c.get("details") or {}
        iv     = c.get("implied_volatility")
        oi     = c.get("open_interest") or 0
        strike = float(det.get("strike_price") or 0)
        ctype  = (det.get("contract_type") or "").lower()
        day    = c.get("day") or c.get("session") or {}
        lq     = c.get("last_quote") or {}
        lt     = c.get("last_trade") or {}
        grk    = c.get("greeks") or {}
        vol    = int(day.get("volume") or 0)
        # mid price: prefer last_quote midpoint, then day vwap, then last_trade price
        mid    = float(lq.get("midpoint") or day.get("vwap") or lt.get("price") or 0)
        delta  = grk.get("delta")
        expiry = det.get("expiration_date") or ""
        if not iv or not strike:
            continue
        iv = float(iv)
        entry = {
            "strike": strike, "iv": iv, "oi": int(oi), "vol": vol,
            "mid": mid, "type": ctype, "delta": delta, "expiry": expiry,
        }
        all_contracts.append(entry)
        if ctype == "call":
            calls.append(entry)
        elif ctype == "put":
            puts.append(entry)

    def _atm(chain):
        near = sorted(chain, key=lambda x: abs(x["strike"] - spot))[:6]
        vals = [x["iv"] for x in near if x["iv"] > 0]
        return float(np.mean(vals)) if vals else None

    def _skew_otm_puts(chain):
        # 25d proxy: strikes 15-30% below spot
        lo, hi = spot * 0.70, spot * 0.85
        otm = [x["iv"] for x in chain if lo <= x["strike"] <= hi and x["iv"] > 0]
        return float(np.mean(otm)) if otm else None

    atm_iv_c  = _atm(calls)
    atm_iv_p  = _atm(puts)
    atm_iv    = float(np.mean([v for v in [atm_iv_c, atm_iv_p] if v])) if any([atm_iv_c, atm_iv_p]) else None
    pc_oi     = sum(p["oi"] for p in puts)
    cc_oi     = sum(c["oi"] for c in calls)
    pc_ratio  = round(pc_oi / cc_oi, 2) if cc_oi > 0 else None
    put_skew  = _skew_otm_puts(puts)
    skew_25d  = round(put_skew / atm_iv, 3) if (put_skew and atm_iv and atm_iv > 0) else None

    # ── new: unusual activity, net premium, max pain, vol/OI ──────────
    # unusual activity: contracts where day_volume > 2× open_interest AND vol > 500
    unusual = [e for e in all_contracts if e["oi"] > 0 and e["vol"] > 500
               and e["vol"] / e["oi"] > 2.0]
    unusual_sorted = sorted(unusual, key=lambda x: x["vol"], reverse=True)[:3]
    unusual_top = []
    for e in unusual_sorted:
        _side = "C" if e["type"] == "call" else "P"
        unusual_top.append(f"{_side}{e['strike']:.0f} v={e['vol']} oi={e['oi']}")

    # net premium: sum(call_vol * mid) - sum(put_vol * mid)  ($ directional flow)
    call_prem = sum(e["vol"] * e["mid"] * 100 for e in calls if e["mid"] > 0)
    put_prem  = sum(e["vol"] * e["mid"] * 100 for e in puts  if e["mid"] > 0)
    net_prem  = call_prem - put_prem

    # max pain: strike that minimises total $ pain across all OI
    all_strikes = sorted({e["strike"] for e in all_contracts})
    max_pain_strike = None
    if all_strikes:
        min_pain = float("inf")
        for s in all_strikes:
            pain = 0.0
            for e in all_contracts:
                if e["type"] == "call" and e["strike"] < s:
                    pain += (s - e["strike"]) * e["oi"] * 100
                elif e["type"] == "put" and e["strike"] > s:
                    pain += (e["strike"] - s) * e["oi"] * 100
            if pain < min_pain:
                min_pain = pain
                max_pain_strike = s

    # vol/OI ratio (aggregate activity gauge)
    total_vol = sum(e["vol"] for e in all_contracts)
    total_oi  = sum(e["oi"]  for e in all_contracts)
    vol_oi    = round(total_vol / total_oi, 2) if total_oi > 0 else None

    return {
        "atm_iv":      round(atm_iv * 100, 1) if atm_iv else None,
        "pc_ratio":    pc_ratio,
        "skew_25d":    skew_25d,
        "chain_count": len(results),
        # ── enhanced options intelligence ──
        "unusual_count": len(unusual),
        "unusual_top":   unusual_top,
        "net_premium":   round(net_prem),
        "net_direction":  "Bullish" if net_prem > 0 else ("Bearish" if net_prem < 0 else "Neutral"),
        "max_pain":      max_pain_strike,
        "max_pain_dist": round((max_pain_strike / spot - 1) * 100, 1) if max_pain_strike and spot else None,
        "vol_oi_ratio":  vol_oi,
        "spot":          spot,
        # full contract list for flow analysis panel
        "_contracts":    all_contracts,
    }

@st.cache_data(ttl=3600, show_spinner=False)
def _tf_short_volume(ticker: str) -> dict:
    """Fetch short-volume ratio from Polygon (free tier). Returns {} on failure."""
    try:
        import requests as _rq
        from datetime import date, timedelta
    except ImportError:
        return {}
    api_key = (os.getenv("POLYGON_API_KEY") or os.getenv("MASSIVE_API_KEY") or "").strip()
    base    = os.getenv("POLYGON_API_BASE") or "https://api.polygon.io"
    if not api_key:
        return {}
    # Polygon short-volume endpoint: /v3/reference/short-volume/{ticker}
    end   = date.today()
    start = end - timedelta(days=7)
    try:
        r = _rq.get(f"{base}/v3/reference/short-volume",
                    params={"ticker": ticker.upper(),
                            "date.gte": start.isoformat(),
                            "date.lte": end.isoformat(),
                            "order": "desc", "limit": 5,
                            "apiKey": api_key},
                    timeout=8)
        rows = r.json().get("results") or [] if r.ok else []
    except Exception:
        return {}
    if not rows:
        return {}
    latest = rows[0]
    short_v = int(latest.get("short_volume") or 0)
    total_v = int(latest.get("short_volume_exempt", 0)) + short_v
    # some endpoints return short_exempt + short as total; others have total_volume
    total_v = int(latest.get("total_volume") or total_v) or total_v
    ratio   = round(short_v / total_v, 3) if total_v > 0 else None
    return {
        "short_vol":       short_v,
        "total_vol":       total_v,
        "short_vol_ratio": ratio,
        "date":            latest.get("date") or latest.get("t") or "",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# SMART MONEY HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _sm_finnhub_get(endpoint: str, params: dict = None, timeout: int = 10):
    """Finnhub REST wrapper. Returns parsed JSON or []."""
    import requests as _rq
    key = (os.getenv("FINNHUB_API_KEY") or "").strip()
    if not key:
        return []
    p = dict(params or {})
    p["token"] = key
    try:
        r = _rq.get(f"https://finnhub.io/api/v1{endpoint}", params=p, timeout=timeout)
        return r.json() if r.ok else []
    except Exception:
        return []

def _sm_fmp_get(endpoint: str, params: dict = None, timeout: int = 10):
    """Financial Modeling Prep REST wrapper. Returns parsed JSON or []."""
    import requests as _rq
    key = (os.getenv("FMP_API_KEY") or "").strip()
    if not key:
        return []
    p = dict(params or {})
    p["apikey"] = key
    try:
        r = _rq.get(f"https://financialmodelingprep.com/api{endpoint}", params=p, timeout=timeout)
        return r.json() if r.ok else []
    except Exception:
        return []

@st.cache_data(ttl=3600, show_spinner=False)
def _sm_congress_trades(ticker: str = "") -> pd.DataFrame:
    """Fetch congressional trading data from QuiverQuant (free). Returns DataFrame."""
    import requests as _rq
    try:
        r = _rq.get(
            "https://api.quiverquant.com/beta/live/congresstrading",
            headers={"Authorization": "Bearer free"},
            timeout=15,
        )
        raw = r.json() if r.ok else []
    except Exception:
        raw = []
    if not raw or not isinstance(raw, list):
        return pd.DataFrame()
    rows = []
    for r in raw:
        tkr = (r.get("Ticker") or "").upper()
        if ticker and tkr != ticker.upper():
            continue
        txn = (r.get("Transaction") or "").strip()
        rows.append({
            "Date": r.get("TransactionDate", ""),
            "ReportDate": r.get("ReportDate", ""),
            "Member": r.get("Representative", ""),
            "Party": r.get("Party", ""),
            "Ticker": tkr,
            "Type": txn,
            "Range": r.get("Range", ""),
            "Chamber": r.get("House", ""),
            "ExcessReturn": r.get("ExcessReturn"),
            "PriceChange": r.get("PriceChange"),
            "SPYChange": r.get("SPYChange"),
        })
    df = pd.DataFrame(rows)
    if not df.empty and "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.sort_values("Date", ascending=False).reset_index(drop=True)
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def _sm_insider_trades(ticker: str = "") -> pd.DataFrame:
    """Fetch insider trades from Finnhub (free tier). Returns DataFrame."""
    if not ticker:
        return pd.DataFrame()
    raw = _sm_finnhub_get("/stock/insider-transactions", {"symbol": ticker.upper()})
    if isinstance(raw, dict):
        raw = raw.get("data", [])
    if not raw or not isinstance(raw, list):
        return pd.DataFrame()
    rows = []
    for r in raw:
        shares = abs(float(r.get("change") or 0))
        rows.append({
            "Date": r.get("filingDate") or r.get("transactionDate") or "",
            "Name": r.get("name", ""),
            "Ticker": ticker.upper(),
            "Type": "Buy" if (r.get("change") or 0) > 0 else "Sale",
            "Shares": shares,
            "SharesOwned": float(r.get("share") or 0),
            "IsDerivative": r.get("isDerivative", False),
            "Source": r.get("source", ""),
        })
    df = pd.DataFrame(rows)
    if not df.empty and "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.sort_values("Date", ascending=False).reset_index(drop=True)
    return df

def _sm_flow_history_path() -> str:
    """Path to rolling flow history parquet."""
    return os.path.join(os.environ.get("PER_TICKER_PARQUET_DIR", ""), "_flow_history.parquet")

def _sm_save_flow(flow_records: list) -> None:
    """Append flow records with ScanTime to _flow_history.parquet. Purge >30 days."""
    if not flow_records:
        return
    fpath = _sm_flow_history_path()
    if not fpath or not os.path.isdir(os.path.dirname(fpath)):
        return
    now = pd.Timestamp.now(tz="US/Eastern")
    new_df = pd.DataFrame(flow_records)
    new_df["ScanTime"] = now
    try:
        if os.path.exists(fpath):
            old = pd.read_parquet(fpath)
            combined = pd.concat([old, new_df], ignore_index=True)
        else:
            combined = new_df
        # Purge older than 30 days
        if "ScanTime" in combined.columns:
            combined["ScanTime"] = pd.to_datetime(combined["ScanTime"], utc=True, errors="coerce")
            cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=30)
            combined = combined[combined["ScanTime"] >= cutoff]
        combined.to_parquet(fpath, index=False)
    except Exception:
        pass

def _sm_load_flow_history() -> pd.DataFrame:
    """Load _flow_history.parquet. Returns empty DataFrame if missing."""
    fpath = _sm_flow_history_path()
    try:
        if os.path.exists(fpath):
            return pd.read_parquet(fpath)
    except Exception:
        pass
    return pd.DataFrame()

def _sm_flow_trends(fh_df: pd.DataFrame, days: int = 5) -> pd.DataFrame:
    """Aggregate flow history into daily net premium per ticker. Returns trend df."""
    if fh_df.empty or "ScanTime" not in fh_df.columns:
        return pd.DataFrame()
    df = fh_df.copy()
    df["ScanTime"] = pd.to_datetime(df["ScanTime"], utc=True, errors="coerce")
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
    df = df[df["ScanTime"] >= cutoff]
    if df.empty:
        return pd.DataFrame()
    df["ScanDate"] = df["ScanTime"].dt.date
    # Need C/P and Est $Prem columns
    if "C/P" not in df.columns or "Est $Prem" not in df.columns:
        return pd.DataFrame()
    # Daily aggregation per ticker
    def _agg(g):
        call_p = g.loc[g["C/P"] == "C", "Est $Prem"].sum()
        put_p = g.loc[g["C/P"] == "P", "Est $Prem"].sum()
        return pd.Series({"CallPrem": call_p, "PutPrem": put_p, "NetPrem": call_p - put_p, "Contracts": len(g)})
    daily = df.groupby(["Ticker", "ScanDate"]).apply(_agg).reset_index()
    # Pivot to get per-ticker daily series, then compute momentum
    tickers = daily.groupby("Ticker")["NetPrem"].sum().abs().nlargest(50).index.tolist()
    daily = daily[daily["Ticker"].isin(tickers)]
    return daily.sort_values(["Ticker", "ScanDate"])


def _tf_score_ticker(sdf: pd.DataFrame, spy_ret_20d: float, sentiment_score: float = 0.0) -> tuple:
    """Score a ticker 0–100 (normalized). Returns (total_score, components_dict)."""
    if sdf is None or len(sdf) < 22:
        return 0.0, {}
    g     = sdf.sort_values("Date")
    close = pd.to_numeric(g["Close"], errors="coerce").dropna()
    if len(close) < 22:
        return 0.0, {}
    px    = close.replace(0, np.nan)
    sma10 = close.rolling(10, min_periods=8).mean()
    FT    = 0.03
    s1d  = float((sma10.diff(1)  / 1  / px * 100).iloc[-1])
    s5d  = float((sma10.diff(5)  / 5  / px * 100).iloc[-1])
    s10d = float((sma10.diff(10) / 10 / px * 100).iloc[-1])
    align = int(s1d > 0) + int(s5d > 0) + int(s10d >= FT)

    # 1. Slope alignment   (0–25): steeper cliff — 0/5/15/25 for align 0/1/2/3
    slope_pts = [0.0, 5.0, 15.0, 25.0][min(align, 3)]

    # 2. RS vs SPY 20d     (0–20): baseline 8
    ret_20d = float((close.iloc[-1] / close.iloc[-21] - 1) * 100) if len(close) >= 21 else 0.0
    rs_diff = ret_20d - spy_ret_20d
    rs_pts  = min(20.0, max(0.0, 8.0 + rs_diff * 0.5))

    # 3. Relative volume   (0–20)
    vol = pd.to_numeric(g["Volume"], errors="coerce") if "Volume" in g.columns else pd.Series(dtype=float)
    avg_v = (float(pd.to_numeric(g["AvgVol_20"], errors="coerce").iloc[-1])
             if "AvgVol_20" in g.columns
             else float(vol.dropna().iloc[-21:-1].mean()) if len(vol.dropna()) >= 21
             else float(vol.dropna().mean()) if not vol.dropna().empty else 1.0)
    cur_v    = float(vol.dropna().iloc[-1]) if not vol.dropna().empty else 0.0
    rel_v    = cur_v / avg_v if avg_v > 0 else 1.0
    rvol_pts = min(20.0, max(0.0, (rel_v - 1.0) * 13.3))

    # 4. Entry quality     (0–20): near rising MA + RSI sweet zone; RSI > 70 = overbought = 0 pts
    sma_last  = float(sma10.iloc[-1]) if not np.isnan(float(sma10.iloc[-1])) else float(close.iloc[-1])
    pull_dist = (float(close.iloc[-1]) / sma_last - 1.0) * 100
    rsi_val   = float(rsi(close, 14).iloc[-1])
    entry_pts = 0.0
    if align >= 2 and rsi_val <= 70:
        if -1.0 <= pull_dist <= 2.5:
            entry_pts += 10.0
        if 35 <= rsi_val <= 65:
            entry_pts += 10.0
        elif 30 <= rsi_val < 35 or 65 < rsi_val <= 70:
            entry_pts += 5.0

    # 5. Momentum          (0–15): 5-day price return acceleration
    ret_5d = float((close.iloc[-1] / close.iloc[-6] - 1) * 100) if len(close) >= 6 else 0.0
    if ret_5d >= 3.0:
        mom_pts = 15.0
    elif ret_5d >= 1.5:
        mom_pts = 10.0
    elif ret_5d >= 0.5:
        mom_pts = 6.0
    elif ret_5d >= 0.0:
        mom_pts = 3.0
    else:
        mom_pts = 0.0

    # 6. Sentiment         (0–9): FinBERT news sentiment (bonus on top of base 100)
    #    Neutral sentiment (score=0) → 4.5 pts → mild baseline bonus
    sent_pts = round(max(0.0, min(9.0, 4.5 + sentiment_score * 4.5)), 1)

    # Base score (0-100) from original 5 components, sentiment added as bonus
    base_total = slope_pts + rs_pts + rvol_pts + entry_pts + mom_pts
    total = round(base_total + sent_pts, 1)
    return total, {
        "Slope(0-25)":      round(slope_pts, 1),
        "RS_SPY(0-20)":     round(rs_pts, 1),
        "RelVol(0-20)":     round(rvol_pts, 1),
        "Entry(0-20)":      round(entry_pts, 1),
        "Momentum(0-15)":   round(mom_pts, 1),
        "Sentiment(0-9)":   sent_pts,
        "Total":            total,
        "align_score":      align,
        "rel_vol":          round(rel_v, 2),
        "rsi":              round(rsi_val, 1),
        "pull_pct":         round(pull_dist, 2),
        "slope_1d":         round(s1d, 4),
        "slope_5d":         round(s5d, 4),
        "slope_10d":        round(s10d, 4),
        "ret_20d":          round(ret_20d, 2),
        "ret_5d":           round(ret_5d, 2),
    }

def _tf_detect_setups(sdf: pd.DataFrame) -> list:
    """Return list of named setup strings for the ticker."""
    if sdf is None or len(sdf) < 22:
        return []
    g     = sdf.sort_values("Date")
    close = pd.to_numeric(g["Close"], errors="coerce")
    px    = close.replace(0, np.nan)
    sma10 = close.rolling(10, min_periods=8).mean()
    FT    = 0.03
    s1d  = sma10.diff(1)  / 1  / px * 100
    s5d  = sma10.diff(5)  / 5  / px * 100
    s10d = sma10.diff(10) / 10 / px * 100
    align = int(float(s1d.iloc[-1]) > 0) + int(float(s5d.iloc[-1]) > 0) + int(float(s10d.iloc[-1]) >= FT)
    vol   = pd.to_numeric(g["Volume"], errors="coerce") if "Volume" in g.columns else pd.Series(dtype=float)
    avg_v = vol.rolling(20, min_periods=10).mean()
    rel_v = float((vol / avg_v.replace(0, np.nan)).iloc[-1]) if len(vol.dropna()) >= 10 else 1.0
    sma_last  = float(sma10.iloc[-1]) if not np.isnan(float(sma10.iloc[-1])) else float(close.iloc[-1])
    pull_dist = (float(close.iloc[-1]) / sma_last - 1.0) * 100
    rsi_val   = float(rsi(close, 14).iloc[-1])
    all_aligned = (s1d > 0) & (s5d > 0) & (s10d >= FT)
    setups = []
    # Stack Ignition: all 3 just aligned for the first time
    if align == 3:
        prev_align = (int(float(s1d.iloc[-2]) > 0) + int(float(s5d.iloc[-2]) > 0)
                      + int(float(s10d.iloc[-2]) >= FT)) if len(s10d) >= 2 else 3
        if prev_align < 3 and rel_v >= 1.3:
            setups.append("★ Stack Ignition")
        else:
            setups.append("⬆ Full Stack")
    # Pullback to Rising MA
    if align >= 2 and -1.0 <= pull_dist <= 2.5 and rsi_val <= 58:
        setups.append("↩ Pullback-to-MA")
    # Outsized volume day
    if rel_v >= 2.5:
        setups.append(f"⚡ Vol Spike ({rel_v:.1f}×)")
    # RS Emerging: align≥2 but was flat recently
    if align >= 2 and len(all_aligned) >= 7:
        if not all_aligned.iloc[-7:-1].any():
            setups.append("🚀 RS Emerging")
    # Stack Break warning
    if float(s1d.iloc[-1]) <= 0 and float(s5d.iloc[-1]) > 0 and float(s10d.iloc[-1]) >= FT:
        setups.append("⚠ Stack Break")
    if not setups and align >= 2:
        setups.append("✓ Upswing")
    return setups

# ═══════════════════════════════════════════════════════════════════════════════

# ---------------- Scans ----------------
if nav == 'Scans':
    st.subheader('Signal Scans')
    _show_scan_banner()
    scan_type = st.selectbox('Mode', ['Single Ticker', 'All Tickers (latest)'], index=0)
    strategy = st.selectbox('Strategy', [
        'Gap >= |X| %',
        'Bollinger Squeeze Breakout',
        '52-week High Breakout',
        'Earnings: Prior & Day-Of',
        'RS breakout (SPY/QQQ)',
        'Slope acceleration',
        'Vol contraction breakout',
        'Volume spike (cont/meanrev)',
        'Deep pullback in uptrend',
        '52w Low Bounce',
    ], index=0)

    if scan_type == 'All Tickers (latest)':
        gap_thr = st.number_input('Gap threshold % (sign indicates direction)', value=3.0, step=0.5, help="Positive = gap up only; negative = gap down only; 0 = either")
        look52 = st.number_input('52w lookback (days)', value=252, min_value=50, max_value=3000, step=10)
        min_price = st.number_input('Min last close ($)', value=5.0, min_value=0.0, step=0.5)
        min_avgvol = st.number_input('Min AvgVol_20', value=500000.0, min_value=0.0, step=50000.0)
        _all_ticker_supported = strategy.startswith("Gap") or strategy.startswith("52-week")
        if not _all_ticker_supported:
            st.info(f"**All Tickers mode** only supports Gap and 52-week High strategies. "
                    f"Switch to **Single Ticker** mode to run '{strategy}'.")
        run_all = st.button('Run All-Ticker Scan', disabled=not _all_ticker_supported)
        if run_all and _all_ticker_supported:
            base_dir = os.environ.get("PER_TICKER_PARQUET_DIR") or ""
            if not base_dir or not os.path.isdir(base_dir):
                st.error("PER_TICKER_PARQUET_DIR is not set or not a directory.")
            else:
                import glob
                files = glob.glob(os.path.join(base_dir, "*.parquet"))
                if not files:
                    st.warning("No parquet files found in PER_TICKER_PARQUET_DIR.")
                else:
                    rows = []
                    total = len(files)
                    prog = st.progress(0.0)
                    for i, fp in enumerate(files, 1):
                        prog.progress(i/total)
                        try:
                            dfp = pd.read_parquet(fp)
                            if dfp is None or dfp.empty:
                                continue
                            d = dfp.sort_values("Timestamp") if "Timestamp" in dfp.columns else dfp.sort_values("Date") if "Date" in dfp.columns else dfp.copy()
                            d = d.tail(max(260, look52 + 10))
                            if d.empty:
                                continue
                            sym = d.get("Ticker")
                            if isinstance(sym, pd.Series):
                                sym = sym.iloc[-1]
                            else:
                                sym = Path(fp).stem
                            sym = str(sym).upper().replace("_", ".")
                            # basic liquidity/price filters
                            last_close = float(d["Close"].iloc[-1]) if "Close" in d.columns else None
                            avg_vol = float(d["AvgVol_20"].iloc[-1]) if "AvgVol_20" in d.columns else None
                            if last_close is not None and last_close < float(min_price):
                                continue
                            if avg_vol is not None and avg_vol < float(min_avgvol):
                                continue
                            if strategy.startswith("Gap"):
                                if "Open" not in d.columns or "Close" not in d.columns:
                                    continue
                                d = d.set_index("Timestamp" if "Timestamp" in d.columns else "Date")
                                d.index = pd.to_datetime(d.index)
                                cur = d.iloc[-1]
                                prev = d.iloc[-2] if len(d) >= 2 else None
                                if prev is None:
                                    continue
                                gap_pct = (float(cur["Open"]) / float(prev["Close"]) - 1.0) * 100.0
                                if gap_thr > 0 and gap_pct < gap_thr:
                                    continue
                                if gap_thr < 0 and gap_pct > gap_thr:
                                    continue
                                if gap_thr == 0 and abs(gap_pct) < 0.0001:
                                    continue
                                rows.append({
                                    "Ticker": sym,
                                    "Date": cur.name.date(),
                                    "Gap %": gap_pct,
                                    "Open": cur.get("Open"),
                                    "Close": cur.get("Close"),
                                    "Volume": cur.get("Volume"),
                                })
                            elif strategy.startswith("52-week"):
                                d = d.set_index("Timestamp" if "Timestamp" in d.columns else "Date")
                                d.index = pd.to_datetime(d.index)
                                if "High" not in d.columns or "Close" not in d.columns:
                                    continue
                                window = d.tail(int(look52))
                                if window.empty:
                                    continue
                                hh = window["High"].max()
                                curc = window["Close"].iloc[-1]
                                if curc >= hh:
                                    rows.append({
                                        "Ticker": sym,
                                        "Date": window.index[-1].date(),
                                        "Close": curc,
                                        "52w High": hh,
                                        "Volume": window["Volume"].iloc[-1] if "Volume" in window.columns else None,
                                    })
                            else:
                                # other strategies not implemented for all-ticker mode
                                continue
                        except Exception:
                            continue
                    prog.empty()
                    if not rows:
                        st.info("No matches found across all tickers.")
                    else:
                        df_out = pd.DataFrame(rows)
                        df_out = df_out.sort_values("Date", ascending=False)
                        st.dataframe(df_out, use_container_width=True)

    if scan_type == 'Single Ticker':
        tkr_in = st.text_input('Ticker (single)', value=ticker).strip().upper()
        c1, c2, c3 = st.columns(3)
        with c1:
            gap_thr = st.number_input('Gap threshold % (sign indicates direction)', value=-3.0, step=0.5)
        with c2:
            bb_pct = st.number_input('BB width percentile <=', value=20.0, min_value=1.0, max_value=100.0, step=1.0)
        with c3:
            look52 = st.number_input('52w lookback (days)', value=252, min_value=50, max_value=3000, step=10)
        f1, f2 = st.columns(2)
        with f1:
            min_price_single = st.number_input('Min last close ($)', value=5.0, min_value=0.0, step=0.5)
        with f2:
            min_avgvol_single = st.number_input('Min AvgVol_20', value=500000.0, min_value=0.0, step=50000.0)
        run_single = st.button('Run Single Scan')
        if run_single:
            sdf = (st.session_state.get('daily_df') if st.session_state.get('ticker','').upper()==tkr_in else None) or _load_daily_df(tkr_in)
            need_open = strategy.startswith('Gap')
            if sdf is None or sdf.empty or (need_open and 'Open' not in sdf.columns):
                st.error('No data found for ticker from Parquet (or missing Open for Gap).')
            else:
                # basic liquidity/price filters
                last_close = float(sdf['Close'].iloc[-1]) if 'Close' in sdf.columns else None
                avg_vol = float(sdf['AvgVol_20'].iloc[-1]) if 'AvgVol_20' in sdf.columns else None
                if (last_close is not None and last_close < float(min_price_single)) or (avg_vol is not None and avg_vol < float(min_avgvol_single)):
                    st.warning("Ticker does not meet min price/volume filters.")
                    ev = pd.DataFrame()
                else:
                    if strategy.startswith('Gap'):
                        ev = _gap_events(sdf, float(gap_thr))
                    elif strategy.startswith('Bollinger'):
                        ev = _bb_squeeze_breakouts(sdf)
                        ev = ev[ev['BB_Width_%ile'] <= float(bb_pct)]
                    elif strategy.startswith('Earnings'):
                        try:
                            ev = _earnings_prior_dayof_scan(tkr_in, sdf)
                        except Exception as e:
                            st.error(f'Earnings scan failed: {e}')
                            ev = pd.DataFrame()
                    elif strategy.startswith('RS breakout'):
                        g = sdf.sort_values("Date").set_index("Date")
                        rs_ok = True
                        if "RS_SPY" not in g.columns or "RS_QQQ" not in g.columns:
                            spy_df = _load_daily_df("SPY")
                            qqq_df = _load_daily_df("QQQ")
                            if spy_df is None or spy_df.empty or qqq_df is None or qqq_df.empty:
                                st.error("SPY/QQQ parquet not found – needed for RS computation.")
                                ev = pd.DataFrame()
                                rs_ok = False
                            else:
                                spy_c = spy_df.sort_values("Date").set_index("Date")["Close"]
                                qqq_c = qqq_df.sort_values("Date").set_index("Date")["Close"]
                                g["RS_SPY"] = g["Close"] / spy_c.reindex(g.index, method="ffill")
                                g["RS_QQQ"] = g["Close"] / qqq_c.reindex(g.index, method="ffill")
                        if rs_ok:
                            rs_spy = pd.to_numeric(g["RS_SPY"], errors="coerce")
                            rs_qqq = pd.to_numeric(g["RS_QQQ"], errors="coerce")
                            rs_h = pd.concat([rs_spy, rs_qqq], axis=1).rolling(60, min_periods=20).max().max(axis=1)
                            px_h = g["Close"].rolling(60, min_periods=20).max()
                            sig = ((rs_spy >= rs_h) | (rs_qqq >= rs_h)) & (g["Close"] >= px_h)
                            ev = g[sig][["Close","RS_SPY","RS_QQQ"]].copy()
                            ev["RS_High60"] = rs_h.reindex(ev.index)
                            ev["Price_High60"] = px_h.reindex(ev.index)
                    elif strategy.startswith('Slope acceleration'):
                        g = sdf.sort_values("Date").set_index("Date")
                        if "SMA_20_Slope1" not in g.columns:
                            sma20 = g["Close"].rolling(20, min_periods=10).mean()
                            g["SMA_20_Slope1"] = (sma20 - sma20.shift(1)) / g["Close"].replace(0, np.nan) * 100
                        if "EMA_21_Slope1" not in g.columns:
                            ema21 = g["Close"].ewm(span=21, min_periods=10).mean()
                            g["EMA_21_Slope1"] = (ema21 - ema21.shift(1)) / g["Close"].replace(0, np.nan) * 100
                        s20 = pd.to_numeric(g["SMA_20_Slope1"], errors="coerce")
                        e21 = pd.to_numeric(g["EMA_21_Slope1"], errors="coerce")
                        sig = ((s20 > 0) & (s20 > s20.shift(1))) | ((e21 > 0) & (e21 > e21.shift(1)))
                        ev = g[sig][["Close","SMA_20_Slope1","EMA_21_Slope1"]].copy()
                    elif strategy.startswith('Vol contraction'):
                        g = sdf.sort_values("Date").set_index("Date")
                        if "BB_Upper_20" not in g.columns or "BB_Lower_20" not in g.columns:
                            mid = g["Close"].rolling(20, min_periods=10).mean()
                            std = g["Close"].rolling(20, min_periods=10).std()
                            g["BB_Upper_20"] = mid + 2 * std
                            g["BB_Lower_20"] = mid - 2 * std
                        if "VolRank_20" not in g.columns:
                            g["VolRank_20"] = g["Volume"].rolling(252, min_periods=20).rank(pct=True) * 100
                        width = (g['BB_Upper_20'] - g['BB_Lower_20']) / g['Close']
                        tight = width.rank(pct=True) <= 0.2
                        squeeze = tight & (g['VolRank_20'] <= 20)
                        breakout = g["Close"] > g["BB_Upper_20"]
                        sig = squeeze.shift(1, fill_value=False) & breakout
                        ev = g[sig][["Close","BB_Upper_20","BB_Lower_20","VolRank_20"]].copy()
                        ev["BB_Width"] = width.reindex(ev.index)
                    elif strategy.startswith('Volume spike'):
                        g = sdf.sort_values("Date").set_index("Date")
                        if "AvgVol_20" not in g.columns:
                            g["AvgVol_20"] = g["Volume"].rolling(20, min_periods=5).mean()
                        vol_rank = (g["Volume"] / g["AvgVol_20"].replace(0, np.nan))
                        spike = vol_rank >= 3
                        body_pos = (g["Close"] - g["Low"]) / (g["High"] - g["Low"]).replace(0, np.nan)
                        continuation = spike & (body_pos >= 0.5)
                        meanrev = spike & (body_pos <= 0.25)
                        sig = continuation | meanrev
                        ev = g[sig][["Close","Volume","AvgVol_20"]].copy()
                        ev["Type"] = np.where(continuation.reindex(ev.index, fill_value=False), "Continuation", "MeanReversion")
                        ev["Vol/AvgVol20"] = vol_rank.reindex(ev.index)
                    elif strategy.startswith('Deep pullback'):
                        g = sdf.sort_values("Date").set_index("Date")
                        if "SMA_50" not in g.columns:
                            g["SMA_50"] = g["Close"].rolling(50, min_periods=20).mean()
                        if "SMA_200" not in g.columns:
                            g["SMA_200"] = g["Close"].rolling(200, min_periods=50).mean()
                        if "RSI_14" not in g.columns:
                            g["RSI_14"] = rsi(g["Close"], 14)
                        uptrend = (g["SMA_50"] > g["SMA_200"]) & (g["SMA_50"].diff() > 0)
                        pull_pct = (g["Close"] / g["High"].rolling(20, min_periods=5).max().replace(0, np.nan) - 1.0) * 100.0
                        sig = uptrend & (pull_pct <= -5.0) & (g["RSI_14"] <= 40)
                        ev = g[sig][["Close","SMA_50","SMA_200","RSI_14"]].copy()
                        ev["Pull_from_20d_High_%"] = pull_pct.reindex(ev.index)
                    elif strategy.startswith('52w Low Bounce'):
                        g = sdf.sort_values("Date").set_index("Date")
                        if "RSI_5" not in g.columns:
                            g["RSI_5"] = rsi(g["Close"], 5)
                        ll = g["Low"].rolling(252, min_periods=50).min()
                        near_ll = (g["Low"] <= ll * 1.01)
                        bounce = near_ll & (g["RSI_5"] >= 30)
                        ev = g[bounce][["Close","Low","RSI_5"]].copy()
                        ev["LL_252"] = ll.reindex(ev.index)
                    else:
                        ev = _breakout_52w(sdf, int(look52))
                    st.caption(f'Events: {len(ev)}')
                    if not ev.empty:
                        if 'Next_Overnight_%' in ev.columns:
                            st.metric('Avg Next Overnight %', f"{ev['Next_Overnight_%'].mean():.2f}%")
                            st.metric('Win rate (Next Overnight > 0)', f"{(ev['Next_Overnight_%']>0).mean()*100:.1f}%")
                        display_df = _prune_scan_columns(ev).tail(200)
                        st.dataframe(display_df, use_container_width=True)
                        csv = ev.to_csv(index=False).encode('utf-8')
                        st.download_button('Download CSV', data=csv, file_name=f'{tkr_in}_{strategy.replace(" ","_")}.csv', mime='text/csv')

        with st.expander("Latest news (Polygon/Massive)", expanded=False):
            news = _fetch_news(tkr_in, limit=10)
            if not news:
                st.info("No news found for this ticker.")
            else:
                for n in news:
                    pub = n.get("published_utc", "")
                    title = n.get("title", "")
                    src = (n.get("publisher") or {}).get("name", "")
                    st.markdown(f"- {pub} | {src} | {title}")

    st.markdown("---")
    st.subheader("Return Visualizers")
    _show_scan_banner()
    vis = st.selectbox(
        "Visualizer",
        [
        "Gap Off-Open Return",
        "Prev Day CC→Close Overnight",
        "Intraday Time Move → Close",
        "Earnings Long-Term Returns",
        "RSI Threshold Backtest",
        "Bollinger + RSI Forward Returns",
        "MA Slope Dynamics",
        "ThetaData Snapshot",
    ],
    index=0,
    key="vis_kind",
)
    tkr_vis = st.text_input("Ticker (or A/B for relative)", value=ticker).strip().upper()

    colA, colB = st.columns(2)
    with colA:
        consec_up = st.number_input("Consecutive up days (>=)", value=0, min_value=0, max_value=10, step=1)
    with colB:
        wd_names = ["Mon","Tue","Wed","Thu","Fri"]
        wd_sel = st.multiselect("Weekdays", options=wd_names, default=wd_names)
    rsi_gate = 0  # RSI filter disabled

    def _weekday_mask(dti: pd.DatetimeIndex, wd_sel_list: list[str]) -> pd.Series:
        wd_map = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri"}
        try:
            dti_n = dti.tz_localize(None) if dti.tzinfo is not None else dti
        except Exception:
            dti_n = dti
        return pd.Series([wd_map.get(int(x),"") in set(wd_sel_list) for x in dti_n.weekday], index=dti)

    def _apply_common_filters(df_d: pd.DataFrame) -> pd.Series:
        idx_ok = pd.Series(True, index=df_d.index)
        if consec_up and consec_up > 0:
            cc = df_d["Close"].pct_change()
            streak = (cc > 0).astype(int)
            ups = streak.rolling(consec_up, min_periods=consec_up).sum().shift(1)
            idx_ok &= (ups >= consec_up)
        if isinstance(df_d.index, pd.DatetimeIndex):
            idx_ok &= _weekday_mask(df_d.index, wd_sel)
        return idx_ok.fillna(False)

    # ---- Shared visual helpers ----
    import re as _re
    def _slug(s: str) -> str:
        return _re.sub(r"[^a-zA-Z0-9_]", "_", str(s))

    def _viz_hist_and_heatmap(series: pd.Series, title_prefix: str = ""):
        try:
            import plotly.graph_objects as _go
        except Exception:
            _go = None
        _k = _slug(vis)
        show_hist = st.checkbox("Show histogram", value=True, key=f"hist_{_k}")
        show_heat = st.checkbox("Show month/weekday heatmap", value=False, key=f"heat_{_k}")
        heat_metric = st.selectbox("Heatmap metric", ["Count", "Mean Return"], index=0, key=f"heat_metric_{_k}") if show_heat else None
        if show_hist and _go is not None and len(series) > 0:
            fig_h = _go.Figure(_go.Histogram(x=series.values, nbinsx=40, marker_color="#4e79a7"))
            fig_h.update_layout(title=f"{title_prefix}Histogram", xaxis_title="Return %", yaxis_title="Freq")
            st.plotly_chart(fig_h, use_container_width=True)
        if show_heat and len(series) > 0:
            idx = pd.to_datetime(series.index)
            dfm = pd.DataFrame({"month": idx.month, "wday": idx.weekday, "ret": series.values})
            if heat_metric == "Mean Return":
                pivot = dfm.pivot_table(index="month", columns="wday", values="ret", aggfunc="mean")
                z_title = "Mean %"
            else:
                pivot = dfm.pivot_table(index="month", columns="wday", values="ret", aggfunc="count")
                z_title = "Count"
            pivot = pivot.reindex(index=range(1,13), columns=range(0,5))
            if _go is not None:
                fig_ht = _go.Figure(data=_go.Heatmap(z=pivot.values, x=["Mon","Tue","Wed","Thu","Fri"], y=[str(m) for m in range(1,13)], colorscale="Blues"))
                fig_ht.update_layout(title=f"{title_prefix}Month x Weekday ({z_title})", xaxis_title="Weekday", yaxis_title="Month")
                st.plotly_chart(fig_ht, use_container_width=True)
        return

    def _viz_calendar_heatmap(series: pd.Series, title_prefix: str = ""):
        try:
            import plotly.graph_objects as _go
        except Exception:
            _go = None
        if len(series) == 0:
            return
        idx = pd.to_datetime(series.index)
        years = sorted(set(idx.year))
        _k = _slug(vis)
        show_cal = st.checkbox("Show calendar heatmap", value=False, key=f"cal_{_k}")
        if not show_cal:
            return
        year = st.selectbox("Year", options=years, index=len(years)-1, key=f"cal_year_{_k}") if years else None
        if year is None:
            return
        sel = series[idx.year == year]
        if len(sel) == 0:
            st.info("No occurrences for selected year.")
            return
        # Build day-of-month x month grid
        idx2 = pd.to_datetime(sel.index)
        dfm = pd.DataFrame({"dom": idx2.day, "mon": idx2.month, "ret": sel.values})
        pv_count = dfm.pivot_table(index="dom", columns="mon", values="ret", aggfunc="count").reindex(index=range(1,32), columns=range(1,13))
        pv_mean = dfm.pivot_table(index="dom", columns="mon", values="ret", aggfunc="mean").reindex(index=range(1,32), columns=range(1,13))
        metric = st.selectbox("Calendar metric", ["Count", "Mean Return"], index=0, key=f"cal_metric_{_k}")
        z = pv_count.values if metric == "Count" else pv_mean.values
        z_title = "Count" if metric == "Count" else "Mean %"
        if _go is not None:
            fig_cal = _go.Figure(data=_go.Heatmap(z=z, x=[str(m) for m in range(1,13)], y=[str(d) for d in range(1,32)], colorscale="Viridis"))
            fig_cal.update_layout(title=f"{title_prefix}Calendar {year} ({z_title})", xaxis_title="Month", yaxis_title="Day")
            st.plotly_chart(fig_cal, use_container_width=True)

    def _summary_table(d: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.DataFrame:
        if d is None or d.empty:
            return pd.DataFrame()
        g_full = d.sort_values("Date").set_index("Date") if "Date" in d.columns else d.copy()
        prevc_full = g_full["Close"].shift(1)
        out_full = pd.DataFrame(index=g_full.index)
        out_full["Prev Close to Close"] = (g_full["Close"] / prevc_full - 1.0) * 100.0
        out_full["Close to Open"] = (g_full["Open"] / prevc_full - 1.0) * 100.0
        out_full["Open to Close"] = (g_full["Close"] / g_full["Open"] - 1.0) * 100.0
        for H in (1,3,5,10):
            out_full[f"{H} Day Return"] = (g_full["Close"].shift(-H) / g_full["Close"] - 1.0) * 100.0
        # All-time high flags and N-day high number (days into history when ATH)
        cummax = g_full["Close"].cummax()
        out_full["All Time High"] = g_full["Close"] >= cummax
        out_full["# Day High"] = np.where(out_full["All Time High"], np.arange(1, len(g_full)+1), np.nan)
        # Consecutive up/down count with sign
        cc = g_full["Close"].diff()
        sgn = np.sign(cc.fillna(0))
        grp = (sgn != sgn.shift(1)).cumsum()
        streak = sgn.groupby(grp).cumsum()
        out_full["Consecutive Up/Down"] = streak
        # Strip tz from both sides — tz mismatch silently returns empty intersection
        idx_dt = pd.to_datetime(idx)
        try:
            idx_dt = idx_dt.tz_localize(None) if idx_dt.tzinfo is not None else idx_dt
        except Exception:
            try: idx_dt = idx_dt.tz_convert(None)
            except Exception: pass
        try:
            if out_full.index.tzinfo is not None:
                out_full.index = out_full.index.tz_localize(None)
        except Exception:
            try: out_full.index = out_full.index.tz_convert(None)
            except Exception: pass
        out = out_full.loc[out_full.index.intersection(idx_dt)]
        return out.dropna(how="all")

    def _viz_price_marks(daily_df: pd.DataFrame, dates_idx: pd.DatetimeIndex, title: str, lookback_days: int | None = None):
        try:
            from plotly.subplots import make_subplots as _mk
            import plotly.graph_objects as _go
        except Exception:
            st.info("plotly unavailable for price visualization.")
            return
        if daily_df is None or daily_df.empty:
            st.info("Daily data unavailable for price visualization.")
            return
        d = daily_df.sort_values("Date").set_index("Date") if "Date" in daily_df.columns else daily_df.copy()
        # Trim to a reasonable window: last lookback_days and also include markers span
        if lookback_days is not None and not d.empty:
            try:
                end_dt = d.index.max()
                start_dt = end_dt - pd.Timedelta(days=int(lookback_days))
                if dates_idx is not None and len(dates_idx) > 0:
                    start_dt = min(start_dt, pd.to_datetime(dates_idx).min() - pd.Timedelta(days=5))
                d = d.loc[d.index >= start_dt]
            except Exception:
                pass
        # Downsample only if very large to keep candles legible
        if len(d) > 1500:
            step = int(np.ceil(len(d) / 1500))
            d = d.iloc[::step]
        # Match the main chart look: white background, light grid, candlesticks + volume
        figp = _mk(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02, row_heights=[0.72, 0.28])
        figp.add_trace(_go.Candlestick(
            x=d.index,
            open=d.get("Open"),
            high=d.get("High"),
            low=d.get("Low"),
            close=d.get("Close"),
            name="OHLC",
            increasing_line_color="#2ecc40",
            decreasing_line_color="#ff4d4f",
            increasing_fillcolor="#2ecc40",
            decreasing_fillcolor="#ff4d4f",
            whiskerwidth=0.5,
            showlegend=False,
        ), row=1, col=1)

        # Bollinger overlay if present (prefer 20-period)
        bb_up = None
        bb_lo = None
        for col in d.columns:
            if isinstance(col, str) and col.startswith("BB_Upper_"):
                bb_up = col
            if isinstance(col, str) and col.startswith("BB_Lower_"):
                bb_lo = col
        if bb_up and bb_lo:
            try:
                figp.add_trace(_go.Scatter(
                    x=d.index, y=d[bb_up], mode="lines", line=dict(color="#1f77b4", width=1),
                    name="BB Upper", showlegend=False
                ), row=1, col=1)
                figp.add_trace(_go.Scatter(
                    x=d.index, y=d[bb_lo], mode="lines", line=dict(color="#1f77b4", width=1),
                    name="BB Lower", fill='tonexty', fillcolor="rgba(31,119,180,0.08)", showlegend=False
                ), row=1, col=1)
            except Exception:
                pass

        if "Volume" in d.columns:
            figp.add_trace(_go.Bar(
                x=d.index,
                y=d["Volume"],
                name="Volume",
                marker_color="#9aa0a6",
                opacity=0.35,
                showlegend=False,
            ), row=2, col=1)
        for dt in pd.to_datetime(dates_idx):
            try:
                figp.add_vrect(x0=dt, x1=dt + pd.Timedelta(days=1), fillcolor="rgba(255,0,0,0.08)", line_width=0)
            except Exception:
                pass
        _tpl_marks = template if "template" in dir() else "plotly_white"
        _is_dark   = ("dark" in str(_tpl_marks).lower())
        _bg        = "#0d1117" if _is_dark else "#fafafa"
        _grid      = "#30363d" if _is_dark else "#dfe3ea"
        figp.update_layout(
            title=title,
            xaxis_rangeslider_visible=False,
            template=_tpl_marks,
            height=650,
            margin=dict(l=30, r=20, t=45, b=35),
            xaxis_showgrid=True,
            xaxis2_showgrid=True,
            yaxis_showgrid=True,
            yaxis2_showgrid=True,
            plot_bgcolor=_bg,
            paper_bgcolor=_bg,
            yaxis_gridcolor=_grid,
            yaxis2_gridcolor=_grid,
        )
        st.plotly_chart(figp, use_container_width=True)

    def _fetch_thetadata_snapshot(ticker: str) -> dict:
        """
        ThetaData stock snapshot (market value) fetch.
        - Uses THETADATA_BASE_URL if set; else uses THETADATA_TERMINAL_URL + path.
        - THETADATA_API_KEY optional (if using local jar that doesn't require auth).
        - THETADATA_SNAPSHOT_PATH optional; defaults to operations/stock_snapshot_market_value.
        - Tries multiple candidate paths to handle jar routing differences.
        """
        import requests
        key = (os.getenv("THETADATA_API_KEY") or "").strip()
        term_url = os.getenv("THETADATA_TERMINAL_URL", "").strip().rstrip("/")
        term_root = term_url.rsplit("/v3", 1)[0] if "/v3" in term_url else term_url
        base_url = os.getenv("THETADATA_BASE_URL", "").strip().rstrip("/")
        snap_path_env = os.getenv("THETADATA_SNAPSHOT_PATH", "operations/stock_snapshot_market_value").strip("/")

        if not ticker:
            return {"error": "ticker required"}

        headers = {}
        if key:
            headers["Authorization"] = f"Bearer {key}"

        # Build candidate base hosts; if a local terminal is configured, prefer it and skip public
        host_candidates = [h for h in [base_url, term_url, term_root] if h]
        if not host_candidates:
            host_candidates = ["https://api.thetadata.us"]

        # Candidate paths to try
        paths = [
            snap_path_env,
            snap_path_env.replace("operations/", "", 1) if snap_path_env.startswith("operations/") else snap_path_env,
            f"v3/{snap_path_env}" if "v3/" not in snap_path_env else snap_path_env,
            "stock_snapshot_market_value",
            "operations/stock_snapshot_market_value",
        ]
        # Deduplicate while preserving order
        seen = set()
        path_list = []
        for p in paths:
            if p and p not in seen:
                seen.add(p)
                path_list.append(p)

        tried = []

        def _try(u: str):
            tried.append(u)
            return requests.get(
                u,
                params={"ticker": ticker.upper()},
                headers=headers,
                timeout=10,
            )

        try:
            resp = None
            for h in host_candidates:
                h_clean = h.rstrip("/")
                for p in path_list:
                    # Avoid double "v3" when host already has /v3
                    p_use = p
                    if h_clean.endswith("/v3") and p.startswith("v3/"):
                        p_use = p[len("v3/") :]
                    url = f"{h_clean}/{p_use.lstrip('/')}"
                    resp = _try(url)
                    if resp.ok:
                        out = resp.json()
                        out["_meta"] = {"tried": tried}
                        return out
            if resp is None:
                return {"error": "no request attempted", "tried": tried}
            return {"error": f"HTTP {resp.status_code}", "body": resp.text[:500], "tried": tried}
        except Exception as e:
            return {"error": str(e), "tried": tried}

    # --- Simple per-visualizer cache helpers to prevent results disappearing on rerun ---
    def _set_vis_cache(name: str, params: dict, table: pd.DataFrame):
        try:
            st.session_state[f'vis_{name}_tbl'] = table.copy()
            st.session_state[f'vis_{name}_params'] = params.copy()
        except Exception:
            pass

    def _get_vis_cache(name: str, params: dict) -> pd.DataFrame | None:
        try:
            prev_p = st.session_state.get(f'vis_{name}_params')
            prev_t = st.session_state.get(f'vis_{name}_tbl')
            if isinstance(prev_p, dict) and prev_t is not None and prev_p == params:
                return prev_t
        except Exception:
            return None
        return None

    def _render_scan_stats(tbl: pd.DataFrame):
        """Show quick mean stats for common return columns if present."""
        if tbl is None or tbl.empty:
            return
        cols_map = {
            "Avg Close→Close %": ["Close to Close", "Prev Close to Close", "Close?Close %"],
            "Avg Next Overnight %": ["Next_Overnight_%", "Next Overnight %", "Next_Open"],
            "Avg Next Intraday %": ["Next_Intraday_%", "Next Intraday %", "Day_%"],
            "Avg Next Close→Close %": ["Next_Total_%", "Next Close?Close %", "Next Close to Close"],
        }
        stats = []
        for label, candidates in cols_map.items():
            col = next((c for c in candidates if c in tbl.columns), None)
            if col:
                s = pd.to_numeric(tbl[col], errors="coerce").dropna()
                if len(s) > 0:
                    stats.append(f"{label}: {s.mean():.2f}% ({len(s)} rows)")
        if stats:
            msg = " | ".join(stats)
            st.caption(msg)
            try:
                st.session_state["last_scan_stats_caption"] = msg
            except Exception:
                pass

    # Global dataframe wrapper to auto-show stats during Scans tab
    try:
        if not getattr(st, "_dataframe_wrapped_for_stats", False):
            _orig_df_fn = st.dataframe
            def _df_with_stats(data, *args, **kwargs):
                res = _orig_df_fn(data, *args, **kwargs)
                try:
                    if st.session_state.get("nav") == "Scans" and isinstance(data, pd.DataFrame):
                        _render_scan_stats(data)
                except Exception:
                    pass
                return res
            st.dataframe = _df_with_stats  # type: ignore
            st._dataframe_wrapped_for_stats = True  # type: ignore
    except Exception:
        pass

    # Gap Off-Open: keep all columns, move timestamps right, default extras on, backfill daily gap
    gap_extra_cols_default = True

    def _reorder_gap_columns(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        cols = list(df.columns)
        ts_cols = [c for c in cols if "Timestamp" in str(c)]
        front = ["Prev Day Earnings"] if "Prev Day Earnings" in cols else []
        middle = [c for c in cols if c not in ts_cols and c not in front]
        return df[front + middle + ts_cols]

    def _backfill_gap_with_daily(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        out = df.copy()
        if "Close to Open" in out.columns:
            gap_src = None
            for c in ("GapPct", "Gap %", "gap_pct"):
                if c in out.columns:
                    gap_src = pd.to_numeric(out[c], errors="coerce")
                    break
            if gap_src is not None:
                mask = out["Close to Open"].isna()
                try:
                    out.loc[mask, "Close to Open"] = gap_src.reindex(out.index)[mask]
                except Exception:
                    out.loc[mask, "Close to Open"] = gap_src[mask] if hasattr(gap_src, "__getitem__") else gap_src
        return out

    def _render_gap_results(sym: str, tbl: pd.DataFrame):
        if tbl is None or tbl.empty:
            return
        # Single table view with optional styled toggle to avoid redundancy
        try:
            try:
                show_tbl = _reorder_gap_columns(_backfill_gap_with_daily(tbl.copy()))
            except Exception:
                show_tbl = tbl.copy()
            ts_cols = [c for c in show_tbl.columns if 'Timestamp' in c]
            for col in ts_cols:
                try:
                    vals = pd.to_datetime(show_tbl[col], errors='coerce')
                    show_tbl[col] = vals.dt.strftime('%Y-%m-%d %H:%M:%S').fillna('')
                except Exception:
                    show_tbl[col] = show_tbl[col].fillna('')
            if 'Reason' in show_tbl.columns:
                show_tbl['Reason'] = show_tbl['Reason'].fillna('')
            num_cols = [c for c in show_tbl.columns if (str(c).endswith('Return') or (c in ('Prev Close to Close','Close to Open','Open to Close','9:24 to Open','Prev Day Return','3D RSI')))]
            styled = (
                show_tbl.style
                .format({c: '{:.3f}' for c in num_cols})
                .map(lambda v: 'background-color:#2ecc40;color:#000' if (isinstance(v,(int,float)) and v>=0) else ('background-color:#ff6961;color:#000' if isinstance(v,(int,float)) else ''), subset=num_cols)
            )
            styled_view = st.checkbox("Styled view", value=True, key="gap_styled_view")
            if styled_view:
                st.write(styled)
            else:
                st.dataframe(show_tbl, use_container_width=True)
            _render_scan_stats(show_tbl)
        except Exception:
            try:
                fixed = _reorder_gap_columns(_backfill_gap_with_daily(tbl.copy()))
                st.dataframe(fixed, use_container_width=True)
                _render_scan_stats(fixed)
            except Exception:
                st.dataframe(tbl, use_container_width=True)

    def _render_chart_for_tbl(title: str, tbl: pd.DataFrame, ticker: str, default_lookback: int = 360):
        """Render an optional price chart with markers for the given table's index."""
        if tbl is None or tbl.empty:
            return
        key_base = title.lower().replace(" ", "_")
        chart_lookback = st.number_input(
            "Chart lookback (days)", value=default_lookback, min_value=30, max_value=1500, step=30, key=f"{key_base}_chart_lb_{ticker}"
        )
        daily_df = _load_daily_df(ticker)
        if daily_df is None or daily_df.empty:
            st.info("Daily data unavailable for chart.")
            return
        dd = daily_df.sort_values('Date')
        if isinstance(dd.index, pd.DatetimeIndex):
            try:
                cutoff = dd.index.max() - pd.Timedelta(days=int(chart_lookback))
                dd = dd.loc[dd.index >= cutoff]
            except Exception:
                dd = dd.tail(int(chart_lookback))
        _viz_price_marks(dd, pd.to_datetime(tbl.index), f"{ticker}: {title}", lookback_days=int(chart_lookback))


    if vis == "Gap Off-Open Return":
        c1, c2, c3 = st.columns(3)
        with c1:
            gap_thr = st.number_input("Gap threshold % (signed)", value=3.0, step=0.5, help="Positive = gap up only, Negative = gap down only, 0 = either")
        with c2:
            lookback_days = st.number_input("Lookback days", value=730, min_value=30, max_value=5000, step=30)
        with c3:
            minutes_list = st.multiselect("Minute marks", options=[1,2,3,5,10,15,30,60], default=[1,3,5,10,15], help="Compute returns from open to these minute marks")
        # Open tolerance (used for selecting the 09:30 bar and minute marks)
        open_tol = st.slider("Open tolerance (minutes)", min_value=0, max_value=10, value=5, help="If the exact minute is missing, use the first bar within this window")
        allow_nearest = st.checkbox("Allow nearest fallback beyond tolerance", value=True, help="If enabled, uses the closest bar even beyond the tolerance window and notes it in the Reason column.")
        # Enforce strict 09:30 selection and daily prev close for consistent gap calculations
        strict_open = True
        use_daily_prevclose = True
        show_extras = st.checkbox('Show extra daily columns (Prev Day Return, 3D RSI, Prev Day Earnings, # Day High, Consecutive)', value=gap_extra_cols_default, key='gap_extras')
        try:
            st.session_state['allow_nearest_offopen'] = bool(allow_nearest)
        except Exception:
            pass
        # Global assumption for naive minute timestamps in local files
        assume_naive = st.checkbox("Treat naive minute timestamps as New York time", value=False, help="Enable if your saved minute files store local NY times without timezone")
        try:
            st.session_state['assume_naive_ny_minutes'] = bool(assume_naive)
        except Exception:
            pass
        run = st.button("Run Gap Off-Open", key="run_gap_offopen")


        # Params fingerprint for caching
        display_tbl = None
        rendered = False
        _gap_params = {
            'sym': tkr_vis,
            'gap_thr': float(gap_thr),
            'lookback': int(lookback_days),
            'consec_up': int(consec_up),
            'wd': list(wd_sel),
            'rsi_gate': float(rsi_gate),
            'rsi_len': int(st.session_state.get('vis_rsi_len', 14)),
            'minutes': tuple(int(m) for m in (minutes_list or [])),
            'open_tol': int(open_tol),
            'allow_nearest': bool(allow_nearest),
            'strict_open': bool(strict_open),
            'use_daily_prevclose': bool(use_daily_prevclose),
        }
        try:
            if 'gap_offopen_last' in st.session_state and st.session_state['gap_offopen_last'] is not None:
                display_tbl = st.session_state['gap_offopen_last']
        except Exception:
            display_tbl = None

        def _add_gap_extras(tbl_in: pd.DataFrame, daily_df: pd.DataFrame) -> pd.DataFrame:
            if tbl_in is None or tbl_in.empty or daily_df is None or daily_df.empty:
                return tbl_in
            try:
                tbl = tbl_in.copy()
                d = daily_df.sort_values('Date').set_index('Date') if 'Date' in daily_df.columns else daily_df
                prev_cc = (d['Close'] / d['Close'].shift(1) - 1.0) * 100.0
                tbl['Prev Day Return'] = prev_cc.reindex(tbl.index)
                tbl['3D RSI'] = rsi(d['Close'].astype(float), 3).reindex(tbl.index)
                ed_list = _get_earnings_dates(tkr_vis) or []
                ed_dates = {pd.to_datetime(x).normalize().date() for x in ed_list if pd.notna(pd.to_datetime(x))}
                def _prev_day_has_earn(ts):
                    try:
                        ts_dt = pd.to_datetime(ts).normalize()
                        prev_date = (ts_dt - pd.Timedelta(days=1)).date()
                        # Mark true if the prior calendar day OR the same day had earnings
                        return (prev_date in ed_dates) or (ts_dt.date() in ed_dates)
                    except Exception:
                        return False
                tbl['Prev Day Earnings'] = [ _prev_day_has_earn(ts) for ts in tbl.index ]
                extra_tbl = _summary_table(daily_df, tbl.index)[['# Day High','Consecutive Up/Down']]
                tbl = tbl.join(extra_tbl, how='left')
                try:
                    st.caption(f"Earnings dates fetched: {len(ed_list)} (Polygon/Massive events -> financials -> yfinance)")
                except Exception:
                    pass
                return tbl
            except Exception:
                return tbl_in
        if run:
            # Helper: compute intraday returns at specified minutes from open for selected dates
            def _intraday_time_returns(sym: str, dates_idx: pd.DatetimeIndex, minute_marks=(1,3,5,10,15), preopen_min=6) -> pd.DataFrame:
                rows = []
                tz = 'America/New_York'
                prev_pref = st.session_state.get('preferred_provider') if 'preferred_provider' in st.session_state else None
                try:
                    st.session_state['preferred_provider'] = 'Polygon'
                except Exception:
                    pass
                diag = []
                assume_local_ny = bool(st.session_state.get('assume_naive_ny_minutes'))

                # ── Pre-load local minute parquet once (avoid re-reading 962K rows per day) ──
                _local_full = None
                _local_max_date = None
                try:
                    from pathlib import Path as _Path
                    _md = _Path(os.environ.get('PER_TICKER_MINUTE_DIR', ''))
                    _mfp = _md / f"{sym.upper().replace('.','_')}.parquet"
                    if _mfp.exists():
                        _ldf = pd.read_parquet(_mfp)
                        _ts_col = next((c for c in _ldf.columns if str(c).lower() in ('timestamp','t','date','datetime')), None)
                        if _ts_col:
                            _lts = pd.to_datetime(_ldf[_ts_col], errors='coerce', utc=True)
                            if pd.api.types.is_datetime64tz_dtype(_ldf[_ts_col]):
                                _lts = _ldf[_ts_col].dt.tz_convert('UTC')
                            _ldf = _ldf.assign(_idx=_lts.dt.tz_convert(tz)).set_index('_idx').sort_index()
                            from offopen_utils import normalize_ohlcv as _norm_ohlcv
                            _ldf = _norm_ohlcv(_ldf)
                            _local_full = _ldf
                            _local_max_date = _ldf.index.max().date()
                except Exception:
                    pass

                # ── Batch API fetch for dates beyond local data ───────────────────────────
                _batch_df = None
                try:
                    _all_dates = pd.to_datetime(dates_idx)
                    _min_req   = _all_dates.min().date()
                    _max_req   = _all_dates.max().date()
                    _need_api  = (_local_max_date is None) or (_max_req > _local_max_date)
                    if _need_api:
                        # Single Polygon call covers the whole date range (~720 bars/day × N days)
                        _api_start = str((_local_max_date + pd.DateOffset(days=1)).date()) if _local_max_date else str(_min_req)
                        _api_end   = str(_max_req)
                        _raw_batch = fetch_ohlc_with_fallback(sym, interval="1m", start=_api_start, end=_api_end)
                        if _raw_batch is not None and not _raw_batch.empty and isinstance(_raw_batch.index, pd.DatetimeIndex):
                            if _raw_batch.index.tz is None:
                                _raw_batch = _raw_batch.copy(); _raw_batch.index = _raw_batch.index.tz_localize('UTC').tz_convert(tz)
                            else:
                                _raw_batch = _raw_batch.copy(); _raw_batch.index = _raw_batch.index.tz_convert(tz)
                            _batch_df = _raw_batch
                except Exception:
                    pass

                for dt in pd.to_datetime(dates_idx):
                    day = dt.date()
                    start_ts = pd.Timestamp(year=day.year, month=day.month, day=day.day, hour=4,  minute=0,  tz=tz)
                    end_ts   = pd.Timestamp(year=day.year, month=day.month, day=day.day, hour=20, minute=10, tz=tz)

                    # 1. Try pre-loaded local data first (fast — already in memory)
                    intr = None
                    if _local_full is not None and _local_max_date is not None and day <= _local_max_date:
                        _seg = _local_full[(_local_full.index >= start_ts) & (_local_full.index <= end_ts)]
                        if not _seg.empty:
                            intr = _seg[['Open','High','Low','Close','Volume']].copy()

                    # 2. Use batch API data (also already in memory)
                    if (intr is None or intr.empty) and _batch_df is not None:
                        _seg = _batch_df[(_batch_df.index >= start_ts) & (_batch_df.index <= end_ts)]
                        if not _seg.empty:
                            intr = _seg[['Open','High','Low','Close'] + (['Volume'] if 'Volume' in _seg.columns else [])].copy()

                    # 3. Per-day API fallback (last resort — correct start/end as plain dates)
                    if intr is None or intr.empty:
                        intr = fetch_ohlc_with_fallback(sym, interval="1m",
                                                         start=str(day), end=str(day))
                    if intr is None or intr.empty:
                        intr = fetch_ohlc_with_fallback(sym, interval="5m",
                                                         start=str(day), end=str(day))
                    prov = st.session_state.get('last_fetch_provider', '-')
                    if intr is None or intr.empty or not isinstance(intr.index, pd.DatetimeIndex):
                        rows.append({"Date": pd.Timestamp(day), **{f"{m} Min Return": np.nan for m in minute_marks}, "9:24 to Open": np.nan})
                        diag.append({'date': str(day), 'provider': prov, 'rows': 0})
                        continue
                    intr = normalize_ohlcv(intr)
                    try:
                        idx_ny = intr.index.tz_convert(tz) if intr.index.tz is not None else intr.index.tz_localize('UTC').tz_convert(tz)
                    except Exception:
                        idx_ny = intr.index
                    intr = intr.copy(); intr.index = idx_ny
                    diag.append({'date': str(day), 'provider': prov, 'rows': int(len(intr)), 'start': str(intr.index.min()), 'end': str(intr.index.max())})
                    # Locate market open (09:30 ET) robustly for 1m or 5m bars
                    t_open = pd.Timestamp(year=day.year, month=day.month, day=day.day, hour=9, minute=30, tz=tz)
                    # Prefer first bar at/after 09:30 within +open_tol minutes
                    win_open = intr[(intr.index >= t_open) & (intr.index <= t_open + pd.Timedelta(minutes=int(open_tol)))]
                    if not win_open.empty:
                        io = 0
                        p_open = float(win_open['Close'].iloc[io]) if 'Close' in win_open.columns else float('nan')
                        open_dt = win_open.index[io]
                    else:
                        # Fallback: nearest within tolerance
                        pos = intr.index.get_indexer([t_open], method='nearest')
                        if not (pos.size and pos[0] != -1):
                            rows.append({"Date": pd.Timestamp(day), **{f"{m} Min Return": np.nan for m in minute_marks}, "9:24 to Open": np.nan})
                            continue
                        io = int(pos[0])
                        dt_found = intr.index[io]
                        if abs((dt_found - t_open).total_seconds()) > 60*int(open_tol):
                            rows.append({"Date": pd.Timestamp(day), **{f"{m} Min Return": np.nan for m in minute_marks}, "9:24 to Open": np.nan})
                            continue
                        p_open = float(intr['Close'].iloc[io]) if 'Close' in intr.columns else float('nan')
                        open_dt = dt_found
                    # Pre-open: choose last bar before 09:30 within 15 minutes window
                    p_pre = np.nan
                    try:
                        pre_win = intr[(intr.index < t_open) & (intr.index >= t_open - pd.Timedelta(minutes=15))]
                        if not pre_win.empty:
                            p_pre = float(pre_win['Close'].iloc[-1])
                    except Exception:
                        p_pre = np.nan
                    out = {"Date": pd.Timestamp(day)}
                    out["9:24 to Open"] = (p_open / p_pre - 1.0) * 100.0 if (not np.isnan(p_pre) and p_pre != 0) else np.nan
                    for m in minute_marks:
                        t_m = t_open + pd.Timedelta(minutes=int(m))
                        # Prefer first bar at/after t_m within +open_tol minutes
                        win_m = intr[(intr.index >= t_m) & (intr.index <= t_m + pd.Timedelta(minutes=int(open_tol)))]
                        if not win_m.empty:
                            p_m = float(win_m['Close'].iloc[0]) if 'Close' in win_m.columns else float('nan')
                            out[f"{int(m)} Min Return"] = (p_m / p_open - 1.0) * 100.0 if p_open else np.nan
                            continue
                        # Fallback: nearest within tolerance
                        im = intr.index.get_indexer([t_m], method='nearest')
                        if im.size and im[0] != -1:
                            idxm = int(im[0])
                            try:
                                dtm = intr.index[idxm]
                                if abs((dtm - t_m).total_seconds()) <= 60*int(open_tol):
                                    p_m = float(intr['Close'].iloc[idxm])
                                    out[f"{int(m)} Min Return"] = (p_m / p_open - 1.0) * 100.0 if p_open else np.nan
                                else:
                                    out[f"{int(m)} Min Return"] = np.nan
                            except Exception:
                                out[f"{int(m)} Min Return"] = np.nan
                        else:
                            out[f"{int(m)} Min Return"] = np.nan
                    rows.append(out)
                # store diagnostics
                try:
                    st.session_state['gap_intra_diag'] = diag
                except Exception:
                    pass
                finally:
                    # restore provider preference
                    try:
                        if prev_pref is None:
                            if 'preferred_provider' in st.session_state:
                                del st.session_state['preferred_provider']
                        else:
                            st.session_state['preferred_provider'] = prev_pref
                    except Exception:
                        pass
                df_ret = pd.DataFrame(rows).set_index('Date') if rows else pd.DataFrame()
                return df_ret
            df_d = _load_daily_df(tkr_vis)
            if df_d is None or df_d.empty or not set(["Open","Close"]).issubset(df_d.columns):
                st.warning("Daily data unavailable or missing Open/Close.")
            else:
                df_d = df_d.sort_values("Date").set_index("Date")
                df_d = df_d.last(f"{int(lookback_days)}D") if isinstance(df_d.index, pd.DatetimeIndex) else df_d.tail(int(lookback_days))
                prevc = df_d["Close"].shift(1)
                gap_pct = (df_d["Open"] / prevc - 1.0) * 100.0
                # Light prefilter by gap sign only; final threshold applied after minute recompute
                thr_val = float(gap_thr)
                if thr_val > 0:
                    mask_gap = gap_pct >= 0
                elif thr_val < 0:
                    mask_gap = gap_pct <= 0
                else:
                    mask_gap = gap_pct.notna()
                mask = mask_gap & _apply_common_filters(df_d)
                sel = df_d[mask]
                day_ret = (sel["Close"] / sel["Open"] - 1.0) * 100.0
                stats_msg = f"Signals: {len(day_ret)} | Mean off-open %: {day_ret.mean():.2f} | Median: {day_ret.median():.2f}"
                st.caption(stats_msg)
                try:
                    st.session_state["last_scan_stats_caption"] = stats_msg
                except Exception:
                    pass
                # Default toggles so outer references are safe even if no rows
                show_marks = False
                send_marks = False
                missing_dates = pd.Index([])
                if len(day_ret) > 0:
                    daily_full = _load_daily_df(tkr_vis)
                    # Build time-based table (minutes), not day-forward returns by default
                    base_idx = pd.to_datetime(day_ret.index)
                    tbl_time = pd.DataFrame(index=base_idx)
                    # Close to Open (gap %)
                    try:
                        tbl_time['Close to Open'] = gap_pct.reindex(base_idx)
                    except Exception:
                        tbl_time['Close to Open'] = np.nan
                    # Intraday minute returns and 9:24?Open
                    chosen_minutes = tuple(int(m) for m in (minutes_list or [1,3,5,10,15]))
                    using_full_minutes = False
                    try:
                        minute_table = _compute_offopen_table_from_minutes(tkr_vis, base_idx, chosen_minutes, int(open_tol), bool(allow_nearest and not strict_open))
                        using_full_minutes = minute_table is not None and not minute_table.empty
                        if using_full_minutes:
                            mt = minute_table.copy()
                            mt.index = pd.to_datetime(mt.index).normalize()
                            align_idx = pd.to_datetime(base_idx).normalize()
                            intr_cols = mt.reindex(align_idx)
                            intr_cols.index = base_idx
                        else:
                            # Local minute data is stale or missing — batch-fetch from Polygon
                            try:
                                _local_mp = Path(os.environ.get('PER_TICKER_MINUTE_DIR','')) / f"{tkr_vis.upper().replace('.','_')}.parquet"
                                if _local_mp.exists():
                                    _ldf_check = pd.read_parquet(_local_mp, columns=['Timestamp'])
                                    _local_cutoff = pd.to_datetime(_ldf_check['Timestamp']).max()
                                    if hasattr(_local_cutoff, 'tz_convert'):
                                        _local_cutoff = _local_cutoff.tz_convert(None)
                                    st.caption(f"📡 Local minute data through **{_local_cutoff.date()}** — fetching recent dates from Polygon")
                                else:
                                    st.caption("📡 No local minute data — fetching from Polygon")
                            except Exception:
                                pass
                            intr_cols = _intraday_time_returns(tkr_vis, base_idx, minute_marks=chosen_minutes)
                        meta_cols = ['Open Timestamp (ET)','Open Price','Prev Close Timestamp (ET)','Prev Close Price','09:24 Timestamp (ET)','9:24 to Open']
                        cols_order = meta_cols + [f"{int(m)} Min Return" for m in chosen_minutes] + ['Reason']
                        if intr_cols is None or intr_cols.empty:
                            intr_cols = pd.DataFrame(index=tbl_time.index, columns=cols_order)
                        else:
                            for col in cols_order:
                                if col not in intr_cols.columns:
                                    intr_cols[col] = np.nan if col not in ('Reason',) else ''
                            intr_cols = intr_cols[cols_order]
                        tbl_time = tbl_time.join(intr_cols, how='left')
                        if using_full_minutes:
                            try:
                                minute_open = intr_cols['Open Price'].astype(float)
                                prev_close = intr_cols['Prev Close Price'].astype(float)
                                new_gap = (minute_open / prev_close - 1.0) * 100.0
                                tbl_time['Close to Open'] = new_gap
                                gap_pct = new_gap.reindex(base_idx)
                                close_series = sel['Close'].reindex(base_idx).astype(float)
                                day_ret = (close_series / minute_open - 1.0) * 100.0
                            except Exception:
                                pass
                        if using_full_minutes:
                            minute_cols = ['9:24 to Open'] + [f"{int(m)} Min Return" for m in chosen_minutes]
                            # keep missing_dates empty since table already sourced from full minutes
                            missing_dates = pd.Index([])
                            try:
                                diag_rows = [{'date': str(idx.date()), 'provider': 'local-minute', 'reason': intr_cols.at[idx, 'Reason'] if 'Reason' in intr_cols.columns else ''} for idx in intr_cols.index]
                                st.session_state['gap_intra_diag'] = diag_rows
                            except Exception:
                                pass
                        else:
                            minute_cols = ['9:24 to Open'] + [f"{int(m)} Min Return" for m in chosen_minutes]
                            if all(c in tbl_time.columns for c in minute_cols):
                                missing_dates = tbl_time.index[tbl_time[minute_cols].isna().all(axis=1)]
                            else:
                                missing_dates = pd.Index([])
                    except Exception:
                        # On any error, still create empty columns so they appear
                        try:
                            chosen_minutes = tuple(int(m) for m in (minutes_list or [1,3,5,10,15]))
                            cols_order = ['Open Timestamp (ET)','Open Price','Prev Close Timestamp (ET)','Prev Close Price','09:24 Timestamp (ET)','9:24 to Open'] + [f"{int(m)} Min Return" for m in chosen_minutes] + ['Reason']
                            intr_cols = pd.DataFrame(index=base_idx, columns=cols_order)
                            tbl_time = tbl_time.join(intr_cols, how='left')
                        except Exception:
                            pass
                        minute_cols = ['9:24 to Open'] + [f"{int(m)} Min Return" for m in chosen_minutes]
                        missing_dates = pd.Index([])
                    # Open to Close (same day)
                    try:
                        tbl_time['Open to Close'] = day_ret.reindex(base_idx)
                    except Exception:
                        tbl_time['Open to Close'] = np.nan

                    # Optionally override prev close with prior daily close for thresholding
                    if use_daily_prevclose and daily_full is not None and not daily_full.empty:
                        try:
                            d_daily = daily_full.sort_values('Date').set_index('Date') if 'Date' in daily_full.columns else daily_full
                            prev_close_daily = d_daily['Close'].shift(1)
                            tbl_time['Prev Close Price'] = prev_close_daily.reindex(tbl_time.index)
                            # keep original minute timestamp but overwrite price; timestamp stays as minute prev close if present
                            if 'Open Price' in tbl_time.columns:
                                opv = pd.to_numeric(tbl_time['Open Price'], errors='coerce')
                                pcv = pd.to_numeric(tbl_time['Prev Close Price'], errors='coerce')
                                tbl_time['Close to Open'] = (opv / pcv - 1.0) * 100.0
                        except Exception:
                            pass

                    # If minute gap is missing, backfill with daily gap to avoid dropping rows
                    try:
                        if 'gap_pct' in locals() and tbl_time['Close to Open'].isna().any():
                            daily_gap = gap_pct.reindex(tbl_time.index)
                            mask_na = tbl_time['Close to Open'].isna()
                            tbl_time.loc[mask_na, 'Close to Open'] = daily_gap[mask_na]
                    except Exception:
                        pass

                    # Optional extras
                    if show_extras:
                        tbl_time = _add_gap_extras(tbl_time, daily_full)

                    # Re-apply gap threshold using the (possibly minute-adjusted) Close to Open values
                    try:
                        cur_gap = tbl_time.get('Close to Open')
                        if cur_gap is not None:
                            thr_val = float(gap_thr)
                            if thr_val > 0:
                                keep = cur_gap >= thr_val
                            elif thr_val < 0:
                                keep = cur_gap <= thr_val
                            else:
                                keep = cur_gap.notna()
                            tbl_time = tbl_time.loc[keep]
                            day_ret = day_ret.reindex(tbl_time.index)
                    except Exception:
                        pass

                    try:
                        st.session_state['gap_offopen_last'] = tbl_time.copy()
                        display_tbl = tbl_time.copy()
                    except Exception:
                        pass

                    _set_vis_cache('gap_offopen', _gap_params, tbl_time)

                    # Show a clear message if nothing matched instead of rendering nothing
                    if tbl_time is None or tbl_time.empty:
                        miss_ct = len(missing_dates) if 'missing_dates' in locals() else 0
                        st.warning(f"No gap off-open rows matched (gap threshold/filters). Missing minute dates: {miss_ct}")
                    else:
                        _render_gap_results(tkr_vis, tbl_time)
                        # remember current table for chart rendering outside button reruns
                        display_tbl = tbl_time.copy()
                        rendered = True

            cached = _get_vis_cache('gap_offopen', _gap_params)
            if not rendered and cached is not None and not cached.empty:
                if show_extras:
                    cached = _add_gap_extras(cached, _load_daily_df(tkr_vis))
                _render_gap_results(tkr_vis, cached)
                rendered = True
                try:
                    display_tbl = cached.copy()
                except Exception:
                    pass
            # Fall back to last computed table so toggles (chart/extras) don't blank the view
            if not rendered:
                try:
                    last_tbl = st.session_state.get('gap_offopen_last')
                    if last_tbl is not None and not last_tbl.empty:
                        if show_extras:
                            last_tbl = _add_gap_extras(last_tbl, _load_daily_df(tkr_vis))
                        _render_gap_results(tkr_vis, last_tbl)
                        rendered = True
                        try:
                            display_tbl = last_tbl.copy()
                        except Exception:
                            pass
                except Exception:
                    pass
            # Chart rendering based on whichever table is available (current, cached, or last)
            if display_tbl is None:
                try:
                    display_tbl = st.session_state.get('gap_offopen_last')
                except Exception:
                    display_tbl = None
            _render_chart_for_tbl("Gap Off-Open occurrences", display_tbl, tkr_vis, default_lookback=360)

    # Minute Cache Inspector removed
    if vis == "Prev Day CC→Close Overnight":
        c1, c2 = st.columns(2)
        with c1:
            cc_thr = st.number_input(
                "Prev close-close % (sign-aware)",
                value=2.0,
                step=0.5,
                help="Positive = require up move ≥ threshold; negative = require down move ≤ threshold."
            )
        with c2:
            relative = "/" in tkr_vis
        run = st.button("Run Prev CC→Close", key="run_prevcc_ov")
        ret_cached = st.session_state.get('prevcc_ov_ret')

        # Earnings diagnostic — lazy (button-triggered) so it doesn't slow every render
        with st.expander("Earnings dates (diagnostic)", expanded=False):
            if st.button("Load earnings dates", key="earn_diag_btn"):
                try:
                    eds = _get_earnings_dates(tkr_vis, limit=200)
                    st.caption(f"Total earnings dates loaded: {len(eds)}")
                    st.write(pd.Series(sorted({pd.to_datetime(x).date() for x in eds})).tail(50))
                except Exception as e:
                    st.caption(f"Earnings diagnostic error: {e}")
            else:
                st.caption("Click to load on demand — avoids slowing page render.")

        def _render_prevcc_table(tbl: pd.DataFrame, ov_series: pd.Series, cc_series: pd.Series):
            """Render the signal table with key columns up front."""
            if tbl is None or tbl.empty:
                return
            tbl = tbl.copy()
            # Enrich
            try: tbl["Close to Next Open %"] = ov_series.reindex(tbl.index)
            except Exception: pass
            try: tbl["Prev Close to Close"] = cc_series.reindex(tbl.index)
            except Exception: pass
            try: tbl["Prev Day Earnings"] = _earnings_flag_for_dates(
                    (tkr_vis.split("/",1)[0]), tbl.index)
            except Exception: pass
            # Column ordering: key metrics first
            priority = ["Prev Close to Close", "Close to Next Open %", "Close to Open",
                        "Open to Close", "1 Day Return", "3 Day Return", "5 Day Return",
                        "10 Day Return", "Prev Day Earnings", "All Time High",
                        "# Day High", "Consecutive Up/Down"]
            ordered = [c for c in priority if c in tbl.columns]
            rest = [c for c in tbl.columns if c not in ordered]
            tbl = tbl[ordered + rest]
            # Style
            pct_cols = [c for c in tbl.columns if any(
                c.endswith(x) for x in ("Return", "Close to Close", "to Next Open %",
                                        "Close to Open", "Open to Close"))]
            try:
                fmt = {c: "{:+.2f}%" for c in pct_cols if c in tbl.columns}
                styled = (tbl.style
                    .format(fmt)
                    .map(
                        lambda v: "background-color:#2ecc40;color:#000" if isinstance(v, (int, float)) and v >= 0
                        else ("background-color:#ff6961;color:#000" if isinstance(v, (int, float)) else ""),
                        subset=[c for c in pct_cols if c in tbl.columns]
                    ))
                st.write(styled)
            except Exception:
                st.dataframe(tbl, use_container_width=True)
            try: _render_scan_stats(tbl)
            except Exception: pass

        if run:
            base_sym, rel_sym = (tkr_vis.split("/",1) + [""])[:2]
            df_d = _load_daily_df(base_sym)  # load once — reused for chart, table, columns
            if df_d is None or df_d.empty:
                st.warning("Daily data unavailable for base.")
            else:
                d = df_d.sort_values("Date").set_index("Date")
                # Strip tz so all index arithmetic is consistent
                try:
                    if d.index.tzinfo is not None: d.index = d.index.tz_localize(None)
                except Exception: pass
                cc = pd.to_numeric((d["Close"] / d["Close"].shift(1) - 1.0) * 100.0, errors="coerce")
                base_mask = cc.notna() & _apply_common_filters(d)
                if cc_thr >= 0:
                    sel_mask = base_mask & (cc >= float(cc_thr))
                else:
                    sel_mask = base_mask & (cc <= float(cc_thr))
                next_open = d["Open"].shift(-1)
                ov = (next_open / d["Close"] - 1.0) * 100.0
                ret = ov[sel_mask]
                if relative and rel_sym:
                    d2 = _load_daily_df(rel_sym)
                    if d2 is not None and not d2.empty:
                        d2 = d2.sort_values("Date").set_index("Date")
                        ov2 = (d2["Open"].shift(-1) / d2["Close"] - 1.0) * 100.0
                        ret = (ret - ov2.reindex(ret.index))
                st.caption(f"Signals: {len(ret)} | Mean overnight %: {ret.mean():.2f} | Median: {ret.median():.2f}")
                if len(ret) > 0:
                    try:
                        st.session_state['prevcc_ov_ret'] = ret
                        st.session_state['prevcc_ov_cc'] = cc
                        st.session_state['prevcc_ov_ov'] = ov
                        ret_cached = ret
                    except Exception:
                        pass
                    # Price chart with signal markers — reuse df_d
                    try:
                        _viz_price_marks(df_d, ret.index, f"{base_sym}: Prev CC→Close Overnight occurrences")
                    except Exception:
                        pass
                    # Build results table directly — don't rely on _summary_table's index join
                    try:
                        tbl = pd.DataFrame(index=ret.index)
                        tbl.index.name = "Date"
                        tbl["Prev Close to Close"] = cc.reindex(ret.index)
                        tbl["Close to Next Open %"] = ov.reindex(ret.index)
                        # Forward returns from d (already tz-stripped)
                        for label, shift in [("1 Day Return", 1), ("3 Day Return", 3),
                                             ("5 Day Return", 5), ("10 Day Return", 10)]:
                            try:
                                tbl[label] = ((d["Close"].shift(-shift) / d["Close"] - 1.0) * 100.0).reindex(ret.index)
                            except Exception:
                                pass
                        try:
                            tbl["Close to Open"] = ((d["Open"] / d["Close"].shift(1) - 1.0) * 100.0).reindex(ret.index)
                        except Exception:
                            pass
                        try:
                            tbl["Open to Close"] = ((d["Close"] / d["Open"] - 1.0) * 100.0).reindex(ret.index)
                        except Exception:
                            pass
                        tbl = tbl.sort_index(ascending=False)
                        _render_prevcc_table(tbl, ov, cc)
                    except Exception as _e_tbl:
                        st.caption(f"Table error: {_e_tbl}")
                        try:
                            st.dataframe(ret.to_frame("Next Open %").sort_index(ascending=False),
                                         use_container_width=True)
                        except Exception:
                            pass

                    show_marks = st.checkbox("Show price chart with occurrences", value=False, key="marks_prevcc_ov")
                    send_marks = st.checkbox("Send these dates to Chart highlights", value=False, key="send_prevcc_ov")
                    if show_marks:
                        _viz_price_marks(df_d, ret.index, f"{base_sym}: Prev CC→Close Overnight occurrences")
                    if send_marks:
                        try:
                            st.session_state['scan_highlights'] = [str(pd.to_datetime(x).date()) for x in ret.index]
                            st.success("Dates sent to Chart tab (toggle 'Show scan highlights').")
                        except Exception:
                            pass
                else:
                    st.info("No matching days for the selected threshold/filters.")
        elif ret_cached is not None and len(ret_cached) > 0:
            base_sym = (tkr_vis.split("/",1) + [""])[0]
            df_d = _load_daily_df(base_sym)
            cc_cached = st.session_state.get('prevcc_ov_cc')
            ov_cached = st.session_state.get('prevcc_ov_ov')
            # Recompute cc/ov if not in session state
            if (cc_cached is None or ov_cached is None) and df_d is not None and not df_d.empty:
                try:
                    d_tmp = df_d.sort_values("Date").set_index("Date")
                    try:
                        if d_tmp.index.tzinfo is not None: d_tmp.index = d_tmp.index.tz_localize(None)
                    except Exception: pass
                    cc_cached = pd.to_numeric((d_tmp["Close"] / d_tmp["Close"].shift(1) - 1.0) * 100.0, errors="coerce")
                    ov_cached = ((d_tmp["Open"].shift(-1) / d_tmp["Close"] - 1.0) * 100.0)
                except Exception:
                    cc_cached = pd.Series(dtype=float)
                    ov_cached = pd.Series(dtype=float)
            # Build cached results table directly (no _summary_table index join)
            try:
                tbl_c = pd.DataFrame(index=ret_cached.index)
                tbl_c.index.name = "Date"
                if cc_cached is not None:
                    tbl_c["Prev Close to Close"] = cc_cached.reindex(ret_cached.index)
                if ov_cached is not None:
                    tbl_c["Close to Next Open %"] = ov_cached.reindex(ret_cached.index)
                if df_d is not None and not df_d.empty:
                    d_c = df_d.sort_values("Date").set_index("Date")
                    try:
                        if d_c.index.tzinfo is not None: d_c.index = d_c.index.tz_localize(None)
                    except Exception: pass
                    for label, shift in [("1 Day Return",1),("3 Day Return",3),("5 Day Return",5),("10 Day Return",10)]:
                        try:
                            tbl_c[label] = ((d_c["Close"].shift(-shift)/d_c["Close"]-1.0)*100.0).reindex(ret_cached.index)
                        except Exception: pass
                    try:
                        tbl_c["Close to Open"] = ((d_c["Open"]/d_c["Close"].shift(1)-1.0)*100.0).reindex(ret_cached.index)
                        tbl_c["Open to Close"] = ((d_c["Close"]/d_c["Open"]-1.0)*100.0).reindex(ret_cached.index)
                    except Exception: pass
                if not tbl_c.empty:
                    tbl_c = tbl_c.sort_index(ascending=False)
                    _render_prevcc_table(tbl_c,
                                         ov_cached if ov_cached is not None else pd.Series(dtype=float),
                                         cc_cached if cc_cached is not None else pd.Series(dtype=float))
            except Exception as _e_ct:
                st.caption(f"Cached table error: {_e_ct}")
                try:
                    st.dataframe(ret_cached.to_frame("Next Open %").sort_index(ascending=False), use_container_width=True)
                except Exception: pass
            show_marks = st.checkbox("Show price chart with occurrences", value=False, key="marks_prevcc_ov_cached")
            if show_marks and df_d is not None:
                _viz_price_marks(df_d, ret_cached.index, f"{base_sym}: Prev CC → Overnight occurrences")

    # Ensure intraday fetch helper is defined before use in the visualizer
    try:
        fetch_ohlc_with_fallback
    except NameError:
        def fetch_ohlc_with_fallback(
            ticker: str,
            *,
            interval: str,
            period: Optional[str] = None,
            start: Optional[str] = None,
            end: Optional[str] = None,
        ):
            try:
                force = bool(st.session_state.get('force_refresh'))
            except Exception:
                force = False
            first_buster = (str(time.time()) if force else None)
            df = _fetch_ohlc_cached(ticker, interval=interval, period=period, start=start, end=end, cache_buster=first_buster)
            try:
                import pandas as _pd
                is_empty = (df is None) or (isinstance(df, _pd.DataFrame) and df.empty)
            except Exception:
                is_empty = df is None
            if not is_empty:
                return df
            buster = str(time.time())
            return _fetch_ohlc_cached(ticker, interval=interval, period=period, start=start, end=end, cache_buster=buster)

    if vis == "Intraday Time Move → Close":
        c1, c2 = st.columns(2)
        with c1:
            times_15 = [f"{h:02d}:{m:02d}" for h in range(9, 16) for m in (30,45,0,15) if not (h==9 and m<45)]
            times_15 = sorted(set(times_15))
            hhmm = st.selectbox("Time of day (HH:MM, NY)", options=times_15, index=times_15.index("10:30") if "10:30" in times_15 else 0)
        with c2:
            move_thr = st.number_input("Move threshold % (sign-aware)", value=2.0, step=0.5)
        anchor_from = st.selectbox("Move from", ["Open", "Prev Close"], index=0)
        c4, c5 = st.columns(2)
        with c4:
            lookback_days_intra = st.number_input("Lookback days", value=200, min_value=10, max_value=365, step=10)
        with c5:
            time_tol_min = st.number_input("Time tolerance (minutes)", value=30, min_value=0, max_value=60, step=1)
        run = st.button("Run Intraday Visualizer", key="run_intra_time")
        s_tm_cached = st.session_state.get('intraday_s_tm')
        if run:
            daily_df = _load_daily_df(tkr_vis)
            # Prefer local minute parquet, fall back to HTTP
            end_ts = pd.Timestamp.now(tz="America/New_York")
            start_ts = end_ts - pd.Timedelta(days=int(lookback_days_intra))
            df_intra = _load_minute_local(tkr_vis, start_ts, end_ts, assume_naive_ny=True)
            if df_intra is None or getattr(df_intra, "empty", True):
                try:
                    df_intra = fetch_ohlc_with_fallback(
                        tkr_vis,
                        interval="5m",
                        start=start_ts.isoformat(),
                        end=end_ts.isoformat(),
                    )
                except Exception:
                    df_intra = fetch_ohlc_with_fallback(tkr_vis, interval="5m", period="365d")
                if df_intra is None or getattr(df_intra, "empty", True):
                    try:
                        df_intra = fetch_ohlc_with_fallback(
                            tkr_vis,
                            interval="15m",
                            start=start_ts.isoformat(),
                            end=end_ts.isoformat(),
                        )
                    except Exception:
                        df_intra = fetch_ohlc_with_fallback(tkr_vis, interval="15m", period="365d")
            if df_intra is None or df_intra.empty or not isinstance(df_intra.index, pd.DatetimeIndex):
                st.warning("Intraday data unavailable.")
            else:
                df = normalize_ohlcv(df_intra)
                try:
                    idx_ny = df.index.tz_convert('America/New_York') if df.index.tz is not None else df.index.tz_localize('UTC').tz_convert('America/New_York')
                except Exception:
                    idx_ny = df.index
                df = df.copy(); df.index = idx_ny
                days = sorted(set(df.index.date))
                if df.empty or not days:
                    st.warning(f"Intraday data loaded but no rows/dates found (rows={len(df)}, days={len(days)}). Check minute source.")
                else:
                    st.caption(f"Intraday rows: {len(df)} | Days: {len(days)}")
                rows = []
                all_rows = []
                anchors_used = {"Open": 0, "Prev Close": 0}
                for day in days:
                    try:
                        # Use .date comparison to avoid TZ-aware vs TZ-naive mismatch
                        mask = pd.Series(df.index.date == day, index=df.index)
                        day_df = df.loc[mask]
                    except Exception:
                        day_df = pd.DataFrame()
                    if day_df.empty:
                        continue
                    day_df = day_df.sort_index()
                    # Anchor = official 09:30 open (nearest within 10 minutes); fallback to first row
                    try:
                        open_ts = pd.Timestamp(year=day.year, month=day.month, day=day.day, hour=9, minute=30, tz=day_df.index.tz)
                        # Prefer first bar at/after 9:30
                        after_open = day_df[day_df.index >= open_ts]
                        if not after_open.empty:
                            open_idx = day_df.index.get_loc(after_open.index[0])
                        else:
                            # fallback to nearest to 9:30
                            near_open_pos = day_df.index.get_indexer([open_ts], method='nearest')
                            open_idx = int(near_open_pos[0]) if near_open_pos.size and near_open_pos[0] != -1 else 0
                        o = float(day_df["Open"].iloc[open_idx])
                    except Exception:
                        o = float('nan')
                    # Prior close (using daily, date-normalized)
                    prev_close_val = None
                    if daily_df is not None and not daily_df.empty:
                        try:
                            d_sorted = daily_df.copy()
                            d_sorted["Date"] = pd.to_datetime(d_sorted["Date"], errors="coerce").dt.normalize()
                            d_sorted = d_sorted.dropna(subset=["Date"]).set_index("Date").sort_index()
                            prev_close_series = d_sorted["Close"].shift(1)
                            prev_close_val = float(prev_close_series.reindex([pd.Timestamp(day)]).iloc[0])
                        except Exception:
                            prev_close_val = None
                    # Fallback: use previous day's last intraday close if daily missing
                    if (prev_close_val is None or np.isnan(prev_close_val)):
                        try:
                            prev_day = (pd.Timestamp(day) - pd.Timedelta(days=1)).date()
                            prev_mask = pd.Series(df.index.date == prev_day, index=df.index)
                            prev_df = df.loc[prev_mask]
                            if prev_df is not None and not prev_df.empty:
                                prev_close_val = float(prev_df["Close"].iloc[-1])
                        except Exception:
                            prev_close_val = None
                    try:
                        hh, mm = [int(x) for x in str(hhmm).split(":",1)]
                        ts = pd.Timestamp(year=day.year, month=day.month, day=day.day, hour=hh, minute=mm, tz='America/New_York')
                        pos = day_df.index.get_indexer([ts], method='nearest')
                        i = int(pos[0]) if pos.size and pos[0] != -1 else None
                    except Exception:
                        i = None
                    if i is None:
                        continue
                    px_t = float(day_df["Close"].iloc[i])
                    anchor = o if anchor_from == "Open" else prev_close_val
                    # Robust fallback: if anchor missing, use first/last close in the day as a proxy
                    try:
                        if anchor is None or (isinstance(anchor, float) and np.isnan(anchor)) or anchor == 0:
                            if anchor_from == "Open":
                                anchor = float(day_df["Close"].iloc[0])
                            else:
                                anchor = float(day_df["Close"].iloc[0])
                    except Exception:
                        pass
                    if anchor is None or (isinstance(anchor, float) and np.isnan(anchor)) or anchor == 0:
                        continue
                    anchors_used[anchor_from] = anchors_used.get(anchor_from, 0) + 1
                    move = (px_t / anchor - 1.0) * 100.0
                    # Track nearest-bar offset
                    try:
                        minutes_off = abs((day_df.index[i] - ts).total_seconds())/60.0
                    except Exception:
                        minutes_off = np.nan
                    # track all moves for diagnostics
                    try:
                        all_rows.append({
                            "Date": pd.Timestamp(day),
                            "Anchor": anchor_from,
                            "Anchor_Price": anchor,
                            "Price_at_Time": px_t,
                            "Move_%": move,
                            "Close": float(day_df["Close"].iloc[-1]),
                            "Minutes_from_target": minutes_off,
                        })
                    except Exception:
                        pass
                    near_ok = True
                    try:
                        if time_tol_min is not None and minutes_off is not None and not np.isnan(minutes_off):
                            near_ok = minutes_off <= float(time_tol_min)
                    except Exception:
                        near_ok = True
                    passed = near_ok and (np.sign(move_thr) * move >= abs(move_thr))
                    if passed:
                        c = float(day_df["Close"].iloc[-1])
                        r = (c / px_t - 1.0) * 100.0
                        rows.append({
                            "Date": pd.Timestamp(day),
                            "Anchor": anchor_from,
                            "Anchor_Price": anchor,
                            "Price_at_Time": px_t,
                            "Move_%": move,
                            "Close": c,
                            "Return_to_Close_%": r,
                            "Minutes_from_target": minutes_off,
                        })
                df_all = pd.DataFrame(all_rows).set_index("Date") if all_rows else pd.DataFrame()
                if not rows:
                    st.info("No matching intraday moves for the selected threshold/filters.")
                    try:
                        st.caption(f"Anchors used — Open: {anchors_used.get('Open',0)}, Prev Close: {anchors_used.get('Prev Close',0)}")
                    except Exception:
                        pass
                    if not df_all.empty:
                        df_all = df_all.sort_index(ascending=False)
                        st.caption(f"Top moves (showing latest {min(len(df_all),50)} days) for context:")
                        st.dataframe(df_all.head(50), use_container_width=True)
                        try:
                            st.caption(f"Max Move % observed: {df_all['Move_%'].max():.2f}% | Min: {df_all['Move_%'].min():.2f}% | Min/Max minutes offset: {df_all['Minutes_from_target'].min():.1f}/{df_all['Minutes_from_target'].max():.1f}")
                        except Exception:
                            pass
                    else:
                        st.caption("No intraday rows to summarize (minute data may be missing for this window).")
                else:
                    ev = pd.DataFrame(rows).set_index("Date").sort_index(ascending=False)
                    try:
                        ev["Prev Day Earnings"] = _earnings_flag_for_dates(tkr_vis, ev.index)
                    except Exception:
                        pass
                    st.caption(f"Signals: {len(ev)} | Mean % to close: {ev['Return_to_Close_%'].mean():.2f} | Median: {ev['Return_to_Close_%'].median():.2f}")
                    try:
                        _render_scan_stats(ev)
                    except Exception:
                        pass
                    st.dataframe(ev, use_container_width=True)
                    try:
                        st.session_state['intraday_s_tm'] = ev['Return_to_Close_%']
                        s_tm_cached = ev['Return_to_Close_%']
                    except Exception:
                        pass
                    with st.expander("Histogram / Heatmap", expanded=False):
                        _viz_hist_and_heatmap(ev['Return_to_Close_%'], title_prefix="Intraday ")

                    show_marks = st.checkbox("Show price chart with occurrences", value=False, key="marks_intraday_time")
                    send_marks = st.checkbox("Send these dates to Chart highlights", value=False, key="send_intraday_time")
                    if show_marks:
                        _viz_price_marks(_load_daily_df(tkr_vis), ev.index, f"{tkr_vis}: Intraday time→close occurrences")
                    if send_marks:
                        try:
                            st.session_state['scan_highlights'] = [str(pd.to_datetime(d).date()) for d in ev.index]
                            st.success("Dates sent to Chart tab (toggle 'Show scan highlights').")
                        except Exception:
                            pass
        elif s_tm_cached is not None and len(s_tm_cached) > 0:
            # Render cached results so toggles don't blank the view
            tbl = _summary_table(_load_daily_df(tkr_vis), s_tm_cached.index)
            if not tbl.empty:
                st.dataframe(tbl, use_container_width=True)
            show_marks = st.checkbox("Show price chart with occurrences", value=False, key="marks_intraday_time_cached")
            if show_marks:
                _viz_price_marks(_load_daily_df(tkr_vis), s_tm_cached.index, f"{tkr_vis}: Intraday time→close occurrences")

    if vis == "Earnings Long-Term Returns":
        c1, c2 = st.columns(2)
        with c1:
            move_type = st.selectbox("Move filter", ["Gap %", "Day %", "Close-Close %"], index=0)
        with c2:
            move_thr = st.number_input("Move threshold % (|x|)", value=3.0, step=0.5)
        horizon = st.select_slider("Horizon (days)", options=[5,10,21,42,63,84,126,189,252], value=126)
        run = st.button("Run Earnings LT", key="run_earn_lt")
        if run:
            df_d = _load_daily_df(tkr_vis)
            if df_d is None or df_d.empty:
                st.warning("Daily data unavailable.")
            else:
                d = df_d.sort_values("Date").set_index("Date")
                eds = _get_earnings_dates(tkr_vis)
                if not eds:
                    st.warning("No earnings dates found (Polygon + yfinance).")
                else:
                    ed_idx = [pd.to_datetime(x) for x in eds]
                    prevc = d["Close"].shift(1)
                    gap = (d["Open"] / prevc - 1.0) * 100.0
                    dayp = (d["Close"] / d["Open"] - 1.0) * 100.0
                    cc = (d["Close"] / d["Close"].shift(1) - 1.0) * 100.0
                    m = gap if move_type.startswith("Gap") else (dayp if move_type.startswith("Day") else cc)
                    sig = (m.abs() >= abs(move_thr)) & d.index.isin(ed_idx) & _apply_common_filters(d)
                    rows = []
                    for ts in d.index[sig]:
                        end_ix = d.index.get_indexer([ts + pd.Timedelta(days=int(horizon))], method='nearest')
                        j = int(end_ix[0]) if end_ix.size and end_ix[0] != -1 else None
                        if j is None:
                            continue
                        start_px = float(d.loc[ts, "Close"]) 
                        end_px = float(d.iloc[j]["Close"])   
                        fr = (end_px / start_px - 1.0) * 100.0
                        rows.append({"Date": ts, "Return%": fr})
                    if not rows:
                        st.info("No earnings events met the move threshold.")
                    else:
                        df_res = pd.DataFrame(rows).set_index("Date").sort_index()
                        s = df_res["Return%"]
                        st.caption(f"Earnings signals: {len(s)} | Mean % (close→close {horizon}d): {s.mean():.2f}")
                        tbl = _summary_table(_load_daily_df(tkr_vis), df_res.index)
                        if not tbl.empty:
                            try:
                                tbl["Prev Day Earnings"] = _earnings_flag_for_dates(tkr_vis, tbl.index)
                            except Exception:
                                pass
                            try:
                                styled = tbl.style.map(
                                    lambda v: "background-color:#2ecc40;color:#000" if (isinstance(v,(int,float)) and v>=0) else ("background-color:#ff6961;color:#000" if isinstance(v,(int,float)) else ""),
                                    subset=[c for c in tbl.columns if c.endswith("Return") or c in ("Prev Close to Close","Close to Open","Open to Close")]
                                )
                                st.write(styled)
                                try:
                                    _render_scan_stats(tbl)
                                except Exception:
                                    pass
                            except Exception:
                                st.dataframe(tbl, use_container_width=True)

                        show_marks = st.checkbox("Show price chart with occurrences", value=False, key=f"marks_earn_{horizon}")
                        send_marks = st.checkbox("Send these dates to Chart highlights", value=False, key=f"send_earn_{horizon}")
                        if show_marks:
                            _viz_price_marks(_load_daily_df(tkr_vis), s.index, f"{tkr_vis}: Earnings LT occurrences ({horizon}d)")
                        if send_marks:
                            try:
                                st.session_state['scan_highlights'] = [str(pd.to_datetime(x).date()) for x in s.index]
                                st.success("Dates sent to Chart tab (toggle 'Show scan highlights').")
                            except Exception:
                                pass

    elif vis == "ThetaData Snapshot":
        st.markdown("### ThetaData Stock Snapshot (streaming)")
        st.caption("Streams CSV from local ThetaData terminal. Default path: v3/stock/snapshot/market_value")
        tt_sym = st.text_input("Ticker", value=tkr_vis).strip().upper()
        term_url = os.getenv("THETADATA_TERMINAL_URL", "http://localhost:25503/v3").strip().rstrip("/")
        snap_path = os.getenv("THETADATA_SNAPSHOT_PATH", "v3/stock/snapshot/market_value").strip("/")
        override_host = st.text_input("Override host (optional)", value=term_url)
        override_path = st.text_input("Override path (optional)", value=snap_path)
        run_tt = st.button("Fetch ThetaData snapshot", key="run_thetadata_snapshot")
        def _build_url(host: str, path: str) -> str:
            h = (host or "").strip().rstrip("/")
            p = (path or "").strip().lstrip("/")
            if h.endswith("/v3") and p.startswith("v3/"):
                p = p[len('v3/'):]
            return f"{h}/{p}"
        if run_tt:
            if not tt_sym:
                st.error("Ticker required.")
            else:
                url = _build_url(override_host or term_url, override_path or snap_path)
                st.caption(f"Requesting: {url}?symbol={tt_sym}")
                try:
                    rows = []
                    with httpx.stream("GET", url, params={"symbol": tt_sym}, timeout=30) as resp:
                        resp.raise_for_status()
                        for line in resp.iter_lines():
                            for row in csv.reader(io.StringIO(line)):
                                rows.append(row)
                                if len(rows) >= 50:
                                    break
                            if len(rows) >= 50:
                                break
                    if rows:
                        st.caption(f"Received {len(rows)} rows (showing up to 50).")
                        st.dataframe(pd.DataFrame(rows))
                    else:
                        st.warning("No rows returned. Check path/host and market hours.")
                except Exception as e:
                    st.error(f"Snapshot request failed: {e}")
    elif vis == "Bollinger + RSI Forward Returns":
        import plotly.graph_objects as _go_bb
        from plotly.subplots import make_subplots as _mk_bb

        st.markdown(
            "Finds days where price is **outside a Bollinger Band** with a "
            "**correspondingly elevated/depressed RSI** — then measures forward returns."
        )
        bc1, bc2, bc3, bc4 = st.columns(4)
        with bc1:
            bb_period = st.number_input("BB period", value=20, min_value=5, max_value=100, step=1, key="bb_rsi_period")
            bb_std_n  = st.number_input("BB std devs", value=2.0, min_value=0.5, max_value=4.0, step=0.5, key="bb_rsi_std")
        with bc2:
            rsi_len_forward = st.number_input("RSI period", value=14, min_value=2, max_value=100, step=1, key="bb_rsi_len")
            lookback_days_forward = st.number_input("Lookback (days)", value=1825, min_value=90, max_value=5000, step=90, key="bb_rsi_lookback")
        with bc3:
            st.markdown("**Upper band (overbought)**")
            include_upper = st.checkbox("Price >= BB Upper", value=True, key="bb_include_upper")
            rsi_high = st.number_input("AND RSI >=", value=60.0, min_value=0.0, max_value=100.0, step=1.0, key="bb_rsi_high",
                                       help="Set 0 to skip RSI filter on upper band.")
            bb_pct_upper = st.number_input("AND BB %B >=", value=1.0, min_value=0.5, max_value=3.0, step=0.05, key="bb_pct_upper",
                                           help="BB %B=(Close-Lower)/(Upper-Lower). 1.0=fully above upper band.")
        with bc4:
            st.markdown("**Lower band (oversold)**")
            include_lower = st.checkbox("Price <= BB Lower", value=True, key="bb_include_lower")
            rsi_low = st.number_input("AND RSI <=", value=40.0, min_value=0.0, max_value=100.0, step=1.0, key="bb_rsi_low",
                                      help="Set 100 to skip RSI filter on lower band.")
            bb_pct_lower = st.number_input("AND BB %B <=", value=0.0, min_value=-2.0, max_value=0.5, step=0.05, key="bb_pct_lower",
                                           help="BB %B <= 0.0 = fully below lower band.")

        bc5, bc6, bc7 = st.columns(3)
        with bc5:
            horizons = st.multiselect(
                "Forward horizons (trading days)",
                options=[1, 2, 3, 5, 10, 15, 21, 42, 63],
                default=[1, 5, 10, 21], key="bb_rsi_horizons"
            )
        with bc6:
            signal_mode = st.selectbox(
                "Signal mode",
                ["Every day in zone (all days outside band)",
                 "First touch only (first day of each new episode)"],
                index=0,
                key="bb_signal_mode",
                help=(
                    "Every day: fires on EACH day price is outside the band + RSI confirmed. "
                    "Shows April 3,4,7,8,9 as separate rows. "
                    "First touch: fires ONCE when price enters the zone, skips consecutive days of the same episode."
                )
            )
        with bc7:
            bb_shares = st.number_input("Shares (equity curve)", value=100, min_value=1, step=10, key="bb_shares")

        if not include_upper and not include_lower:
            st.info("Enable at least one band.")
        elif not horizons:
            st.info("Select at least one forward horizon.")
        else:
            df_d = _load_daily_df_adjusted(tkr_vis)
            if df_d is None or df_d.empty:
                st.error("No daily data found.")
            else:
                d = df_d.sort_values("Date").set_index("Date")
                if isinstance(d.index, pd.DatetimeIndex):
                    cutoff = d.index.max() - pd.Timedelta(days=int(lookback_days_forward))
                    d = d.loc[d.index >= cutoff]

                close  = pd.to_numeric(d["Close"], errors="coerce")
                high_s = pd.to_numeric(d["High"],  errors="coerce") if "High" in d.columns else close
                low_s  = pd.to_numeric(d["Low"],   errors="coerce") if "Low"  in d.columns else close
                open_s = pd.to_numeric(d["Open"],  errors="coerce") if "Open" in d.columns else close

                # Compute BB inline (use parquet columns if available)
                bb_up_col  = f"BB_Upper_{int(bb_period)}"
                bb_lo_col  = f"BB_Lower_{int(bb_period)}"
                bb_mid_col = f"BB_Mid_{int(bb_period)}"
                rsi_col    = f"RSI_{int(rsi_len_forward)}"

                if bb_up_col in d.columns and bb_lo_col in d.columns:
                    bb_upper = pd.to_numeric(d[bb_up_col], errors="coerce")
                    bb_lower = pd.to_numeric(d[bb_lo_col], errors="coerce")
                    bb_mid   = pd.to_numeric(d[bb_mid_col], errors="coerce") if bb_mid_col in d.columns else close.rolling(int(bb_period)).mean()
                else:
                    bb_mid   = close.rolling(int(bb_period), min_periods=int(bb_period)).mean()
                    bb_sigma = close.rolling(int(bb_period), min_periods=int(bb_period)).std()
                    bb_upper = bb_mid + float(bb_std_n) * bb_sigma
                    bb_lower = bb_mid - float(bb_std_n) * bb_sigma
                    st.caption(f"BB computed inline (period={int(bb_period)}, std={bb_std_n})")

                if rsi_col in d.columns:
                    rsiv = pd.to_numeric(d[rsi_col], errors="coerce")
                else:
                    rsiv = rsi(close, int(rsi_len_forward))
                    st.caption(f"RSI computed inline (period={int(rsi_len_forward)})")

                # BB %B: 0=lower band, 1=upper band, >1=above upper, <0=below lower
                bb_bw   = (bb_upper - bb_lower).replace(0, np.nan)
                bb_pctb = (close - bb_lower) / bb_bw

                # Build signal conditions
                cond_up = pd.Series(False, index=d.index)
                cond_lo = pd.Series(False, index=d.index)
                if include_upper:
                    cond_up = bb_pctb >= float(bb_pct_upper)
                    if float(rsi_high) > 0:
                        cond_up = cond_up & (rsiv >= float(rsi_high))
                    cond_up = cond_up.fillna(False)
                if include_lower:
                    cond_lo = bb_pctb <= float(bb_pct_lower)
                    if float(rsi_low) < 100:
                        cond_lo = cond_lo & (rsiv <= float(rsi_low))
                    cond_lo = cond_lo.fillna(False)

                # First-touch isolation
                def _first_touch_bb(cond):
                    out = pd.Series(False, index=cond.index)
                    in_zone = False
                    for i, v in enumerate(cond.values):
                        if v and not in_zone:
                            out.iloc[i] = True
                            in_zone = True
                        elif not v:
                            in_zone = False
                    return out

                if "First touch" in signal_mode:
                    sig_up = _first_touch_bb(cond_up)
                    sig_lo = _first_touch_bb(cond_lo)
                else:
                    sig_up = cond_up
                    sig_lo = cond_lo

                mask_bb = sig_up | sig_lo
                horizons_sorted = sorted(set(int(h) for h in horizons))
                close_arr = close.values
                all_pos = np.where(mask_bb.values)[0]

                events_rows = []
                for pos in all_pos:
                    setup = "Upper Breakout" if sig_up.iloc[pos] else "Lower Breakdown"
                    row = {
                        "_pos": pos,
                        "Setup": setup,
                        "Close": float(close_arr[pos]),
                        "RSI":   float(rsiv.iloc[pos]),
                        "BB_pctB": round(float(bb_pctb.iloc[pos]), 3),
                        "BB Upper": round(float(bb_upper.iloc[pos]), 2),
                        "BB Lower": round(float(bb_lower.iloc[pos]), 2),
                    }
                    for h in horizons_sorted:
                        j = pos + h
                        row[f"Fwd_{h}d_%"] = (close_arr[j] / close_arr[pos] - 1.0) * 100.0 if j < len(close_arr) else np.nan
                    events_rows.append(row)

                if not events_rows:
                    st.info("No signals matched. Try loosening RSI threshold or BB %B.")
                else:
                    ev_df = pd.DataFrame(events_rows)
                    ev_df.index = pd.Index([d.index[r["_pos"]] for r in events_rows], name="Date")
                    ev_df = ev_df.drop(columns=["_pos"])
                    fwd_cols = [f"Fwd_{h}d_%" for h in horizons_sorted]

                    # KPI row
                    k1, k2, k3, k4, k5 = st.columns(5)
                    k1.metric("Total signals", len(ev_df))
                    k2.metric("Upper breakouts", int(sig_up.sum()))
                    k3.metric("Lower breakdowns", int(sig_lo.sum()))
                    if fwd_cols:
                        pc = fwd_cols[0]
                        for ki, snm in enumerate(["Upper Breakout", "Lower Breakdown"]):
                            g2 = pd.to_numeric(ev_df.loc[ev_df["Setup"]==snm, pc], errors="coerce").dropna()
                            if not g2.empty:
                                [k4, k5][ki].metric(
                                    f"Avg Fwd {horizons_sorted[0]}d ({snm[:5]})",
                                    f"{g2.mean():+.2f}%", delta=f"WR {(g2>0).mean()*100:.0f}%"
                                )

                    # Price chart: candlestick + BB bands + RSI panel
                    try:
                        _tpl = template
                    except Exception:
                        _tpl = "plotly_white"

                    chart_days = min(int(lookback_days_forward), 730)
                    d_ch = d.iloc[-chart_days:] if len(d) > chart_days else d
                    dci  = d_ch.index

                    fig_bb = _mk_bb(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                                    row_heights=[0.65, 0.35],
                                    subplot_titles=[
                                        f"{tkr_vis} — Bollinger(period={int(bb_period)}, std={bb_std_n})",
                                        f"RSI({int(rsi_len_forward)})"
                                    ])
                    # Candlesticks
                    fig_bb.add_trace(_go_bb.Candlestick(
                        x=dci, open=open_s.reindex(dci), high=high_s.reindex(dci),
                        low=low_s.reindex(dci), close=close.reindex(dci),
                        name="Price", increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
                        showlegend=False,
                    ), row=1, col=1)
                    # BB bands
                    fig_bb.add_trace(_go_bb.Scatter(
                        x=dci, y=bb_upper.reindex(dci), name="BB Upper",
                        line=dict(color="rgba(100,149,237,0.6)", width=1)
                    ), row=1, col=1)
                    fig_bb.add_trace(_go_bb.Scatter(
                        x=dci, y=bb_lower.reindex(dci), name="BB Lower",
                        line=dict(color="rgba(100,149,237,0.6)", width=1),
                        fill="tonexty", fillcolor="rgba(100,149,237,0.07)"
                    ), row=1, col=1)
                    fig_bb.add_trace(_go_bb.Scatter(
                        x=dci, y=bb_mid.reindex(dci), name="BB Mid",
                        line=dict(color="rgba(100,149,237,0.3)", width=1, dash="dot")
                    ), row=1, col=1)
                    # Signal markers
                    ev_ch = ev_df[ev_df.index.isin(dci)]
                    for snm, sym, clr in [("Upper Breakout","triangle-down","#ef5350"),
                                          ("Lower Breakdown","triangle-up","#26a69a")]:
                        grp = ev_ch[ev_ch["Setup"]==snm]
                        if not grp.empty:
                            fig_bb.add_trace(_go_bb.Scatter(
                                x=grp.index, y=grp["Close"], mode="markers", name=snm,
                                marker=dict(symbol=sym, color=clr, size=10,
                                            line=dict(width=1, color="#fff"))
                            ), row=1, col=1)
                    # RSI panel
                    fig_bb.add_trace(_go_bb.Scatter(
                        x=dci, y=rsiv.reindex(dci), name=f"RSI({int(rsi_len_forward)})",
                        line=dict(color="#ce93d8", width=1.5)
                    ), row=2, col=1)
                    if include_upper and float(rsi_high) > 0:
                        fig_bb.add_hline(y=float(rsi_high), line_dash="dash", line_color="#ef5350",
                                         annotation_text=f"  {int(rsi_high)}", row=2, col=1)
                    if include_lower and float(rsi_low) < 100:
                        fig_bb.add_hline(y=float(rsi_low), line_dash="dash", line_color="#26a69a",
                                         annotation_text=f"  {int(rsi_low)}", row=2, col=1)
                    fig_bb.add_hrect(y0=70, y1=100, fillcolor="rgba(239,83,80,0.06)", line_width=0, row=2, col=1)
                    fig_bb.add_hrect(y0=0,  y1=30,  fillcolor="rgba(38,166,154,0.06)", line_width=0, row=2, col=1)
                    fig_bb.update_yaxes(range=[0, 100], row=2, col=1)
                    fig_bb.update_layout(height=600, template=_tpl, showlegend=True,
                                         xaxis_rangeslider_visible=False,
                                         margin=dict(l=0,r=0,t=40,b=0),
                                         legend=dict(orientation="h", y=1.04, x=0))
                    st.plotly_chart(fig_bb, use_container_width=True)

                    # Forward return stats by setup
                    st.markdown("**Forward Return Statistics by Setup**")
                    summary_rows = []
                    for snm, grp_df in ev_df.groupby("Setup"):
                        for h in horizons_sorted:
                            s = pd.to_numeric(grp_df[f"Fwd_{h}d_%"], errors="coerce").dropna()
                            if s.empty: continue
                            summary_rows.append({
                                "Setup": snm, "Horizon": f"{h}d", "Count": len(s),
                                "Mean %": round(s.mean(), 2), "Median %": round(s.median(), 2),
                                "Win Rate": f"{(s>0).mean()*100:.0f}%",
                                "Best": f"+{s.max():.2f}%", "Worst": f"{s.min():.2f}%",
                            })
                    if summary_rows:
                        sum_df = pd.DataFrame(summary_rows)
                        fig_sum = _go_bb.Figure()
                        clr_map = {"Upper Breakout": "#ef5350", "Lower Breakdown": "#26a69a"}
                        for snm in sum_df["Setup"].unique():
                            grp = sum_df[sum_df["Setup"]==snm]
                            fig_sum.add_trace(_go_bb.Bar(
                                x=grp["Horizon"], y=grp["Mean %"], name=snm,
                                marker_color=clr_map.get(snm,"#90caf9"),
                                text=[f"{v:+.2f}%" for v in grp["Mean %"]], textposition="outside"
                            ))
                        fig_sum.add_hline(y=0, line_color="#888", line_width=1)
                        fig_sum.update_layout(height=300, barmode="group", template=_tpl,
                                               xaxis_title="Horizon (trading days)",
                                               yaxis_title="Mean Forward Return %",
                                               margin=dict(l=0,r=0,t=20,b=0),
                                               legend=dict(orientation="h", y=1.1))
                        st.plotly_chart(fig_sum, use_container_width=True)
                        st.dataframe(sum_df.set_index(["Setup","Horizon"]), use_container_width=True)

                    # Equity curves
                    if fwd_cols:
                        with st.expander("Equity curves (systematic buyer of every signal)", expanded=False):
                            ph = horizons_sorted[0]
                            fig_eq = _go_bb.Figure()
                            for snm, ec in [("Upper Breakout","#ef5350"),("Lower Breakdown","#26a69a"),("All","#90caf9")]:
                                gdf = ev_df if snm=="All" else ev_df[ev_df["Setup"]==snm]
                                if gdf.empty: continue
                                ret_s = pd.to_numeric(gdf[f"Fwd_{ph}d_%"], errors="coerce").fillna(0)
                                cum   = (ret_s / 100.0 * gdf["Close"] * int(bb_shares)).cumsum()
                                fig_eq.add_trace(_go_bb.Scatter(x=cum.index, y=cum.values,
                                                                 mode="lines", name=snm,
                                                                 line=dict(color=ec, width=2)))
                            fig_eq.add_hline(y=0, line_color="#888", line_width=1)
                            fig_eq.update_layout(height=320, template=_tpl,
                                                  title=f"Cum P&L — exit after {ph}d | {int(bb_shares)} shares",
                                                  yaxis_title="Cum P&L ($)",
                                                  margin=dict(l=0,r=0,t=40,b=0))
                            st.plotly_chart(fig_eq, use_container_width=True)

                    # Per-signal table
                    with st.expander("Per-signal detail table", expanded=False):
                        disp = ev_df.copy().sort_index(ascending=False).reset_index()
                        disp["RSI"]      = disp["RSI"].map(lambda v: f"{v:.1f}")
                        disp["Close"]    = disp["Close"].map(lambda v: f"${v:.2f}")
                        disp["BB_pctB"]  = disp["BB_pctB"].map(lambda v: f"{v:.3f}")
                        for c in fwd_cols:
                            disp[c] = pd.to_numeric(disp[c], errors="coerce").map(
                                lambda v: f"{v:+.2f}%" if pd.notna(v) else "-"
                            )
                        st.dataframe(disp, use_container_width=True, hide_index=True)
                        st.download_button("Download CSV",
                            data=ev_df.reset_index().to_csv(index=False).encode(),
                            file_name=f"{tkr_vis}_bb_rsi_signals.csv", mime="text/csv")
    elif vis == "RSI Threshold Backtest":
        import plotly.graph_objects as _go_rsi
        from plotly.subplots import make_subplots as _mk_rsi

        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            rsi_len_days = st.number_input("RSI period", value=14, min_value=2, max_value=100, step=1, key="rsi_bt_len")
        with rc2:
            rsi_mode = st.selectbox("Signal direction", [
                "Oversold — RSI crosses UP through threshold (buy dip)",
                "Overbought — RSI crosses DOWN through threshold (fade rally)",
                "Above threshold (RSI >= value)",
                "Below threshold (RSI <= value)"], key="rsi_mode")
        with rc3:
            rsi_thresh = st.number_input("RSI threshold", value=30, min_value=1, max_value=99, step=1, key="rsi_bt_thr")
        rc4, rc5, rc6 = st.columns(3)
        with rc4:
            horizons = st.multiselect("Forward horizons (trading days)",
                options=[1,2,3,5,10,15,21,42,63], default=[1,5,10,21], key="rsi_horizons")
        with rc5:
            rsi_lookback = st.number_input("Lookback (days)", value=1825, min_value=90, max_value=5000, step=90, key="rsi_lookback")
        with rc6:
            show_sweep = st.checkbox("RSI level sweep heatmap", value=False, key="rsi_sweep",
                help="Sweeps RSI 10-90 to show which level gives best forward returns")
        run_rsi = st.button("Run RSI Backtest", key="run_rsi_bt", use_container_width=True)
        if run_rsi:
            df_d = _load_daily_df_adjusted(tkr_vis)
            if df_d is None or df_d.empty:
                st.error("No daily data found.")
            else:
                d = df_d.sort_values("Date").set_index("Date")
                if isinstance(d.index, pd.DatetimeIndex):
                    cutoff = d.index.max() - pd.Timedelta(days=int(rsi_lookback))
                    d = d.loc[d.index >= cutoff]
                close = d["Close"].astype(float)
                rsiv  = rsi(close, int(rsi_len_days))
                horizons_sorted = sorted(set(int(h) for h in horizons))
                thr = float(rsi_thresh)
                if "crosses UP" in rsi_mode:
                    sig = (rsiv.shift(1) < thr) & (rsiv >= thr)
                    sig_label = f"RSI({int(rsi_len_days)}) crosses UP through {int(thr)}"
                elif "crosses DOWN" in rsi_mode:
                    sig = (rsiv.shift(1) > thr) & (rsiv <= thr)
                    sig_label = f"RSI({int(rsi_len_days)}) crosses DOWN through {int(thr)}"
                elif "Above" in rsi_mode:
                    sig = rsiv >= thr
                    sig_label = f"RSI({int(rsi_len_days)}) >= {int(thr)}"
                else:
                    sig = rsiv <= thr
                    sig_label = f"RSI({int(rsi_len_days)}) <= {int(thr)}"
                sig = sig.fillna(False) & _apply_common_filters(d)
                close_arr  = close.values
                sig_pos    = np.where(sig.values)[0]
                events_rows = []
                for pos in sig_pos:
                    row = {"Date": d.index[pos], "Close": close_arr[pos], "RSI": float(rsiv.iloc[pos])}
                    for h in horizons_sorted:
                        j = pos + h
                        row[f"Fwd_{h}d_%"] = (close_arr[j]/close_arr[pos]-1.0)*100.0 if j < len(close_arr) else np.nan
                    events_rows.append(row)
                if not events_rows:
                    st.info(f"No signals: {sig_label}")
                else:
                    ev_df = pd.DataFrame(events_rows).set_index("Date")
                    fwd_cols = [f"Fwd_{h}d_%" for h in horizons_sorted]
                    # KPIs
                    k1, k2, k3, k4 = st.columns(4)
                    k1.metric("Signals", len(ev_df))
                    if fwd_cols:
                        v = pd.to_numeric(ev_df[fwd_cols[0]], errors="coerce").dropna()
                        k2.metric(f"Avg Fwd {horizons_sorted[0]}d", f"{v.mean():+.2f}%",
                                  delta=f"Median {v.median():+.2f}%")
                        k3.metric("Win Rate", f"{(v>0).mean()*100:.0f}%",
                                  delta=f"{(v>0).mean()*100-50:+.1f}pp vs 50%")
                    k4.metric("RSI range", f"{ev_df['RSI'].min():.0f} - {ev_df['RSI'].max():.0f}")
                    # Price + RSI chart
                    try:
                        _tpl = template
                    except Exception:
                        _tpl = "plotly_white"
                    fig_rc = _mk_rsi(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                                     row_heights=[0.65, 0.35],
                                     subplot_titles=[f"{tkr_vis} Price", f"RSI({int(rsi_len_days)})"])
                    fig_rc.add_trace(_go_rsi.Scatter(x=d.index, y=close, mode="lines", name="Close",
                                                      line=dict(color="#90caf9", width=1.5)), row=1, col=1)
                    fig_rc.add_trace(_go_rsi.Scatter(x=ev_df.index, y=ev_df["Close"], mode="markers",
                                                      name="Signal",
                                                      marker=dict(symbol="triangle-up" if thr < 50 else "triangle-down",
                                                                  color="#ffeb3b", size=8,
                                                                  line=dict(width=1, color="#333"))), row=1, col=1)
                    fig_rc.add_trace(_go_rsi.Scatter(x=rsiv.index, y=rsiv.values, mode="lines",
                                                      name=f"RSI({int(rsi_len_days)})",
                                                      line=dict(color="#ce93d8", width=1.5)), row=2, col=1)
                    fig_rc.add_hline(y=thr, line_dash="dash", line_color="#ef5350",
                                     annotation_text=f"  {int(thr)}", row=2, col=1)
                    fig_rc.add_hrect(y0=0,  y1=30,  fillcolor="rgba(38,166,154,0.06)", line_width=0, row=2, col=1)
                    fig_rc.add_hrect(y0=70, y1=100, fillcolor="rgba(239,83,80,0.06)",  line_width=0, row=2, col=1)
                    fig_rc.update_yaxes(range=[0, 100], row=2, col=1)
                    fig_rc.update_layout(height=520, showlegend=True, template=_tpl,
                                          title=dict(text=sig_label, font=dict(size=13)),
                                          margin=dict(l=0,r=0,t=40,b=0),
                                          xaxis_rangeslider_visible=False)
                    st.plotly_chart(fig_rc, use_container_width=True)
                    # Stats table + bar chart
                    st.markdown("**Forward Return Statistics**")
                    stats_rows = []
                    for h in horizons_sorted:
                        s = pd.to_numeric(ev_df[f"Fwd_{h}d_%"], errors="coerce").dropna()
                        if s.empty: continue
                        stats_rows.append({"Horizon": f"{h}d", "Signals": len(s),
                            "Mean %": round(s.mean(),2), "Median %": round(s.median(),2),
                            "Win Rate": f"{(s>0).mean()*100:.0f}%",
                            "Best": f"+{s.max():.2f}%", "Worst": f"{s.min():.2f}%",
                            "Std %": round(s.std(),2)})
                    if stats_rows:
                        st_df = pd.DataFrame(stats_rows).set_index("Horizon")
                        fig_bar = _go_rsi.Figure()
                        fig_bar.add_trace(_go_rsi.Bar(
                            x=st_df.index, y=st_df["Mean %"], name="Mean %",
                            marker_color=["#26a69a" if v>=0 else "#ef5350" for v in st_df["Mean %"]],
                            text=[f"{v:+.2f}%" for v in st_df["Mean %"]], textposition="outside"))
                        fig_bar.add_trace(_go_rsi.Bar(
                            x=st_df.index, y=st_df["Median %"], name="Median %",
                            marker_color=["rgba(38,166,154,0.5)" if v>=0 else "rgba(239,83,80,0.5)"
                                          for v in st_df["Median %"]],
                            text=[f"{v:+.2f}%" for v in st_df["Median %"]], textposition="outside"))
                        fig_bar.add_hline(y=0, line_color="#888", line_width=1)
                        fig_bar.update_layout(height=280, barmode="group", template=_tpl,
                                               xaxis_title="Horizon", yaxis_title="Return %",
                                               margin=dict(l=0,r=0,t=10,b=0),
                                               legend=dict(orientation="h", y=1.1))
                        st.plotly_chart(fig_bar, use_container_width=True)
                        st.dataframe(st_df, use_container_width=True)
                    with st.expander("Per-signal table", expanded=False):
                        disp = ev_df.copy().sort_index(ascending=False).reset_index()
                        disp["RSI"]   = disp["RSI"].map(lambda v: f"{v:.1f}")
                        disp["Close"] = disp["Close"].map(lambda v: f"${v:.2f}")
                        for c in fwd_cols:
                            disp[c] = pd.to_numeric(disp[c], errors="coerce").map(
                                lambda v: f"{v:+.2f}%" if pd.notna(v) else "-")
                        st.dataframe(disp, use_container_width=True, hide_index=True)
                        st.download_button("Download CSV",
                            data=ev_df.reset_index().to_csv(index=False).encode(),
                            file_name=f"{tkr_vis}_rsi{int(rsi_len_days)}_signals.csv", mime="text/csv")
                    # RSI level sweep heatmap
                    if show_sweep:
                        st.markdown("**RSI Level Sweep** — mean forward return at each threshold")
                        prog = st.progress(0, text="Sweeping...")
                        sweep_levels = list(range(10, 91, 5))
                        sweep_below  = thr < 50
                        sweep_data   = {}
                        for si, lvl in enumerate(sweep_levels):
                            prog.progress(int(si/len(sweep_levels)*100), text=f"RSI {'<=' if sweep_below else '>='} {lvl}")
                            if sweep_below:
                                ss = (rsiv.shift(1) >= lvl) & (rsiv < lvl) if "crosses" in rsi_mode else (rsiv <= lvl)
                            else:
                                ss = (rsiv.shift(1) <= lvl) & (rsiv > lvl) if "crosses" in rsi_mode else (rsiv >= lvl)
                            sp = np.where(ss.fillna(False).values)[0]
                            row_d = {}
                            for h in horizons_sorted:
                                rets = [(close_arr[p+h]/close_arr[p]-1.0)*100.0 for p in sp if p+h < len(close_arr)]
                                row_d[f"{h}d"] = round(np.mean(rets), 3) if rets else np.nan
                            sweep_data[lvl] = row_d
                        prog.empty()
                        sw_df = pd.DataFrame(sweep_data).T
                        sw_df.index.name = "RSI Level"
                        fig_heat = _go_rsi.Figure(data=_go_rsi.Heatmap(
                            z=sw_df.values.T,
                            x=[str(l) for l in sw_df.index],
                            y=sw_df.columns.tolist(),
                            colorscale=[[0,"#ef5350"],[0.5,"#1e2530"],[1,"#26a69a"]],
                            zmid=0,
                            text=[[f"{v:+.2f}%" if not np.isnan(v) else "-" for v in row] for row in sw_df.values.T],
                            texttemplate="%{text}", textfont=dict(size=9),
                            colorbar=dict(title="Mean %")))
                        fig_heat.update_layout(height=300, template=_tpl,
                            title=f"Mean forward return by RSI threshold — {tkr_vis}",
                            xaxis_title=f"RSI ({'<=' if sweep_below else '>='}) Level",
                            yaxis_title="Horizon", margin=dict(l=0,r=0,t=40,b=0))
                        st.plotly_chart(fig_heat, use_container_width=True)
    elif vis == "MA Slope Dynamics":
        import plotly.graph_objects as _go_sma
        from plotly.subplots import make_subplots as _mk_sma

        df_d = _load_daily_df_adjusted(tkr_vis)
        if df_d is None or df_d.empty:
            st.warning("No daily data found for this ticker.")
        else:
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                sma_per = st.number_input("MA period", value=10, min_value=3, max_value=200, step=1,
                                          key="msd_period",
                                          help="Length of the moving average to analyze (default 10).")
                pull_pct = st.number_input("Pullback zone (% from MA)", value=1.0, min_value=0.1,
                                           max_value=10.0, step=0.1, key="msd_pull",
                                           help="Signal when price comes within this % above the MA while the 10d slope is positive.")
            with sc2:
                lookback_days = st.number_input("Lookback days", value=1825, min_value=60,
                                                 max_value=5000, step=90, key="msd_lookback")
                horizons = st.multiselect("Forward horizons (trading days)",
                                          options=[1, 2, 3, 5, 10, 15, 21, 42, 63],
                                          default=[1, 5, 10], key="msd_horizons")
            with sc3:
                flat_thr = st.number_input("Flat zone threshold (%/day)", value=0.03, min_value=0.001,
                                            max_value=1.0, step=0.005, format="%.3f", key="msd_flat",
                                            help="10d slope |value| ≤ this = 'flat'. Tip: ~0.5× slope std shown below.")
                min_up_bars = st.number_input("Min positive-slope bars before pullback",
                                               value=5, min_value=1, max_value=50, step=1,
                                               key="msd_pullback_min",
                                               help="MA must have been rising this many bars before a pullback-to-MA event counts.")

            df = df_d.sort_values("Date").set_index("Date")
            if isinstance(df.index, pd.DatetimeIndex):
                cutoff = df.index.max() - pd.Timedelta(days=int(lookback_days))
                df = df.loc[df.index >= cutoff]

            close = pd.to_numeric(df["Close"], errors="coerce")
            sma   = close.rolling(int(sma_per), min_periods=int(sma_per)).mean()

            # Three slope windows, each normalised as % of price per day
            _px      = close.replace(0, np.nan)
            slope_1d  = (sma.diff(1)  / 1  / _px) * 100   # daily pulse
            slope_5d  = (sma.diff(5)  / 5  / _px) * 100   # short-term trend
            slope_10d = (sma.diff(10) / 10 / _px) * 100   # established direction

            s10_std    = float(slope_10d.dropna().std())
            last_close = float(close.dropna().iloc[-1])
            ma1, ma2, ma3, ma4, ma5 = st.columns(5)
            ma1.metric("Last Close",    f"${last_close:.2f}")
            ma2.metric("Slope 1d now",  f"{float(slope_1d.dropna().iloc[-1]):+.4f}%/d")
            ma3.metric("Slope 5d now",  f"{float(slope_5d.dropna().iloc[-1]):+.4f}%/d")
            ma4.metric("Slope 10d now", f"{float(slope_10d.dropna().iloc[-1]):+.4f}%/d")
            ma5.metric("10d slope std", f"{s10_std:.4f}  (flat ~ ±{s10_std*0.5:.4f})")

            ft  = float(flat_thr)
            pct = float(pull_pct)

            # ── Alignment Score (0-3): how many slopes are positive simultaneously ──
            # Score 3 = all aligned up = maximum momentum (Full Stack)
            # Score 0 = no positive slopes (mixed/down)
            align_score = (
                (slope_1d  > 0).astype(int) +
                (slope_5d  > 0).astype(int) +
                (slope_10d >= ft).astype(int)
            )

            # ── Signal definitions ───────────────────────────────────────────────
            # Turn_Up: 10d slope crosses definitively positive
            sig_turn_up   = (slope_10d.shift(1) < ft)  & (slope_10d >= ft)
            # Turn_Down: 10d slope crosses definitively negative
            sig_turn_down = (slope_10d.shift(1) > -ft) & (slope_10d <= -ft)
            # Goes_Flat: 10d slope enters flat zone from outside
            sig_goes_flat = (slope_10d.abs().shift(1) > ft) & (slope_10d.abs() <= ft)

            # Pullback_to_MA: price enters the pullback zone from above while MA has been rising
            slope_rising = (slope_10d > ft)
            _grp_id  = (~slope_rising).cumsum()
            consec_up = (slope_rising.astype(int)
                         .groupby(_grp_id)
                         .transform(lambda x: x.cumsum()))
            pull_dist    = (close / sma.replace(0, np.nan) - 1.0) * 100
            entering_zone = (pull_dist.abs() <= pct) & (pull_dist >= -pct)
            prev_outside  = pull_dist.shift(1).abs() > pct
            sig_pullback  = (consec_up.shift(1) >= int(min_up_bars)) & entering_zone & prev_outside

            # Stack_Up: first bar all 3 slopes align positive — momentum IGNITION signal
            all_aligned  = (slope_1d > 0) & (slope_5d > 0) & (slope_10d >= ft)
            sig_stack_up = all_aligned & ~all_aligned.shift(1, fill_value=False)

            # Stack_Break: 1d slope turns negative while 5d + 10d still positive — FIRST WARNING
            sig_stack_break = (
                (slope_1d.shift(1) > 0) & (slope_1d <= 0) &
                (slope_5d > 0) & (slope_10d > ft)
            )

            sig_map = {
                "Turn_Up":     sig_turn_up,
                "Stack_Up":    sig_stack_up,
                "Pullback":    sig_pullback,
                "Stack_Break": sig_stack_break,
                "Goes_Flat":   sig_goes_flat,
                "Turn_Down":   sig_turn_down,
            }

            horizons_sorted = sorted(set(int(h) for h in horizons))
            close_arr = close.values
            idx = df.index

            events_rows = []
            for sig_name, sig_mask in sig_map.items():
                for pos in np.where(sig_mask.values)[0]:
                    row = {
                        "Signal":       sig_name,
                        "Close":        round(float(close_arr[pos]), 2),
                        "SMA":          round(float(sma.iloc[pos]), 2) if not np.isnan(sma.iloc[pos]) else np.nan,
                        "Slope_1d":     round(float(slope_1d.iloc[pos]), 4),
                        "Slope_5d":     round(float(slope_5d.iloc[pos]), 4),
                        "Slope_10d":    round(float(slope_10d.iloc[pos]), 4),
                        "Align_Score":  int(align_score.iloc[pos]) if not np.isnan(align_score.iloc[pos]) else 0,
                        "_pos": pos,
                    }
                    for h in horizons_sorted:
                        j = pos + h
                        row[f"Fwd_{h}d_%"] = (close_arr[j] / close_arr[pos] - 1.0) * 100.0 if j < len(close_arr) else np.nan
                    events_rows.append(row)

            try:
                _tpl = template
            except Exception:
                _tpl = "plotly_white"

            if not events_rows:
                st.info("No signals found. Try lowering the flat threshold.")
            else:
                ev_df = pd.DataFrame(events_rows)
                ev_df.index = pd.Index([idx[r["_pos"]] for r in events_rows], name="Date")
                ev_df = ev_df.drop(columns=["_pos"])
                fwd_cols = [f"Fwd_{h}d_%" for h in horizons_sorted]

                sig_style = {
                    "Turn_Up":     ("#26a69a", "triangle-up",    12),
                    "Stack_Up":    ("#00e676", "star",           14),  # bright green star = ignition
                    "Pullback":    ("#90caf9", "circle",         10),
                    "Stack_Break": ("#ff9800", "x",              12),  # orange X = first warning
                    "Goes_Flat":   ("#ffeb3b", "diamond",        10),
                    "Turn_Down":   ("#ef5350", "triangle-down",  12),
                }
                k1, k2, k3, k4, k5, k6 = st.columns(6)
                k1.metric("Turn_Up",     int(sig_turn_up.sum()))
                k2.metric("Stack_Up ★",  int(sig_stack_up.sum()))
                k3.metric("Pullback",    int(sig_pullback.sum()))
                k4.metric("Stack_Break", int(sig_stack_break.sum()))
                k5.metric("Goes_Flat",   int(sig_goes_flat.sum()))
                k6.metric("Turn_Down",   int(sig_turn_down.sum()))

                chart_days = min(int(lookback_days), 730)
                d_ch  = df.iloc[-chart_days:]
                dci   = d_ch.index
                open_s = pd.to_numeric(df["Open"], errors="coerce") if "Open" in df.columns else close
                high_s = pd.to_numeric(df["High"], errors="coerce") if "High" in df.columns else close
                low_s  = pd.to_numeric(df["Low"],  errors="coerce") if "Low"  in df.columns else close

                # ── Build 3-panel chart ────────────────────────────────────
                fig = _mk_sma(
                    rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.03,
                    specs=[[{}], [{"secondary_y": True}], [{}]],
                    row_heights=[0.52, 0.30, 0.18],
                    subplot_titles=[
                        f"{tkr_vis} + SMA({int(sma_per)})  ★=Stack_Up  ✕=Stack_Break",
                        f"Slopes (1d/5d/10d) + Alignment Score (0–3 bars, right axis)",
                        f"% Distance from SMA  (pullback zone ±{pct:.1f}%)",
                    ],
                )

                # Panel 1: green vrect background for "all aligned" periods
                _aa_dci  = all_aligned.reindex(dci, fill_value=False)
                _run_in  = _aa_dci & ~_aa_dci.shift(1, fill_value=False)
                _run_out = ~_aa_dci & _aa_dci.shift(1, fill_value=False)
                _starts  = list(dci[_run_in])
                _ends    = list(dci[_run_out])
                if _aa_dci.iloc[-1]:
                    _ends.append(dci[-1])
                for _s, _e in zip(_starts, _ends):
                    fig.add_vrect(x0=_s, x1=_e,
                                  fillcolor="rgba(0,230,118,0.07)", line_width=0,
                                  row=1, col=1)

                # Panel 1: candles + SMA
                fig.add_trace(_go_sma.Candlestick(
                    x=dci, open=open_s.reindex(dci), high=high_s.reindex(dci),
                    low=low_s.reindex(dci),  close=close.reindex(dci),
                    name="Price", increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
                    showlegend=False,
                ), row=1, col=1)
                fig.add_trace(_go_sma.Scatter(
                    x=dci, y=sma.reindex(dci),
                    name=f"SMA({int(sma_per)})", line=dict(color="#ffeb3b", width=1.5)
                ), row=1, col=1)

                # Panel 1: signal markers
                ev_ch = ev_df[ev_df.index.isin(dci)]
                for sig_name, (clr, sym, sz) in sig_style.items():
                    grp = ev_ch[ev_ch["Signal"] == sig_name]
                    if not grp.empty:
                        fig.add_trace(_go_sma.Scatter(
                            x=grp.index, y=grp["Close"], mode="markers", name=sig_name,
                            marker=dict(symbol=sym, color=clr, size=sz,
                                        line=dict(width=1, color="#fff"))
                        ), row=1, col=1)

                # Panel 2: three slope lines (primary y)
                slope_colors = [
                    (slope_1d,  "Slope 1d",  "#ef5350", 1.2),
                    (slope_5d,  "Slope 5d",  "#ffeb3b", 1.5),
                    (slope_10d, "Slope 10d", "#26a69a", 2.0),
                ]
                for sl, nm, clr, lw in slope_colors:
                    fig.add_trace(_go_sma.Scatter(
                        x=dci, y=sl.reindex(dci), name=nm,
                        line=dict(color=clr, width=lw)
                    ), row=2, col=1, secondary_y=False)
                fig.add_hline(y=0,   line_color="#888", line_width=1, row=2, col=1)
                fig.add_hline(y= ft, line_dash="dash", line_color="rgba(38,166,154,0.5)",
                               annotation_text=f"  +{ft:.3f}", row=2, col=1)
                fig.add_hline(y=-ft, line_dash="dash", line_color="rgba(239,83,80,0.5)",
                               annotation_text=f"  -{ft:.3f}", row=2, col=1)
                fig.add_hrect(y0=-ft, y1=ft, fillcolor="rgba(255,235,59,0.04)",
                               line_width=0, row=2, col=1)

                # Panel 2: alignment score bars (secondary y, 0-3)
                _ascr  = align_score.reindex(dci).fillna(0).astype(int)
                _aclrs = [
                    "#00e676" if v == 3 else
                    "#26a69a" if v == 2 else
                    "#80cbc4" if v == 1 else
                    "#424242"
                    for v in _ascr
                ]
                fig.add_trace(_go_sma.Bar(
                    x=dci, y=_ascr,
                    name="Align Score (0-3)",
                    marker_color=_aclrs, marker_line_width=0,
                    opacity=0.45,
                ), row=2, col=1, secondary_y=True)
                fig.update_yaxes(range=[0, 3.5], row=2, col=1,
                                  secondary_y=True, showgrid=False,
                                  tickvals=[0, 1, 2, 3],
                                  title_text="Score")

                # Panel 3: % distance from MA
                fig.add_trace(_go_sma.Scatter(
                    x=dci, y=pull_dist.reindex(dci), name="% from MA",
                    line=dict(color="#90caf9", width=1.5),
                    fill="tozeroy", fillcolor="rgba(144,202,249,0.08)"
                ), row=3, col=1)
                fig.add_hline(y=0,    line_color="#888", line_width=1, row=3, col=1)
                fig.add_hline(y= pct, line_dash="dash", line_color="rgba(144,202,249,0.6)",
                               annotation_text=f"  +{pct:.1f}%", row=3, col=1)
                fig.add_hline(y=-pct, line_dash="dash", line_color="rgba(144,202,249,0.6)",
                               annotation_text=f"  -{pct:.1f}%", row=3, col=1)
                fig.add_hrect(y0=-pct, y1=pct, fillcolor="rgba(144,202,249,0.05)",
                               line_width=0, row=3, col=1)

                fig.update_layout(height=760, template=_tpl, showlegend=True,
                                   xaxis_rangeslider_visible=False,
                                   margin=dict(l=0, r=0, t=40, b=0),
                                   legend=dict(orientation="h", y=1.04, x=0),
                                   barmode="overlay")
                st.plotly_chart(fig, use_container_width=True)

                # ── Forward Return Stats by Signal ─────────────────────────
                st.markdown("**Forward Return Statistics by Signal**")
                summary_rows = []
                for sig_type, grp_df in ev_df.groupby("Signal"):
                    for h in horizons_sorted:
                        s = pd.to_numeric(grp_df[f"Fwd_{h}d_%"], errors="coerce").dropna()
                        if s.empty:
                            continue
                        summary_rows.append({
                            "Signal": sig_type, "Horizon": f"{h}d", "Count": len(s),
                            "Mean %": round(s.mean(), 2), "Median %": round(s.median(), 2),
                            "Win Rate": f"{(s > 0).mean() * 100:.0f}%",
                            "Best": f"+{s.max():.2f}%", "Worst": f"{s.min():.2f}%",
                        })
                if summary_rows:
                    sum_df = pd.DataFrame(summary_rows)
                    clrs_map = {
                        "Turn_Up":     "#26a69a", "Stack_Up":    "#00e676",
                        "Pullback":    "#90caf9", "Stack_Break": "#ff9800",
                        "Goes_Flat":   "#ffeb3b", "Turn_Down":   "#ef5350",
                    }
                    fig_bar = _go_sma.Figure()
                    for sig_type in ["Turn_Up", "Stack_Up", "Pullback",
                                     "Stack_Break", "Goes_Flat", "Turn_Down"]:
                        grp = sum_df[sum_df["Signal"] == sig_type]
                        if grp.empty:
                            continue
                        fig_bar.add_trace(_go_sma.Bar(
                            x=grp["Horizon"], y=grp["Mean %"], name=sig_type,
                            marker_color=clrs_map.get(sig_type, "#aaa"),
                            text=[f"{v:+.2f}%" for v in grp["Mean %"]], textposition="outside"
                        ))
                    fig_bar.add_hline(y=0, line_color="#888", line_width=1)
                    fig_bar.update_layout(height=300, barmode="group", template=_tpl,
                                           xaxis_title="Horizon (trading days)",
                                           yaxis_title="Mean Return %",
                                           margin=dict(l=0, r=0, t=10, b=0),
                                           legend=dict(orientation="h", y=1.1))
                    st.plotly_chart(fig_bar, use_container_width=True)
                    st.dataframe(sum_df.set_index(["Signal", "Horizon"]), use_container_width=True)

                with st.expander("Per-signal table", expanded=False):
                    disp = ev_df.copy().sort_index(ascending=False).reset_index()
                    disp["Close"] = disp["Close"].map(lambda v: f"${v:.2f}")
                    for c in fwd_cols:
                        disp[c] = pd.to_numeric(disp[c], errors="coerce").map(
                            lambda v: f"{v:+.2f}%" if pd.notna(v) else "-")
                    st.dataframe(disp, use_container_width=True, hide_index=True)
                    st.download_button("Download CSV",
                        data=ev_df.reset_index().to_csv(index=False).encode(),
                        file_name=f"{tkr_vis}_ma{int(sma_per)}_slope_dynamics.csv",
                        mime="text/csv")

            # ── Regime Daily Performance (5 levels by alignment score) ───
            st.markdown("---")
            st.markdown("**Daily Return by Momentum Alignment Score**")
            st.caption(
                "Score = number of slopes aligned positive (1d > 0, 5d > 0, 10d ≥ threshold).  "
                f"Score 3 = Full Stack (max momentum)  |  Score 0 + 10d ≤ −{ft:.3f} = Downswing"
            )

            daily_ret = close.pct_change() * 100

            # 5-level regime
            _score = align_score.reindex(daily_ret.index, fill_value=0)
            _dn    = slope_10d.reindex(daily_ret.index, fill_value=0) <= -ft
            regime_lbl = pd.Series("Mixed (0/3)", index=daily_ret.index, dtype=str)
            regime_lbl[_dn]            = "Downswing"
            regime_lbl[_score == 1]    = "Weak (1/3)"
            regime_lbl[_score == 2]    = "Strong (2/3)"
            regime_lbl[_score == 3]    = "Full Stack (3/3)"

            REGIME_ORDER  = ["Full Stack (3/3)", "Strong (2/3)", "Weak (1/3)",
                              "Mixed (0/3)", "Downswing"]
            REGIME_COLORS = {
                "Full Stack (3/3)": "#00e676",
                "Strong (2/3)":     "#26a69a",
                "Weak (1/3)":       "#80cbc4",
                "Mixed (0/3)":      "#ffeb3b",
                "Downswing":        "#ef5350",
            }

            perf_df = pd.DataFrame({
                "Daily_Ret_%": daily_ret,
                "Regime":      regime_lbl,
                "Close":       close,
                "Align_Score": _score,
            }).dropna(subset=["Daily_Ret_%"])

            stats_rows = []
            for r_name in REGIME_ORDER:
                s = perf_df.loc[perf_df["Regime"] == r_name, "Daily_Ret_%"]
                if s.empty:
                    continue
                cum = (1 + s / 100).prod() - 1
                stats_rows.append({
                    "Regime":               r_name,
                    "Days":                 len(s),
                    "Mean Daily %":         round(s.mean(), 3),
                    "Median Daily %":       round(s.median(), 3),
                    "Win Rate":             f"{(s > 0).mean() * 100:.0f}%",
                    "Best Day":             f"+{s.max():.2f}%",
                    "Worst Day":            f"{s.min():.2f}%",
                    "Cumul. (regime only)": f"{cum * 100:+.1f}%",
                })
            if stats_rows:
                st.dataframe(pd.DataFrame(stats_rows).set_index("Regime"),
                             use_container_width=True)

            _cd       = min(int(lookback_days), 730)
            chart_df2 = perf_df.iloc[-_cd:]
            fig_daily = _go_sma.Figure()
            for r_name in REGIME_ORDER:
                mask_r = chart_df2["Regime"] == r_name
                fig_daily.add_trace(_go_sma.Bar(
                    x=chart_df2.index[mask_r],
                    y=chart_df2.loc[mask_r, "Daily_Ret_%"],
                    name=r_name, marker_color=REGIME_COLORS[r_name],
                    marker_line_width=0,
                ))
            fig_daily.add_hline(y=0, line_color="#555", line_width=1)
            fig_daily.update_layout(
                height=250, barmode="overlay", template=_tpl,
                title=f"{tkr_vis} — daily return % by alignment score (last {_cd} days)",
                yaxis_title="Daily Return %",
                margin=dict(l=0, r=0, t=40, b=0),
                legend=dict(orientation="h", y=1.15),
                bargap=0,
            )
            st.plotly_chart(fig_daily, use_container_width=True)

            # Cumulative equity by regime
            fig_cum = _go_sma.Figure()
            bnh_cum = (1 + perf_df["Daily_Ret_%"].fillna(0) / 100).cumprod()
            fig_cum.add_trace(_go_sma.Scatter(
                x=perf_df.index, y=bnh_cum,
                name="Buy & Hold", line=dict(color="#aaaaaa", width=1.5, dash="dot")
            ))
            for r_name in REGIME_ORDER:
                r_ret = perf_df["Daily_Ret_%"].where(
                    perf_df["Regime"] == r_name, 0).fillna(0)
                cum_r = (1 + r_ret / 100).cumprod()
                fig_cum.add_trace(_go_sma.Scatter(
                    x=perf_df.index, y=cum_r,
                    name=f"{r_name} only",
                    line=dict(color=REGIME_COLORS[r_name], width=2)
                ))
            fig_cum.add_hline(y=1.0, line_color="#555", line_width=1)
            fig_cum.update_layout(
                height=300, template=_tpl,
                title=f"{tkr_vis} — cumulative $1 held only during each alignment regime",
                yaxis_title="Equity ($1 start)", yaxis_tickformat=".2f",
                margin=dict(l=0, r=0, t=40, b=0),
                legend=dict(orientation="h", y=1.15),
            )
            st.plotly_chart(fig_cum, use_container_width=True)

            # ── Upswing Run Analysis ───────────────────────────────────────
            st.markdown("---")
            st.markdown("**Upswing Run Analysis** — every period where 10d slope ≥ threshold")
            st.caption("Each row is one continuous upswing. Max DD = largest intra-run drawdown from peak.")
            in_upswing = (slope_10d >= ft)
            _run_grp   = (~in_upswing).cumsum()
            run_rows   = []
            for rid, grp in close.groupby(_run_grp):
                try:
                    if not bool(in_upswing.loc[grp.index[0]]):
                        continue
                except Exception:
                    continue
                if len(grp) < 2:
                    continue
                entry_px   = float(grp.iloc[0])
                exit_px    = float(grp.iloc[-1])
                peak_px    = float(grp.max())
                run_ret    = (exit_px / entry_px - 1) * 100
                roll_max   = grp.cummax()
                max_dd     = float(((grp / roll_max) - 1).min() * 100)
                n_pullback = int(sig_pullback.loc[grp.index].sum())
                peak_score = int(align_score.loc[grp.index].max())
                avg_score  = round(float(align_score.loc[grp.index].mean()), 1)
                run_rows.append({
                    "Start":          grp.index[0].date(),
                    "End":            grp.index[-1].date(),
                    "Days":           len(grp),
                    "Entry $":        round(entry_px, 2),
                    "Peak $":         round(peak_px, 2),
                    "Exit $":         round(exit_px, 2),
                    "Run Return %":   round(run_ret, 1),
                    "Max DD %":       round(max_dd, 1),
                    "Pullbacks":      n_pullback,
                    "Peak Score":     peak_score,
                    "Avg Score":      avg_score,
                })
            if run_rows:
                run_df = pd.DataFrame(run_rows).sort_values("Start", ascending=False)
                st.dataframe(run_df.set_index("Start"), use_container_width=True)

# ---------------- Overnight ----------------
if nav == 'Overnight':
    _show_scan_banner()
    st.subheader('Pct Chg Overnight')
    c1, c2, c3 = st.columns([2,1,1])
    with c1:
        tkr_ov = st.text_input('Ticker', value=st.session_state.get('ticker','AAPL')).strip().upper()
    with c2:
        strategy_ov = st.selectbox(
            'Strategy',
            ['ATR change', 'Gap', 'Close-to-Close'],
            index=0,
            key='ov_strategy'
        )
    with c3:
        lastn = st.number_input('Last N signals (table)', value=20, min_value=5, max_value=200, step=5, key='ov_lastn')
    c4, c5, c6 = st.columns(3)
    with c4:
        thr = st.number_input('Threshold (%)', value=0.50 if strategy_ov.startswith('ATR') else 3.0, step=0.1, key='ov_threshold')
    with c5:
        # Allow ATR parameters even when strategy is Gap (as optional filter)
        atr_n = st.number_input('ATR lookback', value=14, min_value=5, max_value=60, step=1, key='ov_atr_n')
    with c6:
        run_ov = st.button('Run Overnight Backtest', key='ov_run')

    # ATR options (can be used as primary strategy or as an additional filter when strategy != 'ATR change')
    col_a, col_b = st.columns(2)
    with col_a:
        is_atr = ('ATR' in str(strategy_ov).upper())
        if is_atr:
            atr_change_type = st.selectbox(
                'ATR change type',
                options=['Absolute', 'Increase only', 'Decrease only'],
                index=0,
                key='ov_atr_change_type'
            )
            atr_thr = float(thr)
            use_atr_filter = False
        else:
            use_atr_filter = st.checkbox('Add ATR change filter', value=False, key='ov_use_atr_filter')
            atr_change_type = st.selectbox(
                'ATR change type',
                options=['Absolute', 'Increase only', 'Decrease only'],
                index=0,
                disabled=not use_atr_filter,
                key='ov_atr_change_type'
            )
            atr_thr = st.number_input('ATR change threshold (%)', value=0.50, step=0.1, disabled=not use_atr_filter, key='ov_atr_threshold')
    with col_b:
        exit_at = st.selectbox('Exit at', options=['Next Open', 'Next Close', 'Both'], index=0, key='ov_exit_at')

    col_sh, _ = st.columns([1,3])
    with col_sh:
        ov_shares = st.number_input('Shares (for $ P&L)', value=100, min_value=1, step=10, key='ov_shares')
    if run_ov:
        df_d = _load_daily_df_adjusted(tkr_ov)
        if df_d is None or df_d.empty:
            st.error('No daily data found for ticker from Parquet/providers.')
        elif not set(['Date','Open','Close']).issubset(set(df_d.columns)):
            st.error('Daily data missing Open/Close columns.')
        else:
            try:
                df_d = df_d.sort_values('Date').reset_index(drop=True)
                # Build primary signal
                sig_primary = _build_signal_mask(df_d, strategy_ov, float(thr), int(atr_n), atr_change_type if strategy_ov.startswith('ATR') else None)
                sig = sig_primary
                # Optional ATR filter in addition to Gap/Close-to-Close
                if (not strategy_ov.startswith('ATR')) and bool(use_atr_filter):
                    sig_atr = _build_signal_mask(df_d, 'ATR change', float(atr_thr), int(atr_n), atr_change_type)
                    sig = (sig_primary & sig_atr)
                ev, eq_ov, eq_close = _overnight_results(df_d, sig, shares=int(ov_shares), atr_n=int(atr_n))

                # ── KPI row ────────────────────────────────────────────────────
                k1, k2, k3, k4, k5 = st.columns(5)
                with k1:
                    st.metric('Trades', f"{len(ev):,}")
                with k2:
                    try:
                        wr_o = float((ev['Next Open %'] > 0).mean() * 100.0)
                        st.metric('Win Rate (Next Open)', f"{wr_o:.1f}%")
                    except Exception:
                        st.metric('Win Rate (Next Open)', '-')
                with k3:
                    try:
                        wr_c = float((ev['Next Close %'] > 0).mean() * 100.0)
                        st.metric('Win Rate (Next Close)', f"{wr_c:.1f}%")
                    except Exception:
                        st.metric('Win Rate (Next Close)', '-')
                with k4:
                    try:
                        total_pnl_o = float(eq_ov.dropna().iloc[-1]) if not eq_ov.empty else 0.0
                        st.metric(f'Total $ P&L (Next Open, {ov_shares}sh)', f"${total_pnl_o:,.0f}")
                    except Exception:
                        st.metric('Total $ P&L (Next Open)', '-')
                with k5:
                    try:
                        total_pnl_c = float(eq_close.dropna().iloc[-1]) if not eq_close.empty else 0.0
                        st.metric(f'Total $ P&L (Next Close, {ov_shares}sh)', f"${total_pnl_c:,.0f}")
                    except Exception:
                        st.metric('Total $ P&L (Next Close)', '-')

                # ── Equity curves ──────────────────────────────────────────────
                try:
                    from plotly.subplots import make_subplots as _mk_ov
                    import plotly.graph_objects as _go_ov
                    fig_ov = _mk_ov(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                                    row_heights=[0.5, 0.5],
                                    subplot_titles=[f'{tkr_ov} Cum $ P&L — Exit Next Open ({ov_shares} shares)',
                                                    f'{tkr_ov} Cum $ P&L — Exit Next Close ({ov_shares} shares)'])
                    fig_ov.add_trace(_go_ov.Scatter(x=eq_ov.index,    y=eq_ov.values,    mode='lines', name='Next Open P&L',  line=dict(color='#26a69a')), row=1, col=1)
                    fig_ov.add_trace(_go_ov.Scatter(x=eq_close.index, y=eq_close.values, mode='lines', name='Next Close P&L', line=dict(color='#ef5350')), row=2, col=1)
                    # shade signal bars
                    for _sd in df_d.loc[sig, 'Date']:
                        try:
                            fig_ov.add_vrect(x0=_sd, x1=_sd + pd.Timedelta(days=1),
                                             fillcolor='rgba(200,200,0,0.07)', line_width=0, row='all', col=1)
                        except Exception:
                            pass
                    title_extra = ''
                    if strategy_ov.startswith('ATR') and atr_change_type:
                        title_extra = f' ({atr_change_type})'
                    if (not strategy_ov.startswith('ATR')) and bool(use_atr_filter):
                        title_extra = f' + ATR {atr_change_type}'
                    try:
                        _tpl_ov = template
                    except Exception:
                        _tpl_ov = 'plotly_dark'
                    fig_ov.update_layout(height=650, title=f'{tkr_ov}: {strategy_ov}{title_extra}',
                                         xaxis_rangeslider_visible=False, template=_tpl_ov,
                                         yaxis_title='Cum $ P&L', yaxis2_title='Cum $ P&L')
                    st.plotly_chart(fig_ov, use_container_width=True)
                except Exception as _e_ov_chart:
                    st.warning(f'Chart error: {_e_ov_chart}')

                # ── Trade log + last-N table ────────────────────────────────────
                lcol2, rcol2 = st.columns([2, 1])
                with lcol2:
                    st.markdown('**Trade Log**')
                    st.dataframe(ev.tail(200), use_container_width=True)
                with rcol2:
                    st.caption(f'Last {int(lastn)} signals')
                    try:
                        cols_show = [c for c in ['Trade Date','Exit Date','Next Open %','Next Close %','P&L Open','P&L Close','Cum P&L Open','Cum P&L Close'] if c in ev.columns]
                        st.dataframe(ev.tail(int(lastn))[cols_show], use_container_width=True)
                    except Exception:
                        st.dataframe(ev.tail(int(lastn)), use_container_width=True)

                # ── Download buttons ────────────────────────────────────────────
                try:
                    cex1, cex2 = st.columns(2)
                    with cex1:
                        st.download_button('Download Trade Log CSV',
                                           data=ev.to_csv(index=False).encode('utf-8'),
                                           file_name=f'{tkr_ov}_overnight_trades.csv', mime='text/csv')
                    with cex2:
                        eqdf_ov = pd.DataFrame({'Date': eq_ov.index, 'Equity_NextOpen': eq_ov.values,
                                                'Equity_NextClose': eq_close.reindex(eq_ov.index, fill_value=0.0).values})
                        st.download_button('Download Equity CSV',
                                           data=eqdf_ov.to_csv(index=False).encode('utf-8'),
                                           file_name=f'{tkr_ov}_overnight_equity.csv', mime='text/csv')
                except Exception:
                    pass
            except Exception as e:
                st.error(f'Overnight backtest error: {e}')
# ── Chart sidebar defaults (overridden when nav == 'Chart') ──────────────────
bb_on = False; bb_len = 20; bb_std = 2.0
show_sr = False; sr_lookback = 50; show_patterns = False
show_vwap = False; vwap_anchor = None
sma_selected = [10, 50, 200]; ema_selected = []
append_today_daily = False; auto_refresh_daily = False
lower_selected = ["Volume", "RSI"]
show_volume = True; show_tx = False; show_turnover = False
show_turnover_ma = True; turnover_ma_len = 30
show_rsi = True; rsi_len = 14
show_sto = False; sto_k = 14; sto_d = 3; sto_smooth = 3
show_macd = False; macd_fast = 12; macd_slow = 26; macd_signal = 9
base_height = 1050; show_price_labels = False
show_only_latest_day = False; hide_overnight = True
minimalist_mode = True; thin_candles = False; show_scan_marks = True
live_on = False; refresh_secs = 15
# ─────────────────────────────────────────────────────────────────────────────
# ---------------- Chart sidebar (nav-conditional) ----------------------------
if nav in ('Chart', 'TradingView', 'Options'):
  with st.sidebar:
    # ── All settings collapsed into one expander ──────────────────────────────
    with st.expander("⚙️ Settings", expanded=False):
        use_polygon = st.checkbox("Use Polygon (env key)", value=True)
        st.session_state['use_polygon'] = use_polygon
        try:
            _poly_env = (os.getenv('POLYGON_API_KEY') or '').strip()
            st.session_state['polygon_api_key'] = _poly_env
        except Exception:
            pass
        force_fresh = st.checkbox("Force fresh fetch", value=False)
        try:
            st.session_state['force_refresh'] = force_fresh
        except Exception:
            pass
        try:
            pref = st.selectbox(
                "Preferred provider",
                ["Auto (best available)", "Bloomberg", "Polygon", "Alpaca", "Tiingo"],
                index=0,
                key="preferred_provider_sel",
            )
            st.session_state['preferred_provider'] = pref
        except Exception:
            pass
        st.divider()
        st.caption("API Keys")
        alp_key_id   = st.text_input("Alpaca Key ID", value="", type="password")
        alp_secret   = st.text_input("Alpaca Secret", value="", type="password")
        alp_data_url = st.text_input("Alpaca Data URL",
                                     value=os.getenv('APCA_API_DATA_URL') or 'https://data.alpaca.markets')
        if alp_key_id:   st.session_state['alpaca_key_id']     = alp_key_id.strip()
        if alp_secret:   st.session_state['alpaca_secret_key']  = alp_secret.strip()
        if alp_data_url: st.session_state['alpaca_data_url']    = alp_data_url.strip()
        tradier_token = st.text_input("Tradier Token",
                                      value=os.getenv('TRADIER_TOKEN') or "", type="password")
        tradier_sandbox = st.checkbox("Tradier Sandbox", value=bool(
            os.getenv('TRADIER_SANDBOX', "").strip() not in ("", "0", "false", "False")))
        if tradier_token: st.session_state['tradier_token']   = tradier_token.strip()
        st.session_state['tradier_sandbox'] = bool(tradier_sandbox)
        st.divider()
        try:
            _pk  = st.session_state.get('polygon_api_key') or os.getenv('POLYGON_API_KEY') or ''
            _ak  = st.session_state.get('alpaca_key_id') or os.getenv('ALPACA_API_KEY_ID') or ''
            _src = st.session_state.get('last_fetch_provider', '-')
            _bbg_ok = HAS_XBBG
            st.caption(
                f"Bloomberg: {'✓' if _bbg_ok else '✗'}  "
                f"Polygon: {'✓' if _pk else '✗'}  "
                f"Alpaca: {'✓' if _ak else '✗'}  "
                f"Source: {_src}"
            )
            if ENV_DIAG_CAPTION: st.caption(ENV_DIAG_CAPTION)
        except Exception:
            pass

    # ── Chart + TradingView: height & live refresh ────────────────────────────
    if nav in ('Chart', 'TradingView'):
        st.markdown("### Chart")
        base_height = st.slider("Chart height (px)", 600, 1600, 1050, 50)
        live_on = st.checkbox("Live updates (intraday)", value=False)
        if live_on:
            refresh_secs = st.number_input(
                "Refresh every (sec)", value=15, min_value=5, max_value=300, step=5
            )
            try:
                st.session_state['force_refresh'] = True
                if intraday:
                    st.markdown(
                        f"<meta http-equiv='refresh' content='{int(refresh_secs)}'>",
                        unsafe_allow_html=True,
                    )
            except Exception:
                pass

    # ── Chart only: overlays, MAs, panels, display ───────────────────────────
    if nav == 'Chart':
        with st.expander("Overlays", expanded=False):
            bb_on  = st.checkbox("Bollinger Bands", value=False)
            bb_len = st.number_input("BB Length", value=20, min_value=5,
                                     max_value=200, step=1, disabled=not bb_on)
            bb_std = st.number_input("BB Std Dev", value=2.0, min_value=0.5,
                                     max_value=4.0, step=0.5, disabled=not bb_on)
            show_sr     = st.checkbox("Support / Resistance", value=False)
            sr_lookback = st.number_input("SR Lookback (bars)", value=50, min_value=5,
                                          max_value=500, step=5, disabled=not show_sr)
            show_patterns = st.checkbox("Candlestick Patterns", value=False)
            show_vwap     = st.checkbox("Anchored VWAP", value=False)
            vwap_anchor   = None
            if show_vwap:
                vwap_anchor = (
                    st.text_input("VWAP Anchor (YYYY-MM-DD HH:MM)", value="")
                    if intraday
                    else st.date_input("VWAP Anchor Date", value=start)
                )

        with st.expander("Moving Averages", expanded=False):
            ma_periods = [10, 21, 50, 100, 150, 200]
            _c1, _c2 = st.columns(2)
            with _c1:
                st.caption("SMA")
                sma_selected = [
                    p for p in ma_periods
                    if st.checkbox(f"SMA {p}", value=(p in [10, 50, 200]), key=f"sma_{p}")
                ]
            with _c2:
                st.caption("EMA")
                ema_selected = [
                    p for p in ma_periods
                    if st.checkbox(f"EMA {p}", value=False, key=f"ema_{p}")
                ]

        with st.expander("Lower Panels", expanded=False):
            lower_opts = [
                "Volume", "Transactions", "Dollar Value Traded",
                "RSI", "Stochastic %K/%D", "MACD (panel)",
            ]
            lower_selected = st.multiselect("Panels", options=lower_opts,
                                            default=["Volume", "RSI"])
            show_volume   = "Volume" in lower_selected
            show_tx       = "Transactions" in lower_selected
            show_turnover = "Dollar Value Traded" in lower_selected
            show_turnover_ma = st.checkbox("Show $ Value MA", value=True,
                                           disabled=not show_turnover)
            turnover_ma_len  = st.number_input("$ Value MA length", value=30,
                                               min_value=5, max_value=250, step=5,
                                               disabled=not show_turnover)
            show_rsi  = "RSI" in lower_selected
            rsi_len   = st.number_input("RSI Length", value=14, min_value=2,
                                        max_value=200, step=1, disabled=not show_rsi)
            show_sto  = "Stochastic %K/%D" in lower_selected
            sto_k     = st.number_input("%K Length", value=14, min_value=2,
                                        max_value=200, step=1, disabled=not show_sto)
            sto_d     = st.number_input("%D Length", value=3, min_value=1,
                                        max_value=50, step=1, disabled=not show_sto)
            sto_smooth = st.number_input("%K Smoothing", value=3, min_value=1,
                                         max_value=50, step=1, disabled=not show_sto)
            show_macd   = "MACD (panel)" in lower_selected
            macd_fast   = st.number_input("MACD Fast",   value=12, min_value=2,
                                          max_value=100, step=1)
            macd_slow   = st.number_input("MACD Slow",   value=26, min_value=2,
                                          max_value=100, step=1)
            macd_signal = st.number_input("MACD Signal", value=9,  min_value=1,
                                          max_value=50,  step=1)

        with st.expander("Display Options", expanded=False):
            append_today_daily   = st.checkbox("Append today's bar (1d)",
                                               value=(interval == "1d"))
            auto_refresh_daily   = st.checkbox("Auto-refresh during market hours",
                                               value=(interval == "1d"))
            show_price_labels    = st.checkbox("Price labels (right axis)", value=False)
            show_only_latest_day = st.checkbox("Show only latest day", value=False)
            hide_overnight       = st.checkbox("Hide overnight gaps", value=True)
            minimalist_mode      = st.checkbox("Minimalist grid/ticks", value=True)
            thin_candles         = st.checkbox("Thin candles (no fill)", value=False)
            show_scan_marks      = st.checkbox("Show scan highlights", value=True)
            if st.button("Clear highlights", key="clr_highlights"):
                try:
                    st.session_state['scan_highlights'] = []
                    st.rerun()
                except Exception:
                    pass
if nav == 'Chart':
    with st.sidebar:
        with st.expander("Backtesting", expanded=False):
            enable_backtest = st.checkbox("Enable Backtest", value=False)
            strategy = None
            if enable_backtest:
                strategy = st.selectbox(
                    "Strategy",
                    [
                        "Price crosses above VWAP",
                        "Price crosses below VWAP",
                        "RSI crosses above 70",
                        "RSI crosses below 30",
                        "MACD crosses above Signal",
                        "MACD crosses below Signal",
                    ],
                    index=0,
                )
else:
    enable_backtest = False
    strategy = None

# theme/template are set globally before nav (apply_theme called at top)

# ── Chart helpers ────────────────────────────────────────────────────────
TARGETS = {"open", "high", "low", "close", "adj close", "adj_close", "adjclose", "volume"}

# --- Data fetch fallback helpers ---
def best_period_for(interval_str: str, desired: Optional[str]) -> str:
    if interval_str == "1m":
        return "7d"
    if interval_str in {"5m", "15m", "30m", "60m", "1h"}:
        return desired if desired in {"5d", "7d", "14d", "30d", "60d"} else "30d"
    return desired or "1y"

# --- TradingView symbol mapper (heuristic) ---
def tv_symbol_for(sym: str) -> str:
    s = (sym or "").strip().upper()
    if ":" in s:
        return s  # already namespaced
    # Common indices
    index_map = {
        "^GSPC": "SP:SPX",
        "SPX": "SP:SPX",
        "^NDX": "NASDAQ:NDX",
        "NDX": "NASDAQ:NDX",
        "^DJI": "DJ:DJI",
        "DJI": "DJ:DJI",
        "^VIX": "CBOE:VIX",
        "VIX": "CBOE:VIX",
    }
    if s in index_map:
        return index_map[s]
    if s.startswith("^"):
        return s[1:]
    # Common futures roots to TradingView continuous front contract
    fut_map = {
        "ES": "CME_MINI:ES1!", "NQ": "CME_MINI:NQ1!", "YM": "CBOT_MINI:YM1!", "RTY": "CME:RTY1!",
        "CL": "NYMEX:CL1!", "NG": "NYMEX:NG1!", "RB": "NYMEX:RB1!", "HO": "NYMEX:HO1!",
        "GC": "COMEX:GC1!", "SI": "COMEX:SI1!", "HG": "COMEX:HG1!",
        "ZC": "CBOT:ZC1!", "ZS": "CBOT:ZS1!", "ZW": "CBOT:ZW1!", "ZM": "CBOT:ZM1!", "ZL": "CBOT:ZL1!",
        "KC": "ICEUS:KC1!", "SB": "ICEUS:SB1!", "CC": "ICEUS:CC1!", "CT": "ICEUS:CT1!", "OJ": "ICEUS:OJ1!",
    }
    root = s.split("=")[0]
    if root in fut_map:
        return fut_map[root]
    # Default to NASDAQ namespace for plain equities; users can change in-widget
    if s.isalpha() and 1 <= len(s) <= 5:
        return f"NASDAQ:{s}"
    return s

# --- Symbol normalization (indices and futures continuous contracts) ---
FUTURES_CONTINUOUS = {
    # Equity index futures (CME)
    "ES", "NQ", "YM", "RTY",
    # Rates (CME)
    "ZN", "ZB", "ZF", "ZT",
    # Energies (NYMEX)
    "CL", "NG", "RB", "HO", "BZ", "BRN",
    # Metals (COMEX)
    "GC", "SI", "HG", "PA", "PL",
    # Ags (CBOT)
    "ZC", "ZS", "ZW", "ZM", "ZL",
    # Softs (ICE)
    "KC", "SB", "CC", "CT", "OJ",
}

def normalize_input_symbol(sym: str) -> str:
    s = (sym or "").strip().upper()
    if s.startswith("^"):
        return s  # caret indices handled downstream
    if "=F" in s or ":" in s:
        return s  # already explicit
    root = s.split()[0]
    if root in FUTURES_CONTINUOUS:
        return root + "=F"  # map to Yahoo continuous contract
    return s

# --- Futures specific-contract parser (e.g., ESZ24, CLX2024) ---
_MONTH_MAP = {
    'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
    'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12
}

_FUTURES_SUFFIX = {
    # Exchange code suffix as used by Yahoo
    # CME group equity index
    'ES': 'CME', 'NQ': 'CME', 'RTY': 'CME', 'YM': 'CBT',
    # Treasuries (CBOT)
    'ZN': 'CBT', 'ZB': 'CBT', 'ZF': 'CBT', 'ZT': 'CBT',
    # Energies (NYMEX)
    'CL': 'NYM', 'NG': 'NYM', 'RB': 'NYM', 'HO': 'NYM', 'BZ': 'NYM', 'BRN': 'NYM',
    # Metals (COMEX/NYMEX)
    'GC': 'CMX', 'SI': 'CMX', 'HG': 'CMX', 'PA': 'NYM', 'PL': 'NYM',
    # Ags (CBOT)
    'ZC': 'CBT', 'ZS': 'CBT', 'ZW': 'CBT', 'ZM': 'CBT', 'ZL': 'CBT',
    # Softs (ICE US)
    'KC': 'NYB', 'SB': 'NYB', 'CC': 'NYB', 'CT': 'NYB', 'OJ': 'NYB',
}

def build_futures_contract_candidates(sym: str):
    s = (sym or '').strip().upper()
    # Pattern: ROOT + MONTH_LETTER + YY or YYYY, e.g., ESZ24, CLX2024
    import re
    m = re.fullmatch(r"([A-Z]{1,3})([FGHJKMNQUVXZ])(\d{2}|\d{4})", s)
    if not m:
        return None
    root, mon, year = m.group(1), m.group(2), m.group(3)
    yy = year[-2:]
    yyyy = ("20" + yy) if len(year) == 2 else year
    suffix = _FUTURES_SUFFIX.get(root)
    candidates = []
    # Yahoo with exchange suffix
    if suffix:
        candidates.append(f"{root}{mon}{yy}.{suffix}")
    # Yahoo without suffix
    candidates.append(f"{root}{mon}{yy}")
    # Continuous fallback
    candidates.append(f"{root}=F")
    # Polygon-style candidates (best-effort)
    candidates.append(f"C:{root}{mon}{yyyy}")
    candidates.append(f"{root}{mon}{yyyy}")
    return candidates

def _fetch_ohlc_uncached(
    ticker: str,
    *,
    interval: str,
    period: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
):
    import pandas as _pd
    from time import sleep as _sleep

    # Normalize interval for providers (Yahoo prefers '60m' over '1h')
    _interval = '60m' if interval == '1h' else interval

    # Collect errors for diagnostics
    try:
        st.session_state['last_fetch_errors'] = []
    except Exception:
        pass

    # Decide order for providers: Polygon -> Alpaca -> yfinance -> yahooquery -> Stooq(1d)

    # Helper: Bloomberg Terminal (institutional-grade, highest priority)
    def _try_bloomberg():
        if not HAS_XBBG:
            return None
        try:
            from xbbg import blp as _blp
            import datetime as _dt
            _bbg_tkr = f"{ticker} US Equity"
            per_days = {
                '1d':1,'5d':5,'7d':7,'14d':14,'30d':30,'60d':60,'90d':90,
                '1mo':30,'3mo':90,'6mo':180,'1y':365,'2y':730,'5y':1825,
                '10y':3650,'max':36500
            }
            _days = per_days.get(str(period or '1y').lower(), 365)
            _now  = _dt.datetime.now()
            _s_dt = _pd.to_datetime(start)  if start else (_now - _dt.timedelta(days=_days))
            _e_dt = _pd.to_datetime(end)    if end   else _now
            if _interval == '1d':
                _df = _blp.bdh(
                    _bbg_tkr,
                    ['PX_OPEN','PX_HIGH','PX_LOW','PX_LAST','PX_VOLUME'],
                    start_date=_s_dt.strftime('%Y-%m-%d'),
                    end_date=_e_dt.strftime('%Y-%m-%d'),
                )
                if _df is None or _df.empty:
                    return None
                if isinstance(_df.columns, _pd.MultiIndex):
                    _df.columns = _df.columns.get_level_values(-1)
                _df = _df.rename(columns={
                    'PX_OPEN':'Open','PX_HIGH':'High','PX_LOW':'Low',
                    'PX_LAST':'Close','PX_VOLUME':'Volume',
                    'px_open':'Open','px_high':'High','px_low':'Low',
                    'px_last':'Close','px_volume':'Volume',
                })
            else:
                _int_map = {'1m':1,'5m':5,'15m':15,'30m':30,'60m':60,'1h':60}
                _bbg_int = _int_map.get(_interval, 1)
                _df = _blp.bdib(
                    _bbg_tkr,
                    dt_start=_s_dt,
                    dt_end=_e_dt,
                    event='TRADE',
                    interval=_bbg_int,
                )
                if _df is None or _df.empty:
                    return None
                if isinstance(_df.columns, _pd.MultiIndex):
                    _df.columns = _df.columns.get_level_values(-1)
                _df = _df.rename(columns={
                    'open':'Open','high':'High','low':'Low',
                    'close':'Close','volume':'Volume',
                })
            _keep = [c for c in ['Open','High','Low','Close','Volume'] if c in _df.columns]
            _df = _df[_keep]
            if _df.empty:
                return None
            try:
                st.session_state['last_fetch_provider'] = 'bloomberg'
            except Exception:
                pass
            return _df
        except Exception as e:
            try:
                st.session_state['last_fetch_errors'].append(f"bloomberg: {e}")
            except Exception:
                pass
            return None

    # Helper: Polygon first if configured
    def _try_polygon():
        # Always use Polygon when a key is available (sidebar/session, st.secrets, or env)
        try:
            api_key = st.session_state.get('polygon_api_key')
            if not api_key:
                try:
                    if hasattr(st, 'secrets') and ('POLYGON_API_KEY' in st.secrets):
                        api_key = st.secrets['POLYGON_API_KEY']
                except Exception:
                    pass
            if not api_key:
                api_key = os.getenv('POLYGON_API_KEY')
        except Exception:
            api_key = os.getenv('POLYGON_API_KEY')
        if not api_key:
            return None
        try:
            try:
                from src.data_providers.polygon_fetch import fetch_polygon_ohlc as _poly_fetch  # type: ignore
            except Exception:
                try:
                    from polygon_fetch import fetch_polygon_ohlc as _poly_fetch  # type: ignore
                except Exception:
                    _poly_fetch = None
            if _poly_fetch is None:
                return None
            dfp = _poly_fetch(ticker, interval=_interval, period=period, start=start, end=end, api_key=api_key)
            if dfp is not None and not dfp.empty:
                try:
                    st.session_state['last_fetch_provider'] = 'polygon'
                except Exception:
                    pass
                return dfp
        except Exception as e:
            try:
                st.session_state['last_fetch_errors'].append(f"polygon: {e}")
            except Exception:
                pass
        return None

    # Helper: Alpaca Market Data API
    def _try_alpaca():
        try:
            # Support both ALPACA_* and APCA_* naming conventions + sidebar overrides
            key_id = None
            secret = None
            data_base = None
            try:
                key_id = st.session_state.get('alpaca_key_id')
                secret = st.session_state.get('alpaca_secret_key')
                data_base = st.session_state.get('alpaca_data_url')
            except Exception:
                pass
            # Env fallback (support multiple common names)
            key_id = key_id or os.getenv('ALPACA_API_KEY_ID') or os.getenv('APCA_API_KEY_ID') or os.getenv('ALPACA_API_KEY')
            secret = secret or os.getenv('ALPACA_API_SECRET_KEY') or os.getenv('APCA_API_SECRET_KEY') or os.getenv('ALPACA_SECRET_KEY')
            if not key_id or not secret:
                return None
            # Data API base URL (not the trading API URL)
            data_base = data_base or os.getenv('APCA_API_DATA_URL') or 'https://data.alpaca.markets'
            tf_map = {
                '1m': '1Min', '5m': '5Min', '15m': '15Min', '30m': '30Min', '60m': '1Hour', '1h': '1Hour', '1d': '1Day'
            }
            tf = tf_map.get(_interval, '1Day')

            # Build time window
            _start = start
            _end = end
            if not _start or not _end:
                # Derive from period
                now = datetime.utcnow().replace(tzinfo=timezone.utc)
                end_dt = now
                per_days = {
                    '1d': 1, '5d': 5, '7d': 7, '14d': 14, '30d': 30, '60d': 60, '90d': 90,
                    '1mo': 30, '3mo': 90, '6mo': 180, '1y': 365, '2y': 730, '5y': 1825, '10y': 3650, 'max': 36500
                }
                days = per_days.get(str(period).lower() if period else '1y', 365)
                start_dt = end_dt - timedelta(days=days)
                _start = start_dt.isoformat().replace('+00:00', 'Z')
                _end = end_dt.isoformat().replace('+00:00', 'Z')
            else:
                # Ensure RFC3339
                try:
                    _start = pd.to_datetime(_start).tz_localize('UTC', nonexistent='shift_forward', ambiguous='NaT').isoformat().replace('+00:00', 'Z')
                except Exception:
                    _start = str(_start)
                try:
                    _end = pd.to_datetime(_end).tz_localize('UTC', nonexistent='shift_forward', ambiguous='NaT').isoformat().replace('+00:00', 'Z')
                except Exception:
                    _end = str(_end)

            import requests as _rq
            url = f"{data_base.rstrip('/')}/v2/stocks/{ticker}/bars"
            params = {"timeframe": tf, "start": _start, "end": _end, "limit": 10000}
            headers = {"APCA-API-KEY-ID": key_id, "APCA-API-SECRET-KEY": secret}
            resp = _rq.get(url, params=params, headers=headers, timeout=20)
            if resp.status_code != 200:
                try:
                    st.session_state['last_fetch_errors'].append(f"alpaca: HTTP {resp.status_code} {resp.text[:120]}")
                except Exception:
                    pass
                return None
            js = resp.json() or {}
            bars = js.get('bars') or []
            if not bars:
                return None
            dfa = pd.DataFrame(bars)
            # Columns typically: t (ISO time), o,h,l,c,v
            if 't' in dfa.columns:
                dfa['t'] = pd.to_datetime(dfa['t'], errors='coerce')
                dfa = dfa.set_index('t')
            rename = {'o':'Open','h':'High','l':'Low','c':'Close','v':'Volume'}
            dfa = dfa.rename(columns=rename)
            keep = [c for c in ['Open','High','Low','Close','Volume'] if c in dfa.columns]
            dfa = dfa[keep]
            if dfa.empty:
                return None
            try:
                st.session_state['last_fetch_provider'] = 'alpaca'
            except Exception:
                pass
            return dfa
        except Exception as e:
            try:
                st.session_state['last_fetch_errors'].append(f"alpaca: {e}")
            except Exception:
                pass
            return None

    # Helper: yahooquery block
    # Helper: Tiingo (intraday and daily)
    def _try_tiingo():
        try:
            token = os.getenv('TIINGO_API_KEY') or os.getenv('TIINGO_TOKEN')
            if not token:
                return None
            import requests as _rq
            base = 'https://api.tiingo.com'
            # Decide endpoint by interval
            if _interval == '1d':
                # Daily endpoint
                url = f"{base}/tiingo/daily/{ticker}/prices"
                params = {}
                if start:
                    params['startDate'] = str(pd.to_datetime(start).date())
                if end:
                    params['endDate'] = str(pd.to_datetime(end).date())
                params['token'] = token
                resp = _rq.get(url, params=params, timeout=20)
                if resp.status_code != 200:
                    try:
                        st.session_state['last_fetch_errors'].append(f"tiingo daily: HTTP {resp.status_code}")
                    except Exception:
                        pass
                    return None
                arr = resp.json() or []
                if not arr:
                    return None
                dft = pd.DataFrame(arr)
                # Fields: date, open, high, low, close, volume, adjClose, ...
                if 'date' in dft.columns:
                    dft['date'] = pd.to_datetime(dft['date'], errors='coerce')
                    dft = dft.set_index('date')
                rename = {'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'}
                dft = dft.rename(columns=rename)
                keep = [c for c in ['Open','High','Low','Close','Volume'] if c in dft.columns]
                dft = dft[keep]
            else:
                # Intraday via IEX endpoint, resampled by Tiingo
                tf_map = {'1m':'1min','5m':'5min','15m':'15min','30m':'30min','60m':'60min','1h':'60min'}
                tf = tf_map.get(_interval, '5min')
                url = f"{base}/iex/{ticker}/prices"
                params = {'resampleFreq': tf, 'columns': 'open,high,low,close,volume', 'token': token}
                # Tiingo accepts startDate and endDate (dates) for intraday as well
                if start:
                    params['startDate'] = str(pd.to_datetime(start).date())
                if end:
                    params['endDate'] = str(pd.to_datetime(end).date())
                resp = _rq.get(url, params=params, timeout=20)
                if resp.status_code != 200:
                    try:
                        st.session_state['last_fetch_errors'].append(f"tiingo iex: HTTP {resp.status_code}")
                    except Exception:
                        pass
                    return None
                arr = resp.json() or []
                if not arr:
                    return None
                dft = pd.DataFrame(arr)
                # Fields: date (ISO), open, high, low, close, volume
                if 'date' in dft.columns:
                    dft['date'] = pd.to_datetime(dft['date'], errors='coerce')
                    dft = dft.set_index('date')
                rename = {'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume'}
                dft = dft.rename(columns=rename)
                keep = [c for c in ['Open','High','Low','Close','Volume'] if c in dft.columns]
                dft = dft[keep]
            if dft.empty:
                return None
            try:
                st.session_state['last_fetch_provider'] = 'tiingo'
            except Exception:
                pass
            return dft
        except Exception as e:
            try:
                st.session_state['last_fetch_errors'].append(f"tiingo: {e}")
            except Exception:
                pass
            return None

    # Helper: Twelve Data (intraday and daily)
    def _try_twelvedata():
        try:
            token = os.getenv('TWELVEDATA_API_KEY')
            if not token:
                return None
            import requests as _rq
            base = 'https://api.twelvedata.com'
            tf_map = {'1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min', '60m': '60min', '1h': '1h', '1d': '1day'}
            tf = tf_map.get(_interval, _interval)
            params = {'symbol': ticker, 'interval': tf, 'apikey': token, 'outputsize': 5000, 'order': 'ASC'}
            if not period and start:
                try:
                    params['start_date'] = pd.to_datetime(start).strftime('%Y-%m-%d %H:%M:%S')
                except Exception:
                    params['start_date'] = str(start)
            if not period and end:
                try:
                    params['end_date'] = pd.to_datetime(end).strftime('%Y-%m-%d %H:%M:%S')
                except Exception:
                    params['end_date'] = str(end)
            resp = _rq.get(f"{base}/time_series", params=params, timeout=20)
            if resp.status_code != 200:
                try:
                    st.session_state['last_fetch_errors'].append(f"twelvedata: HTTP {resp.status_code}")
                except Exception:
                    pass
                return None
            js = resp.json() or {}
            if js.get('status') == 'error':
                try:
                    st.session_state['last_fetch_errors'].append(f"twelvedata: {js.get('message','error')}")
                except Exception:
                    pass
                return None
            data = js.get('values') or js.get('data') or []
            if not data:
                return None
            dft = pd.DataFrame(data)
            rename = {'open':'Open','high':'High','low':'Low','close':'Close','volume':'Volume','datetime':'Date'}
            dft = dft.rename(columns=rename)
            if 'Date' in dft.columns:
                dft['Date'] = pd.to_datetime(dft['Date'], errors='coerce')
                dft = dft.set_index('Date')
            keep = [c for c in ['Open','High','Low','Close','Volume'] if c in dft.columns]
            dft = dft[keep].apply(pd.to_numeric, errors='coerce')
            dft = dft.sort_index()
            if dft.empty:
                return None
            try:
                st.session_state['last_fetch_provider'] = 'twelvedata'
            except Exception:
                pass
            return dft
        except Exception as e:
            try:
                st.session_state['last_fetch_errors'].append(f"twelvedata: {e}")
            except Exception:
                pass
            return None

    def _try_yahooquery():
        try:
            from yahooquery import Ticker as _YQTicker
            yq = _YQTicker(ticker)
            if period:
                yq_df = yq.history(period=period, interval=_interval)
            else:
                yq_df = yq.history(start=start, end=end, interval=_interval)
            if yq_df is None or yq_df.empty:
                return None
            # Flatten MultiIndex to DatetimeIndex if needed
            if isinstance(yq_df.index, _pd.MultiIndex):
                if 'date' in yq_df.index.names:
                    yq_df = yq_df.reset_index().set_index('date')
                else:
                    yq_df = yq_df.reset_index()
            if not isinstance(yq_df.index, _pd.DatetimeIndex) and 'date' in yq_df.columns:
                yq_df['date'] = _pd.to_datetime(yq_df['date'])
                yq_df = yq_df.set_index('date')
            rename = {
                "open": "Open","high": "High","low": "Low","close": "Close",
                "adjclose": "Adj Close","adj_close": "Adj Close","volume": "Volume",
            }
            yq_df = yq_df.rename(columns=lambda c: rename.get(str(c).lower(), str(c).title()))
            keep = [c for c in ["Open","High","Low","Close","Adj Close","Volume"] if c in yq_df.columns]
            if keep:
                yq_df = yq_df[keep]
            try:
                st.session_state['last_fetch_provider'] = 'yahooquery'
            except Exception:
                pass
            return yq_df
        except Exception as e:
            try:
                st.session_state['last_fetch_errors'].append(f"yahooquery: {e}")
            except Exception:
                pass
            return None

    # ── Provider waterfall (Bloomberg leads in auto; user pref overrides order) ──
    try:
        pref = st.session_state.get('preferred_provider')
    except Exception:
        pref = None

    def _auto_providers():
        """Auto order: Bloomberg (institutional) > Polygon > Tiingo > Alpaca > TwelveData"""
        _p = []
        if HAS_XBBG:
            _p.append(("Bloomberg", _try_bloomberg))
        _p += [("Polygon", _try_polygon), ("Tiingo", _try_tiingo),
               ("Alpaca", _try_alpaca), ("TwelveData", _try_twelvedata)]
        return _p

    if pref and isinstance(pref, str) and pref.startswith("Bloomberg"):
        providers = [("Bloomberg", _try_bloomberg), ("Polygon", _try_polygon),
                     ("Tiingo", _try_tiingo), ("Alpaca", _try_alpaca), ("TwelveData", _try_twelvedata)]
    elif pref and isinstance(pref, str) and pref.startswith("Polygon"):
        providers = [("Polygon", _try_polygon), ("Bloomberg", _try_bloomberg),
                     ("Tiingo", _try_tiingo), ("Alpaca", _try_alpaca), ("TwelveData", _try_twelvedata)]
    elif pref and isinstance(pref, str) and pref.startswith("Alpaca"):
        providers = [("Alpaca", _try_alpaca), ("Bloomberg", _try_bloomberg),
                     ("Polygon", _try_polygon), ("Tiingo", _try_tiingo), ("TwelveData", _try_twelvedata)]
    elif pref and isinstance(pref, str) and pref.startswith("Tiingo"):
        providers = [("Tiingo", _try_tiingo), ("Bloomberg", _try_bloomberg),
                     ("Polygon", _try_polygon), ("Alpaca", _try_alpaca), ("TwelveData", _try_twelvedata)]
    elif pref and isinstance(pref, str) and pref.lower().startswith("twelvedata"):
        providers = [("TwelveData", _try_twelvedata), ("Bloomberg", _try_bloomberg),
                     ("Polygon", _try_polygon), ("Alpaca", _try_alpaca), ("Tiingo", _try_tiingo)]
    else:
        providers = _auto_providers()
    for _name, _fn in providers:
        try:
            _df = _fn()
            if isinstance(_df, _pd.DataFrame) and not _df.empty:
                return _df
        except Exception:
            pass

    # yfinance with small retries (sanitize TZ-aware start/end)
    for attempt in range(3):
        try:
            _prepost = (_interval != "1d")
            _start_clean = None
            _end_clean = None
            if start:
                _ts = pd.to_datetime(start, errors='coerce')
                if _ts is not pd.NaT:
                    if _ts.tzinfo is not None:
                        _ts = _ts.tz_convert('UTC').tz_localize(None)
                    _start_clean = _ts
            if end:
                _te = pd.to_datetime(end, errors='coerce')
                if _te is not pd.NaT:
                    if _te.tzinfo is not None:
                        _te = _te.tz_convert('UTC').tz_localize(None)
                    _end_clean = _te
            if _interval == '1d':
                if period:
                    df = yf.download(ticker, period=period, interval=_interval, progress=False, auto_adjust=False, threads=False, prepost=False)
                else:
                    df = yf.download(
                        ticker,
                        start=_start_clean.date() if _start_clean is not None else None,
                        end=_end_clean.date() if _end_clean is not None else None,
                        interval=_interval,
                        progress=False,
                        auto_adjust=False,
                        threads=False,
                        prepost=False,
                    )
            else:
                # yfinance is finicky with start/end for intraday; prefer period when possible
                _period = period if period else '7d'
                df = yf.download(
                    ticker,
                    period=_period,
                    interval=_interval,
                    progress=False,
                    auto_adjust=False,
                    threads=False,
                    prepost=_prepost,
                )
            if isinstance(df, _pd.DataFrame) and not df.empty:
                try:
                    st.session_state['last_fetch_provider'] = 'yfinance'
                except Exception:
                    pass
                return df
        except Exception as e:
            try:
                st.session_state['last_fetch_errors'].append(f"yfinance attempt {attempt+1} (auto_adjust=False): {e}")
            except Exception:
                pass
        _sleep(0.6 * (attempt + 1))

    # Retry with auto_adjust=True once
    try:
        _prepost = (_interval != "1d")
        if period:
            df = yf.download(ticker, period=period, interval=_interval, progress=False, auto_adjust=True, threads=False, prepost=_prepost)
        else:
            df = yf.download(ticker, start=start, end=end, interval=_interval, progress=False, auto_adjust=True, threads=False, prepost=_prepost)
        if isinstance(df, _pd.DataFrame) and not df.empty:
            return df
    except Exception as e:
        try:
            st.session_state['last_fetch_errors'].append(f"yfinance (auto_adjust=True): {e}")
        except Exception:
            pass

    # 4) yahooquery fallback
    yq_df = _try_yahooquery()
    if isinstance(yq_df, _pd.DataFrame) and not yq_df.empty:
        if isinstance(yq_df, _pd.DataFrame) and not yq_df.empty:
            try:
                st.session_state['last_fetch_provider'] = 'yahooquery'
            except Exception:
                pass
        return yq_df

    # 5) Stooq daily fallback via pandas-datareader (only for 1d)
    try:
        if _interval == '1d':
            try:
                from pandas_datareader import data as _pdr
            except Exception as e:
                try:
                    st.session_state['last_fetch_errors'].append(f"pandas-datareader not available for Stooq fallback: {e}")
                except Exception:
                    pass
                return _pd.DataFrame()

            # Determine date range
            _start = None
            _end = None
            try:
                if start and end:
                    _start = _pd.to_datetime(start)
                    _end = _pd.to_datetime(end)
                elif period:
                    now = _pd.Timestamp.utcnow().normalize()
                    per_map = {
                        '7d': 7, '14d': 14, '30d': 30, '60d': 60, '90d': 90,
                        '1mo': 30, '3mo': 90, '6mo': 180, '1y': 365, '2y': 730, '5y': 1825, '10y': 3650, 'max': 36500
                    }
                    days = per_map.get(str(period).lower(), 365)
                    _start = now - _pd.Timedelta(days=days)
                    _end = now + _pd.Timedelta(days=1)
                else:
                    _end = _pd.Timestamp.utcnow().normalize() + _pd.Timedelta(days=1)
                    _start = _end - _pd.Timedelta(days=365)
            except Exception:
                pass

            sym = str(ticker).strip().upper()
            stooq_symbol = sym + '.US' if sym.isalpha() else sym
            try:
                stq = _pdr.DataReader(stooq_symbol, 'stooq', start=_start, end=_end)
                if isinstance(stq, _pd.DataFrame) and not stq.empty:
                    stq = stq.sort_index()
                    # ensure yahoo-like column casing
                    stq = stq.rename(columns={
                        'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'
                    })
                    try:
                        st.session_state['last_fetch_provider'] = 'stooq'
                    except Exception:
                        pass
                    return stq
            except Exception as e:
                try:
                    st.session_state['last_fetch_errors'].append(f"stooq fallback: {e}")
                except Exception:
                    pass
    except Exception:
        pass


@st.cache_data(show_spinner=False, ttl=30)
def _fetch_ohlc_cached(
    ticker: str,
    *,
    interval: str,
    period: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
    cache_buster: Optional[str] = None,
):
    # cache_buster participates in cache key but is unused otherwise
    return _fetch_ohlc_uncached(ticker, interval=interval, period=period, start=start, end=end)

# Ensure fetch_ohlc_with_fallback is defined before any call sites
def fetch_ohlc_with_fallback(
    ticker: str,
    *,
    interval: str,
    period: Optional[str] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
):
    """Cached fetch that retries once with a unique cache key to avoid cached-empty results."""
    try:
        force = bool(st.session_state.get('force_refresh'))
    except Exception:
        force = False
    first_buster = (str(time.time()) if force else None)
    df = _fetch_ohlc_cached(ticker, interval=interval, period=period, start=start, end=end, cache_buster=first_buster)
    try:
        import pandas as _pd
        is_empty = (df is None) or (isinstance(df, _pd.DataFrame) and df.empty)
    except Exception:
        is_empty = df is None
    if not is_empty:
        return df
    buster = str(time.time())
    return _fetch_ohlc_cached(ticker, interval=interval, period=period, start=start, end=end, cache_buster=buster)


# --- Anchored VWAP ---
def anchored_vwap(df: pd.DataFrame, anchor_idx: int = 0, price_col: str = "Close", vol_col: str = "Volume") -> pd.Series:
    """
    Calculate Anchored VWAP from a given anchor index (row number).
    Returns a Series with NaN before anchor, and VWAP from anchor forward.
    """
    if price_col not in df.columns or vol_col not in df.columns:
        raise ValueError(f"Columns {price_col} and {vol_col} must be in DataFrame")
    v = df[vol_col].astype(float)
    p = df[price_col].astype(float)
    vwap = pd.Series(np.nan, index=df.index)
    if anchor_idx < 0 or anchor_idx >= len(df):
        return vwap
    cum_vol = v.iloc[anchor_idx:].cumsum()
    cum_pv = (p.iloc[anchor_idx:] * v.iloc[anchor_idx:]).cumsum()
    vwap.iloc[anchor_idx:] = cum_pv / cum_vol
    return vwap

def pick_close_key(cols) -> str | None:
    candidates = {c.lower(): c for c in cols}
    for key in ("close", "adj close", "adj_close", "adjclose"):
        if key in candidates:
            return candidates[key]
    return None

def sma(series: pd.Series, n: int) -> pd.Series:
    return series.rolling(n, min_periods=1).mean().rename(f"SMA({n})")

def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean().rename(f"EMA({n})")

def stoch_kd(high: pd.Series, low: pd.Series, close: pd.Series, k_len: int = 14, d_len: int = 3, smooth_k: int = 3):
    lowest = low.rolling(k_len, min_periods=1).min()
    highest = high.rolling(k_len, min_periods=1).max()
    k = ((close - lowest) / (highest - lowest).replace(0, np.nan) * 100.0).fillna(50.0)
    if smooth_k > 1:
        k = k.rolling(smooth_k, min_periods=1).mean()
    d = k.rolling(d_len, min_periods=1).mean()
    return k.rename(f"%K({k_len})"), d.rename(f"%D({d_len})")

def bbands(series: pd.Series, length: int = 20, n_std: float = 2.0):
    mid = series.rolling(length, min_periods=1).mean()
    sd = series.rolling(length, min_periods=1).std(ddof=0)
    upper = mid + n_std * sd
    lower = mid - n_std * sd
    return lower.rename(f"BB Lower({length},{n_std})"), mid.rename(f"BB Mid({length})"), upper.rename(f"BB Upper({length},{n_std})")
    
# --- MACD ---
def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    fast_ema = series.ewm(span=fast, adjust=False).mean()
    slow_ema = series.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line.rename("MACD"), signal_line.rename("Signal"), hist.rename("Hist")
    
# --- Support/Resistance ---
def find_support_resistance(series: pd.Series, lookback: int = 50):
    # Simple local min/max finder
    supports = []
    resistances = []
    for i in range(lookback, len(series) - lookback):
        window = series[i - lookback:i + lookback + 1]
        if series[i] == window.min():
            supports.append((series.index[i], series[i]))
        if series[i] == window.max():
            resistances.append((series.index[i], series[i]))
    return supports, resistances
    
# --- Candlestick Patterns ---
def detect_patterns(df: pd.DataFrame):
    patterns = {}
    # If pandas-ta is unavailable, return empty/zero series so the app still works
    if not HAS_PANDAS_TA or pta is None:
        for pat in ["cdl_hammer", "cdl_engulfing", "cdl_doji", "cdl_morningstar", "cdl_shootingstar"]:
            patterns[pat] = pd.Series([0] * len(df), index=df.index)
        return patterns

    # Use a few common patterns (guard each call)
    for pat in ["cdl_hammer", "cdl_engulfing", "cdl_doji", "cdl_morningstar", "cdl_shootingstar"]:
        try:
            patterns[pat] = getattr(pta, pat)(df["Open"], df["High"], df["Low"], df["Close"])
        except Exception:
            patterns[pat] = pd.Series([0] * len(df), index=df.index)
    return patterns

def infer_pad_timedelta(interval: str) -> timedelta:
    mapping = {"1d": timedelta(days=1), "1h": timedelta(hours=1), "30m": timedelta(minutes=30),
               "15m": timedelta(minutes=15), "5m": timedelta(minutes=5), "1m": timedelta(minutes=1)}
    return mapping.get(interval, timedelta(days=1))

def extend_right_edge(fig: go.Figure, last_ts, interval: str, rows: int):
    pad = infer_pad_timedelta(interval) * 3
    pad_x = last_ts + pad
    for r in range(1, rows + 1):
        fig.add_trace(go.Scatter(x=[pad_x], y=[None], mode="markers",
                                 marker_opacity=0, showlegend=False, hoverinfo="skip"), row=r, col=1)

# ---------------- Historical Stats (Polygon flat files) ----------------
@st.cache_data(show_spinner=False)
def _normalize_host_path(p: str | None) -> str | None:
    if p is None:
        return None
    try:
        import os, re
        s = str(p).strip().strip('"').strip("'")
        # If running on non-Windows and given a Windows path, map to WSL style
        if os.name != 'nt' and re.match(r'^[a-zA-Z]:[\\/]', s):
            drive = s[0].lower()
            rest = s[2:].replace('\\','/')
            s = f"/mnt/{drive}/{rest.lstrip('/')}"
        return os.path.normpath(s)
    except Exception:
        return p

@st.cache_data(show_spinner=False)
def _autofind_parquet_path(ticker: str) -> str | None:
    """
    Try to locate a per-ticker parquet like <TICKER>.parquet in common folders
    without surfacing paths in the UI.
    Order:
      1) PER_TICKER_PARQUET_DIR env var
      2) Common Windows Documents paths (per_ticker_daily_tech, per_ticker_daily)
      3) WSL equivalents
      4) ./per_ticker_daily_tech or ./per_ticker_daily under CWD
    Returns a normalized path or None.
    """
    import os
    from pathlib import Path
    t = ticker.strip().upper()
    fname = f"{t}.parquet"
    # Also consider alternative filename variants for symbols with punctuation
    alt_names = {fname}
    base = t
    for repl in (('.', '_'), ('/', '_'), ('-', '_'), (' ', '_')):
        base = t.replace(repl[0], repl[1])
        alt_names.add(f"{base}.parquet")
    alt_names.add(f"{t.replace('.', '')}.parquet")
    # 1) Env var
    env_dir = os.environ.get('PER_TICKER_PARQUET_DIR')
    if env_dir:
        # Try exact and alternate names
        for nm in list(alt_names):
            p = _normalize_host_path(os.path.join(env_dir, nm))
            if p and os.path.exists(p):
                return p
    # 2) Windows Documents (standard + 'Visual Code' layout)
    # App directory (so it works regardless of launch CWD)
    try:
        _APP_DIR = str(Path(__file__).resolve().parent)
    except Exception:
        _APP_DIR = ''

    common_dirs = [
        # Standard Documents
        f"C:/Users/{os.environ.get('USERNAME','')}/Documents/Polygon Data/per_ticker_daily_tech",
        f"C:/Users/{os.environ.get('USERNAME','')}/Documents/Polygon Data/per_ticker_daily",
        f"C:/Users/{os.environ.get('USERNAME','')}/Documents/Polygon Data/per_ticker_daily_ohlcv",
        # Visual Code workspace under Documents
        f"C:/Users/{os.environ.get('USERNAME','')}/Documents/Visual Code/Polygon Data/per_ticker_daily_tech",
        f"C:/Users/{os.environ.get('USERNAME','')}/Documents/Visual Code/Polygon Data/per_ticker_daily",
        f"C:/Users/{os.environ.get('USERNAME','')}/Documents/Visual Code/Polygon Data/per_ticker_daily_ohlcv",
        # Next to the app file
        (os.path.join(_APP_DIR, 'per_ticker_daily_tech') if _APP_DIR else ''),
        (os.path.join(_APP_DIR, 'per_ticker_daily') if _APP_DIR else ''),
        (os.path.join(_APP_DIR, 'per_ticker_daily_ohlcv') if _APP_DIR else ''),
    ]
    # 3) WSL equivalents
    common_dirs += [
        f"/mnt/c/Users/{os.environ.get('USERNAME','').lower()}/Documents/Polygon Data/per_ticker_daily_tech",
        f"/mnt/c/Users/{os.environ.get('USERNAME','').lower()}/Documents/Polygon Data/per_ticker_daily",
        f"/mnt/c/Users/{os.environ.get('USERNAME','').lower()}/Documents/Polygon Data/per_ticker_daily_ohlcv",
        f"/mnt/c/Users/{os.environ.get('USERNAME','').lower()}/Documents/Visual Code/Polygon Data/per_ticker_daily_tech",
        f"/mnt/c/Users/{os.environ.get('USERNAME','').lower()}/Documents/Visual Code/Polygon Data/per_ticker_daily",
        f"/mnt/c/Users/{os.environ.get('USERNAME','').lower()}/Documents/Visual Code/Polygon Data/per_ticker_daily_ohlcv",
    ]
    # 4) Local under CWD
    common_dirs += [
        str(Path.cwd() / 'per_ticker_daily_tech'),
        str(Path.cwd() / 'per_ticker_daily'),
        str(Path.cwd() / 'per_ticker_daily_ohlcv'),
    ]
    for d in common_dirs:
        # Try direct matches first
        for nm in list(alt_names):
            p = _normalize_host_path(os.path.join(d, nm))
            if p and os.path.exists(p):
                return p
        # Fallback: scan directory and match by alphanumeric-only stem
        try:
            from pathlib import Path as _P
            dd = _P(_normalize_host_path(d))
            if dd.exists():
                want = ''.join(ch for ch in t if ch.isalnum())
                for fp in dd.glob('*.parquet'):
                    stem = fp.stem
                    if stem.endswith('.csv'):
                        stem = stem[:-4]
                    key = ''.join(ch for ch in stem.upper() if ch.isalnum())
                    if key == want:
                        return str(fp)
        except Exception:
            pass
    return None

@st.cache_data(show_spinner=False)
def _autofind_report_excel_path(ticker: str) -> str | None:
    """
    Try to locate <TICKER>_technicals.xlsx without showing paths in the UI.
    Order:
      1) POLYGON_REPORTS_DIR env var
      2) Common Windows path in user's Documents
      3) WSL-style equivalent under /mnt/c
      4) ./reports relative to CWD
      5) Recursive search under user's Documents Polygon Data
    Returns a normalized path or None.
    """
    import os, glob, getpass
    t = ticker.strip().upper()
    fname = f"{t}_technicals.xlsx"
    # 1) Env var
    env_dir = os.environ.get('POLYGON_REPORTS_DIR')
    if env_dir:
        p = _normalize_host_path(os.path.join(env_dir, fname))
        if p and os.path.exists(p):
            return p
    # 2) Common Windows path
    user = getpass.getuser()
    win_dir = f"C:/Users/{user}/Documents/Polygon Data/reports"
    p = _normalize_host_path(os.path.join(win_dir, fname))
    if p and os.path.exists(p):
        return p
    # 3) WSL equivalent
    wsl_dir = f"/mnt/c/Users/{user}/Documents/Polygon Data/reports"
    p = _normalize_host_path(os.path.join(wsl_dir, fname))
    if p and os.path.exists(p):
        return p
    # 4) ./reports relative to CWD
    local_dir = os.path.join(os.getcwd(), 'reports')
    p = _normalize_host_path(os.path.join(local_dir, fname))
    if p and os.path.exists(p):
        return p
    # 5) Recursive search under user's Documents Polygon Data
    base_dirs = [f"C:/Users/{user}/Documents/Polygon Data", f"/mnt/c/Users/{user}/Documents/Polygon Data"]
    for bd in base_dirs:
        root = _normalize_host_path(bd)
        if root and os.path.exists(root):
            pattern = os.path.join(root, '**', fname)
            matches = glob.glob(pattern, recursive=True)
            if matches:
                return _normalize_host_path(matches[0])
    return None

@st.cache_data(show_spinner=False)
def _load_polygon_daily_for_ticker(
    data_root: str,
    ticker: str,
    reports_dir: str | None = None,
    technicals_script: str | None = None,
    auto_generate_report: bool = True,
    excel_override: object | None = None,
    excel_path_override: str | None = None,
    allow_yahoo_fallback: bool = False,
) -> pd.DataFrame:
    """
    Load per-ticker daily parquet from a Polygon flat-files export.
    Expects a file like `<data_root>/per_ticker_daily/<TICKER>.parquet`.
    Normalizes columns to ['Date','Open','High','Low','Close','Volume'] with Date as datetime.
    Returns sorted ascending by Date.
    """
    import os
    import pandas as pd

    t = ticker.strip().upper()
    df = None
    # 0) Prefer per-ticker parquet if present (already contains technicals)
    try:
        pq_path = _autofind_parquet_path(t)
        if pq_path and os.path.exists(pq_path):
            df_pq = pd.read_parquet(pq_path)
            if isinstance(df_pq, pd.DataFrame) and not df_pq.empty:
                # Normalize to expected columns
                cols = {str(c).lower(): c for c in df_pq.columns}
                date_col = cols.get('timestamp') or cols.get('date')
                if date_col is not None:
                    out = pd.DataFrame({'Date': pd.to_datetime(df_pq[date_col], errors='coerce')})
                    for c in ('Open','High','Low','Close','Volume'):
                        src = cols.get(c.lower())
                        if src is not None and src in df_pq.columns:
                            out[c] = pd.to_numeric(df_pq[src], errors='coerce')
                    out = out.dropna(subset=['Date','Close']) if 'Close' in out.columns else out.dropna(subset=['Date'])
                    out = out.sort_values('Date').reset_index(drop=True)
                    return out
    except Exception:
        pass
    # 0) If a file-like Excel was provided (uploaded), parse it first
    if excel_override is not None:
        try:
            try:
                import openpyxl  # noqa: F401
                xl = pd.ExcelFile(excel_override, engine='openpyxl')
            except ImportError:
                xl = pd.ExcelFile(excel_override)
            def _try_sheet_to_ohlc(_df: pd.DataFrame) -> pd.DataFrame | None:
                d = _df.copy()
                try:
                    if not isinstance(d.index, pd.RangeIndex):
                        idx_as_date = pd.to_datetime(d.index, errors='coerce')
                        if idx_as_date.notna().mean() > 0.8 and 'Date' not in d.columns:
                            d.insert(0, 'Date', idx_as_date)
                            d.reset_index(drop=True, inplace=True)
                except Exception:
                    pass
                rename_map = {}
                for c in list(d.columns):
                    lc = str(c).strip().lower().replace(' ', '').replace('_','')
                    if lc in {'date','day','sessiondate','windowstart','t','timestamp'}:
                        rename_map[c] = 'Date'
                    elif lc in {'open','o'}:
                        rename_map[c] = 'Open'
                    elif lc in {'high','h'}:
                        rename_map[c] = 'High'
                    elif lc in {'low','l'}:
                        rename_map[c] = 'Low'
                    elif lc in {'close','c'}:
                        rename_map[c] = 'Close'
                    elif lc in {'adjclose','adjustedclose','adj_close'}:
                        rename_map[c] = 'Adj Close'
                    elif lc in {'volume','v','vol'}:
                        rename_map[c] = 'Volume'
                if rename_map:
                    d = d.rename(columns=rename_map)
                if 'Date' not in d.columns and len(d.columns) > 0:
                    try:
                        cand = pd.to_datetime(d.iloc[:,0], errors='coerce')
                        if cand.notna().mean() > 0.8:
                            d.insert(0, 'Date', cand)
                    except Exception:
                        pass
                if 'Close' not in d.columns and 'Adj Close' in d.columns:
                    d['Close'] = pd.to_numeric(d['Adj Close'], errors='coerce')
                for col in ['Open','High','Low','Close','Volume']:
                    if col in d.columns:
                        d[col] = pd.to_numeric(d[col], errors='coerce')
                if 'Date' in d.columns and 'Close' in d.columns:
                    d['Date'] = pd.to_datetime(d['Date'], errors='coerce', utc=True)
                    d = d.dropna(subset=['Date','Close'])
                    d['Date'] = d['Date'].dt.tz_convert(None) if hasattr(d['Date'].dt, 'tz_convert') else d['Date']
                    d = d.sort_values('Date').reset_index(drop=True)
                    return d
                return None
            best = None
            for sheet in xl.sheet_names:
                try:
                    cand = _try_sheet_to_ohlc(xl.parse(sheet))
                    if cand is not None and len(cand) >= 5:
                        best = cand
                        if str(sheet).lower() in {'daily','prices','ohlc','price','history'}:
                            break
                except Exception:
                    continue
            if best is not None:
                return best
        except Exception as e:
            raise RuntimeError(f"Failed reading uploaded Excel: {e}")

    # 1) Primary source: reports Excel (e.g., reports/AAPL_technicals.xlsx)
    data_root = _normalize_host_path(data_root) or data_root
    reports_dir = _normalize_host_path(reports_dir or os.path.join(data_root, 'reports'))
    # exact Excel override path (string path) wins if exists
    if excel_path_override:
        excel_path_override = _normalize_host_path(excel_path_override)
    path_xlsx = excel_path_override or os.path.join(reports_dir or '', f"{t}_technicals.xlsx")
    # Normalize common user input issues (extra quotes/spaces, mixed separators)
    path_xlsx = _normalize_host_path(path_xlsx) or path_xlsx
    reports_dir = _normalize_host_path(reports_dir) or reports_dir

    # If not exactly in reports_dir, try to discover anywhere under data_root
    if not os.path.exists(path_xlsx):
        try:
            import glob as _glob
            pattern = os.path.join(_normalize_host_path(data_root) or data_root, '**', f'{t}_technicals.xlsx')
            candidates = sorted(_glob.glob(pattern, recursive=True))
            if candidates:
                path_xlsx = candidates[0]
        except Exception:
            pass

    if not os.path.exists(path_xlsx) and auto_generate_report and technicals_script:
            # Try to generate the report via external script
            try:
                import sys, subprocess
                # Attempt common CLIs
                tried_cmds = [
                    [sys.executable, technicals_script, t, reports_dir],
                    [sys.executable, technicals_script, '--ticker', t, '--out', reports_dir],
                    [sys.executable, technicals_script, '--ticker', t],
                ]
                for cmd in tried_cmds:
                    try:
                        subprocess.run(cmd, check=True, timeout=180, capture_output=True)
                        break
                    except Exception:
                        continue
            except Exception:
                pass  # Non-fatal; will try reading if the script happened to succeed

    if os.path.exists(path_xlsx):
            try:
                try:
                    import openpyxl  # noqa: F401
                    xl = pd.ExcelFile(path_xlsx, engine='openpyxl')
                except ImportError as _e:
                    raise RuntimeError("Excel engine 'openpyxl' is required to read .xlsx reports. Install with: pip install openpyxl")
                def _try_sheet_to_ohlc(_df: pd.DataFrame) -> pd.DataFrame | None:
                    d = _df.copy()
                    # If index looks like dates and not a default RangeIndex, lift to a column
                    try:
                        if not isinstance(d.index, pd.RangeIndex):
                            idx_as_date = pd.to_datetime(d.index, errors='coerce')
                            if idx_as_date.notna().mean() > 0.8 and 'Date' not in d.columns:
                                d.insert(0, 'Date', idx_as_date)
                                d.reset_index(drop=True, inplace=True)
                    except Exception:
                        pass
                    # Normalize headers
                    rename_map = {}
                    for c in list(d.columns):
                        lc = str(c).strip().lower().replace(' ', '').replace('_','')
                        if lc in {'date','day','sessiondate','windowstart','t','timestamp'}:
                            rename_map[c] = 'Date'
                        elif lc in {'open','o'}:
                            rename_map[c] = 'Open'
                        elif lc in {'high','h'}:
                            rename_map[c] = 'High'
                        elif lc in {'low','l'}:
                            rename_map[c] = 'Low'
                        elif lc in {'close','c'}:
                            rename_map[c] = 'Close'
                        elif lc in {'adjclose','adjustedclose','adj_close'}:
                            rename_map[c] = 'Adj Close'
                        elif lc in {'volume','v','vol'}:
                            rename_map[c] = 'Volume'
                    if rename_map:
                        d = d.rename(columns=rename_map)
                    # If Date still missing, try first column
                    if 'Date' not in d.columns and len(d.columns) > 0:
                        try:
                            cand = pd.to_datetime(d.iloc[:,0], errors='coerce')
                            if cand.notna().mean() > 0.8:
                                d.insert(0, 'Date', cand)
                        except Exception:
                            pass
                    # If Close missing but Adj Close present
                    if 'Close' not in d.columns and 'Adj Close' in d.columns:
                        d['Close'] = pd.to_numeric(d['Adj Close'], errors='coerce')
                    # Ensure numeric types where available
                    for col in ['Open','High','Low','Close','Volume']:
                        if col in d.columns:
                            d[col] = pd.to_numeric(d[col], errors='coerce')
                    # Final sanity: need Date + Close; for gap studies we also need Open
                    if 'Date' in d.columns and 'Close' in d.columns:
                        d['Date'] = pd.to_datetime(d['Date'], errors='coerce', utc=True)
                        d = d.dropna(subset=['Date','Close'])
                        d['Date'] = d['Date'].dt.tz_convert(None) if hasattr(d['Date'].dt, 'tz_convert') else d['Date']
                        d = d.sort_values('Date').reset_index(drop=True)
                        return d
                    return None

                best = None
                for sheet in xl.sheet_names:
                    try:
                        cand = _try_sheet_to_ohlc(xl.parse(sheet))
                        if cand is not None and len(cand) >= 10:
                            best = cand
                            # Prefer sheet names that sound like daily or price
                            if str(sheet).lower() in {'daily','prices','ohlc','price','history'}:
                                break
                    except Exception:
                        continue
                if best is not None:
                    df = best
            except Exception as e:
                # Surface Excel parsing issues without exposing paths
                raise RuntimeError(f"Failed reading Excel report: {e}")

    if df is None:
        # 2) Final fallback: build from daily_aggs_v1 flat files (CSV/CSV.GZ). This avoids Excel dependency.
        daily_root = _normalize_host_path(os.path.join(data_root, 'daily_aggs_v1')) or os.path.join(data_root, 'daily_aggs_v1')
        if not os.path.exists(daily_root):
            daily_root = None
        try:
            import glob
            parts: list[pd.DataFrame] = []
            files: list[str] = []
            if daily_root:
                pattern1 = os.path.join(daily_root, '**', '*.csv')
                pattern2 = os.path.join(daily_root, '**', '*.csv.gz')
                files = sorted(glob.glob(pattern1, recursive=True)) + sorted(glob.glob(pattern2, recursive=True))
            usecols = None
            for fp in files:
                try:
                    try:
                        hdr = pd.read_csv(fp, nrows=0, encoding="utf-8")
                    except UnicodeDecodeError:
                        hdr = pd.read_csv(fp, nrows=0, encoding="latin1")
                    lower = {str(c).lower(): c for c in hdr.columns}
                    tcol = next((lower[k] for k in ('ticker','symbol','t') if k in lower), None)
                    if not tcol:
                        continue
                    # Detect column names present in this file
                    ocol = next((lower[k] for k in ('open','o') if k in lower), None)
                    hcol = next((lower[k] for k in ('high','h') if k in lower), None)
                    lcol = next((lower[k] for k in ('low','l') if k in lower), None)
                    ccol = next((lower[k] for k in ('close','c') if k in lower), None)
                    vcol = next((lower[k] for k in ('volume','v') if k in lower), None)
                    dcol = next((lower[k] for k in ('date','day','window_start','timestamp','t') if k in lower), None)
                    cols = [c for c in (tcol, dcol, ocol, hcol, lcol, ccol, vcol) if c]
                    try:
                        dfp = pd.read_csv(fp, usecols=cols, encoding="utf-8")
                    except UnicodeDecodeError:
                        dfp = pd.read_csv(fp, usecols=cols, encoding="latin1")
                    dfp = dfp[dfp[tcol].astype(str).str.upper() == t]
                    if dfp.empty:
                        continue
                    # Normalize
                    mapping = {'Ticker': tcol, 'Date': dcol, 'Open': ocol, 'High': hcol, 'Low': lcol, 'Close': ccol, 'Volume': vcol}
                    mapping = {k: v for k, v in mapping.items() if v}
                    dfp = dfp.rename(columns={v: k for k, v in mapping.items()})
                    # Parse date robustly
                    rawd = dfp['Date']
                    if pd.api.types.is_numeric_dtype(rawd):
                        mx = pd.to_numeric(rawd, errors='coerce').dropna().astype(float).max() if len(rawd) else 0
                        if mx > 1e12:
                            parsed = pd.to_datetime(rawd, unit='ns', errors='coerce', utc=True)
                        elif mx > 1e9:
                            parsed = pd.to_datetime(rawd, unit='s', errors='coerce', utc=True)
                        else:
                            parsed = pd.to_datetime(rawd, errors='coerce', utc=True)
                    else:
                        parsed = pd.to_datetime(rawd, errors='coerce', utc=True)
                    dfp['Date'] = parsed.dt.tz_convert(None) if hasattr(parsed.dt, 'tz_convert') else parsed
                    for col in ['Open','High','Low','Close','Volume']:
                        if col in dfp.columns:
                            dfp[col] = pd.to_numeric(dfp[col], errors='coerce')
                    dfp = dfp.dropna(subset=['Date','Close'])
                    parts.append(dfp[['Date','Open','High','Low','Close','Volume']])
                except Exception:
                    continue
            if parts:
                df = pd.concat(parts, ignore_index=True)
        except Exception as e:
            df = None

    # 3) Final safety: Yahoo Finance fallback (optional)
    if df is None and allow_yahoo_fallback:
        try:
            import yfinance as _yf
            ydf = _yf.download(
                tickers=str(t),
                period="max",
                interval="1d",
                auto_adjust=False,
                progress=False,
                group_by=None,
            )
            if ydf is not None and not ydf.empty:
                if isinstance(ydf.index, pd.DatetimeIndex):
                    out = pd.DataFrame({
                        'Date': pd.to_datetime(ydf.index, utc=True),
                        'Open': pd.to_numeric(ydf.get('Open'), errors='coerce'),
                        'High': pd.to_numeric(ydf.get('High'), errors='coerce'),
                        'Low': pd.to_numeric(ydf.get('Low'), errors='coerce'),
                        'Close': pd.to_numeric(ydf.get('Close') if 'Close' in ydf.columns else ydf.get('Adj Close'), errors='coerce'),
                        'Volume': pd.to_numeric(ydf.get('Volume'), errors='coerce'),
                    })
                    out = out.dropna(subset=['Date','Close'])
                    out['Date'] = out['Date'].dt.tz_convert(None)
                    out = out.sort_values('Date').reset_index(drop=True)
                    df = out
        except Exception:
            df = None

    # If still nothing, raise a generic error (no paths)
    if df is None:
        raise ValueError("No historical data found via Excel, flat files, or Yahoo fallback.")

    # Normalize columns case-insensitively
    cols = {str(c).lower(): c for c in df.columns}
    def pick(*names):
        for n in names:
            key = n.lower()
            if key in cols:
                return cols[key]
        return None

    c_open = pick('open')
    c_high = pick('high')
    c_low = pick('low')
    c_close = pick('close')
    c_volume = pick('volume','v')
    c_date = pick('date','day','session_date','window_start','t','timestamp')

    if c_date is None or c_open is None or c_close is None:
        raise ValueError(f"Unexpected schema for {t}. Columns: {list(df.columns)}")

    # Parse date robustly (supports epoch ns/s or ISO strings)
    _raw_date = df[c_date]
    try:
        import numpy as _np
        import pandas as _pd
        if _pd.api.types.is_numeric_dtype(_raw_date):
            mx = _pd.to_numeric(_raw_date, errors='coerce').dropna().astype(float).max() if len(_raw_date) else 0
            if mx > 1e12:
                parsed_date = _pd.to_datetime(_raw_date, unit='ns', errors='coerce', utc=True)
            elif mx > 1e9:
                parsed_date = _pd.to_datetime(_raw_date, unit='s', errors='coerce', utc=True)
            else:
                parsed_date = _pd.to_datetime(_raw_date, errors='coerce', utc=True)
        else:
            parsed_date = _pd.to_datetime(_raw_date, errors='coerce', utc=True)
    except Exception:
        parsed_date = pd.to_datetime(_raw_date, errors='coerce', utc=True)

    out = pd.DataFrame({
        'Date': parsed_date,
        'Open': pd.to_numeric(df[c_open], errors='coerce'),
        'High': pd.to_numeric(df[c_high], errors='coerce') if c_high in df.columns else pd.NA,
        'Low': pd.to_numeric(df[c_low], errors='coerce') if c_low in df.columns else pd.NA,
        'Close': pd.to_numeric(df[c_close], errors='coerce'),
        'Volume': pd.to_numeric(df[c_volume], errors='coerce') if c_volume in df.columns else pd.NA,
    })
    out = out.dropna(subset=['Date'])
    # Convert to naive date (no timezone) for grouping/joins; keep time for safety
    out['Date'] = out['Date'].dt.tz_convert(None) if hasattr(out['Date'].dt, 'tz_convert') else out['Date']
    out = out.sort_values('Date').reset_index(drop=True)
    return out

def _compute_gap_drop_stats(daily: pd.DataFrame, mode: str, threshold_pct: float, direction: str | None = None) -> pd.DataFrame:
    """
    Compute event table for either:
      - mode='close_drop': prior close -> today close <= -threshold
      - mode='gap': gap % at open >= threshold (direction 'Up'/'Down')
    Returns a DataFrame with Date, Gap_% , Intraday_% , Next_Overnight_% , Next_Intraday_% , Next_Total_% .
    """
    import numpy as np
    df = daily.copy()
    df['PrevClose'] = df['Close'].shift(1)
    df['NextOpen'] = df['Open'].shift(-1)
    df['NextClose'] = df['Close'].shift(-1)
    df['Gap_%'] = (df['Open'] - df['PrevClose']) / df['PrevClose'] * 100.0
    df['Intraday_%'] = (df['Close'] - df['Open']) / df['Open'] * 100.0
    # Event-day close-to-close move
    df['Close_to_Close_%'] = (df['Close'] - df['PrevClose']) / df['PrevClose'] * 100.0
    df['Next_Overnight_%'] = (df['NextOpen'] - df['Close']) / df['Close'] * 100.0
    df['Next_Intraday_%'] = (df['NextClose'] - df['NextOpen']) / df['NextOpen'] * 100.0
    df['Next_Total_%'] = (df['NextClose'] - df['Close']) / df['Close'] * 100.0
    df['Next_Close_to_Close_%'] = df['Next_Total_%']

    if mode == 'close_drop':
        drop_pct = (df['Close'] - df['PrevClose']) / df['PrevClose'] * 100.0
        # Signed threshold: positive means up >= threshold; negative means down <= threshold
        if threshold_pct >= 0:
            mask = drop_pct >= threshold_pct
        else:
            mask = drop_pct <= threshold_pct
    elif mode == 'gap':
        # Signed threshold: positive means gap up >= threshold; negative means gap down <= threshold
        if threshold_pct >= 0:
            mask = df['Gap_%'] >= threshold_pct
        else:
            mask = df['Gap_%'] <= threshold_pct
    else:
        raise ValueError("mode must be 'close_drop' or 'gap'")

    if mode == 'close_drop':
        cols = ['Date','Close_to_Close_%','Next_Overnight_%','Next_Intraday_%','Next_Close_to_Close_%']
    else:
        cols = ['Date','Gap_%','Intraday_%','Next_Overnight_%','Next_Intraday_%','Next_Total_%']
    out = df.loc[mask, cols].dropna().reset_index(drop=True)
    return out

# ---------------- CHART TAB ----------------
if nav == 'Chart':
    _show_scan_banner()

    # Optional auto-refresh during US market hours for 1d aggregation from intraday
    try:
        if (interval == '1d') and bool(append_today_daily) and bool(auto_refresh_daily):
            import pytz as _pytz
            _ny = _pytz.timezone('America/New_York')
            now_ny = pd.Timestamp.now(tz=_ny)
            is_weekday = now_ny.weekday() < 5
            # refresh between 09:25 and 16:10 ET
            hm = now_ny.hour + now_ny.minute/60.0
            within = (9 + 25/60.0) <= hm <= (16 + 10/60.0)
            if is_weekday and within:
                try:
                    st.session_state['force_refresh'] = True
                except Exception:
                    pass
                import streamlit as _st
                _st.components.v1.html('<script>setTimeout(function(){window.location.reload();}, 60000);</script>', height=0)
    except Exception:
        pass
    # --- Simple Backtest Logic ---
    def backtest_price_crosses_vwap(price: pd.Series, vwap: pd.Series):
            signals = (price > vwap) & (price.shift(1) <= vwap.shift(1))
            trades = []
            in_trade = False
            entry_idx = None
            for i in range(1, len(price)):
                if signals.iloc[i] and not in_trade:
                    entry_idx = i
                    in_trade = True
                elif in_trade and (price.iloc[i] < vwap.iloc[i]):
                    trades.append({
                        'entry_time': price.index[entry_idx],
                        'entry_price': price.iloc[entry_idx],
                        'exit_time': price.index[i],
                        'exit_price': price.iloc[i],
                        'pnl': price.iloc[i] - price.iloc[entry_idx]
                    })
                    in_trade = False
                    entry_idx = None
            return trades

    try:
            # Map allowed intraday periods for Yahoo
            def best_period_for(interval_str: str, desired: str | None) -> str:
                if interval_str == "1m":
                    return "7d"  # Yahoo max for 1m
                if interval_str in {"5m", "15m", "30m", "60m", "1h"}:
                    return desired if desired in {"5d", "7d", "14d", "30d", "60d"} else "30d"
                return desired or "1y"

            # Normalize symbol for futures continuous aliases (ES -> ES=F, etc.)
            # and build candidate list for specific contracts (e.g., ESZ24)
            contract_candidates = build_futures_contract_candidates(ticker)
            if contract_candidates:
                fetch_tickers = contract_candidates
            else:
                fetch_tickers = [normalize_input_symbol(ticker)]

            if intraday:
                p = best_period_for(interval, str(period))
                df = None
                for tk in fetch_tickers:
                    df = fetch_ohlc_with_fallback(tk, interval=interval, period=p)
                    if df is not None and not df.empty:
                        break
            else:
                # Ensure valid date order and include selected end date (Yahoo end is exclusive)
                s = min(pd.to_datetime(start), pd.to_datetime(end)).date()
                e = max(pd.to_datetime(start), pd.to_datetime(end)).date()
                e_inclusive = (e + timedelta(days=1)).isoformat()
                df = None
                for tk in fetch_tickers:
                    df = fetch_ohlc_with_fallback(tk, interval=interval, start=s.isoformat(), end=e_inclusive)
                    if df is not None and not df.empty:
                        break

                # Daily-specific resilience: widen window if empty, try Ticker().history, then period fallback
                if (df is None or df.empty) and interval == "1d":
                    s_wide = (s - timedelta(days=3)).isoformat()
                    e_wide_inclusive = (e + timedelta(days=3 + 1)).isoformat()
                    if df is None or df.empty:
                        for tk in fetch_tickers:
                            df = fetch_ohlc_with_fallback(tk, interval=interval, start=s_wide, end=e_wide_inclusive)
                            if df is not None and not df.empty:
                                break
                    # Try direct Ticker().history as another path (sometimes succeeds when download() fails)
                    if (df is None or df.empty):
                        try:
                            tkr = yf.Ticker(fetch_tickers[0])
                            df_hist = tkr.history(start=s.isoformat(), end=e_inclusive, interval="1d", auto_adjust=False)
                            if isinstance(df_hist, pd.DataFrame) and not df_hist.empty:
                                df = df_hist
                        except Exception as e:
                            try:
                                st.session_state['last_fetch_errors'].append(f"yfinance Ticker.history: {e}")
                            except Exception:
                                pass
                    if df is None or df.empty:
                        span_days = (e - s).days + 1
                        period_map = [(7, "7d"), (31, "1mo"), (93, "3mo"), (183, "6mo"), (365, "1y"), (730, "2y")]
                        sel_period = "1y"
                        for lim, per in period_map:
                            if span_days <= lim:
                                sel_period = per
                                break
                        for tk in fetch_tickers:
                            df = fetch_ohlc_with_fallback(tk, interval=interval, period=sel_period)
                            if df is not None and not df.empty:
                                break
                    # Final safety: fetch full history and slice
                    if df is None or df.empty:
                        try:
                            full = None
                            for tk in fetch_tickers:
                                full = fetch_ohlc_with_fallback(tk, interval="1d", period="max")
                                if full is not None and not full.empty:
                                    break
                            if isinstance(full, pd.DataFrame) and not full.empty and isinstance(full.index, pd.DatetimeIndex):
                                df = full[(full.index.date >= s) & (full.index.date <= e)]
                        except Exception as e:
                            try:
                                st.session_state['last_fetch_errors'].append(f"max-period slice fallback: {e}")
                            except Exception:
                                pass

            if df is None or df.empty:
                details = None
                try:
                    errs = st.session_state.get('last_fetch_errors')
                    if errs:
                        details = " | Details: " + " | ".join(errs[-3:])
                except Exception:
                    pass
                msg = f"No data for {ticker} @ interval={interval}. Tried polygon, yfinance, and yahooquery."
                if details:
                    msg += details
                st.warning(msg)
                st.stop()

            df = normalize_ohlcv(df)
            for col in ["Open", "High", "Low", "Close", "Adj Close", "Volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")

            # Quick data freshness panel
            try:
                last_dt = None
                if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 0:
                    last_dt = df.index.max()
                info_col1, info_col2 = st.columns([3,1])
                with info_col1:
                    if last_dt is not None:
                        prov = st.session_state.get('last_fetch_provider', '-')
                        st.caption(f"Data ends: {pd.to_datetime(last_dt).strftime('%Y-%m-%d %H:%M')} • Source: {prov}")
                with info_col2:
                    if st.button('Refresh Data'):
                        try:
                            st.session_state['force_refresh'] = True
                        except Exception:
                            pass
                        st.rerun()
            except Exception:
                pass

            # Optionally append today's partial daily bar using local minute data first, then API
            try:
                if (interval == "1d") and append_today_daily:
                    # Determine 'today' in NY and the target symbol
                    try:
                        import pytz as _pytz
                        _ny_tz = _pytz.timezone('America/New_York')
                    except Exception:
                        _ny_tz = None
                    today = pd.Timestamp.now(tz=_ny_tz).date() if _ny_tz else pd.Timestamp.utcnow().tz_localize(None).date()
                    sym0 = fetch_tickers[0] if isinstance(fetch_tickers, (list, tuple)) and fetch_tickers else tkr
                    tz = 'America/New_York'
                    # Use a wide window to include pre/post market
                    start_ts = pd.Timestamp(year=today.year, month=today.month, day=today.day, hour=4, minute=0, tz=tz)
                    end_ts   = pd.Timestamp(year=today.year, month=today.month, day=today.day, hour=20, minute=10, tz=tz)

                    intr_today = _load_minute_local(sym0, start_ts, end_ts)
                    provider_tag = 'local-minute'

                    if intr_today is None or intr_today.empty:
                        # Fallback: API 5m last 5d, then filter for today
                        provider_tag = 'api-5m'
                        intr = fetch_ohlc_with_fallback(sym0, interval="5m", period="5d")
                        if isinstance(intr, pd.DataFrame) and not intr.empty and isinstance(intr.index, pd.DatetimeIndex):
                            intr = normalize_ohlcv(intr)
                            try:
                                idx_ny = intr.index.tz_convert('America/New_York') if intr.index.tz is not None else intr.index.tz_localize('UTC').tz_convert('America/New_York')
                            except Exception:
                                idx_ny = intr.index
                            intr_today = intr[(pd.Index(idx_ny).date) == today]

                    if (intr_today is None) or intr_today.empty:
                        # Force Polygon and try 1m then 5m for today's explicit window
                        try:
                            prev_pref = st.session_state.get('preferred_provider') if 'preferred_provider' in st.session_state else None
                            st.session_state['preferred_provider'] = 'Polygon'
                        except Exception:
                            prev_pref = None
                        try:
                            provider_tag = 'api-1m/5m'
                            intr_retry = fetch_ohlc_with_fallback(sym0, interval='1m', start=start_ts.isoformat(), end=end_ts.isoformat())
                            if (intr_retry is None) or intr_retry.empty:
                                intr_retry = fetch_ohlc_with_fallback(sym0, interval='5m', start=start_ts.isoformat(), end=end_ts.isoformat())
                            if isinstance(intr_retry, pd.DataFrame) and not intr_retry.empty and isinstance(intr_retry.index, pd.DatetimeIndex):
                                intr_retry = normalize_ohlcv(intr_retry)
                                try:
                                    idx_ny2 = intr_retry.index.tz_convert('America/New_York') if intr_retry.index.tz is not None else intr_retry.index.tz_localize('UTC').tz_convert('America/New_York')
                                except Exception:
                                    idx_ny2 = intr_retry.index
                                intr_today = intr_retry[(pd.Index(idx_ny2).date) == today]
                        except Exception:
                            pass
                        finally:
                            try:
                                if prev_pref is None:
                                    if 'preferred_provider' in st.session_state:
                                        del st.session_state['preferred_provider']
                                else:
                                    st.session_state['preferred_provider'] = prev_pref
                            except Exception:
                                pass

                    # Record debug info for UI
                    try:
                        st.session_state['today_debug'] = {
                            'provider': provider_tag,
                            'intraday_rows': int(0 if intr_today is None else len(intr_today)),
                            'intraday_start': '-' if (intr_today is None or intr_today.empty) else str(intr_today.index.min()),
                            'intraday_end': '-' if (intr_today is None or intr_today.empty) else str(intr_today.index.max()),
                            'target_date': str(today),
                        }
                    except Exception:
                        pass

                    if isinstance(intr_today, pd.DataFrame) and not intr_today.empty:
                        o = float(intr_today["Open"].iloc[0]) if "Open" in intr_today.columns else float('nan')
                        h = float(intr_today["High"].max()) if "High" in intr_today.columns else float('nan')
                        l = float(intr_today["Low"].min()) if "Low" in intr_today.columns else float('nan')
                        c = float(intr_today["Close"].iloc[-1]) if "Close" in intr_today.columns else float('nan')
                        v = float(intr_today["Volume"].sum()) if "Volume" in intr_today.columns else float('nan')
                        idx = pd.to_datetime(pd.Timestamp(today))
                        new_row = pd.DataFrame({"Open":[o],"High":[h],"Low":[l],"Close":[c],"Volume":[v]}, index=[idx])
                        if isinstance(df.index, pd.DatetimeIndex):
                            if (df.index.date == today).any():
                                df = pd.concat([df[~(df.index.date == today)], new_row])
                            else:
                                df = pd.concat([df, new_row])
                            df = df.sort_index()
                            try:
                                df = df[~df.index.duplicated(keep='last')]
                            except Exception:
                                pass

            except Exception:
                pass

            # Determine VWAP anchor index from input (overlay not required for backtest)
            vwap_idx = None
            if vwap_anchor:
                try:
                    anchor_dt = pd.to_datetime(vwap_anchor)
                    if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 0:
                        pos = df.index.get_indexer([anchor_dt], method='nearest')
                        vwap_idx = int(pos[0]) if pos.size and pos[0] != -1 else None
                except Exception:
                    vwap_idx = None

            # Option: if daily interval and user wants only the latest day, slice to last date
            try:
                if (interval == "1d") and bool(show_only_latest_day) and isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 0:
                    _last_date = df.index.max().date()
                    df = df[(df.index.date == _last_date)]
            except Exception:
                pass

            # Count rows for subplots (optional Volume, Transactions, Turnover, RSI, Stoch, MACD panel)
            base_rows = 1 + (1 if show_volume else 0) + (1 if show_tx else 0) + (1 if show_turnover else 0)
            rows = base_rows + (1 if show_rsi else 0) + (1 if show_sto else 0) + (1 if show_macd else 0)
            row_heights = [0.55]
            if show_volume:
                row_heights.append(0.13)
            if show_tx:
                row_heights.append(0.13)
            if show_turnover:
                row_heights.append(0.13)
            if show_rsi:
                row_heights.append(0.09)
            if show_sto:
                row_heights.append(0.09)
            if show_macd:
                row_heights.append(0.09)
            fig = make_subplots(rows=rows, cols=1, shared_xaxes=True, vertical_spacing=0.03, row_heights=row_heights)

            # Plotting block (encapsulate to align indentation)
            if True:
                if True:

                    # Main candlestick (cleaner aesthetics with optional thin/no-fill)
                    if thin_candles:
                        inc_fill = "rgba(0,0,0,0)"
                        dec_fill = "rgba(0,0,0,0)"
                        line_w = 1.2
                    else:
                        inc_fill = "#2ecc40"
                        dec_fill = "#ff4136"
                        line_w = 1.4
                    fig.add_trace(go.Candlestick(
                        x=df.index,
                        open=df.get("Open"), high=df.get("High"), low=df.get("Low"), close=df.get("Close"),
                        name="OHLC",
                        increasing_line_color="#2ecc40", decreasing_line_color="#ff4136",
                        increasing_fillcolor=inc_fill, decreasing_fillcolor=dec_fill,
                        increasing=dict(line=dict(width=line_w)),
                        decreasing=dict(line=dict(width=line_w)),
                        opacity=0.95 if not thin_candles else 1.0
                    ), row=1, col=1)

                    close_key = pick_close_key(df.columns)
                    if close_key is None:
                        st.error(f"No usable 'Close' found after normalization. Columns: {list(df.columns)}")
                        st.stop()
                    close = df[close_key].astype(float)

                    # Add price labels to right side if enabled
                    def add_price_label(trace_name, y_val, color):
                        # Place label just inside the plotting area so it's not clipped
                        fig.add_annotation(
                            xref="paper", x=0.995, y=y_val,
                            xanchor="right", yanchor="middle",
                            text=f"{trace_name}: {y_val:.2f}",
                            font=dict(color=color, size=13),
                            showarrow=False, align="right",
                            bgcolor="#222" if template=="plotly_dark" else "#fff",
                            bordercolor=color, borderwidth=1, opacity=0.95
                        )

                    if show_price_labels:
                        # Candlestick close
                        if len(close) > 0:
                            add_price_label("Close", close.iloc[-1], "#00bfff")
                        # Anchored VWAP
                        if show_vwap and vwap_idx is not None:
                            vwap_series = anchored_vwap(df, anchor_idx=vwap_idx)
                            last_vwap = vwap_series.dropna().iloc[-1] if vwap_series.dropna().size > 0 else None
                            if last_vwap is not None:
                                add_price_label("VWAP", last_vwap, "#ff9900")
                        # SMA/EMA
                        for n in sma_selected:
                            s = sma(close, int(n))
                            if len(s.dropna()) > 0:
                                add_price_label(f"SMA({n})", s.dropna().iloc[-1], "#8888ff")
                        for n in ema_selected:
                            e = ema(close, int(n))
                            if len(e.dropna()) > 0:
                                add_price_label(f"EMA({n})", e.dropna().iloc[-1], "#ff88ff")

                    # Plot anchored VWAP if enabled
                    if show_vwap and vwap_idx is not None:
                        vwap_series = anchored_vwap(df, anchor_idx=vwap_idx)
                        fig.add_trace(go.Scatter(x=df.index, y=vwap_series, mode="lines", name="Anchored VWAP", line=dict(color="#ff9900", width=2, dash="dash")), row=1, col=1)

                    if bb_on:
                        lb, mb, ub = bbands(close, int(bb_len), float(bb_std))
                        fig.add_trace(go.Scatter(x=df.index, y=lb, mode="lines", name=lb.name), row=1, col=1)
                        fig.add_trace(go.Scatter(x=df.index, y=mb, mode="lines", name=mb.name), row=1, col=1)
                        fig.add_trace(go.Scatter(x=df.index, y=ub, mode="lines", name=ub.name), row=1, col=1)
                
                        # Support/Resistance
                        if show_sr:
                            supports, resistances = find_support_resistance(close, int(sr_lookback))
                            for t, y in supports:
                                fig.add_hline(y=y, line_dash="dot", line_color="#2ecc40", annotation_text="Support", annotation_position="left", row=1, col=1)
                            for t, y in resistances:
                                fig.add_hline(y=y, line_dash="dot", line_color="#ff4136", annotation_text="Resistance", annotation_position="left", row=1, col=1)
                
                        # Candlestick Patterns
                        if show_patterns:
                            patterns = detect_patterns(df)
                            for pat, ser in patterns.items():
                                # Markers for detected patterns
                                pat_idx = ser[ser != 0].index
                                fig.add_trace(go.Scatter(x=pat_idx, y=close.loc[pat_idx], mode="markers", marker_symbol="star", marker_size=12, name=pat), row=1, col=1)

                    for n in sma_selected:
                        s = sma(close, int(n))
                        fig.add_trace(go.Scatter(x=df.index, y=s, mode="lines", name=s.name), row=1, col=1)
                    for n in ema_selected:
                        e = ema(close, int(n))
                        fig.add_trace(go.Scatter(x=df.index, y=e, mode="lines", name=e.name), row=1, col=1)

                    # Optional Volume and Transactions panels
                    next_row_idx = 2
                    if show_volume:
                        if "Volume" in df.columns and df["Volume"].notna().any():
                            vol_colors = np.where(close >= df.get("Open", close), "rgba(0,200,0,0.6)", "rgba(200,0,0,0.6)")
                            fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color=vol_colors, showlegend=False), row=next_row_idx, col=1)
                        else:
                            fig.add_trace(go.Bar(x=df.index, y=[0]*len(df), name="Volume", showlegend=False), row=next_row_idx, col=1)
                        next_row_idx += 1

                    if show_tx:
                        if "Transactions" in df.columns and df["Transactions"].notna().any():
                            fig.add_trace(go.Bar(x=df.index, y=df["Transactions"], name="Transactions", marker_color="rgba(100,149,237,0.7)", showlegend=False), row=next_row_idx, col=1)
                        else:
                            fig.add_trace(go.Bar(x=df.index, y=[0]*len(df), name="Transactions", showlegend=False), row=next_row_idx, col=1)
                        next_row_idx += 1

                    # Optional Dollar Value Traded panel (Close * Volume)
                    if show_turnover:
                        turn_row = 2 + (1 if show_volume else 0) + (1 if show_tx else 0)
                        if ("Close" in df.columns) and ("Volume" in df.columns) and df[["Close","Volume"]].notna().all(axis=1).any():
                            dollar_val = (df["Close"].astype(float) * df["Volume"].astype(float))
                            fig.add_trace(
                                go.Bar(x=df.index, y=dollar_val, name="$ Value Traded", marker_color="rgba(0,123,255,0.6)", showlegend=False),
                                row=turn_row, col=1
                            )
                            # Optional 30-day moving average overlay
                            try:
                                if show_turnover_ma and int(turnover_ma_len) > 1:
                                    dv_ma = dollar_val.rolling(int(turnover_ma_len)).mean()
                                    fig.add_trace(
                                        go.Scatter(x=df.index, y=dv_ma, mode="lines", name=f"$ Value {int(turnover_ma_len)}d MA", line=dict(color="#ffae42", width=2)),
                                        row=turn_row, col=1
                                    )
                            except Exception:
                                pass
                        else:
                            fig.add_trace(go.Bar(x=df.index, y=[0]*len(df), name="$ Value Traded", showlegend=False), row=turn_row, col=1)

                    next_row = 2 + (1 if show_volume else 0) + (1 if show_turnover else 0)
                    if show_rsi:
                        r = rsi(close, int(rsi_len))
                        fig.add_trace(go.Scatter(x=df.index, y=r, mode="lines", name=r.name), row=next_row, col=1)
                        fig.add_hline(y=30, line_dash="dot", line_color="#555", row=next_row, col=1)
                        fig.add_hline(y=70, line_dash="dot", line_color="#555", row=next_row, col=1)
                        fig.update_yaxes(range=[0, 100], row=next_row, col=1, title_text="RSI")
                        next_row += 1

                    if show_sto:
                        k, d = stoch_kd(df["High"], df["Low"], close, int(sto_k), int(sto_d), int(sto_smooth))
                        fig.add_trace(go.Scatter(x=df.index, y=k, mode="lines", name=k.name), row=next_row, col=1)
                        fig.add_trace(go.Scatter(x=df.index, y=d, mode="lines", name=d.name), row=next_row, col=1)
                        fig.add_hline(y=20, line_dash="dot", line_color="#555", row=next_row, col=1)
                        fig.add_hline(y=80, line_dash="dot", line_color="#555", row=next_row, col=1)
                        fig.update_yaxes(range=[0, 100], row=next_row, col=1, title_text="Stoch")
                        next_row += 1

                    if show_macd:
                        macd_line, signal_line, hist = macd(close, int(macd_fast), int(macd_slow), int(macd_signal))
                        fig.add_trace(go.Scatter(x=df.index, y=macd_line, mode="lines", name="MACD"), row=next_row, col=1)
                        fig.add_trace(go.Scatter(x=df.index, y=signal_line, mode="lines", name="Signal"), row=next_row, col=1)
                        fig.add_trace(go.Bar(x=df.index, y=hist, name="Hist", marker_color=np.where(hist>=0, "#2ecc40", "#ff4136")), row=next_row, col=1)
                        fig.update_yaxes(title_text="MACD", row=next_row, col=1)
                        next_row += 1

                    last_ts = pd.to_datetime(df.index[-1])
                    extend_right_edge(fig, last_ts, interval, rows)

                    # --- Run backtest if enabled and strategy is selected ---
                    trades = []
                    backtest_missing_anchor = False
                    if enable_backtest:
                        def backtest_hold_condition(price_series: pd.Series, cond: pd.Series):
                            cond = cond.fillna(False)
                            entries = (cond & ~cond.shift(1).fillna(False))
                            exits = (~cond & cond.shift(1).fillna(False))
                            in_pos = False
                            entry_idx = None
                            out = []
                            for i, ts in enumerate(price_series.index):
                                if entries.iloc[i] and not in_pos:
                                    in_pos = True
                                    entry_idx = i
                                elif exits.iloc[i] and in_pos:
                                    out.append({
                                        'entry_time': price_series.index[entry_idx],
                                        'entry_price': float(price_series.iloc[entry_idx]),
                                        'exit_time': ts,
                                        'exit_price': float(price_series.iloc[i]),
                                        'pnl': float(price_series.iloc[i] - price_series.iloc[entry_idx])
                                    })
                                    in_pos = False
                                    entry_idx = None
                            return out

                        if strategy == "Price crosses above VWAP":
                            if vwap_idx is None:
                                backtest_missing_anchor = True
                            else:
                                vwap_series = anchored_vwap(df, anchor_idx=vwap_idx)
                                trades = backtest_price_crosses_vwap(close, vwap_series)
                        elif strategy == "RSI crosses above 70":
                            r = rsi(close, int(rsi_len))
                            hold = r > 70
                            trades = backtest_hold_condition(close, hold)
                        elif strategy == "RSI crosses below 30":
                            r = rsi(close, int(rsi_len))
                            hold = r < 30
                            trades = backtest_hold_condition(close, hold)
                        elif strategy == "Bollinger Bands":
                            lb, mb, ub = bbands(close, int(bb_len), float(bb_std))
                            hold = (close > ub) | (close < lb)
                            trades = backtest_hold_condition(close, hold)

                        # Plot buy/sell markers on chart
                        for t in trades:
                            fig.add_trace(go.Scatter(
                                x=[t['entry_time']], y=[t['entry_price']],
                                mode="markers", marker_symbol="triangle-up", marker_color="#00ff00", marker_size=12,
                                name="Buy"
                            ), row=1, col=1)
                            fig.add_trace(go.Scatter(
                                x=[t['exit_time']], y=[t['exit_price']],
                                mode="markers", marker_symbol="triangle-down", marker_color="#ff0000", marker_size=12,
                                name="Sell"
                            ), row=1, col=1)

                    # Tighter margins if price labels hidden; x unified hover for clarity
                    right_margin = 120 if show_price_labels else 60
                    fig.update_layout(
                        title=f"{ticker} - {interval}",
                        xaxis_rangeslider_visible=False,
                        height=base_height,
                        margin=dict(l=50, r=right_margin, t=50, b=50),
                        hovermode="x unified",
                        hoverlabel=dict(bgcolor="#111" if template=="plotly_dark" else "#f8f8f8", font_size=12)
                    )
                    # Add weekend breaks and optional intraday overnight removal
                    extra_rb = []
                    if intraday and hide_overnight:
                        extra_rb.append(dict(bounds=[16, 9.5], pattern="hour"))  # 4pm -> 9:30am ET
                    style_axes(fig, dark=(template == "plotly_dark"), rows=rows, extra_rangebreaks=extra_rb,
                               minimalist=minimalist_mode, nticks=(6 if minimalist_mode else None))

                    # Overlay scan highlights if present
                    try:
                        if bool(show_scan_marks):
                            hl = st.session_state.get('scan_highlights') or []
                            if hl:
                                for dstr in hl:
                                    try:
                                        dt0 = pd.to_datetime(dstr)
                                        fig.add_vrect(x0=dt0, x1=dt0 + pd.Timedelta(days=1), fillcolor='rgba(255,0,0,0.08)', line_width=0)
                                    except Exception:
                                        pass
                    except Exception:
                        pass

                    # Quick KPI for turnover if enabled
                    try:
                        if show_turnover and ("Close" in df.columns) and ("Volume" in df.columns) and len(df) > 0:
                            last_turn = float(df["Close"].iloc[-1]) * float(df["Volume"].iloc[-1])
                            ma_len = int(turnover_ma_len) if show_turnover_ma else 30
                            avgN = float((df["Close"].astype(float) * df["Volume"].astype(float)).rolling(ma_len).mean().iloc[-1]) if len(df) >= ma_len else None
                            c1, c2 = st.columns(2)
                            with c1:
                                st.metric("Last $ Traded", f"${last_turn:,.0f}")
                            with c2:
                                if avgN is not None and not np.isnan(avgN):
                                    st.metric(f"{ma_len}d Avg $ Traded", f"${avgN:,.0f}")
                    except Exception:
                        pass

                    # Today Debug expander (when daily aggregation from intraday is enabled)
                    try:
                        if False and (interval == '1d') and append_today_daily:
                            dbg = st.session_state.get('today_debug') or {}
                            with st.expander('Today Debug'):
                                st.caption(f"Provider: {dbg.get('provider','-')} | Rows today: {dbg.get('intraday_rows','-')} | Window: {dbg.get('intraday_start','-')} ? {dbg.get('intraday_end','-')}")
                    except Exception:
                        pass

                    # Additional safe Today Debug block
                    try:
                        if False and (interval == '1d') and append_today_daily:
                            dbg = st.session_state.get('today_debug') or {}
                            with st.expander('Today Debug (alt)'):
                                st.caption(
                                    f"Provider: {dbg.get('provider','-')} | Rows today: {dbg.get('intraday_rows','-')}"
                                )
                                st.caption(
                                    f"Window: {dbg.get('intraday_start','-')} -> {dbg.get('intraday_end','-')}"
                                )
                    except Exception:
                        pass

                    st.plotly_chart(fig, use_container_width=True)
                    provider = None
                    try:
                        provider = st.session_state.get('last_fetch_provider')
                    except Exception:
                        pass
                    src = f" | Source: {provider}" if provider else ""
                    st.caption(f"Rows: {len(df)} | Columns: {list(df.columns)}{src}")
                    try:
                        import pandas as _pd
                        if isinstance(df.index, _pd.DatetimeIndex) and len(df.index) > 0:
                            st.caption(f"Last bar: {str(df.index.max())}")
                    except Exception:
                        pass


                    # --- Historical Stats (from uploaded Excel only) ---
                    with st.expander("Historical Gap/Drop Stats"):
                        # Parquet-first: no Excel upload; rely on per-ticker parquet dir
                        stats_ticker = ticker
                        # Show which parquet path will be used (if found)
                        try:
                            pq_path_hint = _autofind_parquet_path(stats_ticker)
                            if pq_path_hint:
                                st.caption(f"Parquet source: {pq_path_hint}")
                            else:
                                st.caption("Parquet source: not found - set directory in sidebar.")
                                curdir = os.environ.get('PER_TICKER_PARQUET_DIR')
                                if curdir and os.path.isdir(curdir):
                                    try:
                                        import glob as _glob
                                        some = sorted(_glob.glob(os.path.join(curdir, '*.parquet')))[:5]
                                        if some:
                                            st.caption(f"Example files in directory: {[os.path.basename(x) for x in some]}")
                                    except Exception:
                                        pass
                        except Exception:
                            pass
                        mode = st.selectbox("Study", [
                            "Close down N% day -> next day",
                            "Gap up/down >= N% -> same day + next day",
                        ], index=0)
                        threshold = st.number_input(
                            "Threshold (%) — use + for up, - for down",
                            min_value=-50.0,
                            max_value=50.0,
                            value=-3.0,
                            step=0.1,
                        )

                        run = st.button("Run stats", key="run_gap_stats")
                        if run:
                            try:
                                # Load strictly from per-ticker Parquet (no Excel dependency)
                                daily_hist = _load_polygon_daily_for_ticker(
                                    data_root="",
                                    ticker=stats_ticker,
                                    reports_dir=None,
                                    technicals_script=None,
                                    auto_generate_report=False,
                                    excel_override=None,
                                    excel_path_override=None,
                                    allow_yahoo_fallback=False,
                                )
                                if daily_hist is None or daily_hist.empty:
                                    st.error("No historical data available from Parquet. Set 'Per-ticker Parquet directory' in the sidebar and ensure <TICKER>.parquet exists.")
                                    st.stop()
                                m = 'close_drop' if mode.startswith("Close") else 'gap'
                                thr = float(threshold)
                                result_df = _compute_gap_drop_stats(daily_hist, m, thr, None)
                                st.caption(f"Matches: {len(result_df)} events")
                                if not result_df.empty:
                                    # Summary KPIs
                                    c1,c2,c3,c4 = st.columns(4)
                                    if m == 'close_drop':
                                        with c1:
                                            st.metric("Avg Close?Close %", f"{result_df['Close_to_Close_%'].mean():.2f}%")
                                        with c2:
                                            st.metric("Avg Next Overnight %", f"{result_df['Next_Overnight_%'].mean():.2f}%")
                                        with c3:
                                            st.metric("Avg Next Intraday %", f"{result_df['Next_Intraday_%'].mean():.2f}%")
                                        with c4:
                                            st.metric("Avg Next Close?Close %", f"{result_df['Next_Close_to_Close_%'].mean():.2f}%")
                                    else:
                                        with c1:
                                            st.metric("Avg Intraday %", f"{result_df['Intraday_%'].mean():.2f}%")
                                        with c2:
                                            st.metric("Avg Next Overnight %", f"{result_df['Next_Overnight_%'].mean():.2f}%")
                                        with c3:
                                            st.metric("Avg Next Intraday %", f"{result_df['Next_Intraday_%'].mean():.2f}%")
                                        with c4:
                                            st.metric("Avg Next Total %", f"{result_df['Next_Total_%'].mean():.2f}%")

                                    st.dataframe(result_df, use_container_width=True)
                                else:
                                    st.info("No matching events for the selected criteria.")
                            except Exception as e:
                                st.error(f"Stats error: {e}")

                    # Optional: TradingView fallback expander (if the TV tab isn't visible in your setup)
                    with st.expander("TradingView (embedded)"):
                        _tv_interval_map = {"1m": "1", "5m": "5", "15m": "15", "30m": "30", "1h": "60", "60m": "60", "1d": "D"}
                        tv_interval = _tv_interval_map.get(interval, "D")
                        tv_theme = "dark" if template == "plotly_dark" else "light"
                        tv_symbol = tv_symbol_for(ticker)
                        tv_cfg = {
                            "container_id": "tv_container_exp",
                            "symbol": tv_symbol,
                            "interval": tv_interval,
                            "timezone": "Etc/UTC",
                            "theme": tv_theme,
                            "style": "1",
                            "hide_side_toolbar": True,
                            "allow_symbol_change": True,
                            "studies": [],
                            "autosize": True,
                        }
                        html_code = f"""
                        <div id=\"tv_container_exp\" style=\"height:{base_height}px; width:100%\"></div>
                        <script src=\"https://s3.tradingview.com/tv.js\"></script>
                        <script type=\"text/javascript\">
                          new TradingView.widget({json.dumps(tv_cfg)});
                        </script>
                        """
                        st.components.v1.html(html_code, height=base_height+20, scrolling=False)

                    # --- Backtest results section (always shows below chart when enabled) ---
                    if enable_backtest and strategy == "Price crosses above VWAP":
                        st.subheader("Backtest Trades")
                        if backtest_missing_anchor:
                            st.info("Set Anchored VWAP and provide an anchor to run this backtest.")
                        else:
                            if trades:
                                trade_df = pd.DataFrame(trades)
                                trade_df['holding_period'] = (trade_df['exit_time'] - trade_df['entry_time']).astype(str)
                                # KPI grid
                                k1, k2, k3 = st.columns(3)
                                total_trades = len(trade_df)
                                total_pnl = float(trade_df['pnl'].sum())
                                win_rate = float((trade_df['pnl'] > 0).mean() * 100.0)
                                with k1:
                                    st.metric("Total Trades", f"{total_trades}")
                                with k2:
                                    st.metric("Total P&L", f"{total_pnl:.2f}")
                                with k3:
                                    st.metric("Win Rate", f"{win_rate:.1f}%")
                                # Trades grid
                                st.dataframe(
                                    trade_df[['entry_time','entry_price','exit_time','exit_price','pnl','holding_period']],
                                    use_container_width=True,
                                )
                            else:
                                st.caption("No signals generated for the selected range and settings.")
    except Exception as e:
        st.error(f"Error fetching or plotting data: {e}")

    # Bottom Diagnostics expander (moved debug/info here to declutter UI)
    with st.expander("Diagnostics", expanded=False):
        try:
            dbg = st.session_state.get('today_debug') or {}
            provider = st.session_state.get('last_fetch_provider', '-')
            st.caption(f"Provider: {provider} | Rows today: {dbg.get('intraday_rows','-')}")
            if dbg:
                st.caption(f"Window: {dbg.get('intraday_start','-')} -> {dbg.get('intraday_end','-')} | Target: {dbg.get('target_date','-')}")
        except Exception:
            pass

# ---------------- TradingView ----------------
if nav == 'TradingView':
    st.subheader("TradingView (embedded)")
    _tv_interval_map = {"1m": "1", "5m": "5", "15m": "15", "30m": "30", "1h": "60", "60m": "60", "1d": "D"}
    tv_interval = _tv_interval_map.get(interval, "D")
    tv_theme = "dark" if template == "plotly_dark" else "light"
    tv_symbol = tv_symbol_for(ticker)
    tv_cfg = {
        "container_id": "tv_container_tab",
        "symbol": tv_symbol,
        "interval": tv_interval,
        "timezone": "Etc/UTC",
        "theme": tv_theme,
        "style": "1",
        "hide_side_toolbar": True,
        "allow_symbol_change": True,
        "studies": [],
        "autosize": True,
    }
    html_code = f"""
    <div id="tv_container_tab" style="height:{base_height}px; width:100%"></div>
    <script src="https://s3.tradingview.com/tv.js"></script>
    <script type="text/javascript">
      new TradingView.widget({json.dumps(tv_cfg)});
    </script>
    """
    st.components.v1.html(html_code, height=base_height+20, scrolling=False)

# ---------------- GPT-5 Agent ----------------
if nav == 'GPT-5 Agent':
    st.subheader("🤖 AI Equity Research Agent")

    # ── Config row ────────────────────────────────────────────────────────────
    _ai_c1, _ai_c2 = st.columns([3, 2])
    with _ai_c1:
        ticker_query = st.text_input("Ticker", value="AAPL", max_chars=12, key="ai_ticker")
        prompt_extra = st.text_area(
            "Research prompt",
            "Give me a concise equity snapshot: current price & market cap, recent earnings, "
            "key technicals (trend, RSI), top 3 risks, and analyst sentiment. Under 250 words.",
            height=100, key="ai_prompt",
        )
    with _ai_c2:
        _ai_provider = st.selectbox(
            "AI provider",
            ["Claude (Anthropic)", "GPT-4o (OpenAI)", "GPT-4o-mini (OpenAI)"],
            index=0, key="ai_provider_sel",
        )
        if _ai_provider.startswith("Claude"):
            _ai_key_default = os.getenv("ANTHROPIC_API_KEY", "").strip()
            _ai_key = st.text_input(
                "Anthropic API Key", value=_ai_key_default, type="password", key="ai_anthropic_key",
                help="Or set ANTHROPIC_API_KEY in your .env file",
            )
        else:
            _ai_key_default = os.getenv("OPENAI_API_KEY", "").strip()
            _ai_key = st.text_input(
                "OpenAI API Key", value=_ai_key_default, type="password", key="ai_openai_key",
                help="Or set OPENAI_API_KEY in your .env file",
            )

    # ── Build a rich market snapshot (Bloomberg → yfinance) ──────────────────
    def _rich_snapshot(sym: str) -> str:
        parts = []
        if HAS_XBBG:
            try:
                from xbbg import blp as _blp
                _flds = [
                    'PX_LAST', 'MARKET_CAP', 'PE_RATIO', 'BEST_EPS',
                    'EARN_ANN_DT', 'DVD_YLD_EST', '52WK_HIGH', '52WK_LOW', 'RSI_14D',
                ]
                _bd = _blp.bdp(f"{sym} US Equity", _flds)
                if _bd is not None and not _bd.empty:
                    r = _bd.iloc[0]
                    for f, label in [
                        ('px_last','Price'), ('market_cap','Mkt Cap'), ('pe_ratio','P/E'),
                        ('best_eps','EPS Est'), ('earn_ann_dt','Next Earnings'),
                        ('dvd_yld_est','Div Yield%'), ('52wk_high','52W Hi'),
                        ('52wk_low','52W Lo'), ('rsi_14d','RSI14'),
                    ]:
                        v = r.get(f, '')
                        if v not in ('', None):
                            parts.append(f"{label}: {v}")
                    if parts:
                        return " | ".join(parts)
            except Exception:
                pass
        # yfinance fallback
        try:
            import yfinance as _yf_ai
            _t = _yf_ai.Ticker(sym)
            _i = _t.info or {}
            for key, label in [
                ('currentPrice','Price'), ('marketCap','Mkt Cap'),
                ('trailingPE','P/E'), ('trailingEps','EPS'),
                ('fiftyTwoWeekHigh','52W Hi'), ('fiftyTwoWeekLow','52W Lo'),
            ]:
                v = _i.get(key, '')
                if v not in ('', None):
                    parts.append(f"{label}: {v}")
        except Exception:
            pass
        return " | ".join(parts) if parts else f"{sym}: data unavailable"

    # ── Run analysis ─────────────────────────────────────────────────────────
    if st.button("🔍 Run Analysis", type="primary", key="ai_run"):
        if not ticker_query.strip():
            st.error("Enter a ticker symbol.")
        elif not _ai_key.strip():
            st.error("Enter an API key (or set it in your .env file).")
        else:
            _sym = ticker_query.strip().upper()
            with st.spinner(f"Fetching data for {_sym} and calling AI…"):
                _snap = _rich_snapshot(_sym)
                _sys_msg = (
                    "You are a concise equity research assistant with access to real-time market data. "
                    "Format your response in clean markdown. Always end with: "
                    "*Disclaimer: This is not financial advice.*"
                )
                _usr_msg = (
                    f"Ticker: {_sym}\n"
                    f"Current Market Data: {_snap}\n\n"
                    f"Task: {prompt_extra}"
                )
                _response = None
                _err = None

                if _ai_provider.startswith("Claude"):
                    try:
                        import anthropic as _anthr
                        _ac = _anthr.Anthropic(api_key=_ai_key.strip())
                        _msg = _ac.messages.create(
                            model="claude-sonnet-4-6",
                            max_tokens=1024,
                            system=_sys_msg,
                            messages=[{"role": "user", "content": _usr_msg}],
                        )
                        _response = _msg.content[0].text if _msg.content else None
                    except Exception as e:
                        _err = str(e)
                        if "Could not resolve authentication" in _err or "api_key" in _err.lower():
                            _err = "Invalid or missing Anthropic API key. Add ANTHROPIC_API_KEY to your .env file."
                else:
                    try:
                        from openai import OpenAI as _OAI
                        _model = "gpt-4o-mini" if "mini" in _ai_provider else "gpt-4o"
                        _oc = _OAI(api_key=_ai_key.strip())
                        _or = _oc.chat.completions.create(
                            model=_model,
                            messages=[
                                {"role": "system", "content": _sys_msg},
                                {"role": "user", "content": _usr_msg},
                            ],
                            max_tokens=1024,
                        )
                        _response = _or.choices[0].message.content if _or and _or.choices else None
                    except Exception as e:
                        _err = str(e)

                if _response:
                    st.markdown(_response)
                    with st.expander("📊 Raw market data used", expanded=False):
                        st.code(_snap)
                elif _err:
                    st.error(f"AI call failed: {_err}")
                else:
                    st.warning("No response returned.")
    st.stop()
# --- Black-Scholes Delta function ---
def bs_delta(S, K, T, r, sigma, q, kind="call"):
    """
    Black-Scholes option delta.
    S: spot price
    K: strike price
    T: time to expiry (in years)
    r: risk-free rate
    sigma: volatility (annualized)
    q: dividend yield
    kind: "call" or "put"
    """
    import math
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return float("nan")
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    # Normal CDF without SciPy using math.erf
    def _phi(x: float) -> float:
        try:
            return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))
        except Exception:
            # Fallback rough approximation
            t = 1.0 / (1.0 + 0.2316419 * abs(x))
            d = 0.3989423 * math.exp(-x * x / 2.0)
            prob = 1.0 - d * (1.330274 * t**5 - 1.821256 * t**4 + 1.781478 * t**3 - 0.356538 * t**2 + 0.3193815 * t)
            return prob if x >= 0 else 1.0 - prob
    if str(kind).lower().startswith("c"):
        return math.exp(-q * T) * _phi(d1)
    else:
        return -math.exp(-q * T) * _phi(-d1)

def is_equity_symbol(sym: str) -> bool:
    s = (sym or "").strip().upper()
    if not s:
        return False
    if s.startswith('^'):
        return False
    if '=' in s or ':' in s:
        return False
    # crude: letters/digits up to 6 chars
    return s.replace('.', '').isalnum() and len(s) <= 6

def fetch_expirations(ticker: str, cache_buster: str | None = None) -> list[str]:
    """Fetch option expirations. yfinance first (always works), then Polygon."""
    st.session_state['opt_errors'] = []
    tkr_clean = ticker.strip().upper().lstrip('^')
    # yfinance first — free, no entitlement needed
    try:
        import yfinance as _yf_exp
        exps_yf = list(_yf_exp.Ticker(tkr_clean).options or [])
        if exps_yf:
            st.session_state['opt_last_provider'] = 'yfinance'
            st.session_state['opt_errors'].append(f'yfinance: OK — {len(exps_yf)} expirations')
            return sorted(exps_yf)
        else:
            st.session_state['opt_errors'].append('yfinance: 0 expirations returned')
    except Exception as _e:
        st.session_state['opt_errors'].append(f'yfinance: EXCEPTION — {_e}')
    # Polygon HTTP fallback
    try:
        api_key = (os.getenv('POLYGON_API_KEY') or os.getenv('MASSIVE_API_KEY') or '').strip()
        if api_key:
            import requests as _rq
            base_url = (os.getenv('POLYGON_API_BASE') or 'https://api.polygon.io') + '/v3/reference/options/contracts'
            params = {'underlying_ticker': tkr_clean, 'active': 'true', 'limit': 1000, 'apiKey': api_key}
            exps_set = set()
            url = base_url
            for _ in range(10):
                resp = _rq.get(url, params=params, timeout=10)
                if not resp.ok:
                    st.session_state['opt_errors'].append(f'polygon-http: HTTP {resp.status_code}')
                    break
                js = resp.json()
                for r in js.get('results') or []:
                    ed = r.get('expiration_date')
                    if ed:
                        exps_set.add(str(ed))
                next_url = js.get('next_url')
                if not next_url:
                    break
                url = next_url
                params = None
            if exps_set:
                st.session_state['opt_last_provider'] = 'polygon-http'
                return sorted(exps_set)
            else:
                st.session_state['opt_errors'].append('polygon-http: 0 expirations')
    except Exception as _e2:
        st.session_state['opt_errors'].append(f'polygon-http: EXCEPTION — {_e2}')
    return []

def fetch_thetadata_chain(ticker: str, expiration: str):
    """Fetch option chain from Theta Terminal (greeks/all), normalize columns, split calls/puts."""
    import pandas as _pd
    import requests as _rq, csv, io
    st.session_state['opt_last_provider'] = 'thetadata'
    base = (os.getenv("THETADATA_TERMINAL_URL") or "").strip().rstrip("/")
    if not base:
        base = "http://localhost:25503/v3"
    def _stream_csv(path: str) -> list[list[str]]:
        resp = _rq.get(f"{base}{path}", params={"symbol": ticker.strip().upper().lstrip("^"), "expiration": expiration}, timeout=10)
        if not resp.ok or not resp.text:
            st.session_state['opt_errors'].append(f"thetadata[{path}]: HTTP {resp.status_code}")
            return []
        rows = list(csv.reader(io.StringIO(resp.text)))
        return rows
    rows = _stream_csv("/option/snapshot/greeks/all")
    if not rows:
        rows = _stream_csv("/option/snapshot/ohlc")
    header = rows[0] if rows else []
    data_rows = rows[1:] if rows and header and any(not r.replace('.','',1).isdigit() for r in header) else rows
    if rows:
        if header and any(k.lower() in ("symbol","ticker","contract") for k in header):
            cols = [str(c).strip().lower().replace(' ', '_') for c in header]
        else:
            cols = ["symbol","expiration","strike","right","bid","ask","last","volume","open_interest","iv","delta","gamma","theta","vega","rho","timestamp"][:len(data_rows[0])]
        df = _pd.DataFrame(data_rows, columns=cols)
    else:
        df = _pd.DataFrame()

    # Quotes snapshot to fill OI/Vol if greeks snapshot is sparse
    rows_q = _stream_csv("/option/snapshot/quotes")
    df_q = _pd.DataFrame()
    if rows_q:
        hq = rows_q[0]
        drq = rows_q[1:] if hq and any(not r.replace('.','',1).isdigit() for r in hq) else rows_q
        if hq and any(k.lower() in ("symbol","ticker","contract") for k in hq):
            cols_q = [str(c).strip().lower().replace(' ', '_') for c in hq]
        else:
            cols_q = ["symbol","expiration","strike","right","bid","ask","last","volume","open_interest","iv","timestamp"][:len(drq[0])]
        df_q = _pd.DataFrame(drq, columns=cols_q)
        df_q.columns = [str(c).replace(' ', '_').lower() for c in df_q.columns]

    def _normalize(df_in: _pd.DataFrame) -> _pd.DataFrame:
        df_local = df_in.copy()
        df_local.columns = [str(c).replace(' ', '_').lower() for c in df_local.columns]
        ren = {
            'symbol': 'contractsymbol',
            'ticker': 'contractsymbol',
            'contract': 'contractsymbol',
            'right': 'option_type',
            'type': 'option_type',
            'last': 'last_price',
            'last_price': 'last_price',
            'close': 'last_price',
            'oi': 'open_interest',
        }
        for src, dst in ren.items():
            if src in df_local.columns and dst not in df_local.columns:
                df_local[dst] = df_local[src]
        for c in ('strike','bid','ask','last_price','volume','open_interest','implied_volatility','iv','delta','gamma','theta','vega','rho'):
            if c in df_local.columns:
                df_local[c] = _pd.to_numeric(df_local[c], errors="coerce")
        if 'expiration' in df_local.columns:
            try:
                df_local['expiration'] = _pd.to_datetime(df_local['expiration'], errors='coerce').dt.date.astype(str)
            except Exception:
                pass
        return df_local

    df = _normalize(df) if not df.empty else df
    df_q = _normalize(df_q) if not df_q.empty else df_q

    # Merge quotes info into main greeks snapshot if needed
    if not df.empty and not df_q.empty and 'contractsymbol' in df.columns and 'contractsymbol' in df_q.columns:
        df = df.merge(df_q, on=['contractsymbol','expiration','strike','option_type'], how='left', suffixes=('', '_q'))
        for col in ('open_interest','volume','implied_volatility'):
            if col in df.columns and col + '_q' in df.columns:
                primary = df[col]
                fallback = df[col + '_q']
                df[col] = primary.where(primary.notna(), fallback)
                df.drop(columns=[col + '_q'], inplace=True)
    elif df.empty and not df_q.empty:
        df = df_q

    if df.empty:
        return df, df
    calls = df[df['option_type'].astype(str).str.startswith(('c','C'), na=False)].copy()
    puts  = df[df['option_type'].astype(str).str.startswith(('p','P'), na=False)].copy()
    return calls, puts

def fetch_chain(ticker: str, expiration: str, prefer_theta: bool = False):
    """Fetch option chain: yfinance first (free), then Polygon if entitlement available."""
    import pandas as _pd
    st.session_state['opt_errors'] = []  # reset — stale errors from prior calls were showing

    def _from_thetadata() -> tuple[_pd.DataFrame, _pd.DataFrame]:
        try:
            return fetch_thetadata_chain(ticker, expiration)
        except Exception as e:
            st.session_state['opt_errors'].append(f'thetadata: {e}')
            return _pd.DataFrame(), _pd.DataFrame()

    def _from_polygon() -> tuple[_pd.DataFrame, _pd.DataFrame]:
        try:
            api_key = (os.getenv('POLYGON_API_KEY') or os.getenv('MASSIVE_API_KEY') or '').strip()
            if api_key:
                import requests as _rq
                bases = [
                    os.getenv('POLYGON_API_BASE') or 'https://api.polygon.io',
                    os.getenv('MASSIVE_API_BASE') or 'https://api.massive.com',
                ]
                rows = []
                # Polygon/Massive snapshot endpoint expects underlying in the path
                under_sym = ticker.strip().upper().lstrip('^')
                params = {
                    "expiration_date": expiration,
                    "limit": 1000,
                    "apiKey": api_key,
                }
                for base_try in bases:
                    snap_url = f"{base_try}/v3/snapshot/options/{under_sym}"
                    try:
                        resp = _rq.get(snap_url, params=params, timeout=10)
                        if not resp.ok:
                            st.session_state['opt_errors'].append(f'polygon-snapshot[{base_try}]: HTTP {resp.status_code}')
                            continue
                        js = resp.json()
                        for r in js.get("results") or []:
                            det = r.get("details", {})
                            lq = r.get("last_quote", {}) or {}
                            lt = r.get("last_trade", {}) or {}
                            sess = r.get("session", {}) or {}
                            greeks = r.get("greeks", {}) or {}
                            rows.append({
                                "contractsymbol": r.get("ticker"),
                                "expiration": det.get("expiration_date"),
                                "strike": det.get("strike_price"),
                                "option_type": (det.get("contract_type") or "").lower(),
                                "bid": lq.get("bid"),
                                "ask": lq.get("ask"),
                                "last_price": lt.get("price") or sess.get("close") or lq.get("midpoint"),
                                "volume": sess.get("volume"),
                                "open_interest": r.get("open_interest"),
                                "implied_volatility": r.get("implied_volatility"),
                                "delta": greeks.get("delta"),
                                "gamma": greeks.get("gamma"),
                                "theta": greeks.get("theta"),
                                "vega": greeks.get("vega"),
                            })
                        if rows:
                            break
                    except Exception as ie:
                        st.session_state['opt_errors'].append(f'polygon-snapshot[{base_try}]: {ie}')
                if not rows:
                    st.session_state['opt_last_provider'] = 'polygon-snapshot'
                    st.session_state['opt_errors'].append('polygon-snapshot: empty result (check entitlements for options snapshot)')
                    # Fallback: reference/contracts endpoint to at least show strikes/types
                    try:
                        base_ref = os.getenv('POLYGON_API_BASE') or 'https://api.polygon.io'
                        def _walk_contracts(active_flag: str | None):
                            url_ref = f'{base_ref}/v3/reference/options/contracts'
                            params_ref = {
                                "underlying_ticker": under_sym,
                                "expiration_date": expiration,
                                "limit": 1000,
                                "apiKey": api_key,
                            }
                            if active_flag is not None:
                                params_ref["active"] = active_flag
                            rows_local = []
                            url_walk = url_ref
                            params_walk = params_ref
                            for _ in range(10):
                                resp = _rq.get(url_walk, params=params_walk, timeout=10)
                                if not resp.ok:
                                    st.session_state['opt_errors'].append(f'polygon-http: HTTP {resp.status_code}')
                                    break
                                js = resp.json()
                                for r in js.get('results') or []:
                                    rows_local.append({
                                        'contractsymbol': r.get('ticker'),
                                        'expiration': r.get('expiration_date'),
                                        'strike': r.get('strike_price'),
                                        'option_type': (r.get('contract_type') or '').lower(),
                                        'bid': None,
                                        'ask': None,
                                        'last_price': None,
                                        'volume': r.get('volume'),
                                        'open_interest': r.get('open_interest'),
                                        'implied_volatility': r.get('implied_volatility'),
                                        'delta': None,
                                        'gamma': None,
                                        'theta': None,
                                        'vega': None,
                                    })
                                next_url = js.get('next_url')
                                if not next_url:
                                    break
                                url_walk = next_url
                                params_walk = None
                            return rows_local
                        rows_ref = _walk_contracts("true")
                        if not rows_ref:
                            rows_ref = _walk_contracts(None)
                        if rows_ref:
                            df = _pd.DataFrame(rows_ref)
                            df.columns = [str(c).replace(' ', '_').lower() for c in df.columns]
                            calls = df[df['option_type'].astype(str).str.startswith(('c','C'), na=False)].copy()
                            puts  = df[df['option_type'].astype(str).str.startswith(('p','P'), na=False)].copy()
                            st.session_state['opt_last_provider'] = 'polygon-http'
                            st.session_state['opt_errors'].append('polygon-http: contracts fallback (no quotes/greeks)')
                            return calls, puts
                    except Exception as ie2:
                        st.session_state['opt_errors'].append(f'polygon-http fallback: {ie2}')
                else:
                    df = _pd.DataFrame(rows)
                    df.columns = [str(c).replace(' ', '_').lower() for c in df.columns]
                    calls = df[df['option_type'].astype(str).str.startswith('c', na=False)].copy()
                    puts  = df[df['option_type'].astype(str).str.startswith('p', na=False)].copy()
                    st.session_state['opt_last_provider'] = 'polygon-snapshot'
                    return calls, puts
        except Exception as e:
            st.session_state['opt_errors'].append(f'polygon-snapshot: {e}')
        try:
            api_key = (os.getenv('POLYGON_API_KEY') or os.getenv('MASSIVE_API_KEY') or '').strip()
            if api_key:
                import requests as _rq
                _base = os.getenv('MASSIVE_API_BASE') or os.getenv('POLYGON_API_BASE') or 'https://api.polygon.io'
                base_url = f'{_base}/v3/reference/options/contracts'
                params = {
                    "underlying_ticker": ticker.strip().upper().lstrip('^'),
                    "expiration_date": expiration,
                    "active": "true",
                    "limit": 1000,
                    "apiKey": api_key,
                }
                url = base_url
                rows = []
                for _ in range(10):
                    resp = _rq.get(url, params=params, timeout=10)
                    if not resp.ok:
                        st.session_state['opt_errors'].append(f'polygon-http: HTTP {resp.status_code}')
                        break
                    js = resp.json()
                    results = js.get('results') or []
                    for r in results:
                        rows.append({
                            'contractsymbol': r.get('ticker'),
                            'strike': r.get('strike_price'),
                            'option_type': (r.get('contract_type') or '').lower(),
                            'expiration': r.get('expiration_date'),
                            'bid': r.get('bid_price'),
                            'ask': r.get('ask_price'),
                            'last_price': r.get('last_quote',{}).get('last',{}).get('price'),
                            'volume': r.get('day',{}).get('volume'),
                            'open_interest': r.get('open_interest'),
                            'implied_volatility': r.get('implied_volatility'),
                        })
                    next_url = js.get('next_url')
                    if not next_url:
                        break
                    url = next_url
                    params = None
                if rows:
                    df = _pd.DataFrame(rows)
                    df.columns = [str(c).replace(' ', '_').lower() for c in df.columns]
                    calls = df[df['option_type'].astype(str).str.startswith('c', na=False)].copy()
                    puts  = df[df['option_type'].astype(str).str.startswith('p', na=False)].copy()
                    st.session_state['opt_last_provider'] = 'polygon-http'
                    return calls, puts
                else:
                    st.session_state['opt_errors'].append('polygon-http: empty chain')
        except Exception as e:
            st.session_state['opt_errors'].append(f'polygon-http: {e}')
        return _pd.DataFrame(), _pd.DataFrame()

    def _from_yfinance() -> tuple[_pd.DataFrame, _pd.DataFrame]:
        try:
            import yfinance as _yf2
            tkr_clean = ticker.strip().upper().lstrip('^')
            chain = _yf2.Ticker(tkr_clean).option_chain(expiration)
            calls_yf = chain.calls.copy() if chain.calls is not None and not chain.calls.empty else _pd.DataFrame()
            puts_yf  = chain.puts.copy()  if chain.puts  is not None and not chain.puts.empty  else _pd.DataFrame()
            if calls_yf.empty and puts_yf.empty:
                st.session_state['opt_errors'].append(f'yfinance: returned 0 rows for {tkr_clean} exp={expiration}')
                return _pd.DataFrame(), _pd.DataFrame()
            # Normalize all column names to lowercase_underscore
            for df in (calls_yf, puts_yf):
                df.columns = [
                    c.replace('contractSymbol','contract_symbol')
                     .replace('lastPrice','last_price')
                     .replace('openInterest','open_interest')
                     .replace('impliedVolatility','implied_volatility')
                     .replace('inTheMoney','in_the_money')
                     .replace('lastTradeDate','last_trade_date')
                     .replace('Strike','strike')
                     .replace('percentChange','percent_change')
                     .replace('contractSize','contract_size')
                     .replace('currency','currency')
                    for c in df.columns
                ]
                # Add contractsymbol alias so Strike Viewer works
                if 'contract_symbol' in df.columns and 'contractsymbol' not in df.columns:
                    df['contractsymbol'] = df['contract_symbol']
            st.session_state['opt_last_provider'] = 'yfinance'
            st.session_state['opt_errors'].append(
                f'yfinance: OK — calls={len(calls_yf)} puts={len(puts_yf)}'
            )
            return calls_yf, puts_yf
        except Exception as _e_yf:
            st.session_state['opt_errors'].append(f'yfinance: EXCEPTION — {_e_yf}')
            return _pd.DataFrame(), _pd.DataFrame()

    # order: ThetaData if preferred, then yfinance (always works, no entitlement),
    # then Polygon snapshot (only if options entitlement available)
    if prefer_theta:
        calls, puts = _from_thetadata()
        if (calls is not None and not calls.empty) or (puts is not None and not puts.empty):
            st.session_state['opt_last_provider'] = 'thetadata'
            return calls, puts
    # yfinance first — free, full bid/ask/IV/OI/volume, no entitlement needed
    calls, puts = _from_yfinance()
    if (calls is not None and not calls.empty) or (puts is not None and not puts.empty):
        return calls, puts
    # Polygon fallback — only useful if you have the options snapshot entitlement
    calls, puts = _from_polygon()
    return calls, puts

def spot_price(ticker: str) -> float | None:
    # Try local parquet first (fast, no network)
    try:
        df_loc = _load_daily_df(ticker)
        if df_loc is not None and not df_loc.empty:
            last = df_loc.sort_values("Date")["Close"].dropna()
            if not last.empty:
                return float(last.iloc[-1])
    except Exception:
        pass
    # yfinance fallback
    try:
        t = yf.Ticker(ticker)
        h = t.history(period="1d")
        if not h.empty:
            return float(h["Close"].iloc[-1])
    except Exception:
        pass
    return None

if nav == 'Options':
    st.subheader("Options Chain")
    # Config section for greeks & charts
    cfg_col1, cfg_col2, cfg_col3, cfg_col4 = st.columns([1,1,1,2])
    with cfg_col1:
        r_rate = st.number_input("Risk-free r (decimal)", value=0.02, min_value=0.0, max_value=0.2, step=0.005, format="%.3f")
    with cfg_col2:
        q_div  = st.number_input("Dividend q (decimal)", value=0.0, min_value=0.0, max_value=0.2, step=0.005, format="%.3f")
    with cfg_col3:
        show_oi_chart  = st.checkbox("OI by Strike", value=True)
        show_vol_chart = st.checkbox("Vol by Strike", value=True)
        show_iv_chart  = st.checkbox("IV (Smile)", value=True)
    with cfg_col4:
        load_opts = st.button("Load / Refresh Chain", use_container_width=True)
    strike_window_pct = st.number_input("Strike window around spot (%)", value=20.0, min_value=1.0, max_value=200.0, step=5.0, help="Keep strikes within ± this % of spot")
    use_theta = st.checkbox("Use ThetaData (Theta Terminal required)", value=bool(os.getenv("THETADATA_TERMINAL_URL","").strip()), help="If enabled, fetch chain from Theta Terminal first, then fall back to Polygon.")

    if not load_opts and "opts_expirations" not in st.session_state:
        st.info("Click **Load / Refresh Chain** to fetch expirations and the nearest chain.")
    else:
        try:
            if load_opts or "opts_expirations" not in st.session_state:
                exps = fetch_expirations(ticker)
                if not exps:
                    st.warning("No options data available for this ticker.")
                    try:
                        diag = st.session_state.get('opt_errors') or []
                        prov = st.session_state.get('opt_last_provider', '-')
                        if diag:
                            st.caption(f"Options diagnostics (last provider={prov}): {' | '.join(diag[-3:])}")
                    except Exception:
                        pass
                    exps = []
                st.session_state["opts_expirations"] = exps

                if exps:
                    today = datetime.now(timezone.utc).date()
                    try:
                        exps_dt = [datetime.strptime(e, "%Y-%m-%d").date() for e in exps]
                    except Exception:
                        exps_dt = []
                    future_pairs = [(e, d) for e, d in zip(exps, exps_dt) if d and d >= today]
                    if future_pairs:
                        nearest = min(future_pairs, key=lambda p: (p[1] - today).days)[0]
                    else:
                        nearest = min(exps, key=lambda e: abs((datetime.strptime(e, "%Y-%m-%d").date() - today).days))
                    st.session_state["opts_selected_exp"] = nearest

            exps = st.session_state.get("opts_expirations", [])
            if exps:
                today = datetime.now(timezone.utc).date()
                try:
                    exps_dt = [datetime.strptime(e, "%Y-%m-%d").date() for e in exps]
                except Exception:
                    exps_dt = []
                future = [e for e, d in zip(exps, exps_dt) if d and d >= today]
                exps_use = future or exps
                sel_default = st.session_state.get("opts_selected_exp", (exps_use[0] if exps_use else None))
                sel = st.selectbox("Expiration", options=exps_use, index=(exps_use.index(sel_default) if sel_default in exps_use else 0))
                st.session_state["opts_selected_exp"] = sel

                with st.spinner(f"Fetching chain for {sel}..."):
                    calls, puts = fetch_chain(ticker, sel, prefer_theta=bool(use_theta))
                    if (calls is None or calls.empty) and (puts is None or puts.empty):
                        tried = [sel]
                        alt_found = False
                        for alt in exps_use:
                            if alt in tried:
                                continue
                            c2, p2 = fetch_chain(ticker, alt)
                            if (c2 is not None and not c2.empty) or (p2 is not None and not p2.empty):
                                calls, puts = c2, p2
                                sel = alt
                                st.session_state["opts_selected_exp"] = alt
                                st.info(f"Selected nearest expiration with data: {alt}")
                                alt_found = True
                                break
                        if not alt_found:
                            st.warning("No option chain returned for this expiration.")
                            try:
                                diag = st.session_state.get('opt_errors') or []
                                prov = st.session_state.get('opt_last_provider', '-')
                                if diag:
                                    st.caption(f"Options diagnostics (last provider={prov}): {' | '.join(diag[-3:])}")
                            except Exception:
                                pass
                    if (calls is None or calls.empty) and (puts is None or puts.empty):
                        diag = st.session_state.get('opt_errors') or []
                        prov = st.session_state.get('opt_last_provider', '-')
                        st.warning("No option chain returned.")
                        if diag:
                            st.caption(f"Options diagnostics (last provider={prov}): {' | '.join(diag[-5:])}")
                        calls, puts = pd.DataFrame(), pd.DataFrame()

                    S = spot_price(ticker)
                    exp_dt = datetime.strptime(sel, "%Y-%m-%d").date()
                    T = max((exp_dt - datetime.now(timezone.utc).date()).days, 0) / 365.0
                    if S is not None and not pd.isna(S):
                        lo = S * (1 - strike_window_pct / 100.0)
                        hi = S * (1 + strike_window_pct / 100.0)
                        try:
                            if calls is not None and not calls.empty and 'strike' in calls.columns:
                                calls = calls[(calls['strike'] >= lo) & (calls['strike'] <= hi)]
                            if puts is not None and not puts.empty and 'strike' in puts.columns:
                                puts = puts[(puts['strike'] >= lo) & (puts['strike'] <= hi)]
                        except Exception:
                            pass

            def add_delta(df: pd.DataFrame, kind: str) -> pd.DataFrame:
                out = df.copy()
                # If provider already gave delta, use it
                if "delta" in out.columns:
                    try:
                        out["delta"] = pd.to_numeric(out["delta"], errors="coerce")
                        if out["delta"].notna().any():
                            return out
                    except Exception:
                        pass
                # Resolve columns robustly for BS backfill
                iv_col = None
                for c in ("implied_volatility", "impliedvolatility", "iv", "impl_vol"):
                    if c in out.columns:
                        iv_col = c
                        break
                strike_col = None
                for c in ("strike", "strike_price", "strikeprice", "k"):
                    if c in out.columns:
                        strike_col = c
                        break
                if S is None or T <= 0 or iv_col is None or strike_col is None:
                    out["delta"] = np.nan
                    return out
                iv = pd.to_numeric(out[iv_col], errors="coerce").astype(float)
                strikes = pd.to_numeric(out[strike_col], errors="coerce").astype(float)
                deltas = []
                for k, sig in zip(strikes, iv):
                    deltas.append(bs_delta(S, float(k), float(T), float(r_rate), float(sig or 0.0), float(q_div), kind))
                out["delta"] = deltas
                return out

            calls = add_delta(calls, "call")
            puts  = add_delta(puts,  "put")

            # Filters
            f1, f2, f3 = st.columns(3)
            with f1:
                min_oi = st.number_input("Min Open Interest", value=0, min_value=0, step=10)
            with f2:
                min_vol = st.number_input("Min Volume", value=0, min_value=0, step=10)
            with f3:
                moneyness = st.selectbox("Moneyness", ["All", "OTM", "ATM (+/-1%)", "ITM"])

            # moneyness tagging (uses spot S if available)
            def tag_filter(df: pd.DataFrame, kind: str) -> pd.DataFrame:
                out = df.copy()
                # find strike column
                s_col = None
                for c in ("strike", "strike_price", "strikeprice", "k"):
                    if c in out.columns:
                        s_col = c
                        break
                if S is not None and s_col is not None:
                    out["moneyness"] = (out[s_col] - S) / S
                    if kind == "call":
                        itm_mask = out[s_col] < S
                    else:
                        itm_mask = out[s_col] > S
                    if moneyness == "OTM":
                        out = out[~itm_mask]
                    elif moneyness == "ITM":
                        out = out[itm_mask]
                    elif moneyness == "ATM (+/-1%)":
                        out = out[abs(out["moneyness"]) <= 0.01]
                if "open_interest" in out.columns:
                    out = out[out["open_interest"].fillna(0) >= min_oi]
                if "volume" in out.columns:
                    out = out[out["volume"].fillna(0) >= min_vol]
                return out

            cc = tag_filter(calls if calls is not None else pd.DataFrame(), "call").sort_values("strike") if calls is not None else pd.DataFrame()
            pp = tag_filter(puts if puts is not None else pd.DataFrame(),  "put").sort_values("strike") if puts is not None else pd.DataFrame()

            # Tables (black on white ensured by global CSS)
            t1, t2 = st.tabs(["Calls", "Puts"])
            with t1:
                if cc.empty:
                    st.info("No call data to display for this expiration/filters.")
                else:
                    st.dataframe(cc, use_container_width=True)
            with t2:
                if pp.empty:
                    st.info("No put data to display for this expiration/filters.")
                else:
                    st.dataframe(pp, use_container_width=True)

            # Diagnostics: provider + non-null counts
            try:
                prov = st.session_state.get('opt_last_provider', '-')
                diag = st.session_state.get('opt_errors') or []
                def _nn(df, col):
                    return int(pd.to_numeric(df[col], errors="coerce").notna().sum()) if (col in df.columns and not df.empty) else 0
                diag_msg = (
                    f"Provider: {prov} | "
                    f"OI (calls/puts): {_nn(cc,'open_interest')}/{_nn(pp,'open_interest')} | "
                    f"Vol (calls/puts): {_nn(cc,'volume')}/{_nn(pp,'volume')} | "
                    f"IV (calls/puts): {_nn(cc,'implied_volatility')}/{_nn(pp,'implied_volatility')}"
                )
                st.caption(diag_msg)
                if diag:
                    st.caption("Diagnostics: " + " | ".join(diag[-5:]))
                if cc.empty and pp.empty:
                    st.warning("No option rows returned from Theta/Polygon for this expiration (check filters, strike window, provider).")
            except Exception:
                pass

            # Charts vs strike
            def add_vline(fig, xval, label):
                if xval is None:
                    return
                fig.add_vline(x=xval, line_dash="dash", line_color="#888", annotation_text=label, annotation_position="top")

            if show_oi_chart:
                fig_oi = go.Figure()
                if "open_interest" in cc.columns and not cc.empty:
                    fig_oi.add_trace(go.Scatter(x=cc["strike"], y=cc["open_interest"], mode="lines+markers", name="Calls OI"))
                if "open_interest" in pp.columns and not pp.empty:
                    fig_oi.add_trace(go.Scatter(x=pp["strike"], y=pp["open_interest"], mode="lines+markers", name="Puts OI"))
                if fig_oi.data:
                    add_vline(fig_oi, S, "Spot")
                    fig_oi.update_layout(title=f"Open Interest by Strike - {sel}", xaxis_title="Strike", yaxis_title="Open Interest", height=380)
                    st.plotly_chart(fig_oi, use_container_width=True)
                else:
                    st.info("No OI data available to chart for this expiration/filters.")

            if show_vol_chart:
                fig_vol = go.Figure()
                if "volume" in cc.columns and not cc.empty:
                    fig_vol.add_trace(go.Scatter(x=cc["strike"], y=cc["volume"], mode="lines+markers", name="Calls Volume"))
                if "volume" in pp.columns and not pp.empty:
                    fig_vol.add_trace(go.Scatter(x=pp["strike"], y=pp["volume"], mode="lines+markers", name="Puts Volume"))
                if fig_vol.data:
                    add_vline(fig_vol, S, "Spot")
                    fig_vol.update_layout(title=f"Volume by Strike - {sel}", xaxis_title="Strike", yaxis_title="Volume", height=380)
                    st.plotly_chart(fig_vol, use_container_width=True)
                else:
                    st.info("No volume data available to chart for this expiration/filters.")

            if show_iv_chart:
                fig_iv = go.Figure()
                if "implied_volatility" in cc.columns and not cc.empty:
                    fig_iv.add_trace(go.Scatter(x=cc["strike"], y=cc["implied_volatility"]*100, mode="lines+markers", name="Calls IV%"))
                if "implied_volatility" in pp.columns and not pp.empty:
                    fig_iv.add_trace(go.Scatter(x=pp["strike"], y=pp["implied_volatility"]*100, mode="lines+markers", name="Puts IV%"))
                if fig_iv.data:
                    add_vline(fig_iv, S, "Spot")
                    fig_iv.update_layout(title=f"Implied Volatility (Smile) - {sel}", xaxis_title="Strike", yaxis_title="IV (%)", height=380)
                    st.plotly_chart(fig_iv, use_container_width=True)
                else:
                    st.info("No IV data available to chart for this expiration/filters.")

            # Strike-level viewer for charts
            if not cc.empty or not pp.empty:
                st.subheader("Strike Viewer")
                merged = pd.concat([cc.assign(side="Call"), pp.assign(side="Put")], axis=0, ignore_index=True)
                merged = merged.sort_values("strike")
                opt_sym = st.selectbox(
                    "Select contract",
                    options=merged["contractsymbol"] if "contractsymbol" in merged.columns else merged.index,
                    format_func=lambda x: str(x),
                )
                sel_row = merged[merged["contractsymbol"] == opt_sym].iloc[0] if "contractsymbol" in merged.columns and opt_sym in set(merged["contractsymbol"]) else None
                if sel_row is not None:
                    st.caption(f"{sel_row.get('side','')} | Strike: {sel_row.get('strike')} | Exp: {sel_row.get('expiration')} | Bid/Ask: {sel_row.get('bid')}/{sel_row.get('ask')} | Last: {sel_row.get('last_price')}")
                    # Daily mini-chart placeholder using bid/ask/last if available
                    try:
                        px_cols = [c for c in ("bid","ask","last_price") if c in merged.columns]
                        if px_cols:
                            st.line_chart(merged.set_index("strike")[px_cols], height=200)
                    except Exception:
                        pass

            # theme for options charts (match main)
            for f in ["fig_oi", "fig_vol", "fig_iv"]:
                pass  # (plotly template is fine; page-wide colors already set)

            if S is not None:
                st.caption(f"Spot: {S:.2f} | Expirations: {len(exps)} | Showing: {sel}")
            else:
                st.caption(f"Spot unavailable | Expirations: {len(exps)} | Showing: {sel}")

        except Exception as e:
            st.error(f"Options error: {e}")

# ---------------- Premarket ----------------
if nav == 'Premarket':
    st.subheader("Premarket Movers")

    # ── Controls ─────────────────────────────────────────────────────────────
    _pm1, _pm2, _pm3, _pm4, _pm5 = st.columns([2, 1, 1, 1, 1])
    with _pm1:
        scan_date = st.date_input("Scan date", value=pd.Timestamp("today").date(), key="pm_scan_date")
    with _pm2:
        top_n = st.number_input("Top N", value=25, min_value=5, max_value=200, step=5, key="pm_top_n")
    with _pm3:
        min_prev_close = st.number_input("Min price ($)", value=5.0, min_value=0.0, step=0.5, key="pm_min_price")
    with _pm4:
        min_volume = st.number_input("Min volume", value=0.0, min_value=0.0, step=100000.0, format="%.0f", key="pm_min_vol")
    with _pm5:
        min_premarket = st.number_input("Min abs %", value=1.0, min_value=0.0, step=0.5, key="pm_min_pct")

    _pd1, _pd2 = st.columns(2)
    with _pd1:
        pm_source = st.radio(
            "Source",
            ["Local minutes (fast)", "API snapshot (Polygon / Massive)"],
            horizontal=True, key="pm_source_sel",
        )
    with _pd2:
        sort_by_pm = st.radio(
            "Sort by",
            ["Abs Premarket %", "Volume"],
            horizontal=True, key="pm_sort",
        )
    run_pm = st.button("🔍 Run Premarket Scan", type="primary", use_container_width=True, key="pm_run_main")

    st.markdown("---")

    # ── Execute scan ─────────────────────────────────────────────────────────
    _key_ok = bool(_massive_key())
    if run_pm:
        if pm_source == "Local minutes (fast)":
            _pm_tickers = _list_minute_tickers(limit=2000)
            if not _pm_tickers:
                st.warning("No local minute data found — switch to API snapshot.")
                _df_pm = None
            else:
                with st.spinner(f"Scanning {len(_pm_tickers):,} local tickers for {scan_date}…"):
                    _df_pm = _scan_premarket_local_minutes(
                        pd.Timestamp(scan_date),
                        _pm_tickers,
                        min_vol=float(min_volume),
                        max_requests=2000,
                    )
                if _df_pm is None or _df_pm.empty:
                    st.info("No premarket movers found in local minutes for that date. Try API snapshot.")
                    _df_pm = None
        else:
            if not _key_ok:
                st.error("MASSIVE_API_KEY or POLYGON_API_KEY not set in .env.")
                _df_pm = None
            else:
                with st.spinner("Fetching premarket snapshot…"):
                    _g_df, _l_df, _pm_errs = _massive_premarket_movers(
                        min_volume=float(min_volume), limit=int(top_n) * 4,
                    )
                if _pm_errs:
                    st.caption(" | ".join(_pm_errs[-3:]))
                _dfs = [d for d in [_g_df, _l_df] if d is not None and not d.empty]
                _df_pm = pd.concat(_dfs, ignore_index=True) if _dfs else None
                if _df_pm is None or _df_pm.empty:
                    st.info("No premarket movers returned (check API key / entitlement).")
                    _df_pm = None
        try:
            st.session_state["premarket_scan_df"] = _df_pm.copy() if _df_pm is not None else pd.DataFrame()
        except Exception:
            pass
    else:
        _cached = st.session_state.get("premarket_scan_df")
        _df_pm = _cached if isinstance(_cached, pd.DataFrame) and not _cached.empty else None

    # ── Results table ─────────────────────────────────────────────────────────
    if _df_pm is not None and not _df_pm.empty:
        _df_show = _df_pm.copy()
        for _col in ["Premarket %", "Volume", "Prev Close", "Open", "Close"]:
            if _col in _df_show.columns:
                _df_show[_col] = pd.to_numeric(_df_show[_col], errors="coerce")
        if min_prev_close > 0 and "Prev Close" in _df_show.columns:
            _df_show = _df_show[_df_show["Prev Close"].fillna(0) >= min_prev_close]
        if min_volume > 0 and "Volume" in _df_show.columns:
            _df_show = _df_show[_df_show["Volume"].fillna(0) >= min_volume]
        if min_premarket > 0 and "Premarket %" in _df_show.columns:
            _df_show = _df_show[_df_show["Premarket %"].abs() >= min_premarket]
        _df_show = _df_show.dropna(subset=["Premarket %"])
        _df_show["_abs_pct"] = _df_show["Premarket %"].abs()
        if sort_by_pm == "Abs Premarket %":
            _df_show = _df_show.sort_values("_abs_pct", ascending=False, na_position="last")
        elif "Volume" in _df_show.columns:
            _df_show = _df_show.sort_values(["Volume", "_abs_pct"], ascending=[False, False], na_position="last")
        _df_show = _df_show.drop(columns=["_abs_pct"], errors="ignore")
        st.caption(f"{len(_df_show):,} movers • {scan_date}")
        st.dataframe(_df_show.head(int(top_n)), use_container_width=True)
    else:
        st.info("Click **Scan** to load premarket movers.")

    # ── Historical Context ────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("Historical Context — similar premarket days", expanded=False):
        _scan_ref = st.session_state.get("premarket_scan_df")
        _scan_tickers = (
            sorted(_scan_ref["Ticker"].dropna().unique().tolist())
            if isinstance(_scan_ref, pd.DataFrame) and not _scan_ref.empty and "Ticker" in _scan_ref.columns
            else []
        )
        _hc1, _hc2, _hc3 = st.columns([2, 1, 1])
        with _hc1:
            if _scan_tickers:
                tkr_pm = st.selectbox("Ticker", options=_scan_tickers, key="pm_hist_ticker")
            else:
                tkr_pm = st.text_input("Ticker", value=ticker, key="pm_hist_ticker_txt").strip().upper()
        with _hc2:
            lookback_days = st.number_input("Lookback (days)", value=120, min_value=30, max_value=2000, step=30, key="pm_lookback")
        with _hc3:
            tol_pct = st.number_input("\u00b1 % tolerance", value=0.5, min_value=0.1, max_value=10.0, step=0.1, key="pm_tol")

        _hm1, _hm2 = st.columns([3, 1])
        with _hm1:
            minute_marks = st.multiselect(
                "Minute marks after open",
                options=[1, 2, 3, 5, 10, 15, 30, 60],
                default=[1, 3, 5, 10, 15],
                key="pm_marks",
            )
        with _hm2:
            _default_target = 0.0
            if isinstance(_scan_ref, pd.DataFrame) and not _scan_ref.empty and tkr_pm:
                try:
                    _row = _scan_ref[
                        (_scan_ref["Ticker"] == tkr_pm) &
                        (pd.to_datetime(_scan_ref["Date"]).dt.date == pd.Timestamp(scan_date).date())
                    ]
                    if not _row.empty and "Premarket %" in _row.columns:
                        _default_target = float(_row["Premarket %"].iloc[0])
                except Exception:
                    pass
            target_pct = st.number_input("Target %", value=float(_default_target), step=0.1, format="%.2f", key="pm_target")

        run_hist = st.button("Find similar days", key="pm_hist_btn", type="primary")

        if run_hist and tkr_pm:
            with st.spinner(f"Loading {lookback_days}d premarket history for {tkr_pm}…"):
                _hist = _premarket_history_for_ticker(tkr_pm, lookback_days=int(lookback_days))
            if _hist is None or _hist.empty:
                st.info("No premarket history found for that ticker.")
            else:
                _hist = _hist.copy()
                _hist["Premarket %"] = pd.to_numeric(_hist["Premarket %"], errors="coerce")
                _hist["Delta"] = (_hist["Premarket %"] - float(target_pct)).abs()
                _matches = _hist[_hist["Delta"] <= float(tol_pct)].sort_values("Delta")
                if _matches.empty:
                    st.info(f"No days within \u00b1{tol_pct}% of {target_pct:.1f}%.")
                else:
                    st.caption(f"{len(_matches)} similar days found")
                    _off_tbl = _compute_offopen_table_from_minutes(
                        tkr_pm, _matches.index, minute_marks, 5, True,
                    )
                    _joined = _matches.join(_off_tbl, how="left") if _off_tbl is not None and not _off_tbl.empty else _matches
                    _joined = _joined.drop(columns=["Delta"], errors="ignore")
                    st.dataframe(_joined, use_container_width=True)

                    _day_opts = [pd.Timestamp(d).date() for d in _matches.index]
                    _day_sel = st.selectbox("View intraday chart", options=_day_opts, key="pm_day_sel")
                    if _day_sel:
                        _show_intraday_chart(tkr_pm, pd.Timestamp(_day_sel))


# ---------------- Scanners ----------------
if nav == 'Scanners':
    st.subheader("Massive Scanners")
    if not _massive_key():
        st.error("MASSIVE_API_KEY not set. Add it to .env.")
    else:
        st.markdown("### Most Active Stocks (by Volume)")
        c1, c2, c3 = st.columns(3)
        with c1:
            active_limit = st.number_input("Rows", value=50, min_value=10, max_value=500, step=10, key="sc_active_rows")
        with c2:
            active_min_vol = st.number_input("Min volume", value=0.0, min_value=0.0, step=100000.0, key="sc_active_min_vol")
        with c3:
            run_active = st.button("Load Most Active", key="run_active")
        if run_active:
            snap_df, errs = _massive_snapshot_all_tickers(limit=1000)
            if errs:
                st.caption(" | ".join(errs))
            if snap_df is None or snap_df.empty:
                st.info("No snapshot data returned.")
            else:
                snap_df["Volume"] = pd.to_numeric(snap_df.get("Volume"), errors="coerce")
                snap_df["% Chg"] = pd.to_numeric(snap_df.get("% Chg"), errors="coerce")
                if float(active_min_vol) > 0:
                    snap_df = snap_df[snap_df["Volume"].fillna(0) >= float(active_min_vol)]
                snap_df = snap_df.sort_values(["Volume", "% Chg"], ascending=[False, False], na_position="last")
                st.dataframe(snap_df.head(int(active_limit)), use_container_width=True)

        st.markdown("### Relative Volume Leaders (today vs 10d avg)")
        r1, r2, r3, r4 = st.columns(4)
        with r1:
            rv_universe = st.number_input("Universe size (by volume)", value=200, min_value=50, max_value=2000, step=50, key="sc_rvol_universe")
        with r2:
            rv_limit = st.number_input("Rows", value=30, min_value=10, max_value=200, step=10, key="sc_rvol_rows")
        with r3:
            rv_min_ratio = st.number_input("Min RVOL", value=2.0, min_value=0.5, max_value=20.0, step=0.5, key="sc_rvol_min")
        with r4:
            run_rvol = st.button("Scan RVOL", key="run_rvol")
        if run_rvol:
            snap_df, errs = _massive_snapshot_all_tickers(limit=int(rv_universe))
            if errs:
                st.caption(" | ".join(errs))
            if snap_df is None or snap_df.empty:
                st.info("No snapshot data returned.")
            else:
                snap_df["Volume"] = pd.to_numeric(snap_df.get("Volume"), errors="coerce")
                snap_df = snap_df.sort_values("Volume", ascending=False, na_position="last")
                tickers = [t for t in snap_df["Ticker"].dropna().head(int(rv_universe)).tolist()]
                rows = []
                end_dt = pd.Timestamp("today").date().isoformat()
                start_dt = (pd.Timestamp("today") - pd.Timedelta(days=20)).date().isoformat()
                prog = st.progress(0.0)
                total = len(tickers)
                for i, tkr in enumerate(tickers, 1):
                    if i % 20 == 0:
                        prog.progress(min(i / total, 1.0))
                    df = _massive_aggs_1d(tkr, start_dt, end_dt)
                    if df is None or df.empty or "Volume" not in df.columns:
                        continue
                    vols = pd.to_numeric(df["Volume"], errors="coerce").dropna()
                    if len(vols) < 5:
                        continue
                    today_vol = vols.iloc[-1]
                    avg10 = vols.tail(10).mean()
                    if avg10 and avg10 > 0:
                        rvol = today_vol / avg10
                        if rvol >= float(rv_min_ratio):
                            rows.append({"Ticker": tkr, "Today Vol": today_vol, "Avg10 Vol": avg10, "RVOL": rvol})
                prog.progress(1.0)
                out = pd.DataFrame(rows)
                if out.empty:
                    st.info("No RVOL leaders found.")
                else:
                    out = out.sort_values("RVOL", ascending=False)
                    st.dataframe(out.head(int(rv_limit)), use_container_width=True)

        st.markdown("### Unusual Options Volume (Volume > OI)")
        o1, o2, o3, o4 = st.columns(4)
        with o1:
            opt_ticker = st.text_input("Ticker", value="NVDA").strip().upper()
        with o2:
            opt_min_vol = st.number_input("Min volume", value=500, min_value=0, step=100, key="sc_opt_min_vol")
        with o3:
            opt_ratio = st.number_input("Min Vol/OI", value=2.0, min_value=1.0, max_value=20.0, step=0.5, key="sc_opt_ratio")
        with o4:
            run_opt = st.button("Scan Options", key="run_opt")
        if run_opt:
            chain = _massive_snapshot_option_chain(opt_ticker)
            if not chain:
                st.info("No option chain returned (check ticker or Massive SDK).")
            else:
                rows = []
                for c in chain:
                    try:
                        vol = getattr(getattr(c, "day", None), "volume", None)
                        oi = getattr(c, "open_interest", None)
                        if vol is None or oi is None or oi == 0:
                            continue
                        if vol >= float(opt_min_vol) and (vol / oi) >= float(opt_ratio):
                            det = getattr(c, "details", None)
                            rows.append({
                                "Contract": getattr(c, "ticker", None),
                                "Vol": vol,
                                "OI": oi,
                                "Vol/OI": round(vol / oi, 2),
                                "Strike": getattr(det, "strike_price", None) if det else None,
                                "Exp": getattr(det, "expiration_date", None) if det else None,
                                "Right": getattr(det, "contract_type", None) if det else None,
                            })
                    except Exception:
                        continue
                out = pd.DataFrame(rows)
                if out.empty:
                    st.info("No unusual contracts found for thresholds.")
                else:
                    out = out.sort_values("Vol/OI", ascending=False)
                    st.dataframe(out, use_container_width=True)

        st.markdown("### News Momentum (last N hours)")
        n1, n2, n3 = st.columns(3)
        with n1:
            news_hours = st.number_input("Hours", value=6, min_value=1, max_value=72, step=1, key="sc_news_hours")
        with n2:
            news_universe = st.number_input("Universe size (by volume)", value=200, min_value=50, max_value=2000, step=50, key="sc_news_universe")
        with n3:
            run_news = st.button("Scan News", key="run_news")
        if run_news:
            snap_df, errs = _massive_snapshot_all_tickers(limit=int(news_universe))
            if errs:
                st.caption(" | ".join(errs))
            if snap_df is None or snap_df.empty:
                st.info("No snapshot data returned.")
            else:
                snap_df["Volume"] = pd.to_numeric(snap_df.get("Volume"), errors="coerce")
                snap_df = snap_df.sort_values("Volume", ascending=False, na_position="last")
                tickers = [t for t in snap_df["Ticker"].dropna().head(int(news_universe)).tolist()]
                rows = []
                prog = st.progress(0.0)
                total = len(tickers)
                for i, tkr in enumerate(tickers, 1):
                    if i % 20 == 0:
                        prog.progress(min(i / total, 1.0))
                    cnt = _massive_news_count(tkr, hours=int(news_hours))
                    if cnt > 0:
                        rows.append({"Ticker": tkr, "News Count": cnt})
                prog.progress(1.0)
                out = pd.DataFrame(rows)
                if out.empty:
                    st.info("No news momentum found.")
                else:
                    out = out.sort_values("News Count", ascending=False)
                    st.dataframe(out, use_container_width=True)

# ---------------- Movers ----------------
if nav == 'Movers':
    st.subheader("Top Movers (Massive/Polygon snapshots)")
    cols = st.columns(4)
    with cols[0]:
        direction = st.selectbox("Direction", ["gainers", "losers"], index=0)
    with cols[1]:
        limit = st.number_input("Rows", value=50, min_value=10, max_value=200, step=10)
    with cols[2]:
        min_price = st.number_input("Min price ($)", value=0.0, min_value=0.0, step=0.5)
    with cols[3]:
        min_vol = st.number_input("Min volume", value=0.0, min_value=0.0, step=50000.0)

    df, errs = _fetch_movers(direction=direction, limit=int(limit))
    if errs:
        st.caption(" | ".join(errs))
    if df is None or df.empty:
        st.info("No movers returned (check API key/entitlement).")
    else:
        df_f = df.copy()
        if min_price > 0 and "Last" in df_f.columns:
            df_f = df_f[df_f["Last"].fillna(0) >= min_price]
        if min_vol > 0 and "Volume" in df_f.columns:
            df_f = df_f[df_f["Volume"].fillna(0) >= min_vol]
        sort_col = "% Chg" if "% Chg" in df_f.columns else "Last"
        df_f = df_f.sort_values(sort_col, ascending=(direction=="losers"))
        st.dataframe(df_f, use_container_width=True)


 


# ═══════════════════════════════════════════════════════════════════════════════
# Calendar (Corporate Events)
# ═══════════════════════════════════════════════════════════════════════════════
if nav == 'Calendar':
    st.subheader("📅 Corporate Events Calendar")

    # ── Pill CSS for calendar controls ────────────────────────────────────
    st.markdown("""<style>
    .st-key-cal_range_sel label:first-child { display: none !important; }
    .st-key-cal_range_sel [data-testid="stRadio"] > div {
        gap: 6px !important; padding: 6px 0 10px !important; flex-wrap: wrap;
    }
    .st-key-cal_range_sel [data-testid="stRadio"] label {
        background: #1e1e2e !important; border: 1px solid #444 !important;
        border-radius: 20px !important; padding: 5px 18px !important;
        font-size: 13px !important; font-weight: 500 !important;
        color: #bbb !important; cursor: pointer; transition: all 0.12s;
    }
    .st-key-cal_range_sel [data-testid="stRadio"] label:hover {
        border-color: #1f77b4 !important; color: #1f77b4 !important;
        background: rgba(31,119,180,0.08) !important;
    }
    .st-key-cal_range_sel [data-testid="stRadio"] label:has(input:checked) {
        background: #1f77b4 !important; color: #fff !important;
        border-color: #1f77b4 !important; font-weight: 600 !important;
    }
    .st-key-cal_range_sel [data-testid="stRadio"] div[data-baseweb="radio"] > div:first-child {
        display: none !important;
    }
    </style>""", unsafe_allow_html=True)

    today = pd.Timestamp.today().normalize()

    # ── Date range pills ──────────────────────────────────────────────────
    _presets = ["This week", "Next week", "Next 2 weeks", "This month", "Next 30 days", "Custom"]
    with st.container(key="cal_range_sel"):
        cal_preset = st.radio("Date range", _presets, index=4,
                              key="cal_preset", horizontal=True)

    if cal_preset == "This week":
        cal_from = (today - pd.Timedelta(days=today.dayofweek)).date()
        cal_to   = (today + pd.Timedelta(days=6 - today.dayofweek)).date()
    elif cal_preset == "Next week":
        nxt = today + pd.Timedelta(days=7 - today.dayofweek)
        cal_from, cal_to = nxt.date(), (nxt + pd.Timedelta(days=6)).date()
    elif cal_preset == "Next 2 weeks":
        cal_from = today.date()
        cal_to   = (today + pd.Timedelta(days=13)).date()
    elif cal_preset == "This month":
        cal_from = today.replace(day=1).date()
        cal_to   = (today + pd.offsets.MonthEnd(0)).date()
    elif cal_preset == "Next 30 days":
        cal_from = today.date()
        cal_to   = (today + pd.Timedelta(days=30)).date()
    else:
        _d1, _d2 = st.columns(2)
        cal_from = _d1.date_input("From", value=today.date(), key="cal_from")
        cal_to   = _d2.date_input("To", value=(today + pd.Timedelta(days=14)).date(), key="cal_to")

    st.markdown(
        f"<div style='display:inline-block;background:#1a2a3a;border:1px solid #2a4a6a;"
        f"border-radius:12px;padding:3px 14px;font-size:12px;color:#8ab4f8;margin-bottom:8px'>"
        f"📅 {cal_from} → {cal_to}</div>".replace("{cal_from}", str(cal_from)).replace("{cal_to}", str(cal_to)),
        unsafe_allow_html=True)

    # ── Watchlist + Event types + Buttons ─────────────────────────────────
    # Load full beta universe — ALL tickers selected by default
    try:
        _cal_beta = _tf_load_beta_universe()
    except Exception:
        _cal_beta = []

    # Filter out index names / header rows from beta universe
    _beta_noise = {"TICKER", "DOW", "SPX", "NASDAQ", "NDX", "RTY", "SOX",
                   "DAX", "GXA", "MESA"}
    _cal_beta_clean = [t for t in _cal_beta if t not in _beta_noise]
    _cal_universe = sorted(set(_cal_beta_clean)) if _cal_beta_clean else sorted({
        "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "TSLA",
        "NFLX", "BABA", "V", "JPM", "GS", "MS", "BAC", "XOM",
        "CVX", "PFE", "JNJ", "WMT", "COST"})

    # If email scan added tickers, merge them into options list
    _email_extra = st.session_state.get("_cal_email_extra", [])
    if _email_extra:
        _cal_universe = sorted(set(_cal_universe) | set(_email_extra))

    # Default to ALL beta universe tickers on first run
    if "cal_wl_picker" not in st.session_state:
        st.session_state["cal_wl_picker"] = _cal_universe[:]

    _wl1, _wl2 = st.columns([3, 1])
    with _wl1:
        # Email scan button row
        _btn1, _btn2, _btn3 = st.columns([1, 1, 2])
        with _btn1:
            _email_scan = st.button("📧 Scan Emails", key="cal_email_scan",
                                    use_container_width=True,
                                    help="Scan Outlook for TMT, earnings, catalyst, analyst day, eco data emails (last 7 days)")
        with _btn2:
            _scan_days = st.selectbox("Lookback", [3, 7, 14, 30], index=1,
                                      key="cal_scan_days", label_visibility="collapsed")
        with _btn3:
            if "cal_scan_status" in st.session_state:
                st.markdown(
                    f"<span style='font-size:12px;color:#8ab4f8'>{st.session_state['cal_scan_status']}</span>",
                    unsafe_allow_html=True)

        if _email_scan:
            with st.spinner("Scanning Outlook for TMT / earnings / catalyst / eco emails..."):
                _scan_result = _scan_calendar_emails(lookback_days=_scan_days)
            _found = _scan_result["tickers"]
            _stats = _scan_result["stats"]
            if _found:
                _cur = list(st.session_state.get("cal_wl_picker", []))
                _new = [t for t in _found if t not in _cur]
                st.session_state["_cal_email_extra"] = _found
                st.session_state["_cal_email_events"] = _scan_result["events"]
                st.session_state["cal_wl_picker"] = sorted(set(_cur) | set(_found))
                st.session_state["cal_scan_status"] = (
                    f"✅ {_stats['emails_matched']} emails matched, "
                    f"{_stats['tickers_found']} tickers found, {len(_new)} new added")
                st.rerun()
            else:
                st.session_state["cal_scan_status"] = (
                    f"⚠️ Scanned {_stats.get('emails_scanned',0)} emails, "
                    f"{_stats.get('emails_matched',0)} matched — no new tickers")
                st.rerun()

        # Show email-sourced events if available
        _email_events = st.session_state.get("_cal_email_events", [])
        if _email_events:
            with st.expander(f"📧 Email events ({len(_email_events)} emails)", expanded=False):
                for ev in _email_events[:20]:
                    _type_emoji = {"earnings": "📅", "catalyst": "🔥",
                                   "eco_data": "📊", "tmt": "💻"}.get(ev["type"], "📧")
                    st.markdown(
                        f"{_type_emoji} **{ev['type'].upper()}** — {ev['subject']}  \n"
                        f"<span style='font-size:11px;color:#6e7681'>"
                        f"{ev['date']} · {', '.join(ev['tickers'][:10])}</span>",
                        unsafe_allow_html=True)

        cal_watchlist_sel = st.multiselect(
            f"Watchlist ({len(_cal_universe)} tickers available)",
            options=_cal_universe,
            placeholder="Search and add/remove tickers...",
            key="cal_wl_picker")
        cal_watchlist_raw = "\n".join(cal_watchlist_sel)
    with _wl2:
        st.markdown("<p style='font-size:13px;font-weight:600;color:#aaa;margin-bottom:4px'>Event Types</p>",
                    unsafe_allow_html=True)
        _ce1, _ce2, _ce3 = st.columns(3)
        _chk_earn = _ce1.checkbox("📅 Earn", value=True, key="cal_chk_earn")
        _chk_div  = _ce2.checkbox("💰 Div",  value=True, key="cal_chk_div")
        _chk_spl  = _ce3.checkbox("✂️ Split", value=True, key="cal_chk_split")
        cal_event_filter = []
        if _chk_earn: cal_event_filter.append("earnings")
        if _chk_div:  cal_event_filter.append("dividend")
        if _chk_spl:  cal_event_filter.append("split")

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        _bc1, _bc2 = st.columns(2)
        cal_fetch = _bc1.button("⚡ Fetch", key="cal_fetch", use_container_width=True, type="primary")
        if _bc2.button("🔄 Clear", key="cal_refresh", use_container_width=True):
            for k in list(st.session_state.keys()):
                if k.startswith("cal_cache_") or k == "cal_tmt_status":
                    del st.session_state[k]
            st.rerun()

    # ── Build ticker list ─────────────────────────────────────────────────
    watchlist_tickers = [
        t.strip().upper() for t in
        cal_watchlist_raw.replace(",", "\n").splitlines()
        if t.strip()
    ]

    # ── Fetch events ──────────────────────────────────────────────────────
    cache_key = f"cal_cache_{cal_from}_{cal_to}_{',' .join(sorted(watchlist_tickers))}_{','.join(sorted(cal_event_filter))}"
    frames = st.session_state.get(cache_key, None)

    if cal_fetch or frames is None:
        from_dt = pd.Timestamp(cal_from)
        to_dt   = pd.Timestamp(cal_to)
        rows = []
        seen = set()

        def _safe_ts(d) -> pd.Timestamp:
            ts = pd.Timestamp(d)
            if ts.tzinfo is not None:
                ts = ts.tz_convert("UTC").tz_localize(None)
            return ts.normalize()

        def _fmt_rev(v):
            if v is None: return None
            try:
                v = float(v)
                if abs(v) >= 1000: return f"${v/1000:.1f}B"
                return f"${v:.0f}M"
            except Exception:
                return None

        prog = st.progress(0, text="Fetching events...")

        # ── Source 1: Bloomberg (direct blpapi — mixed-type) ──────────────
        _bbg_data = {}
        if HAS_XBBG and watchlist_tickers:
            try:
                _bbg_tks = tuple(f"{t} US Equity" for t in watchlist_tickers)
                _bbg_data = _cal_bbg_fetch(_bbg_tks)
            except Exception:
                pass

            # Also try xbbg as fallback if direct blpapi returned nothing
            if not _bbg_data:
                try:
                    from xbbg import blp as _blp_cal
                    _bbg_tks_list = [f"{t} US Equity" for t in watchlist_tickers]
                    _xflds = ['EARN_ANN_DT', 'NEXT_EARN_DT', 'BEST_EPS', 'BEST_SALES',
                              'DVD_EX_DT', 'DVD_PAY_DT', 'DVD_CASH_GROSS', 'ANNOUNCEMENT_DT']
                    _bd = _blp_cal.bdp(_bbg_tks_list, _xflds)
                    if _bd is not None and not _bd.empty:
                        for bbg_tkr, row in _bd.iterrows():
                            _bbg_data[str(bbg_tkr)] = {}
                            for c in _bd.columns:
                                v = row[c]
                                if pd.isna(v) if isinstance(v, (float, int)) else (str(v) in ('nan','NaT','')):
                                    _bbg_data[str(bbg_tkr)][c.upper()] = None
                                else:
                                    _bbg_data[str(bbg_tkr)][c.upper()] = v
                except Exception:
                    pass

            # Process Bloomberg data into event rows
            for tkr in watchlist_tickers:
                bkey = f"{tkr} US Equity"
                bd = _bbg_data.get(bkey, {})
                if not bd:
                    continue
                _timing = bd.get('EARN_ANN_DT_TIME_HIST')
                _eps = bd.get('BEST_EPS')
                _rev = bd.get('BEST_SALES')
                _tgt = bd.get('BEST_TARGET_PRICE')
                _tdiv = bd.get('EQY_DVD_SH_12M')

                # Earnings
                if "earnings" in cal_event_filter:
                    for _ef in ('EARN_ANN_DT', 'NEXT_EARN_DT'):
                        _ed = bd.get(_ef)
                        if _ed is None: continue
                        try:
                            ts = _safe_ts(_ed)
                            if from_dt <= ts <= to_dt:
                                k2 = f"{tkr}|earn|{ts.date()}"
                                if k2 not in seen:
                                    seen.add(k2)
                                    _tstr = str(_timing).upper() if _timing else ""
                                    _tdisp = "BMO" if "BEF" in _tstr or "BMO" in _tstr else (
                                             "AMC" if "AFT" in _tstr or "AMC" in _tstr else "")
                                    rows.append({"ticker": tkr, "date": ts,
                                                 "event_type": "earnings",
                                                 "event_name": f"Earnings {_tdisp} (BBG)" if _tdisp else "Earnings (BBG)",
                                                 "status": "confirmed",
                                                 "timing": _tdisp or None,
                                                 "eps_est": float(_eps) if _eps is not None else None,
                                                 "rev_est": float(_rev) if _rev is not None else None,
                                                 "target_px": float(_tgt) if _tgt is not None else None,
                                                 "div_amount": None, "pay_date": None})
                        except Exception:
                            pass

                # Dividends
                if "dividend" in cal_event_filter:
                    _dd = bd.get('DVD_EX_DT')
                    _da = bd.get('DVD_CASH_GROSS')
                    _dp = bd.get('DVD_PAY_DT')
                    if _dd is not None:
                        try:
                            ts = _safe_ts(_dd)
                            if from_dt <= ts <= to_dt:
                                k2 = f"{tkr}|div|{ts.date()}"
                                if k2 not in seen:
                                    seen.add(k2)
                                    amt_s = f"${float(_da):.3f}" if _da is not None else ""
                                    pay_s = str(pd.Timestamp(_dp).date()) if _dp is not None else None
                                    rows.append({"ticker": tkr, "date": ts,
                                                 "event_type": "dividend",
                                                 "event_name": f"Ex-Div {amt_s} (BBG)" if amt_s else "Ex-Div (BBG)",
                                                 "status": "confirmed",
                                                 "timing": None, "eps_est": None, "rev_est": None,
                                                 "target_px": None,
                                                 "div_amount": float(_da) if _da is not None else None,
                                                 "pay_date": pay_s})
                        except Exception:
                            pass

                # Announcement
                _ann = bd.get('ANNOUNCEMENT_DT')
                if _ann is not None:
                    try:
                        ts = _safe_ts(_ann)
                        if from_dt <= ts <= to_dt:
                            k2 = f"{tkr}|ann|{ts.date()}"
                            if k2 not in seen:
                                seen.add(k2)
                                rows.append({"ticker": tkr, "date": ts,
                                             "event_type": "earnings",
                                             "event_name": "Corp Announcement (BBG)",
                                             "status": "confirmed",
                                             "timing": None, "eps_est": None, "rev_est": None,
                                             "target_px": None, "div_amount": None, "pay_date": None})
                    except Exception:
                        pass

        prog.progress(0.3, text="Fetching from yfinance...")

        # ── Source 2: yfinance (per-ticker fallback) ──────────────────────
        try:
            import yfinance as _yf_cal
            _yf_ok = True
        except ImportError:
            _yf_ok = False

        if _yf_ok:
            for idx_t, tkr in enumerate(watchlist_tickers):
                prog.progress(0.3 + 0.7 * (idx_t + 1) / max(len(watchlist_tickers), 1),
                              text=f"yfinance: {tkr}...")
                try:
                    yftk = _yf_cal.Ticker(tkr)

                    # Earnings
                    if "earnings" in cal_event_filter:
                        _yf_eps = None
                        _yf_rev = None
                        try:
                            cal_dict = yftk.calendar
                            if isinstance(cal_dict, dict):
                                _yf_eps = cal_dict.get("Earnings Average") or cal_dict.get("EPS Estimate")
                                _yf_rev = cal_dict.get("Revenue Average") or cal_dict.get("Revenue Estimate")
                                earn_date = cal_dict.get("Earnings Date")
                                if earn_date is not None:
                                    dates = earn_date if (hasattr(earn_date, '__iter__') and
                                                          not isinstance(earn_date, str)) else [earn_date]
                                    for d in dates:
                                        try:
                                            ts = _safe_ts(d)
                                            if from_dt <= ts <= to_dt:
                                                k2 = f"{tkr}|earn|{ts.date()}"
                                                if k2 not in seen:
                                                    seen.add(k2)
                                                    rows.append({"ticker": tkr, "date": ts,
                                                                 "event_type": "earnings",
                                                                 "event_name": "Earnings",
                                                                 "status": "confirmed",
                                                                 "timing": None,
                                                                 "eps_est": float(_yf_eps) if _yf_eps else None,
                                                                 "rev_est": float(_yf_rev) / 1e6 if _yf_rev else None,
                                                                 "target_px": None,
                                                                 "div_amount": None, "pay_date": None})
                                        except Exception:
                                            pass
                        except Exception:
                            pass
                        try:
                            ed = yftk.earnings_dates
                            if ed is not None and not ed.empty:
                                for d_ts in ed.index:
                                    try:
                                        ts = _safe_ts(d_ts)
                                        if from_dt <= ts <= to_dt:
                                            k2 = f"{tkr}|earn|{ts.date()}"
                                            if k2 not in seen:
                                                seen.add(k2)
                                                rows.append({"ticker": tkr, "date": ts,
                                                             "event_type": "earnings",
                                                             "event_name": "Earnings",
                                                             "status": "confirmed",
                                                             "timing": None, "eps_est": _yf_eps,
                                                             "rev_est": float(_yf_rev)/1e6 if _yf_rev else None,
                                                             "target_px": None,
                                                             "div_amount": None, "pay_date": None})
                                    except Exception:
                                        pass
                        except Exception:
                            pass

                    # Dividends
                    if "dividend" in cal_event_filter:
                        try:
                            divs = yftk.dividends
                            if divs is not None and not divs.empty:
                                for d_ts, val in divs.items():
                                    try:
                                        ts = _safe_ts(d_ts)
                                        if from_dt <= ts <= to_dt:
                                            k2 = f"{tkr}|div|{ts.date()}"
                                            if k2 not in seen:
                                                seen.add(k2)
                                                rows.append({"ticker": tkr, "date": ts,
                                                             "event_type": "dividend",
                                                             "event_name": f"Div ${val:.3f}",
                                                             "status": "confirmed",
                                                             "timing": None, "eps_est": None, "rev_est": None,
                                                             "target_px": None,
                                                             "div_amount": float(val), "pay_date": None})
                                    except Exception:
                                        pass
                        except Exception:
                            pass
                        try:
                            cal_dict = yftk.calendar
                            if isinstance(cal_dict, dict):
                                ex_date = cal_dict.get("Ex-Dividend Date") or cal_dict.get("Dividend Date")
                                div_amt = cal_dict.get("Dividend Rate") or cal_dict.get("dividendRate", 0)
                                if ex_date:
                                    ts = _safe_ts(ex_date)
                                    if from_dt <= ts <= to_dt:
                                        k2 = f"{tkr}|div|{ts.date()}"
                                        if k2 not in seen:
                                            seen.add(k2)
                                            rows.append({"ticker": tkr, "date": ts,
                                                         "event_type": "dividend",
                                                         "event_name": f"Ex-Div ${div_amt:.3f}" if div_amt else "Ex-Div",
                                                         "status": "confirmed",
                                                         "timing": None, "eps_est": None, "rev_est": None,
                                                         "target_px": None,
                                                         "div_amount": float(div_amt) if div_amt else None,
                                                         "pay_date": None})
                        except Exception:
                            pass

                    # Splits
                    if "split" in cal_event_filter:
                        try:
                            splits = yftk.splits
                            if splits is not None and not splits.empty:
                                for d_ts, ratio in splits.items():
                                    try:
                                        ts = _safe_ts(d_ts)
                                        if from_dt <= ts <= to_dt:
                                            k2 = f"{tkr}|split|{ts.date()}"
                                            if k2 not in seen:
                                                seen.add(k2)
                                                rows.append({"ticker": tkr, "date": ts,
                                                             "event_type": "split",
                                                             "event_name": f"Split {ratio:.0f}:1",
                                                             "status": "confirmed",
                                                             "timing": None, "eps_est": None, "rev_est": None,
                                                             "target_px": None,
                                                             "div_amount": None, "pay_date": None})
                                    except Exception:
                                        pass
                        except Exception:
                            pass
                except Exception:
                    pass

        prog.empty()
        frames = rows
        st.session_state[cache_key] = frames

    # ── Render ────────────────────────────────────────────────────────────
    if not frames:
        st.info(f"No events found for **{cal_from}** → **{cal_to}**. "
                "Try clicking **Fetch**, expanding the date range, or adding more tickers.")
    else:
        events = pd.DataFrame(frames)
        events["date"] = pd.to_datetime(events["date"])
        events = events[events["event_type"].isin(cal_event_filter)].sort_values("date")

        TYPE_ICON  = {"earnings": "📅", "dividend": "💰", "split": "✂️"}
        TYPE_COLOR = {"earnings": "#4a9eff", "dividend": "#4caf50", "split": "#ff9800"}

        def _cal_cell_html(r):
            """Build a rich mini-card HTML string for a single event."""
            icon = TYPE_ICON.get(r.get('event_type'), '📌')
            color = TYPE_COLOR.get(r.get('event_type'), '#888')
            tkr = r.get('ticker', '')
            parts = [f"<b>{tkr}</b>"]
            if r.get('event_type') == 'earnings':
                if r.get('timing'): parts.append(r['timing'])
                if r.get('eps_est') is not None:
                    parts.append(f"EPS ${r['eps_est']:.2f}")
                if r.get('rev_est') is not None:
                    rv = r['rev_est']
                    parts.append(f"Rev ${rv/1000:.1f}B" if abs(rv) >= 1000 else f"Rev ${rv:.0f}M")
            elif r.get('event_type') == 'dividend':
                if r.get('div_amount') is not None:
                    parts.append(f"${r['div_amount']:.3f}")
                if r.get('pay_date'):
                    parts.append(f"Pay {r['pay_date']}")
            elif r.get('event_type') == 'split':
                parts.append(r.get('event_name', ''))
            detail = " · ".join(parts)
            return (f"<div style='border-left:3px solid {color};padding:2px 6px;margin:2px 0;"
                    f"font-size:10px;line-height:1.4;color:#ddd'>{icon} {detail}</div>")

        tab_cal, tab_table, tab_ticker = st.tabs([
            "📅 Calendar Grid", "📋 All Events", "🔍 Ticker Lookup"])

        # ── Tab 1: Calendar grid ──────────────────────────────────────────
        with tab_cal:
            events["date_only"] = events["date"].dt.date
            # Group events per day as list of row dicts
            _day_evts = {};
            for _, r in events.iterrows():
                d = r["date_only"]
                if d not in _day_evts: _day_evts[d] = []
                _day_evts[d].append(r.to_dict())

            all_dates = pd.date_range(str(cal_from), str(cal_to), freq="D")
            weeks = []
            week = []
            for d in all_dates:
                if d.dayofweek == 0 and week:
                    weeks.append(week); week = []
                week.append(d)
            if week:
                weeks.append(week)

            DAY_LABELS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            st.markdown(
                " ".join(f"<span style='display:inline-block;width:13%;text-align:center;"
                         f"font-weight:700;color:#aaa'>{d}</span>" for d in DAY_LABELS),
                unsafe_allow_html=True)

            for week in weeks:
                cols = st.columns(7)
                start_dow = week[0].dayofweek
                for i in range(7):
                    col = cols[i]
                    if i < start_dow or (i - start_dow) >= len(week):
                        col.markdown("&nbsp;", unsafe_allow_html=True)
                        continue
                    d = week[i - start_dow]
                    d_date = d.date()
                    is_today = d_date == today.date()
                    evs = _day_evts.get(d_date, [])
                    is_weekend = d.dayofweek >= 5
                    if is_today:
                        bg = "#1a3a5c"
                        border = "2px solid #4a9eff"
                        shadow = "box-shadow:0 0 8px rgba(74,158,255,0.3);"
                    elif evs:
                        bg = "#1a1a2e"
                        border = "1px solid #333"
                        shadow = ""
                    else:
                        bg = "#0d0d15" if is_weekend else "#111"
                        border = "1px solid #1a1a1a"
                        shadow = ""
                    dt_color = '#4a9eff' if is_today else '#888'
                    inner = (f"<div style='font-size:12px;font-weight:700;color:{dt_color};"
                             f"margin-bottom:4px'>{d.strftime('%b %d')}</div>")
                    if len(evs) <= 6:
                        # Show all events if 6 or fewer
                        for ev in evs:
                            inner += _cal_cell_html(ev)
                    else:
                        # Show first 5, then collapsible overflow
                        for ev in evs[:5]:
                            inner += _cal_cell_html(ev)
                        _overflow_id = f"cal_overflow_{d_date}"
                        _overflow_html = "".join(_cal_cell_html(ev) for ev in evs[5:])
                        inner += (
                            f"<details style='margin-top:2px'>"
                            f"<summary style='font-size:9px;color:#8ab4f8;background:#2a2a3a;"
                            f"border-radius:8px;padding:1px 6px;display:inline-block;"
                            f"cursor:pointer;border:none;outline:none'>"
                            f"+{len(evs)-5} more</summary>"
                            f"<div style='margin-top:2px'>{_overflow_html}</div>"
                            f"</details>")
                    col.markdown(
                        f"<div style='background:{bg};border:{border};border-radius:8px;"
                        f"padding:8px 6px;min-height:90px;{shadow}'>{inner}</div>",
                        unsafe_allow_html=True)

        # ── Tab 2: All events table ───────────────────────────────────────
        with tab_table:
            disp = events.copy()
            disp["Date"]    = disp["date"].dt.strftime("%Y-%m-%d")
            disp["Icon"]    = disp["event_type"].map(TYPE_ICON).fillna("📌")
            disp["Type"]    = disp["event_type"].map({
                "earnings": "Earnings", "dividend": "Dividend", "split": "Split"
            }).fillna(disp["event_type"])
            disp["Timing"]  = disp["timing"].fillna("")
            disp["EPS Est"] = disp["eps_est"].apply(
                lambda x: f"${x:.2f}" if pd.notna(x) and x is not None else "")
            disp["Rev Est"] = disp["rev_est"].apply(
                lambda x: (f"${x/1000:.1f}B" if abs(x)>=1000 else f"${x:.0f}M")
                if (pd.notna(x) and x is not None) else "")
            disp["Div Amt"] = disp["div_amount"].apply(
                lambda x: f"${x:.3f}" if pd.notna(x) and x is not None else "")
            disp = disp[["Date","Icon","ticker","Type","event_name","Timing",
                         "EPS Est","Rev Est","Div Amt","status"]].rename(
                columns={"ticker": "Ticker", "event_name": "Event", "status": "Status"})
            st.dataframe(disp, use_container_width=True, hide_index=True)

        # ── Tab 3: Ticker lookup ──────────────────────────────────────────
        with tab_ticker:
            lu_ticker = st.text_input("Ticker", value=st.session_state.get("ticker", "AAPL"),
                                       key="cal_lu_ticker").strip().upper()
            if lu_ticker:
                tkr_events = events[events["ticker"] == lu_ticker].copy()
                if tkr_events.empty:
                    st.info(f"No events found for {lu_ticker} in this date range.")
                else:
                    # Summary card
                    _earn_rows = tkr_events[tkr_events["event_type"]=="earnings"]
                    _div_rows  = tkr_events[tkr_events["event_type"]=="dividend"]
                    _cards = []
                    if not _earn_rows.empty:
                        _er = _earn_rows.iloc[0]
                        _ep = f"EPS Est ${_er['eps_est']:.2f}" if pd.notna(_er.get('eps_est')) else ""
                        _rv = ""
                        if pd.notna(_er.get('rev_est')) and _er['rev_est'] is not None:
                            rv = _er['rev_est']
                            _rv = f"Rev ${rv/1000:.1f}B" if abs(rv)>=1000 else f"Rev ${rv:.0f}M"
                        _tm = _er.get('timing') or ""
                        _cards.append(f"<div style='display:inline-block;background:#1a2a3a;"
                                      f"border:1px solid #2a4a6a;border-radius:10px;padding:8px 16px;"
                                      f"margin:4px 8px 4px 0;font-size:12px;color:#8ab4f8'>"
                                      f"📅 <b>Earnings</b> {_er['date'].strftime('%b %d')} "
                                      f"{_tm} {_ep} {_rv}</div>")
                    if not _div_rows.empty:
                        _dr = _div_rows.iloc[0]
                        _da = f"${_dr['div_amount']:.3f}" if pd.notna(_dr.get('div_amount')) else ""
                        _dp = f"Pay {_dr['pay_date']}" if _dr.get('pay_date') else ""
                        _cards.append(f"<div style='display:inline-block;background:#1a2e1a;"
                                      f"border:1px solid #2a6a2a;border-radius:10px;padding:8px 16px;"
                                      f"margin:4px 8px 4px 0;font-size:12px;color:#81c784'>"
                                      f"💰 <b>Ex-Div</b> {_dr['date'].strftime('%b %d')} "
                                      f"{_da} {_dp}</div>")
                    if _cards:
                        st.markdown("".join(_cards), unsafe_allow_html=True)
                        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

                    tkr_events["Date"]  = tkr_events["date"].dt.strftime("%Y-%m-%d")
                    tkr_events["Type"]  = tkr_events["event_type"].map({
                        "earnings": "📅 Earnings",
                        "dividend": "💰 Dividend",
                        "split": "✂️ Split",
                    }).fillna(tkr_events["event_type"])
                    tkr_events["Timing"]  = tkr_events["timing"].fillna("")
                    tkr_events["EPS Est"] = tkr_events["eps_est"].apply(
                        lambda x: f"${x:.2f}" if pd.notna(x) and x is not None else "")
                    st.dataframe(
                        tkr_events[["Date","Type","event_name","Timing","EPS Est","status"]].rename(
                            columns={"event_name":"Event","status":"Status"}),
                        use_container_width=True, hide_index=True)




# ═══════════════════════════════════════════════════════════════════════════════
# Signal Scanner — find tickers that meet visualizer criteria right now
# ═══════════════════════════════════════════════════════════════════════════════
if nav == 'Signal Scanner':
    st.subheader("Signal Scanner")
    st.markdown(
        "Scans your **entire** parquet universe instantly using a pre-built signals index. "
        "Build the index once (or daily) — then every scan runs in seconds against all tickers."
    )

    # ── Index path ────────────────────────────────────────────────────────────
    _pq_dir_ss = os.environ.get("PER_TICKER_PARQUET_DIR", "")
    _idx_path  = os.path.join(_pq_dir_ss, "_signals_index.parquet") if _pq_dir_ss else ""

    # ── Index status banner ───────────────────────────────────────────────────
    _idx_exists = bool(_idx_path and os.path.exists(_idx_path))
    _idx_age_h  = None
    if _idx_exists:
        import time as _time
        _idx_age_h = (_time.time() - os.path.getmtime(_idx_path)) / 3600.0

    _ib1, _ib2 = st.columns([3, 1])
    with _ib1:
        if not _idx_exists:
            st.warning("⚠️ No signals index found. Build it below to enable full-universe scanning.")
        elif _idx_age_h is not None and _idx_age_h > 24:
            st.warning(f"⚠️ Index is {_idx_age_h:.0f}h old — consider rebuilding for fresh signals.")
        else:
            st.success(f"✅ Index ready ({_idx_age_h:.1f}h old)" if _idx_age_h is not None else "✅ Index ready")
    with _ib2:
        _force_rebuild = st.button("🔨 Build / Rebuild Index", key="ss_build_idx", use_container_width=True)

    # ── Index builder ─────────────────────────────────────────────────────────
    # Chunked build: processes _SS_BCHUNK files per render cycle so that
    # Streamlit's WebSocket heartbeat keeps running — no more "crash to homescreen".
    _SS_BCHUNK = 60   # ≈ 60 files × ~30 ms each ≈ 1-2 s per cycle with 4 workers

    def _process_one(fp):
        # Read one parquet; returns summary dict or None (stale/delisted/bad data).
        try:
            df = pd.read_parquet(fp)
            if df is None or df.empty:
                return None
            date_col = next((c for c in ("Date","Timestamp","date","timestamp") if c in df.columns), None)
            if date_col is None:
                return None
            df = df.sort_values(date_col)
            # Handle both TZ-aware and TZ-naive timestamps robustly
            _ts = pd.to_datetime(df[date_col], errors="coerce")
            if _ts.dt.tz is not None:
                _ts = _ts.dt.tz_convert("UTC").dt.tz_localize(None)
            df[date_col] = _ts
            df = df.set_index(date_col)
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index, errors="coerce")

            # ── Staleness gate: skip delisted / acquired tickers ──────────────
            # If the last bar is > 10 calendar days ago the ticker is no
            # longer actively trading (acquired, delisted, halted, etc.).
            try:
                _days_stale = (pd.Timestamp.today().normalize()
                               - pd.Timestamp(df.index.max()).normalize()).days
                if _days_stale > 10:
                    return None
            except Exception:
                pass

            # Skip files whose name looks like a raw timestamp (not a ticker)
            raw_sym = (str(df["Ticker"].iloc[-1]).upper().replace("_",".")
                       if "Ticker" in df.columns
                       else os.path.basename(fp).replace(".parquet","").upper().replace("_","."))
            if raw_sym.isdigit() or len(raw_sym) > 12:
                return None
            sym = raw_sym

            df = df.tail(220)   # 220 days needed for SMA200 calculation
            close = pd.to_numeric(df.get("Close", pd.Series(dtype=float)), errors="coerce").dropna()
            if len(close) < 30:
                return None

            last_close = float(close.iloc[-1])
            avg_vol = (pd.to_numeric(df["AvgVol_20"].iloc[-1], errors="coerce")
                       if "AvgVol_20" in df.columns
                       else pd.to_numeric(df["Volume"].tail(20), errors="coerce").mean()
                       if "Volume" in df.columns else np.nan)

            # ── RSI (14): prefer pre-computed RSI_14 ─────────────────────────
            if "RSI_14" in df.columns:
                _rsi_s = pd.to_numeric(df["RSI_14"], errors="coerce")
                rsi_last = float(_rsi_s.iloc[-1])
                rsi_prev = float(_rsi_s.iloc[-2]) if len(_rsi_s) >= 2 else np.nan
            else:
                rsi14 = rsi(close, 14)
                rsi_last = float(rsi14.iloc[-1]) if not rsi14.empty else np.nan
                rsi_prev = float(rsi14.iloc[-2]) if len(rsi14) >= 2 else np.nan

            # ── Bollinger (20, 2): prefer pre-computed columns ────────────────
            if "BB_Upper_20" in df.columns and "BB_Lower_20" in df.columns:
                _bb_up = pd.to_numeric(df["BB_Upper_20"], errors="coerce")
                _bb_dn = pd.to_numeric(df["BB_Lower_20"], errors="coerce")
                bb_upper_last = float(_bb_up.iloc[-1])
                bb_lower_last = float(_bb_dn.iloc[-1])
                _bb_rng = bb_upper_last - bb_lower_last
                bb_pct_b_last = ((last_close - bb_lower_last) / _bb_rng) if _bb_rng != 0 else np.nan
            else:
                bb_mid   = close.rolling(20).mean()
                bb_std_s = close.rolling(20).std()
                _bb_up   = bb_mid + 2.0 * bb_std_s
                _bb_dn   = bb_mid - 2.0 * bb_std_s
                bb_upper_last = float(_bb_up.iloc[-1])
                bb_lower_last = float(_bb_dn.iloc[-1])
                _bb_rng_s = (_bb_up - _bb_dn).replace(0, np.nan)
                bb_pct_b_last = float(((close - _bb_dn) / _bb_rng_s).iloc[-1])

            # ── SMA slope (10, 20, 50): normalized % slope ────────────────────
            def _slope(period):
                s = close.rolling(period, min_periods=period).mean()
                sl = s.diff() / close.replace(0, np.nan) * 100.0
                return float(sl.iloc[-1]) if not sl.empty else np.nan

            sma10_slope = _slope(10)
            sma20_slope = _slope(20)
            sma50_slope = _slope(50)

            # ── SMA200 slope ──────────────────────────────────────────────
            sma200_slope = _slope(200)

            # ── Previous-day slopes (for "just turned" detection) ─────────
            def _slope_prev(period):
                s2 = close.rolling(period, min_periods=int(period * 0.5)).mean()
                sl2 = s2.diff() / close.replace(0, np.nan) * 100.0
                return float(sl2.iloc[-2]) if len(sl2.dropna()) >= 2 else np.nan
            sma10_slope_prev = _slope_prev(10)
            sma50_slope_prev = _slope_prev(50)

            # ── SMA levels and price-to-MA distances ─────────────────────
            def _sma_val(period, min_pct=0.6):
                s3 = close.rolling(period, min_periods=int(period * min_pct)).mean()
                v = float(s3.iloc[-1])
                return v if not np.isnan(v) else np.nan
            def _pct_to_ma(ma_val):
                if not ma_val or np.isnan(ma_val) or ma_val <= 0:
                    return np.nan
                return round((last_close / ma_val - 1) * 100, 2)
            sma10_val  = _sma_val(10)
            sma20_val  = _sma_val(20)
            sma50_val  = _sma_val(50, min_pct=0.7)
            sma200_val = _sma_val(200, min_pct=0.6)
            close_to_sma10  = _pct_to_ma(sma10_val)
            close_to_sma20  = _pct_to_ma(sma20_val)
            close_to_sma50  = _pct_to_ma(sma50_val)
            close_to_sma200 = _pct_to_ma(sma200_val)

            # ── ATR%: prefer pre-computed ATR_14 (absolute) / Close × 100 ─────
            if "ATR_14" in df.columns and last_close > 0:
                _atr_abs = pd.to_numeric(df["ATR_14"].iloc[-1], errors="coerce")
                atr_pct_last = float(_atr_abs / last_close * 100.0) if pd.notna(_atr_abs) else np.nan
            else:
                try:
                    atr_pct_last = float(_atr_percent(df, n=14).iloc[-1])
                except Exception:
                    atr_pct_last = np.nan

            # ── Gap%: prefer pre-computed GapPct (fraction → ×100) ────────────
            if "GapPct" in df.columns:
                gap_last = float(pd.to_numeric(df["GapPct"].iloc[-1], errors="coerce") * 100.0)
            elif "Open" in df.columns and len(close) >= 2:
                open_s = pd.to_numeric(df["Open"], errors="coerce")
                gap_s  = (open_s / close.shift(1) - 1.0) * 100.0
                gap_last = float(gap_s.iloc[-1])
            else:
                gap_last = np.nan

            # ── CC%: prefer pre-computed DayPct (fraction → ×100) ─────────────
            if "DayPct" in df.columns:
                _dp = pd.to_numeric(df["DayPct"], errors="coerce")
                cc_last = float(_dp.iloc[-1] * 100.0)
                cc_prev = float(_dp.iloc[-2] * 100.0) if len(_dp) >= 2 else np.nan
            else:
                cc_last = float(((close / close.shift(1)) - 1.0).iloc[-1] * 100.0) if len(close) >= 2 else np.nan
                cc_prev = float(((close / close.shift(1)) - 1.0).iloc[-2] * 100.0) if len(close) >= 3 else np.nan

            last_date = str(df.index[-1].date())

            return {
                "Ticker":       sym,
                "Last Date":    last_date,
                "Close":        round(last_close, 4),
                "Avg Vol":      round(avg_vol, 0) if pd.notna(avg_vol) else np.nan,
                "RSI14":        round(rsi_last, 2) if pd.notna(rsi_last) else np.nan,
                "RSI14_Prev":   round(rsi_prev, 2) if pd.notna(rsi_prev) else np.nan,
                "BB_PctB":      round(bb_pct_b_last, 4) if pd.notna(bb_pct_b_last) else np.nan,
                "BB_Upper":     round(bb_upper_last, 4) if pd.notna(bb_upper_last) else np.nan,
                "BB_Lower":     round(bb_lower_last, 4) if pd.notna(bb_lower_last) else np.nan,
                "SMA10_Slope":  round(sma10_slope, 4) if pd.notna(sma10_slope) else np.nan,
                "SMA20_Slope":  round(sma20_slope, 4) if pd.notna(sma20_slope) else np.nan,
                "SMA50_Slope":  round(sma50_slope, 4) if pd.notna(sma50_slope) else np.nan,
                "ATR_Pct":      round(atr_pct_last, 4) if pd.notna(atr_pct_last) else np.nan,
                "Gap_Pct":      round(gap_last, 4) if pd.notna(gap_last) else np.nan,
                "CC_Pct":       round(cc_last, 4) if pd.notna(cc_last) else np.nan,
                "CC_Pct_Prev":  round(cc_prev, 4) if pd.notna(cc_prev) else np.nan,
                "SMA200_Slope":    round(sma200_slope, 4)     if pd.notna(sma200_slope)    else np.nan,
                "SMA10_Slope_Prev": round(sma10_slope_prev, 4) if pd.notna(sma10_slope_prev) else np.nan,
                "SMA50_Slope_Prev": round(sma50_slope_prev, 4) if pd.notna(sma50_slope_prev) else np.nan,
                "Close_to_SMA10":  close_to_sma10,
                "Close_to_SMA20":  close_to_sma20,
                "Close_to_SMA50":  close_to_sma50,
                "Close_to_SMA200": close_to_sma200,
            }
        except Exception:
            return None

    # ── Kick off: store file list in session state when button is clicked ──────
    if _force_rebuild:
        import glob as _glob
        import time as _time
        if not _pq_dir_ss or not os.path.isdir(_pq_dir_ss):
            st.error("⚠️ PER_TICKER_PARQUET_DIR not set or not a valid directory.")
        else:
            _all_files = sorted(_glob.glob(os.path.join(_pq_dir_ss, "*.parquet")))
            _all_files = [f for f in _all_files if "_signals_index" not in f]
            if not _all_files:
                st.error("⚠️ No .parquet files found in the directory.")
            else:
                # ── mtime pre-filter ──────────────────────────────────────────
                # Skip parquets whose file was last modified > 14 days ago.
                # Delisted / acquired tickers stop receiving new bars so their
                # file mtime goes stale. This stat() call costs microseconds
                # per file — far cheaper than opening 19k parquets.
                _cutoff   = _time.time() - 14 * 86400   # 14 days in seconds
                _raw_n    = len(_all_files)
                _filtered = [f for f in _all_files if os.path.getmtime(f) > _cutoff]
                # Safety: if ALL files are older than cutoff (data not refreshed
                # recently), fall back to the full list to avoid blocking the user.
                if _filtered:
                    _all_files    = _filtered
                    _prefilter_n  = _raw_n - len(_all_files)
                else:
                    _prefilter_n  = 0   # fallback: scan everything
                st.session_state["ss_build_pending"]   = _all_files
                st.session_state["ss_build_rows"]      = []
                st.session_state["ss_build_total"]     = len(_all_files)
                st.session_state["ss_build_raw_total"] = _raw_n
                st.session_state["ss_build_prefilter"] = _prefilter_n
                st.rerun()

    # ── Chunked build loop: one _SS_BCHUNK per render cycle ───────────────────
    # Each cycle completes in ~1-2 s, keeping the WebSocket heartbeat alive.
    if st.session_state.get("ss_build_pending") is not None:
        from concurrent.futures import ThreadPoolExecutor, as_completed as _as_completed

        _pending = st.session_state["ss_build_pending"]
        _n_total = st.session_state["ss_build_total"]
        _done_n  = _n_total - len(_pending)

        _build_prog = st.progress(
            _done_n / max(_n_total, 1),
            text=f"Building index… {_done_n:,}/{_n_total:,} files scanned"
        )
        _build_status = st.empty()

        # Process the next chunk using 4 workers
        _chunk = _pending[:_SS_BCHUNK]
        with ThreadPoolExecutor(max_workers=4) as _pool:
            _futs = {_pool.submit(_process_one, fp): fp for fp in _chunk}
            for _fut in _as_completed(_futs):
                try:
                    _res = _fut.result()
                except Exception:
                    _res = None
                if _res:
                    st.session_state["ss_build_rows"].append(_res)

        st.session_state["ss_build_pending"] = _pending[_SS_BCHUNK:]

        if not st.session_state["ss_build_pending"]:
            # ── Build complete ─────────────────────────────────────────────────
            _build_prog.empty()
            _rows        = st.session_state["ss_build_rows"]
            _n_total     = st.session_state["ss_build_total"]
            _raw_total   = st.session_state.get("ss_build_raw_total", _n_total)
            _prefilter_n = st.session_state.get("ss_build_prefilter", 0)
            st.session_state["ss_build_pending"]   = None
            st.session_state["ss_build_rows"]      = []
            st.session_state["ss_build_total"]     = 0
            st.session_state["ss_build_raw_total"] = 0
            st.session_state["ss_build_prefilter"] = 0
            if _rows:
                _idx_df  = pd.DataFrame(_rows)
                _idx_df.to_parquet(_idx_path, index=False)
                _active       = len(_rows)
                _stale_gated  = _n_total - _active   # passed mtime but failed date check
                _msg = (f"✅ Index built — {_active:,} active tickers  "
                        f"| {_prefilter_n:,} skipped by mtime  "
                        f"| {_stale_gated:,} skipped by last-bar date  "
                        f"| {_raw_total:,} total files scanned")
                _build_status.success(_msg)
            else:
                _build_status.error(
                    "⚠️ No active tickers found — all may be stale or the directory is empty."
                )
            st.rerun()   # refresh the age banner so it shows "0.0h old"
        else:
            st.rerun()   # trigger next chunk

    # ── Scan controls ─────────────────────────────────────────────────────────
    st.markdown("---")
    _uf1, _uf2 = st.columns(2)
    with _uf1:
        ss_min_price = st.number_input("Min price ($)", value=5.0, min_value=0.0, step=1.0, key="ss_min_price")
    with _uf2:
        ss_min_vol   = st.number_input("Min avg vol (20d)", value=200000.0, min_value=0.0, step=50000.0, key="ss_min_vol")

    # ── Strategy picker ───────────────────────────────────────────────────────
    strategy = st.selectbox("Strategy to scan for", [
        "Bollinger + RSI  —  price outside band with RSI confirmation",
        "RSI Threshold    —  RSI crosses / sits above or below a level",
        "SMA Slope        —  SMA turns up or down (momentum flip)",
        "Overnight Setup  —  ATR expansion signals overnight edge",
        "Gap Open         —  opens with a gap above/below threshold",
        "Prev Day CC      —  previous day close-to-close move met threshold",
    ], key="ss_strategy")

    strat = strategy.split("—")[0].strip()

    # ── Per-strategy parameters (pre-filled from Visualizer session state) ────
    with st.expander("Strategy parameters", expanded=True):
        if "Bollinger" in strat:
            p1, p2, p3, p4 = st.columns(4)
            with p1:
                ss_bb_period = st.number_input("BB period", value=int(st.session_state.get("bb_rsi_period", 20)),
                                               min_value=5, max_value=100, key="ss_bb_period")
                ss_bb_std    = st.number_input("BB std devs", value=float(st.session_state.get("bb_rsi_std", 2.0)),
                                               min_value=0.5, max_value=4.0, step=0.5, key="ss_bb_std")
            with p2:
                ss_rsi_len   = st.number_input("RSI period", value=int(st.session_state.get("bb_rsi_len", 14)),
                                               min_value=2, max_value=100, key="ss_rsi_len")
            with p3:
                ss_bb_upper  = st.checkbox("Scan upper band (overbought)", value=True, key="ss_bb_upper")
                ss_rsi_high  = st.number_input("RSI >=", value=float(st.session_state.get("bb_rsi_high", 60.0)),
                                               min_value=0.0, max_value=100.0, key="ss_rsi_high")
                ss_pct_upper = st.number_input("BB %B >=", value=float(st.session_state.get("bb_pct_upper", 1.0)),
                                               min_value=0.0, max_value=3.0, step=0.05, key="ss_pct_upper")
            with p4:
                ss_bb_lower  = st.checkbox("Scan lower band (oversold)", value=True, key="ss_bb_lower")
                ss_rsi_low   = st.number_input("RSI <=", value=float(st.session_state.get("bb_rsi_low", 40.0)),
                                               min_value=0.0, max_value=100.0, key="ss_rsi_low")
                ss_pct_lower = st.number_input("BB %B <=", value=float(st.session_state.get("bb_pct_lower", 0.0)),
                                               min_value=-2.0, max_value=0.5, step=0.05, key="ss_pct_lower")

        elif "RSI Threshold" in strat:
            q1, q2, q3 = st.columns(3)
            with q1:
                ss_rsi_len2  = st.number_input("RSI period", value=int(st.session_state.get("rsi_bt_len", 14)),
                                               min_value=2, max_value=100, key="ss_rsi_len2")
            with q2:
                ss_rsi_mode  = st.selectbox("Signal direction", [
                    "Oversold — RSI crosses UP through threshold (buy dip)",
                    "Overbought — RSI crosses DOWN through threshold (fade rally)",
                    "Above threshold (RSI >= value)",
                    "Below threshold (RSI <= value)",
                ], key="ss_rsi_mode")
            with q3:
                ss_rsi_thr   = st.number_input("RSI threshold", value=int(st.session_state.get("rsi_bt_thr", 30)),
                                               min_value=1, max_value=99, key="ss_rsi_thr")

        elif "SMA Slope" in strat:
            s1, s2, s3 = st.columns(3)
            with s1:
                ss_sma_period   = st.number_input("SMA period", value=int(st.session_state.get("sma_period", 10)),
                                                  min_value=3, max_value=200, key="ss_sma_period")
                ss_sma_normalize = st.checkbox("Normalize slope", value=bool(st.session_state.get("sma_normalize", True)), key="ss_sma_norm")
            with s2:
                ss_sma_up   = st.number_input("Upslope trigger >=", value=float(st.session_state.get("sma_up", 0.05)),
                                              step=0.01, format="%.3f", key="ss_sma_up")
            with s3:
                ss_sma_dn   = st.number_input("Downslope trigger <=", value=float(st.session_state.get("sma_dn", -0.05)),
                                              step=0.01, format="%.3f", key="ss_sma_dn")
            ss_sma_direction = st.radio("Scan for", ["Upslope crossing", "Downslope crossing", "Either"],
                                        horizontal=True, key="ss_sma_dir")

        elif "Overnight" in strat:
            o1, o2, o3 = st.columns(3)
            with o1:
                ss_ov_strategy = st.selectbox("Overnight strategy", ["ATR change", "Gap", "Close-to-Close"],
                                              key="ss_ov_strat")
            with o2:
                ss_ov_thr = st.number_input("Threshold", value=float(st.session_state.get("ov_threshold", 0.5)),
                                            step=0.1, key="ss_ov_thr")
            with o3:
                ss_ov_atr_n = st.number_input("ATR period", value=int(st.session_state.get("ov_atr_n", 14)),
                                              min_value=5, max_value=60, key="ss_ov_atr_n")

        elif "Gap Open" in strat:
            g1, g2 = st.columns(2)
            with g1:
                ss_gap_thr = st.number_input("Gap threshold % (signed)",
                                             value=3.0, step=0.5, key="ss_gap_thr",
                                             help="Positive=gap up, negative=gap down, 0=either direction")
            with g2:
                ss_gap_today_only = st.checkbox("Today's gap only (last row)", value=True, key="ss_gap_today",
                                                help="Uncheck to also catch gaps from the last N signal-window days")

        elif "Prev Day CC" in strat:
            cc1, cc2 = st.columns(2)
            with cc1:
                ss_cc_thr = st.number_input("Prev C-C threshold % (signed)",
                                            value=2.0, step=0.5, key="ss_cc_thr",
                                            help="Positive=require up day, negative=require down day")
            with cc2:
                ss_cc_today_only = st.checkbox("Today only (last row)", value=True, key="ss_cc_today")

    run_scan = st.button("⚡ Scan Full Universe", type="primary", use_container_width=True, key="ss_run",
                         disabled=not _idx_exists)

    if not _idx_exists:
        st.info("👆 Build the index first — it takes 2-3 minutes once, then every scan is instant.")

    if run_scan and _idx_exists:
        # Clear any previous results and row selection so fresh scan starts clean
        for _k in ("ss_hits","ss_total_universe","ss_n_ok","ss_idx_age_h","ss_strat"):
            st.session_state.pop(_k, None)

        # ── Load index ────────────────────────────────────────────────────────
        _idx = pd.read_parquet(_idx_path)

        # ── Apply liquidity filters ───────────────────────────────────────────
        _idx = _idx[pd.to_numeric(_idx["Close"], errors="coerce") >= float(ss_min_price)]
        if float(ss_min_vol) > 0:
            _idx = _idx[pd.to_numeric(_idx["Avg Vol"], errors="coerce").fillna(0) >= float(ss_min_vol)]

        total_universe = len(pd.read_parquet(_idx_path))
        n_ok   = len(_idx)
        n_skip = total_universe - n_ok

        # ── Apply strategy filter ─────────────────────────────────────────────
        hits = []

        if "Bollinger" in strat:
            _bb_p   = int(ss_bb_period)
            _bb_std = float(ss_bb_std)
            _upper  = pd.to_numeric(_idx["BB_Upper"], errors="coerce")
            _lower  = pd.to_numeric(_idx["BB_Lower"], errors="coerce")
            _close  = pd.to_numeric(_idx["Close"],    errors="coerce")
            _rsi    = pd.to_numeric(_idx["RSI14"],    errors="coerce")
            _pctb   = pd.to_numeric(_idx["BB_PctB"],  errors="coerce")
            _mask   = pd.Series(False, index=_idx.index)
            _side   = pd.Series("", index=_idx.index)
            if ss_bb_upper:
                _up = (_close >= _upper) & (_rsi >= float(ss_rsi_high)) & (_pctb >= float(ss_pct_upper))
                _mask |= _up
                _side[_up] = "UPPER"
            if ss_bb_lower:
                _dn = (_close <= _lower) & (_rsi <= float(ss_rsi_low)) & (_pctb <= float(ss_pct_lower))
                _mask |= _dn
                _side[_dn] = "LOWER"
            _hits = _idx[_mask].copy()
            _hits["Signal"] = _side[_mask].values

        elif "RSI Threshold" in strat:
            _rsi  = pd.to_numeric(_idx["RSI14"],      errors="coerce")
            _rsi_p= pd.to_numeric(_idx["RSI14_Prev"], errors="coerce")
            _thr  = float(ss_rsi_thr)
            if "crosses UP"   in ss_rsi_mode:  _mask = (_rsi_p < _thr) & (_rsi >= _thr)
            elif "crosses DOWN" in ss_rsi_mode: _mask = (_rsi_p > _thr) & (_rsi <= _thr)
            elif "Above"       in ss_rsi_mode:  _mask = _rsi >= _thr
            else:                               _mask = _rsi <= _thr
            _hits = _idx[_mask.fillna(False)].copy()
            _hits["Signal"] = _rsi[_mask.fillna(False)].map(lambda v: f"RSI={v:.1f}" if pd.notna(v) else "").values

        elif "SMA Slope" in strat:
            _col_map = {10: "SMA10_Slope", 20: "SMA20_Slope", 50: "SMA50_Slope"}
            _scol    = _col_map.get(int(ss_sma_period), "SMA20_Slope")
            _slope   = pd.to_numeric(_idx.get(_scol, _idx.get("SMA20_Slope")), errors="coerce")
            if ss_sma_direction == "Upslope crossing":   _mask = _slope >= float(ss_sma_up)
            elif ss_sma_direction == "Downslope crossing": _mask = _slope <= float(ss_sma_dn)
            else: _mask = (_slope >= float(ss_sma_up)) | (_slope <= float(ss_sma_dn))
            _hits = _idx[_mask.fillna(False)].copy()
            _dir  = _slope[_mask.fillna(False)].map(lambda v: f"Slope={v:+.3f}%" if pd.notna(v) else "")
            _hits["Signal"] = _dir.values

        elif "Overnight" in strat:
            if ss_ov_strategy == "ATR change":
                _val  = pd.to_numeric(_idx["ATR_Pct"], errors="coerce")
                _mask = _val.abs() >= float(ss_ov_thr)
            elif ss_ov_strategy == "Gap":
                _val  = pd.to_numeric(_idx["Gap_Pct"], errors="coerce")
                _mask = _val.abs() >= float(ss_ov_thr)
            else:
                _val  = pd.to_numeric(_idx["CC_Pct"], errors="coerce")
                _mask = _val.abs() >= float(ss_ov_thr)
            _hits = _idx[_mask.fillna(False)].copy()
            _hits["Signal"] = _val[_mask.fillna(False)].map(lambda v: f"{v:+.2f}%" if pd.notna(v) else "").values

        elif "Gap Open" in strat:
            _gap  = pd.to_numeric(_idx["Gap_Pct"], errors="coerce")
            _gthr = float(ss_gap_thr)
            if _gthr > 0:   _mask = _gap >= _gthr
            elif _gthr < 0: _mask = _gap <= _gthr
            else:            _mask = _gap.abs() >= 0.5
            _hits = _idx[_mask.fillna(False)].copy()
            _hits["Signal"] = _gap[_mask.fillna(False)].map(lambda v: f"Gap={v:+.2f}%" if pd.notna(v) else "").values

        elif "Prev Day CC" in strat:
            _cc   = pd.to_numeric(_idx["CC_Pct"], errors="coerce")
            _ccthr = float(ss_cc_thr)
            _mask = (_cc >= _ccthr) if _ccthr >= 0 else (_cc <= _ccthr)
            _hits = _idx[_mask.fillna(False)].copy()
            _hits["Signal"] = _cc[_mask.fillna(False)].map(lambda v: f"CC={v:+.2f}%" if pd.notna(v) else "").values

        else:
            _hits = _idx.iloc[0:0].copy()
            _hits["Signal"] = []

        # ── Format results ────────────────────────────────────────────────────
        if not _hits.empty:
            _close_num = pd.to_numeric(_hits["Close"],    errors="coerce")
            _upper_num = pd.to_numeric(_hits["BB_Upper"], errors="coerce")
            _lower_num = pd.to_numeric(_hits["BB_Lower"], errors="coerce")
            _sig_col   = _hits.get("Signal", pd.Series("", index=_hits.index))

            # BB Distance: positive = price above band, negative = price below band
            # For UPPER signals: close - upper band. For LOWER signals: close - lower band.
            def _bb_dist(row):
                sig   = str(row.get("Signal", ""))
                close = row.get("_close")
                upper = row.get("_upper")
                lower = row.get("_lower")
                if pd.isna(close):
                    return np.nan, np.nan
                if "UPPER" in sig or ("BB" in strat and "UPPER" not in sig and "LOWER" not in sig):
                    band = upper
                    dist = close - band if pd.notna(band) else np.nan
                else:
                    band = lower
                    dist = close - band if pd.notna(band) else np.nan  # negative when below
                pct  = dist / band * 100.0 if (pd.notna(dist) and pd.notna(band) and band != 0) else np.nan
                return dist, pct

            _tmp = _hits.copy()
            _tmp["_close"] = _close_num.values
            _tmp["_upper"] = _upper_num.values
            _tmp["_lower"] = _lower_num.values
            _bb_dists = _tmp.apply(_bb_dist, axis=1)

            _out = pd.DataFrame()
            _out["Ticker"]      = _hits["Ticker"].values
            _out["Signal"]      = _hits["Signal"].values
            _out["Last Date"]   = _hits["Last Date"].values
            _out["Close"]       = _close_num.map(lambda v: f"${v:.2f}" if pd.notna(v) else "—").values
            _out["BB Dist$"]    = [f"{d[0]:+.2f}" if pd.notna(d[0]) else "—" for d in _bb_dists]
            _out["BB Dist%"]    = [f"{d[1]:+.2f}%" if pd.notna(d[1]) else "—" for d in _bb_dists]
            _out["BB %B"]       = pd.to_numeric(_hits["BB_PctB"], errors="coerce").map(
                lambda v: f"{v:.2f}" if pd.notna(v) else "—").values
            _out["RSI14"]       = pd.to_numeric(_hits["RSI14"], errors="coerce").map(
                lambda v: f"{v:.1f}" if pd.notna(v) else "—").values
            _out["ATR%"]        = pd.to_numeric(_hits["ATR_Pct"], errors="coerce").map(
                lambda v: f"{v:.2f}%" if pd.notna(v) else "—").values
            _out["Day Chg%"]    = pd.to_numeric(_hits["CC_Pct"], errors="coerce").map(
                lambda v: f"{v:+.2f}%" if pd.notna(v) else "—").values
            _out["Gap%"]        = pd.to_numeric(_hits["Gap_Pct"], errors="coerce").map(
                lambda v: f"{v:+.2f}%" if pd.notna(v) else "—").values
            _out["Avg Vol"]     = pd.to_numeric(_hits["Avg Vol"], errors="coerce").map(
                lambda v: f"{v/1e6:.1f}M" if pd.notna(v) and v >= 1e6
                else f"{v/1e3:.0f}K" if pd.notna(v) and v >= 1e3 else "—").values
            # keep raw numeric columns for the detail panel (not shown in table)
            _out["_Close"]      = _close_num.values
            _out["_BB_Upper"]   = _upper_num.values
            _out["_BB_Lower"]   = _lower_num.values
            _out["_RSI14"]      = pd.to_numeric(_hits["RSI14"],   errors="coerce").values
            _out["_ATR_Pct"]    = pd.to_numeric(_hits["ATR_Pct"], errors="coerce").values
            _out["_CC_Pct"]     = pd.to_numeric(_hits["CC_Pct"],  errors="coerce").values
            hits = _out.to_dict("records")

        # ── Save results to session state so row-click reruns don't wipe them ─
        st.session_state["ss_hits"]           = hits
        st.session_state["ss_total_universe"] = total_universe
        st.session_state["ss_n_ok"]           = n_ok
        st.session_state["ss_idx_age_h"]      = _idx_age_h
        st.session_state["ss_strat"]          = strat

    # ── Read results from session state (survives reruns from row clicks) ─────
    hits           = st.session_state.get("ss_hits", [])
    total_universe = st.session_state.get("ss_total_universe", 0)
    n_ok           = st.session_state.get("ss_n_ok", 0)
    _idx_age_h_disp = st.session_state.get("ss_idx_age_h", _idx_age_h)

    if not hits and not run_scan:
        st.info("Configure your strategy above and click **⚡ Scan Full Universe** to see results.")
    elif hits or run_scan:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Universe",      f"{total_universe:,}")
        m2.metric("After filters", f"{n_ok:,}")
        m3.metric("Signals found", f"{len(hits):,}")
        m4.metric("Index age",     f"{_idx_age_h_disp:.1f}h" if _idx_age_h_disp is not None else "—")

        # columns visible in the table (hide raw _ columns)
        _TABLE_COLS = ["Ticker","Signal","Last Date","Close","BB Dist$","BB Dist%",
                       "BB %B","RSI14","ATR%","Day Chg%","Gap%","Avg Vol"]

        if not hits:
            st.info("No tickers matched. Try lowering min price/vol filters or relaxing strategy thresholds.")
        else:
            _hits_df    = pd.DataFrame(hits)
            today_str   = str(pd.Timestamp.now().date())
            hits_today  = _hits_df[_hits_df["Last Date"] == today_str]
            hits_recent = _hits_df[_hits_df["Last Date"] != today_str]

            # ── Post-signal stats helper ──────────────────────────────────────
            def _post_signal_stats(tkr: str, signal_label: str):
                """Load parquet for tkr, find all historical instances of same signal,
                compute forward returns at 1/3/5/10/20 day horizons."""
                if not _pq_dir_ss:
                    return None, None
                import glob as _g2
                # find file
                candidates = (
                    _g2.glob(os.path.join(_pq_dir_ss, f"{tkr}.parquet")) +
                    _g2.glob(os.path.join(_pq_dir_ss, f"{tkr.replace('.','_')}.parquet"))
                )
                if not candidates:
                    return None, None
                try:
                    _df = pd.read_parquet(candidates[0])
                    date_col = next((c for c in ("Date","Timestamp","date","timestamp") if c in _df.columns), None)
                    if date_col is None:
                        return None, None
                    _df = _df.sort_values(date_col)
                    _df[date_col] = pd.to_datetime(_df[date_col]).dt.tz_localize(None)
                    _df = _df.set_index(date_col)
                    _close = pd.to_numeric(_df.get("Close"), errors="coerce").dropna()
                    if len(_close) < 40:
                        return None, None

                    # Recompute signal on full history
                    _bb_mid   = _close.rolling(20).mean()
                    _bb_std_s = _close.rolling(20).std()
                    _bb_upper = _bb_mid + 2.0 * _bb_std_s
                    _bb_lower = _bb_mid - 2.0 * _bb_std_s
                    _bb_range = (_bb_upper - _bb_lower).replace(0, np.nan)
                    _pctb     = (_close - _bb_lower) / _bb_range
                    _rsi14    = rsi(_close, 14)
                    _cc       = (_close / _close.shift(1) - 1.0) * 100.0
                    _gap      = np.nan
                    if "Open" in _df.columns:
                        _open_s = pd.to_numeric(_df["Open"], errors="coerce")
                        _gap    = (_open_s / _close.shift(1) - 1.0) * 100.0

                    # Reconstruct same mask as query
                    if "Bollinger" in strat:
                        if "UPPER" in signal_label:
                            _sig = (_close >= _bb_upper) & (_rsi14 >= float(ss_rsi_high)) & (_pctb >= float(ss_pct_upper))
                        else:
                            _sig = (_close <= _bb_lower) & (_rsi14 <= float(ss_rsi_low)) & (_pctb <= float(ss_pct_lower))
                    elif "RSI Threshold" in strat:
                        _thr = float(ss_rsi_thr)
                        if   "crosses UP"   in ss_rsi_mode: _sig = (_rsi14.shift(1) < _thr) & (_rsi14 >= _thr)
                        elif "crosses DOWN" in ss_rsi_mode: _sig = (_rsi14.shift(1) > _thr) & (_rsi14 <= _thr)
                        elif "Above"        in ss_rsi_mode: _sig = _rsi14 >= _thr
                        else:                               _sig = _rsi14 <= _thr
                    elif "SMA Slope" in strat:
                        _col_map2 = {10: 10, 20: 20, 50: 50}
                        _p2 = _col_map2.get(int(ss_sma_period), 20)
                        _sma2 = _close.rolling(_p2).mean()
                        _sl2  = _sma2.diff() / _close.replace(0, np.nan) * 100.0
                        if   ss_sma_direction == "Upslope crossing":   _sig = _sl2 >= float(ss_sma_up)
                        elif ss_sma_direction == "Downslope crossing":  _sig = _sl2 <= float(ss_sma_dn)
                        else: _sig = (_sl2 >= float(ss_sma_up)) | (_sl2 <= float(ss_sma_dn))
                    elif "Prev Day CC" in strat:
                        _ccthr2 = float(ss_cc_thr)
                        _sig = (_cc >= _ccthr2) if _ccthr2 >= 0 else (_cc <= _ccthr2)
                    elif "Gap Open" in strat:
                        if not isinstance(_gap, pd.Series):
                            return None, None
                        _gthr2 = float(ss_gap_thr)
                        if _gthr2 > 0:   _sig = _gap >= _gthr2
                        elif _gthr2 < 0: _sig = _gap <= _gthr2
                        else:            _sig = _gap.abs() >= 0.5
                    else:
                        return None, None

                    _sig = _sig.fillna(False)
                    _signal_dates = _close.index[_sig]
                    if len(_signal_dates) == 0:
                        return None, None

                    # Forward returns
                    horizons = [1, 3, 5, 10, 20]
                    fwd_rows = []
                    for sd in _signal_dates:
                        loc = _close.index.get_loc(sd)
                        row_d = {"Signal Date": str(sd.date()), "Close": round(float(_close.iloc[loc]), 2)}
                        for h in horizons:
                            if loc + h < len(_close):
                                ret = (_close.iloc[loc + h] / _close.iloc[loc] - 1.0) * 100.0
                                row_d[f"+{h}d"] = round(ret, 2)
                            else:
                                row_d[f"+{h}d"] = np.nan
                        fwd_rows.append(row_d)

                    fwd_df = pd.DataFrame(fwd_rows)

                    # Summary stats
                    stat_rows = []
                    for h in horizons:
                        col = f"+{h}d"
                        vals = pd.to_numeric(fwd_df[col], errors="coerce").dropna()
                        if vals.empty:
                            continue
                        stat_rows.append({
                            "Horizon":   col,
                            "Avg Ret%":  f"{vals.mean():+.2f}%",
                            "Win Rate":  f"{(vals > 0).mean()*100:.0f}%",
                            "Median%":   f"{vals.median():+.2f}%",
                            "Best%":     f"{vals.max():+.2f}%",
                            "Worst%":    f"{vals.min():+.2f}%",
                            "Samples":   len(vals),
                        })
                    stats_df = pd.DataFrame(stat_rows)
                    return fwd_df, stats_df

                except Exception:
                    return None, None

            # ── Display helper — row-click selection ──────────────────────────
            def _show_hits(df_hits, label, table_key):
                if df_hits.empty:
                    return None
                if label:
                    st.markdown(f"### {label}")
                display_df = df_hits[_TABLE_COLS].reset_index(drop=True)
                sel = st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True,
                    on_select="rerun",
                    selection_mode="single-row",
                    key=table_key,
                )
                # return the selected full row (with raw _ cols) if any
                rows = sel.selection.get("rows", []) if sel and hasattr(sel, "selection") else []
                if rows:
                    return df_hits.iloc[rows[0]]
                return None

            _selected_row = None
            if not hits_today.empty:
                _r = _show_hits(hits_today, "🟢 Today's signals", "ss_tbl_today")
                if _r is not None:
                    _selected_row = _r
            if not hits_recent.empty:
                _ss_show_recent = st.toggle(
                    f"🔵 Show recent signals ({len(hits_recent)})",
                    value=False, key="ss_show_recent_toggle"
                )
                if _ss_show_recent:
                    _r = _show_hits(hits_recent, "", "ss_tbl_recent")
                    if _r is not None:
                        _selected_row = _r

            # ── Detail panel (renders immediately below the table on row click) ──
            if _selected_row is not None:
                _sel_tkr = str(_selected_row.get("Ticker", ""))
                st.markdown("---")
                st.markdown(f"## 🔍 {_sel_tkr} — Signal Detail")

                _dc1, _dc2, _dc3, _dc4, _dc5 = st.columns(5)
                _dc1.metric("Close",    _selected_row.get("Close",    "—"))
                _dc2.metric("Signal",   _selected_row.get("Signal",   "—"))
                _dc3.metric("BB Dist%", _selected_row.get("BB Dist%", "—"))
                _dc4.metric("RSI14",    _selected_row.get("RSI14",    "—"))
                _dc5.metric("Day Chg%", _selected_row.get("Day Chg%", "—"))

                _chart_tab, _stats_tab = st.tabs(["📈 Chart", "📊 Post-Signal History"])

                with _chart_tab:
                    try:
                        import plotly.graph_objects as _ss_go
                        _ss_cdf = _load_daily_df(_sel_tkr)
                        if _ss_cdf is None or len(_ss_cdf) < 10:
                            st.warning(f"No daily data for {_sel_tkr}")
                        else:
                            _ss_cdf = _ss_cdf.sort_values("Date").tail(90).reset_index(drop=True)
                            _ss_close = pd.to_numeric(_ss_cdf["Close"], errors="coerce")
                            _ss_s10 = _ss_close.rolling(10, min_periods=5).mean()
                            _ss_s20 = _ss_close.rolling(20, min_periods=10).mean()
                            _ss_fig = _ss_go.Figure()
                            _ss_fig.add_trace(_ss_go.Candlestick(
                                x=_ss_cdf["Date"],
                                open=pd.to_numeric(_ss_cdf.get("Open", _ss_close), errors="coerce"),
                                high=pd.to_numeric(_ss_cdf.get("High", _ss_close), errors="coerce"),
                                low=pd.to_numeric(_ss_cdf.get("Low", _ss_close), errors="coerce"),
                                close=_ss_close,
                                name="Price",
                                increasing_line_color="#26a69a",
                                decreasing_line_color="#ef5350",
                            ))
                            _ss_fig.add_trace(_ss_go.Scatter(
                                x=_ss_cdf["Date"], y=_ss_s10, name="SMA10",
                                line=dict(color="#f5a623", width=1),
                            ))
                            _ss_fig.add_trace(_ss_go.Scatter(
                                x=_ss_cdf["Date"], y=_ss_s20, name="SMA20",
                                line=dict(color="#7b68ee", width=1),
                            ))
                            _ss_fig.update_layout(
                                title=f"{_sel_tkr} — 90-Day",
                                xaxis_rangeslider_visible=False,
                                height=420,
                                margin=dict(l=40, r=20, t=40, b=20),
                                legend=dict(orientation="h"),
                            )
                            st.plotly_chart(_ss_fig, use_container_width=True)
                    except Exception as _ce:
                        st.warning(f"Chart error: {type(_ce).__name__}: {_ce}")

                with _stats_tab:
                    with st.spinner(f"Computing post-signal returns for {_sel_tkr}…"):
                        _fwd_df, _stats_df = _post_signal_stats(
                            _sel_tkr, str(_selected_row.get("Signal", ""))
                        )
                    if _stats_df is None or _stats_df.empty:
                        st.info("Not enough history — need 40+ days of data with multiple signal occurrences.")
                    else:
                        st.markdown("#### Forward Return Summary")
                        st.dataframe(_stats_df, use_container_width=True, hide_index=True)

                        if _fwd_df is not None and not _fwd_df.empty:
                            st.markdown("#### All Historical Instances")
                            _ret_cols = [c for c in _fwd_df.columns if c.startswith("+")]
                            def _color_ret(val):
                                try:
                                    v = float(val)
                                    if v > 0: return "background-color:#1a3a2a;color:#26a69a"
                                    if v < 0: return "background-color:#3a1a1a;color:#ef5350"
                                except Exception:
                                    pass
                                return ""
                            st.dataframe(
                                _fwd_df.style.map(_color_ret, subset=_ret_cols),
                                use_container_width=True, hide_index=True
                            )
                            try:
                                import plotly.graph_objects as _pgo2
                                _wr_data = []
                                for _h in [1, 3, 5, 10, 20]:
                                    _col = f"+{_h}d"
                                    if _col in _fwd_df.columns:
                                        _v = pd.to_numeric(_fwd_df[_col], errors="coerce").dropna()
                                        if not _v.empty:
                                            _wr_data.append({"Horizon": _col,
                                                             "Win Rate %": round((_v > 0).mean() * 100, 1),
                                                             "Avg Ret %":  round(_v.mean(), 2)})
                                if _wr_data:
                                    _wr_df = pd.DataFrame(_wr_data)
                                    _fig_wr = _pgo2.Figure()
                                    _fig_wr.add_bar(
                                        x=_wr_df["Horizon"], y=_wr_df["Win Rate %"], name="Win Rate %",
                                        marker_color=["#26a69a" if w >= 50 else "#ef5350" for w in _wr_df["Win Rate %"]]
                                    )
                                    _fig_wr.add_scatter(
                                        x=_wr_df["Horizon"], y=_wr_df["Avg Ret %"],
                                        name="Avg Ret %", mode="lines+markers",
                                        line=dict(color="#ffd700", width=2), yaxis="y2"
                                    )
                                    _fig_wr.update_layout(
                                        title=f"{_sel_tkr} — Post-Signal Win Rate & Avg Return",
                                        yaxis=dict(title="Win Rate %", range=[0, 100]),
                                        yaxis2=dict(title="Avg Ret %", overlaying="y", side="right",
                                                    zeroline=True, zerolinecolor="#555"),
                                        legend=dict(orientation="h"),
                                        height=320,
                                        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                                        font=dict(color="#fafafa"),
                                    )
                                    st.plotly_chart(_fig_wr, use_container_width=True)
                            except Exception:
                                pass

            st.download_button(
                "📥 Download hits as CSV",
                pd.DataFrame(hits)[_TABLE_COLS].to_csv(index=False),
                file_name=f"signal_scan_{strat.replace(' ','_')}_{today_str}.csv",
                mime="text/csv",
                key="ss_dl"
            )


if ENV_DIAG_CAPTION and nav in ('Chart', 'TradingView', 'Options'):
    try:
        with st.sidebar.expander("Env / Keys", expanded=False):
            st.caption(ENV_DIAG_CAPTION)
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Shared LLM call helper — supports Anthropic and OpenAI
# ─────────────────────────────────────────────────────────────────────────────
def _call_llm(provider: str, api_key: str, model: str,
               system_prompt: str, user_msg: str, max_tokens: int = 2000) -> str:
    """
    Call Anthropic or OpenAI and return the assistant text.
    Raises RuntimeError with a readable message on failure.
    """
    import urllib.request as _ur, json as _js, urllib.error as _ue
    import re as _re
    def _cln(s): return _re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', ' ', str(s))

    if provider == "Anthropic":
        payload = _js.dumps({
            "model": model,
            "max_tokens": max_tokens,
            "system": _cln(system_prompt),
            "messages": [{"role": "user", "content": _cln(user_msg)}],
        }, ensure_ascii=True).encode("utf-8")
        req = _ur.Request(
            "https://api.anthropic.com/v1/messages",
            data=payload,
            headers={"x-api-key": api_key, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            method="POST",
        )
        try:
            with _ur.urlopen(req, timeout=120) as r:
                data = _js.loads(r.read())
        except _ue.HTTPError as e:
            raise RuntimeError(f"Anthropic API {e.code}: {e.read().decode('utf-8','replace')}")
        return data["content"][0]["text"]

    elif provider == "OpenAI":
        payload = _js.dumps({
            "model": model,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": _cln(system_prompt)},
                {"role": "user",   "content": _cln(user_msg)},
            ],
        }, ensure_ascii=True).encode("utf-8")
        req = _ur.Request(
            "https://api.openai.com/v1/chat/completions",
            data=payload,
            headers={"Authorization": f"Bearer {api_key}",
                     "content-type": "application/json"},
            method="POST",
        )
        try:
            with _ur.urlopen(req, timeout=120) as r:
                data = _js.loads(r.read())
        except _ue.HTTPError as e:
            raise RuntimeError(f"OpenAI API {e.code}: {e.read().decode('utf-8','replace')}")
        return data["choices"][0]["message"]["content"]

    else:
        raise ValueError(f"Unknown provider: {provider}")

# ─────────────────────────────────────────────────────────────────────────────
# Goldman Sachs Fundamental Analysis
# ─────────────────────────────────────────────────────────────────────────────
if nav == 'GS Fundamental':
    st.markdown(
        """
        <div style='background:linear-gradient(90deg,#1a1a2e 0%,#16213e 60%,#0f3460 100%);
                    padding:18px 24px;border-radius:6px;margin-bottom:18px;
                    border-left:4px solid #c8a951'>
            <span style='color:#c8a951;font-size:11px;font-weight:700;letter-spacing:3px'>
                GOLDMAN SACHS
            </span><br>
            <span style='color:#fff;font-size:20px;font-weight:700;letter-spacing:0.5px'>
                Equity Research — Fundamental Analysis
            </span><br>
            <span style='color:#9ab;font-size:12px'>
                Institutional-grade fundamental research for the firm's asset management division
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    gsp1, gsp2 = st.columns([1, 3])
    with gsp1:
        gs_provider = st.selectbox("AI Provider", ["OpenAI", "Anthropic"], key="gs_provider")
    with gsp2:
        _gs_key_default = (os.getenv("OPENAI_API_KEY","") if gs_provider=="OpenAI" else os.getenv("ANTHROPIC_API_KEY","")).strip()
        gs_api_key = st.text_input(f"{gs_provider} API Key", value=_gs_key_default, type="password", key=f"gs_api_key_{gs_provider}", help=f"Auto-loaded from .env — overwrite if needed")
    anth_key_gs = gs_api_key

    gs1, gs2, gs3 = st.columns([2, 1, 1])
    with gs1:
        gs_ticker = st.text_input("Ticker", value=st.session_state.get("ticker", "AAPL"),
                                   max_chars=10, key="gs_ticker",
                                   placeholder="e.g. AAPL, MSFT, NVDA").strip().upper()
    with gs2:
        gs_focus = st.selectbox("Report focus", [
            "Full Fundamental Report",
            "Valuation Deep Dive",
            "Business Quality & Moat",
            "Bull vs Bear — Price Targets Only",
        ], key="gs_focus")
    with gs3:
        _gs_mdls = ["gpt-4.1","gpt-4o","gpt-4o-mini"] if gs_provider=="OpenAI" else ["claude-opus-4-6","claude-sonnet-4-6","claude-haiku-4-5-20251001"]
        gs_model = st.selectbox("Model", _gs_mdls, key="gs_model")

    with st.expander("Advanced options", expanded=False):
        gsa1, gsa2 = st.columns(2)
        with gsa1:
            gs_extra = st.text_area("Additional context / instructions",
                placeholder="e.g. 'Focus on AI monetization' or 'Compare vs MSFT and GOOGL'",
                key="gs_extra", height=80)
        with gsa2:
            gs_peers = st.text_input("Peer tickers (comma-separated)",
                placeholder="e.g. MSFT, GOOGL, META", key="gs_peers")
            gs_model_detail = st.selectbox("Depth", ["Standard (2,500 tokens)", "Deep (4,000 tokens)"],
                                            key="gs_depth")

    run_gs = st.button("Generate Fundamental Analysis", key="gs_run",
                        use_container_width=True, type="primary")

    if run_gs:
        if not gs_ticker:
            st.error("Enter a ticker symbol.")
        elif not gs_api_key:
            st.error(f"{gs_provider} API key required — paste above or set in .env")
        else:
            import traceback as _tb3
            with st.spinner(f"Pulling fundamental data for {gs_ticker}…"):
                try:
                    import yfinance as _yf3

                    _tk3 = _yf3.Ticker(gs_ticker)

                    # ── Info dict ─────────────────────────────────────────
                    try:
                        _info3 = _tk3.info or {}
                    except Exception:
                        _info3 = {}

                    def _g(key, default=None):
                        v = _info3.get(key, default)
                        return v if v not in (None, "None", "", "N/A") else default

                    def _pct(v):
                        try: return f"{float(v)*100:.1f}%"
                        except: return "N/A"

                    def _fmt3(v, prefix="$", suffix="", dec=2):
                        if v is None: return "N/A"
                        try:
                            fv = float(v)
                            if abs(fv) >= 1e12: return f"{prefix}{fv/1e12:.2f}T{suffix}"
                            if abs(fv) >= 1e9:  return f"{prefix}{fv/1e9:.2f}B{suffix}"
                            if abs(fv) >= 1e6:  return f"{prefix}{fv/1e6:.2f}M{suffix}"
                            return f"{prefix}{fv:.{dec}f}{suffix}"
                        except: return str(v)

                    # Core identity
                    company_name3   = _g("longName") or _g("shortName") or gs_ticker
                    sector3         = _g("sector", "N/A")
                    industry3       = _g("industry", "N/A")
                    description     = (_g("longBusinessSummary") or "")[:800]
                    employees       = _g("fullTimeEmployees")
                    country         = _g("country", "N/A")
                    exchange3       = _g("exchange", "N/A")

                    # Price & market
                    last_price3     = _g("currentPrice") or _g("regularMarketPrice")
                    market_cap3     = _g("marketCap")
                    enterprise_val  = _g("enterpriseValue")
                    beta3           = _g("beta")
                    week52_hi       = _g("fiftyTwoWeekHigh")
                    week52_lo       = _g("fiftyTwoWeekLow")

                    # Valuation multiples
                    pe_trailing     = _g("trailingPE")
                    pe_forward      = _g("forwardPE")
                    ps_ratio        = _g("priceToSalesTrailing12Months")
                    pb_ratio        = _g("priceToBook")
                    ev_ebitda       = _g("enterpriseToEbitda")
                    ev_revenue      = _g("enterpriseToRevenue")
                    peg_ratio       = _g("pegRatio")

                    # Income statement metrics
                    revenue_ttm3    = _g("totalRevenue")
                    rev_growth      = _g("revenueGrowth")
                    rev_per_share   = _g("revenuePerShare")
                    gross_margin3   = _g("grossMargins")
                    op_margin       = _g("operatingMargins")
                    ebitda_margin   = _g("ebitdaMargins")
                    net_margin3     = _g("profitMargins")
                    ebitda3         = _g("ebitda")
                    net_income      = _g("netIncomeToCommon")
                    eps_trail       = _g("trailingEps")
                    eps_fwd3        = _g("forwardEps")
                    eps_growth      = _g("earningsGrowth")
                    eps_growth_5y   = _g("earningsQuarterlyGrowth")

                    # Balance sheet
                    total_cash      = _g("totalCash")
                    cash_per_share  = _g("totalCashPerShare")
                    total_debt      = _g("totalDebt")
                    debt_equity     = _g("debtToEquity")
                    current_ratio3  = _g("currentRatio")
                    quick_ratio3    = _g("quickRatio")
                    book_val        = _g("bookValue")
                    return_equity   = _g("returnOnEquity")
                    return_assets   = _g("returnOnAssets")

                    # Cash flow
                    fcf             = _g("freeCashflow")
                    op_cashflow     = _g("operatingCashflow")
                    capex           = None
                    try:
                        if fcf and op_cashflow:
                            capex = float(op_cashflow) - float(fcf)
                    except: pass
                    fcf_yield       = None
                    try:
                        if fcf and market_cap3:
                            fcf_yield = float(fcf) / float(market_cap3) * 100
                    except: pass

                    # Dividends & buybacks
                    div_rate        = _g("dividendRate")
                    div_yield       = _g("dividendYield")
                    payout_ratio    = _g("payoutRatio")
                    five_yr_div_yld = _g("fiveYearAvgDividendYield")

                    # Analyst consensus
                    target_mean3    = _g("targetMeanPrice")
                    target_low3     = _g("targetLowPrice")
                    target_high3    = _g("targetHighPrice")
                    rec_key3        = _g("recommendationKey", "N/A")
                    rec_mean3       = _g("recommendationMean")
                    num_analysts3   = _g("numberOfAnalystOpinions")

                    # Ownership
                    inst_own3       = _g("heldPercentInstitutions")
                    insider_own     = _g("heldPercentInsiders")
                    short_ratio     = _g("shortRatio")
                    short_float3    = _g("shortPercentOfFloat")

                    # ── Historical financials (income stmt / balance sheet) ──
                    income_str  = "N/A"
                    balance_str = "N/A"
                    cashflow_str= "N/A"
                    try:
                        _is = _tk3.income_stmt
                        if _is is not None and not _is.empty:
                            # Key rows only
                            key_rows = [r for r in ["Total Revenue","Gross Profit","Operating Income",
                                                     "Net Income","EBITDA","Basic EPS"] if r in _is.index]
                            income_str = _is.loc[key_rows].iloc[:, :5].to_string() if key_rows else _is.iloc[:6, :4].to_string()
                    except Exception: pass
                    try:
                        _bs = _tk3.balance_sheet
                        if _bs is not None and not _bs.empty:
                            key_rows_b = [r for r in ["Total Assets","Total Debt","Stockholders Equity",
                                                       "Cash And Cash Equivalents","Total Current Assets",
                                                       "Total Current Liabilities"] if r in _bs.index]
                            balance_str = _bs.loc[key_rows_b].iloc[:, :4].to_string() if key_rows_b else _bs.iloc[:6, :4].to_string()
                    except Exception: pass
                    try:
                        _cf = _tk3.cashflow
                        if _cf is not None and not _cf.empty:
                            key_rows_c = [r for r in ["Operating Cash Flow","Free Cash Flow",
                                                       "Capital Expenditure","Repurchase Of Capital Stock",
                                                       "Cash Dividends Paid"] if r in _cf.index]
                            cashflow_str = _cf.loc[key_rows_c].iloc[:, :4].to_string() if key_rows_c else _cf.iloc[:6, :4].to_string()
                    except Exception: pass

                    # ── Peer data ─────────────────────────────────────────
                    peer_snap = ""
                    peer_list = [p.strip().upper() for p in (gs_peers or "").split(",") if p.strip()]
                    if peer_list:
                        peer_rows = []
                        for _pt in peer_list[:4]:
                            try:
                                _pi = _yf3.Ticker(_pt).info or {}
                                peer_rows.append(
                                    f"  {_pt}: P/E={_pi.get('trailingPE','N/A')} "
                                    f"Fwd P/E={_pi.get('forwardPE','N/A')} "
                                    f"EV/EBITDA={_pi.get('enterpriseToEbitda','N/A')} "
                                    f"Rev Growth={_pct(_pi.get('revenueGrowth'))} "
                                    f"Net Margin={_pct(_pi.get('profitMargins'))} "
                                    f"Price=${_pi.get('currentPrice','N/A')}"
                                )
                            except Exception:
                                peer_rows.append(f"  {_pt}: data unavailable")
                        peer_snap = "=== PEER COMPARISON ===\n" + "\n".join(peer_rows) + "\n"

                    # ── Price performance ─────────────────────────────────
                    try:
                        _ph3 = _tk3.history(period="5y", interval="1d", auto_adjust=True)
                        _ph3 = _ph3.dropna(subset=["Close"]).sort_index()
                        _last3 = float(_ph3["Close"].iloc[-1])
                        _last_date3 = _ph3.index[-1].strftime("%Y-%m-%d")
                        ret_1y = (_last3 / float(_ph3["Close"].iloc[-252]) - 1)*100 if len(_ph3) > 252 else None
                        ret_3y = (_last3 / float(_ph3["Close"].iloc[-756]) - 1)*100 if len(_ph3) > 756 else None
                        ret_5y = (_last3 / float(_ph3["Close"].iloc[0])    - 1)*100 if len(_ph3) > 5   else None
                        ytd3   = _ph3[_ph3.index.year == _ph3.index[-1].year]
                        ret_ytd = (float(ytd3["Close"].iloc[-1]) / float(ytd3["Close"].iloc[0]) - 1)*100 if len(ytd3) > 1 else None
                        avg_vol3 = float(_ph3["Volume"].tail(20).mean()) if "Volume" in _ph3.columns else None
                    except Exception:
                        _last3 = last_price3
                        _last_date3 = "N/A"
                        ret_1y = ret_3y = ret_5y = ret_ytd = avg_vol3 = None

                    # ── Build data snapshot ───────────────────────────────
                    data_snapshot3 = f"""
COMPANY: {company_name3} ({gs_ticker})
DATE: {_last_date3}
SECTOR: {sector3} | INDUSTRY: {industry3} | COUNTRY: {country}
EMPLOYEES: {f'{int(employees):,}' if employees else 'N/A'} | EXCHANGE: {exchange3}

BUSINESS DESCRIPTION:
{description}

=== MARKET DATA ===
Current Price:     {_fmt3(last_price3)}
Market Cap:        {_fmt3(market_cap3)}
Enterprise Value:  {_fmt3(enterprise_val)}
52-Week Range:     ${week52_lo} — ${week52_hi}
Beta:              {beta3 if beta3 else 'N/A'}

=== PRICE PERFORMANCE ===
YTD:    {f'{ret_ytd:+.1f}%' if ret_ytd is not None else 'N/A'}
1-Year: {f'{ret_1y:+.1f}%' if ret_1y is not None else 'N/A'}
3-Year: {f'{ret_3y:+.1f}%' if ret_3y is not None else 'N/A'}
5-Year: {f'{ret_5y:+.1f}%' if ret_5y is not None else 'N/A'}

=== VALUATION MULTIPLES ===
P/E (Trailing):     {pe_trailing if pe_trailing else 'N/A'}
P/E (Forward):      {pe_forward if pe_forward else 'N/A'}
PEG Ratio:          {peg_ratio if peg_ratio else 'N/A'}
P/S (TTM):          {ps_ratio if ps_ratio else 'N/A'}
P/B:                {pb_ratio if pb_ratio else 'N/A'}
EV/EBITDA:          {ev_ebitda if ev_ebitda else 'N/A'}
EV/Revenue:         {ev_revenue if ev_revenue else 'N/A'}

=== INCOME STATEMENT (TTM) ===
Revenue:            {_fmt3(revenue_ttm3)}
Revenue Growth YoY: {_pct(rev_growth)}
Gross Margin:       {_pct(gross_margin3)}
Operating Margin:   {_pct(op_margin)}
EBITDA Margin:      {_pct(ebitda_margin)}
Net Margin:         {_pct(net_margin3)}
EBITDA:             {_fmt3(ebitda3)}
Net Income:         {_fmt3(net_income)}
EPS (Trailing):     {f'${eps_trail:.2f}' if eps_trail else 'N/A'}
EPS (Forward):      {f'${eps_fwd3:.2f}' if eps_fwd3 else 'N/A'}
EPS Growth (YoY):   {_pct(eps_growth)}

=== BALANCE SHEET ===
Total Cash:         {_fmt3(total_cash)}
Total Debt:         {_fmt3(total_debt)}
Net Cash/(Debt):    {_fmt3(float(total_cash or 0) - float(total_debt or 0))}
Debt/Equity:        {f'{debt_equity:.1f}x' if debt_equity else 'N/A'}
Current Ratio:      {f'{current_ratio3:.2f}x' if current_ratio3 else 'N/A'}
Quick Ratio:        {f'{quick_ratio3:.2f}x' if quick_ratio3 else 'N/A'}
Return on Equity:   {_pct(return_equity)}
Return on Assets:   {_pct(return_assets)}

=== FREE CASH FLOW ===
Free Cash Flow:     {_fmt3(fcf)}
Operating Cash Flow:{_fmt3(op_cashflow)}
CapEx:              {_fmt3(capex)}
FCF Yield:          {f'{fcf_yield:.2f}%' if fcf_yield else 'N/A'}
FCF/Net Income:     {f'{float(fcf)/float(net_income):.2f}x' if fcf and net_income and float(net_income) != 0 else 'N/A'}

=== DIVIDENDS & CAPITAL RETURN ===
Dividend Rate:      {f'${div_rate:.2f}/yr' if div_rate else 'N/A (no dividend)'}
Dividend Yield:     {_pct(div_yield) if div_yield else 'N/A'}
Payout Ratio:       {_pct(payout_ratio) if payout_ratio else 'N/A'}
5-Yr Avg Div Yield: {f'{five_yr_div_yld:.2f}%' if five_yr_div_yld else 'N/A'}

=== OWNERSHIP ===
Institutional:      {_pct(inst_own3)}
Insider:            {_pct(insider_own)}
Short Float:        {_pct(short_float3)}
Short Ratio:        {f'{short_ratio:.1f} days to cover' if short_ratio else 'N/A'}

=== ANALYST CONSENSUS ===
Rating:             {rec_key3.upper() if rec_key3 and rec_key3 != 'N/A' else 'N/A'} (score: {rec_mean3 if rec_mean3 else 'N/A'})
# of Analysts:      {num_analysts3 if num_analysts3 else 'N/A'}
Price Target Mean:  {_fmt3(target_mean3)} ({f'{((float(target_mean3)/float(last_price3))-1)*100:+.1f}% from current' if target_mean3 and last_price3 else ''})
Price Target Range: {_fmt3(target_low3)} — {_fmt3(target_high3)}

=== HISTORICAL INCOME STATEMENT ===
{income_str}

=== HISTORICAL BALANCE SHEET ===
{balance_str}

=== HISTORICAL CASH FLOWS ===
{cashflow_str}

{peer_snap}"""

                    st.success(f"Fundamental data loaded for {company_name3}. Generating Goldman Sachs analysis…")

                except Exception as _e5:
                    st.error(f"Failed to load data: {_e5}")
                    st.code(_tb3.format_exc())
                    st.stop()

            # ── System prompt ─────────────────────────────────────────────
            gs_system = """You are a Managing Director in Equity Research at Goldman Sachs, 
covering the company for the firm's $2 trillion asset management division.
You have 20 years of experience analyzing businesses for institutional investors.

Your research notes are precise, data-driven, and opinionated. 
You do not hedge everything — you take a clear Buy, Neutral, or Sell stance with a conviction level.
You think about business quality, competitive moats, and capital allocation above all else.

Format your response EXACTLY as a Goldman Sachs Equity Research note with these sections:

1. RATING BOX (at top — ticker, rating, 12-month price target, conviction level, key stats in a table)
2. INVESTMENT THESIS (2-3 sentence core argument — why buy, hold, or avoid)
3. BUSINESS MODEL BREAKDOWN (how the company makes money, simply explained)
4. REVENUE STREAMS & SEGMENT ANALYSIS (each segment: % of revenue, growth rate, margin)
5. PROFITABILITY ANALYSIS (gross/operating/net margin trends, quality of earnings)
6. BALANCE SHEET HEALTH (debt load, liquidity, net cash position interpretation)
7. FREE CASH FLOW ANALYSIS (FCF yield, FCF quality, capital allocation grade A-F)
8. COMPETITIVE ADVANTAGES / MOAT (rate each: pricing power, brand, switching costs, network effects, scale — score 1-10 each)
9. MANAGEMENT QUALITY ASSESSMENT (capital allocation track record, insider ownership, compensation structure)
10. VALUATION ANALYSIS (current multiples vs 5-year history and sector peers, DCF implied value if data sufficient)
11. BULL CASE — 12-month price target with assumptions
12. BEAR CASE — 12-month price target with assumptions
13. BASE CASE — 12-month price target (this is your official target)
14. VERDICT (one paragraph: buy / hold / avoid with conviction level 1-5 stars)
15. KEY RISKS
16. DISCLAIMER

Use exact numbers from the data. Be specific about what the moat score means.
For capital allocation grade, consider: buybacks at good prices, dividends, M&A discipline, R&D ROI."""

            focus_addon = {
                "Full Fundamental Report": "Write the complete institutional research note covering all 16 sections.",
                "Valuation Deep Dive": "Emphasize sections 10-13 (Valuation, Bull/Bear/Base case). Build a thorough valuation framework using all available multiples. Compare to peers if provided. Less focus on qualitative sections.",
                "Business Quality & Moat": "Emphasize sections 3-4 (Business Model), 8 (Competitive Advantages), and 9 (Management Quality). Score each moat dimension carefully with specific evidence from the data.",
                "Bull vs Bear — Price Targets Only": "Focus almost entirely on sections 11-14. Give detailed Bull/Bear/Base cases with specific price targets, assumptions, and catalysts for each scenario.",
            }.get(gs_focus, "")

            max_tok = 4000 if "Deep" in gs_model_detail else 2500

            user_msg3 = f"""Here is the complete fundamental data for {gs_ticker}:

{data_snapshot3}

Report focus: {gs_focus}
{focus_addon}
{f'Additional context: {gs_extra}' if gs_extra and gs_extra.strip() else ''}

Generate the Goldman Sachs equity research note now.
Use exact numbers. Be opinionated. Give a clear Buy/Neutral/Sell with conviction level."""

            with st.spinner("Generating Goldman Sachs fundamental analysis…"):
                try:
                    # Read directly from session_state to avoid stale widget cache
                    _gs_prov_live = st.session_state.get("gs_provider", "OpenAI")
                    _gs_key_live = (os.getenv("OPENAI_API_KEY","") if _gs_prov_live=="OpenAI" else os.getenv("ANTHROPIC_API_KEY","")).strip()
                    _gs_key_field = st.session_state.get(f"gs_api_key_{_gs_prov_live}", "").strip()
                    if _gs_key_field: _gs_key_live = _gs_key_field
                    _gs_mdl_live = st.session_state.get("gs_model", gs_model)
                    if not _gs_key_live:
                        st.error(f"No {_gs_prov_live} key found — set OPENAI_API_KEY or ANTHROPIC_API_KEY in .env"); st.stop()
                    st.caption(f"Provider: {_gs_prov_live} | Model: {_gs_mdl_live} | Key: ...{_gs_key_live[-4:]}")
                    _output3 = _call_llm(_gs_prov_live, _gs_key_live, _gs_mdl_live, gs_system, user_msg3, max_tokens=max_tok)
                    if not _output3:
                        st.error("No response returned from model.")
                    else:
                        # ── GS-style header ───────────────────────────────
                        st.markdown(
                            "<div style='background:#1a1a2e;border:1px solid #c8a951;"
                            "border-radius:6px;margin-bottom:12px'>"
                            "<div style='background:linear-gradient(90deg,#1a1a2e,#16213e);padding:10px 20px;border-radius:5px 5px 0 0;border-bottom:1px solid #c8a951'>"
                            "<span style='color:#c8a951;font-weight:700;font-size:13px;letter-spacing:2px'>GOLDMAN SACHS</span>"
                            f"<span style='color:#fff;font-weight:700;font-size:14px'> | EQUITY RESEARCH | {company_name3} ({gs_ticker}) | {_last_date3}</span>"
                            "</div></div>",
                            unsafe_allow_html=True,
                        )

                        # KPI bar
                        gk1, gk2, gk3, gk4, gk5, gk6 = st.columns(6)
                        gk1.metric("Price", f"${_last3:.2f}" if _last3 else "N/A")
                        gk2.metric("Market Cap", _fmt3(market_cap3))
                        gk3.metric("P/E (Fwd)", f"{pe_forward:.1f}x" if pe_forward else "N/A")
                        gk4.metric("EV/EBITDA", f"{ev_ebitda:.1f}x" if ev_ebitda else "N/A")
                        gk5.metric("FCF Yield", f"{fcf_yield:.1f}%" if fcf_yield else "N/A")
                        gk6.metric("Analyst Target", _fmt3(target_mean3),
                                   delta=f"{((float(target_mean3)/float(_last3))-1)*100:+.1f}%" if target_mean3 and _last3 else None)

                        st.markdown(_output3)

                        with st.expander("Raw fundamental data", expanded=False):
                            st.code(data_snapshot3, language="text")

                        st.download_button(
                            "Download as text",
                            data=f"GOLDMAN SACHS EQUITY RESEARCH\n{company_name3} ({gs_ticker}) | {_last_date3}\n\n{_output3}\n\n--- RAW DATA ---\n{data_snapshot3}",
                            file_name=f"GS_Fundamental_{gs_ticker}_{_last_date3}.txt",
                            mime="text/plain",
                        )

                except Exception as _e6:
                    st.error(f"API call failed: {_e6}")
                    st.code(_tb3.format_exc())


if nav == 'MS Analysis':
    st.markdown(
        """<div style='background:linear-gradient(90deg,#003087 0%,#0066cc 100%);
                    padding:18px 24px;border-radius:6px;margin-bottom:18px'>
            <span style='color:#fff;font-size:22px;font-weight:700'>
                Morgan Stanley - Technical Strategy Desk
            </span><br>
            <span style='color:#a8c8ff;font-size:13px'>
                Institutional-grade technical analysis powered by real market data
            </span></div>""",
        unsafe_allow_html=True,
    )
    msp1, msp2 = st.columns([1, 3])
    with msp1:
        ms_provider = st.selectbox("AI Provider", ["OpenAI", "Anthropic"], key="ms_provider")
    with msp2:
        _ms_key_default = (os.getenv("OPENAI_API_KEY","") if ms_provider=="OpenAI" else os.getenv("ANTHROPIC_API_KEY","")).strip()
        ms_api_key = st.text_input(f"{ms_provider} API Key", value=_ms_key_default, type="password", key=f"ms_api_key_{ms_provider}", help=f"Auto-loaded from .env — overwrite if needed")
    ms1, ms2, ms3 = st.columns([2, 1, 1])
    with ms1:
        ms_ticker = st.text_input("Ticker", value=st.session_state.get("ticker","AAPL"),
                                   max_chars=10, key="ms_ticker").strip().upper()
    with ms2:
        ms_timeframe = st.selectbox("Primary timeframe", ["Daily","Weekly","Monthly"], key="ms_tf")
    with ms3:
        ms_style = st.selectbox("Report style", [
            "Full institutional note",
            "Trade setup only (entry/stop/targets)",
            "Pattern & momentum focus",
        ], key="ms_style")
    with st.expander("Advanced options", expanded=False):
        msadv1, msadv2 = st.columns(2)
        with msadv1:
            ms_extra = st.text_area("Additional context",
                                     placeholder="e.g. Focus on options flow",
                                     key="ms_extra", height=80)
        with msadv2:
            ms_period = st.selectbox("Data lookback", ["1y","2y","3y","5y"], index=1, key="ms_period")
            _ms_mdls = ["gpt-4.1","gpt-4o","gpt-4o-mini"] if ms_provider=="OpenAI" else ["claude-opus-4-6","claude-sonnet-4-6","claude-haiku-4-5-20251001"]
            ms_model  = st.selectbox("Model", _ms_mdls, key="ms_model")
    run_ms = st.button("Generate Technical Analysis", key="ms_run", use_container_width=True, type="primary")
    if run_ms:
        if not ms_ticker:
            st.error("Enter a ticker.")
        elif not ms_api_key:
            st.error(f"{ms_provider} API key required — paste above or set in .env")
        else:
            import traceback as _tb_ms
            with st.spinner("Computing indicators..."):
                try:
                    import yfinance as _yf_ms
                    _hms = _yf_ms.Ticker(ms_ticker).history(period=ms_period, interval="1d", auto_adjust=True)
                    if _hms is None or _hms.empty:
                        st.error("No price data.")
                        st.stop()
                    _hms = _hms.dropna(subset=["Close"]).sort_index()
                    _cms = _hms["Close"].astype(float)
                    _hhs = _hms["High"].astype(float)
                    _lls = _hms["Low"].astype(float)
                    _vms = _hms["Volume"].astype(float)

                    def _smams(s, n): return s.rolling(n, min_periods=n).mean()
                    def _emams(s, n): return s.ewm(span=n, adjust=False).mean()
                    def _lv(s): d = s.dropna(); return float(d.iloc[-1]) if len(d) > 0 else None

                    last_ms      = float(_cms.iloc[-1])
                    last_date_ms = _cms.index[-1].strftime("%Y-%m-%d")
                    sma20v  = _lv(_smams(_cms, 20))
                    sma50v  = _lv(_smams(_cms, 50))
                    sma100v = _lv(_smams(_cms, 100))
                    sma200v = _lv(_smams(_cms, 200))

                    _dms  = _cms.diff()
                    _agms = _dms.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
                    _alms = (-_dms.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
                    rsi_ms   = (100 - 100/(1+_agms/_alms.replace(0, float("nan")))).fillna(50)
                    rsi_v_ms = float(rsi_ms.iloc[-1])

                    macd_ms    = _emams(_cms, 12) - _emams(_cms, 26)
                    sig_ms     = _emams(macd_ms, 9)
                    hist_ms    = macd_ms - sig_ms
                    macd_v_ms  = float(macd_ms.iloc[-1])
                    sig_v_ms   = float(sig_ms.iloc[-1])
                    hist_v_ms  = float(hist_ms.iloc[-1])
                    hist_pv_ms = float(hist_ms.iloc[-2]) if len(hist_ms) > 2 else hist_v_ms

                    sma20_ms  = _smams(_cms, 20)
                    bb_std_ms = _cms.rolling(20, min_periods=20).std()
                    bb_u_ms   = _lv(sma20_ms + 2*bb_std_ms)
                    bb_l_ms   = _lv(sma20_ms - 2*bb_std_ms)
                    bb_m_ms   = _lv(sma20_ms)
                    bb_w_ms   = ((bb_u_ms - bb_l_ms)/bb_m_ms*100) if bb_u_ms and bb_l_ms and bb_m_ms else None
                    bb_pctb_ms = ((last_ms-bb_l_ms)/(bb_u_ms-bb_l_ms)) if bb_u_ms and bb_l_ms and (bb_u_ms-bb_l_ms)>0 else 0.5

                    avg_vol_ms  = float(_vms.tail(20).mean())
                    last_vol_ms = float(_vms.iloc[-1])
                    vol_ratio_ms = last_vol_ms/avg_vol_ms if avg_vol_ms > 0 else 1.0

                    hi52_ms = float(_hhs.tail(252).max())
                    lo52_ms = float(_lls.tail(252).min())
                    fib_rng = hi52_ms - lo52_ms
                    fibs_ms = {
                        "23.6%": round(hi52_ms - 0.236*fib_rng, 2),
                        "38.2%": round(hi52_ms - 0.382*fib_rng, 2),
                        "50.0%": round(hi52_ms - 0.500*fib_rng, 2),
                        "61.8%": round(hi52_ms - 0.618*fib_rng, 2),
                        "78.6%": round(hi52_ms - 0.786*fib_rng, 2),
                    }
                    nf_ms = min(fibs_ms.items(), key=lambda x: abs(x[1] - last_ms))

                    _tr_ms = pd.concat([_hhs-_lls, (_hhs-_cms.shift()).abs(), (_lls-_cms.shift()).abs()], axis=1).max(axis=1)
                    atr_ms = float(_tr_ms.rolling(14).mean().dropna().iloc[-1])

                    def _ab(p, m):
                        if m is None: return "N/A"
                        return "ABOVE" if p > m else "BELOW"

                    # Pre-format all values — no inline conditionals in format specs
                    sma20_s  = "${:.2f}".format(sma20v)  if sma20v  else "N/A"
                    sma50_s  = "${:.2f}".format(sma50v)  if sma50v  else "N/A"
                    sma100_s = "${:.2f}".format(sma100v) if sma100v else "N/A"
                    sma200_s = "${:.2f}".format(sma200v) if sma200v else "N/A"
                    bb_u_s   = "${:.2f}".format(bb_u_ms) if bb_u_ms else "N/A"
                    bb_m_s   = "${:.2f}".format(bb_m_ms) if bb_m_ms else "N/A"
                    bb_l_s   = "${:.2f}".format(bb_l_ms) if bb_l_ms else "N/A"
                    bb_w_s   = "{:.2f}%".format(bb_w_ms) if bb_w_ms else "N/A"
                    daily_trend_ms  = ("Bullish" if sma20v and sma50v and last_ms > sma20v and sma20v > sma50v
                                       else "Bearish" if sma20v and sma50v and last_ms < sma20v and sma20v < sma50v
                                       else "Mixed/Neutral")
                    weekly_trend_ms = ("Bullish" if sma50v and sma200v and sma50v > sma200v
                                       else "Bearish" if sma50v and sma200v and sma50v < sma200v else "Neutral")
                    rsi_lbl_ms = "Overbought" if rsi_v_ms > 70 else ("Oversold" if rsi_v_ms < 30 else "Neutral")
                    hist_dir_ms = "Expanding" if abs(hist_v_ms) > abs(hist_pv_ms) else "Contracting"
                    macd_dir_ms = "ABOVE" if macd_v_ms > sig_v_ms else "BELOW"
                    fib_line = "  ".join(f"{k}=${v:.2f}" for k, v in fibs_ms.items())

                    snap_ms = "\n".join([
                        f"TICKER: {ms_ticker}  DATE: {last_date_ms}",
                        f"PRICE: ${last_ms:.2f}  52w: ${lo52_ms:.2f}-${hi52_ms:.2f}  ATR: ${atr_ms:.2f} ({atr_ms/last_ms*100:.2f}% of price)",
                        "",
                        f"TREND:",
                        f"  Daily:  {daily_trend_ms}",
                        f"  Weekly: {weekly_trend_ms}",
                        f"  SMA20={sma20_s} ({_ab(last_ms,sma20v)})  SMA50={sma50_s} ({_ab(last_ms,sma50v)})",
                        f"  SMA100={sma100_s} ({_ab(last_ms,sma100v)})  SMA200={sma200_s} ({_ab(last_ms,sma200v)})",
                        "",
                        f"RSI(14): {rsi_v_ms:.1f} - {rsi_lbl_ms}",
                        "",
                        f"MACD(12,26,9):",
                        f"  Line={macd_v_ms:.4f}  Signal={sig_v_ms:.4f}  Hist={hist_v_ms:.4f} ({hist_dir_ms})",
                        f"  MACD {macd_dir_ms} signal line",
                        "",
                        f"BOLLINGER(20,2):",
                        f"  Upper={bb_u_s}  Mid={bb_m_s}  Lower={bb_l_s}",
                        f"  %B={bb_pctb_ms:.3f}  Width={bb_w_s}",
                        "",
                        f"VOLUME: {int(last_vol_ms):,} vs 20d avg {int(avg_vol_ms):,} ({vol_ratio_ms:.2f}x)",
                        "",
                        "FIBONACCI (52w Low to High):",
                        f"  {fib_line}",
                        f"  Nearest to current: {nf_ms[0]} at ${nf_ms[1]:.2f}",
                    ])
                    st.success("Data loaded. Generating Morgan Stanley analysis...")

                except Exception as _e_ms:
                    st.error("Failed to load data: {}".format(_e_ms))
                    st.code(_tb_ms.format_exc())
                    st.stop()

            ms_system = """You are a Managing Director and senior technical strategist at Morgan Stanley.
Format EXACTLY as a Morgan Stanley technical analysis note:
1. TRADE PLAN SUMMARY (direction, entry, stop, 2 targets, R/R ratio)
2. TREND ANALYSIS (daily, weekly, monthly)
3. MOVING AVERAGES  4. RSI ANALYSIS  5. MACD ANALYSIS
6. BOLLINGER BANDS  7. VOLUME ANALYSIS  8. SUPPORT & RESISTANCE
9. FIBONACCI LEVELS  10. CHART PATTERNS  11. RISK FACTORS  12. DISCLAIMER
Use exact price levels. Be opinionated. Take a clear directional view."""

            style_map_ms = {
                "Full institutional note": "Write the complete note covering all 12 sections.",
                "Trade setup only (entry/stop/targets)": "Focus on Trade Plan (1) and S/R (8). Make entry, stop, targets very specific.",
                "Pattern & momentum focus": "Emphasize RSI (4), MACD (5), Bollinger (6), and Chart Patterns (10).",
            }
            extra_ms = ("Context: " + ms_extra) if ms_extra and ms_extra.strip() else ""
            user_msg_ms = "\n".join([
                "Technical data for {}:".format(ms_ticker),
                "",
                snap_ms,
                "",
                "Timeframe: {}".format(ms_timeframe),
                "Style: {}".format(ms_style),
                style_map_ms.get(ms_style, ""),
                extra_ms,
                "",
                "Generate the Morgan Stanley technical analysis note with exact price levels.",
            ])

            with st.spinner("Generating Morgan Stanley analysis..."):
                try:
                    _ms_prov_live = st.session_state.get("ms_provider", "OpenAI")
                    _ms_key_live = (os.getenv("OPENAI_API_KEY","") if _ms_prov_live=="OpenAI" else os.getenv("ANTHROPIC_API_KEY","")).strip()
                    _ms_key_field = st.session_state.get(f"ms_api_key_{_ms_prov_live}", "").strip()
                    if _ms_key_field: _ms_key_live = _ms_key_field
                    _ms_mdl_live = st.session_state.get("ms_model", ms_model)
                    if not _ms_key_live:
                        st.error(f"No {_ms_prov_live} key found — set OPENAI_API_KEY or ANTHROPIC_API_KEY in .env"); st.stop()
                    st.caption(f"Provider: {_ms_prov_live} | Model: {_ms_mdl_live} | Key: ...{_ms_key_live[-4:]}")
                    _out_ms = _call_llm(_ms_prov_live, _ms_key_live, _ms_mdl_live, ms_system, user_msg_ms, max_tokens=2000)
                    if not _out_ms:
                        st.error("No response returned.")
                    else:
                        hdr_ms = (
                            "<div style='background:#003087;padding:10px 20px;border-radius:5px;margin-bottom:12px'>"
                            "<span style='color:#fff;font-weight:700'>MORGAN STANLEY | TECHNICAL ANALYSIS | "
                            + ms_ticker + " | " + last_date_ms + "</span></div>"
                        )
                        st.markdown(hdr_ms, unsafe_allow_html=True)
                        msk1, msk2, msk3, msk4 = st.columns(4)
                        msk1.metric("Price", "${:.2f}".format(last_ms))
                        msk2.metric("RSI(14)", "{:.1f}".format(rsi_v_ms))
                        msk3.metric("MACD", "{:.3f}".format(macd_v_ms))
                        msk4.metric("52w Range", "${:.0f} - ${:.0f}".format(lo52_ms, hi52_ms))
                        st.markdown(_out_ms)
                        with st.expander("Raw indicator data", expanded=False):
                            st.code(snap_ms, language="text")
                        dl_ms = "MORGAN STANLEY TECHNICAL ANALYSIS\n{} | {}\n\n{}\n\n--- RAW DATA ---\n{}".format(
                            ms_ticker, last_date_ms, _out_ms, snap_ms)
                        st.download_button("Download as text", data=dl_ms,
                            file_name="MS_TechAnalysis_{}_{}.txt".format(ms_ticker, last_date_ms),
                            mime="text/plain")
                except Exception as _e_ms2:
                    st.error("Claude API call failed: {}".format(_e_ms2))
                    st.code(_tb_ms.format_exc())


# ─────────────────────────────────────────────────────────────────────────────
# JPMorgan Earnings Analyzer
# ─────────────────────────────────────────────────────────────────────────────
if nav == 'JPM Earnings':
    st.markdown(
        """<div style='background:linear-gradient(90deg,#003087 0%,#00539b 100%);
                    padding:18px 24px;border-radius:6px;margin-bottom:18px'>
            <span style='color:#fff;font-size:22px;font-weight:700'>
                JPMorgan Chase - Equity Research
            </span><br>
            <span style='color:#a8d4ff;font-size:13px'>Pre and Post-Earnings Analysis</span>
        </div>""",
        unsafe_allow_html=True,
    )
    jpmp1, jpmp2 = st.columns([1, 3])
    with jpmp1:
        jpm_provider = st.selectbox("AI Provider", ["OpenAI", "Anthropic"], key="jpm_provider")
    with jpmp2:
        _jpm_key_default = (os.getenv("OPENAI_API_KEY","") if jpm_provider=="OpenAI" else os.getenv("ANTHROPIC_API_KEY","")).strip()
        jpm_api_key = st.text_input(f"{jpm_provider} API Key", value=_jpm_key_default, type="password", key=f"jpm_api_key_{jpm_provider}", help=f"Auto-loaded from .env — overwrite if needed")
    jpm1, jpm2, jpm3 = st.columns([2, 1, 1])
    with jpm1:
        jpm_ticker = st.text_input("Ticker", value=st.session_state.get("ticker","AAPL"),
                                    max_chars=10, key="jpm_ticker").strip().upper()
    with jpm2:
        jpm_mode = st.selectbox("Report type", [
            "Pre-Earnings Preview", "Post-Earnings Reaction", "Full Earnings Dossier"], key="jpm_mode")
    with jpm3:
        _jpm_mdls = ["gpt-4.1","gpt-4o","gpt-4o-mini"] if jpm_provider=="OpenAI" else ["claude-opus-4-6","claude-sonnet-4-6","claude-haiku-4-5-20251001"]
        jpm_model = st.selectbox("Model", _jpm_mdls, key="jpm_model")
    with st.expander("Advanced options", expanded=False):
        jpma1, jpma2 = st.columns(2)
        with jpma1:
            jpm_extra = st.text_area("Additional context",
                                      placeholder="e.g. Focus on AI segment", key="jpm_extra", height=80)
        with jpma2:
            jpm_quarters = st.number_input("Earnings history (quarters)", value=8,
                min_value=4, max_value=16, step=1, key="jpm_quarters")
            jpm_price_period = st.selectbox("Price history", ["1y","2y","3y"], index=1, key="jpm_period")
    run_jpm = st.button("Generate Earnings Analysis", key="jpm_run", use_container_width=True, type="primary")

    if run_jpm:
        if not jpm_ticker:
            st.error("Enter a ticker.")
        elif not jpm_api_key:
            st.error(f"{jpm_provider} API key required — paste above or set in .env")
        else:
            import traceback as _tb_jpm

            def _jfmt(v, prefix="$", dec=2):
                if v is None: return "N/A"
                try:
                    fv = float(v)
                    if abs(fv) >= 1e12: return "{}{:.2f}T".format(prefix, fv/1e12)
                    if abs(fv) >= 1e9:  return "{}{:.2f}B".format(prefix, fv/1e9)
                    if abs(fv) >= 1e6:  return "{}{:.2f}M".format(prefix, fv/1e6)
                    return "{}{:.{}f}".format(prefix, fv, dec)
                except: return "N/A"

            def _jpct(v):
                try: return "{:.1f}%".format(float(v)*100)
                except: return "N/A"

            def _jf2(v, suf="", dec=2):
                try: return "{:.{}f}{}".format(float(v), dec, suf)
                except: return "N/A"

            with st.spinner("Pulling earnings data..."):
                try:
                    import yfinance as _yf_jpm
                    _tkj = _yf_jpm.Ticker(jpm_ticker)
                    try: _infj = _tkj.info or {}
                    except: _infj = {}

                    def _jg(k, d=None):
                        v = _infj.get(k, d)
                        return v if v not in (None, "None", "", "N/A") else d

                    company_jpm  = _jg("longName") or _jg("shortName") or jpm_ticker
                    sector_jpm   = _jg("sector", "N/A")
                    mktcap_jpm   = _jg("marketCap")
                    beta_jpm     = _jg("beta")
                    pe_trail_j   = _jg("trailingPE")
                    pe_fwd_j     = _jg("forwardPE")
                    eps_t_j      = _jg("trailingEps")
                    eps_f_j      = _jg("forwardEps")
                    rev_j        = _jg("totalRevenue")
                    rev_g_j      = _jg("revenueGrowth")
                    gm_j         = _jg("grossMargins")
                    nm_j         = _jg("profitMargins")
                    tgt_mean_j   = _jg("targetMeanPrice")
                    tgt_lo_j     = _jg("targetLowPrice")
                    tgt_hi_j     = _jg("targetHighPrice")
                    rec_jpm      = _jg("recommendationKey", "N/A")
                    num_an_j     = _jg("numberOfAnalystOpinions")
                    inst_j       = _jg("heldPercentInstitutions")
                    sht_j        = _jg("shortPercentOfFloat")

                    _phj = _tkj.history(period=jpm_price_period, interval="1d", auto_adjust=True)
                    _phj = _phj.dropna(subset=["Close"]).sort_index()
                    last_px_j   = float(_phj["Close"].iloc[-1]) if not _phj.empty else None
                    last_date_j = _phj.index[-1].strftime("%Y-%m-%d") if not _phj.empty else "N/A"
                    hi52_j = float(_phj["High"].tail(252).max()) if len(_phj) >= 50 else None
                    lo52_j = float(_phj["Low"].tail(252).min())  if len(_phj) >= 50 else None
                    ytd_j  = _phj[_phj.index.year == _phj.index[-1].year]
                    ytd_ret_j = ((float(ytd_j["Close"].iloc[-1])/float(ytd_j["Close"].iloc[0]))-1)*100 if len(ytd_j) > 1 else None

                    # Earnings history
                    earn_j = []
                    try:
                        _ehj = _tkj.earnings_history
                        if _ehj is not None and not _ehj.empty:
                            for _, row in _ehj.tail(int(jpm_quarters)).iterrows():
                                est = row.get("epsEstimate") or row.get("EPS Estimate")
                                act = row.get("epsActual")   or row.get("Reported EPS")
                                sp  = row.get("surprisePercent")
                                bt  = None
                                try: bt = "BEAT" if float(act) >= float(est) else "MISS"
                                except: pass
                                earn_j.append({
                                    "date": str(row.get("Date", row.get("quarter","?"))),
                                    "est": est, "act": act, "sp": sp, "beat": bt,
                                })
                    except: pass

                    # Day moves
                    dr_j = {}
                    if not _phj.empty:
                        _clj = _phj["Close"].astype(float)
                        for dt, ret in ((_clj/_clj.shift(1)-1)*100).items():
                            dr_j[dt.date()] = float(ret)
                    edm_j = []
                    for er in earn_j:
                        try:
                            dp = pd.Timestamp(str(er["date"])).date()
                            for off in range(0, 4):
                                chk = dp + pd.Timedelta(days=off)
                                if chk in dr_j:
                                    edm_j.append({"date": er["date"], "beat": er["beat"], "move": round(dr_j[chk],2)})
                                    break
                        except: pass

                    avg_abs_j = sum(abs(m["move"]) for m in edm_j)/len(edm_j) if edm_j else None
                    up_j  = [m["move"] for m in edm_j if m["move"] > 0]
                    dn_j  = [m["move"] for m in edm_j if m["move"] < 0]

                    # Implied move
                    impl_j = None
                    exp_j  = None
                    try:
                        _exj = _tkj.options
                        if _exj and last_px_j:
                            exp_j = _exj[0]
                            _chj  = _tkj.option_chain(exp_j)
                            _cj   = _chj.calls
                            _pj   = _chj.puts
                            ac = _cj[(_cj["strike"] - float(last_px_j)).abs() <= float(last_px_j)*0.02]
                            ap = _pj[(_pj["strike"]  - float(last_px_j)).abs() <= float(last_px_j)*0.02]
                            if not ac.empty and not ap.empty:
                                cm = float(ac.iloc[0].get("lastPrice", ac.iloc[0].get("ask", 0)))
                                pm = float(ap.iloc[0].get("lastPrice", ap.iloc[0].get("ask", 0)))
                                impl_j = round((cm+pm)/float(last_px_j)*100, 2)
                    except: pass

                    eps_est_str_j = "N/A"
                    rev_est_str_j = "N/A"
                    try:
                        _eej = _tkj.earnings_estimate
                        if _eej is not None and not _eej.empty: eps_est_str_j = _eej.to_string()
                    except: pass
                    try:
                        _rej = _tkj.revenue_estimate
                        if _rej is not None and not _rej.empty: rev_est_str_j = _rej.to_string()
                    except: pass

                    # Pre-format all values using helper functions
                    px_s     = _jfmt(last_px_j)
                    mc_s     = _jfmt(mktcap_jpm)
                    hi52_s   = _jfmt(hi52_j)
                    lo52_s   = _jfmt(lo52_j)
                    ytd_s    = "{:+.1f}%".format(ytd_ret_j) if ytd_ret_j is not None else "N/A"
                    beta_s   = _jf2(beta_jpm)
                    pe_t_s   = _jf2(pe_trail_j)
                    pe_f_s   = _jf2(pe_fwd_j)
                    eps_t_s  = _jfmt(eps_t_j)
                    eps_f_s  = _jfmt(eps_f_j)
                    rev_s    = _jfmt(rev_j)
                    rg_s     = _jpct(rev_g_j)
                    gm_s     = _jpct(gm_j)
                    nm_s     = _jpct(nm_j)
                    tgt_s    = _jfmt(tgt_mean_j)
                    tgt_lo_s = _jfmt(tgt_lo_j)
                    tgt_hi_s = _jfmt(tgt_hi_j)
                    inst_s   = _jpct(inst_j)
                    sht_s    = _jpct(sht_j)
                    impl_s   = "+-{:.2f}%".format(impl_j) if impl_j else "N/A"
                    avg_mv_s = "{:.2f}%".format(avg_abs_j) if avg_abs_j else "N/A"
                    up_s     = "{:+.2f}% ({} times)".format(sum(up_j)/len(up_j), len(up_j)) if up_j else "N/A"
                    dn_s     = "{:+.2f}% ({} times)".format(sum(dn_j)/len(dn_j), len(dn_j)) if dn_j else "N/A"
                    upside_s = "N/A"
                    try:
                        if tgt_mean_j and last_px_j:
                            upside_s = "{:+.1f}%".format((float(tgt_mean_j)/float(last_px_j)-1)*100)
                    except: pass
                    impl_vs_hist = (
                        "Implied ABOVE hist avg - expensive options" if impl_j and avg_abs_j and impl_j > avg_abs_j
                        else "Implied BELOW hist avg - cheap options" if impl_j and avg_abs_j
                        else "N/A"
                    )
                    rec_s = rec_jpm.upper() if rec_jpm and rec_jpm != "N/A" else "N/A"

                    eh_txt = "\n".join(
                        "  {}: Est={} Actual={} Surprise={}% => {}".format(
                            r["date"], r["est"], r["act"], r["sp"], r["beat"])
                        for r in earn_j
                    ) or "  No earnings history available"

                    edm_txt = "\n".join(
                        "  {}: {} => {:+.2f}% on earnings day".format(m["date"], m["beat"], m["move"])
                        for m in edm_j
                    ) or "  No earnings day moves available"

                    snap_jpm = "\n".join([
                        "COMPANY: {} ({})  DATE: {}".format(company_jpm, jpm_ticker, last_date_j),
                        "SECTOR: {}".format(sector_jpm),
                        "",
                        "MARKET SNAPSHOT:",
                        "  Price: {}  Market Cap: {}  52w: {}-{}".format(px_s, mc_s, lo52_s, hi52_s),
                        "  YTD: {}  Beta: {}  Inst Own: {}  Short: {}".format(ytd_s, beta_s, inst_s, sht_s),
                        "",
                        "VALUATION:",
                        "  Trailing P/E: {}  Forward P/E: {}".format(pe_t_s, pe_f_s),
                        "  EPS Trailing: {}  EPS Forward: {}".format(eps_t_s, eps_f_s),
                        "  Revenue TTM: {}  Rev Growth: {}".format(rev_s, rg_s),
                        "  Gross Margin: {}  Net Margin: {}".format(gm_s, nm_s),
                        "",
                        "ANALYST CONSENSUS:",
                        "  Target: {} (range {}-{})  Upside: {}".format(tgt_s, tgt_lo_s, tgt_hi_s, upside_s),
                        "  Rating: {} ({} analysts)".format(rec_s, num_an_j),
                        "",
                        "EPS ESTIMATES:",
                        eps_est_str_j,
                        "",
                        "REVENUE ESTIMATES:",
                        rev_est_str_j,
                        "",
                        "EARNINGS HISTORY (last {} qtrs):".format(jpm_quarters),
                        eh_txt,
                        "",
                        "EARNINGS DAY MOVES:",
                        edm_txt,
                        "Avg Absolute Move: {}".format(avg_mv_s),
                        "Avg UP: {}  Avg DOWN: {}".format(up_s, dn_s),
                        "",
                        "OPTIONS IMPLIED MOVE:",
                        "  Nearest Expiry: {}  Straddle Implied: {}".format(exp_j or "N/A", impl_s),
                        "  {}".format(impl_vs_hist),
                    ])
                    st.success("Data loaded for {}. Generating JPMorgan analysis...".format(company_jpm))

                except Exception as _e_jpm:
                    st.error("Failed to load data: {}".format(_e_jpm))
                    st.code(_tb_jpm.format_exc())
                    st.stop()

            jpm_sys = """You are a Managing Director in Equity Research at JPMorgan Chase.
Format EXACTLY as a JPMorgan earnings note:
1. DECISION SUMMARY & TRADE PLAN
2. EARNINGS SCORECARD (beat/miss record, avg surprise)
3. CONSENSUS ESTIMATES (EPS and Revenue with context)
4. THE WHISPER NUMBER (what the market actually needs to rally - not just meet consensus)
5. KEY METRICS TO WATCH (3-5 specific numbers)
6. SEGMENT & REVENUE BREAKDOWN EXPECTATIONS
7. MANAGEMENT GUIDANCE CREDIBILITY
8. OPTIONS MARKET ANALYSIS (implied move, straddle, skew)
9. HISTORICAL EARNINGS DAY PATTERNS
10. PRE-EARNINGS POSITIONING (buy/sell/wait with rationale and levels)
11. POST-EARNINGS PLAYBOOK (gap up / gap down / flat - specific levels for each)
12. RISK FACTORS  13. DISCLAIMER
Use exact numbers from the data. Be opinionated. Specify exact price levels."""

            mode_map_j = {
                "Pre-Earnings Preview": "Focus on Whisper (4), Key Metrics (5), Pre-Positioning (10), Playbook (11).",
                "Post-Earnings Reaction": "Earnings just reported. Focus entirely on Post-Earnings Playbook (11). Is the reaction justified or overdone?",
                "Full Earnings Dossier": "Write the complete note covering all 13 sections.",
            }
            extra_jpm = ("Context: " + jpm_extra) if jpm_extra and jpm_extra.strip() else ""
            user_msg_j = "\n".join([
                "Earnings data for {}:".format(jpm_ticker),
                "",
                snap_jpm,
                "",
                "Report: {}".format(jpm_mode),
                mode_map_j.get(jpm_mode, ""),
                extra_jpm,
                "",
                "Generate the JPMorgan earnings note. Be specific about price levels and the whisper number.",
            ])

            with st.spinner("Generating JPMorgan earnings analysis..."):
                try:
                    _jpm_prov_live = st.session_state.get("jpm_provider", "OpenAI")
                    _jpm_key_live = (os.getenv("OPENAI_API_KEY","") if _jpm_prov_live=="OpenAI" else os.getenv("ANTHROPIC_API_KEY","")).strip()
                    _jpm_key_field = st.session_state.get(f"jpm_api_key_{_jpm_prov_live}", "").strip()
                    if _jpm_key_field: _jpm_key_live = _jpm_key_field
                    _jpm_mdl_live = st.session_state.get("jpm_model", jpm_model)
                    if not _jpm_key_live:
                        st.error(f"No {_jpm_prov_live} key found — set OPENAI_API_KEY or ANTHROPIC_API_KEY in .env"); st.stop()
                    st.caption(f"Provider: {_jpm_prov_live} | Model: {_jpm_mdl_live} | Key: ...{_jpm_key_live[-4:]}") 
                    _out_jpm = _call_llm(_jpm_prov_live, _jpm_key_live, _jpm_mdl_live, jpm_sys, user_msg_j, max_tokens=2000)
                    if not _out_jpm:
                        st.error("No response returned.")
                    else:
                        hdr_jpm = (
                            "<div style='background:#003087;padding:10px 20px;border-radius:5px;margin-bottom:12px'>"
                            "<span style='color:#fff;font-weight:700'>JPMORGAN EQUITY RESEARCH | EARNINGS | "
                            + company_jpm + " (" + jpm_ticker + ") | " + last_date_j + "</span></div>"
                        )
                        st.markdown(hdr_jpm, unsafe_allow_html=True)
                        jk1, jk2, jk3, jk4, jk5 = st.columns(5)
                        jk1.metric("Price", px_s)
                        jk2.metric("Implied Move", impl_s)
                        jk3.metric("Hist Avg Move", avg_mv_s)
                        jk4.metric("Analyst Target", tgt_s, delta=upside_s if upside_s != "N/A" else None)
                        jk5.metric("Rating", rec_s)
                        st.markdown(_out_jpm)
                        with st.expander("Raw data", expanded=False):
                            st.code(snap_jpm, language="text")
                        dl_j = "JPMORGAN EQUITY RESEARCH\n{} ({}) | {}\n\n{}\n\n--- DATA ---\n{}".format(
                            company_jpm, jpm_ticker, last_date_j, _out_jpm, snap_jpm)
                        st.download_button("Download as text", data=dl_j,
                            file_name="JPM_Earnings_{}_{}.txt".format(jpm_ticker, last_date_j),
                            mime="text/plain")
                except Exception as _e_jpm2:
                    st.error("Claude API call failed: {}".format(_e_jpm2))
                    st.code(_tb_jpm.format_exc())

# ─────────────────────────────────────────────────────────────────────────────
# SENTIMENT TAB  (FinBERT news sentiment scanner)
# ─────────────────────────────────────────────────────────────────────────────
if nav == 'Sentiment':
    import glob as _sent_glob

    st.subheader("\U0001f9e0 FinBERT News Sentiment Scanner")

    if not _HAS_FINBERT:
        st.warning(
            "FinBERT is not available. Install with:\n\n"
            "```\npip install transformers torch\n```\n\n"
            "Then restart the app."
        )
        st.stop()

    # ── Universe selector ──────────────────────────────────────────────────
    _sent_c1, _sent_c2, _sent_c3 = st.columns([2, 1, 1])
    with _sent_c1:
        _sent_univ = st.selectbox(
            "Universe",
            ["Beta Universe", "NDX 100", "SPX 150", "Custom List"],
            key="sent_univ",
        )
    with _sent_c2:
        _sent_limit = st.number_input(
            "Headlines per ticker", min_value=3, max_value=25, value=10, key="sent_limit"
        )
    with _sent_c3:
        _sent_sort = st.selectbox(
            "Sort by",
            ["Most Bullish", "Most Bearish", "Abs Sentiment", "Ticker"],
            key="sent_sort",
        )

    # Resolve universe tickers
    if _sent_univ == "NDX 100":
        _sent_tickers = list(_TF_NDX100)
    elif _sent_univ == "SPX 150":
        _sent_tickers = list(_TF_SPX_LIQUID)
    elif _sent_univ == "Custom List":
        _sent_custom = st.text_input(
            "Enter tickers (comma separated)", key="sent_custom_list"
        )
        _sent_tickers = [t.strip().upper() for t in _sent_custom.split(",") if t.strip()] if _sent_custom else []
    else:  # Beta Universe
        _pq_dir_sent = os.environ.get("PER_TICKER_PARQUET_DIR", "")
        if _pq_dir_sent:
            _sent_tickers = sorted(
                os.path.splitext(os.path.basename(f))[0]
                for f in _sent_glob.glob(os.path.join(_pq_dir_sent, "*.parquet"))
                if not os.path.basename(f).startswith("_")
            )
        else:
            _sent_tickers = list(_TF_NDX100)

    st.caption(f"{len(_sent_tickers)} tickers in universe")

    _sent_run = st.button("\u26a1 Scan Sentiment", type="primary", key="sent_run_btn")

    if _sent_run and _sent_tickers:
        _sent_results = []
        _sent_prog = st.progress(0, text="Scoring sentiment...")
        for _si, _stk in enumerate(_sent_tickers):
            _sent_prog.progress((_si + 1) / len(_sent_tickers), text=f"Scoring {_stk}...")
            try:
                _sr = _tf_news_sentiment(_stk, limit=_sent_limit)
                if _sr["count"] > 0:
                    _sent_results.append({
                        "Ticker":     _stk,
                        "Score":      _sr["score"],
                        "Label":      _sr["label"],
                        "Headlines":  _sr["count"],
                        "Top":        _sr["headlines"][0]["title"][:60] if _sr["headlines"] else "",
                        "_data":      _sr,
                    })
            except Exception:
                pass
        _sent_prog.empty()
        st.session_state["sent_results"] = _sent_results

    # ── Display results ────────────────────────────────────────────────────
    _sent_data = st.session_state.get("sent_results", [])
    if _sent_data:
        if _sent_sort == "Most Bullish":
            _sent_data = sorted(_sent_data, key=lambda x: x["Score"], reverse=True)
        elif _sent_sort == "Most Bearish":
            _sent_data = sorted(_sent_data, key=lambda x: x["Score"])
        elif _sent_sort == "Abs Sentiment":
            _sent_data = sorted(_sent_data, key=lambda x: abs(x["Score"]), reverse=True)
        else:
            _sent_data = sorted(_sent_data, key=lambda x: x["Ticker"])

        _sent_df = pd.DataFrame([{
            "Ticker":    r["Ticker"],
            "Sentiment": r["Score"],
            "Label":     r["Label"],
            "Headlines": r["Headlines"],
            "Top Headline": r["Top"],
        } for r in _sent_data])

        st.dataframe(
            _sent_df.style.background_gradient(
                subset=["Sentiment"],
                cmap="RdYlGn",
                vmin=-1.0,
                vmax=1.0,
            ),
            use_container_width=True,
            height=min(35 * len(_sent_df) + 38, 700),
        )

        _sent_csv = _sent_df.to_csv(index=False)
        st.download_button(
            "Download CSV", _sent_csv,
            file_name="sentiment_scan.csv", mime="text/csv"
        )

        # ── Drill-down ──────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### Headline Detail")
        _sent_drill = st.selectbox(
            "Select ticker to view headlines",
            [r["Ticker"] for r in _sent_data],
            key="sent_drill_tkr",
        )
        if _sent_drill:
            _sd_match = [r for r in _sent_data if r["Ticker"] == _sent_drill]
            if _sd_match:
                _sd = _sd_match[0]["_data"]
                st.metric("Overall Sentiment", f"{_sd['label'].title()} ({_sd['score']:+.3f})")
                for _h in _sd.get("headlines", []):
                    _h_color = (
                        "#2e7d32" if _h["sentiment"] == "positive"
                        else "#c62828" if _h["sentiment"] == "negative"
                        else "#757575"
                    )
                    st.markdown(
                        f"<div style='padding:6px 12px;margin:4px 0;"
                        f"border-left:4px solid {_h_color};background:#1e1e1e;"
                        f"border-radius:4px'>"
                        f"<span style='color:{_h_color};font-weight:600'>"
                        f"{_h['sentiment'].upper()} ({_h['confidence']:.0%})</span>"
                        f"&nbsp;&nbsp;{_h['title']}</div>",
                        unsafe_allow_html=True,
                    )
    elif st.session_state.get("sent_results") is not None:
        st.info("No headlines found for the selected universe.")

# ─────────────────────────────────────────────────────────────────────────────
# TRADE FINDER TAB  (appended inline into app18.py)
# ─────────────────────────────────────────────────────────────────────────────
if nav == 'Trade Finder':
    import plotly.graph_objects as _go_tf
    import plotly.subplots as _psp_tf
    import glob as _glob_tf

    st.subheader("🎯 Trade Finder")

    # ── 0. helpers local to this tab ─────────────────────────────────────────
    def _tf_regime(vix: float | None, spy_chg: float | None) -> tuple[str, str]:
        """Return (label, css_color) for current market regime."""
        if vix is None:
            return "Unknown", "#888"
        if vix < 18:
            lbl = "Risk-On 🟢"
            col = "#00c853"
        elif vix < 25:
            lbl = "Elevated ⚠️"
            col = "#ff9800"
        else:
            lbl = "Risk-Off 🔴"
            col = "#e53935"
        if spy_chg is not None:
            if spy_chg <= -1.5:
                lbl += " (Selling)"
                col = "#e53935"
            elif spy_chg >= 1.5:
                lbl += " (Rally)"
        return lbl, col

    def _tf_get_all_parquet_tickers() -> list[str]:
        """Return all ticker names found in PER_TICKER_PARQUET_DIR."""
        pq_dir = os.environ.get("PER_TICKER_PARQUET_DIR", "")
        if not pq_dir or not os.path.isdir(pq_dir):
            return []
        files = sorted(_glob_tf.glob(os.path.join(pq_dir, "*.parquet")))
        return [os.path.splitext(os.path.basename(f))[0].upper() for f in files]

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 1 — MARKET PULSE
    # ─────────────────────────────────────────────────────────────────────────
    with st.expander("📡 Market Pulse", expanded=True):
        if st.button("Refresh Market Pulse", key="tf_refresh_pulse"):
            st.cache_data.clear()

        _macro_snap = _tf_snapshot_batch(tuple(_TF_MACRO_TICKERS))
        _vix_val    = _tf_vix()

        # SPY data for regime
        _spy_snap = _macro_snap.get("SPY") or {}
        _spy_chg  = _spy_snap.get("chg_pct")

        regime_lbl, regime_col = _tf_regime(_vix_val, _spy_chg)

        # ── Fetch supplemental data ───────────────────────────────────────────
        _idx_px   = _tf_index_prices()
        _cmdty_px = _tf_commodity_spot()

        # ─── Compact Market Pulse: 2 rows, no empty cards ────────────────────
        # Row 1: Regime | VIX | S&P | NDX | RTY | TLT | DXY
        _mp_r1 = st.columns([1.5, 1, 1, 1, 1, 1, 1])
        # Regime badge
        _mp_r1[0].markdown(
            f"<div style='background:{regime_col};padding:4px 10px;border-radius:5px;"
            f"font-size:0.85rem;font-weight:700;color:#fff;display:inline-block'>"
            f"Regime: {regime_lbl}</div>",
            unsafe_allow_html=True,
        )
        if _vix_val is not None:
            _mp_r1[0].metric("VIX", f"{_vix_val:.2f}")
        # Equity indices — index price preferred, ETF*10 fallback for SPX
        _ix_map = [("S&P", "I:SPX", "SPY"), ("NDX", "I:NDX", "QQQ"), ("RTY", "I:RUT", "IWM")]
        for _ci, (_lbl, _ix_t, _etf_t) in enumerate(_ix_map, 1):
            _ip = _idx_px.get(_ix_t) or {}
            _ep = _macro_snap.get(_etf_t) or {}
            if not _ip.get("price") and _ix_t == "I:SPX" and _ep.get("price"):
                _ip = {"price": float(_ep["price"]) * 10.0, "chg_pct": float(_ep.get("chg_pct", 0))}
            _px = _ip.get("price") or (_ep.get("price"))
            _delt = f"{_ip.get('chg_pct', _ep.get('chg_pct',0)):+.2f}%" if _px else None
            _mp_r1[_ci].metric(_lbl, f"{_px:,.0f}" if _px else chr(0x2014), delta=_delt)
        # TLT
        _et_tlt = _macro_snap.get("TLT") or {}
        _pt = _et_tlt.get("price")
        _mp_r1[4].metric("TLT", f"${_pt:.2f}" if _pt else chr(0x2014),
                          delta=f"{_et_tlt.get('chg_pct',0):+.2f}%" if _pt else None)
        # DXY: index preferred, UUP ETF fallback (show as ETF price, not index level)
        _ix_dxy = _idx_px.get("I:DXY") or {}
        _et_uup = _macro_snap.get("UUP") or {}
        _dxy = _ix_dxy.get("price")
        if _dxy:
            _mp_r1[5].metric("DXY", f"{_dxy:.2f}",
                              delta=f"{_ix_dxy.get('chg_pct',0):+.2f}%")
        elif _et_uup.get("price"):
            _mp_r1[5].metric("Dollar(UUP)", f"${float(_et_uup['price']):.2f}",
                              delta=f"{_et_uup.get('chg_pct',0):+.2f}%")
        # VIX already shown with regime
        _mp_r1[6].markdown("")  # spacer

        # Row 2: USD/JPY | EUR/USD | Gold | Silver | WTI | Copper
        _mp_r2 = st.columns(6)
        # FX: Polygon forex preferred, ETF fallback
        _fx_usdjpy = _cmdty_px.get("USDJPY") or {}
        _fx_eurusd = _cmdty_px.get("EURUSD") or {}
        _et_fxy = _macro_snap.get("FXY") or {}
        _et_fxe = _macro_snap.get("FXE") or {}
        _pj = _fx_usdjpy.get("price")
        if _pj:
            _mp_r2[0].metric("USD/JPY", f"{_pj:.2f}",
                              delta=f"{_fx_usdjpy.get('chg_pct',0):+.2f}%")
        elif _et_fxy.get("price"):
            _mp_r2[0].metric("Yen(FXY)", f"${float(_et_fxy['price']):.2f}",
                              delta=f"{_et_fxy.get('chg_pct',0):+.2f}%")
        _pe = _fx_eurusd.get("price")
        if _pe:
            _mp_r2[1].metric("EUR/USD", f"{_pe:.4f}",
                              delta=f"{_fx_eurusd.get('chg_pct',0):+.2f}%")
        elif _et_fxe.get("price"):
            _mp_r2[1].metric("Euro(FXE)", f"${float(_et_fxe['price']):.2f}",
                              delta=f"{_et_fxe.get('chg_pct',0):+.2f}%")
        # Commodities: Polygon spot preferred, ETF fallback
        _cm_items = [
            ("Gold",   "GOLD",   "GLD",  "${:,.0f}", "${:.2f}"),
            ("Silver", "SILVER", "SLV",  "${:,.2f}", "${:.2f}"),
            ("WTI",    "OIL",    "USO",  "${:,.2f}", "${:.2f}"),
            ("Copper", "COPPER", "CPER", "${:,.3f}", "${:.2f}"),
        ]
        for _ci2, (_lbl2, _ck2, _etf2, _fmt_spot, _fmt_etf) in enumerate(_cm_items, 2):
            _sp2 = _cmdty_px.get(_ck2) or {}
            _et2 = _macro_snap.get(_etf2) or {}
            _px2 = _sp2.get("price")
            if _px2:
                _mp_r2[_ci2].metric(_lbl2, _fmt_spot.format(_px2),
                                      delta=f"{_sp2.get('chg_pct',0):+.2f}%")
            elif _et2.get("price"):
                _mp_r2[_ci2].metric(f"{_lbl2}({_etf2})", _fmt_etf.format(float(_et2["price"])),
                                      delta=f"{_et2.get('chg_pct',0):+.2f}%")
            else:
                _mp_r2[_ci2].metric(_lbl2, chr(0x2014))

        # ── Row 3: Prediction Markets ──
        st.markdown("**Prediction Markets**")
        _pm_fin_kw = ["fed", "rate", "recession", "inflation", "cpi", "tariff",
                       "s&p", "spx", "treasury", "gdp", "mortgage", "ecb",
                       "oil", "gold", "interest"]
        _pm_pulse_mkts = _pm_fetch_markets(limit=200)
        _pm_fin = []
        for _pmm in _pm_pulse_mkts:
            _pmq = ((_pmm.get("question") or "") + " " + (_pmm.get("slug") or "")).lower()
            if any(_fk in _pmq for _fk in _pm_fin_kw):
                _pm_fin.append(_pmm)
            if len(_pm_fin) >= 6:
                break
        if _pm_fin:
            _pm_pc = st.columns(len(_pm_fin))
            for _pci, _pcm in enumerate(_pm_fin):
                _pc_price = _pm_safe_float(_pcm.get("lastTradePrice"))
                _pc_chg = _pm_safe_float(_pcm.get("oneWeekPriceChange"))
                _pc_title = (_pcm.get("question") or _pcm.get("groupItemTitle") or "")[:30]
                _pm_pc[_pci].metric(
                    _pc_title,
                    f"{_pc_price:.0%}" if _pc_price else "\u2014",
                    delta=f"{_pc_chg:+.1%} 1w" if _pc_chg else None,
                    help=(_pcm.get("question") or ""),
                )
        else:
            st.caption("Polymarket API unavailable")


    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 2 — SCANNER SETUP
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("### Scanner Setup")
    _sc1, _sc2, _sc3 = st.columns([2, 2, 3])

    with _sc1:
        _tf_universe = st.selectbox(
            "Universe",
            ["Beta Universe", "NDX 100", "SPX 150", "Watchlist", "All Parquets"],
            key="tf_universe",
        )
        _tf_min_score = st.slider("Min Score", 0, 100, 60, 5, key="tf_min_score")
        _tf_min_align = st.selectbox("Min Alignment", [0, 1, 2, 3], index=2, key="tf_min_align")

    with _sc2:
        _tf_min_rvol = st.number_input("Min Rel Volume", 0.0, 10.0, 0.9, 0.1, key="tf_min_rvol")
        _tf_max_rsi  = st.slider("Max RSI", 30, 100, 72, 1, key="tf_max_rsi")
        _tf_fetch_opts = st.checkbox("Fetch Options Data (slower)", value=False, key="tf_fetch_opts")

    with _sc3:
        st.markdown("**Setup Filters** (leave all off = show any)")
        _tf_filt_ignition  = st.checkbox("★ Stack Ignition",  key="tf_f_ignition",
            help="All 3 MA slopes (1d/5d/10d) just flipped positive TOGETHER for the first time "
                 "on above-avg volume (≥1.3×). Early-stage momentum ignition — the MAs are "
                 "just waking up. Highest-conviction early entry.")
        _tf_filt_fullstack = st.checkbox("⬆ Full Stack",       key="tf_f_fullstack",
            help="All 3 MA slopes (1d/5d/10d) are positive and have been for at least 1 bar. "
                 "Ticker is in a confirmed, sustained uptrend. Best for momentum continuation / "
                 "adding to winners.")
        _tf_filt_pullback  = st.checkbox("↩ Pullback-to-MA",  key="tf_f_pullback",
            help="Price has pulled back to within −1% to +2.5% of the rising SMA10, with RSI ≤58. "
                 "Classic buy-the-dip-in-an-uptrend setup — MA alignment ≥2, trend intact, "
                 "waiting for a bounce.")
        _tf_filt_volspike  = st.checkbox("⚡ Vol Spike",        key="tf_f_volspike",
            help="Today's volume is 2.5× or more above the 20-day average. Unusual institutional "
                 "interest — often precedes a directional move. Works for both breakouts and "
                 "reversals depending on price context.")
        _tf_filt_rs        = st.checkbox("🚀 RS Emerging",      key="tf_f_rs",
            help="MA alignment just reached 2+ after ≥7 consecutive days of flat or no alignment. "
                 "Ticker is breaking out of dormancy and showing early relative strength vs. the "
                 "market. Catch it before the crowd notices.")
        _tf_filt_break     = st.checkbox("⚠ Stack Break",       key="tf_f_break",
            help="The 1-day MA slope just went negative while the 5d and 10d slopes are still "
                 "positive — uptrend is cracking at the short end. Use as an AVOID or short-side "
                 "alert. Not a buy signal.")

    # Watchlist input (only shown if Watchlist selected)
    if _tf_universe == "Watchlist":
        _wl_default = st.session_state.get("tf_watchlist_raw", "AAPL,NVDA,MSFT,META,TSLA,AMD,MU,SNDK")
        _wl_raw = st.text_area(
            "Watchlist Tickers (comma-separated)",
            value=_wl_default,
            height=80,
            key="tf_watchlist_input",
        )
        st.session_state["tf_watchlist_raw"] = _wl_raw

    # Build universe
    def _tf_build_universe() -> list[str]:
        u = st.session_state.get("tf_universe", "Beta Universe")
        if u == "Beta Universe":
            _bu = _tf_load_beta_universe()
            return _bu if _bu else list(dict.fromkeys(_TF_NDX100))
        if u == "NDX 100":
            return list(dict.fromkeys(_TF_NDX100))
        if u == "SPX 150":
            return list(dict.fromkeys(_TF_SPX_LIQUID))
        if u == "Watchlist":
            raw = st.session_state.get("tf_watchlist_raw", "")
            return [t.strip().upper() for t in raw.replace("\n", ",").split(",") if t.strip()]
        # All Parquets
        return _tf_get_all_parquet_tickers()

    # Active setup filters
    def _tf_active_filters() -> set[str]:
        mapping = {
            "tf_f_ignition":  "★ Stack Ignition",
            "tf_f_fullstack": "⬆ Full Stack",
            "tf_f_pullback":  "↩ Pullback-to-MA",
            "tf_f_volspike":  "⚡",
            "tf_f_rs":        "🚀 RS Emerging",
            "tf_f_break":     "⚠ Stack Break",
        }
        active = set()
        for k, label in mapping.items():
            if st.session_state.get(k, False):
                active.add(label)
        return active

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 3 — RUN SCAN
    # ─────────────────────────────────────────────────────────────────────────
    _tf_run = st.button("🔍 Run Scan", type="primary", key="tf_run_scan")

    if _tf_run:
        _universe = _tf_build_universe()
        if not _universe:
            st.error("No tickers in universe — check PER_TICKER_PARQUET_DIR or watchlist.")
        else:
            # ── Fetch live intraday snapshot from Polygon for entire universe ──
            _today_norm = pd.Timestamp.today().normalize()
            _live_snap = {}
            _snap_status = st.empty()
            _snap_status.info("Fetching live Polygon snapshots…")
            for _ci in range(0, len(_universe), 250):
                _chunk = tuple(_universe[_ci:_ci + 250])
                _live_snap.update(_tf_snapshot_batch(_chunk))
            _n_live = sum(1 for v in _live_snap.values() if v.get("price"))
            _snap_status.success(f"✅ Live data: {_n_live}/{len(_universe)} tickers from Polygon")

            def _tf_append_live(df, tkr):
                """Append or overwrite today's bar with live Polygon snapshot."""
                snap = _live_snap.get(tkr)
                if snap is None or not snap.get("price"):
                    return df
                _dates = pd.to_datetime(df["Date"])
                _has_today = (_dates.dt.normalize() == _today_norm).any()
                if _has_today:
                    # Overwrite today's row with live intraday data
                    _mask = _dates.dt.normalize() == _today_norm
                    df = df.copy()
                    df.loc[_mask, "Close"]  = snap["price"]
                    df.loc[_mask, "High"]   = max(snap.get("high") or snap["price"],
                                                  float(df.loc[_mask, "High"].iloc[0]))
                    df.loc[_mask, "Low"]    = min(snap.get("low") or snap["price"],
                                                  float(df.loc[_mask, "Low"].iloc[0]))
                    df.loc[_mask, "Volume"] = snap.get("volume") or float(df.loc[_mask, "Volume"].iloc[0])
                    return df
                new_row = {
                    "Date":   _today_norm,
                    "Open":   snap.get("open") or snap["price"],
                    "High":   snap.get("high") or snap["price"],
                    "Low":    snap.get("low")  or snap["price"],
                    "Close":  snap["price"],
                    "Volume": snap.get("volume") or 0,
                }
                return pd.concat([df, pd.DataFrame([new_row])],
                                 ignore_index=True)

            # SPY 20d return for RS scoring
            _spy_df = _load_daily_df("SPY")
            if _spy_df is not None:
                _spy_df = _tf_append_live(_spy_df, "SPY")
            if _spy_df is not None and len(_spy_df) >= 21:
                _spy_close = pd.to_numeric(_spy_df["Close"], errors="coerce")
                _spy_ret20 = float((_spy_close.iloc[-1] / _spy_close.iloc[-21] - 1) * 100)
            else:
                _spy_ret20 = 0.0

            _results = []
            _flow_all = []          # collect unusual options flow across all tickers
            _prog = st.progress(0.0, text="Scanning…")
            _n = len(_universe)
            for _idx, _tkr in enumerate(_universe):
                _prog.progress((_idx + 1) / _n, text=f"Scanning {_tkr} ({_idx+1}/{_n})")
                _sdf = _load_daily_df(_tkr)
                if _sdf is not None:
                    _sdf = _tf_append_live(_sdf, _tkr)
                if _sdf is None or len(_sdf) < 22:
                    continue
                # ── Staleness gate: skip delisted/acquired tickers ─────────────
                # If last bar is >10 calendar days old the stock likely no longer trades
                try:
                    _last_bar_dt = pd.to_datetime(_sdf["Date"]).max()
                    if (_today_norm - pd.Timestamp(_last_bar_dt).normalize()).days > 10:
                        continue
                except Exception:
                    pass
                # ─────────────────────────────────────────────────────────────────
                # ── Collect flow data BEFORE score filter (all tickers) ───────
                _opts = {}
                if st.session_state.get("tf_fetch_opts", False):
                    try:
                        _opts = _tf_options_summary(_tkr)
                    except Exception:
                        pass
                    _fc_list = _opts.get("_contracts", [])
                    _fc_spot = _opts.get("spot")
                    if _fc_list and _fc_spot:
                        for _fc in _fc_list:
                            _fc_vol = _fc.get("vol", 0)
                            _fc_oi  = _fc.get("oi", 0)
                            _fc_mid = _fc.get("mid", 0)
                            _fc_voi = round(_fc_vol / _fc_oi, 1) if _fc_oi > 0 else 0.0
                            if _fc_voi < 1.0 or _fc_vol < 100:
                                continue
                            _fc_strike = _fc.get("strike", 0)
                            _fc_dist = round((_fc_strike / _fc_spot - 1) * 100, 1)
                            if abs(_fc_dist) < 2.0:
                                _fc_money = "ATM"
                            elif (_fc["type"] == "call" and _fc_strike > _fc_spot) or (_fc["type"] == "put" and _fc_strike < _fc_spot):
                                _fc_money = "OTM"
                            else:
                                _fc_money = "ITM"
                            _flow_all.append({
                                "Ticker": _tkr,
                                "C/P": _fc["type"][0].upper(),
                                "Strike": _fc_strike,
                                "Expiry": _fc.get("expiry", ""),
                                "Vol": _fc_vol,
                                "OI": _fc_oi,
                                "Vol/OI": _fc_voi,
                                "Mid": round(_fc_mid, 2),
                                "Est $Prem": round(_fc_vol * _fc_mid * 100),
                                "IV": round(_fc.get("iv", 0) * 100, 1),
                                "Delta": round(float(_fc.get("delta")), 2) if _fc.get("delta") is not None else None,
                                "Dist%": _fc_dist,
                                "Moneyness": _fc_money,
                                "Spot": round(_fc_spot, 2),
                            })
                # ── Score filter — only scored tickers make the scorecard ─────
                _sent = _tf_news_sentiment(_tkr, limit=10)
                _score, _comps = _tf_score_ticker(_sdf, _spy_ret20, _sent.get('score', 0.0))
                if _score < st.session_state.get("tf_min_score", 60):
                    continue
                if _comps.get("align_score", 0) < st.session_state.get("tf_min_align", 2):
                    continue
                if _comps.get("rel_vol", 1.0) < st.session_state.get("tf_min_rvol", 0.9):
                    continue
                if _comps.get("rsi", 50) > st.session_state.get("tf_max_rsi", 72):
                    continue
                _setups = _tf_detect_setups(_sdf)
                # Setup filter
                _active_f = _tf_active_filters()
                if _active_f:
                    if not any(any(fk in s for s in _setups) for fk in _active_f):
                        continue
                _px_last = float(pd.to_numeric(_sdf["Close"], errors="coerce").iloc[-1])
                _results.append({
                    "Ticker":         _tkr,
                    "Price":          round(_px_last, 2),
                    "Score":          _score,
                    "Slope(0-25)":    _comps.get("Slope(0-25)", 0),
                    "RS(0-20)":       _comps.get("RS_SPY(0-20)", 0),
                    "RVol(0-20)":     _comps.get("RelVol(0-20)", 0),
                    "Entry(0-20)":    _comps.get("Entry(0-20)", 0),
                    "Momentum(0-15)": _comps.get("Momentum(0-15)", 0),
                    "Sent(0-9)":      _comps.get("Sentiment(0-9)", 0),
                    "SentLabel":      _sent.get("label", "neutral"),
                    "Align":          _comps.get("align_score", 0),
                    "RelVol":         _comps.get("rel_vol", 1.0),
                    "RSI":            _comps.get("rsi", 50),
                    "Pull%":          _comps.get("pull_pct", 0),
                    "Ret20d%":        _comps.get("ret_20d", 0),
                    "Setups":         ", ".join(_setups) if _setups else "—",
                    "ATM_IV":         _opts.get("atm_iv"),
                    "PC_Ratio":       _opts.get("pc_ratio"),
                    "Skew25d":        _opts.get("skew_25d"),
                })
            _prog.empty()
            _results.sort(key=lambda x: x["Score"], reverse=True)
            st.session_state["tf_results"] = _results[:60]
            st.session_state["tf_spy_ret20"] = _spy_ret20
            # Store flow data (sorted by Est $Prem descending)
            _flow_all.sort(key=lambda x: x.get("Est $Prem", 0), reverse=True)
            st.session_state["tf_flow_data"] = _flow_all[:500]
            # Persist flow to rolling history parquet
            if _flow_all:
                try:
                    _sm_save_flow(_flow_all)
                except Exception:
                    pass
            _flow_msg = f" | Flow: {len(_flow_all)} contracts across {len({f['Ticker'] for f in _flow_all})} tickers" if _flow_all else ""
            st.success(f"Scan complete — {len(_results)} tickers scored ≥ {st.session_state.get('tf_min_score',60)}{_flow_msg}")

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 4 — SCORECARD
    # ─────────────────────────────────────────────────────────────────────────
    _tf_results = st.session_state.get("tf_results", [])
    if _tf_results:
        st.markdown("### Scorecard — Top Results")
        with st.expander("Column legend", expanded=False):
            st.markdown(
                "| Column | Range | Meaning |\n"
                "|---|---|---|\n"
                "| **Score** | 0\u2013109 | Composite total (sum of all 6 components) |\n"
                "| **Slope** | 0/5/15/25 | SMA-10 slope alignment \u2014 how many of 1d/5d/10d slopes positive |\n"
                "| **RS** | 0\u201320 | Relative strength vs SPY (20d). 8 = matching SPY |\n"
                "| **RVol** | 0\u201320 | Volume vs 20-day avg. 0 = below avg, 20 = 1.5\u00d7+ |\n"
                "| **Entry** | 0\u201320 | Price near rising MA + RSI 35\u201365 sweet spot |\n"
                "| **Momentum** | 0\u201315 | 5-day return tiers: 15 = \u22653%, 10 = \u22651.5%, 6 = \u22650.5% |\n"
                "| **Sent** | 0\u20139 | FinBERT news sentiment. 4.5 = neutral, 9 = bullish, 0 = bearish |\n"
                "| **SentLabel** | text | positive / neutral / negative |\n"
                "| **Align** | 0\u20133 | Raw slope alignment count |\n"
                "| **RelVol** | ratio | Raw relative volume (1.0 = average) |\n"
                "| **RSI** | 0\u2013100 | RSI(14). <30 oversold, >70 overbought |\n"
                "| **Pull%** | % | Distance from SMA-10. Negative = dip below MA |\n"
                "| **Ret20d%** | % | 20-day price return |"
            )
        _df_sc = pd.DataFrame(_tf_results)

        # Color score column
        def _score_color(val):
            if val >= 70:
                return "background-color:#1b5e20;color:#fff"
            if val >= 55:
                return "background-color:#2e7d32;color:#fff"
            if val >= 40:
                return "background-color:#f57f17;color:#000"
            return ""

        # Format display
        _display_cols = [c for c in [
            "Ticker","Price","Score","Slope(0-25)","RS(0-20)","RVol(0-20)","Entry(0-20)","Momentum(0-15)","Sent(0-9)",
            "SentLabel","Align","RelVol","RSI","Pull%","Ret20d%","Setups","ATM_IV","PC_Ratio","Skew25d"
        ] if c in _df_sc.columns]

        _df_display = _df_sc[_display_cols].copy()
        # Format numeric columns
        for _col in ["Score","Slope(0-25)","RS(0-20)","RVol(0-20)","Entry(0-20)","Momentum(0-15)","Sent(0-9)",
                     "RelVol","RSI","Pull%","Ret20d%","ATM_IV","PC_Ratio","Skew25d"]:
            if _col in _df_display.columns:
                _df_display[_col] = pd.to_numeric(_df_display[_col], errors="coerce").round(1)

        st.caption("💡 Click any row to drill down into that ticker")
        _sc_sel = st.dataframe(
            _df_display.style.applymap(_score_color, subset=["Score"]),
            use_container_width=True,
            height=400,
            on_select="rerun",
            selection_mode="single-row",
            key="tf_scorecard_table",
        )
        # Store clicked ticker in session state for the Drill-Down section
        _sc_rows = _sc_sel.selection.get("rows", []) if _sc_sel and hasattr(_sc_sel, "selection") else []
        if _sc_rows:
            _clicked_tkr = str(_df_display.iloc[_sc_rows[0]]["Ticker"])
            st.session_state["tf_drill_select"] = _clicked_tkr

        # Download button
        _csv_sc = _df_display.to_csv(index=False)
        st.download_button(
            "⬇ Download CSV",
            data=_csv_sc,
            file_name="trade_finder_results.csv",
            mime="text/csv",
            key="tf_dl_csv",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 4b — FLOW SCANNER (aggregated unusual options activity)
    # ─────────────────────────────────────────────────────────────────────────
    _tf_flow = st.session_state.get("tf_flow_data", [])
    if _tf_flow:
        with st.expander("**Flow Scanner** — unusual options activity across scanned universe", expanded=False):
            _fdf = pd.DataFrame(_tf_flow)

            # Summary bar: total call vs put premium
            _fs_call_prem = _fdf[_fdf["C/P"] == "C"]["Est $Prem"].sum()
            _fs_put_prem  = _fdf[_fdf["C/P"] == "P"]["Est $Prem"].sum()
            _fs_total     = _fs_call_prem + _fs_put_prem
            _fs_bias = "Bullish" if _fs_call_prem > _fs_put_prem else ("Bearish" if _fs_put_prem > _fs_call_prem else "Neutral")
            _fs_color = "#00c853" if _fs_bias == "Bullish" else ("#e53935" if _fs_bias == "Bearish" else "#888")
            _fs_tickers = _fdf["Ticker"].nunique()
            _fs_contracts = len(_fdf)

            _fsc1, _fsc2, _fsc3, _fsc4 = st.columns(4)
            _fsc1.markdown(
                f"<div style='font-size:.85rem;color:#888'>Market Flow Bias</div>"
                f"<div style='font-size:1.5rem;font-weight:700;color:{_fs_color}'>{_fs_bias}</div>",
                unsafe_allow_html=True,
            )
            _fsc2.metric("Call Premium", f"${_fs_call_prem:,.0f}")
            _fsc3.metric("Put Premium", f"${_fs_put_prem:,.0f}")
            _fsc4.metric("Contracts / Tickers", f"{_fs_contracts} / {_fs_tickers}")

            # Tabs: Top Flow, By Ticker, Heatmap
            _flt1, _flt2, _flt3 = st.tabs(["Top Unusual Flow", "By Ticker Summary", "Premium by Ticker"])

            with _flt1:
                # Show top contracts by $ premium with Vol/OI > 1.5
                _fdf_uu = _fdf[_fdf["Vol/OI"] >= 1.5].copy()
                if _fdf_uu.empty:
                    _fdf_uu = _fdf.nlargest(30, "Est $Prem")
                _fdf_uu = _fdf_uu.sort_values("Est $Prem", ascending=False).head(50)
                st.caption(f"Top 50 contracts by est. premium (Vol/OI >= 1.5x)  |  {len(_fdf)} total active contracts")
                st.dataframe(
                    _fdf_uu.style.format({
                        "Strike": "${:,.0f}", "Mid": "${:.2f}", "Est $Prem": "${:,.0f}",
                        "IV": "{:.1f}%", "Dist%": "{:+.1f}%", "Vol/OI": "{:.1f}x",
                        "Spot": "${:,.2f}",
                    }).background_gradient(subset=["Vol/OI"], cmap="YlOrRd")
                     .background_gradient(subset=["Est $Prem"], cmap="Greens"),
                    use_container_width=True, hide_index=True, height=500,
                )

            with _flt2:
                # Aggregate by ticker: total call prem, put prem, net, unusual count, bias
                _fgrp = _fdf.groupby("Ticker").agg(
                    Call_Prem=("Est $Prem", lambda x: x[_fdf.loc[x.index, "C/P"] == "C"].sum()),
                    Put_Prem=("Est $Prem", lambda x: x[_fdf.loc[x.index, "C/P"] == "P"].sum()),
                    Contracts=("Vol", "count"),
                    Max_VolOI=("Vol/OI", "max"),
                    Avg_IV=("IV", "mean"),
                ).reset_index()
                _fgrp["Net $"] = _fgrp["Call_Prem"] - _fgrp["Put_Prem"]
                _fgrp["Bias"] = _fgrp["Net $"].apply(lambda x: "Bullish" if x > 0 else ("Bearish" if x < 0 else "Neutral"))
                _fgrp = _fgrp.sort_values("Net $", key=abs, ascending=False)
                st.caption("Aggregated flow by ticker — sorted by absolute net premium")
                st.dataframe(
                    _fgrp.style.format({
                        "Call_Prem": "${:,.0f}", "Put_Prem": "${:,.0f}", "Net $": "${:+,.0f}",
                        "Max_VolOI": "{:.1f}x", "Avg_IV": "{:.1f}%",
                    }),
                    use_container_width=True, hide_index=True, height=400,
                )

            with _flt3:
                # Bar chart: net premium by ticker (top 20)
                import plotly.graph_objects as _go_fs
                _top_flow = _fgrp.head(20).copy()
                _bar_colors = ["#26a69a" if v > 0 else "#ef5350" for v in _top_flow["Net $"]]
                _fig_fs = _go_fs.Figure()
                _fig_fs.add_trace(_go_fs.Bar(
                    x=_top_flow["Ticker"],
                    y=_top_flow["Net $"],
                    marker_color=_bar_colors,
                    text=[f"${v:+,.0f}" for v in _top_flow["Net $"]],
                    textposition="outside",
                ))
                _fig_fs.update_layout(
                    template="plotly_dark",
                    title="Net Options Premium by Ticker (Call - Put)",
                    xaxis_title="Ticker", yaxis_title="Net $ Premium",
                    height=420, margin=dict(l=50, r=30, t=40, b=30),
                )
                st.plotly_chart(_fig_fs, use_container_width=True)

            # Download button for flow data
            _flow_csv = _fdf.to_csv(index=False)
            st.download_button(
                "Download Flow Data CSV",
                data=_flow_csv,
                file_name="flow_scanner_results.csv",
                mime="text/csv",
                key="tf_flow_dl",
            )

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 5 — DRILL-DOWN
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("### Drill-Down")
    _tf_manual_tkr = st.text_input("Or type any ticker:", key="tf_manual_ticker", placeholder="e.g. NVDA").strip().upper()

    _drill_tkr = None
    if _tf_manual_tkr:
        _drill_tkr = _tf_manual_tkr
    else:
        _sel_sc = st.session_state.get("tf_drill_select", "—")
        if _sel_sc and _sel_sc != "—":
            _drill_tkr = _sel_sc

    if _drill_tkr:
        _drill_df = _load_daily_df(_drill_tkr)
        if _drill_df is None or len(_drill_df) < 10:
            st.warning(f"No data for {_drill_tkr}")
        else:
            _drill_df = _drill_df.sort_values("Date").tail(90).reset_index(drop=True)
            _dc = pd.to_numeric(_drill_df["Close"], errors="coerce")
            _sma10_d = _dc.rolling(10, min_periods=5).mean()
            _sma20_d = _dc.rolling(20, min_periods=10).mean()

            # Score for this ticker
            _full_df = _load_daily_df(_drill_tkr)
            _spy_ret20_dd = st.session_state.get("tf_spy_ret20", 0.0)
            _dsent = _tf_news_sentiment(_drill_tkr, limit=10)
            _dscore, _dcomps = _tf_score_ticker(_full_df, _spy_ret20_dd, _dsent.get('score', 0.0)) if _full_df is not None else (0, {})
            _dsetups = _tf_detect_setups(_full_df) if _full_df is not None else []

            # Header metrics
            _hc = st.columns(6)
            _hc[0].metric("Score", f"{_dscore:.1f}")
            _hc[1].metric("Alignment", f"{_dcomps.get('align_score',0)}/3")
            _hc[2].metric("Rel Vol", f"{_dcomps.get('rel_vol',1.0):.2f}×")
            _hc[3].metric("RSI(14)", f"{_dcomps.get('rsi',50):.1f}")
            _hc[4].metric("Pull%", f"{_dcomps.get('pull_pct',0):+.2f}%")
            _hc[5].metric("Ret 20d%", f"{_dcomps.get('ret_20d',0):+.2f}%")

            if _dsetups:
                st.info("**Setups:** " + "  |  ".join(_dsetups))

            # Two-panel chart: candlestick + score bars
            _fig_dd = _psp_tf.make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                row_heights=[0.72, 0.28],
                vertical_spacing=0.03,
            )

            # Candles
            _fig_dd.add_trace(
                _go_tf.Candlestick(
                    x=_drill_df["Date"],
                    open=pd.to_numeric(_drill_df["Open"], errors="coerce"),
                    high=pd.to_numeric(_drill_df["High"], errors="coerce"),
                    low=pd.to_numeric(_drill_df["Low"], errors="coerce"),
                    close=_dc,
                    name=_drill_tkr,
                    increasing_line_color="#26a69a",
                    decreasing_line_color="#ef5350",
                    showlegend=False,
                ),
                row=1, col=1,
            )
            _fig_dd.add_trace(
                _go_tf.Scatter(x=_drill_df["Date"], y=_sma10_d, name="SMA10",
                               line=dict(color="#ff9800", width=1.5), showlegend=True),
                row=1, col=1,
            )
            _fig_dd.add_trace(
                _go_tf.Scatter(x=_drill_df["Date"], y=_sma20_d, name="SMA20",
                               line=dict(color="#42a5f5", width=1.5, dash="dot"), showlegend=True),
                row=1, col=1,
            )

            # Score component bars
            if _dcomps:
                _score_labels = ["Slope", "RS_SPY", "RelVol", "Entry", "Momentum", "Sentiment"]
                _score_vals   = [
                    _dcomps.get("Slope(0-25)", 0),
                    _dcomps.get("RS_SPY(0-20)", 0),
                    _dcomps.get("RelVol(0-20)", 0),
                    _dcomps.get("Entry(0-20)", 0),
                    _dcomps.get("Momentum(0-15)", 0),
                    _dcomps.get("Sentiment(0-9)", 0),
                ]
                _bar_colors   = ["#4caf50","#2196f3","#ff9800","#9c27b0","#f44336","#00bcd4"]
                _fig_dd.add_trace(
                    _go_tf.Bar(
                        x=_score_labels, y=_score_vals,
                        marker_color=_bar_colors,
                        name="Score Components",
                        showlegend=False,
                    ),
                    row=2, col=1,
                )

            _fig_dd.update_layout(
                title=f"{_drill_tkr} — 90-Day View  |  Score: {_dscore:.1f}/100",
                template="plotly_dark",
                height=560,
                margin=dict(l=50, r=30, t=40, b=30),
                xaxis_rangeslider_visible=False,
            )
            st.plotly_chart(_fig_dd, use_container_width=True)

            # Options snapshot + enhanced intelligence
            _dopts = _tf_options_summary(_drill_tkr)
            _dshort = _tf_short_volume(_drill_tkr)
            if _dopts:
                st.markdown("**Options Snapshot (next 45 days)**")
                _oc = st.columns(4)
                _oc[0].metric("ATM IV", f"{_dopts.get('atm_iv','---')}%" if _dopts.get('atm_iv') else "---")
                _oc[1].metric("P/C Ratio (OI)", str(_dopts.get("pc_ratio") or "---"))
                _oc[2].metric("25d Put Skew", str(_dopts.get("skew_25d") or "---"))
                _oc[3].metric("Chain Count", str(_dopts.get("chain_count") or "---"))

                # Row 2: enhanced options intelligence
                st.markdown("**Options Flow Intelligence**")
                _oc2 = st.columns(4)
                # max pain with distance from spot
                _mp = _dopts.get("max_pain")
                _mpd = _dopts.get("max_pain_dist")
                _mp_label = f"${_mp:,.0f} ({_mpd:+.1f}%)" if _mp and _mpd is not None else "---"
                _oc2[0].metric("Max Pain", _mp_label)
                # net premium with direction colour
                _np = _dopts.get("net_premium", 0)
                _nd = _dopts.get("net_direction", "---")
                _np_str = f"${abs(_np):,.0f}" if _np else "$0"
                _np_color = "#00c853" if _np > 0 else ("#e53935" if _np < 0 else "#888")
                _oc2[1].markdown(
                    f"<div style='font-size:.85rem;color:#888'>Net Premium</div>"
                    f"<div style='font-size:1.4rem;font-weight:700;color:{_np_color}'>"
                    f"{_np_str} {_nd}</div>",
                    unsafe_allow_html=True,
                )
                # unusual activity
                _uu = _dopts.get("unusual_count", 0)
                _ut = _dopts.get("unusual_top", [])
                _uu_help = " | ".join(_ut) if _ut else "None detected"
                _oc2[2].metric("Unusual Contracts", str(_uu), help=_uu_help)
                # vol/OI ratio
                _voi = _dopts.get("vol_oi_ratio")
                _oc2[3].metric("Vol/OI Ratio", str(_voi) if _voi is not None else "---",
                               help="Total option volume / total OI. >0.5 = active day")

                # Row 3: short volume (if available)
                if _dshort:
                    _sc = st.columns(3)
                    _svr = _dshort.get("short_vol_ratio")
                    _sc[0].metric("Short Vol Ratio",
                                  f"{_svr:.1%}" if _svr is not None else "---",
                                  help="Short volume as % of total volume (FINRA/Polygon)")
                    _sc[1].metric("Short Volume", f"{_dshort.get('short_vol',0):,.0f}")
                    _sc[2].caption(f"as of {_dshort.get('date','')}")
            else:
                st.caption("Options data unavailable (check Polygon API key or click 'Fetch Options Data').")

            # ── FLOW ANALYSIS PANEL ──────────────────────────────────────────
            _flow_contracts = _dopts.get("_contracts", []) if _dopts else []
            _flow_spot = _dopts.get("spot") if _dopts else None
            if _flow_contracts and _flow_spot:
                with st.expander("**Flow Analysis** — unusual activity & premium breakdown", expanded=False):
                    # Build flow DataFrame from all contracts
                    _flow_rows = []
                    for _fc in _flow_contracts:
                        _fc_vol = _fc.get("vol", 0)
                        _fc_oi = _fc.get("oi", 0)
                        _fc_mid = _fc.get("mid", 0)
                        _fc_strike = _fc.get("strike", 0)
                        _fc_voi = round(_fc_vol / _fc_oi, 1) if _fc_oi > 0 else 0.0
                        _fc_prem = _fc_vol * _fc_mid * 100
                        _fc_dist = round((_fc_strike / _flow_spot - 1) * 100, 1) if _flow_spot else 0
                        _fc_delta = _fc.get("delta")
                        if _fc_strike <= _flow_spot:
                            _money = "ITM" if _fc["type"] == "call" else "OTM"
                        else:
                            _money = "OTM" if _fc["type"] == "call" else "ITM"
                        if abs(_fc_dist) < 2.0:
                            _money = "ATM"
                        _flow_rows.append({
                            "C/P": _fc["type"][0].upper(),
                            "Strike": _fc_strike,
                            "Expiry": _fc.get("expiry", ""),
                            "Vol": _fc_vol,
                            "OI": _fc_oi,
                            "Vol/OI": _fc_voi,
                            "Mid": round(_fc_mid, 2),
                            "Est $Prem": round(_fc_prem),
                            "IV": round(_fc.get("iv", 0) * 100, 1),
                            "Delta": round(float(_fc_delta), 2) if _fc_delta is not None else None,
                            "Dist%": _fc_dist,
                            "Moneyness": _money,
                        })
                    _flow_df = pd.DataFrame(_flow_rows)

                    # Tab 1: Unusual only (vol/OI > 1.5), Tab 2: All contracts, Tab 3: Premium chart
                    _ft1, _ft2, _ft3 = st.tabs(["Unusual Flow", "All Contracts", "Premium Chart"])

                    with _ft1:
                        _uf = _flow_df[(_flow_df["Vol/OI"] > 1.5) & (_flow_df["Vol"] > 200)].copy()
                        if _uf.empty:
                            _uf = _flow_df.nlargest(10, "Vol/OI")
                        _uf = _uf.sort_values("Vol", ascending=False).head(20)
                        st.caption(f"Contracts with Vol/OI > 1.5 and Vol > 200  |  Spot: ${_flow_spot:,.2f}")
                        st.dataframe(
                            _uf.style.format({
                                "Strike": "${:,.0f}", "Mid": "${:.2f}", "Est $Prem": "${:,.0f}",
                                "IV": "{:.1f}%", "Dist%": "{:+.1f}%", "Vol/OI": "{:.1f}x",
                            }).background_gradient(subset=["Vol/OI"], cmap="YlOrRd"),
                            use_container_width=True, hide_index=True, height=400,
                        )
                        # Quick verdict
                        _uf_calls = _uf[_uf["C/P"] == "C"]["Est $Prem"].sum()
                        _uf_puts = _uf[_uf["C/P"] == "P"]["Est $Prem"].sum()
                        _uf_bias = "Bullish" if _uf_calls > _uf_puts else ("Bearish" if _uf_puts > _uf_calls else "Neutral")
                        _uf_color = "#00c853" if _uf_bias == "Bullish" else ("#e53935" if _uf_bias == "Bearish" else "#888")
                        st.markdown(
                            f"Unusual flow bias: <span style='color:{_uf_color};font-weight:700'>"
                            f"{_uf_bias}</span> "
                            f"(calls ${_uf_calls:,.0f} vs puts ${_uf_puts:,.0f})",
                            unsafe_allow_html=True,
                        )

                    with _ft2:
                        _all_sorted = _flow_df.sort_values("Vol", ascending=False).head(60)
                        st.caption(f"Top 60 contracts by volume  |  {len(_flow_df)} total contracts")
                        st.dataframe(
                            _all_sorted.style.format({
                                "Strike": "${:,.0f}", "Mid": "${:.2f}", "Est $Prem": "${:,.0f}",
                                "IV": "{:.1f}%", "Dist%": "{:+.1f}%", "Vol/OI": "{:.1f}x",
                            }),
                            use_container_width=True, hide_index=True, height=500,
                        )

                    with _ft3:
                        # Stacked bar: call premium vs put premium by strike bucket
                        import plotly.graph_objects as _go_flow
                        _cprem = _flow_df[_flow_df["C/P"] == "C"].groupby("Strike")["Est $Prem"].sum()
                        _pprem = _flow_df[_flow_df["C/P"] == "P"].groupby("Strike")["Est $Prem"].sum()
                        _all_k = sorted(set(_cprem.index) | set(_pprem.index))
                        _fig_flow = _go_flow.Figure()
                        _fig_flow.add_trace(_go_flow.Bar(
                            x=_all_k, y=[_cprem.get(k, 0) for k in _all_k],
                            name="Call $", marker_color="#26a69a",
                        ))
                        _fig_flow.add_trace(_go_flow.Bar(
                            x=_all_k, y=[_pprem.get(k, 0) for k in _all_k],
                            name="Put $", marker_color="#ef5350",
                        ))
                        # Max pain + spot reference lines
                        _mp_val = _dopts.get("max_pain")
                        if _mp_val:
                            _fig_flow.add_vline(x=_mp_val, line_dash="dot", line_color="#ff9800",
                                                annotation_text="Max Pain", annotation_position="top left")
                        _fig_flow.add_vline(x=_flow_spot, line_dash="solid", line_color="#ffffff",
                                            annotation_text="Spot", annotation_position="top right")
                        _fig_flow.update_layout(
                            barmode="stack", template="plotly_dark",
                            title="Premium by Strike (Call vs Put)",
                            xaxis_title="Strike", yaxis_title="Est $ Premium",
                            height=420, margin=dict(l=50, r=30, t=40, b=30),
                            legend=dict(orientation="h", y=1.02),
                        )
                        st.plotly_chart(_fig_flow, use_container_width=True)

        # ─────────────────────────────────────────────────────────────────────
        # SECTION 6 — AI TRADE BRIEF
        # ─────────────────────────────────────────────────────────────────────
        st.markdown("### AI Trade Brief")
        _tb1, _tb2 = st.columns([2, 1])
        with _tb1:
            _tf_ai_provider = st.selectbox(
                "Provider",
                ["Anthropic", "OpenAI"],
                key="tf_ai_provider",
            )
        with _tb2:
            if _tf_ai_provider == "Anthropic":
                _tf_ai_models = ["claude-sonnet-4-6", "claude-opus-4-6", "claude-haiku-4-5-20251001"]
            else:
                _tf_ai_models = ["gpt-4.1", "gpt-4o", "gpt-4o-mini"]
            _tf_ai_model = st.selectbox("Model", _tf_ai_models, key="tf_ai_model")

        _tf_brief_btn = st.button("Generate AI Trade Brief", key="tf_brief_btn")

        if _tf_brief_btn and _drill_tkr:
            _ai_key = (
                os.getenv("ANTHROPIC_API_KEY", "").strip()
                if _tf_ai_provider == "Anthropic"
                else os.getenv("OPENAI_API_KEY", "").strip()
            )
            if not _ai_key:
                st.error(f"No API key found — set {'ANTHROPIC_API_KEY' if _tf_ai_provider=='Anthropic' else 'OPENAI_API_KEY'} in .env")
            else:
                # Build data snapshot for AI
                _opts_txt = ""
                if _dopts:
                    _opts_txt = (
                        f"Options: ATM IV={_dopts.get('atm_iv','N/A')}%, "
                        f"P/C Ratio={_dopts.get('pc_ratio','N/A')}, "
                        f"25d Skew={_dopts.get('skew_25d','N/A')}, "
                        f"Chain={_dopts.get('chain_count','N/A')} contracts\n"
                        f"Options Flow: Net Premium=${_dopts.get('net_premium',0):+,.0f} ({_dopts.get('net_direction','N/A')}), "
                        f"Unusual Contracts={_dopts.get('unusual_count',0)}, "
                        f"Vol/OI={_dopts.get('vol_oi_ratio','N/A')}"
                    )
                    _mp = _dopts.get('max_pain')
                    _mpd = _dopts.get('max_pain_dist')
                    if _mp:
                        _opts_txt += f"\nMax Pain: ${_mp:,.0f} ({_mpd:+.1f}% from spot)" if _mpd is not None else f"\nMax Pain: ${_mp:,.0f}"
                    _ut = _dopts.get('unusual_top', [])
                    if _ut:
                        _opts_txt += f"\nUnusual contracts: {' | '.join(_ut)}"
                _short_txt = ""
                if _dshort and _dshort.get("short_vol_ratio") is not None:
                    _short_txt = (
                        f"Short Interest: Short Vol Ratio={_dshort['short_vol_ratio']:.1%}, "
                        f"Short Vol={_dshort.get('short_vol',0):,.0f} "
                        f"(as of {_dshort.get('date','N/A')})"
                    )
                _vix_txt = f"VIX={_vix_val:.1f}" if _vix_val else "VIX=N/A"
                _macro_summary = ", ".join(
                    f"{_TF_MACRO_LABELS.get(t,t)}: "
                    f"{((_macro_snap.get(t) or {}).get('chg_pct') or 0):+.2f}%"
                    for t in ["SPY","QQQ","IWM","TLT"]
                )
                _tf_sys = (
                    "You are a senior equity trader and technical analyst with 20+ years of experience "
                    "in short-term and swing trading. Your job is to produce a concise, actionable Trade Brief "
                    "in the following exact format:\n\n"
                    "1. SETUP SUMMARY (2-3 sentences on pattern/alignment)\n"
                    "2. TRADE THESIS (bullish/bearish thesis, catalysts)\n"
                    "3. ENTRY ZONE (price range, conditions)\n"
                    "4. STOP LOSS (level and rationale)\n"
                    "5. TARGETS (T1, T2, T3 with price levels)\n"
                    "6. RISK/REWARD (ratio and context)\n"
                    "7. HOLDING PERIOD (intraday / swing / position)\n"
                    "8. OPTIONS ANGLE (use IV, net premium flow direction, unusual activity, max pain, "
                    "and short interest data if available — strategy suggestion)\n"
                    "9. CONFIDENCE (1–10 with reasoning)\n\n"
                    "Be specific with price levels. Avoid vague language. "
                    "If options data is not available, skip section 8. "
                    "Incorporate recent news sentiment data (FinBERT-scored) into your Trade Thesis if available."
                )
                _tf_user = "\n".join([
                    f"Ticker: {_drill_tkr}",
                    f"Composite Score: {_dscore:.1f}/100",
                    f"Alignment Score: {_dcomps.get('align_score',0)}/3",
                    f"Slope 1d: {_dcomps.get('slope_1d',0):.4f}%/day, "
                    f"5d: {_dcomps.get('slope_5d',0):.4f}%/day, "
                    f"10d: {_dcomps.get('slope_10d',0):.4f}%/day",
                    f"RSI(14): {_dcomps.get('rsi',50):.1f}",
                    f"Relative Volume: {_dcomps.get('rel_vol',1.0):.2f}×",
                    f"Pull from MA10: {_dcomps.get('pull_pct',0):+.2f}%",
                    f"20d Return: {_dcomps.get('ret_20d',0):+.2f}%",
                    f"RS vs SPY score: {_dcomps.get('RS_SPY(0-20)',0):.1f}/20",
                    f"Entry quality: {_dcomps.get('Entry(0-20)',0):.1f}/20",
                    f"Momentum (5d): {_dcomps.get('Momentum(0-15)',0):.1f}/15  (5d ret: {_dcomps.get('ret_5d',0):+.2f}%)",
                    f"Sentiment: {_dcomps.get('Sentiment(0-9)',0):.1f}/9",
                    f"Detected setups: {', '.join(_dsetups) if _dsetups else 'None'}",
                    f"Market: {regime_lbl} | {_vix_txt}",
                    f"Macro moves: {_macro_summary}",
                    _opts_txt,
                    _short_txt,
                    (f"News Sentiment: {_dsent['label']} ({_dsent['score']:+.2f}), "
                     f"{_dsent['count']} headlines scored"
                     if _dsent.get('count', 0) > 0 else "News Sentiment: N/A"),
                    ("\n".join(
                        f"  - \"{h['title'][:80]}\" ({h['sentiment']}, {h['confidence']:.0%})"
                        for h in _dsent.get("headlines", [])[:5]
                    ) if _dsent.get("headlines") else ""),
                    "",
                    "Generate the Trade Brief in the 9-section format specified.",
                ])

                with st.spinner("Generating Trade Brief…"):
                    try:
                        _brief_out = _call_llm(
                            _tf_ai_provider,
                            _ai_key,
                            _tf_ai_model,
                            _tf_sys,
                            _tf_user,
                            max_tokens=1200,
                        )
                        if _brief_out:
                            _brief_hdr = (
                                f"<div style='background:#1a237e;padding:10px 20px;"
                                f"border-radius:6px;margin-bottom:12px'>"
                                f"<span style='color:#fff;font-weight:700;font-size:1.05rem'>"
                                f"AI TRADE BRIEF — {_drill_tkr} | Score: {_dscore:.1f}/100 | "
                                f"Setups: {', '.join(_dsetups) if _dsetups else '—'}"
                                f"</span></div>"
                            )
                            st.markdown(_brief_hdr, unsafe_allow_html=True)
                            st.markdown(_brief_out)
                            _dl_brief = (
                                f"AI TRADE BRIEF — {_drill_tkr}\n"
                                f"Score: {_dscore:.1f}/100 | {regime_lbl} | {_vix_txt}\n"
                                f"{'='*60}\n\n{_brief_out}\n\n"
                                f"{'='*60}\nDATA SNAPSHOT\n{_tf_user}"
                            )
                            st.download_button(
                                "⬇ Download Brief",
                                data=_dl_brief,
                                file_name=f"Trade_Brief_{_drill_tkr}.txt",
                                mime="text/plain",
                                key="tf_dl_brief",
                            )
                        else:
                            st.error("No response from AI — check API key and model.")
                    except Exception as _e_tf:
                        st.error(f"AI call failed: {_e_tf}")


# ============================================================
# AI Scanner tab
# ============================================================
if nav == 'AI Scanner':
    import json as _asjson
    import plotly.graph_objects as _asgo

    st.subheader('AI Scanner — Natural Language Screener')
    st.markdown(
        "Describe what you're looking for in plain English. "
        "The AI parses your query into filter conditions and scans your universe instantly."
    )

    _as_pq_dir   = os.environ.get('PER_TICKER_PARQUET_DIR', '')
    _as_idx_path = os.path.join(_as_pq_dir, '_signals_index.parquet') if _as_pq_dir else ''
    _as_idx_ok   = bool(_as_pq_dir and os.path.exists(_as_idx_path))

    @st.cache_data(show_spinner=False, ttl=120)
    def _as_load_index(path):
        return pd.read_parquet(path)

    if _as_idx_ok:
        _as_idx   = _as_load_index(_as_idx_path)
        _as_total = len(_as_idx)
        _AS_ENH = ['SMA200_Slope', 'SMA10_Slope_Prev', 'Close_to_SMA50', 'Close_to_SMA200']
        if not all(c in _as_idx.columns for c in _AS_ENH):
            st.info('ℹ️ Index missing enhanced MA columns — rebuild in Signal Scanner for '
                    'SMA50/200 distance queries. Basic RSI, BB, and slope queries still work.')
        st.caption(f'Index: **{_as_total:,} tickers**')
    else:
        _as_idx   = None
        _as_total = 0
        st.warning(
            '⚠️ No signals index found. '
            'Go to **Signal Scanner** and click **Build / Rebuild Index** first, '
            'then return here.'
        )
    st.markdown('---')

    # -- Example query chips
    st.markdown('**Try an example query:**')
    _AS_EXAMPLES = [
        "Stocks pulling back to the upward-sloping 50-day MA",
        "10-day slope just turned positive, stock moving up with it",
        "Near the 200-day MA with RSI under 50",
        "Above upper Bollinger Band with RSI over 70",
        "RSI under 35 within 3% of the 20-day MA",
        "Strong momentum: CC% over 1.5 with RSI between 45 and 65",
    ]
    _as_ec1, _as_ec2, _as_ec3 = st.columns(3)
    _as_ecols = [_as_ec1, _as_ec2, _as_ec3]
    for _asi, _asex in enumerate(_AS_EXAMPLES):
        if _as_ecols[_asi % 3].button(_asex, key=f"as_ex_{_asi}", use_container_width=True):
            st.session_state["as_query"] = _asex
            st.rerun()

    # -- Query text area
    _as_query = st.text_area(
        "Describe your screen:",
        key="as_query",
        height=80,
        placeholder="e.g. stocks pulling back to upsloping 50-day MA with RSI under 55",
    )

    # -- Rule-based NL parser (no API key needed) ----------------------------
    import re as _as_re

    def _as_parse_query(_q):
        """Translate natural-language screen query into filter conditions."""
        q = _q.lower().strip()
        filters = []

        # ── MA-period detection ───────────────────────────────────────────
        _AS_MA_MAP  = [("200","SMA200"),("50","SMA50"),("20","SMA20"),("10","SMA10")]
        _AS_PREV    = {"SMA10": "SMA10_Slope_Prev", "SMA50": "SMA50_Slope_Prev"}

        for _period, _col in _AS_MA_MAP:
            # Does this period appear? e.g. "50-day", "50d", "50 day"
            if not _as_re.search(r'\b' + _period + r'[\s\-]?d(?:ay)?\b', q):
                continue
            _sc   = f'{_col}_Slope'
            _dc   = f'Close_to_{_col}'
            _prev = _AS_PREV.get(_col)

            # slope just turned positive / turning up
            if _as_re.search(r'(just\s+)?(turn(ed|ing)?\s+(up|positive)|slope\s+turn)', q):
                filters.append({"col": _sc, "op": ">", "val": 0})
                if _prev:
                    filters.append({"col": _prev, "op": "<=", "val": 0})

            # pulling back / dipping to
            elif _as_re.search(r'pull(ing)?\s*(back|down)|pullback|retrac|dip(ping)?\s*(to|toward)', q):
                if _as_re.search(r'(up(ward)?(\s*slop)?|ris(ing)?|upslop)', q):
                    filters.append({"col": _sc, "op": ">", "val": 0})
                filters.append({"col": _dc, "op": "between", "lo": -4.0, "hi": 1.5})

            # near / at / testing
            elif _as_re.search(r'\b(near|around|at the|close to|touch(ing)?|test(ing)?|within\s+\d)', q):
                if _as_re.search(r'(up(ward)?|ris(ing)?|upslop)', q):
                    filters.append({"col": _sc, "op": ">", "val": 0})
                # "within X%" - extract the number
                _wm = _as_re.search(r'within\s+([\d.]+)\s*%', q)
                _tol = float(_wm.group(1)) if _wm else 2.0
                filters.append({"col": _dc, "op": "between", "lo": -_tol, "hi": _tol})

            # above MA
            elif _as_re.search(r'\babove\b', q) and 'bollinger' not in q and 'bb' not in q:
                filters.append({"col": _dc, "op": ">", "val": 0})

            # below MA
            elif _as_re.search(r'\bbelow\b', q) and 'bollinger' not in q and 'bb' not in q:
                filters.append({"col": _dc, "op": "<", "val": 0})

            # upward / rising slope
            elif _as_re.search(r'(up(ward)?|ris(ing)?|positive|upslop)', q):
                filters.append({"col": _sc, "op": ">", "val": 0})

            # downward / falling slope
            elif _as_re.search(r'(down(ward)?|fall(ing)?|declin|negative)', q):
                filters.append({"col": _sc, "op": "<", "val": 0})

            # flattening
            elif _as_re.search(r'(flat(ten)?|consolidat)', q):
                filters.append({"col": _sc, "op": "between", "lo": -0.05, "hi": 0.05})

        # ── Standalone "slope turning up" without explicit MA period → SMA10 ─
        if _as_re.search(r'(10.?day\s+slope|slope\s+just\s+turn|slope.{0,10}turn(ed)?\s+up)', q):
            if not any(f.get("col") == "SMA10_Slope" for f in filters):
                filters.append({"col": "SMA10_Slope",      "op": ">",  "val": 0})
                filters.append({"col": "SMA10_Slope_Prev", "op": "<=", "val": 0})

        # ── RSI ───────────────────────────────────────────────────────────
        _rm = _as_re.search(r'rsi\s*(?:under|below|<|less\s*than)\s*(\d+)', q)
        if _rm:
            filters.append({"col": "RSI14", "op": "<", "val": float(_rm.group(1))})

        _rm2 = _as_re.search(r'rsi\s*(?:over|above|>|greater\s*than)\s*(\d+)', q)
        if _rm2:
            filters.append({"col": "RSI14", "op": ">", "val": float(_rm2.group(1))})

        _rb = _as_re.search(r'rsi\s*(?:between|from)\s*(\d+)\s*(?:and|to|-)\s*(\d+)', q)
        if _rb:
            filters.append({"col": "RSI14", "op": "between",
                            "lo": float(_rb.group(1)), "hi": float(_rb.group(2))})

        if 'oversold' in q and not any(f.get("col") == "RSI14" for f in filters):
            filters.append({"col": "RSI14", "op": "<", "val": 35})
        if 'overbought' in q and not any(f.get("col") == "RSI14" for f in filters):
            filters.append({"col": "RSI14", "op": ">", "val": 70})

        # ── Bollinger Bands ───────────────────────────────────────────────
        if _as_re.search(r'above\s+(?:the\s+)?(?:upper\s+)?(bollinger|bb)', q):
            filters.append({"col": "BB_PctB", "op": ">", "val": 1.0})
        if _as_re.search(r'below\s+(?:the\s+)?(?:lower\s+)?(bollinger|bb)', q):
            filters.append({"col": "BB_PctB", "op": "<", "val": 0.0})

        # ── CC% / momentum ─────────────────────────────────────────────────
        _cm = _as_re.search(r'cc%?\s*(?:over|above|>)\s*([\d.]+)', q)
        if _cm:
            filters.append({"col": "CC_Pct", "op": ">", "val": float(_cm.group(1))})
        elif _as_re.search(r'(strong\s+momentum|strong\s+move\s+up|big\s+move)', q):
            filters.append({"col": "CC_Pct", "op": ">", "val": 1.5})

        # ── Volatility / consolidation ──────────────────────────────────────
        if _as_re.search(r'(low\s+volat|consolidat|quiet|tight)', q):
            filters.append({"col": "ATR_Pct", "op": "<", "val": 2.0})

        # ── News sentiment (FinBERT) ─────────────────────────────────────────────
        if _as_re.search(r'(bullish|positive)\s+(sentiment|news)', q):
            filters.append({"col": "Sentiment", "op": ">", "val": 0.2})
        elif _as_re.search(r'(bearish|negative)\s+(sentiment|news)', q):
            filters.append({"col": "Sentiment", "op": "<", "val": -0.2})
        elif _as_re.search(r'neutral\s+sentiment', q):
            filters.append({"col": "Sentiment", "op": "between", "lo": -0.2, "hi": 0.2})

        return filters

    # -- Parse & Scan button
    _as_run = st.button(
        "\u26a1 Parse & Scan",
        type="primary",
        key="as_run_btn",
        disabled=(not _as_query.strip()) or (not _as_idx_ok),
        help="Build the Signal Scanner index first" if not _as_idx_ok else None,
    )

    if _as_run and _as_query.strip():
        _as_parsed = _as_parse_query(_as_query)
        if _as_parsed:
            st.session_state["as_filters"]     = _as_parsed
            st.session_state["as_query_label"] = _as_query.strip()
            st.rerun()
        else:
            st.warning(
                "Could not extract filter conditions from that query. "
                "Try specifying a period (e.g. '50-day') and a condition "
                "(e.g. 'RSI under 45', 'pulling back', 'slope turning up')."
            )

    _as_filters = st.session_state.get("as_filters", [])

    # -- Show parsed filters & results
    if _as_filters:
        _as_ql = st.session_state.get("as_query_label", "")
        st.markdown(f"**Query:** *{_as_ql}*")

        # Display filter pills
        _as_fdisp = []
        for _asf in _as_filters:
            _afc = _asf.get("col", "?")
            _afo = _asf.get("op",  "?")
            if _afo == "between":
                _as_lo = _asf.get("lo")
                _as_hi = _asf.get("hi")
                _as_fdisp.append(f"`{_afc}` between {_as_lo} and {_as_hi}")
            else:
                _as_val = _asf.get("val")
                _as_fdisp.append(f"`{_afc}` {_afo} {_as_val}")
        st.caption("Filters: " + "  |  ".join(_as_fdisp))

        if st.button("↺ New Query", key="as_clear_btn"):
            st.session_state.pop("as_filters", None)
            st.session_state.pop("as_query_label", None)
            st.session_state.pop("as_query", None)
            st.rerun()

        # Apply filters to index
        def _as_apply(df, filters):
            mask = pd.Series([True] * len(df), index=df.index)
            for _asf in filters:
                _afc = _asf.get("col", "")
                _afo = _asf.get("op",  "")
                if _afc not in df.columns:
                    continue
                _afs = pd.to_numeric(df[_afc], errors="coerce")
                if   _afo == ">":       mask &= _afs >  float(_asf["val"])
                elif _afo == ">=":      mask &= _afs >= float(_asf["val"])
                elif _afo == "<":       mask &= _afs <  float(_asf["val"])
                elif _afo == "<=":      mask &= _afs <= float(_asf["val"])
                elif _afo == "==":      mask &= _afs == float(_asf["val"])
                elif _afo == "between":
                    mask &= (_afs >= float(_asf["lo"])) & (_afs <= float(_asf["hi"]))
            return df[mask]

        if _as_idx is None:
            st.warning("Build the Signal Scanner index first.")
            st.stop()
        _as_results = _as_apply(_as_idx, _as_filters).copy()
        st.metric("Matches", f"{len(_as_results):,} / {_as_total:,}")

        if _as_results.empty:
            st.info("No tickers matched those conditions. Try relaxing the filters.")
        else:
            _as_disp_cols = [c for c in [
                "Ticker", "Last Date", "Close", "RSI14", "BB_PctB",
                "SMA50_Slope", "Close_to_SMA50",
                "SMA200_Slope", "Close_to_SMA200",
                "CC_Pct", "ATR_Pct", "Avg Vol",
            ] if c in _as_results.columns]
            _as_disp = _as_results[_as_disp_cols].reset_index(drop=True)
            for _arc in [
                "RSI14", "BB_PctB", "SMA50_Slope", "Close_to_SMA50",
                "SMA200_Slope", "Close_to_SMA200", "CC_Pct", "ATR_Pct"
            ]:
                if _arc in _as_disp.columns:
                    _as_disp[_arc] = pd.to_numeric(_as_disp[_arc], errors="coerce").round(2)

            st.markdown("### Results")
            _as_sel = st.dataframe(
                _as_disp,
                use_container_width=True,
                hide_index=True,
                on_select="rerun",
                selection_mode="single-row",
                key="as_result_tbl",
            )

            st.download_button(
                "⬇ Download CSV",
                data=_as_disp.to_csv(index=False),
                file_name="ai_scanner_results.csv",
                mime="text/csv",
                key="as_dl_csv",
            )

            # -- Row-click drill-down chart
            _as_sel_rows = (
                _as_sel.selection.get("rows", [])
                if _as_sel and hasattr(_as_sel, "selection") else []
            )
            if _as_sel_rows:
                _as_sel_tkr = str(_as_results.iloc[_as_sel_rows[0]].get("Ticker", ""))
                st.markdown("---")
                st.markdown(f"## {_as_sel_tkr} — Chart & Score")

                # Load daily OHLCV
                _as_cdf = _load_daily_df(_as_sel_tkr)
                if _as_cdf is None or len(_as_cdf) < 10:
                    st.warning(f"No daily data found for {_as_sel_tkr}")
                else:
                    _as_cdf = _as_cdf.sort_values("Date").tail(90).reset_index(drop=True)
                    _as_dc  = pd.to_numeric(_as_cdf["Close"], errors="coerce")
                    _as_s10 = _as_dc.rolling(10, min_periods=5).mean()
                    _as_s20 = _as_dc.rolling(20, min_periods=10).mean()
                    _as_s50 = _as_dc.rolling(50, min_periods=25).mean()

                    _as_fig = _asgo.Figure()
                    _as_fig.add_trace(_asgo.Candlestick(
                        x=_as_cdf["Date"],
                        open=pd.to_numeric(_as_cdf.get("Open",  _as_dc), errors="coerce"),
                        high=pd.to_numeric(_as_cdf.get("High",  _as_dc), errors="coerce"),
                        low=pd.to_numeric( _as_cdf.get("Low",   _as_dc), errors="coerce"),
                        close=_as_dc,
                        name="Price",
                        increasing_line_color="#26a69a",
                        decreasing_line_color="#ef5350",
                    ))
                    _as_fig.add_trace(_asgo.Scatter(
                        x=_as_cdf["Date"], y=_as_s10, name="SMA10",
                        line=dict(color="#f5a623", width=1),
                    ))
                    _as_fig.add_trace(_asgo.Scatter(
                        x=_as_cdf["Date"], y=_as_s20, name="SMA20",
                        line=dict(color="#7b68ee", width=1),
                    ))
                    _as_fig.add_trace(_asgo.Scatter(
                        x=_as_cdf["Date"], y=_as_s50, name="SMA50",
                        line=dict(color="#00bcd4", width=1.5),
                    ))
                    _as_fig.update_layout(
                        title=f"{_as_sel_tkr} — 90-Day",
                        xaxis_rangeslider_visible=False,
                        height=420,
                        margin=dict(l=40, r=20, t=40, b=20),
                        legend=dict(orientation="h"),
                        template="plotly_dark",
                    )
                    st.plotly_chart(_as_fig, use_container_width=True)

                    # Score metrics from Trade Finder scorer
                    try:
                        _as_spy_r = float(st.session_state.get("tf_spy_ret20", 0.0))
                        _as_score, _as_comps = _tf_score_ticker(_as_cdf, _as_spy_r, 0.0)
                        _asm1, _asm2, _asm3, _asm4 = st.columns(4)
                        _asm1.metric("Score",     f"{_as_score:.1f} / 100")
                        _ask_a = "Slope(0-25)"
                        _ask_e = "Entry(0-20)"
                        _ask_m = "Momentum(0-15)"
                        _asm2.metric("Slope",     f"{_as_comps.get(_ask_a, 0):.0f} / 25")
                        _asm3.metric("Entry",     f"{_as_comps.get(_ask_e, 0):.0f} / 20")
                        _asm4.metric("Momentum",  f"{_as_comps.get(_ask_m, 0):.0f} / 15")
                    except Exception:
                        pass

    # -- No filters yet
    else:
        st.info(
            "Enter a query above (or click an example) and press **Parse & Scan** to run a screen. "
            "The AI will translate your description into filter conditions and show matching tickers."
        )




# =============================================================
# SMART MONEY TAB
# =============================================================
if nav == 'Smart Money':
    import plotly.graph_objects as _smgo

    st.markdown("## Smart Money Intelligence")
    st.caption("Flow trends, congressional trades & insider activity " + chr(0x2014) + " no Unusual Whales API needed")

    _sm_t1, _sm_t2, _sm_t3 = st.tabs([
        chr(0x1F4CA) + " Flow Trends",
        chr(0x1F3DB) + " Congress Trades",
        chr(0x1F50D) + " Insider Activity",
    ])

    # ── Tab 1: Flow Trends ──────────────────────────────────────────────
    with _sm_t1:
        _sm_fh = _sm_load_flow_history()
        if _sm_fh.empty:
            st.info(
                chr(0x1F4A1) + " **No flow history yet.** Run a Trade Finder scan with "
                'the **"Fetch Options Data (slower)"** checkbox enabled in Scanner Setup. '
                "Each scan captures unusual options activity across your entire scanned universe."
            )
        else:
            # Days selector
            _sm_days = st.select_slider(
                "Lookback", options=[1, 2, 3, 5, 7, 14, 30], value=5,
                key="sm_flow_days",
            )
            _sm_trends = _sm_flow_trends(_sm_fh, days=_sm_days)

            if _sm_trends.empty:
                st.warning("No flow data in the selected window.")
            else:
                # Summary metrics
                _sm_total_scans = _sm_fh["ScanTime"].nunique() if "ScanTime" in _sm_fh.columns else 0
                _sm_uniq_tkrs = _sm_trends["Ticker"].nunique() if "Ticker" in _sm_trends.columns else 0
                _sm_net_all = _sm_trends["NetPrem"].sum() if "NetPrem" in _sm_trends.columns else 0
                _sm_call_all = _sm_trends["CallPrem"].sum() if "CallPrem" in _sm_trends.columns else 0
                _sm_put_all = _sm_trends["PutPrem"].sum() if "PutPrem" in _sm_trends.columns else 0
                _sm_bc_ratio = round(_sm_call_all / max(_sm_put_all, 1), 2)
                _sm_net_dir = chr(0x1F7E2) + " Bullish" if _sm_net_all > 0 else chr(0x1F534) + " Bearish"

                _smc1, _smc2, _smc3, _smc4 = st.columns(4)
                _smc1.metric("Scans Captured", f"{_sm_total_scans:,}")
                _smc2.metric("Tickers Tracked", f"{_sm_uniq_tkrs:,}")
                _smc3.metric("Net Premium", f"${_sm_net_all/1e6:+.1f}M", delta=_sm_net_dir)
                _smc4.metric("Bull/Bear Ratio", f"{_sm_bc_ratio:.2f}x")

                st.markdown("---")

                # Top tickers by absolute net premium
                _sm_tkr_agg = _sm_trends.groupby("Ticker").agg(
                    CallPrem=("CallPrem", "sum"),
                    PutPrem=("PutPrem", "sum"),
                    NetPrem=("NetPrem", "sum"),
                    Contracts=("Contracts", "sum"),
                ).reset_index()
                _sm_tkr_agg["AbsNet"] = _sm_tkr_agg["NetPrem"].abs()
                _sm_tkr_agg = _sm_tkr_agg.sort_values("AbsNet", ascending=False).head(20)

                st.markdown("### Top 20 Tickers by Premium Momentum")

                # Format for display
                _sm_disp = _sm_tkr_agg[["Ticker", "CallPrem", "PutPrem", "NetPrem", "Contracts"]].copy()
                for _smc in ["CallPrem", "PutPrem", "NetPrem"]:
                    _sm_disp[_smc] = _sm_disp[_smc].apply(
                        lambda v: f"${v/1e6:+.2f}M" if abs(v) >= 1e6 else f"${v/1e3:+.1f}K"
                    )
                st.dataframe(_sm_disp, use_container_width=True, hide_index=True)

                # Bar chart: top 10 tickers net premium
                _sm_top10 = _sm_tkr_agg.head(10)
                _sm_bar = _smgo.Figure()
                _sm_colors = [
                    "#26a69a" if v > 0 else "#ef5350" for v in _sm_top10["NetPrem"]
                ]
                _sm_bar.add_trace(_smgo.Bar(
                    x=_sm_top10["Ticker"],
                    y=_sm_top10["NetPrem"],
                    marker_color=_sm_colors,
                    text=_sm_top10["NetPrem"].apply(
                        lambda v: f"${v/1e6:+.1f}M" if abs(v) >= 1e6 else f"${v/1e3:+.0f}K"
                    ),
                    textposition="outside",
                ))
                _sm_bar.update_layout(
                    title="Net Premium by Ticker (Top 10)",
                    yaxis_title="Net Premium ($)",
                    height=380,
                    template="plotly_dark",
                    margin=dict(l=40, r=20, t=50, b=30),
                    showlegend=False,
                )
                st.plotly_chart(_sm_bar, use_container_width=True)

                # Daily trend chart for top 5
                _sm_top5_list = _sm_tkr_agg.head(5)["Ticker"].tolist()
                _sm_daily_top5 = _sm_trends[_sm_trends["Ticker"].isin(_sm_top5_list)].copy()
                if not _sm_daily_top5.empty and "ScanDate" in _sm_daily_top5.columns:
                    st.markdown("### Daily Flow Trend (Top 5)")
                    _sm_dfig = _smgo.Figure()
                    _sm_pal = ["#f5a623", "#7b68ee", "#26a69a", "#ef5350", "#00bcd4"]
                    for _smi, _smt in enumerate(_sm_top5_list):
                        _sm_td = _sm_daily_top5[_sm_daily_top5["Ticker"] == _smt].sort_values("ScanDate")
                        _sm_dfig.add_trace(_smgo.Scatter(
                            x=_sm_td["ScanDate"].astype(str),
                            y=_sm_td["NetPrem"],
                            mode="lines+markers",
                            name=_smt,
                            line=dict(color=_sm_pal[_smi % len(_sm_pal)], width=2),
                        ))
                    _sm_dfig.update_layout(
                        yaxis_title="Net Premium ($)",
                        height=350,
                        template="plotly_dark",
                        margin=dict(l=40, r=20, t=30, b=30),
                        legend=dict(orientation="h"),
                    )
                    st.plotly_chart(_sm_dfig, use_container_width=True)

                # Narrative bullets
                st.markdown("### Flow Narratives")
                for _, _smr in _sm_tkr_agg.head(10).iterrows():
                    _smt = _smr["Ticker"]
                    _smn = _smr["NetPrem"]
                    _smd = "call" if _smn > 0 else "put"
                    _smv = abs(_smn)
                    _sme = chr(0x1F7E2) if _smn > 0 else chr(0x1F534)
                    if _smv >= 1e6:
                        _smvs = f"${_smv/1e6:.1f}M"
                    else:
                        _smvs = f"${_smv/1e3:.0f}K"
                    st.markdown(f"{_sme} **{_smt}** " + chr(0x2014) + f" {_smd} premium dominant ({_smvs} net), {int(_smr['Contracts']):,} contracts")

                st.download_button(
                    chr(0x2B07) + " Download Flow History CSV",
                    data=_sm_fh.to_csv(index=False),
                    file_name="flow_history.csv",
                    mime="text/csv",
                    key="sm_dl_flow",
                )

    # ── Tab 2: Congress Trades (QuiverQuant – free, no key needed) ─────
    with _sm_t2:
        st.markdown("### Congressional Trading Activity")
        st.caption("Source: QuiverQuant " + chr(0x2014) + " 1,000 most recent disclosures")

        _sm_cf1, _sm_cf2, _sm_cf3, _sm_cf4 = st.columns(4)
        _sm_ct_tkr = _sm_cf1.text_input("Filter Ticker", "", key="sm_ct_tkr").strip().upper()
        _sm_ct_party = _sm_cf2.selectbox("Party", ["All", "Democrat", "Republican"], key="sm_ct_party")
        _sm_ct_type = _sm_cf3.selectbox("Type", ["All", "Purchase", "Sale"], key="sm_ct_type")
        _sm_ct_since = _sm_cf4.date_input("Since", value=pd.Timestamp.now() - pd.Timedelta(days=90), key="sm_ct_since")

        with st.spinner("Fetching congressional trades..."):
            _sm_cdf = _sm_congress_trades(_sm_ct_tkr)

        if _sm_cdf.empty:
            st.info("No congressional trades found. Try a different ticker or broadening the date range.")
        else:
            # Apply filters
            if _sm_ct_party != "All":
                _sm_p_map = {"Democrat": "D", "Republican": "R"}
                _sm_cdf = _sm_cdf[_sm_cdf["Party"].str.upper().str.startswith(_sm_p_map.get(_sm_ct_party, ""))]
            if _sm_ct_type != "All":
                _sm_cdf = _sm_cdf[_sm_cdf["Type"].str.lower().str.contains(_sm_ct_type.lower(), na=False)]
            if "Date" in _sm_cdf.columns:
                _sm_cdf = _sm_cdf[_sm_cdf["Date"] >= pd.Timestamp(_sm_ct_since)]

            if _sm_cdf.empty:
                st.info("No trades match those filters.")
            else:
                # Summary
                _sm_cc1, _sm_cc2, _sm_cc3, _sm_cc4 = st.columns(4)
                _sm_cc1.metric("Total Trades", f"{len(_sm_cdf):,}")
                _sm_cc2.metric("Unique Members", f"{_sm_cdf['Member'].nunique():,}")
                _sm_top_ct = _sm_cdf["Ticker"].value_counts().head(3)
                _sm_cc3.metric("Top Tickers", ", ".join(_sm_top_ct.index.tolist()))
                # Buy vs sell ratio
                _sm_buys_c = len(_sm_cdf[_sm_cdf["Type"].str.lower() == "purchase"])
                _sm_sells_c = len(_sm_cdf[_sm_cdf["Type"].str.lower() == "sale"])
                _sm_cc4.metric("Buy / Sell", f"{_sm_buys_c} / {_sm_sells_c}")

                # Highlight tickers in user's universe
                _sm_parq_dir = os.environ.get("PER_TICKER_PARQUET_DIR", "")
                _sm_universe = set()
                if _sm_parq_dir and os.path.isdir(_sm_parq_dir):
                    _sm_universe = {
                        f.replace(".parquet", "").upper()
                        for f in os.listdir(_sm_parq_dir)
                        if f.endswith(".parquet") and not f.startswith("_")
                    }

                _sm_cd = _sm_cdf.copy()
                if _sm_universe:
                    _sm_cd["In Universe"] = _sm_cd["Ticker"].apply(
                        lambda t: chr(0x2705) if t.upper() in _sm_universe else ""
                    )

                # Excess return column — color-grade it
                _sm_disp_cols = [c for c in [
                    "Date", "Member", "Party", "Chamber", "Ticker", "Type",
                    "Range", "ExcessReturn", "In Universe",
                ] if c in _sm_cd.columns]
                _sm_cd_disp = _sm_cd[_sm_disp_cols].copy()
                if "ExcessReturn" in _sm_cd_disp.columns:
                    _sm_cd_disp["ExcessReturn"] = pd.to_numeric(
                        _sm_cd_disp["ExcessReturn"], errors="coerce"
                    ).round(2)

                st.dataframe(_sm_cd_disp, use_container_width=True, hide_index=True)

                # Most-traded tickers bar chart
                _sm_tkr_counts = _sm_cdf["Ticker"].value_counts().head(15)
                if len(_sm_tkr_counts) > 1:
                    st.markdown("### Most-Traded Tickers by Congress")
                    _sm_ct_fig = _smgo.Figure()
                    _sm_ct_fig.add_trace(_smgo.Bar(
                        x=_sm_tkr_counts.index.tolist(),
                        y=_sm_tkr_counts.values.tolist(),
                        marker_color="#7b68ee",
                    ))
                    _sm_ct_fig.update_layout(
                        yaxis_title="# Trades",
                        height=340,
                        template="plotly_dark",
                        margin=dict(l=40, r=20, t=30, b=30),
                    )
                    st.plotly_chart(_sm_ct_fig, use_container_width=True)

                st.download_button(
                    chr(0x2B07) + " Download CSV",
                    data=_sm_cd.to_csv(index=False),
                    file_name="congress_trades.csv",
                    mime="text/csv",
                    key="sm_dl_congress",
                )

    # ── Tab 3: Insider Activity (Finnhub free tier) ──────────────────────
    with _sm_t3:
        _sm_fh_key = (os.getenv("FINNHUB_API_KEY") or "").strip()
        if not _sm_fh_key:
            st.warning(
                chr(0x1F511) + " **Finnhub API key not found.** "
                "Add `FINNHUB_API_KEY=your_key` to your `.env` file. "
                "[Get a free key at finnhub.io](https://finnhub.io/register)"
            )
        else:
            st.markdown("### SEC Insider Trading Activity")
            st.caption("Source: Finnhub insider-transactions (SEC Form 4 filings)")

            _sm_if1, _sm_if2, _sm_if3 = st.columns(3)
            _sm_it_tkr = _sm_if1.text_input(
                "Ticker (required)", "", key="sm_it_tkr",
                help="Finnhub free tier requires a ticker symbol",
            ).strip().upper()
            _sm_it_type = _sm_if2.selectbox("Type", ["All", "Buy", "Sale"], key="sm_it_type")
            _sm_it_since = _sm_if3.date_input("Since", value=pd.Timestamp.now() - pd.Timedelta(days=180), key="sm_it_since")

            if not _sm_it_tkr:
                st.info("Enter a ticker symbol above to see insider trades.")
            else:
                with st.spinner("Fetching insider trades..."):
                    _sm_idf = _sm_insider_trades(_sm_it_tkr)

                if _sm_idf.empty:
                    st.info(f"No insider trades found for {_sm_it_tkr}.")
                else:
                    # Apply filters
                    if _sm_it_type == "Buy":
                        _sm_idf = _sm_idf[_sm_idf["Type"] == "Buy"]
                    elif _sm_it_type == "Sale":
                        _sm_idf = _sm_idf[_sm_idf["Type"] == "Sale"]
                    if "Date" in _sm_idf.columns:
                        _sm_idf = _sm_idf[_sm_idf["Date"] >= pd.Timestamp(_sm_it_since)]

                    if _sm_idf.empty:
                        st.info("No trades match those filters.")
                    else:
                        # Summary
                        _sm_buy_cnt = len(_sm_idf[_sm_idf["Type"] == "Buy"])
                        _sm_sell_cnt = len(_sm_idf[_sm_idf["Type"] == "Sale"])
                        _sm_buy_shares = _sm_idf.loc[_sm_idf["Type"] == "Buy", "Shares"].sum()
                        _sm_sell_shares = _sm_idf.loc[_sm_idf["Type"] == "Sale", "Shares"].sum()

                        _sm_ic1, _sm_ic2, _sm_ic3, _sm_ic4 = st.columns(4)
                        _sm_ic1.metric("Total Filings", f"{len(_sm_idf):,}")
                        _sm_ic2.metric("Buys / Sales", f"{_sm_buy_cnt} / {_sm_sell_cnt}")
                        _sm_ic3.metric(
                            "Shares Bought",
                            f"{_sm_buy_shares:,.0f}" if _sm_buy_shares < 1e6 else f"{_sm_buy_shares/1e6:.1f}M",
                        )
                        _sm_ic4.metric(
                            "Shares Sold",
                            f"{_sm_sell_shares:,.0f}" if _sm_sell_shares < 1e6 else f"{_sm_sell_shares/1e6:.1f}M",
                        )

                        st.markdown("---")

                        # Full table
                        st.markdown("### All Insider Trades")
                        _sm_id = _sm_idf.copy()

                        def _sm_style_insider(row):
                            t = str(row.get("Type", ""))
                            if t == "Buy":
                                return ["background-color: rgba(38, 166, 154, 0.15)"] * len(row)
                            elif t == "Sale":
                                return ["background-color: rgba(239, 83, 80, 0.15)"] * len(row)
                            return [""] * len(row)

                        _sm_id_cols = [c for c in [
                            "Date", "Name", "Type", "Shares", "SharesOwned", "IsDerivative",
                        ] if c in _sm_id.columns]
                        _sm_styled = _sm_id[_sm_id_cols].style.apply(_sm_style_insider, axis=1)
                        st.dataframe(_sm_styled, use_container_width=True, hide_index=True)

                    st.download_button(
                        chr(0x2B07) + " Download CSV",
                        data=_sm_idf.to_csv(index=False),
                        file_name="insider_trades.csv",
                        mime="text/csv",
                        key="sm_dl_insider",
                    )



# =============================================================
# Bloomberg direct API helper -- shared by all Global tabs
# Uses blpapi (pip install blpapi) when Bloomberg Terminal is running.
# Falls back to empty dict when session unavailable.
# =============================================================
@st.cache_data(ttl=60, show_spinner=False)
def _bbg_bdp(_secs: tuple, _flds: tuple, _batch_size: int = 50) -> dict:
    """BDP via blpapi. Returns {security: {field: float or nan}}.
    Sends requests in batches within a SINGLE session to avoid
    Bloomberg rejecting rapid successive connections.
    Returns empty dict if no real (non-NaN) values were retrieved."""
    import math as _bmath
    try:
        import blpapi as _blp
        _opts = _blp.SessionOptions()
        _opts.setServerHost("localhost")
        _opts.setServerPort(8194)
        _sess = _blp.Session(_opts)
        if not _sess.start():
            return {}
        if not _sess.openService("//blp/refdata"):
            _sess.stop(); return {}
        _svc = _sess.getService("//blp/refdata")
        _out = {}
        _real_count = 0
        # Send in batches within the same session
        for _bi in range(0, max(len(_secs), 1), _batch_size):
            _batch = _secs[_bi:_bi + _batch_size]
            if not _batch:
                break
            _req = _svc.createRequest("ReferenceDataRequest")
            for _s in _batch:
                _req.append("securities", _s)
            for _f in _flds:
                _req.append("fields", _f)
            _sess.sendRequest(_req)
            while True:
                _ev = _sess.nextEvent(10000)
                for _msg in _ev:
                    if str(_msg.messageType()) == "ReferenceDataResponse":
                        _sd = _msg.getElement("securityData")
                        for _i in range(_sd.numValues()):
                            _se = _sd.getValueAsElement(_i)
                            _sec = _se.getElementAsString("security")
                            _fd = _se.getElement("fieldData")
                            _out[_sec] = {}
                            for _f in _flds:
                                try:
                                    _v = _fd.getElementAsFloat(_f)
                                    _out[_sec][_f] = _v
                                    if not _bmath.isnan(_v):
                                        _real_count += 1
                                except Exception:
                                    try:
                                        if _fd.hasElement(_f):
                                            _v = float(_fd.getElement(_f).getValue())
                                            _out[_sec][_f] = _v
                                            if not _bmath.isnan(_v):
                                                _real_count += 1
                                        else:
                                            _out[_sec][_f] = float("nan")
                                    except Exception:
                                        _out[_sec][_f] = float("nan")
                if _ev.eventType() in (_blp.Event.RESPONSE, _blp.Event.TIMEOUT):
                    break
        _sess.stop()
        if _real_count == 0:
            return {}
        return _out
    except Exception:
        return {}


# ── Calendar: mixed-type BDP fetch (handles strings like BMO/AMC) ─────────
_CAL_BBG_FIELDS = (
    'EARN_ANN_DT', 'NEXT_EARN_DT', 'EARN_ANN_DT_TIME_HIST',
    'BEST_EPS', 'BEST_SALES', 'BEST_TARGET_PRICE',
    'DVD_EX_DT', 'DVD_PAY_DT', 'DVD_CASH_GROSS', 'EQY_DVD_SH_12M',
    'ANNOUNCEMENT_DT',
)

@st.cache_data(ttl=120, show_spinner=False)
def _cal_bbg_fetch(_secs: tuple, _flds: tuple = _CAL_BBG_FIELDS,
                   _batch_size: int = 50) -> dict:
    """BDP via blpapi returning mixed types (float/str/date).
    Returns {security: {field: value}} where value can be float, str, or None."""
    import math as _bmath
    try:
        import blpapi as _blp
        _opts = _blp.SessionOptions()
        _opts.setServerHost("localhost")
        _opts.setServerPort(8194)
        _sess = _blp.Session(_opts)
        if not _sess.start():
            return {}
        if not _sess.openService("//blp/refdata"):
            _sess.stop(); return {}
        _svc = _sess.getService("//blp/refdata")
        _out = {}
        _real_count = 0
        for _bi in range(0, max(len(_secs), 1), _batch_size):
            _batch = _secs[_bi:_bi + _batch_size]
            if not _batch:
                break
            _req = _svc.createRequest("ReferenceDataRequest")
            for _s in _batch:
                _req.append("securities", _s)
            for _f in _flds:
                _req.append("fields", _f)
            _sess.sendRequest(_req)
            while True:
                _ev = _sess.nextEvent(10000)
                for _msg in _ev:
                    if str(_msg.messageType()) == "ReferenceDataResponse":
                        _sd = _msg.getElement("securityData")
                        for _i in range(_sd.numValues()):
                            _se = _sd.getValueAsElement(_i)
                            _sec = _se.getElementAsString("security")
                            _fd = _se.getElement("fieldData")
                            _out[_sec] = {}
                            for _f in _flds:
                                try:
                                    if not _fd.hasElement(_f):
                                        _out[_sec][_f] = None; continue
                                    _el = _fd.getElement(_f)
                                    # Try float first
                                    try:
                                        _v = _el.getValueAsFloat()
                                        if not _bmath.isnan(_v):
                                            _out[_sec][_f] = _v
                                            _real_count += 1
                                        else:
                                            _out[_sec][_f] = None
                                    except Exception:
                                        # Fall back to generic getValue (string/date)
                                        _raw = _el.getValue()
                                        _out[_sec][_f] = str(_raw) if _raw is not None else None
                                        if _raw is not None and str(_raw) not in ('', 'nan', 'NaT'):
                                            _real_count += 1
                                except Exception:
                                    _out[_sec][_f] = None
                if _ev.eventType() in (_blp.Event.RESPONSE, _blp.Event.TIMEOUT):
                    break
        _sess.stop()
        if _real_count == 0:
            return {}
        return _out
    except Exception:
        return {}



# =============================================================
# Index Breadth tab
# =============================================================
if nav == 'Index Breadth':
    import pandas as _ib_pd
    st.subheader('📊 Index & Sector Breadth')
    _IB_XLS = os.path.join(os.path.expanduser("~"), "Documents",
                           "Futures Spread2xlsb (version 1).xlsx")
    if not os.path.exists(_IB_XLS):
        st.error("Futures Spread workbook not found.")
        st.stop()

    @st.cache_data(ttl=60, show_spinner=False)
    def _ib_load(path):
        raw = _ib_pd.read_excel(path, sheet_name="Futures", header=None)
        # -- Futures section rows 2-5 --
        futs = raw.iloc[2:6, [0,1,2,3,6,7,12,13,14]].copy()
        futs.columns = ["Ticker","Last","Change","Chg%","VWAP","VWAP Diff","MTD%","QTD%","YTD%"]
        futs = futs[futs["Ticker"].notna()].reset_index(drop=True)
        futs.insert(0, "Name", ["S&P 500 Fut","Nasdaq Fut","Russell Fut","DJIA Fut"])
        # -- Breadth section -- find start row (has "Last" in col 2)
        br_start = 28
        for _ri in range(20, 45):
            if str(raw.iloc[_ri, 2]) == "Last":
                br_start = _ri + 1
                break
        rows = []
        for _ri in range(br_start, len(raw)):
            tk = raw.iloc[_ri, 0]
            if _ib_pd.isna(tk) or not str(tk).strip():
                break  # stop at first gap — avoids picking up AAII/MSXX noise rows
            _lbl = str(raw.iloc[_ri,1]).strip() if not _ib_pd.isna(raw.iloc[_ri,1]) else ""
            if not _lbl or _lbl == str(tk).strip():
                # No display label: clean up Bloomberg ticker suffix
                _lbl = str(tk).strip()
                for _sfx in [" Index", " Equity", " Comdty", " Curncy"]:
                    if _lbl.endswith(_sfx):
                        _lbl = _lbl[:-len(_sfx)]; break
            rows.append({"BBTicker": str(tk).strip(),
                         "Label":    _lbl,
                         "Last":     raw.iloc[_ri,2],
                         "Pt Chg":   raw.iloc[_ri,3],
                         "Chg%":     raw.iloc[_ri,4],
                         "%Abv10d":  raw.iloc[_ri,5],
                         "%52wHi":   raw.iloc[_ri,6],
                         "%52wLo":   raw.iloc[_ri,7],
                         "%AbvBB":   raw.iloc[_ri,8],
                         "%BlwBB":   raw.iloc[_ri,9],
                         "%Abv20d":  raw.iloc[_ri,10],
                         "%Abv50d":  raw.iloc[_ri,11],
                         "%Abv200d": raw.iloc[_ri,12],
                         "%RSI>70":  raw.iloc[_ri,13],
                         "%RSI<30":  raw.iloc[_ri,14],
                         "RSI 3d":   raw.iloc[_ri,15],
                         "RSI 14d":  raw.iloc[_ri,16],
                         "RSI 30d":  raw.iloc[_ri,17],
                         "Volume":   raw.iloc[_ri,18],
                         "20d Avg Vol": raw.iloc[_ri,19],
                         "MTD%":     raw.iloc[_ri,20],
                         "QTD%":     raw.iloc[_ri,21],
                         "YTD%":     raw.iloc[_ri,22],
                         "52w High": raw.iloc[_ri,23],
                         "%Off Hi":  raw.iloc[_ri,24],
                         "Hi Date":  raw.iloc[_ri,25],
                         "Members":  raw.iloc[_ri,26]})
        breadth = _ib_pd.DataFrame(rows)
        # -- Worldwide sheet -- track regional groups ----------------------
        _ww_raw = _ib_pd.read_excel(path, sheet_name="Worldwide", header=0)
        _ww_raw = _ww_raw.dropna(subset=[_ww_raw.columns[0]])
        _ww_px_col = _ww_raw.columns[1]  # "Curr Px"
        _curr_grp = "US Indices"
        _grp_rows = []
        for _, _wrow in _ww_raw.iterrows():
            if _ib_pd.isna(_wrow[_ww_px_col]):
                _curr_grp = str(_wrow.iloc[0]).strip()
            else:
                _rd = _wrow.to_dict()
                _rd["_Group"] = _curr_grp
                _grp_rows.append(_rd)
        ww = _ib_pd.DataFrame(_grp_rows) if _grp_rows else _ww_raw.iloc[0:0]
        return futs, breadth, ww

    _ib_c1, _ib_c2 = st.columns([6,1])
    if _ib_c2.button("🔄 Reload", key="ib_reload"):
        st.cache_data.clear(); st.rerun()
    try:
        _ib_futs, _ib_br, _ib_ww = _ib_load(_IB_XLS)
    except Exception as _ibe:
        st.error(f"Could not load Futures Spread workbook: {_ibe}")
        st.stop()

    # ── Live futures data via blpapi ───────────────────────────────────────
    _IB_FUT_SECS = ("ESA Index", "NQA Index", "RTYA Index", "DMA Index")
    _IB_FUT_FLDS = ("PX_LAST", "CHG_NET_1D", "CHG_PCT_1D",
                    "PER_TRADE_VWAP_REALTIME",
                    "CHG_PCT_MTD", "RETURN_MTD",
                    "CHG_PCT_QTD", "RETURN_QTD",
                    "CHG_PCT_YTD", "YTD_RETURN")
    _ib_bbg = _bbg_bdp(_IB_FUT_SECS, _IB_FUT_FLDS)
    if _ib_bbg:
        _IB_FUT_NAMES = {
            "ESA Index": "S&P 500 Fut", "NQA Index": "Nasdaq Fut",
            "RTYA Index": "Russell Fut", "DMA Index": "DJIA Fut",
        }
        import math as _ibm
        def _ib_get(sec, fld, fb=None):
            """Get field, fallback to fb field if NaN."""
            v = _ib_bbg[sec].get(fld, float("nan"))
            if fb and _ibm.isnan(float(v)): v = _ib_bbg[sec].get(fb, float("nan"))
            return v
        _ib_bbg_rows = [{
            "Name":      _IB_FUT_NAMES.get(_s, _s),
            "Ticker":    _s.split()[0],
            "Last":      _ib_get(_s, "PX_LAST"),
            "Change":    _ib_get(_s, "CHG_NET_1D"),
            "Chg%":      _ib_get(_s, "CHG_PCT_1D"),
            "VWAP":      _ib_get(_s, "PER_TRADE_VWAP_REALTIME"),
            "VWAP Diff": (lambda _l, _v: _l - _v if not (_ibm.isnan(float(_l)) or _ibm.isnan(float(_v))) else float("nan"))(_ib_get(_s, "PX_LAST"), _ib_get(_s, "PER_TRADE_VWAP_REALTIME")),
            "MTD%":      _ib_get(_s, "CHG_PCT_MTD", "RETURN_MTD"),
            "QTD%":      _ib_get(_s, "CHG_PCT_QTD", "RETURN_QTD"),
            "YTD%":      _ib_get(_s, "CHG_PCT_YTD", "YTD_RETURN"),
        } for _s in _IB_FUT_SECS if _s in _ib_bbg]
        if _ib_bbg_rows:
            _ib_futs = _ib_pd.DataFrame(_ib_bbg_rows)
        st.success("✅ Bloomberg connected (live data via API)")
    else:
        _ib_ok = _ib_pd.to_numeric(_ib_futs["Last"], errors="coerce").notna().any()
        if not _ib_ok:
            st.warning("⚠️ Bloomberg not connected — prices are stale. "
                       "Open the Futures Spread workbook in Excel with Bloomberg and click Reload.")
        else:
            st.info("ℹ️ Bloomberg via Excel (open API unavailable)")

    _ib_sort = st.radio("Sort breadth by:",
        ["Most Oversold (🔽 % above 10d)", "Most Overbought (🔼 % above 10d)", "Index name"],
        horizontal=True, key="ib_sort")

    st.markdown("#### 📈 Major Futures")
    _ib_fd = _ib_futs.copy()
    for _c in ["Last","Change","Chg%","VWAP","VWAP Diff","MTD%","QTD%","YTD%"]:
        if _c in _ib_fd.columns:
            _ib_fd[_c] = _ib_pd.to_numeric(_ib_fd[_c], errors="coerce")
    _FUT_FMT = {
        "Last":     lambda x: f"{x:,.2f}",   "Change":   lambda x: f"{x:+.2f}",
        "Chg%":     lambda x: f"{x:+.2f}%",  "VWAP":     lambda x: f"{x:,.2f}",
        "VWAP Diff":lambda x: f"{x:+.2f}",   "MTD%":     lambda x: f"{x:+.2f}%",
        "QTD%":     lambda x: f"{x:+.2f}%",  "YTD%":     lambda x: f"{x:+.2f}%",
    }
    for _fc, _ff in _FUT_FMT.items():
        if _fc in _ib_fd.columns:
            _ib_fd[_fc] = _ib_fd[_fc].apply(lambda v, _ff=_ff: "-" if _ib_pd.isna(v) else _ff(float(v)))
    _fut_cc = {
        "Name":     st.column_config.TextColumn("Name",      width="medium"),
        "Ticker":   st.column_config.TextColumn("Ticker",    width="small"),
        "Last":     st.column_config.TextColumn("Last",      width="small"),
        "Change":   st.column_config.TextColumn("Change",    width="small"),
        "Chg%":     st.column_config.TextColumn("Chg%",      width="small"),
        "VWAP":     st.column_config.TextColumn("VWAP",      width="small"),
        "VWAP Diff":st.column_config.TextColumn("VWAP Diff", width="small"),
        "MTD%":     st.column_config.TextColumn("MTD%",      width="small"),
        "QTD%":     st.column_config.TextColumn("QTD%",      width="small"),
        "YTD%":     st.column_config.TextColumn("YTD%",      width="small"),
    }
    st.dataframe(_ib_fd.style, hide_index=True, use_container_width=False,
                 column_config={k:v for k,v in _fut_cc.items() if k in _ib_fd.columns})

    st.markdown("#### \U0001f9ee Index & Sector Breadth")
    _ib_b = _ib_br.copy()
    _IB_NUM_COLS = ["Last","Pt Chg","Chg%","%Abv10d","%52wHi","%52wLo",
                    "%AbvBB","%BlwBB","%Abv20d","%Abv50d","%Abv200d",
                    "%RSI>70","%RSI<30","RSI 3d","RSI 14d","RSI 30d",
                    "Volume","20d Avg Vol","MTD%","QTD%","YTD%",
                    "52w High","%Off Hi","Members"]
    for _nc in _IB_NUM_COLS:
        if _nc in _ib_b.columns:
            _ib_b[_nc] = _ib_pd.to_numeric(_ib_b[_nc], errors="coerce")
    if "Most Oversold" in _ib_sort:
        _ib_b = _ib_b.sort_values("%Abv10d", ascending=True)
    elif "Most Overbought" in _ib_sort:
        _ib_b = _ib_b.sort_values("%Abv10d", ascending=False)
    else:
        _ib_b = _ib_b.sort_values("Label")
    _IB_SHOW_COLS = ["Label","Last","Pt Chg","Chg%","%Abv10d",
                     "%52wHi","%52wLo","%AbvBB","%BlwBB",
                     "%Abv20d","%Abv50d","%Abv200d",
                     "%RSI>70","%RSI<30","RSI 3d","RSI 14d","RSI 30d",
                     "Volume","20d Avg Vol","MTD%","QTD%","YTD%",
                     "52w High","%Off Hi","Hi Date","Members"]
    _ib_bshow = _ib_b[[c for c in _IB_SHOW_COLS if c in _ib_b.columns]].copy()
    _ib_bshow = _ib_bshow.dropna(subset=["%Abv10d"]).reset_index(drop=True)
    if not _ib_bshow.empty:
        _ib_bnum = _ib_bshow.copy()
        _IB_BFMT = {
            "Last":     lambda x: f"{x:,.2f}",   "Pt Chg":   lambda x: f"{x:+.2f}",
            "Chg%":     lambda x: f"{x:+.2f}%",  "%Abv10d":  lambda x: f"{x:.1f}%",
            "%52wHi":   lambda x: f"{x:.1f}%",   "%52wLo":   lambda x: f"{x:.1f}%",
            "%AbvBB":   lambda x: f"{x:.1f}%",   "%BlwBB":   lambda x: f"{x:.1f}%",
            "%Abv20d":  lambda x: f"{x:.1f}%",   "%Abv50d":  lambda x: f"{x:.1f}%",
            "%Abv200d": lambda x: f"{x:.1f}%",
            "%RSI>70":  lambda x: f"{x:.1f}%",   "%RSI<30":  lambda x: f"{x:.1f}%",
            "RSI 3d":   lambda x: f"{x:.1f}",    "RSI 14d":  lambda x: f"{x:.1f}",
            "RSI 30d":  lambda x: f"{x:.1f}",
            "Volume":   lambda x: f"{int(x):,}",
            "20d Avg Vol": lambda x: f"{int(x):,}",
            "MTD%":     lambda x: f"{x:+.2f}%",  "QTD%":     lambda x: f"{x:+.2f}%",
            "YTD%":     lambda x: f"{x:+.2f}%",
            "52w High": lambda x: f"{x:,.2f}",   "%Off Hi":  lambda x: f"{x:+.2f}%",
            "Members":  lambda x: f"{int(x)}",
        }
        for _bc, _bf in _IB_BFMT.items():
            if _bc in _ib_bshow.columns:
                _ib_bshow[_bc] = _ib_bshow[_bc].apply(
                    lambda v, _bf=_bf: "-" if _ib_pd.isna(v) else _bf(float(v)))
        # Format Hi Date
        if "Hi Date" in _ib_bshow.columns:
            _ib_bshow["Hi Date"] = _ib_bshow["Hi Date"].apply(
                lambda v: "-" if (_ib_pd.isna(v) or str(v) in ("","nan","None","NaT")) else str(v)[:10])
        try:
            import matplotlib as _mpl3
            def _bgrad(num_s, cmap_n, vmin, vmax):
                _cm3 = _mpl3.colormaps.get_cmap(cmap_n)
                _nm3 = _mpl3.colors.Normalize(vmin=vmin, vmax=vmax)
                def _fn3(col):
                    return [
                        f"background-color:rgba({int(r*255)},{int(g*255)},{int(b*255)},0.85)"
                        if not _ib_pd.isna(nv) else ""
                        for nv, (r, g, b, _a) in zip(
                            num_s, [_cm3(_nm3(min(max(float(v2),vmin),vmax)))
                                    for v2 in num_s.fillna(vmin)])
                    ]
                return _fn3
            _bsty = _ib_bshow.style
            if "%Abv10d" in _ib_bnum.columns and _ib_bnum["%Abv10d"].notna().any():
                _bsty = _bsty.apply(
                    _bgrad(_ib_bnum["%Abv10d"], "RdYlGn", 20, 80),
                    subset=["%Abv10d"])
            if "Chg%" in _ib_bnum.columns and _ib_bnum["Chg%"].notna().any():
                _bsty = _bsty.apply(
                    _bgrad(_ib_bnum["Chg%"], "RdYlGn", -3, 3),
                    subset=["Chg%"])
            if "RSI 14d" in _ib_bnum.columns and _ib_bnum["RSI 14d"].notna().any():
                _bsty = _bsty.apply(
                    _bgrad(_ib_bnum["RSI 14d"], "RdYlGn", 30, 70),
                    subset=["RSI 14d"])
            if "%Off Hi" in _ib_bnum.columns and _ib_bnum["%Off Hi"].notna().any():
                _bsty = _bsty.apply(
                    _bgrad(_ib_bnum["%Off Hi"], "RdYlGn", -20, 0),
                    subset=["%Off Hi"])
            st.dataframe(_bsty, hide_index=True, use_container_width=True)
        except Exception:
            st.dataframe(_ib_bshow, hide_index=True, use_container_width=True)
    else:
        st.info("No breadth data \u2014 connect Bloomberg and reload.")

    st.markdown("#### 🌍 Worldwide Breadth")
    _ib_wwd_raw = _ib_ww.copy()

    # ── Fetch extra fields for US Indices + SPX Sectors via blpapi ──────────
    _IB_EXTRA_GRPS = {"US Indices", "SPX Sectors"}
    _ib_extra_secs = ()
    _ib_extra_raw  = {}
    if "_Group" in _ib_wwd_raw.columns and "Ticker" not in _ib_wwd_raw.columns:
        # find ticker column
        _tc = _ib_wwd_raw.columns[0]
    else:
        _tc = "Ticker" if "Ticker" in _ib_wwd_raw.columns else _ib_wwd_raw.columns[0]
    _extra_mask = _ib_wwd_raw.get("_Group", _ib_pd.Series(dtype=str)).isin(_IB_EXTRA_GRPS)
    if _extra_mask.any():
        _ib_extra_secs = tuple(
            str(t).strip() for t in _ib_wwd_raw.loc[_extra_mask, _tc]
            if not _ib_pd.isna(t) and str(t).strip()
        )
    _IB_EXTRA_FLDS = ("CHG_PCT_MTD", "RETURN_MTD", "CHG_PCT_QTD", "RETURN_QTD",
                       "CHG_PCT_YTD", "YTD_RETURN", "RETURN_YTD",
                       "HIGH_52WEEK", "PX_HIGH_52WK", "HIGH_DT_52WEEK")
    if _ib_extra_secs:
        _ib_extra_raw = _bbg_bdp(_ib_extra_secs, _IB_EXTRA_FLDS)

    def _ib_extra_row(bbg_tk):
        """Return dict of extra fields for a given Bloomberg ticker."""
        _d = _ib_extra_raw.get(str(bbg_tk).strip(), {})
        _isnan = _ib_pd.isna
        _mtd = _d.get("CHG_PCT_MTD",  float("nan"))
        if _isnan(_mtd): _mtd = _d.get("RETURN_MTD",  float("nan"))
        _qtd = _d.get("CHG_PCT_QTD",  float("nan"))
        if _isnan(_qtd): _qtd = _d.get("RETURN_QTD",  float("nan"))
        _ytd = _d.get("CHG_PCT_YTD",  float("nan"))
        if _isnan(_ytd): _ytd = _d.get("YTD_RETURN",  float("nan"))
        if _isnan(_ytd): _ytd = _d.get("RETURN_YTD",  float("nan"))
        _hi52 = _d.get("HIGH_52WEEK", float("nan"))
        if _isnan(_hi52): _hi52 = _d.get("PX_HIGH_52WK", float("nan"))
        _hi_dt = _d.get("HIGH_DT_52WEEK", "")
        return {"MTD%": _mtd, "QTD%": _qtd, "YTD%": _ytd,
                "52W Hi": _hi52, "Hi Date": str(_hi_dt) if _hi_dt else ""}

    # numeric-coerce all cols except Ticker and _Group
    for _c2 in _ib_wwd_raw.columns:
        if _c2 not in ("Ticker", "_Group"):
            _ib_wwd_raw[_c2] = _ib_pd.to_numeric(_ib_wwd_raw[_c2], errors="coerce")
    _IB_WW_RN = {
        "Curr Px":"Last","Pct Chg on Day":"Chg%","# of Members":"Mbrs",
        "Daily Advancers - Decliners":"Adv-Dec","Weekly Advancers - Decliners":"Wk A-D",
        "Daily ARMS Index":"ARMS","Weekly ARMS Index":"Wk ARMS",
        "New 12 Week Highs %":"12W Hi","New 12 Week Lows %":"12W Lo",
        "New 52 Week Highs %":"52W Hi","New 52 Week Lows %":"52W Lo",
        "%  14D RSI > 70":"RSI>70","%  14D RSI < 30":"RSI<30",
        "% MACD Buy Last 10D":"MACD Buy","% MACD Sell Last 10D":"MACD Sel",
        "% Px > Upper Boll Bnd":"BB Hi%","% Px < Lower Boll Bnd":"BB Lo%",
    }
    _ib_wwd_raw = _ib_wwd_raw.rename(columns=_IB_WW_RN)
    _IB_WW_SHOW = ["Ticker","Last","Chg%","Mbrs","Adv-Dec","Wk A-D",
                   "ARMS","Wk ARMS","12W Hi","12W Lo","52W Hi","52W Lo",
                   "RSI>70","RSI<30"]
    _ib_disp_cols = [c for c in _IB_WW_SHOW if c in _ib_wwd_raw.columns]
    # pre-format strings, keep numeric copy for gradient
    _WW_RFMT = {
        "Last":    lambda x: f"{x:,.2f}", "Chg%": lambda x: f"{x:+.2f}%",
        "Mbrs":    lambda x: f"{int(x)}", "Adv-Dec": lambda x: f"{int(x):+d}",
        "Wk A-D":  lambda x: f"{int(x):+d}", "ARMS": lambda x: f"{x:.2f}",
        "Wk ARMS": lambda x: f"{x:.2f}",
        "12W Hi":  lambda x: f"{x:.1f}", "12W Lo":  lambda x: f"{x:.1f}",
        "52W Hi":  lambda x: f"{x:.1f}", "52W Lo":  lambda x: f"{x:.1f}",
        "RSI>70":  lambda x: f"{x:.1f}", "RSI<30":  lambda x: f"{x:.1f}",
    }
    import matplotlib as _mpl2
    _IB_WW_GRP_ICONS = {
        "US Indices":"📈",
        "SPX Sectors":"📊",
        "LATAM":"🌎",
        "Europe":"🇪🇺",
        "MID-EAST":"🌍",
        "ASIA PACIFIC":"🌏",
    }
    _ww_cc_base = {
        "Ticker":    st.column_config.TextColumn("Ticker",    width="small"),
        "Last":      st.column_config.TextColumn("Last",      width="small"),
        "Chg%":      st.column_config.TextColumn("Chg%",      width="small"),
        "MTD%":      st.column_config.TextColumn("MTD%",      width="small"),
        "QTD%":      st.column_config.TextColumn("QTD%",      width="small"),
        "YTD%":      st.column_config.TextColumn("YTD%",      width="small"),
        "52W Hi":    st.column_config.TextColumn("52W Hi",    width="small"),
        "% Off Hi":  st.column_config.TextColumn("% Off Hi",  width="small"),
        "Hi Date":   st.column_config.TextColumn("Hi Date",   width="small"),
        "Mbrs":     st.column_config.TextColumn("Mbrs",     width="small"),
        "Adv-Dec":  st.column_config.TextColumn("Adv-Dec",  width="small"),
        "Wk A-D":   st.column_config.TextColumn("Wk A-D",   width="small"),
        "ARMS":     st.column_config.TextColumn("ARMS",     width="small"),
        "Wk ARMS":  st.column_config.TextColumn("Wk ARMS",  width="small"),
        "12W Hi":   st.column_config.TextColumn("12W Hi",   width="small"),
        "12W Lo":   st.column_config.TextColumn("12W Lo",   width="small"),
        "52W Hi":   st.column_config.TextColumn("52W Hi",   width="small"),
        "52W Lo":   st.column_config.TextColumn("52W Lo",   width="small"),
        "RSI>70":   st.column_config.TextColumn("RSI>70",   width="small"),
        "RSI<30":   st.column_config.TextColumn("RSI<30",   width="small"),
    }
    if "_Group" in _ib_wwd_raw.columns and not _ib_wwd_raw.empty:
        for _grp in _ib_wwd_raw["_Group"].unique():
            _icon = _IB_WW_GRP_ICONS.get(_grp, "📍")
            st.markdown(f"##### {_icon} {_grp}")
            _g_rows = _ib_wwd_raw[_ib_wwd_raw["_Group"]==_grp].reset_index(drop=True)
            _g_num = _g_rows[_ib_disp_cols].reset_index(drop=True)
            # For US Indices + SPX Sectors, add extra blpapi columns
            import math as _ibm2
            _exfmt = {}  # always defined to avoid NameError
            if _grp in _IB_EXTRA_GRPS:
                _exfmt = {"MTD%": lambda x: f"{x:+.2f}%",
                          "QTD%": lambda x: f"{x:+.2f}%",
                          "YTD%": lambda x: f"{x:+.2f}%",
                          "52W Hi": lambda x: f"{x:,.2f}",
                          "% Off Hi": lambda x: f"{x:+.2f}%",
                          "Hi Date": lambda x: str(x)}
                for _ecol in ("MTD%","QTD%","YTD%","52W Hi","% Off Hi","Hi Date"):
                    _g_num[_ecol] = None
                # Debug expander — shows raw Bloomberg return for first row
                if _ib_extra_raw:
                    _first_tk = str(_g_rows["Ticker"].iloc[0]).strip() if "Ticker" in _g_rows.columns else ""
                    _dbg_d = _ib_extra_raw.get(_first_tk, {})
                    with st.expander(f"🔍 Bloomberg debug: {_grp} ({_first_tk})", expanded=False):
                        st.write({k: round(v,4) if not _ibm2.isnan(float(v)) else "NaN" for k,v in _dbg_d.items() if isinstance(v,(int,float))})
                        if not _dbg_d:
                            st.warning("No data returned for this ticker — check Bloomberg connection or field names.")
                for _ri2, _grow in _g_rows.iterrows():
                    _btk = str(_grow.get("Ticker", _grow.get(_tc, ""))).strip()
                    _ex = _ib_extra_row(_btk) if _ib_extra_raw else {"MTD%": float("nan"), "QTD%": float("nan"), "YTD%": float("nan"), "52W Hi": float("nan"), "Hi Date": ""}
                    # also compute %Off Hi
                    _last_v = _ib_pd.to_numeric(_grow.get("Last", float("nan")), errors="coerce")
                    _hi_v   = _ex["52W Hi"]
                    _poff   = float("nan")
                    if not (_ib_pd.isna(_last_v) or _ib_pd.isna(_hi_v) or
                            _ibm2.isnan(float(_hi_v)) or float(_hi_v) == 0):
                        _poff = (float(_last_v) - float(_hi_v)) / float(_hi_v) * 100
                    _g_num.at[_ri2, "MTD%"]    = _ex["MTD%"]
                    _g_num.at[_ri2, "QTD%"]    = _ex["QTD%"]
                    _g_num.at[_ri2, "YTD%"]    = _ex["YTD%"]
                    _g_num.at[_ri2, "52W Hi"]  = _ex["52W Hi"]
                    _g_num.at[_ri2, "% Off Hi"] = _poff
                    _g_num.at[_ri2, "Hi Date"]  = _ex["Hi Date"]
            _g_str = _g_num.copy()
            for _wc, _wf in _WW_RFMT.items():
                if _wc in _g_str.columns:
                    _g_str[_wc] = _g_str[_wc].apply(
                        lambda v, _wf=_wf: "-" if _ib_pd.isna(v) else _wf(float(v)))
            # Format extra columns — and drop any that are ALL empty/NaN
            if _grp in _IB_EXTRA_GRPS:
                for _ec2, _ef2 in _exfmt.items():
                    if _ec2 in _g_str.columns:
                        if _ec2 == "Hi Date":
                            _g_str[_ec2] = _g_str[_ec2].apply(
                                lambda v: "-" if (not v or str(v) in ("-","nan","None","")) else str(v)[:10])
                        else:
                            _g_str[_ec2] = _g_num[_ec2].apply(
                                lambda v, _ef2=_ef2: "-" if (_ib_pd.isna(v) or str(v) in ("","nan","None")) else _ef2(float(v)))
                # Drop extra columns that have zero real data (all "-" or NaN)
                for _ec3 in ("MTD%","QTD%","YTD%","52W Hi","% Off Hi","Hi Date"):
                    if _ec3 in _g_str.columns:
                        _has_data = _g_str[_ec3].apply(
                            lambda v: str(v).strip() not in ("-", "", "nan", "None", "NaN")).any()
                        if not _has_data:
                            _g_str.drop(columns=[_ec3], inplace=True)
                            if _ec3 in _g_num.columns:
                                _g_num.drop(columns=[_ec3], inplace=True)
            try:
                _gsty = _g_str.style
                def _wgrad2(num_s, cmap_n, vmin, vmax):
                    _cm2 = _mpl2.colormaps.get_cmap(cmap_n)
                    _nm2 = _mpl2.colors.Normalize(vmin=vmin, vmax=vmax)
                    def _fn2g(col):
                        return [
                            f"background-color:rgba({int(r*255)},{int(g*255)},{int(b*255)},0.85)"
                            if not _ib_pd.isna(nv) else ""
                            for nv, (r, g, b, _a) in zip(
                                num_s, [_cm2(_nm2(min(max(float(v2),vmin),vmax)))
                                        for v2 in num_s.fillna(vmin)])
                        ]
                    return _fn2g
                if "Chg%" in _g_num.columns and _g_num["Chg%"].notna().any():
                    _gsty = _gsty.apply(_wgrad2(_g_num["Chg%"], "RdYlGn", -3, 3),
                                        subset=["Chg%"])
                st.dataframe(_gsty, hide_index=True, use_container_width=False,
                             column_config={k:v for k,v in _ww_cc_base.items()
                                            if k in _g_str.columns})
            except Exception:
                st.dataframe(_g_str, hide_index=True, use_container_width=False)
    else:
        st.info("No worldwide breadth data.")


# =============================================================
# 10-Day Screen tab
# =============================================================
if nav == '10-Day Screen':
    import pandas as _td_pd
    st.subheader('📅 10-Day Screen')

    # ── Custom universe (blpapi-based, no Excel required) ───────────────────
    _TD_UNIVERSE = (
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NFLX", "TSLA", "SPOT",
    "DASH", "APP", "RDDT", "MAGS", "SMH", "NVDA", "AMD", "TSM",
    "AVGO", "MRVL", "AMAT", "LRCX", "ASML", "ON", "TXN", "INTC",
    "ANET", "DELL", "SMCI", "NBIS", "CRWV", "CLS", "SNDK", "MU",
    "WDC", "STX", "SOLS", "AXTI", "AEHR", "TER", "COHR", "LITE",
    "GLW", "AAOI", "CIEN", "ALAB", "CRDO", "IGV", "PLTR", "ORCL",
    "NOW", "CRWD", "PANW", "NET", "ZS", "SNOW", "DDOG", "ADBE",
    "FIG", "CRM", "TEAM", "SHOP", "UPST", "AFRM", "OKTA", "TTD",
    "DOCU", "TWLO", "MDB", "ESTC", "DOCS", "BBAI", "MNDY", "CFLT",
    "WDAY", "IBIT", "ETHA", "SOLZ", "BITO", "XBTUSD", "XETUSD", "COIN",
    "HOOD", "GLXY", "CRCL", "MSTR", "RIOT", "MARA", "IREN", "HUT",
    "CIFR", "WULF", "APLD", "SOUN", "IONQ", "RGTI", "INFQ", "QBTS",
    "QUBT", "QMCO", "ARQQ", "BE", "CCJ", "CEG", "GEV", "OKLO",
    "NNE", "SMR", "LEU", "NRG", "VRT", "VST", "CW", "UUUU",
    "UEC", "URA", "NEE", "PWR", "JOBY", "ACHR", "ONDS", "OUST",
    "AEVA", "LASR", "LPTH", "SERV", "RR", "AVAV", "KTOS", "DPRO",
    "RCAT", "UMAC", "LMT", "RKLB", "ASTS", "SATS", "LUNR", "RDW",
    "FLY", "SPIR", "BKSY", "KRMN", "VOYG", "VSAT", "SIDU", "VELO",
    "MP", "METC", "USAR", "UAMY", "KWEB", "BABA", "BIDU", "SE",
    "JD", "VNET", "GDS", "PDD", "ARKG", "TEM", "RXRX", "DNA",
    "SDGR", "TWST", "CRSP", "BEAM", "KRE", "GS", "MS", "BAC",
    "JPM", "C", "WFC", "APO", "KKR", "BX", "XYZ", "PYPL",
    "MA", "V", "AXP", "SOFI", "TRAN", "FDX", "UPS", "CSX",
    "UNP", "JETS", "UAL", "DAL", "LUV", "UBER", "LYFT", "ABNB",
    "CCL", "RCL", "NCLH", "EXPE", "MAR", "DIS", "ITB", "TOL",
    "PHM", "DHI", "KBH", "HD", "LOW", "Z", "XLP", "PG",
    "PEP", "KO", "NKE", "WMT", "COST", "XLV", "JNJ", "PFE",
    "MRK", "LLY", "UNH", "CI", "HUM", "XRT", "KSS", "VSCO",
    "BBY", "TGT", "XLY", "LULU", "CROX", "RH", "TPR", "CAVA",
    "MCD", "SBUX", "CMG", "SHAK", "BYND", "F", "GM", "CVNA",
    "RIVN", "NIO", "LCID", "XLI", "CAT", "URI", "CMI", "HON",
    "GE", "BA", "TDG", "TAN", "SEDG", "ENPH", "FSLR", "PLUG",
    "LVS", "WYNN", "PENN", "DKNG", "CSCO", "IBM", "SNAP", "PINS",
    "IAC", "LUMN", "WBD", "CMCSA", "T", "VZ", "XLE", "XOP",
    "XOM", "CVX", "COP", "OXY", "APA", "EOG", "DVN", "VLO",
    "MPC", "XME", "FCX", "CLF", "MT", "GDX", "PTON", "CME",
    "IBB", "XBI", "AMGN", "GILD", "MRNA", "NVAX", "VKTX", "IYR",
    "AMT", "SPG", "DLR", "EQIX", "DBRG", "CRML", "TMQ", "TMC",
    "NVA", "NMG", "AREC", "LAC", "SQM", "XLU", "EXC", "DUK",
    "SO", "KULR", "NTR", "MOS", "CF", "HYM", "OMDA", "TTWO",
)
    _TD_CURNCY_SYMS = frozenset({'XBTUSD', 'XETUSD'})
    _TD_ETF_SYMS = frozenset({
        "IBIT","ETHA","SOLZ","BITO","XLP","XLV","XRT","XLY","XLI",
        "XLE","XOP","XME","XLU","MAGS","SMH","IGV","KWEB","ARKG",
        "KRE","JETS","ITB","IBB","XBI","IYR","URA","TAN","GDX",
    })

    _TD_XLS = os.path.join(os.path.expanduser("~"), "Documents",
                           "10Day Screen 2025v2.xls")
    _TD_SI  = os.path.join(os.environ.get("PER_TICKER_PARQUET_DIR",""), "_signals_index.parquet")

    @st.cache_data(ttl=60, show_spinner=False)
    def _td_load(path):
        df = _td_pd.read_excel(path, sheet_name="BETA List", header=1)
        df = df.dropna(subset=["Symbol"]).copy()
        df["Symbol"] = df["Symbol"].astype(str).str.strip()
        df = df[df["Symbol"] != "Symbol"].reset_index(drop=True)
        keep = ["Symbol","Type","Last","10Day MA","Upper Boll","Lower Boll",
                "Percent Above or Below 10Day","RSI 14D","RSI 3D","MTD Chg","YTD chg"]
        df = df[[c for c in keep if c in df.columns]]
        _num = ["Last","10Day MA","Upper Boll","Lower Boll",
                "Percent Above or Below 10Day","RSI 14D","RSI 3D","MTD Chg","YTD chg"]
        for _c in _num:
            if _c in df.columns:
                df[_c] = _td_pd.to_numeric(df[_c], errors="coerce")
        return df

    @st.cache_data(ttl=120, show_spinner=False)
    def _td_si_load(path):
        if path and os.path.exists(path):
            return _td_pd.read_parquet(path)
        return None

    _td_c1, _td_c2 = st.columns([6,1])
    if _td_c2.button("🔄 Reload", key="td_reload"):
        st.cache_data.clear(); st.rerun()

    _td_si = _td_si_load(_TD_SI)

    # Build skeleton DataFrame from custom universe
    @st.cache_data(ttl=3600, show_spinner=False)
    def _td_build_uni():
        _rows = []
        for _s in _TD_UNIVERSE:
            _t = ('Curncy' if _s in _TD_CURNCY_SYMS
                  else 'ETF' if _s in _TD_ETF_SYMS else 'Equity')
            _rows.append({'Symbol': _s, 'Type': _t,
                          'Last': float('nan'), '10Day MA': float('nan'),
                          'Upper Boll': float('nan'), 'Lower Boll': float('nan'),
                          'Percent Above or Below 10Day': float('nan'),
                          'RSI 14D': float('nan'), 'RSI 3D': float('nan'),
                          'MTD Chg': float('nan'), 'YTD chg': float('nan')})
        return _td_pd.DataFrame(_rows)

    _td_df = _td_build_uni()

    # Optional: load RSI3D from Excel if available
    if os.path.exists(_TD_XLS):
        try:
            _td_xls = _td_load(_TD_XLS)
            if 'RSI 3D' in _td_xls.columns and _td_xls['RSI 3D'].notna().any():
                _rsi3_map = (_td_xls[['Symbol','RSI 3D']]
                             .dropna(subset=['RSI 3D'])
                             .set_index('Symbol')['RSI 3D'].to_dict())
                for _idx2, _row2 in _td_df.iterrows():
                    if _row2['Symbol'] in _rsi3_map:
                        _td_df.at[_idx2, 'RSI 3D'] = _rsi3_map[_row2['Symbol']]
        except Exception:
            pass


    # ── Build Bloomberg tickers from Symbol + Type column ─────────────────
    def _td_bbg_tk(sym, typ):
        t = str(typ).strip().lower()
        if t in ("equity", "etf"):    return f"{sym} US Equity"
        if t == "index":              return f"{sym} Index"
        if t in ("comdty", "cmdty"): return f"{sym} Comdty"
        if t == "curncy":             return f"{sym} Curncy"
        return f"{sym} US Equity"

    _TD_BBG_FLDS = ("PX_LAST", "MOV_AVG_10D", "RSI_14D",
                    "RETURN_MTD",  "CHG_PCT_MTD",
                    "YTD_RETURN",  "CHG_PCT_YTD",
                    "EQY_BOLLINGER_UPPER", "EQY_BOLLINGER_LOWER")
    if "Type" in _td_df.columns:
        _td_tks = tuple(
            _td_bbg_tk(str(r["Symbol"]).strip(), r.get("Type", "Equity"))
            for _, r in _td_df.iterrows())
    else:
        _td_tks = tuple(f"{str(r['Symbol']).strip()} US Equity"
                        for _, r in _td_df.iterrows())

    _td_bbg = _bbg_bdp(_td_tks, _TD_BBG_FLDS)

    if _td_bbg:
        _td_has_px = sum(1 for _v in _td_bbg.values()
                         if not __import__('math').isnan(_v.get("PX_LAST", float("nan"))))
        st.success(f"✅ Bloomberg connected — {_td_has_px}/{len(_td_bbg)} tickers with prices")
        import math as _tdm
        for _idx, _row in _td_df.iterrows():
            _sym = str(_row["Symbol"]).strip()
            _typ = str(_row.get("Type", "Equity")).strip() if "Type" in _td_df.columns else "Equity"
            _tk = _td_bbg_tk(_sym, _typ)
            if _tk in _td_bbg:
                _d = _td_bbg[_tk]
                # Only overwrite if blpapi returned a real value -- preserve Excel data otherwise
                _px  = _d.get("PX_LAST",       float("nan"))
                _ma  = _d.get("MOV_AVG_10D",   float("nan"))
                _rsi = _d.get("RSI_14D",       float("nan"))
                _mtd = _d.get("CHG_PCT_MTD",  float("nan"))
                if _tdm.isnan(_mtd): _mtd = _d.get("RETURN_MTD",  float("nan"))
                _ytd = _d.get("CHG_PCT_YTD",  float("nan"))
                if _tdm.isnan(_ytd): _ytd = _d.get("YTD_RETURN",  float("nan"))
                _ubb = _d.get("EQY_BOLLINGER_UPPER", float("nan"))
                _lbb = _d.get("EQY_BOLLINGER_LOWER", float("nan"))
                if not _tdm.isnan(_px):  _td_df.at[_idx, "Last"]    = _px
                if not _tdm.isnan(_ma):  _td_df.at[_idx, "10Day MA"] = _ma
                if not _tdm.isnan(_rsi): _td_df.at[_idx, "RSI 14D"]  = _rsi
                if not _tdm.isnan(_mtd): _td_df.at[_idx, "MTD Chg"]  = _mtd
                if not _tdm.isnan(_ytd): _td_df.at[_idx, "YTD chg"]  = _ytd
                if not _tdm.isnan(_ubb): _td_df.at[_idx, "Upper Boll"] = _ubb
                if not _tdm.isnan(_lbb): _td_df.at[_idx, "Lower Boll"] = _lbb
                if not (_tdm.isnan(_px) or _tdm.isnan(_ma) or _ma == 0):
                    _td_df.at[_idx, "Percent Above or Below 10Day"] = (_px - _ma) / _ma * 100
    else:
        # Fallback: check if Excel had live Bloomberg data
        _td_bberg = _td_pd.to_numeric(_td_df["Last"], errors="coerce").notna().sum() > 5
        if not _td_bberg:
            st.warning("⚠️ Bloomberg not connected. Falling back to parquet data where available.")
            if _td_si is not None:
                for _idx, _row in _td_df.iterrows():
                    _sym = str(_row["Symbol"]).strip()
                    _si_match = (_td_si[_td_si["Ticker"] == _sym]
                                 if "Ticker" in _td_si.columns else _td_pd.DataFrame())
                    if not _si_match.empty:
                        _sir = _si_match.iloc[0]
                        if _td_pd.isna(_row.get("Last")):
                            _td_df.at[_idx, "Last"] = _sir.get("Close", float("nan"))
                        if _td_pd.isna(_row.get("RSI 14D")):
                            _td_df.at[_idx, "RSI 14D"] = _sir.get("RSI14", float("nan"))
                        if _td_pd.isna(_row.get("Percent Above or Below 10Day")):
                            _td_df.at[_idx, "Percent Above or Below 10Day"] = _sir.get("Close_to_SMA10", float("nan"))
        else:
            st.info("ℹ️ Bloomberg via Excel (open API unavailable)")

    _td_col_a, _td_col_b, _td_col_c = st.columns(3)
    _td_sort = _td_col_a.selectbox("Sort by",
        ["% From 10d MA","RSI14","RSI3","MTD%","YTD%"],
        key="td_sort")
    _td_asc  = _td_col_b.checkbox("Ascending (most oversold first)", value=True, key="td_asc")
    _td_view = _td_col_c.selectbox("Show",
        ["All","Indices Only","Stocks Only"], key="td_view")

    _TD_IDX_SYMS = ["INDU","SPX","CCMP","RTY","NDX","CLA","GCA","NGA","JYA","VXX","TLT","XBTUSD","XETUSD"]
    _td_is_idx = (_td_df["Symbol"].isin(_TD_IDX_SYMS)
                  | _td_df["Symbol"].isin(_TD_ETF_SYMS))
    if "Type" in _td_df.columns:
        _td_is_idx = _td_is_idx | (_td_df["Type"].astype(str).str.lower()
                                   .str.contains("index|etf|cmdty|curncy", na=False))

    _TD_SORT_MAP = {"% From 10d MA": "Percent Above or Below 10Day",
                    "RSI14": "RSI 14D", "RSI3": "RSI 3D",
                    "MTD%": "MTD Chg", "YTD%": "YTD chg"}
    _td_sort_col = _TD_SORT_MAP.get(_td_sort, _td_sort)
    if _td_sort_col not in _td_df.columns:
        _td_sort_col = "RSI 14D"

    _td_disp_cols = ["Symbol","Last","10Day MA","Upper Boll","Lower Boll",
                     "Percent Above or Below 10Day","RSI 14D","RSI 3D","MTD Chg","YTD chg"]
    _td_disp_cols = [c for c in _td_disp_cols if c in _td_df.columns]
    _TD_RENAME = {"Percent Above or Below 10Day": "% vs 10d",
                  "10Day MA": "10d MA", "RSI 14D": "RSI14",
                  "RSI 3D": "RSI3", "MTD Chg": "MTD%", "YTD chg": "YTD%",
                  "Upper Boll": "UpperBB", "Lower Boll": "LowerBB"}

    import math as _tdmath
    def _td_fmt_v(col):
        def _d(v, fn):
            try:
                if v is None or (isinstance(v, float) and _tdmath.isnan(v)): return "-"
                return fn(v)
            except Exception: return "-"
        _m = {
            "Last":     lambda v: _d(v, lambda x: f"{x:,.2f}"),
            "10Day MA": lambda v: _d(v, lambda x: f"{x:,.2f}"),
            "Percent Above or Below 10Day": lambda v: _d(v, lambda x: f"{x:+.2f}%"),
            "RSI 14D":  lambda v: _d(v, lambda x: f"{x:.1f}"),
            "RSI 3D":   lambda v: _d(v, lambda x: f"{x:.1f}"),
            "MTD Chg":  lambda v: _d(v, lambda x: f"{x:+.2f}%"),
            "YTD chg":  lambda v: _d(v, lambda x: f"{x:+.2f}%"),
        }
        return _m.get(col)

    def _td_render(data, label, container=None):
        if data.empty: return
        _ct = container or st
        _d = data.sort_values(_td_sort_col, ascending=_td_asc, na_position="last")
        _d = _d[_td_disp_cols].reset_index(drop=True)
        _d = _d.rename(columns=_TD_RENAME)
        _ct.markdown(f"#### {label}")
        # ── Compute BB Pos% from numeric values (before string-formatting) ──
        # positive  = % above upper band  (red = extended / potential breakdown)
        # negative  = % below lower band  (green = oversold / potential snap-back)
        # zero/NaN  = inside bands        (no highlight)
        if all(c in _d.columns for c in ("Last", "UpperBB", "LowerBB")):
            def _bb_pos_fn(row):
                l, u, lo = row["Last"], row["UpperBB"], row["LowerBB"]
                if _td_pd.isna(l) or _td_pd.isna(u) or _td_pd.isna(lo): return float("nan")
                lf, uf, lof = float(l), float(u), float(lo)
                if lf > uf:  return (lf - uf)  / uf  * 100
                if lf < lof: return (lf - lof) / lof * 100
                return 0.0
            def _bb_dollar_fn(row):
                l, u, lo = row["Last"], row["UpperBB"], row["LowerBB"]
                if _td_pd.isna(l) or _td_pd.isna(u) or _td_pd.isna(lo): return float("nan")
                lf, uf, lof = float(l), float(u), float(lo)
                if lf > uf:  return lf - uf    # positive = $ above upper BB
                if lf < lof: return lf - lof   # negative = $ below lower BB
                return 0.0
            _d["BB Pos%"] = _d.apply(_bb_pos_fn, axis=1)
            _d["BB Pos$"] = _d.apply(_bb_dollar_fn, axis=1)
        # Reorder columns: put BB Pos% and BB Pos$ right after % vs 10d
        _COL_ORDER = [
            "Symbol","Last","10d MA","UpperBB","LowerBB",
            "% vs 10d","BB Pos%","BB Pos$",
            "RSI14","RSI3","MTD%","YTD%",
        ]
        _d = _d[[c for c in _COL_ORDER if c in _d.columns]]
        # Keep numeric copy for gradient before converting to strings
        _d_num = _d.copy()
        # Pre-format to strings — NaN becomes "-" reliably
        _RAW_FMT = {
            "Last":    lambda x: f"{x:,.2f}", "10d MA":  lambda x: f"{x:,.2f}",
            "UpperBB": lambda x: f"{x:,.2f}", "LowerBB": lambda x: f"{x:,.2f}",
            "BB Pos%": lambda x: "—" if abs(x) < 0.01 else f"{x:+.2f}%",
            "BB Pos$": lambda x: "—" if abs(x) < 0.01 else f"{x:+.2f}",
            "% vs 10d": lambda x: f"{x:+.2f}%",
            "RSI14": lambda x: f"{x:.1f}", "RSI3": lambda x: f"{x:.1f}",
            "MTD%": lambda x: f"{x:+.2f}%", "YTD%": lambda x: f"{x:+.2f}%",
        }
        for _c, _f in _RAW_FMT.items():
            if _c in _d.columns:
                _d[_c] = _d[_c].apply(
                    lambda v, _f=_f: "-" if _td_pd.isna(v) else _f(float(v)))
        try:
            _sty = _d.style
            import matplotlib as _mpl
            def _grad(num_s, cmap_n, vmin, vmax):
                _cm = _mpl.colormaps.get_cmap(cmap_n)
                _nm = _mpl.colors.Normalize(vmin=vmin, vmax=vmax)
                def _fn(col):
                    return [
                        f"background-color:rgba({int(r*255)},{int(g*255)},{int(b*255)},0.85)"
                        if not _td_pd.isna(nv) else ""
                        for nv, (r, g, b, _a) in zip(
                            num_s, [_cm(_nm(min(max(float(v2), vmin), vmax)))
                                    for v2 in num_s.fillna(vmin)])
                    ]
                return _fn
            if "RSI14" in _d.columns and _d_num["RSI14"].notna().any():
                _sty = _sty.apply(_grad(_d_num["RSI14"], "RdYlGn_r", 25, 75),
                                  subset=["RSI14"])
            if "% vs 10d" in _d.columns and _d_num["% vs 10d"].notna().any():
                _sty = _sty.apply(_grad(_d_num["% vs 10d"], "RdYlGn", -10, 10),
                                  subset=["% vs 10d"])
            # BB Pos%: red for above upper band, green for below lower band
            if "BB Pos%" in _d.columns and _d_num["BB Pos%"].notna().any():
                def _bb_color(col):
                    out = []
                    for _v in _d_num["BB Pos%"]:
                        if _td_pd.isna(_v) or abs(float(_v)) < 0.01:
                            out.append("")
                        elif float(_v) > 0:  # above upper BB
                            _i = min(abs(float(_v)) / 5.0, 1.0) * 0.80
                            out.append(f"background-color:rgba(220,53,69,{_i:.2f})")
                        else:               # below lower BB
                            _i = min(abs(float(_v)) / 5.0, 1.0) * 0.80
                            out.append(f"background-color:rgba(40,167,69,{_i:.2f})")
                    return out
                _sty = _sty.apply(_bb_color, subset=["BB Pos%"])
            # BB Pos$: same color logic as BB Pos% (red=above upper, green=below lower)
            if "BB Pos$" in _d.columns and _d_num["BB Pos$"].notna().any():
                def _bb_dollar_color(col):
                    out = []
                    for _v in _d_num["BB Pos$"]:
                        if _td_pd.isna(_v) or abs(float(_v)) < 0.01:
                            out.append("")
                        elif float(_v) > 0:
                            # scale intensity by $ relative to last price; 5% = full intensity
                            _last_ref = _d_num["Last"].iloc[list(_d_num["BB Pos$"]).index(_v)] if "Last" in _d_num.columns else float("nan")
                            _ref = max(abs(float(_last_ref)), 1.0) if not _td_pd.isna(_last_ref) else 1.0
                            _i = min(abs(float(_v)) / (_ref * 0.05), 1.0) * 0.80
                            out.append(f"background-color:rgba(220,53,69,{_i:.2f})")
                        else:
                            _last_ref = _d_num["Last"].iloc[list(_d_num["BB Pos$"]).index(_v)] if "Last" in _d_num.columns else float("nan")
                            _ref = max(abs(float(_last_ref)), 1.0) if not _td_pd.isna(_last_ref) else 1.0
                            _i = min(abs(float(_v)) / (_ref * 0.05), 1.0) * 0.80
                            out.append(f"background-color:rgba(40,167,69,{_i:.2f})")
                    return out
                _sty = _sty.apply(_bb_dollar_color, subset=["BB Pos$"])
            _td_cc = {
                "Symbol":  st.column_config.TextColumn("Symbol",  width="medium"),
                "Last":    st.column_config.TextColumn("Last",    width="small"),
                "10d MA":  st.column_config.TextColumn("10d MA",  width="small"),
                "UpperBB": st.column_config.TextColumn("UpperBB", width="small"),
                "LowerBB": st.column_config.TextColumn("LowerBB", width="small"),
                "% vs 10d":st.column_config.TextColumn("% vs 10d",width="small"),
                "BB Pos%": st.column_config.TextColumn("BB Pos%", width="small"),
                "BB Pos$": st.column_config.TextColumn("BB Pos$", width="small"),
                "RSI14":   st.column_config.TextColumn("RSI14",   width="small"),
                "RSI3":    st.column_config.TextColumn("RSI3",    width="small"),
                "MTD%":    st.column_config.TextColumn("MTD%",    width="small"),
                "YTD%":    st.column_config.TextColumn("YTD%",    width="small"),
            }
            _ct.dataframe(_sty, hide_index=True, use_container_width=True,
                         column_config=_td_cc)
        except Exception:
            _ct.dataframe(_d, hide_index=True, use_container_width=True)

    if _td_view == "All":
        _td_left, _td_right = st.columns(2)
        _td_render(_td_df[_td_is_idx], "📊 Indices & Macro", container=_td_left)
        _td_render(_td_df[~_td_is_idx], "📈 Stocks", container=_td_right)
    elif _td_view == "Indices Only":
        _td_render(_td_df[_td_is_idx], "📊 Indices & Macro")
    elif _td_view == "Stocks Only":
        _td_render(_td_df[~_td_is_idx], "📈 Stocks")


# =============================================================
# ADR Parity tab
# =============================================================
if nav == 'ADR Parity':
    import pandas as _ap_pd
    st.subheader('🌎 ADR Parity — Premium / Discount to Home Country')
    _AP_XLS = os.path.join(os.path.expanduser("~"), "OneDrive - Merus Global",
                           "PARITY SHEET (9).xlsx")
    if not os.path.exists(_AP_XLS):
        st.error("Parity Sheet not found.")
        st.stop()

    @st.cache_data(ttl=60, show_spinner=False)
    def _ap_load(path):
        _PDT = "P/D Today"
        _PD1 = "P/D T-1"
        frames = []
        for _sheet, _region in [("ToUS","European"), ("Parity","Canadian")]:
            try:
                _s = _ap_pd.read_excel(path, sheet_name=_sheet, header=0)
                _s.columns = _s.iloc[0].tolist()
                _s = _s.iloc[1:].reset_index(drop=True)
                _t_col = _s.columns[0]
                _pd_c  = [c for c in _s.columns if str(c).strip() == "P/D T" or str(c).strip() == "P/D Today"]
                _pd1_c = [c for c in _s.columns if str(c).strip() == "P/D T-1"]
                if not _pd_c: continue
                _sub = _s[[_t_col] + (_pd1_c[:1] or []) + _pd_c[:1]].copy()
                _sub.columns = ["BBTicker"] + ([_PD1] if _pd1_c else []) + [_PDT]
                _sub["Region"] = _region
                frames.append(_sub)
            except Exception: pass
        if not frames: return _ap_pd.DataFrame()
        out = _ap_pd.concat(frames, ignore_index=True)
        out["Ticker"] = (out["BBTicker"].astype(str)
                         .str.replace(r"\s+(US|HK|LN|GY|FP|NA|SE|SM)\s+Equity.*","",regex=True)
                         .str.strip())
        out[_PDT] = _ap_pd.to_numeric(out[_PDT], errors="coerce")
        if _PD1 in out.columns:
            out[_PD1] = _ap_pd.to_numeric(out[_PD1], errors="coerce")
            out["P/D Chg"] = out[_PDT] - out[_PD1]
        out["Abs P/D"] = out[_PDT].abs()
        return out.dropna(subset=["Ticker"]).reset_index(drop=True)

    _ap_c1, _ap_c2 = st.columns([6,1])
    if _ap_c2.button("🔄 Reload", key="ap_reload"):
        st.cache_data.clear(); st.rerun()
    try:
        _ap_df = _ap_load(_AP_XLS)
    except Exception as _ape:
        st.error(f"Could not load Parity Sheet: {_ape}")
        st.stop()

    _ap_PDT = "P/D Today"
    _ap_bberg = (not _ap_df.empty) and (_ap_df[_ap_PDT].abs().sum() > 0.0001)
    if not _ap_bberg:
        st.warning("⚠️ Bloomberg not connected — P/D values are zero. Connect and Reload.")
    else:
        st.success("✅ Bloomberg connected")

    _ap_r1, _ap_r2, _ap_r3 = st.columns(3)
    _ap_reg  = _ap_r1.selectbox("Region", ["All","European","Canadian"], key="ap_reg")
    _ap_sort = _ap_r2.selectbox("Sort",
        ["Biggest Premium","Biggest Discount","Abs Move","Ticker"], key="ap_sort")
    _ap_top  = int(_ap_r3.number_input("Top N", 10, 500, 60, step=10, key="ap_top"))

    _ap_show = _ap_df[_ap_df["Region"] == _ap_reg].copy() if _ap_reg != "All" else _ap_df.copy()
    if _ap_sort == "Biggest Premium":  _ap_show = _ap_show.sort_values(_ap_PDT, ascending=False)
    elif _ap_sort == "Biggest Discount": _ap_show = _ap_show.sort_values(_ap_PDT, ascending=True)
    elif _ap_sort == "Abs Move":       _ap_show = _ap_show.sort_values("Abs P/D", ascending=False)
    else:                              _ap_show = _ap_show.sort_values("Ticker")
    _ap_show = _ap_show.head(_ap_top).reset_index(drop=True)

    if _ap_bberg and not _ap_df.empty:
        _ap_pdt_s = _ap_df[_ap_PDT]
        _ap_tkr_s = _ap_df["Ticker"]
        _ap_max_i = _ap_pdt_s.idxmax()
        _ap_min_i = _ap_pdt_s.idxmin()
        _apm1, _apm2, _apm3 = st.columns(3)
        _apm1.metric("Avg P/D", f"{_ap_pdt_s.mean()*100:+.3f}%")
        _apm2.metric("Largest Premium",
            f"{_ap_pdt_s.max()*100:+.3f}%  ({_ap_tkr_s[_ap_max_i]})")
        _apm3.metric("Largest Discount",
            f"{_ap_pdt_s.min()*100:+.3f}%  ({_ap_tkr_s[_ap_min_i]})")

    _ap_dcols = ["Ticker","Region",_ap_PDT]
    if "P/D T-1"  in _ap_show.columns: _ap_dcols.append("P/D T-1")
    if "P/D Chg"  in _ap_show.columns: _ap_dcols.append("P/D Chg")
    _ap_disp = _ap_show[[c for c in _ap_dcols if c in _ap_show.columns]]

    if not _ap_disp.empty:
        import matplotlib as _mpl_ap
        _ap_num = _ap_disp.copy()
        _AP_FMTS = {_ap_PDT: lambda x: f"{x:+.3%}"}
        if "P/D T-1" in _ap_disp.columns: _AP_FMTS["P/D T-1"] = lambda x: f"{x:+.3%}"
        if "P/D Chg" in _ap_disp.columns: _AP_FMTS["P/D Chg"] = lambda x: f"{x:+.3%}"
        _ap_str = _ap_disp.copy()
        for _ac, _af in _AP_FMTS.items():
            if _ac in _ap_str.columns:
                _ap_str[_ac] = _ap_str[_ac].apply(
                    lambda v, _af=_af: "-" if _ap_pd.isna(v) or isinstance(v, str) else _af(float(v)))
        try:
            _ap_sty = _ap_str.style
            def _ap_grad(num_s, cmap_n, vmin, vmax):
                _cm_ap = _mpl_ap.colormaps.get_cmap(cmap_n)
                _nm_ap = _mpl_ap.colors.Normalize(vmin=vmin, vmax=vmax)
                def _fn_ap(col):
                    return [
                        f"background-color:rgba({int(r*255)},{int(g*255)},{int(b*255)},0.85)"
                        if not _ap_pd.isna(nv) else ""
                        for nv, (r, g, b, _a) in zip(
                            num_s, [_cm_ap(_nm_ap(min(max(float(v2),vmin),vmax)))
                                    for v2 in num_s.fillna(0)])
                    ]
                return _fn_ap
            if _ap_num[_ap_PDT].notna().any():
                _ap_sty = _ap_sty.apply(
                    _ap_grad(_ap_num[_ap_PDT], "RdYlGn_r", -0.01, 0.01),
                    subset=[_ap_PDT])
            _ap_cc = {
                "Ticker":  st.column_config.TextColumn("Ticker",  width="small"),
                "Region":  st.column_config.TextColumn("Region",  width="small"),
                _ap_PDT:   st.column_config.TextColumn("P/D Today", width="small"),
                "P/D T-1": st.column_config.TextColumn("P/D T-1", width="small"),
                "P/D Chg": st.column_config.TextColumn("P/D Chg", width="small"),
            }
            st.dataframe(_ap_sty, hide_index=True, use_container_width=False,
                         column_config={k:v for k,v in _ap_cc.items()
                                        if k in _ap_str.columns})
        except Exception:
            st.dataframe(_ap_disp, hide_index=True, use_container_width=False)
    else:
        st.info("No matching tickers.")


# =============================================================
# Macro Driver Map tab
# =============================================================
if nav == 'Macro Drivers':
    import pandas as _md_pd
    st.subheader('\U0001f4e1 Macro Driver Map')
    st.caption(
        'Live macro indicators with their correlated equity baskets. '
        'Click any driver row to see how the basket is tracking today.'
    )

    # ── Static correlation map ─────────────────────────────────────────────────
    # (Display Name, Bloomberg Ticker, Category, [Stock Basket], Basket Label)
    _MD_MAP = [
        # ── Commodities ───────────────────────────────────────────────────────
        ("Gold",          "GCA Comdty",     "Commodities",
         ["GDX","NEM","AEM","RGLD","WPM","PAAS","NVA"],
         "Gold Miners & Royalties"),
        ("Silver",        "SIA Comdty",     "Commodities",
         ["PAAS","AG","MAG"],
         "Silver Miners"),
        ("Copper",        "HGA Comdty",     "Commodities",
         ["FCX","SCCO","CLF","TECK","CRML"],
         "Copper Producers"),
        ("WTI Crude",     "CLA Comdty",     "Commodities",
         ["XOM","CVX","COP","OXY","DVN","EOG","APA","XLE","XOP"],
         "US Oil Producers & ETFs"),
        ("Brent Crude",   "COA Comdty",     "Commodities",
         ["BP","SHEL","TTE","EQNR","XOM","CVX","COP","OXY"],
         "Euro Oil Majors + US Peers"),
        ("Crack Spread",  "CRK321M1 Index", "Commodities",
         ["VLO","MPC","PSX"],
         "Refining Stocks (3:2:1 Crack)"),
        ("Heating Oil",   "HOA Comdty",     "Commodities",
         ["VLO","MPC","PSX","XOM"],
         "Refiners / Distillate Proxies"),
        ("RBOB Gasoline", "XBA Comdty",     "Commodities",
         ["VLO","MPC","PSX","CVI"],
         "Gasoline / Refinery Proxies"),
        ("Nat Gas",       "NGA Comdty",     "Commodities",
         ["CTRA","RRC","EQT","AR","LNG"],
         "Gas Producers"),
        # ── Metals / Mining ───────────────────────────────────────────────────
        ("Iron Ore",      "IOEA Comdty",    "Metals / Mining",
         ["CLF","MT","VALE","X"],
         "Steel & Iron Ore Producers"),
        ("DRAM Spot",     "ISPPDR37 Index", "Metals / Mining",
         ["MU","WDC","SNDK","STX","AMAT","LRCX"],
         "Memory & Semi-Cap Correlated"),
        # ── FX / Rates ────────────────────────────────────────────────────────
        ("DXY Dollar",    "DXY Index",      "FX / Rates",
         ["GLD","TLT","EEM","EFA","FXI"],
         "Dollar-Inverse Proxies"),
        ("JPY / USD",     "JYA Curncy",     "FX / Rates",
         ["TM","SONY","NKY","FXY"],
         "JPY-Sensitive Equities"),
        ("10yr Yield",    "USGG10YR Index", "FX / Rates",
         ["GS","JPM","BAC","C","WFC","MS","TBT","KRE"],
         "Rate-Sensitive Banks"),
        ("2yr Yield",     "USGG2YR Index",  "FX / Rates",
         ["GS","MS","JPM","SOFI"],
         "Short-Rate Sensitive"),
        # ── Volatility ────────────────────────────────────────────────────────
        ("VIX",           "VIX Index",      "Volatility",
         ["SPHB","SPLV","GLD","TLT","XLU"],
         "Vol Hedges & Defensives"),
        # ── Crypto ────────────────────────────────────────────────────────────
        ("Bitcoin",       "XBTUSD Curncy",  "Crypto",
         ["COIN","MSTR","RIOT","MARA","IBIT","BITO"],
         "Crypto Proxies"),
        ("Ethereum",      "XETUSD Curncy",  "Crypto",
         ["COIN","ETHA","BITO","HOOD"],
         "ETH Proxies"),
        # ── Indices ───────────────────────────────────────────────────────────
        ("SPX",           "SPX Index",      "Indices",
         ["QQQ","IWM","SPY","XLK","XLF","XLE"],
         "Index ETFs"),
        ("NDX",           "NDX Index",      "Indices",
         ["AAPL","MSFT","NVDA","AMZN","META","GOOGL","TSLA"],
         "NDX Mega-Cap Top 7"),
        ("Russell 2000",  "RTY Index",      "Indices",
         ["IWM","TNA","VTWO"],
         "Small-Cap ETFs"),
        ("SOX Semi",      "SOX Index",      "Indices",
         ["NVDA","AMD","AVGO","AMAT","LRCX","MRVL","MU","INTC"],
         "Semiconductor Basket"),
    ]

    # ── Reload button ──────────────────────────────────────────────────────────
    _md_hdr, _md_rld = st.columns([7, 1])
    if _md_rld.button("\U0001f504 Reload", key="md_reload"):
        st.cache_data.clear(); st.rerun()

    # ── Fetch all driver + basket prices in ONE Bloomberg call ──────────────
    _MD_SECS = tuple(r[1] for r in _MD_MAP)
    _MD_FLDS = ("PX_LAST", "CHG_NET_1D", "CHG_PCT_1D",
                "YTD_RETURN", "CHG_PCT_YTD",
                "RETURN_WTD", "CHG_PCT_WTD")
    # Collect all unique basket stock tickers
    _md_all_basket_syms = sorted(set(s for r in _MD_MAP for s in r[3]))
    _md_all_basket_secs = tuple(f"{s} US Equity" for s in _md_all_basket_syms)
    _md_all_secs = _MD_SECS + _md_all_basket_secs
    _md_raw_all  = _bbg_bdp(_md_all_secs, _MD_FLDS)
    _md_raw  = {k: v for k, v in _md_raw_all.items() if k in set(_MD_SECS)}
    _md_bk_raw = _md_raw_all  # basket stocks also in here
    _md_live = bool(_md_raw)
    if not _md_live:
        st.warning(
            "\u26a0\ufe0f Bloomberg not connected \u2014 driver prices unavailable. "
            "Connect and click Reload."
        )
    else:
        st.success("\u2705 Bloomberg connected (live data via API)")

    # ── Build driver DataFrame ─────────────────────────────────────────────────
    _md_rows = []
    for _mn, _mbbg, _mcat, _mbasket, _mdesc in _MD_MAP:
        _d = _md_raw.get(_mbbg, {})
        _ytd_drv = _d.get("YTD_RETURN", float("nan"))
        if _md_pd.isna(_ytd_drv): _ytd_drv = _d.get("CHG_PCT_YTD", float("nan"))
        _wtd_drv = _d.get("RETURN_WTD", float("nan"))
        if _md_pd.isna(_wtd_drv): _wtd_drv = _d.get("CHG_PCT_WTD", float("nan"))
        _md_rows.append({
            "Driver":   _mn,
            "Category": _mcat,
            "Last":     _d.get("PX_LAST",    float("nan")),
            "Change":   _d.get("CHG_NET_1D", float("nan")),
            "Chg%":     _d.get("CHG_PCT_1D", float("nan")),
            "WTD%":     _wtd_drv,
            "YTD%":     _ytd_drv,
        })
    _md_df  = _md_pd.DataFrame(_md_rows)
    _md_num = _md_df.copy()   # keep numeric copy for gradient

    # ── Pre-format to strings ─────────────────────────────────────────────────
    _MD_FMTMAP = {
        "Last":   lambda x: f"{x:,.3f}",
        "Change": lambda x: f"{x:+.3f}",
        "Chg%":   lambda x: f"{x:+.3f}%",
        "WTD%":   lambda x: f"{x:+.2f}%",
        "YTD%":   lambda x: f"{x:+.2f}%",
    }
    _md_str = _md_df.copy()
    for _dc, _df_fn in _MD_FMTMAP.items():
        _md_str[_dc] = _md_str[_dc].apply(
            lambda v, _fn=_df_fn: "-" if _md_pd.isna(v) else _fn(float(v)))

    import matplotlib as _mpl_md
    def _md_grad(num_s, cmap_n, vmin, vmax):
        _cm = _mpl_md.colormaps.get_cmap(cmap_n)
        _nm = _mpl_md.colors.Normalize(vmin=vmin, vmax=vmax)
        def _fn(col):
            return [
                f"background-color:rgba({int(r*255)},{int(g*255)},{int(b*255)},0.80)"
                if not _md_pd.isna(nv) else ""
                for nv, (r, g, b, _a) in zip(
                    num_s, [_cm(_nm(min(max(float(v2), vmin), vmax)))
                            for v2 in num_s.fillna(0)])
            ]
        return _fn

    _md_sty = _md_str.style
    if _md_num["Chg%"].notna().any():
        _md_sty = _md_sty.apply(
            _md_grad(_md_num["Chg%"], "RdYlGn", -3, 3), subset=["Chg%"])

    _md_cc = {
        "Driver":   st.column_config.TextColumn("Driver",   width="medium"),
        "Category": st.column_config.TextColumn("Category", width="medium"),
        "Last":     st.column_config.TextColumn("Last",     width="small"),
        "Change":   st.column_config.TextColumn("Change",   width="small"),
        "Chg%":     st.column_config.TextColumn("Chg%",     width="small"),
        "WTD%":     st.column_config.TextColumn("WTD%",     width="small"),
        "YTD%":     st.column_config.TextColumn("YTD%",     width="small"),
    }

    _md_col_left, _md_col_right = st.columns([11, 9])

    # ── Driver selector (dropdown) ────────────────────────────────────────────
    with _md_col_left:
        _md_event = st.dataframe(
            _md_sty, hide_index=True, use_container_width=True,
            column_config=_md_cc,
            on_select="rerun", selection_mode="single-row",
            key="md_driver_table",
        )

    if hasattr(_md_event, "selection") and _md_event.selection.rows:
        st.session_state["md_sel_row"] = _md_event.selection.rows[0]
    _md_sel_row = st.session_state.get("md_sel_row", 0)
    _md_sel_row = min(_md_sel_row, len(_MD_MAP) - 1)

    _md_sel      = _MD_MAP[_md_sel_row]
    _md_drv_name = _md_sel[0]
    _md_drv_bbg  = _md_sel[1]
    _md_basket   = _md_sel[3]
    _md_bkt_lbl  = _md_sel[4]

    _drv_last = _md_num["Last"].iloc[_md_sel_row]
    _drv_chg  = _md_num["Chg%"].iloc[_md_sel_row]
    import math as _md_math
    _drv_chg_v = float(_drv_chg) if not _md_math.isnan(float(_drv_chg)) else float("nan")

    # ── Basket header + table (right column) ────────────────────────────────
    _last_str = (f"{float(_drv_last):,.3f}" if not _md_math.isnan(float(_drv_last)) else "\u2014")
    _chg_str  = (f"{_drv_chg_v:+.3f}%"     if not _md_math.isnan(_drv_chg_v)        else "\u2014")
    _chg_col  = ("#28a745" if (not _md_math.isnan(_drv_chg_v) and _drv_chg_v >= 0)
                else "#dc3545")
    _md_col_right.markdown(
        '<div style="margin:0 0 8px;padding:5px 12px;background:#1a2130;'
        'border-left:3px solid #58a6ff;border-radius:3px;'
        'font-size:13px;font-family:Consolas,monospace;line-height:1.6;">'
        f'<span style="color:#58a6ff;font-weight:700">{_md_drv_name}</span>'
        f'&nbsp;&nbsp;<span style="color:#6e7681">{_md_bkt_lbl}</span>'
        f'&emsp;Last <span style="color:#e6edf3">{_last_str}</span>'
        f'&emsp;Today <span style="color:{_chg_col};font-weight:600">{_chg_str}</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # ── Build basket rows from pre-fetched data ──────────────────────────────
    _bk_rows = []
    for _bsym in _md_basket:
        _bsec = f"{_bsym} US Equity"
        _d = _md_bk_raw.get(_bsec, {})
        _ytd_bk = _d.get("YTD_RETURN", float("nan"))
        if _md_pd.isna(_ytd_bk): _ytd_bk = _d.get("CHG_PCT_YTD", float("nan"))
        _bk_rows.append({
            "Symbol": _bsym,
            "Last":   _d.get("PX_LAST",    float("nan")),
            "Change": _d.get("CHG_NET_1D", float("nan")),
            "Chg%":   _d.get("CHG_PCT_1D", float("nan")),
            "YTD%":   _ytd_bk,
        })

    # Add "vs Driver" column (stock Chg% minus driver Chg%) ─────────────────────
    for _r in _bk_rows:
        _sc = _r["Chg%"]
        if not (_md_pd.isna(_sc) or _md_math.isnan(_drv_chg_v)):
            _r["vs Driver"] = float(_sc) - _drv_chg_v
        else:
            _r["vs Driver"] = float("nan")

    _bk_df  = _md_pd.DataFrame(_bk_rows)

    if not _bk_df.empty:
        _bk_num = _bk_df.copy()

        # Pre-format
        _BK_FMT = {
            "Last":      lambda x: f"{x:,.2f}",
            "Change":    lambda x: f"{x:+.2f}",
            "Chg%":      lambda x: f"{x:+.3f}%",
            "vs Driver": lambda x: f"{x:+.3f}%",
            "YTD%":      lambda x: f"{x:+.2f}%",
        }
        _bk_str = _bk_df.copy()
        for _bc, _bf in _BK_FMT.items():
            if _bc in _bk_str.columns:
                _bk_str[_bc] = _bk_str[_bc].apply(
                    lambda v, _f=_bf: "-" if _md_pd.isna(v) else _f(float(v)))

        _bk_sty = _bk_str.style
        if "Chg%" in _bk_num and _bk_num["Chg%"].notna().any():
            _bk_sty = _bk_sty.apply(
                _md_grad(_bk_num["Chg%"], "RdYlGn", -3, 3), subset=["Chg%"])
        if "vs Driver" in _bk_num and _bk_num["vs Driver"].notna().any():
            _bk_sty = _bk_sty.apply(
                _md_grad(_bk_num["vs Driver"], "RdYlGn", -2, 2), subset=["vs Driver"])

        _bk_cc = {
            "Symbol":    st.column_config.TextColumn("Symbol",     width="small"),
            "Last":      st.column_config.TextColumn("Last",       width="small"),
            "Change":    st.column_config.TextColumn("Change",     width="small"),
            "Chg%":      st.column_config.TextColumn("Chg%",       width="small"),
            "vs Driver": st.column_config.TextColumn("vs Driver",  width="small"),
            "YTD%":      st.column_config.TextColumn("YTD%",       width="small"),
        }
        _md_col_right.caption(f"vs Driver = Chg% \u2212 {_md_drv_name} today \u00b7 \u25b2 outperforming \u00b7 \u25bc lagging")
        _md_col_right.dataframe(
            _bk_sty, hide_index=True, use_container_width=True,
            column_config={k: v for k, v in _bk_cc.items() if k in _bk_str.columns},
        )

        # ── Basket summary metrics ─────────────────────────────────────────────
        _bk_chg_vals = _bk_num["Chg%"].dropna()
        if not _bk_chg_vals.empty:
            _bk_avg   = float(_bk_chg_vals.mean())
            _bk_best  = int(_bk_chg_vals.idxmax())
            _bk_worst = int(_bk_chg_vals.idxmin())
            _sm1, _sm2, _sm3 = _md_col_right.columns(3)
            _rel_avg = (_bk_avg - _drv_chg_v) if not _md_math.isnan(_drv_chg_v) else None
            _sm1.metric(
                "Basket Avg",
                f"{_bk_avg:+.3f}%",
                delta=f"{_rel_avg:+.3f}% vs driver" if _rel_avg is not None else None,
            )
            _sm2.metric(
                "Best  \U0001f7e2",
                f"{_bk_str['Symbol'].iloc[_bk_best]}"
                f"  {float(_bk_num['Chg%'].iloc[_bk_best]):+.3f}%",
            )
            _sm3.metric(
                "Laggard  \U0001f534",
                f"{_bk_str['Symbol'].iloc[_bk_worst]}"
                f"  {float(_bk_num['Chg%'].iloc[_bk_worst]):+.3f}%",
            )
    else:
        _md_col_right.info("No basket data \u2014 connect Bloomberg and click Reload.")

# ═══════════════════════════════════════════════════════════════════════════════
# TAB — Polymarket
# ═══════════════════════════════════════════════════════════════════════════════
if nav == 'Polymarket':
    import json as _pm_json
    import pandas as _pm_pd
    from datetime import datetime as _pm_dt

    st.subheader("\U0001f52e Polymarket \u2014 Prediction Markets")

    # ── Controls bar ─────────────────────────────────────────────────────────
    _pm_ctrl1, _pm_ctrl2, _pm_ctrl3, _pm_ctrl4 = st.columns([2, 1, 1, 1])
    _pm_search_q = _pm_ctrl1.text_input(
        "Search contracts", key="pm_q",
        placeholder="e.g. fed, recession, bitcoin, trump, oil, tariff...")
    _pm_cat_opts = ["All"] + list(_PM_CATEGORIES.keys())
    _pm_cat_sel = _pm_ctrl2.selectbox("Category", _pm_cat_opts, key="pm_cat")
    _pm_sort_opts = ["Vol 24h", "Vol 1W", "Price", "1W Chg", "1M Chg", "Liquidity"]
    _pm_sort_sel = _pm_ctrl3.selectbox("Sort by", _pm_sort_opts, key="pm_sort")
    _pm_n_show = _pm_ctrl4.selectbox("Show", [50, 100, 200, 500], index=1, key="pm_nshow")

    if _pm_ctrl4.button("\U0001f504 Refresh", key="pm_refresh"):
        st.cache_data.clear()

    # ── Fetch contracts ──────────────────────────────────────────────────────
    # Pull up to 500 contracts in first batch; if user wants 500, do a second fetch
    _pm_raw = _pm_fetch_markets(limit=500, offset=0)
    if _pm_n_show > 500 and len(_pm_raw) == 500:
        _pm_raw += _pm_fetch_markets(limit=500, offset=500)

    _pm_all_rows = [_pm_market_to_row(m) for m in _pm_raw]
    _pm_df = _pm_pd.DataFrame(_pm_all_rows) if _pm_all_rows else _pm_pd.DataFrame()

    if not _pm_df.empty:
        # ── Client-side filters ──────────────────────────────────────────────
        if _pm_search_q:
            _sq = _pm_search_q.lower()
            _pm_df = _pm_df[_pm_df["Contract"].str.lower().str.contains(_sq, na=False)
                            | _pm_df["_slug"].str.lower().str.contains(_sq, na=False)]

        if _pm_cat_sel != "All":
            _pm_df = _pm_df[_pm_df["Category"] == _pm_cat_sel]

        # Sort
        _sort_asc = False
        if _pm_sort_sel == "Price":
            _sort_asc = False
        _pm_df = _pm_df.sort_values(_pm_sort_sel, ascending=_sort_asc, na_position="last")
        _pm_df = _pm_df.head(_pm_n_show)

        st.caption(f"{len(_pm_df)} contracts shown \u2022 {len(_pm_all_rows)} total fetched from Polymarket")

        # ── Section 1: Active Contracts Table ────────────────────────────────
        st.markdown("### Active Contracts")
        _pm_show_df = _pm_df[["Contract", "Price", "1W Chg", "1M Chg",
                               "Vol 24h", "Vol 1W", "Liquidity", "Spread", "Category"]].copy()

        # Styled dataframe
        def _pm_style_table(styler):
            styler.format({
                "Price": "{:.1%}",
                "1W Chg": "{:+.1%}",
                "1M Chg": "{:+.1%}",
                "Vol 24h": "${:,.0f}",
                "Vol 1W": "${:,.0f}",
                "Liquidity": "${:,.0f}",
                "Spread": "{:.3f}",
            })
            styler.background_gradient(subset=["Price"], cmap="RdYlGn", vmin=0, vmax=1)
            styler.background_gradient(subset=["1W Chg"], cmap="RdYlGn", vmin=-0.15, vmax=0.15)
            styler.background_gradient(subset=["1M Chg"], cmap="RdYlGn", vmin=-0.3, vmax=0.3)
            return styler

        st.dataframe(
            _pm_show_df.style.pipe(_pm_style_table),
            use_container_width=True, hide_index=True,
            height=min(35 * len(_pm_show_df) + 38, 800),
        )

        # CSV download
        st.download_button("\U0001f4e5 Download CSV", _pm_show_df.to_csv(index=False),
                           "polymarket_contracts.csv", "text/csv", key="pm_csv")

        # ── Category summary ─────────────────────────────────────────────────
        with st.expander("Category Breakdown", expanded=False):
            _cat_summary = _pm_pd.DataFrame(_pm_all_rows).groupby("Category").agg(
                Contracts=("Contract", "count"),
                Avg_Price=("Price", "mean"),
                Total_Vol_24h=("Vol 24h", "sum"),
                Total_Liquidity=("Liquidity", "sum"),
            ).sort_values("Total_Vol_24h", ascending=False)
            _cat_summary["Avg_Price"] = _cat_summary["Avg_Price"].map("{:.1%}".format)
            _cat_summary["Total_Vol_24h"] = _cat_summary["Total_Vol_24h"].map("${:,.0f}".format)
            _cat_summary["Total_Liquidity"] = _cat_summary["Total_Liquidity"].map("${:,.0f}".format)
            st.dataframe(_cat_summary, use_container_width=True)

        st.markdown("---")

        # ── Section 2: Watchlist ─────────────────────────────────────────────
        st.markdown("### \U0001f4cb Watchlist")
        _wl_data = _pm_load_watchlist()

        # Add to watchlist from current view
        _pm_add_c1, _pm_add_c2 = st.columns([4, 1])
        _pm_add_opts = _pm_df["Contract"].tolist()
        _pm_add_pick = _pm_add_c1.selectbox("Add contract to watchlist:",
                                              ["(select)"] + _pm_add_opts, key="pm_wl_add")
        if _pm_add_pick != "(select)" and _pm_add_c2.button("\u2795 Add", key="pm_wl_add_btn"):
            _match_row = _pm_df[_pm_df["Contract"] == _pm_add_pick].iloc[0]
            _wl = _pm_load_watchlist()
            if not any(w.get("slug") == _match_row["_slug"] for w in _wl):
                _wl.append({
                    "slug": _match_row["_slug"],
                    "title": _match_row["Contract"],
                    "market_id": _match_row["_id"],
                    "category": _match_row["Category"],
                    "added": _pm_dt.now().strftime("%Y-%m-%d %H:%M"),
                })
                _pm_save_watchlist(_wl)
                st.success(f"Added: {_pm_add_pick[:50]}")
                st.rerun()
            else:
                st.warning("Already in watchlist.")

        if _wl_data:
            # Enrich watchlist with live data from our fetched contracts
            _pm_slug_idx = {r["_slug"]: r for _, r in _pm_df.iterrows()} if not _pm_df.empty else {}
            _wl_rows = []
            for _w in _wl_data:
                _ws = _w.get("slug", "")
                if _ws in _pm_slug_idx:
                    _wr = _pm_slug_idx[_ws]
                    _wl_rows.append({
                        "Contract": _wr["Contract"][:70],
                        "Price": _wr["Price"],
                        "1W Chg": _wr["1W Chg"],
                        "Vol 24h": _wr["Vol 24h"],
                        "Category": _wr["Category"],
                        "Added": _w.get("added", ""),
                    })
                else:
                    # Not in current batch — try event slug lookup
                    _wev = _pm_fetch_event_by_slug(_ws)
                    if _wev:
                        _woc, _wpr, _ = _pm_extract_best_outcome(_wev)
                        _wl_rows.append({
                            "Contract": _w.get("title", "")[:70],
                            "Price": _wpr,
                            "1W Chg": 0.0,
                            "Vol 24h": _pm_safe_float(_wev.get("volume24hr")),
                            "Category": _w.get("category", ""),
                            "Added": _w.get("added", ""),
                        })
                    else:
                        _wl_rows.append({
                            "Contract": _w.get("title", "")[:70],
                            "Price": 0.0, "1W Chg": 0.0, "Vol 24h": 0.0,
                            "Category": _w.get("category", ""),
                            "Added": _w.get("added", ""),
                        })

            _wl_df = _pm_pd.DataFrame(_wl_rows)
            st.dataframe(
                _wl_df.style.format({
                    "Price": "{:.1%}", "1W Chg": "{:+.1%}", "Vol 24h": "${:,.0f}",
                }),
                use_container_width=True, hide_index=True,
            )

            # Remove from watchlist
            _wl_c1, _wl_c2 = st.columns([4, 1])
            _wl_titles = [w.get("title", "") for w in _wl_data]
            _wl_rm = _wl_c1.selectbox("Remove:", ["(select)"] + _wl_titles, key="pm_wl_rm")
            if _wl_rm != "(select)" and _wl_c2.button("\u274c Remove", key="pm_wl_rm_btn"):
                _wl_data = [w for w in _wl_data if w.get("title") != _wl_rm]
                _pm_save_watchlist(_wl_data)
                st.rerun()

            st.download_button("\U0001f4e5 Watchlist CSV", _wl_df.to_csv(index=False),
                               "polymarket_watchlist.csv", "text/csv", key="pm_wl_csv")
        else:
            st.info("Watchlist empty \u2014 select a contract above and click Add.")

        st.markdown("---")

        # ── Section 3: Drill-Down ────────────────────────────────────────────
        st.markdown("### Market Drill-Down")
        _pm_drill_opts = _pm_df[["Contract", "_slug"]].drop_duplicates("_slug")
        if not _pm_drill_opts.empty:
            _pm_drill_slug = st.selectbox(
                "Select contract",
                _pm_drill_opts["_slug"].tolist(),
                format_func=lambda s: _pm_drill_opts[_pm_drill_opts["_slug"] == s]["Contract"].iloc[0][:60]
                    if not _pm_drill_opts[_pm_drill_opts["_slug"] == s].empty else s,
                key="pm_drill",
            )
            if _pm_drill_slug:
                _pm_drill_ev = _pm_fetch_event_by_slug(_pm_drill_slug)
                if _pm_drill_ev:
                    st.markdown(f"**{_pm_drill_ev.get('title', '')}**")
                    _pm_desc = _pm_drill_ev.get("description") or ""
                    if _pm_desc:
                        with st.expander("Description", expanded=False):
                            st.markdown(_pm_desc[:1000])

                    # All outcomes as metrics
                    _pm_drill_mkts = _pm_drill_ev.get("markets") or []
                    for _dm in _pm_drill_mkts:
                        _dm_q = _dm.get("question") or _dm.get("groupItemTitle") or ""
                        if _dm_q and len(_pm_drill_mkts) > 1:
                            st.caption(_dm_q[:80])
                        _dm_outcomes = _dm.get("outcomes") or "[]"
                        _dm_prices = _dm.get("outcomePrices") or "[]"
                        _dm_chg1w = _pm_safe_float(_dm.get("oneWeekPriceChange"))
                        _dm_chg1m = _pm_safe_float(_dm.get("oneMonthPriceChange"))
                        _dm_v24 = _pm_safe_float(_dm.get("volume24hr"))
                        _dm_liq = _pm_safe_float(_dm.get("liquidityNum") or _dm.get("liquidity"))
                        if isinstance(_dm_outcomes, str):
                            try: _dm_outcomes = _pm_json.loads(_dm_outcomes)
                            except Exception: _dm_outcomes = []
                        if isinstance(_dm_prices, str):
                            try: _dm_prices = _pm_json.loads(_dm_prices)
                            except Exception: _dm_prices = []
                        if _dm_outcomes:
                            _n_oc = min(len(_dm_outcomes), 8)
                            _dm_cols = st.columns(_n_oc + 2)
                            for _oi in range(_n_oc):
                                _olbl = _dm_outcomes[_oi] if _oi < len(_dm_outcomes) else ""
                                _opr = float(_dm_prices[_oi]) if _oi < len(_dm_prices) else 0.0
                                _dm_cols[_oi].metric(_olbl, f"{_opr:.0%}")
                            _dm_cols[-2].metric("Vol 24h", f"${_dm_v24:,.0f}")
                            _dm_cols[-1].metric("Liquidity", f"${_dm_liq:,.0f}")

                    # Summary stats
                    _drill_sc1, _drill_sc2, _drill_sc3, _drill_sc4 = st.columns(4)
                    _ev_vol = _pm_safe_float(_pm_drill_ev.get("volume"))
                    _ev_liq = _pm_safe_float(_pm_drill_ev.get("liquidity"))
                    _ev_oi = _pm_safe_float(_pm_drill_ev.get("openInterest"))
                    _drill_sc1.metric("Total Volume", f"${_ev_vol:,.0f}")
                    _drill_sc2.metric("Liquidity", f"${_ev_liq:,.0f}")
                    _drill_sc3.metric("Open Interest", f"${_ev_oi:,.0f}" if _ev_oi else "\u2014")
                    _drill_sc4.metric("Markets", str(len(_pm_drill_mkts)))

                    # Watchlist toggle + link
                    _in_wl = any(w.get("slug") == _pm_drill_slug for w in _wl_data)
                    _dr_c1, _dr_c2 = st.columns(2)
                    if _in_wl:
                        if _dr_c1.button("\u274c Remove from Watchlist", key="pm_dr_rm"):
                            _wl_data = [w for w in _wl_data if w.get("slug") != _pm_drill_slug]
                            _pm_save_watchlist(_wl_data)
                            st.rerun()
                    else:
                        if _dr_c1.button("\u2795 Add to Watchlist", key="pm_dr_add"):
                            _oc, _pr, _tid = _pm_extract_best_outcome(_pm_drill_ev)
                            _wl_data.append({
                                "slug": _pm_drill_slug,
                                "title": _pm_drill_ev.get("title", ""),
                                "market_id": "",
                                "category": _pm_categorize(
                                    _pm_drill_ev.get("title", ""), _pm_drill_slug),
                                "added": _pm_dt.now().strftime("%Y-%m-%d %H:%M"),
                            })
                            _pm_save_watchlist(_wl_data)
                            st.rerun()
                    _dr_c2.markdown(
                        f"[\U0001f517 View on Polymarket](https://polymarket.com/event/{_pm_drill_slug})")
                else:
                    st.warning("Could not load event details for this contract.")

    else:
        st.warning("Could not fetch markets from Polymarket. API may be unreachable.")
