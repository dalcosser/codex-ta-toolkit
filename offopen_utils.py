"""Shared helpers for computing off-open minute returns."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

TZ_NY = 'America/New_York'


def find_minute_file(minute_dir: Path, ticker: str) -> Path | None:
    """Locate the best parquet file for a ticker inside a minute directory."""
    if not minute_dir:
        return None
    t = (ticker or "").strip().upper().replace('.', '_')
    if not t:
        return None
    direct = minute_dir / f"{t}.parquet"
    if direct.exists():
        return direct
    sub = minute_dir / t
    if sub.is_dir():
        candidates = sorted(sub.glob('*.parquet'), key=lambda fp: fp.stat().st_size, reverse=True)
        if candidates:
            return candidates[0]
    for fp in minute_dir.glob(f"{t}*.parquet"):
        return fp
    return None


def _ensure_numeric_series(series: pd.Series) -> pd.Series:
    try:
        return pd.to_numeric(series, errors='coerce')
    except Exception:
        return pd.Series(dtype='float64')


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure standard OHLCV column names exist (Close/Open/High/Low/Volume)."""
    out = df.copy()
    lc = {str(c).lower(): c for c in out.columns}

    def ensure(target: str, *cands: str) -> None:
        if target in out.columns:
            return
        for name in cands:
            key = lc.get(name.lower())
            if key in out.columns:
                try:
                    out[target] = pd.to_numeric(out[key], errors='coerce')
                except Exception:
                    out[target] = out[key]
                return
        out[target] = np.nan

    ensure('Close', 'close', 'c')
    ensure('Open', 'open', 'o')
    ensure('High', 'high', 'h')
    ensure('Low', 'low', 'l')
    ensure('Volume', 'volume', 'v')
    return out


def ensure_ny_index(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with a tz-aware America/New_York DatetimeIndex."""
    if isinstance(df.index, pd.DatetimeIndex):
        idx = df.index
        try:
            if idx.tz is None:
                idx = idx.tz_localize('UTC')
            idx = idx.tz_convert(TZ_NY)
        except Exception:
            idx = pd.to_datetime(idx, errors='coerce', utc=True).tz_convert(TZ_NY)
        out = df.copy(); out.index = idx
        return out.sort_index()

    cols = {str(c).lower(): c for c in df.columns}
    col = None
    for name in ('timestamp','t','start','window_start','time','date'):
        key = cols.get(name.lower())
        if key in df.columns:
            col = key
            break
    if col is None:
        dcol = cols.get('date'); tcol = cols.get('time')
        if dcol and tcol and dcol in df.columns and tcol in df.columns:
            ds = pd.to_datetime(df[dcol], errors='coerce')
            tt = df[tcol].astype(str).str.extract(r'^(\d{1,2}:\d{2})')[0]
            idx = pd.to_datetime(ds.dt.strftime('%Y-%m-%d') + ' ' + tt, errors='coerce')
        else:
            raise RuntimeError(f"No datetime column/index found. Columns={list(df.columns)}")
    else:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            sv = pd.to_numeric(s, errors='coerce')
            mx = sv.dropna().abs().max()
            unit = 'ns' if (mx is not None and mx >= 1e18) else ('us' if (mx is not None and mx >= 1e15) else ('ms' if (mx is not None and mx >= 1e12) else 's'))
            idx = pd.to_datetime(sv, errors='coerce', utc=True, unit=unit)
        else:
            idx = pd.to_datetime(s.astype(str), errors='coerce', utc=True)

    if isinstance(idx, pd.Series):
        idx = pd.DatetimeIndex(idx)
    if getattr(idx, 'tz', None) is None:
        idx = idx.tz_localize('UTC')
    idx = idx.tz_convert(TZ_NY)
    out = df.copy(); out.index = idx
    return out.sort_index()


def load_minute_dataframe(minute_dir: Path, ticker: str) -> pd.DataFrame | None:
    """Load and normalize the full minute parquet for a ticker."""
    fp = find_minute_file(minute_dir, ticker)
    if not fp:
        return None
    df = pd.read_parquet(fp)
    df = normalize_ohlcv(df)
    return ensure_ny_index(df)


def _normalize_day(day: pd.Timestamp | str | pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(day)
    if ts.tzinfo is None:
        return ts.tz_localize(TZ_NY)
    return ts.tz_convert(TZ_NY)


def _to_naive_ts(ts: pd.Timestamp | None) -> pd.Timestamp:
    if ts is None or ts is pd.NaT:
        return pd.NaT
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        return ts
    try:
        ts = ts.tz_convert(TZ_NY)
    except Exception:
        pass
    try:
        return ts.tz_localize(None)
    except Exception:
        return ts


def compute_offopen_for_dates(
    intr_all: pd.DataFrame,
    dates: Sequence[pd.Timestamp],
    marks: Sequence[int],
    tol_min: int,
    allow_nearest: bool = True,
) -> pd.DataFrame:
    rows = []
    date_list = list(dates)
    if not date_list:
        return pd.DataFrame()
    for day in date_list:
        rows.append(_compute_offopen_for_day(intr_all, day, marks, tol_min, allow_nearest))
    df = pd.DataFrame(rows)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date').sort_index()
        df = df.sort_index(ascending=False)
    return df


def _compute_offopen_for_day(
    intr_all: pd.DataFrame,
    day: pd.Timestamp | str,
    marks: Sequence[int],
    tol_min: int,
    allow_nearest: bool,
) -> dict:
    marks = [int(m) for m in (marks or [])]
    tol_min = int(max(tol_min, 0))
    day_et = _normalize_day(day)
    start = day_et.normalize()
    end = start + pd.Timedelta(days=1)
    seg = intr_all[(intr_all.index >= start) & (intr_all.index < end)]
    mark_cols = {f"{int(m)} Min Return": np.nan for m in marks}
    row: dict[str, object] = {
        'Date': day_et.tz_localize(None),
        'Open Timestamp (ET)': pd.NaT,
        'Open Price': np.nan,
        'Prev Close Timestamp (ET)': pd.NaT,
        'Prev Close Price': np.nan,
        '09:24 Timestamp (ET)': pd.NaT,
        '9:24 to Open': np.nan,
        **mark_cols,
        'Reason': '',
    }
    reasons: list[str] = []

    def add_reason(msg: str) -> None:
        if msg and msg not in reasons:
            reasons.append(msg)

    if seg.empty:
        add_reason('no_data_for_date')
        row['Reason'] = '; '.join(reasons)
        return row

    t_open = day_et.replace(hour=9, minute=30, second=0, microsecond=0)
    win_open = seg[(seg.index >= t_open) & (seg.index <= t_open + pd.Timedelta(minutes=tol_min))]
    open_dt = None
    if not win_open.empty:
        open_dt = win_open.index[0]
        p_open = float(win_open['Close'].iloc[0]) if 'Close' in win_open.columns else float('nan')
    else:
        pos = seg.index.get_indexer([t_open], method='nearest')
        if not (pos.size and pos[0] != -1):
            add_reason('no_open_bar')
            row['Reason'] = '; '.join(reasons)
            return row
        open_dt = seg.index[int(pos[0])]
        delta = abs((open_dt - t_open).total_seconds())
        if delta > 60*tol_min and not allow_nearest:
            add_reason('open_beyond_tolerance')
            row['Reason'] = '; '.join(reasons)
            return row
        if delta > 60*tol_min:
            add_reason(f'used_nearest_open(+{int(delta)}s)')
        p_open = float(seg['Close'].iloc[int(pos[0])]) if 'Close' in seg.columns else float('nan')

    row['Open Timestamp (ET)'] = _to_naive_ts(open_dt)
    row['Open Price'] = p_open

    t_pre = day_et.replace(hour=9, minute=24, second=0, microsecond=0)
    pre_bar = seg[seg.index == t_pre]
    if not pre_bar.empty:
        pre_dt = pre_bar.index[-1]
        p_pre = float(pre_bar['Close'].iloc[-1]) if 'Close' in pre_bar.columns else float('nan')
        row['09:24 Timestamp (ET)'] = _to_naive_ts(pre_dt)
        row['9:24 to Open'] = (p_open / p_pre - 1.0) * 100.0 if (p_pre == p_pre and p_pre) else np.nan
    else:
        add_reason('no_924_bar')

    for m in marks:
        col = f"{int(m)} Min Return"
        t_m = t_open + pd.Timedelta(minutes=int(m))
        win_m = seg[(seg.index >= t_m) & (seg.index <= t_m + pd.Timedelta(minutes=tol_min))]
        if not win_m.empty:
            p_m = float(win_m['Close'].iloc[0]) if 'Close' in win_m.columns else float('nan')
            row[col] = (p_m / p_open - 1.0) * 100.0 if p_open else np.nan
            continue
        im = seg.index.get_indexer([t_m], method='nearest')
        if not (im.size and im[0] != -1):
            add_reason(f'no_{int(m)}m_bar')
            row[col] = np.nan
            continue
        dtm = seg.index[int(im[0])]
        delta = abs((dtm - t_m).total_seconds())
        if delta > 60*tol_min and not allow_nearest:
            add_reason(f'{int(m)}m_beyond_tol')
            row[col] = np.nan
            continue
        if delta > 60*tol_min:
            add_reason(f'used_nearest_{int(m)}m(+{int(delta)}s)')
        p_m = float(seg['Close'].iloc[int(im[0])]) if 'Close' in seg.columns else float('nan')
        row[col] = (p_m / p_open - 1.0) * 100.0 if p_open else np.nan

    start_prev_window = day_et.normalize() - pd.Timedelta(days=1)
    prev_cutoff = day_et.normalize()
    prev_seg = intr_all[(intr_all.index >= start_prev_window) & (intr_all.index < prev_cutoff)]
    if prev_seg.empty:
        # Weekend/holiday: take the last bar strictly before the day start (e.g., Friday for Monday)
        prev_idx = intr_all.index[intr_all.index < prev_cutoff]
        if len(prev_idx):
            prev_ts = prev_idx.max()
            prev_row = intr_all.loc[prev_ts]
            prev_price = float(prev_row['Close']) if 'Close' in prev_row else float('nan')
        else:
            prev_ts = pd.NaT
            prev_price = np.nan
    else:
        prev_ts = prev_seg.index[-1]
        prev_price = float(prev_seg['Close'].iloc[-1]) if 'Close' in prev_seg.columns else float('nan')

    if pd.notna(prev_ts):
        row['Prev Close Timestamp (ET)'] = _to_naive_ts(prev_ts)
        row['Prev Close Price'] = prev_price
    else:
        add_reason('no_prev_close')

    row['Reason'] = '; '.join(reasons)
    return row
