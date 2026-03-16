#!/usr/bin/env python
r"""
Compute simple off-open returns for one ticker/date from a local minute parquet.

Usage (PowerShell example):
  py -3.11 .\codex_ta_toolkit\tools\compute_offopen.py \
      --ticker AAPL --date 2025-01-21 \
      --minute "C:\Users\David Alcosser\Documents\Visual Code\codex_ta_toolkit\per_ticker_minute" \
      --marks 1 3 5 10 15 --tol-min 5 --out aapl_2025-01-21_offopen.csv

Outputs a one-row CSV with:
  - Open (picked) timestamp
  - 9:24 to Open return (%)
  - N Min Return (%) for each requested minute mark
and prints a diagnostics block to the console (picked bars/timestamps).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd


def _find_minute_file(minute_dir: Path, ticker: str) -> Optional[Path]:
    t = ticker.strip().upper().replace('.', '_')
    cand = minute_dir / f"{t}.parquet"
    if cand.exists():
        return cand
    # also support per-ticker subfolder
    sub = minute_dir / t
    if sub.is_dir():
        best: Optional[Path] = None
        for f in sub.glob("*.parquet"):
            if best is None or f.stat().st_size > best.stat().st_size:
                best = f
        return best
    # loose scan
    for f in minute_dir.glob(f"{t}*.parquet"):
        return f
    return None


def _ensure_ny_index(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of df with a tz-aware America/New_York DatetimeIndex.

    Handles:
      - Existing DatetimeIndex (naive -> assume UTC, then convert)
      - TZ-aware DatetimeIndex
      - Columns: Timestamp/t/start/window_start (epoch s/ms/us/ns or strings)
      - Fallback: compose from 'date' + 'time' (use left HH:MM if range)
    """
    # 1) Already has a datetime-like index
    if isinstance(df.index, pd.DatetimeIndex):
        idx = df.index
        try:
            if idx.tz is None:
                idx = idx.tz_localize('UTC')
            idx = idx.tz_convert('America/New_York')
        except Exception:
            # If localization fails, coerce via to_datetime
            idx = pd.to_datetime(idx, errors='coerce', utc=True).tz_convert('America/New_York')
        out = df.copy(); out.index = idx
        return out.sort_index()

    # 2) Try common datetime columns
    col_order = ['Timestamp','timestamp','t','start','window_start','time','date']
    cols = {str(c).lower(): c for c in df.columns}
    chosen = None
    for name in col_order:
        c = cols.get(name.lower())
        if c in df.columns:
            chosen = c
            break
    idx = None
    if chosen is not None:
        s = df[chosen]
        # If tz-aware already
        try:
            tz_attr = getattr(getattr(s, 'dtype', None), 'tz', None)
        except Exception:
            tz_attr = None
        try:
            if tz_attr is not None:
                idx = pd.DatetimeIndex(s).tz_convert('America/New_York')
            elif pd.api.types.is_numeric_dtype(s):
                sv = pd.to_numeric(s, errors='coerce')
                mx = sv.dropna().abs().max()
                if mx is None or pd.isna(mx):
                    idx = pd.to_datetime(sv, errors='coerce', utc=True)
                else:
                    unit = 'ns' if mx >= 1e18 else ('us' if mx >= 1e15 else ('ms' if mx >= 1e12 else 's'))
                    idx = pd.to_datetime(sv, errors='coerce', utc=True, unit=unit)
            else:
                # text/ISO-like
                idx = pd.to_datetime(s.astype(str), errors='coerce', utc=True)
            if getattr(idx, 'tz', None) is None:
                idx = idx.tz_localize('UTC')
            idx = idx.tz_convert('America/New_York')
        except Exception as e:
            idx = None
    # 3) Fallback: compose from date + time (extract left HH:MM if range)
    if idx is None:
        dcol = cols.get('date'); tcol = cols.get('time')
        if dcol and tcol and (dcol in df.columns) and (tcol in df.columns):
            ds = pd.to_datetime(df[dcol], errors='coerce')
            tt = df[tcol].astype(str).str.extract(r'^(\d{1,2}:\d{2})')[0]
            combo = pd.to_datetime(ds.dt.strftime('%Y-%m-%d') + ' ' + tt, errors='coerce')
            idx = combo.dt.tz_localize('America/New_York')
        else:
            raise RuntimeError(f"No datetime column/index found. Columns={list(df.columns)}")
    out = df.copy(); out.index = pd.DatetimeIndex(idx)
    return out.sort_index()


def compute_offopen(df: pd.DataFrame, date_str: str, marks: List[int], tol_min: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    tz = 'America/New_York'
    day = pd.Timestamp(date_str, tz=tz)
    d0 = day.normalize()
    d1 = d0 + pd.Timedelta(days=1)
    d = df.loc[d0:d1]
    if d.empty:
        raise RuntimeError(f"No minutes for {date_str} (ET)")

    t_open = day.replace(hour=9, minute=30, second=0)
    win_open = d[(d.index >= t_open) & (d.index <= t_open + pd.Timedelta(minutes=tol_min))]
    if not win_open.empty:
        open_dt = win_open.index[0]
        p_open = float(win_open['Close'].iloc[0] if 'Close' in win_open.columns else win_open.iloc[0]['close'])
    else:
        pos = d.index.get_indexer([t_open], method='nearest')
        if not (pos.size and pos[0] != -1):
            raise RuntimeError("Could not locate 09:30 bar within tolerance")
        open_dt = d.index[int(pos[0])]
        if abs((open_dt - t_open).total_seconds()) > 60 * tol_min:
            raise RuntimeError("Nearest 09:30 bar beyond tolerance")
        p_open = float(d['Close'].iloc[int(pos[0])] if 'Close' in d.columns else d.iloc[int(pos[0])]['close'])

    # Pre 09:24: last bar at or before 09:24 within morning window
    t_924 = day.replace(hour=9, minute=24, second=0)
    pre = d.loc[:t_924]
    p_pre = float('nan')
    pre_dt = None
    if not pre.empty:
        pre_dt = pre.index[-1]
        p_pre = float(pre['Close'].iloc[-1] if 'Close' in pre.columns else pre.iloc[-1]['close'])

    out = {
        'Date': d0.tz_localize(None).date(),
        'Open (picked)': str(open_dt),
        '9:24 to Open': ((p_open / p_pre - 1.0) * 100.0) if (p_pre == p_pre and p_pre != 0) else pd.NA,
    }

    rows_dbg = [
        {'target': '09:30', 'picked': str(open_dt), 'delta_s': int((open_dt - t_open).total_seconds()), 'price': p_open},
        {'target': '09:24', 'picked': str(pre_dt) if pre_dt is not None else '-', 'delta_s': None, 'price': p_pre},
    ]

    for m in marks:
        t_m = t_open + pd.Timedelta(minutes=int(m))
        win_m = d[(d.index >= t_m) & (d.index <= t_m + pd.Timedelta(minutes=tol_min))]
        if not win_m.empty:
            pick = win_m.index[0]
            p_m = float(win_m['Close'].iloc[0] if 'Close' in win_m.columns else win_m.iloc[0]['close'])
        else:
            im = d.index.get_indexer([t_m], method='nearest')
            if not (im.size and im[0] != -1):
                pick = None
                p_m = float('nan')
            else:
                pick = d.index[int(im[0])]
                if abs((pick - t_m).total_seconds()) > 60 * tol_min:
                    pick = None
                    p_m = float('nan')
                else:
                    p_m = float(d['Close'].iloc[int(im[0])] if 'Close' in d.columns else d.iloc[int(im[0])]['close'])
        out[f"{int(m)} Min Return"] = ((p_m / p_open - 1.0) * 100.0) if (pick is not None) else pd.NA
        rows_dbg.append({'target': f'09:30+{int(m)}m', 'picked': str(pick) if pick is not None else '-', 'delta_s': (None if pick is None else int((pick - t_m).total_seconds())), 'price': (None if pick is None else p_m)})

    return pd.DataFrame([out]).set_index('Date'), pd.DataFrame(rows_dbg)


def main() -> None:
    ap = argparse.ArgumentParser(description='Compute off-open returns from local minute parquet')
    ap.add_argument('--ticker', required=True)
    ap.add_argument('--date', required=True, help='YYYY-MM-DD (ET)')
    ap.add_argument('--minute', required=True, help='Folder containing per-ticker minute parquet files')
    ap.add_argument('--marks', nargs='*', type=int, default=[1,3,5,10,15])
    ap.add_argument('--tol-min', type=int, default=5, help='Tolerance window in minutes')
    ap.add_argument('--out', type=str, default=None, help='Optional CSV output path')
    args = ap.parse_args()

    minute_dir = Path(args.minute)
    if not minute_dir.exists():
        raise SystemExit(f"Minute folder not found: {minute_dir}")
    fp = _find_minute_file(minute_dir, args.ticker)
    if not fp or not fp.exists():
        raise SystemExit(f"Minute parquet not found for {args.ticker} in {minute_dir}")

    df = pd.read_parquet(fp)
    # normalize column names a bit
    cols = {str(c): c for c in df.columns}
    if 'Close' not in cols and 'close' not in cols:
        # try typical Polygon names
        if 'c' in cols:
            df = df.rename(columns={'c': 'Close'})
    try:
        df = _ensure_ny_index(df)
    except Exception as e:
        raise SystemExit(f"Failed to normalize timestamps to ET: {e}")

    res, dbg = compute_offopen(df, args.date, args.marks, tol_min=int(args.tol_min))
    print("\n== Off-Open ==")
    print(res.to_string())
    print("\n== Diagnostics ==")
    print(dbg.to_string(index=False))

    if args.out:
        outp = Path(args.out)
        try:
            res.to_csv(outp)
            print(f"\nSaved: {outp}")
        except Exception as e:
            print(f"Failed to write CSV: {e}")


if __name__ == '__main__':
    main()
