#!/usr/bin/env python
"""
Momentum scanner using Polygon OHLCV parquet files.

Usage:
  python scripts/momentum_scan.py --daily-dir "per_ticker_daily" --top-n 25 --min-dollar-vol 5e6
  # With a whitelist:
  python scripts/momentum_scan.py --daily-dir "per_ticker_daily" --tickers AAPL MSFT NVDA
"""
from __future__ import annotations
import argparse, math, os
from pathlib import Path
import numpy as np
import pandas as pd

DATE_CANDIDATES = ("date","timestamp","t","time","datetime","dt","asofdate","asof_date")

def _z(series: pd.Series) -> pd.Series:
    return (series - series.mean()) / series.std(ddof=0)

def _load_parquet_ohlcv(fp: Path) -> pd.DataFrame | None:
    try:
        dfp = pd.read_parquet(fp)
    except Exception:
        return None
    cols = {str(c).lower(): c for c in dfp.columns}
    dcol = next((cols[k] for k in DATE_CANDIDATES if k in cols), None)
    if not dcol:
        return None
    raw_date = dfp[dcol]
    if pd.api.types.is_numeric_dtype(raw_date):
        mx = pd.to_numeric(raw_date, errors="coerce").max()
        if pd.notna(mx) and mx > 1e12:
            dvals = pd.to_datetime(raw_date, unit="ms", errors="coerce")
        elif pd.notna(mx) and mx > 1e9:
            dvals = pd.to_datetime(raw_date, unit="s", errors="coerce")
        else:
            dvals = pd.to_datetime(raw_date, errors="coerce")
    else:
        dvals = pd.to_datetime(raw_date, errors="coerce")

    def pick(*names: str):
        for n in names:
            if n in cols:
                return cols[n]
        return None

    out = pd.DataFrame({"Date": dvals})
    out["Open"] = pd.to_numeric(dfp[pick("open","o","open_price","openprice")], errors="coerce") if pick("open","o","open_price","openprice") else np.nan
    out["High"] = pd.to_numeric(dfp[pick("high","h","high_price","highprice")], errors="coerce") if pick("high","h","high_price","highprice") else np.nan
    out["Low"]  = pd.to_numeric(dfp[pick("low","l","low_price","lowprice")],  errors="coerce") if pick("low","l","low_price","lowprice") else np.nan
    out["Close"] = pd.to_numeric(dfp[pick("close","c","adj_close","adjclose","close_price","closeprice")], errors="coerce") if pick("close","c","adj_close","adjclose","close_price","closeprice") else np.nan
    out["Volume"] = pd.to_numeric(dfp[pick("volume","v","totalvolume","volume_")], errors="coerce") if pick("volume","v","totalvolume","volume_") else np.nan
    out = out.dropna(subset=["Date","Close"]).sort_values("Date").reset_index(drop=True)
    return out

def _maybe_patch_with_minute(daily: pd.DataFrame, minute_fp: Path | None) -> pd.DataFrame:
    if not minute_fp or not minute_fp.exists():
        return daily
    try:
        m = pd.read_parquet(minute_fp)
        cols = {str(c).lower(): c for c in m.columns}
        tcol = next((cols[k] for k in DATE_CANDIDATES if k in cols), None)
        if not tcol:
            return daily
        ts = pd.to_datetime(m[tcol], unit="ms", errors="coerce") if pd.api.types.is_numeric_dtype(m[tcol]) else pd.to_datetime(m[tcol], errors="coerce")
        m = m.assign(Date=ts).dropna(subset=["Date"])
        m = m.set_index("Date")
        o = m["open"] if "open" in cols else m[cols.get("o")] if cols.get("o") else None
        h = m["high"] if "high" in cols else m[cols.get("h")] if cols.get("h") else None
        l = m["low"] if "low" in cols else m[cols.get("l")] if cols.get("l") else None
        c = m["close"] if "close" in cols else m[cols.get("c")] if cols.get("c") else None
        v = m["volume"] if "volume" in cols else m[cols.get("v")] if cols.get("v") else None
        if any(s is None for s in (o,h,l,c,v)):
            return daily
        daily_m = pd.DataFrame({
            "Open": o.resample("1D").first(),
            "High": h.resample("1D").max(),
            "Low": l.resample("1D").min(),
            "Close": c.resample("1D").last(),
            "Volume": v.resample("1D").sum(),
        }).dropna(subset=["Close"]).reset_index().rename(columns={"index":"Date"})
        merged = pd.concat([daily, daily_m]).sort_values("Date")
        return merged.drop_duplicates(subset=["Date"], keep="last").reset_index(drop=True)
    except Exception:
        return daily

def _reg_slope_tstat(series: pd.Series, window: int) -> pd.Series:
    logp = np.log(series)
    out = []
    idx = logp.index
    for i in range(len(logp)):
        if i + 1 < window:
            out.append(np.nan)
            continue
        y = logp.iloc[i + 1 - window : i + 1].values
        x = np.arange(len(y))
        beta, alpha = np.polyfit(x, y, 1)
        y_hat = alpha + beta * x
        resid = y - y_hat
        s_err = np.sqrt(np.sum(resid ** 2) / (len(y) - 2))
        denom = s_err / math.sqrt(np.sum((x - x.mean()) ** 2)) if len(y) > 2 else np.nan
        tstat = beta / denom if denom and denom != 0 else np.nan
        out.append(tstat)
    return pd.Series(out, index=idx)

def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    g = df.sort_values("Date").copy()
    g["Ret1"] = g["Close"].pct_change()
    for n in (3,5,10,20,60,120,252):
        g[f"R{n}d"] = g["Close"].pct_change(n)
    g["MA10"] = g["Close"].rolling(10).mean()
    g["MA50"] = g["Close"].rolling(50).mean()
    g["MA200"] = g["Close"].rolling(200).mean()
    g["Break20"] = g["Close"] / g["Close"].rolling(20).max() - 1
    hi52 = g["Close"].rolling(252).max()
    lo52 = g["Close"].rolling(252).min()
    g["Pos52w"] = (g["Close"] - lo52) / (hi52 - lo52)
    g["Vol20"] = g["Ret1"].rolling(20).std(ddof=0)
    g["DollarVol20"] = (g["Close"] * g["Volume"]).rolling(20).mean()
    g["SlopeT_60"] = _reg_slope_tstat(g["Close"], 60)
    g["BB_UpPct20"] = (g["Close"] > g["Close"].rolling(20).max()).rolling(20).mean()
    return g

def _compute_scores(feat: pd.DataFrame) -> pd.DataFrame:
    f = feat.copy()
    for col in ("R5d","R10d","R20d","R60d","R120d","SlopeT_60","Vol20","Break20","Pos52w"):
        if col in f:
            f[col] = _z(f[col])
    f["Score_daily"] = 0.6*f.get("R5d") + 0.8*f.get("R20d") + 0.5*f.get("Break20") + 0.3*f.get("SlopeT_60") + 0.2*f.get("Pos52w") - 0.2*f.get("Vol20")
    f["Score_weekly"] = 0.3*f.get("R20d") + 0.5*f.get("R60d") + 0.4*f.get("R120d") + 0.3*f.get("SlopeT_60") + 0.2*f.get("Pos52w")
    return f

def _load_benchmark(daily_dir: Path, benchmark: str) -> dict[str, float]:
    candidates = [
        benchmark,
        benchmark.replace(".","_"),
        benchmark.upper(),
        benchmark.lower(),
    ]
    bench_fp = None
    for name in candidates:
        cand = daily_dir / f"{name}.parquet"
        if cand.exists():
            bench_fp = cand
            break
    if not bench_fp:
        raise SystemExit(f"Benchmark parquet not found for: {benchmark}")
    dfb = _load_parquet_ohlcv(bench_fp)
    if dfb is None or dfb.empty:
        raise SystemExit(f"Benchmark parquet unreadable or empty: {bench_fp}")
    feat = _compute_scores(_build_features(dfb))
    valid = feat.dropna(subset=["R20d","R60d","R120d"])
    if valid.empty:
        raise SystemExit(f"Benchmark lacks sufficient data: {bench_fp}")
    last = valid.iloc[-1]
    return {
        "ticker": bench_fp.stem.upper(),
        "R20d": last.get("R20d"),
        "R60d": last.get("R60d"),
        "R120d": last.get("R120d"),
    }

def _scan(daily_dir: Path, minute_dir: Path | None, tickers: set[str] | None, top_n: int, min_dollar_vol: float, bench_stats: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows_daily, rows_weekly = [], []
    last_dates = []
    for i, fp in enumerate(sorted(daily_dir.glob("*.parquet")), start=1):
        if i % 50 == 0:
            print(f"... processed {i} tickers (latest: {fp.stem})")
        tkr = fp.stem.upper()
        if tickers and tkr not in tickers:
            continue
        df = _load_parquet_ohlcv(fp)
        if df is None or df.empty:
            continue
        minute_fp = None
        if minute_dir and minute_dir.is_dir():
            for alt in (f"{tkr}.parquet", f"{tkr.replace('.','_')}.parquet", tkr.lower()+".parquet"):
                cand = minute_dir / alt
                if cand.exists():
                    minute_fp = cand
                    break
        df = _maybe_patch_with_minute(df, minute_fp)
        feat = _compute_scores(_build_features(df))
        valid = feat.dropna(subset=["Score_daily","Score_weekly","DollarVol20"])
        if valid.empty:
            continue
        last = valid.iloc[-1]
        # Require constructive trend: above 10/50/200, MA50 > MA200, and positive momentum.
        if pd.isna(last.get("MA10")) or pd.isna(last.get("MA50")) or pd.isna(last.get("MA200")):
            continue
        if last.get("R20d", 0) <= 0 or last.get("R60d", 0) <= 0:
            continue
        if last["Close"] < last["MA10"] or last["Close"] < last["MA50"] or last["Close"] < last["MA200"]:
            continue
        if last["MA50"] < last["MA200"]:
            continue
        if pd.isna(last.get("Pos52w")) or last["Pos52w"] < (2/3):
            continue
        if last.get("Break20", 0) <= 0:
            continue
        # Relative strength vs benchmark on 20d and 60d.
        if bench_stats and (
            last.get("R20d", float("-inf")) <= bench_stats.get("R20d", float("inf"))
            or last.get("R60d", float("-inf")) <= bench_stats.get("R60d", float("inf"))
        ):
            continue
        if last["DollarVol20"] < min_dollar_vol:
            continue
        last_dates.append(last["Date"])
        rows_daily.append({"Ticker": tkr, "Date": last["Date"], "Score": last["Score_daily"], "R5d%": last["R5d"]*100, "R20d%": last["R20d"]*100, "R60d%": last["R60d"]*100, "DV20": last["DollarVol20"]})
        rows_weekly.append({"Ticker": tkr, "Date": last["Date"], "Score": last["Score_weekly"], "R20d%": last["R20d"]*100, "R60d%": last["R60d"]*100, "R120d%": last["R120d"]*100, "DV20": last["DollarVol20"]})
    if not last_dates:
        return pd.DataFrame(), pd.DataFrame()
    freshest = pd.to_datetime(max(last_dates))
    daily_rank = pd.DataFrame(rows_daily)
    weekly_rank = pd.DataFrame(rows_weekly)
    daily_rank = daily_rank[daily_rank["Date"] == freshest].sort_values("Score", ascending=False).head(top_n)
    weekly_rank = weekly_rank[weekly_rank["Date"] == freshest].sort_values("Score", ascending=False).head(top_n)
    return daily_rank, weekly_rank

def main():
    p = argparse.ArgumentParser(description="Polygon-based momentum scanner")
    p.add_argument("--daily-dir", required=True, type=Path, help="Folder with per-ticker daily parquet")
    p.add_argument("--minute-dir", type=Path, help="Optional folder with per-ticker minute parquet (used only to fill the most recent day)")
    p.add_argument("--tickers", nargs="+", help="Optional whitelist of tickers to scan")
    p.add_argument("--top-n", type=int, default=25)
    p.add_argument("--min-dollar-vol", type=float, default=5_000_000, help="Min 20d avg dollar volume")
    p.add_argument("--benchmark", default="QQQ", help="Ticker to use as benchmark for relative strength (expects matching parquet in daily-dir)")
    args = p.parse_args()

    daily_dir = args.daily_dir
    if not daily_dir.is_dir():
        raise SystemExit(f"Daily dir not found: {daily_dir}")

    bench_stats = _load_benchmark(daily_dir, args.benchmark)
    tickers = set(t.upper() for t in args.tickers) if args.tickers else None
    daily_rank, weekly_rank = _scan(daily_dir, args.minute_dir, tickers, args.top_n, args.min_dollar_vol, bench_stats)

    print(f"Benchmark: {bench_stats['ticker']} (R20d={bench_stats['R20d']:.4f}, R60d={bench_stats['R60d']:.4f}, R120d={bench_stats['R120d']:.4f})")
    print("\nTop daily momentum")
    print(daily_rank.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))
    print("\nTop weekly momentum")
    print(weekly_rank.to_string(index=False, float_format=lambda x: f"{x:,.2f}"))

if __name__ == "__main__":
    main()
