import os
import pickle
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import dash
import pandas as pd
import plotly.graph_objects as go
import requests
from dash import Input, Output, State, dcc, html, dash_table, no_update


BASE_URL = "https://api.massive.com"
CACHE_FILE = Path("ticker_cache.pkl")
DEBUG_LOG = Path("app_debug.log")
ET = ZoneInfo("America/New_York")
REPLAY_CACHE: dict[tuple[str, int], pd.DataFrame] = {}
MINUTE_CACHE: dict[tuple[str, str, str], pd.DataFrame] = {}


def log_msg(msg: str) -> None:
    try:
        with DEBUG_LOG.open("a", encoding="utf-8") as f:
            f.write(f"{datetime.now().isoformat()} | {msg}\n")
    except Exception:
        pass


def get_api_key() -> str:
    key = os.getenv("MASSIVE_API_KEY", "").strip()
    if key:
        return key

    key_file = Path("massive_api_key.txt")
    if key_file.exists():
        v = key_file.read_text(encoding="utf-8").strip()
        if v:
            os.environ["MASSIVE_API_KEY"] = v
            return v
    return ""


def req_json(url: str, params: dict | None = None, timeout: int = 30) -> dict:
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError("MASSIVE_API_KEY is not set")

    q = dict(params or {})
    q["apiKey"] = api_key
    resp = requests.get(url, params=q, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def get_all_us_tickers_from_api() -> pd.DataFrame:
    all_rows: list[pd.DataFrame] = []
    next_url = f"{BASE_URL}/v3/reference/tickers"

    while next_url:
        try:
            payload = req_json(
                next_url,
                {
                    "market": "stocks",
                    "locale": "us",
                    "active": "true",
                    "type": "CS",
                    "limit": 1000,
                },
                timeout=60,
            )

            rows = payload.get("results", [])
            if rows:
                df = pd.DataFrame(rows)
                keep = [c for c in ["ticker", "name", "primary_exchange"] if c in df.columns]
                all_rows.append(df[keep])

            next_url = payload.get("next_url")
            time.sleep(0.1)
        except Exception as exc:
            log_msg(f"Error fetching tickers: {exc}")
            next_url = None

    if not all_rows:
        return pd.DataFrame(columns=["ticker", "name", "primary_exchange"])
    return pd.concat(all_rows, ignore_index=True).drop_duplicates(subset=["ticker"])


def get_all_us_tickers_cached() -> pd.DataFrame:
    today = date.today().isoformat()
    if CACHE_FILE.exists():
        try:
            with CACHE_FILE.open("rb") as f:
                cache = pickle.load(f)
            if cache.get("date") == today:
                v = cache.get("tickers")
                if isinstance(v, pd.DataFrame):
                    return v
        except Exception:
            pass

    df = get_all_us_tickers_from_api()
    if not df.empty:
        with CACHE_FILE.open("wb") as f:
            pickle.dump({"date": today, "tickers": df}, f)
    return df


def is_premarket_session(now_et: datetime | None = None) -> bool:
    now_et = now_et or datetime.now(ET)
    if now_et.weekday() >= 5:
        return False
    hm = now_et.strftime("%H:%M")
    return "04:00" <= hm < "09:30"


def get_market_snapshot() -> pd.DataFrame:
    payload = req_json(f"{BASE_URL}/v2/snapshot/locale/us/markets/stocks/tickers", timeout=60)
    rows = payload.get("tickers", [])
    if not rows:
        return pd.DataFrame(columns=["ticker", "base_price", "end_price", "change_pct", "volume"])

    out = []
    for r in rows:
        prev_day = r.get("prevDay") or {}
        day = r.get("day") or {}
        minute = r.get("min") or {}

        base_price = prev_day.get("c")
        end_price = minute.get("c")
        if base_price is None or end_price is None or base_price <= 0:
            continue

        out.append(
            {
                "ticker": r.get("ticker"),
                "change_pct": r.get("todaysChangePerc"),
                "todays_change": r.get("todaysChange"),
                "base_price": base_price,
                "open_price": day.get("o"),
                "end_price": end_price,
                "volume": day.get("v"),
                "updated": r.get("updated"),
            }
        )

    return pd.DataFrame(out)


def get_unified_snapshot(tickers: list[str]) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame(columns=["ticker", "base_price", "end_price", "change_pct", "volume"])

    chunks = [tickers[i : i + 250] for i in range(0, len(tickers), 250)]
    out: list[dict] = []

    for chunk in chunks:
        try:
            payload = req_json(
                f"{BASE_URL}/v3/snapshot",
                {"ticker.any_of": ",".join(chunk), "limit": 250},
                timeout=60,
            )
            for item in payload.get("results", []):
                sess = item.get("session") or {}
                prev_close = sess.get("previous_close")
                early_change = sess.get("early_trading_change")
                pm_price = sess.get("price")
                if prev_close is not None and early_change is not None:
                    pm_price = prev_close + early_change

                out.append(
                    {
                        "ticker": item.get("ticker"),
                        "base_price": prev_close,
                        "end_price": pm_price,
                        "change_pct": sess.get("early_trading_change_percent"),
                        "volume": sess.get("volume"),
                    }
                )
        except Exception as exc:
            log_msg(f"Error in unified snapshot: {exc}")

        time.sleep(0.25)

    df = pd.DataFrame(out)
    if df.empty:
        return df
    return df[df["ticker"].notna() & df["base_price"].notna() & (df["base_price"] > 0)].drop_duplicates(subset=["ticker"])


def get_market_caps_bulk(tickers: list[str]) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame(columns=["ticker", "name", "market_cap"])

    chunks = [tickers[i : i + 100] for i in range(0, len(tickers), 100)]
    out = []

    for chunk in chunks:
        try:
            payload = req_json(
                f"{BASE_URL}/v3/reference/tickers",
                {
                    "ticker.any_of": ",".join(chunk),
                    "active": "true",
                    "limit": 250,
                },
                timeout=30,
            )
            rows = payload.get("results", [])
            if rows:
                df = pd.DataFrame(rows)
                keep = [c for c in ["ticker", "name", "market_cap"] if c in df.columns]
                out.append(df[keep])
        except Exception as exc:
            log_msg(f"Error getting market caps: {exc}")

        time.sleep(0.25)

    if not out:
        return pd.DataFrame(columns=["ticker", "name", "market_cap"])

    return pd.concat(out, ignore_index=True).drop_duplicates(subset=["ticker"])


def get_minute_bars(ticker: str, from_date: str, to_date: str) -> pd.DataFrame:
    cache_key = (ticker, from_date, to_date)
    cached = MINUTE_CACHE.get(cache_key)
    if cached is not None:
        return cached.copy()

    for attempt in range(3):
        try:
            payload = req_json(
                f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/1/minute/{from_date}/{to_date}",
                {"adjusted": "true", "sort": "asc", "limit": 50000},
                timeout=45,
            )
            rows = payload.get("results", [])
            if not rows:
                return pd.DataFrame()

            df = pd.DataFrame(rows)
            df["datetime_et"] = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert(ET)
            df["ticker"] = ticker
            cols = [c for c in ["ticker", "datetime_et", "t", "o", "h", "l", "c", "v", "vw", "n"] if c in df.columns]
            out = df[cols]
            MINUTE_CACHE[cache_key] = out.copy()
            if len(MINUTE_CACHE) > 400:
                MINUTE_CACHE.pop(next(iter(MINUTE_CACHE)))
            return out
        except Exception as exc:
            if attempt < 2:
                time.sleep(0.35 * (attempt + 1))
                continue
            log_msg(f"Minute bar error {ticker}: {exc}")
            return pd.DataFrame()


def get_daily_bars(ticker: str, from_date: str, to_date: str) -> pd.DataFrame:
    try:
        payload = req_json(
            f"{BASE_URL}/v2/aggs/ticker/{ticker}/range/1/day/{from_date}/{to_date}",
            {"adjusted": "true", "sort": "asc", "limit": 50000},
            timeout=45,
        )
        rows = payload.get("results", [])
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows)
        dt = pd.to_datetime(df["t"], unit="ms", utc=True).dt.tz_convert(ET)
        df["date"] = dt.dt.date
        df["ticker"] = ticker
        return df[["ticker", "date", "o", "h", "l", "c", "v"]]
    except Exception as exc:
        log_msg(f"Daily bar error {ticker}: {exc}")
        return pd.DataFrame()


def get_previous_close(ticker: str, replay_date: date) -> float | None:
    start = (replay_date - timedelta(days=10)).strftime("%Y-%m-%d")
    end = replay_date.strftime("%Y-%m-%d")
    d = get_daily_bars(ticker, start, end)
    if d.empty:
        return None
    d = d[d["date"] < replay_date].sort_values("date")
    if d.empty:
        return None
    val = pd.to_numeric(d["c"], errors="coerce").iloc[-1]
    return None if pd.isna(val) else float(val)


def build_replay_row(ticker: str, replay_date: date) -> dict | None:
    prev_close = get_previous_close(ticker, replay_date)
    if prev_close is None or prev_close <= 0:
        return None

    day = replay_date.strftime("%Y-%m-%d")
    m = get_minute_bars(ticker, day, day)
    if m.empty:
        return None

    m = m.copy()
    m["time_hm"] = m["datetime_et"].dt.strftime("%H:%M")
    pm = m[(m["time_hm"] >= "04:00") & (m["time_hm"] < "09:30")].sort_values("datetime_et")
    if pm.empty:
        return None

    open_price = pd.to_numeric(pm["o"], errors="coerce").iloc[0]
    end_price = pd.to_numeric(pm["c"], errors="coerce").iloc[-1]
    pm_v = pd.to_numeric(pm["v"], errors="coerce").fillna(0)
    pm_c = pd.to_numeric(pm["c"], errors="coerce").fillna(0)
    volume = pm_v.sum()
    notional = (pm_v * pm_c).sum()
    if pd.isna(end_price):
        return None

    end_price_f = float(end_price)
    open_price_f = None if pd.isna(open_price) else float(open_price)
    volume_f = float(volume)
    change = end_price_f - prev_close
    change_pct = (change / prev_close) * 100.0 if prev_close > 0 else None

    return {
        "ticker": ticker,
        "change_pct": change_pct,
        "todays_change": change,
        "base_price": float(prev_close),
        "open_price": open_price_f,
        "end_price": end_price_f,
        "volume": volume_f,
        "notional": float(notional),
    }


def get_liquid_universe(max_tickers: int) -> list[str]:
    try:
        live = get_market_snapshot()
        if not live.empty:
            live["end_price"] = pd.to_numeric(live.get("end_price"), errors="coerce")
            live["volume"] = pd.to_numeric(live.get("volume"), errors="coerce")
            live["notional"] = live["end_price"].fillna(0) * live["volume"].fillna(0)
            tickers = (
                live.sort_values("notional", ascending=False)["ticker"]
                .dropna()
                .astype(str)
                .drop_duplicates()
                .head(max(10, int(max_tickers)))
                .tolist()
            )
            if tickers:
                return tickers
    except Exception as exc:
        log_msg(f"Liquid universe fallback error: {exc}")

    universe = get_all_us_tickers_cached()
    if universe.empty or "ticker" not in universe.columns:
        return []
    return (
        universe["ticker"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .sort_values()
        .head(max(10, int(max_tickers)))
        .tolist()
    )


def get_replay_snapshot(replay_date: date, max_tickers: int = 500) -> tuple[pd.DataFrame, dict]:
    cache_key = (replay_date.isoformat(), int(max_tickers))
    if cache_key in REPLAY_CACHE:
        df = REPLAY_CACHE[cache_key].copy()
        return df, {"universe": int(max_tickers), "rows": len(df), "cached": True}

    tickers = get_liquid_universe(max_tickers=max_tickers)
    if not tickers:
        return pd.DataFrame(columns=["ticker", "base_price", "end_price", "change_pct", "volume"]), {
            "universe": 0,
            "rows": 0,
        }

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for row in ex.map(lambda t: build_replay_row(t, replay_date), tickers):
            if row is not None:
                rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["ticker", "base_price", "end_price", "change_pct", "volume"]), {
            "universe": len(tickers),
            "rows": 0,
            "cached": False,
        }
    out = pd.DataFrame(rows)
    REPLAY_CACHE[cache_key] = out.copy()
    if len(REPLAY_CACHE) > 20:
        REPLAY_CACHE.pop(next(iter(REPLAY_CACHE)))
    return out, {"universe": len(tickers), "rows": len(rows), "cached": False}


def find_historical_gaps(ticker_daily: pd.DataFrame, current_change_pct: float, direction: str) -> pd.DataFrame:
    if ticker_daily.empty or len(ticker_daily) < 2:
        return pd.DataFrame()

    d = ticker_daily.sort_values("date").copy()
    d["prev_close"] = d["c"].shift(1)
    d["gap_pct"] = (d["o"] - d["prev_close"]) / d["prev_close"] * 100.0
    d = d[d["gap_pct"].notna()]
    if d.empty:
        return d

    threshold = 0.75 * current_change_pct
    if direction == "gainer":
        m = d[(d["gap_pct"] >= threshold) & (d["gap_pct"] > 0)]
    else:
        m = d[(d["gap_pct"] <= threshold) & (d["gap_pct"] < 0)]

    today = date.today()
    m = m[m["date"] < today].sort_values("date", ascending=False).head(10)
    if m.empty:
        return m

    return m[["ticker", "date", "gap_pct", "prev_close", "o"]].rename(columns={"o": "open_price"})


def calculate_interval_changes(minute_data: pd.DataFrame, gap_date: date) -> pd.DataFrame:
    if minute_data.empty:
        return pd.DataFrame()

    d = minute_data.copy()
    d["date"] = d["datetime_et"].dt.date
    d["time_hm"] = d["datetime_et"].dt.strftime("%H:%M")
    d = d[(d["date"] == gap_date) & (d["time_hm"] >= "09:30") & (d["time_hm"] <= "10:30")].sort_values("datetime_et")
    if d.empty:
        return pd.DataFrame()

    open_row = d[d["time_hm"] >= "09:30"].head(1)
    if open_row.empty:
        return pd.DataFrame()

    open_930 = float(open_row["o"].iloc[0])

    def px(t: str) -> float | None:
        r = d[d["time_hm"] == t]
        if r.empty:
            return None
        return float(r["c"].iloc[0])

    def chg(t: str) -> float | None:
        p = px(t)
        if p is None:
            return None
        return (p - open_930) / open_930 * 100.0

    return pd.DataFrame(
        [
            {
                "gap_date": gap_date,
                "open_930": open_930,
                "chg_1m": chg("09:31"),
                "chg_3m": chg("09:33"),
                "chg_5m": chg("09:35"),
                "chg_10m": chg("09:40"),
                "chg_15m": chg("09:45"),
                "chg_30m": chg("10:00"),
                "chg_60m": chg("10:30"),
            }
        ]
    )


def aggregate_interval_stats(interval_results: pd.DataFrame, ticker: str) -> dict:
    out = {"ticker": ticker, "n_instances": int(len(interval_results))}

    for m in [1, 3, 5, 10, 15, 30, 60]:
        col = f"chg_{m}m"
        if interval_results.empty or col not in interval_results.columns:
            out[f"pct_positive_{m}m"] = None
            out[f"avg_change_{m}m"] = None
            out[f"median_change_{m}m"] = None
            continue

        x = pd.to_numeric(interval_results[col], errors="coerce").dropna()
        if x.empty:
            out[f"pct_positive_{m}m"] = None
            out[f"avg_change_{m}m"] = None
            out[f"median_change_{m}m"] = None
            continue

        out[f"pct_positive_{m}m"] = float((x > 0).mean() * 100.0)
        out[f"avg_change_{m}m"] = float(x.mean())
        out[f"median_change_{m}m"] = float(x.median())

    return out


def calculate_3day_chart_window(gap_date: date) -> tuple[str, str]:
    from_date = (gap_date - timedelta(days=2)).strftime("%Y-%m-%d")
    to_date = gap_date.strftime("%Y-%m-%d")
    return from_date, to_date


def prepare_chart_data(minute_data: pd.DataFrame, gap_date: date) -> pd.DataFrame:
    if minute_data.empty:
        return pd.DataFrame()

    d = minute_data.copy().sort_values("datetime_et")
    d["datetime_et"] = pd.to_datetime(d["datetime_et"], errors="coerce")
    d["c"] = pd.to_numeric(d["c"], errors="coerce")
    d = d[d["datetime_et"].notna() & d["c"].notna()].copy()
    if d.empty:
        return d
    d["date"] = d["datetime_et"].dt.date
    d["time_hm"] = d["datetime_et"].dt.strftime("%H:%M")

    d = d[(d["time_hm"] >= "04:00") & (d["time_hm"] <= "20:00")].copy()
    if d.empty:
        return d

    def session(t: str) -> str:
        if "04:00" <= t < "09:30":
            return "Pre-Market"
        if "09:30" <= t <= "16:00":
            return "Regular"
        return "Post-Market"

    d["session"] = d["time_hm"].map(session)
    d["is_gap_day"] = d["date"] == gap_date
    d["bar_index"] = range(1, len(d) + 1)
    return d


def prepare_zoom_chart_data(minute_data: pd.DataFrame, gap_date: date) -> pd.DataFrame:
    if minute_data.empty:
        return pd.DataFrame()

    d = minute_data.copy()
    d["datetime_et"] = pd.to_datetime(d["datetime_et"], errors="coerce")
    d["c"] = pd.to_numeric(d["c"], errors="coerce")
    d = d[d["datetime_et"].notna() & d["c"].notna()].copy()
    if d.empty:
        return d
    d["date"] = d["datetime_et"].dt.date
    d["time_hm"] = d["datetime_et"].dt.strftime("%H:%M")
    d = d[(d["date"] == gap_date) & (d["time_hm"] >= "09:00") & (d["time_hm"] <= "10:00")].sort_values("datetime_et")
    if d.empty:
        return d

    d["session"] = d["time_hm"].apply(lambda t: "Pre-Market" if t < "09:30" else "Regular")
    d["bar_index"] = range(1, len(d) + 1)
    d["time_label"] = d["datetime_et"].dt.strftime("%Y-%m-%d %H:%M")
    return d


def build_zoom_shapes(zoom_data: pd.DataFrame) -> list[dict]:
    shapes: list[dict] = []
    if zoom_data.empty:
        return shapes

    pm = zoom_data[zoom_data["session"] == "Pre-Market"]
    if not pm.empty:
        shapes.append(
            {
                "type": "rect",
                "x0": float(pm["bar_index"].min()) - 0.5,
                "x1": float(pm["bar_index"].max()) + 0.5,
                "y0": 0,
                "y1": 1,
                "xref": "x",
                "yref": "paper",
                "fillcolor": "rgba(255, 255, 255, 0.05)",
                "line": {"width": 0},
                "layer": "below",
            }
        )

    boundary = zoom_data[zoom_data["time_hm"] == "09:30"]
    if not boundary.empty:
        x = float(boundary["bar_index"].min()) - 0.5
        shapes.append(
            {
                "type": "line",
                "x0": x,
                "x1": x,
                "y0": 0,
                "y1": 1,
                "xref": "x",
                "yref": "paper",
                "line": {"color": "rgba(255,255,255,0.18)", "width": 1, "dash": "dot"},
            }
        )
    return shapes


def build_session_shapes(chart_data: pd.DataFrame, gap_date: date | None = None) -> list[dict]:
    shapes: list[dict] = []
    if chart_data.empty:
        return shapes

    for (d, sess), seg in chart_data.groupby(["date", "session"], sort=False):
        x0 = float(seg["bar_index"].min()) - 0.5
        x1 = float(seg["bar_index"].max()) + 0.5

        if sess == "Pre-Market":
            fill = "rgba(255,255,255,0.035)"
        elif sess == "Post-Market":
            fill = "rgba(255,255,255,0.02)"
        else:
            fill = "rgba(0,191,255,0.045)" if (gap_date is not None and d == gap_date) else "rgba(255,255,255,0.0)"

        if fill != "rgba(255,255,255,0.0)":
            shapes.append(
                {
                    "type": "rect",
                    "x0": x0,
                    "x1": x1,
                    "y0": 0,
                    "y1": 1,
                    "xref": "x",
                    "yref": "paper",
                    "fillcolor": fill,
                    "line": {"width": 0},
                    "layer": "below",
                }
            )

        if sess == "Regular":
            x = x0
            shapes.append(
                {
                    "type": "line",
                    "x0": x,
                    "x1": x,
                    "y0": 0,
                    "y1": 1,
                    "xref": "x",
                    "yref": "paper",
                    "line": {"color": "rgba(255,255,255,0.14)", "width": 1, "dash": "dot"},
                }
            )

    return shapes


def format_money(x: float | None) -> str:
    if x is None or pd.isna(x):
        return ""
    return f"${x:,.0f}"


def base_table_style() -> dict:
    return {
        "style_table": {"overflowX": "auto", "border": "1px solid #2f2f2f", "borderRadius": "6px"},
        "style_cell": {
            "backgroundColor": "#1b1b1b",
            "color": "#eee",
            "fontSize": 12,
            "padding": "6px",
            "border": "1px solid #2a2a2a",
            "textAlign": "right",
        },
        "style_cell_conditional": [{"if": {"column_id": "ticker"}, "textAlign": "left"}, {"if": {"column_id": "name"}, "textAlign": "left"}, {"if": {"column_id": "Date"}, "textAlign": "left"}],
        "style_header": {"backgroundColor": "#262626", "color": "#fff", "fontWeight": "bold"},
        "style_data": {"whiteSpace": "normal", "height": "auto"},
        "style_data_conditional": [{"if": {"row_index": "odd"}, "backgroundColor": "#171717"}],
    }


def labeled_control(label: str, control, width: str = "140px"):
    return html.Div(
        style={"display": "flex", "flexDirection": "column", "gap": "4px", "minWidth": width},
        children=[
            html.Div(label, style={"fontSize": 11, "color": "#90a4ae", "fontWeight": "bold"}),
            control,
        ],
    )


def build_base_layout() -> html.Div:
    tbl = base_table_style()
    detail_tbl = dict(tbl)
    detail_tbl["style_data_conditional"] = tbl.get("style_data_conditional", []) + [
        {"if": {"column_id": "Date"}, "textAlign": "left"},
        {"if": {"filter_query": "{1m} > 0", "column_id": "1m"}, "color": "#66bb6a", "fontWeight": "bold"},
        {"if": {"filter_query": "{1m} < 0", "column_id": "1m"}, "color": "#ef5350", "fontWeight": "bold"},
        {"if": {"filter_query": "{3m} > 0", "column_id": "3m"}, "color": "#66bb6a"},
        {"if": {"filter_query": "{3m} < 0", "column_id": "3m"}, "color": "#ef5350"},
        {"if": {"filter_query": "{5m} > 0", "column_id": "5m"}, "color": "#66bb6a"},
        {"if": {"filter_query": "{5m} < 0", "column_id": "5m"}, "color": "#ef5350"},
        {"if": {"filter_query": "{10m} > 0", "column_id": "10m"}, "color": "#66bb6a"},
        {"if": {"filter_query": "{10m} < 0", "column_id": "10m"}, "color": "#ef5350"},
        {"if": {"filter_query": "{15m} > 0", "column_id": "15m"}, "color": "#66bb6a"},
        {"if": {"filter_query": "{15m} < 0", "column_id": "15m"}, "color": "#ef5350"},
        {"if": {"filter_query": "{30m} > 0", "column_id": "30m"}, "color": "#66bb6a"},
        {"if": {"filter_query": "{30m} < 0", "column_id": "30m"}, "color": "#ef5350"},
        {"if": {"filter_query": "{60m} > 0", "column_id": "60m"}, "color": "#66bb6a"},
        {"if": {"filter_query": "{60m} < 0", "column_id": "60m"}, "color": "#ef5350"},
        {"if": {"filter_query": "{Gap %} > 0", "column_id": "Gap %"}, "color": "#66bb6a"},
        {"if": {"filter_query": "{Gap %} < 0", "column_id": "Gap %"}, "color": "#ef5350"},
    ]
    return html.Div(
        style={"fontFamily": "Segoe UI, sans-serif", "padding": "14px", "backgroundColor": "#111", "color": "#eee"},
        children=[
            html.H2("Pre-Market Screener (Python)", style={"margin": "0 0 10px 0"}),
            html.Div(
                style={"display": "flex", "gap": "10px", "marginBottom": "12px", "flexWrap": "wrap", "alignItems": "flex-end"},
                children=[
                    labeled_control(
                        "Actions",
                        html.Div(
                            style={"display": "flex", "gap": "6px"},
                            children=[
                                html.Button("Refresh Snapshot", id="refresh", n_clicks=0),
                                html.Button("Run Stats", id="run_stats", n_clicks=0),
                            ],
                        ),
                        width="230px",
                    ),
                    labeled_control(
                        "Snapshot Mode",
                        dcc.Dropdown(
                            id="snapshot_mode",
                            options=[
                                {"label": "Live", "value": "live"},
                                {"label": "Replay", "value": "replay"},
                            ],
                            value="live",
                            clearable=False,
                            style={"color": "#111"},
                        ),
                        width="140px",
                    ),
                    labeled_control(
                        "Replay Date",
                        dcc.DatePickerSingle(
                            id="replay_date",
                            date=date.today().isoformat(),
                            display_format="YYYY-MM-DD",
                        ),
                        width="170px",
                    ),
                    labeled_control(
                        "Replay Universe",
                        dcc.Input(
                            id="replay_universe",
                            type="number",
                            value=500,
                            min=50,
                            max=5000,
                            step=50,
                        ),
                        width="130px",
                    ),
                    labeled_control(
                        "Stat Display",
                        dcc.Dropdown(
                            id="stat_display",
                            options=[
                                {"label": "% > 0", "value": "pct"},
                                {"label": "Avg Change", "value": "avg"},
                                {"label": "Median Change", "value": "med"},
                            ],
                            value="pct",
                            clearable=False,
                            style={"color": "#111"},
                        ),
                        width="160px",
                    ),
                    labeled_control(
                        "Top N",
                        dcc.Input(id="top_n", type="number", value=50, min=5, max=500, step=5),
                        width="90px",
                    ),
                    labeled_control(
                        "Min Notional",
                        dcc.Input(id="min_notional", type="number", value=100000, step=50000),
                        width="130px",
                    ),
                    labeled_control(
                        "Min Market Cap",
                        dcc.Input(id="min_market_cap", type="number", value=0, step=1000000),
                        width="130px",
                    ),
                    labeled_control(
                        "Min Instances",
                        dcc.Input(id="min_instances", type="number", value=0, min=0, max=10, step=1),
                        width="110px",
                    ),
                ],
            ),
            html.Div(id="status", style={"marginBottom": "8px", "color": "#9ecbff"}),
            html.Div(
                style={"display": "flex", "gap": "14px", "marginBottom": "8px", "fontSize": 13, "color": "#b0bec5"},
                children=[html.Div(id="current_time"), html.Div(id="current_session")],
            ),
            html.Details(
                style={"marginBottom": "12px"},
                children=[
                    html.Summary("How to Use"),
                    html.Div(
                        style={"paddingTop": "8px", "color": "#cfd8dc", "fontSize": 13},
                        children=[
                            html.Div("1) Click Refresh Snapshot to load movers."),
                            html.Div("2) Select a ticker from Gainers/Losers."),
                            html.Div("3) Click Run Stats for historical open behavior."),
                            html.Div("4) Click a historical row to render zoom + 3-day charts."),
                        ],
                    ),
                ],
            ),
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "12px"},
                children=[
                    html.Div(
                        children=[
                            html.H4("Gainers"),
                            dash_table.DataTable(
                                id="gainers_table",
                                page_size=12,
                                row_selectable="single",
                                sort_action="native",
                                style_as_list_view=True,
                                **tbl,
                            ),
                        ]
                    ),
                    html.Div(
                        children=[
                            html.H4("Losers"),
                            dash_table.DataTable(
                                id="losers_table",
                                page_size=12,
                                row_selectable="single",
                                sort_action="native",
                                style_as_list_view=True,
                                **tbl,
                            ),
                        ]
                    ),
                ],
            ),
            html.H4("Historical Instances", style={"marginTop": "12px"}),
            dash_table.DataTable(
                id="detail_table",
                page_size=10,
                row_selectable="single",
                sort_action="native",
                style_as_list_view=True,
                **detail_tbl,
            ),
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "12px", "marginTop": "12px"},
                children=[
                    dcc.Graph(id="zoom_chart", style={"height": "320px"}),
                    dcc.Graph(id="intraday_chart", style={"height": "320px"}),
                ],
            ),
            dcc.Store(id="snapshot_store"),
            dcc.Store(id="stats_store"),
            dcc.Store(id="details_store"),
            dcc.Store(id="selected_ticker_store"),
            dcc.Interval(id="clock_tick", interval=1000, n_intervals=0),
        ],
    )


def to_table(df: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    if df is None or df.empty:
        return [], []
    cols = [{"name": c, "id": c} for c in df.columns]
    safe = df.where(pd.notna(df), None)
    return safe.to_dict("records"), cols


def build_stats_view(df: pd.DataFrame, stat_display: str) -> pd.DataFrame:
    if df.empty:
        return df

    view_cols = ["ticker", "today_change_pct", "n_instances"]
    metric = "pct_positive" if stat_display == "pct" else "avg_change" if stat_display == "avg" else "median_change"
    for m in [1, 3, 5, 10, 15, 30, 60]:
        view_cols.append(f"{metric}_{m}m")

    use = [c for c in view_cols if c in df.columns]
    out = df[use].copy()
    for c in out.columns:
        if c != "ticker":
            out[c] = pd.to_numeric(out[c], errors="coerce").round(2)
    rename = {"n_instances": "n", "today_change_pct": "Today %"}
    for m in [1, 3, 5, 10, 15, 30, 60]:
        rename[f"{metric}_{m}m"] = f"{m}m"
    out = out.rename(columns=rename)
    return out


def fetch_stats_for_ticker(ticker: str, current_change_pct: float, direction: str) -> tuple[dict, pd.DataFrame]:
    end = date.today()
    start = end - timedelta(days=365 * 3)

    daily = get_daily_bars(ticker, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"))
    gaps = find_historical_gaps(daily, current_change_pct=current_change_pct, direction=direction)
    # Fallback: if strict threshold yields no rows, use most recent same-direction gaps
    # so interval table/chart still has usable instances.
    if gaps.empty and (not daily.empty) and len(daily) >= 2:
        d = daily.sort_values("date").copy()
        d["prev_close"] = d["c"].shift(1)
        d["gap_pct"] = (d["o"] - d["prev_close"]) / d["prev_close"] * 100.0
        d = d[d["gap_pct"].notna()]
        if direction == "gainer":
            d = d[d["gap_pct"] > 0]
        else:
            d = d[d["gap_pct"] < 0]
        d = d[d["date"] < date.today()].sort_values("date", ascending=False).head(10)
        if not d.empty:
            gaps = d[["ticker", "date", "gap_pct", "prev_close", "o"]].rename(columns={"o": "open_price"})

    if gaps.empty:
        return aggregate_interval_stats(pd.DataFrame(), ticker), pd.DataFrame()

    rows = []
    for g in gaps.itertuples(index=False):
        gd = g.date
        minute = get_minute_bars(ticker, gd.strftime("%Y-%m-%d"), gd.strftime("%Y-%m-%d"))
        intr = calculate_interval_changes(minute, gd)
        if not intr.empty:
            intr["ticker"] = ticker
            intr["gap_pct"] = float(g.gap_pct)
            rows.append(intr)
        else:
            # Keep the historical instance even if interval bars are missing,
            # so detail table still shows dates/instances like the R app.
            rows.append(
                pd.DataFrame(
                    [
                        {
                            "gap_date": gd,
                            "open_930": None,
                            "chg_1m": None,
                            "chg_3m": None,
                            "chg_5m": None,
                            "chg_10m": None,
                            "chg_15m": None,
                            "chg_30m": None,
                            "chg_60m": None,
                            "ticker": ticker,
                            "gap_pct": float(g.gap_pct),
                        }
                    ]
                )
            )

    details = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    stats = aggregate_interval_stats(details, ticker)
    return stats, details


app = dash.Dash(__name__)
app.layout = build_base_layout()


@app.callback(
    Output("snapshot_store", "data"),
    Output("status", "children"),
    Input("refresh", "n_clicks"),
    State("snapshot_mode", "value"),
    State("replay_date", "date"),
    State("replay_universe", "value"),
    State("top_n", "value"),
    State("min_notional", "value"),
    State("min_market_cap", "value"),
    prevent_initial_call=True,
)
def refresh_snapshot(
    _clicks: int,
    snapshot_mode: str,
    replay_date_val: str,
    replay_universe: int,
    top_n: int,
    min_notional: float,
    min_market_cap: float,
):
    t0 = time.time()
    try:
        mode = (snapshot_mode or "live").lower()
        replay_diag = {"universe": None, "rows": None}
        cap_filter_note = ""
        if mode == "replay":
            if not replay_date_val:
                return {"all": []}, "Replay mode requires a date"
            replay_date = pd.to_datetime(replay_date_val).date()
            if replay_date.weekday() >= 5:
                return {"all": []}, f"Replay date {replay_date} is a weekend. Choose a trading day."
            snap, replay_diag = get_replay_snapshot(replay_date, max_tickers=int(replay_universe or 500))
        else:
            snap = get_market_snapshot()

        if snap.empty:
            if mode == "replay":
                return {"all": []}, (
                    f"No replay rows found for {replay_date_val}. "
                    f"Universe={replay_diag.get('universe', 0)}. "
                    "Try a different date or larger Replay Universe."
                )
            return {"all": []}, "No snapshot data returned"

        for c in ["change_pct", "todays_change", "base_price", "open_price", "end_price", "volume"]:
            if c in snap.columns:
                snap[c] = pd.to_numeric(snap[c], errors="coerce")

        before_filter_count = len(snap)
        if mode == "replay" and "notional" in snap.columns:
            snap["notional"] = pd.to_numeric(snap["notional"], errors="coerce").fillna(0)
        else:
            snap["notional"] = pd.to_numeric(snap["volume"], errors="coerce").fillna(0) * pd.to_numeric(
                snap["end_price"], errors="coerce"
            ).fillna(0)

        notional_threshold = float(min_notional or 0)
        notional_hits = int((snap["notional"] >= notional_threshold).sum())
        snap = snap[snap["notional"] >= notional_threshold]

        top_n_int = int(top_n or 50)
        gainers = snap.sort_values("change_pct", ascending=False).head(top_n_int)
        losers = snap.sort_values("change_pct", ascending=True).head(top_n_int)

        # Mirror the R app behavior: use v3 unified snapshot for accurate
        # pre-market change/volume on the active candidate set.
        if mode == "live":
            candidate = (
                pd.concat(
                    [
                        snap.sort_values("change_pct", ascending=False).head(max(50, top_n_int * 3))["ticker"],
                        snap.sort_values("change_pct", ascending=True).head(max(50, top_n_int * 3))["ticker"],
                    ],
                    ignore_index=True,
                )
                .dropna()
                .drop_duplicates()
                .tolist()
            )
            if candidate:
                usnap = get_unified_snapshot(candidate)
                if not usnap.empty:
                    snap = snap.drop(columns=["base_price", "end_price", "change_pct", "volume"], errors="ignore").merge(
                        usnap, on="ticker", how="left", suffixes=("", "_u")
                    )
                    snap["base_price"] = pd.to_numeric(snap["base_price"], errors="coerce")
                    snap["end_price"] = pd.to_numeric(snap["end_price"], errors="coerce")
                    snap["change_pct"] = pd.to_numeric(snap["change_pct"], errors="coerce")
                    snap["volume"] = pd.to_numeric(snap["volume"], errors="coerce")
                    snap = snap[snap["base_price"].notna() & snap["end_price"].notna() & (snap["base_price"] > 0)]
                    snap["notional"] = snap["volume"].fillna(0) * snap["end_price"].fillna(0)
                    snap = snap[snap["notional"] >= float(min_notional or 0)]

        if "change_pct" in snap.columns:
            snap["change_pct"] = pd.to_numeric(snap["change_pct"], errors="coerce")
        snap = snap[snap["change_pct"].notna()].copy()

        gainers = snap.sort_values("change_pct", ascending=False).head(top_n_int)
        losers = snap.sort_values("change_pct", ascending=True).head(top_n_int)
        cap_tickers = snap["ticker"].dropna().astype(str).drop_duplicates().head(1500).tolist()
        caps = get_market_caps_bulk(cap_tickers)
        if "market_cap" not in caps.columns:
            caps["market_cap"] = pd.Series(dtype="float64")
        caps["market_cap"] = pd.to_numeric(caps["market_cap"], errors="coerce")
        merged = snap.merge(caps, on="ticker", how="left")
        if "market_cap" not in merged.columns:
            merged["market_cap"] = None

        if min_market_cap and float(min_market_cap) > 0:
            cap_series = pd.to_numeric(merged["market_cap"], errors="coerce")
            coverage = float(cap_series.notna().mean()) if len(cap_series) else 0.0
            if coverage < 0.10:
                cap_filter_note = " | Min Market Cap skipped (insufficient market_cap coverage)"
            else:
                merged = merged[cap_series.fillna(0) >= float(min_market_cap)]

        gainers = merged.sort_values("change_pct", ascending=False).head(top_n_int).copy()
        losers = merged.sort_values("change_pct", ascending=True).head(top_n_int).copy()

        for df in [gainers, losers]:
            if "market_cap" not in df.columns:
                df["market_cap"] = None
            df["market_cap_fmt"] = df["market_cap"].apply(format_money)
            df["notional_fmt"] = df["notional"].apply(format_money)

        elapsed = time.time() - t0
        merged = merged.where(pd.notna(merged), None)
        gainers = gainers.where(pd.notna(gainers), None)
        losers = losers.where(pd.notna(losers), None)
        mode_txt = "LIVE" if mode == "live" else f"REPLAY {replay_date_val}"
        if len(merged) == 0 and mode == "replay":
            return {"all": []}, (
                f"Replay built {before_filter_count} rows, {notional_hits} met Min Notional, but 0 remain after filters. "
                "Try lowering Min Notional/Min Market Cap."
            )
        return {
            "all": merged.to_dict("records"),
            "gainers": gainers.to_dict("records"),
            "losers": losers.to_dict("records"),
        }, (
            f"Snapshot loaded: {len(merged)} rows in {elapsed:.1f}s | "
            f"Mode: {mode_txt} | "
            f"{'ReplayRows=' + str(replay_diag.get('rows')) + ' Universe=' + str(replay_diag.get('universe')) + ' Cached=' + str(replay_diag.get('cached')) + ' | ' if mode == 'replay' else ''}"
            f"MinNotionalHits={notional_hits} | "
            f"Session: {'LIVE' if is_premarket_session() else 'Closed'}"
            f"{cap_filter_note}"
        )
    except Exception as exc:
        log_msg(f"Refresh error: {exc}\n{traceback.format_exc()}")
        return {"all": []}, f"Refresh failed: {exc}"


@app.callback(
    Output("gainers_table", "data"),
    Output("gainers_table", "columns"),
    Output("losers_table", "data"),
    Output("losers_table", "columns"),
    Input("snapshot_store", "data"),
)
def render_snapshot_tables(snapshot_data):
    if not snapshot_data:
        return [], [], [], []

    def pick_cols(df: pd.DataFrame) -> pd.DataFrame:
        want = [
            "ticker",
            "end_price",
            "change_pct",
            "todays_change",
            "volume",
            "market_cap_fmt",
        ]
        use = [c for c in want if c in df.columns]
        out = df[use].copy()
        for c in ["end_price", "change_pct", "todays_change"]:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce").round(2)
        return out

    g = pick_cols(pd.DataFrame(snapshot_data.get("gainers", [])))
    l = pick_cols(pd.DataFrame(snapshot_data.get("losers", [])))
    g_data, g_cols = to_table(g)
    l_data, l_cols = to_table(l)
    return g_data, g_cols, l_data, l_cols


@app.callback(
    Output("selected_ticker_store", "data"),
    Input("gainers_table", "selected_rows"),
    Input("losers_table", "selected_rows"),
    State("gainers_table", "data"),
    State("losers_table", "data"),
    State("selected_ticker_store", "data"),
)
def set_selected_ticker(g_sel, l_sel, g_data, l_data, current_ticker):
    if g_sel and g_data:
        idx = g_sel[0]
        if idx < len(g_data):
            return g_data[idx].get("ticker")
    if l_sel and l_data:
        idx = l_sel[0]
        if idx < len(l_data):
            return l_data[idx].get("ticker")
    if current_ticker:
        return current_ticker
    return no_update


@app.callback(
    Output("stats_store", "data"),
    Output("details_store", "data"),
    Output("selected_ticker_store", "data", allow_duplicate=True),
    Output("status", "children", allow_duplicate=True),
    Input("run_stats", "n_clicks"),
    Input("snapshot_store", "data"),
    State("snapshot_store", "data"),
    State("min_instances", "value"),
    prevent_initial_call=True,
)
def run_stats(_clicks: int, _snapshot_trigger, snapshot_data, min_instances: int):
    if not snapshot_data:
        return {"gainers": [], "losers": []}, {}, None, "Load snapshot first"

    gainers = pd.DataFrame(snapshot_data.get("gainers", []))
    losers = pd.DataFrame(snapshot_data.get("losers", []))
    g_chg = {
        str(r["ticker"]): pd.to_numeric(r.get("change_pct"), errors="coerce")
        for _, r in gainers.iterrows()
        if pd.notna(r.get("ticker"))
    }
    l_chg = {
        str(r["ticker"]): pd.to_numeric(r.get("change_pct"), errors="coerce")
        for _, r in losers.iterrows()
        if pd.notna(r.get("ticker"))
    }

    tasks = []
    details_map: dict[str, list[dict]] = {}
    stats_g, stats_l = [], []
    g_total = 0
    l_total = 0
    min_inst = int(min_instances or 0)

    with ThreadPoolExecutor(max_workers=3) as ex:
        for row in gainers.itertuples(index=False):
            if pd.isna(getattr(row, "change_pct", None)):
                continue
            g_total += 1
            fut = ex.submit(fetch_stats_for_ticker, row.ticker, float(row.change_pct), "gainer")
            tasks.append((fut, "gainer"))

        for row in losers.itertuples(index=False):
            if pd.isna(getattr(row, "change_pct", None)):
                continue
            l_total += 1
            fut = ex.submit(fetch_stats_for_ticker, row.ticker, float(row.change_pct), "loser")
            tasks.append((fut, "loser"))

        for fut, side in tasks:
            try:
                stats, details = fut.result()
                if int(stats.get("n_instances", 0)) < min_inst:
                    continue
                if side == "gainer":
                    stats["today_change_pct"] = None if pd.isna(g_chg.get(stats.get("ticker"))) else float(g_chg.get(stats.get("ticker")))
                    stats_g.append(stats)
                else:
                    stats["today_change_pct"] = None if pd.isna(l_chg.get(stats.get("ticker"))) else float(l_chg.get(stats.get("ticker")))
                    stats_l.append(stats)
                if isinstance(details, pd.DataFrame) and not details.empty:
                    if "gap_date" in details.columns:
                        details["gap_date"] = details["gap_date"].astype(str)
                    details_map[stats["ticker"]] = details.to_dict("records")
            except Exception as exc:
                log_msg(f"Stats task error: {exc}")

    default_ticker = None
    if stats_g:
        default_ticker = stats_g[0].get("ticker")
    elif stats_l:
        default_ticker = stats_l[0].get("ticker")
    elif details_map:
        default_ticker = next(iter(details_map.keys()))
    if details_map and (default_ticker not in details_map):
        default_ticker = next(iter(details_map.keys()))

    return (
        {"gainers": stats_g, "losers": stats_l},
        details_map,
        default_ticker,
        (
            f"Stats complete: {len(stats_g)} gainers, {len(stats_l)} losers | "
            f"candidates G={g_total} L={l_total} | min_instances={min_inst} | "
            f"detail_tickers={len(details_map)} | selected={default_ticker}"
        ),
    )


@app.callback(
    Output("gainers_table", "data", allow_duplicate=True),
    Output("gainers_table", "columns", allow_duplicate=True),
    Output("losers_table", "data", allow_duplicate=True),
    Output("losers_table", "columns", allow_duplicate=True),
    Input("stats_store", "data"),
    Input("stat_display", "value"),
    prevent_initial_call=True,
)
def render_stats_tables(stats_data, stat_display: str):
    if not stats_data:
        return [], [], [], []

    g = build_stats_view(pd.DataFrame(stats_data.get("gainers", [])), stat_display)
    l = build_stats_view(pd.DataFrame(stats_data.get("losers", [])), stat_display)

    g_data, g_cols = to_table(g)
    l_data, l_cols = to_table(l)
    return g_data, g_cols, l_data, l_cols


@app.callback(
    Output("detail_table", "data"),
    Output("detail_table", "columns"),
    Input("selected_ticker_store", "data"),
    Input("details_store", "data"),
)
def render_detail_table(selected_ticker: str, details_store):
    if not details_store:
        return [], []
    if (not selected_ticker) or (selected_ticker not in details_store):
        selected_ticker = next(iter(details_store.keys()), None)
        if not selected_ticker:
            return [], []

    rows = details_store.get(selected_ticker, [])
    df = pd.DataFrame(rows)
    if df.empty:
        return [], []

    cols = [c for c in ["gap_date", "gap_pct", "chg_1m", "chg_3m", "chg_5m", "chg_10m", "chg_15m", "chg_30m", "chg_60m"] if c in df.columns]
    out = df[cols].copy()
    for c in cols:
        if c != "gap_date":
            out[c] = pd.to_numeric(out[c], errors="coerce").round(2)

    out = out.rename(
        columns={
            "gap_date": "Date",
            "gap_pct": "Gap %",
            "chg_1m": "1m",
            "chg_3m": "3m",
            "chg_5m": "5m",
            "chg_10m": "10m",
            "chg_15m": "15m",
            "chg_30m": "30m",
            "chg_60m": "60m",
        }
    )
    cols = [{"name": c, "id": c} for c in out.columns]
    records = out.assign(ticker=selected_ticker).where(pd.notna(out.assign(ticker=selected_ticker)), None).to_dict("records")
    return records, cols


@app.callback(
    Output("detail_table", "selected_rows"),
    Input("detail_table", "data"),
)
def default_select_detail_row(rows):
    if rows and len(rows) > 0:
        return [0]
    return []


@app.callback(
    Output("current_time", "children"),
    Output("current_session", "children"),
    Input("clock_tick", "n_intervals"),
)
def update_clock(_n):
    now = datetime.now(ET)
    t = now.strftime("Time: %H:%M:%S ET")
    s = "Session: Pre-Market (LIVE)" if is_premarket_session(now) else "Session: Pre-Market Closed"
    return t, s


@app.callback(
    Output("zoom_chart", "figure"),
    Output("intraday_chart", "figure"),
    Input("detail_table", "selected_rows"),
    State("detail_table", "data"),
    State("selected_ticker_store", "data"),
)
def render_charts(selected_rows, detail_rows, ticker):
    blank = go.Figure().update_layout(template="plotly_dark", margin=dict(l=10, r=10, t=30, b=10))
    try:
        if not detail_rows or not ticker:
            return blank, blank

        idx = 0 if not selected_rows else selected_rows[0]
        if idx >= len(detail_rows):
            return blank, blank

        gd_val = detail_rows[idx].get("gap_date") or detail_rows[idx].get("Date")
        if not gd_val:
            return blank, blank

        gd = pd.to_datetime(gd_val, errors="coerce")
        if pd.isna(gd):
            return blank.update_layout(title="No chart: invalid gap date"), blank.update_layout(title="No chart: invalid gap date")
        gd = gd.date()
        from_d, to_d = calculate_3day_chart_window(gd)
        minute = get_minute_bars(ticker, from_d, to_d)
        if minute.empty:
            return (
                blank.update_layout(title=f"No minute bars for {ticker} ({from_d} to {to_d})"),
                blank.update_layout(title=f"No minute bars for {ticker} ({from_d} to {to_d})"),
            )

        zoom = prepare_zoom_chart_data(minute, gd)
        chart3 = prepare_chart_data(minute, gd)

        zfig = go.Figure()
        if not zoom.empty:
            zoom = zoom.copy()
            zoom["bar_index"] = pd.to_numeric(zoom["bar_index"], errors="coerce")
            zoom["c"] = pd.to_numeric(zoom["c"], errors="coerce")
            zoom = zoom[zoom["bar_index"].notna() & zoom["c"].notna()]

            tick = (
                zoom[zoom["time_hm"].isin(["09:00", "09:10", "09:20", "09:30", "09:40", "09:50", "10:00"])]
                .groupby("time_hm", as_index=False)["bar_index"]
                .min()
            )
            if not zoom.empty:
                zfig.add_trace(go.Scatter(
                    x=zoom["bar_index"],
                    y=zoom["c"],
                    mode="lines",
                    name=ticker,
                    line={"color": "#00bfff", "width": 1.8},
                    text=[
                        f"{t}<br>Price: ${p:.2f}<br>{s}"
                        for t, p, s in zip(zoom["time_label"], zoom["c"], zoom["session"])
                    ],
                    hovertemplate="%{text}<extra></extra>",
                ))
                zfig.update_layout(
                    template="plotly_dark",
                    title=f"{ticker} Open Zoom 9:00-10:00 ({gd})",
                    margin=dict(l=10, r=10, t=40, b=20),
                    xaxis={
                        "title": "",
                        "tickmode": "array",
                        "tickvals": tick["bar_index"].tolist(),
                        "ticktext": tick["time_hm"].tolist(),
                        "gridcolor": "#3d3d3d",
                    },
                    yaxis={"title": "", "tickprefix": "$", "gridcolor": "#3d3d3d"},
                    shapes=build_zoom_shapes(zoom),
                    showlegend=False,
                    hovermode="closest",
                )

        cfig = go.Figure()
        if not chart3.empty:
            chart3 = chart3.copy()
            chart3["bar_index"] = pd.to_numeric(chart3["bar_index"], errors="coerce")
            chart3["c"] = pd.to_numeric(chart3["c"], errors="coerce")
            chart3 = chart3[chart3["bar_index"].notna() & chart3["c"].notna()]

            for session, sdf in chart3.groupby(["date", "session"], sort=False):
                s = sdf.sort_values("bar_index")
                if s.empty:
                    continue
                is_gap = bool(s["is_gap_day"].iloc[0])
                sess_name = session[1]
                is_rth = sess_name == "Regular"

                if is_gap and is_rth:
                    lc, lw, op = "#00bfff", 1.9, 1.0
                elif is_gap and not is_rth:
                    lc, lw, op = "rgba(0,191,255,0.55)", 1.3, 1.0
                elif (not is_gap) and is_rth:
                    lc, lw, op = "rgba(0,191,255,0.40)", 1.2, 1.0
                else:
                    lc, lw, op = "rgba(0,191,255,0.20)", 0.9, 1.0

                cfig.add_trace(
                    go.Scatter(
                        x=s["bar_index"],
                        y=s["c"],
                        mode="lines",
                        name=f"{session[0]} {session[1]}",
                        line=dict(width=lw, color=lc),
                        opacity=op,
                        text=[
                            f"{dt.strftime('%b %d %H:%M')}<br>Price: ${p:.2f}<br>{sess_name}{' (T)' if is_gap else ''}"
                            for dt, p in zip(s["datetime_et"], s["c"])
                        ],
                        hovertemplate="%{text}<extra></extra>",
                        showlegend=False,
                    )
                )
            ticks = (
                chart3[chart3["time_hm"].isin(["04:00", "09:30", "16:00"])]
                .groupby(["date", "time_hm"], as_index=False)["bar_index"]
                .min()
            )
            ticks["label"] = pd.to_datetime(ticks["date"].astype(str)).dt.strftime("%b %d") + "\n" + ticks["time_hm"]
            cfig.update_layout(
                template="plotly_dark",
                title=f"{ticker} 3-Day Intraday ({gd})",
                margin=dict(l=10, r=10, t=40, b=30),
                xaxis={
                    "title": "",
                    "tickmode": "array",
                    "tickvals": ticks["bar_index"].tolist(),
                    "ticktext": ticks["label"].tolist(),
                    "gridcolor": "#3d3d3d",
                    "tickfont": {"size": 10},
                },
                yaxis={"title": "", "tickprefix": "$", "gridcolor": "#3d3d3d"},
                shapes=build_session_shapes(chart3, gap_date=gd),
                showlegend=False,
                hovermode="closest",
            )

        if len(zfig.data) == 0:
            zfig.update_layout(title=f"No zoom data for {ticker} ({gd})")
        if len(cfig.data) == 0:
            cfig.update_layout(title=f"No 3-day data for {ticker} ({gd})")
        return zfig, cfig
    except Exception as exc:
        log_msg(f"render_charts error: {exc}\n{traceback.format_exc()}")
        return (
            blank.update_layout(title=f"Chart error: {exc}"),
            blank.update_layout(title=f"Chart error: {exc}"),
        )


if __name__ == "__main__":
    log_msg("Starting Python premarket screener")
    app.run(host="127.0.0.1", port=8050, debug=True)
