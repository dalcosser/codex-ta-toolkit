import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from ..indicators.ta import bbands, vwap, macd, sma, ema, rsi, stoch_kd

def plot_candles_with_signal(
    df: pd.DataFrame,
    signal: pd.Series,
    ticker: str = "TICKER",
    overlays: dict | None = None,
    indicators: dict | None = None,
    template: str = "plotly_dark",
):
    """
    overlays keys:
      - "bbands": bool
      - "vwap": bool
      - "macd": bool   (overlay lines on price)
      - "sma_periods": list[int]
      - "ema_periods": list[int]

    indicators keys:
      - "rsi": {"show": bool, "length": int}
      - "stoch": {"show": bool, "k_len": int, "d_len": int, "smooth_k": int}
    """
    overlays = overlays or {}
    indicators = indicators or {}

    show_rsi = indicators.get("rsi", {}).get("show", False)
    rsi_len  = int(indicators.get("rsi", {}).get("length", 14))

    show_sto = indicators.get("stoch", {}).get("show", False)
    k_len    = int(indicators.get("stoch", {}).get("k_len", 14))
    d_len    = int(indicators.get("stoch", {}).get("d_len", 3))
    smooth_k = int(indicators.get("stoch", {}).get("smooth_k", 3))

    # rows: Price + optional RSI + optional Stoch (separate panes)
    rows = 1 + (1 if show_rsi else 0) + (1 if show_sto else 0)
    specs = [[{"secondary_y": False}] for _ in range(rows)]
    heights = [0.58] + ([0.21] if show_rsi else []) + ([0.21] if show_sto else [])

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        specs=specs,
        row_heights=heights if heights else None,
    )

    data = df.copy()
    for c in ["Open", "High", "Low", "Close"]:
        if c in data.columns:
            data[c] = pd.to_numeric(data[c], errors="coerce")

    price_row = 1
    next_row = 2
    rsi_row = next_row if show_rsi else None
    stoch_row = (next_row + (1 if show_rsi else 0)) if show_sto else None

    # Price panel
    fig.add_trace(
        go.Candlestick(
            x=data.index, open=data.get("Open"), high=data.get("High"),
            low=data.get("Low"), close=data.get("Close"), name="OHLC"
        ),
        row=price_row, col=1
    )

    # Buy/Sell markers
    sig = signal.reindex(data.index).fillna(0)
    buys = data[sig > 0]
    sells = data[sig < 0]
    if not buys.empty and "Low" in buys.columns:
        fig.add_trace(
            go.Scatter(
                x=buys.index, y=buys["Low"] * 0.995, mode="markers",
                marker_symbol="triangle-up", marker_size=9, name="Buy"
            ),
            row=price_row, col=1
        )
    if not sells.empty and "High" in sells.columns:
        fig.add_trace(
            go.Scatter(
                x=sells.index, y=sells["High"] * 1.005, mode="markers",
                marker_symbol="triangle-down", marker_size=9, name="Sell"
            ),
            row=price_row, col=1
        )

    # SMAs / EMAs overlays
    for length in overlays.get("sma_periods", []):
        try:
            s = sma(data["Close"], int(length))
            fig.add_trace(go.Scatter(x=data.index, y=s, mode="lines", name=s.name), row=price_row, col=1)
        except Exception:
            pass
    for length in overlays.get("ema_periods", []):
        try:
            e = ema(data["Close"], int(length))
            fig.add_trace(go.Scatter(x=data.index, y=e, mode="lines", name=e.name), row=price_row, col=1)
        except Exception:
            pass

    # BBands / VWAP / MACD overlays
    if overlays.get("bbands"):
        l, m, u = bbands(data["Close"])
        fig.add_trace(go.Scatter(x=data.index, y=l, mode="lines", name=l.name), row=price_row, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=m, mode="lines", name=m.name), row=price_row, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=u, mode="lines", name=u.name), row=price_row, col=1)

    if overlays.get("vwap") and all(k in data.columns for k in ["High","Low","Close","Volume"]):
        vw = vwap(data["High"], data["Low"], data["Close"], data["Volume"]).rename("VWAP")
        fig.add_trace(go.Scatter(x=data.index, y=vw, mode="lines", name=vw.name), row=price_row, col=1)

    if overlays.get("macd"):
        macd_line, macd_sig, _ = macd(data["Close"])
        fig.add_trace(go.Scatter(x=data.index, y=macd_line, mode="lines", name=macd_line.name), row=price_row, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=macd_sig,  mode="lines", name=macd_sig.name),  row=price_row, col=1)

    # RSI pane
    if rsi_row is not None:
        r = rsi(data["Close"], rsi_len).rename(f"RSI({rsi_len})")
        fig.add_trace(go.Scatter(x=data.index, y=r, mode="lines", name=r.name), row=rsi_row, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="#555", row=rsi_row, col=1)
        fig.add_hline(y=70, line_dash="dot", line_color="#555", row=rsi_row, col=1)
        fig.update_yaxes(range=[0, 100], row=rsi_row, col=1, title_text="RSI")

    # Stochastic pane
    if stoch_row is not None:
        k, d = stoch_kd(data["High"], data["Low"], data["Close"], k_len, d_len, smooth_k)
        fig.add_trace(go.Scatter(x=data.index, y=k, mode="lines", name=k.name), row=stoch_row, col=1)
        fig.add_trace(go.Scatter(x=data.index, y=d, mode="lines", name=d.name), row=stoch_row, col=1)
        fig.add_hline(y=20, line_dash="dot", line_color="#555", row=stoch_row, col=1)
        fig.add_hline(y=80, line_dash="dot", line_color="#555", row=stoch_row, col=1)
        fig.update_yaxes(range=[0, 100], row=stoch_row, col=1, title_text="Stoch")
    # --- Final layout styling ---
    dark_mode = template == "plotly_dark"

    fig.update_layout(
        title=f"{ticker} - Price, Signals & Indicators",
        xaxis_rangeslider_visible=False,
        template=template,
        height=950 if rows == 3 else (880 if rows == 2 else 760),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            font=dict(color="#ffffff" if dark_mode else "#111111")
        ),
        plot_bgcolor="#000000" if dark_mode else "#ffffff",
        paper_bgcolor="#000000" if dark_mode else "#ffffff",
        font=dict(color="#e6e6e6" if dark_mode else "#111111"),
    )

    grid_color = "#333333" if dark_mode else "#cccccc"
    fig.update_xaxes(
        showgrid=True,
        gridcolor=grid_color,
        zerolinecolor=grid_color,
        tickfont=dict(color="#e6e6e6" if dark_mode else "#111111"),
        title_font=dict(color="#e6e6e6" if dark_mode else "#111111"),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=grid_color,
        zerolinecolor=grid_color,
        tickfont=dict(color="#e6e6e6" if dark_mode else "#111111"),
        title_font=dict(color="#e6e6e6" if dark_mode else "#111111"),
    )

    # Tooltips readable in dark
    fig.update_traces(hoverlabel=dict(
        bgcolor="#111111" if dark_mode else "#ffffff",
        font_color="#e6e6e6" if dark_mode else "#111111",
        bordercolor="#444444" if dark_mode else "#cccccc"
    ), selector=dict(type="scatter"))
