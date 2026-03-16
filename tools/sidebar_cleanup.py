"""
app18.py sidebar & nav cleanup script.
Run:  python tools/sidebar_cleanup.py
"""
import re, subprocess, sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent.parent
path = ROOT / "app18.py"

with open(path, encoding="utf-8") as f:
    text = f.read()

orig_lines = len(text.splitlines())
changes = []

# ─── CHANGE 1: Nav radio – collapsed label + readable line breaks ─────────────
OLD1 = (
    'nav = st.radio(\n'
    '    "Navigation",\n'
    '    ["Chart", "Options", "TradingView", "GPT-5 Agent", "MS Technical Analysis",'
    ' "JPM Earnings", "GS Fundamental", "Scans", "Premarket", "Calendar", "Scanners",'
    ' "Movers", "Overnight", "Signal Scanner"],\n'
    '    horizontal=True,\n'
    '    index=0,\n'
    '    key="top_nav",\n'
    ')'
)
NEW1 = (
    'nav = st.radio(\n'
    '    "Navigation",\n'
    '    [\n'
    '        "Chart", "Options", "TradingView", "GPT-5 Agent",\n'
    '        "MS Analysis", "JPM Earnings", "GS Fundamental",\n'
    '        "Scans", "Premarket", "Calendar",\n'
    '        "Scanners", "Movers", "Overnight", "Signal Scanner",\n'
    '    ],\n'
    '    horizontal=True,\n'
    '    index=0,\n'
    '    key="top_nav",\n'
    '    label_visibility="collapsed",\n'
    ')\n'
    'st.markdown("---")'
)
if OLD1 in text:
    text = text.replace(OLD1, NEW1)
    changes.append("1 ✓ nav radio cleaned up")
else:
    changes.append("1 ⚠ nav radio – not found, skipped")

# ─── CHANGE 2: Rename MS Technical Analysis → MS Analysis everywhere ──────────
n = text.count("'MS Technical Analysis'")
text = text.replace("'MS Technical Analysis'", "'MS Analysis'")
changes.append(f"2 ✓ MS Analysis rename ({n} occurrences)")

# ─── CHANGE 3: Remove duplicate scan banner inside Scans section ──────────────
OLD3 = (
    "    _show_scan_banner()\n"
    "    try:\n"
    "        banner = st.session_state.get(\"last_scan_stats_caption\")\n"
    "        if banner:\n"
    "            st.info(f\"Last scan stats: {banner}\")\n"
    "    except Exception:\n"
    "        pass"
)
if OLD3 in text:
    text = text.replace(OLD3, "    _show_scan_banner()")
    changes.append("3 ✓ removed duplicate scan banner")
else:
    changes.append("3 ⚠ duplicate scan banner – not found, skipped")

# ─── CHANGE 4: Insert chart-sidebar defaults block before sidebar block 2 ─────
DEFAULTS = """\
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
"""
MARKER4 = "# ---------------- Chart controls ----------------\n"
if MARKER4 in text and DEFAULTS not in text:
    text = text.replace(MARKER4, DEFAULTS + MARKER4)
    changes.append("4 ✓ chart sidebar defaults block inserted")
else:
    changes.append("4 ⚠ defaults block already present or marker not found")

# ─── CHANGE 5: Replace sidebar block 2 (all chart-specific controls) ──────────
START5 = "# ---------------- Chart controls ----------------\nwith st.sidebar:\n    with st.expander(\"Overlays\", expanded=False):"
END5   = "\n# ---------------- Helpers (chart) ----------------\nwith st.sidebar:"

idx_s = text.find(START5)
idx_e = text.find(END5)

if idx_s != -1 and idx_e != -1 and idx_e > idx_s:
    NEW5 = """\
# ---------------- Chart sidebar (nav-conditional) ----------------------------
with st.sidebar:
    # ── Always visible: Data provider & API keys ─────────────────────────────
    st.markdown("### Data")
    use_polygon = st.checkbox("Use Polygon (env key)", value=True)
    st.session_state['use_polygon'] = use_polygon
    try:
        _poly_env = (os.getenv('POLYGON_API_KEY') or '').strip()
        st.session_state['polygon_api_key'] = _poly_env
    except Exception:
        pass

    try:
        pref = st.selectbox(
            "Preferred provider",
            ["Auto (best available)", "Polygon", "Alpaca", "Tiingo"],
            index=0,
            help="Try this provider first when fetching chart data.",
            key="preferred_provider_sel",
        )
        st.session_state['preferred_provider'] = pref
    except Exception:
        pass

    force_fresh = st.checkbox("Force fresh fetch", value=False)
    try:
        st.session_state['force_refresh'] = force_fresh
    except Exception:
        pass

    with st.expander("API Keys", expanded=False):
        alp_key_id  = st.text_input("Alpaca Key ID", value="", type="password")
        alp_secret  = st.text_input("Alpaca Secret", value="", type="password")
        alp_data_url = st.text_input(
            "Alpaca Data URL",
            value=os.getenv('APCA_API_DATA_URL') or 'https://data.alpaca.markets',
        )
        if alp_key_id:   st.session_state['alpaca_key_id']    = alp_key_id.strip()
        if alp_secret:   st.session_state['alpaca_secret_key'] = alp_secret.strip()
        if alp_data_url: st.session_state['alpaca_data_url']   = alp_data_url.strip()
        tradier_token = st.text_input(
            "Tradier Token", value=os.getenv('TRADIER_TOKEN') or "", type="password"
        )
        tradier_sandbox = st.checkbox(
            "Tradier Sandbox",
            value=bool(
                os.getenv('TRADIER_SANDBOX', "").strip()
                not in ("", "0", "false", "False")
            ),
        )
        if tradier_token: st.session_state['tradier_token'] = tradier_token.strip()
        st.session_state['tradier_sandbox'] = bool(tradier_sandbox)

    with st.expander("Diagnostics", expanded=False):
        try:
            _pk  = st.session_state.get('polygon_api_key') or os.getenv('POLYGON_API_KEY') or ''
            _ak  = (st.session_state.get('alpaca_key_id')
                    or os.getenv('ALPACA_API_KEY_ID') or os.getenv('APCA_API_KEY_ID') or '')
            _src = st.session_state.get('last_fetch_provider', '-')
            st.caption(
                f"Polygon: {'✓' if _pk else '✗'}  "
                f"Alpaca: {'✓' if _ak else '✗'}  "
                f"Last source: {_src}"
            )
            if ENV_DIAG_CAPTION:
                st.caption(ENV_DIAG_CAPTION)
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
                    pass"""
    text = text[:idx_s] + NEW5 + text[idx_e:]
    changes.append("5 ✓ sidebar block 2 fully restructured")
else:
    changes.append(f"5 ⚠ sidebar block 2 not found (start={idx_s}, end={idx_e})")

# ─── CHANGE 6: Sidebar block 3 (backtesting + appearance) ─────────────────────
OLD6 = (
    "# ---------------- Helpers (chart) ----------------\n"
    "with st.sidebar:\n"
    "    st.markdown(\"### Backtesting\")\n"
    "    enable_backtest = st.checkbox(\"Enable Backtest\", value=False)\n"
    "    strategy = None\n"
    "    if enable_backtest:\n"
    "        strategy = st.selectbox(\n"
    "            \"Strategy\",\n"
    "            [\n"
    "                \"Price crosses above VWAP\",\n"
    "                \"Price crosses below VWAP\",\n"
    "                \"RSI crosses above 70\",\n"
    "                \"RSI crosses below 30\",\n"
    "                \"MACD crosses above Signal\",\n"
    "                \"MACD crosses below Signal\"\n"
    "            ],\n"
    "            index=0\n"
    "        )\n"
    "\n"
    "    st.markdown(\"### Appearance\")\n"
    "    theme = st.selectbox(\"Theme\", [\"System\", \"Dark\", \"Light\"], index=0)"
)
NEW6 = (
    "with st.sidebar:\n"
    "    if nav == 'Chart':\n"
    "        with st.expander(\"Backtesting\", expanded=False):\n"
    "            enable_backtest = st.checkbox(\"Enable Backtest\", value=False)\n"
    "            strategy = None\n"
    "            if enable_backtest:\n"
    "                strategy = st.selectbox(\n"
    "                    \"Strategy\",\n"
    "                    [\n"
    "                        \"Price crosses above VWAP\",\n"
    "                        \"Price crosses below VWAP\",\n"
    "                        \"RSI crosses above 70\",\n"
    "                        \"RSI crosses below 30\",\n"
    "                        \"MACD crosses above Signal\",\n"
    "                        \"MACD crosses below Signal\",\n"
    "                    ],\n"
    "                    index=0,\n"
    "                )\n"
    "    else:\n"
    "        enable_backtest = False\n"
    "        strategy = None\n"
    "\n"
    "    st.markdown(\"### Appearance\")\n"
    "    theme = st.selectbox(\"Theme\", [\"System\", \"Dark\", \"Light\"], index=0)"
)
if OLD6 in text:
    text = text.replace(OLD6, NEW6)
    changes.append("6 ✓ sidebar block 3 restructured")
else:
    changes.append("6 ⚠ sidebar block 3 not found exactly")

# ─── CHANGE 7: Collapse 3+ blank lines → max 2 ────────────────────────────────
before = len(text.splitlines())
text = re.sub(r'\n{4,}', '\n\n\n', text)
after = len(text.splitlines())
changes.append(f"7 ✓ collapsed blank lines ({before - after} lines removed)")

# ─── CHANGE 8: Remove orphan duplicate section header ─────────────────────────
text = text.replace(
    "# ---------------- Helpers (chart) ----------------\nTARGETS",
    "# ── Chart helpers ────────────────────────────────────────────────────────\nTARGETS",
)
changes.append("8 ✓ cleaned orphan section header")

# ─── WRITE & VERIFY ───────────────────────────────────────────────────────────
with open(path, "w", encoding="utf-8") as f:
    f.write(text)

print(f"Original lines : {orig_lines}")
print(f"Final lines    : {len(text.splitlines())}")
print("\nChanges applied:")
for c in changes:
    print(f"  {c}")

result = subprocess.run(
    [sys.executable, "-m", "py_compile", str(path)],
    capture_output=True, text=True,
)
if result.returncode == 0:
    print("\n✓ SYNTAX OK — py_compile passed")
else:
    print("\n✗ SYNTAX ERROR:\n" + result.stderr)
