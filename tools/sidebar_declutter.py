"""
Two changes:
  1. Collapse the exposed DATA widgets into a single "⚙ Settings" expander
  2. Remove the DoltHub rank_score expander from the Chart page
Run: python tools/sidebar_declutter.py
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

# ─── CHANGE 1: Collapse exposed DATA widgets + API Keys + Diagnostics
#               into one "⚙ Settings" expander ────────────────────────────────
OLD_DATA = """\
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
                f"Polygon: {'\\u2713' if _pk else '\\u2717'}  "
                f"Alpaca: {'\\u2713' if _ak else '\\u2717'}  "
                f"Last source: {_src}"
            )
            if ENV_DIAG_CAPTION:
                st.caption(ENV_DIAG_CAPTION)
        except Exception:
            pass"""

NEW_DATA = """\
    # ── Settings (collapsed by default — keeps sidebar clean) ────────────────
    with st.expander("⚙ Settings", expanded=False):
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
                ["Auto (best available)", "Polygon", "Alpaca", "Tiingo"],
                index=0,
                help="Try this provider first when fetching chart data.",
                key="preferred_provider_sel",
            )
            st.session_state['preferred_provider'] = pref
        except Exception:
            pass

        st.divider()
        st.caption("API Keys")
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

        st.divider()
        st.caption("Diagnostics")
        try:
            _pk  = st.session_state.get('polygon_api_key') or os.getenv('POLYGON_API_KEY') or ''
            _ak  = (st.session_state.get('alpaca_key_id')
                    or os.getenv('ALPACA_API_KEY_ID') or os.getenv('APCA_API_KEY_ID') or '')
            _src = st.session_state.get('last_fetch_provider', '-')
            st.caption(
                f"Polygon: {'\\u2713' if _pk else '\\u2717'}  "
                f"Alpaca: {'\\u2713' if _ak else '\\u2717'}  "
                f"Last source: {_src}"
            )
            if ENV_DIAG_CAPTION:
                st.caption(ENV_DIAG_CAPTION)
        except Exception:
            pass"""

if OLD_DATA in text:
    text = text.replace(OLD_DATA, NEW_DATA)
    changes.append("1 ✓ Sidebar DATA section collapsed into ⚙ Settings expander")
else:
    changes.append("1 ✗ OLD_DATA block not matched — check whitespace")

# ─── CHANGE 2: Remove the DoltHub rank_score expander from Chart page ─────────
# Find the block: with st.expander("DoltHub rank_score", expanded=False): ...
# It ends before the next top-level try: block
OLD_DOLTHUB = """\
    with st.expander("DoltHub rank_score", expanded=False):
        default_t = (
            st.session_state.get("tkr_vis")
            or st.session_state.get("ticker")
            or "AAPL"
        )
        tkr_rank = st.text_input("Ticker for rank_score", value=default_t, key="rank_score_ticker").strip().upper()
        if tkr_rank:
            rs_df = _fetch_dolthub_rank_score(tkr_rank)
            if rs_df is None or rs_df.empty:
                st.caption("No DoltHub rank_score rows for that ticker.")
            else:
                st.dataframe(rs_df, use_container_width=True)"""

if OLD_DOLTHUB in text:
    text = text.replace(OLD_DOLTHUB, "")
    changes.append("2 ✓ DoltHub rank_score expander removed from Chart page")
else:
    changes.append("2 ✗ DoltHub block not matched — check exact text")

# ─── CHANGE 3: Tighten the APPEARANCE section label ──────────────────────────
# Remove "### Appearance" markdown header — Theme label on the widget is enough
OLD_APP = '    st.markdown("### Appearance")\n    theme = st.selectbox("Theme"'
NEW_APP = '    theme = st.selectbox("Theme"'
if OLD_APP in text:
    text = text.replace(OLD_APP, NEW_APP)
    changes.append("3 ✓ Removed redundant ### Appearance header")
else:
    changes.append("3 ⚠ Appearance header not found (may already be gone)")

# ─── CHANGE 4: Clean blank lines ─────────────────────────────────────────────
before = len(text.splitlines())
text = re.sub(r'\n{4,}', '\n\n\n', text)
changes.append(f"4 ✓ Collapsed blank lines ({before - len(text.splitlines())} removed)")

# ─── Write & verify ───────────────────────────────────────────────────────────
with open(path, "w", encoding="utf-8") as f:
    f.write(text)

print(f"Lines: {orig_lines} → {len(text.splitlines())}")
print("\nChanges:")
for c in changes: print(f"  {c}")

result = subprocess.run([sys.executable, "-m", "py_compile", str(path)],
                        capture_output=True, text=True)
print("\n✓ SYNTAX OK" if result.returncode == 0 else "\n✗ ERROR:\n" + result.stderr)
