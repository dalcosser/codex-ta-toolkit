"""
Add Bloomberg Terminal as the first data provider in app18.py.
- Enables HAS_XBBG with a safe try/import
- Inserts _try_bloomberg() helper inside _fetch_ohlc_uncached()
- Adds "Bloomberg" to the preferred-provider selectbox
- Updates the provider waterfall so Bloomberg leads in auto mode
- Adds Bloomberg status to the diagnostics caption
Run: python tools/add_bloomberg.py
"""
import sys, subprocess
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
path = ROOT / "app18.py"

text = path.read_text(encoding="utf-8")
orig_lines = len(text.splitlines())
changes = []

# ── 1. Enable HAS_XBBG ────────────────────────────────────────────────────────
OLD_XBBG = "# Bloomberg not used\nHAS_XBBG = False"
NEW_XBBG = (
    "# Bloomberg terminal integration (requires blpapi + terminal running)\n"
    "try:\n"
    "    import xbbg  # noqa: F401\n"
    "    HAS_XBBG = True\n"
    "except ImportError:\n"
    "    HAS_XBBG = False"
)
if OLD_XBBG in text:
    text = text.replace(OLD_XBBG, NEW_XBBG, 1)
    changes.append("1 \u2713 HAS_XBBG enabled with try/import guard")
else:
    changes.append("1 \u2717 HAS_XBBG block not matched")

# ── 2. Add _try_bloomberg() helper before _try_polygon() ──────────────────────
BLOOMBERG_HELPER = (
    "    # Helper: Bloomberg Terminal (institutional-grade, highest priority)\n"
    "    def _try_bloomberg():\n"
    "        if not HAS_XBBG:\n"
    "            return None\n"
    "        try:\n"
    "            from xbbg import blp as _blp\n"
    "            import datetime as _dt\n"
    "            _bbg_tkr = f\"{ticker} US Equity\"\n"
    "            per_days = {\n"
    "                '1d':1,'5d':5,'7d':7,'14d':14,'30d':30,'60d':60,'90d':90,\n"
    "                '1mo':30,'3mo':90,'6mo':180,'1y':365,'2y':730,'5y':1825,\n"
    "                '10y':3650,'max':36500\n"
    "            }\n"
    "            _days = per_days.get(str(period or '1y').lower(), 365)\n"
    "            _now  = _dt.datetime.now()\n"
    "            _s_dt = _pd.to_datetime(start)  if start else (_now - _dt.timedelta(days=_days))\n"
    "            _e_dt = _pd.to_datetime(end)    if end   else _now\n"
    "            if _interval == '1d':\n"
    "                _df = _blp.bdh(\n"
    "                    _bbg_tkr,\n"
    "                    ['PX_OPEN','PX_HIGH','PX_LOW','PX_LAST','PX_VOLUME'],\n"
    "                    start_date=_s_dt.strftime('%Y-%m-%d'),\n"
    "                    end_date=_e_dt.strftime('%Y-%m-%d'),\n"
    "                )\n"
    "                if _df is None or _df.empty:\n"
    "                    return None\n"
    "                if isinstance(_df.columns, _pd.MultiIndex):\n"
    "                    _df.columns = _df.columns.get_level_values(-1)\n"
    "                _df = _df.rename(columns={\n"
    "                    'PX_OPEN':'Open','PX_HIGH':'High','PX_LOW':'Low',\n"
    "                    'PX_LAST':'Close','PX_VOLUME':'Volume',\n"
    "                    'px_open':'Open','px_high':'High','px_low':'Low',\n"
    "                    'px_last':'Close','px_volume':'Volume',\n"
    "                })\n"
    "            else:\n"
    "                _int_map = {'1m':1,'5m':5,'15m':15,'30m':30,'60m':60,'1h':60}\n"
    "                _bbg_int = _int_map.get(_interval, 1)\n"
    "                _df = _blp.bdib(\n"
    "                    _bbg_tkr,\n"
    "                    dt_start=_s_dt,\n"
    "                    dt_end=_e_dt,\n"
    "                    event='TRADE',\n"
    "                    interval=_bbg_int,\n"
    "                )\n"
    "                if _df is None or _df.empty:\n"
    "                    return None\n"
    "                if isinstance(_df.columns, _pd.MultiIndex):\n"
    "                    _df.columns = _df.columns.get_level_values(-1)\n"
    "                _df = _df.rename(columns={\n"
    "                    'open':'Open','high':'High','low':'Low',\n"
    "                    'close':'Close','volume':'Volume',\n"
    "                })\n"
    "            _keep = [c for c in ['Open','High','Low','Close','Volume'] if c in _df.columns]\n"
    "            _df = _df[_keep]\n"
    "            if _df.empty:\n"
    "                return None\n"
    "            try:\n"
    "                st.session_state['last_fetch_provider'] = 'bloomberg'\n"
    "            except Exception:\n"
    "                pass\n"
    "            return _df\n"
    "        except Exception as e:\n"
    "            try:\n"
    "                st.session_state['last_fetch_errors'].append(f\"bloomberg: {e}\")\n"
    "            except Exception:\n"
    "                pass\n"
    "            return None\n"
    "\n"
)

ANCHOR = "    # Helper: Polygon first if configured\n    def _try_polygon():"
if ANCHOR in text:
    text = text.replace(ANCHOR, BLOOMBERG_HELPER + ANCHOR, 1)
    changes.append("2 \u2713 _try_bloomberg() inserted before _try_polygon()")
else:
    changes.append("2 \u2717 Polygon anchor not found")

# ── 3. Add Bloomberg to preferred-provider selectbox ─────────────────────────
OLD_SEL = '["Auto (best available)", "Polygon", "Alpaca", "Tiingo"]'
NEW_SEL = '["Auto (best available)", "Bloomberg", "Polygon", "Alpaca", "Tiingo"]'
if OLD_SEL in text:
    text = text.replace(OLD_SEL, NEW_SEL, 1)
    changes.append("3 \u2713 Bloomberg added to preferred-provider selectbox")
else:
    changes.append("3 \u2717 Selectbox options not matched")

# ── 4. Update diagnostics caption to show Bloomberg status ────────────────────
OLD_DIAG = (
    "            st.caption(f\"Polygon: {'\\u2713' if _pk else '\\u2717'}  "
    "Alpaca: {'\\u2713' if _ak else '\\u2717'}  Source: {_src}\")"
)
NEW_DIAG = (
    "            _bbg_ok = HAS_XBBG\n"
    "            st.caption(\n"
    "                f\"Bloomberg: {'\\u2713' if _bbg_ok else '\\u2717'}  \"\n"
    "                f\"Polygon: {'\\u2713' if _pk else '\\u2717'}  \"\n"
    "                f\"Alpaca: {'\\u2713' if _ak else '\\u2717'}  \"\n"
    "                f\"Source: {_src}\"\n"
    "            )"
)
if OLD_DIAG in text:
    text = text.replace(OLD_DIAG, NEW_DIAG, 1)
    changes.append("4 \u2713 Diagnostics caption updated with Bloomberg status")
else:
    # Try alternate unicode
    OLD_DIAG2 = (
        '            st.caption(f"Polygon: {\'\\u2713\' if _pk else \'\\u2717\'}  '
        'Alpaca: {\'\\u2713\' if _ak else \'\\u2717\'}  Source: {_src}")'
    )
    if OLD_DIAG2 in text:
        text = text.replace(OLD_DIAG2, NEW_DIAG, 1)
        changes.append("4 \u2713 Diagnostics caption updated (alt match)")
    else:
        changes.append("4 \u26a0 Diagnostics caption not matched — search manually")

# ── 5. Update provider waterfall ─────────────────────────────────────────────
OLD_WATERFALL = """\
    # Preferred real-time providers (Polygon, Alpaca, Tiingo)
    providers = [("Polygon", _try_polygon), ("Alpaca", _try_alpaca), ("Tiingo", _try_tiingo)]
    try:
        pref = st.session_state.get('preferred_provider')
    except Exception:
        pref = None
    # Build provider order including Twelve Data (Polygon and Tiingo first)
    def _auto_providers():
        return [("Polygon", _try_polygon), ("Tiingo", _try_tiingo), ("Alpaca", _try_alpaca), ("TwelveData", _try_twelvedata)]
    if pref and isinstance(pref, str) and pref.startswith("Polygon"):
        providers = [("Polygon", _try_polygon), ("Tiingo", _try_tiingo), ("Alpaca", _try_alpaca), ("TwelveData", _try_twelvedata)]
    elif pref and isinstance(pref, str) and pref.startswith("Alpaca"):
        providers = [("Alpaca", _try_alpaca), ("Polygon", _try_polygon), ("Tiingo", _try_tiingo), ("TwelveData", _try_twelvedata)]
    elif pref and isinstance(pref, str) and pref.startswith("Tiingo"):
        providers = [("Tiingo", _try_tiingo), ("Polygon", _try_polygon), ("Alpaca", _try_alpaca), ("TwelveData", _try_twelvedata)]
    elif pref and isinstance(pref, str) and pref.lower().startswith("twelvedata"):
        providers = [("TwelveData", _try_twelvedata), ("Polygon", _try_polygon), ("Alpaca", _try_alpaca), ("Tiingo", _try_tiingo)]
    else:
        providers = _auto_providers()"""

NEW_WATERFALL = """\
    # ── Provider waterfall (Bloomberg leads in auto; user pref overrides order) ──
    try:
        pref = st.session_state.get('preferred_provider')
    except Exception:
        pref = None

    def _auto_providers():
        \"\"\"Auto order: Bloomberg (institutional) > Polygon > Tiingo > Alpaca > TwelveData\"\"\"
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
        providers = _auto_providers()"""

if OLD_WATERFALL in text:
    text = text.replace(OLD_WATERFALL, NEW_WATERFALL, 1)
    changes.append("5 \u2713 Provider waterfall updated — Bloomberg leads auto order")
else:
    changes.append("5 \u2717 Waterfall block not matched exactly")

# ── Write & verify ────────────────────────────────────────────────────────────
path.write_text(text, encoding="utf-8")
new_lines = len(text.splitlines())
print(f"Lines: {orig_lines} \u2192 {new_lines}  (+{new_lines - orig_lines})")
print("\nChanges:")
for c in changes:
    print(f"  {c}")

result = subprocess.run([sys.executable, "-m", "py_compile", str(path)],
                        capture_output=True, text=True)
print("\n\u2713 SYNTAX OK" if result.returncode == 0 else "\n\u2717 SYNTAX ERROR:\n" + result.stderr)
