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

        # Regime badge + VIX
        rc1, rc2, rc3 = st.columns([2, 1, 1])
        with rc1:
            st.markdown(
                f"<div style='background:{regime_col};padding:8px 16px;border-radius:6px;"
                f"font-size:1.1rem;font-weight:700;color:#fff;display:inline-block'>"
                f"Regime: {regime_lbl}</div>",
                unsafe_allow_html=True,
            )
        with rc2:
            if _vix_val is not None:
                st.metric("VIX", f"{_vix_val:.2f}")
            else:
                st.metric("VIX", "N/A")
        with rc3:
            st.metric("SPY % Chg", f"{_spy_chg:+.2f}%" if _spy_chg is not None else "N/A")

        st.markdown("---")

        # ── Fetch supplemental data ───────────────────────────────────────────
        _idx_px   = _tf_index_prices()
        _cmdty_px = _tf_commodity_spot()

        # ─── ROW 1: Equity Indices (cash level + ETF) ─────────────────────────
        st.markdown("**📊 Equity Indices**")
        _ix_cols = st.columns(3)
        _ix_map = [
            ("S&P 500",    "I:SPX", "SPY"),
            ("NDX 100",    "I:NDX", "QQQ"),
            ("Russell 2K", "I:RUT", "IWM"),
        ]
        for _col_w, (_lbl, _ix_t, _etf_t) in zip(_ix_cols, _ix_map):
            _ip = _idx_px.get(_ix_t) or {}
            _ep = _macro_snap.get(_etf_t) or {}
            with _col_w:
                if _ip.get("price"):
                    st.metric(_lbl, f"{_ip['price']:,.0f}",
                              delta=f"{_ip['chg_pct']:+.2f}%")
                    if _ep.get("price"):
                        st.caption(f"{_etf_t}: ${_ep['price']:.2f}  ({_ep.get('chg_pct', 0):+.2f}%)")
                elif _ep.get("price"):
                    st.metric(f"{_lbl} ({_etf_t})", f"${_ep['price']:.2f}",
                              delta=f"{_ep.get('chg_pct', 0):+.2f}%")
                else:
                    st.metric(_lbl, "—")

        st.markdown("---")

        # ─── ROW 2: Rates + Dollar ─────────────────────────────────────────────
        st.markdown("**🏦 Rates & Dollar**")
        _r_cols = st.columns(4)
        _et_tlt  = _macro_snap.get("TLT") or {}
        _ix_tnx  = _idx_px.get("I:TNX") or {}
        _ix_tyx  = _idx_px.get("I:TYX") or {}
        _ix_dxy  = _idx_px.get("I:DXY") or {}
        with _r_cols[0]:
            _pt = _et_tlt.get("price")
            st.metric("Bonds (TLT)", f"${_pt:.2f}" if _pt else "—",
                      delta=f"{_et_tlt.get('chg_pct', 0):+.2f}%" if _pt else None)
        with _r_cols[1]:
            _yt = _ix_tnx.get("price")
            st.metric("10yr Yield", f"{_yt:.3f}%" if _yt else "—",
                      delta=f"{_ix_tnx.get('chg_pct', 0):+.2f}%" if _yt else None)
        with _r_cols[2]:
            _yt30 = _ix_tyx.get("price")
            st.metric("30yr Yield", f"{_yt30:.3f}%" if _yt30 else "—",
                      delta=f"{_ix_tyx.get('chg_pct', 0):+.2f}%" if _yt30 else None)
        with _r_cols[3]:
            _dxy = _ix_dxy.get("price")
            st.metric("DXY (Dollar)", f"{_dxy:.2f}" if _dxy else "—",
                      delta=f"{_ix_dxy.get('chg_pct', 0):+.2f}%" if _dxy else None)

        st.markdown("---")

        # ─── ROW 3: FX ────────────────────────────────────────────────────────
        st.markdown("**💱 FX**")
        _fx_cols = st.columns(4)
        _fx_usdjpy = _cmdty_px.get("USDJPY") or {}
        _fx_eurusd = _cmdty_px.get("EURUSD") or {}
        with _fx_cols[0]:
            _pj = _fx_usdjpy.get("price")
            st.metric("USD/JPY", f"{_pj:.2f}" if _pj else "—",
                      delta=f"{_fx_usdjpy.get('chg_pct', 0):+.2f}%" if _pj else None)
        with _fx_cols[1]:
            _pe = _fx_eurusd.get("price")
            st.metric("EUR/USD", f"{_pe:.4f}" if _pe else "—",
                      delta=f"{_fx_eurusd.get('chg_pct', 0):+.2f}%" if _pe else None)

        st.markdown("---")

        # ─── ROW 4: Commodities (spot price + ETF) ────────────────────────────
        st.markdown("**🪙 Commodities**")
        _cm_cols = st.columns(4)
        _cm_map = [
            ("Gold",      "GOLD",   "GLD",  "${:,.2f}/oz",  "GLD"),
            ("Silver",    "SILVER", "SLV",  "${:,.2f}/oz",  "SLV"),
            ("WTI Crude", "OIL",    "USO",  "${:,.2f}/bbl", "USO"),
            ("Copper",    "COPPER", "CPER", "${:,.3f}/lb",  "CPER"),
        ]
        for _col_w, (_lbl, _ck, _etf_t, _fmt, _etf_lbl) in zip(_cm_cols, _cm_map):
            _sp = _cmdty_px.get(_ck) or {}
            _ep = _macro_snap.get(_etf_t) or {}
            with _col_w:
                if _sp.get("price"):
                    st.metric(_lbl, _fmt.format(_sp["price"]),
                              delta=f"{_sp['chg_pct']:+.2f}%")
                    if _ep.get("price"):
                        st.caption(f"{_etf_lbl}: ${_ep['price']:.2f}  ({_ep.get('chg_pct', 0):+.2f}%)")
                elif _ep.get("price"):
                    st.metric(f"{_lbl} ({_etf_lbl})", f"${_ep['price']:.2f}",
                              delta=f"{_ep.get('chg_pct', 0):+.2f}%")
                else:
                    st.metric(_lbl, "—")

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 2 — SCANNER SETUP
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("### Scanner Setup")
    _sc1, _sc2, _sc3 = st.columns([2, 2, 3])

    with _sc1:
        _tf_universe = st.selectbox(
            "Universe",
            ["NDX 100", "SPX 150", "Watchlist", "All Parquets"],
            key="tf_universe",
        )
        _tf_min_score = st.slider("Min Score", 0, 100, 40, 5, key="tf_min_score")
        _tf_min_align = st.selectbox("Min Alignment", [0, 1, 2, 3], index=1, key="tf_min_align")

    with _sc2:
        _tf_min_rvol = st.number_input("Min Rel Volume", 0.0, 10.0, 0.5, 0.1, key="tf_min_rvol")
        _tf_max_rsi  = st.slider("Max RSI", 30, 100, 75, 1, key="tf_max_rsi")
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
        u = st.session_state.get("tf_universe", "NDX 100")
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
            # SPY 20d return for RS scoring
            _spy_df = _load_daily_df("SPY")
            if _spy_df is not None and len(_spy_df) >= 21:
                _spy_close = pd.to_numeric(_spy_df["Close"], errors="coerce")
                _spy_ret20 = float((_spy_close.iloc[-1] / _spy_close.iloc[-21] - 1) * 100)
            else:
                _spy_ret20 = 0.0

            _results = []
            _prog = st.progress(0.0, text="Scanning…")
            _n = len(_universe)
            _today_norm = pd.Timestamp.today().normalize()
            for _idx, _tkr in enumerate(_universe):
                _prog.progress((_idx + 1) / _n, text=f"Scanning {_tkr} ({_idx+1}/{_n})")
                _sdf = _load_daily_df(_tkr)
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
                _score, _comps = _tf_score_ticker(_sdf, _spy_ret20, 0)
                if _score < st.session_state.get("tf_min_score", 40):
                    continue
                if _comps.get("align_score", 0) < st.session_state.get("tf_min_align", 1):
                    continue
                if _comps.get("rel_vol", 1.0) < st.session_state.get("tf_min_rvol", 0.5):
                    continue
                if _comps.get("rsi", 50) > st.session_state.get("tf_max_rsi", 75):
                    continue
                _setups = _tf_detect_setups(_sdf)
                # Setup filter
                _active_f = _tf_active_filters()
                if _active_f:
                    if not any(any(fk in s for s in _setups) for fk in _active_f):
                        continue
                # Optional options data
                _opts = {}
                if st.session_state.get("tf_fetch_opts", False):
                    try:
                        _opts = _tf_options_summary(_tkr)
                    except Exception:
                        pass
                _px_last = float(pd.to_numeric(_sdf["Close"], errors="coerce").iloc[-1])
                _results.append({
                    "Ticker":         _tkr,
                    "Price":          round(_px_last, 2),
                    "Score":          _score,
                    "Slope(0-25)":    _comps.get("Slope(0-25)", 0),
                    "RS(0-20)":       _comps.get("RS_SPY(0-20)", 0),
                    "RVol(0-20)":     _comps.get("RelVol(0-20)", 0),
                    "Setup(0-20)":    _comps.get("Setup(0-20)", 0),
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
            st.success(f"Scan complete — {len(_results)} tickers scored ≥ {st.session_state.get('tf_min_score',40)}")

    # ─────────────────────────────────────────────────────────────────────────
    # SECTION 4 — SCORECARD
    # ─────────────────────────────────────────────────────────────────────────
    _tf_results = st.session_state.get("tf_results", [])
    if _tf_results:
        st.markdown("### Scorecard — Top Results")
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
            "Ticker","Price","Score","Slope(0-25)","RS(0-20)","RVol(0-20)","Setup(0-20)",
            "Align","RelVol","RSI","Pull%","Ret20d%","Setups","ATM_IV","PC_Ratio","Skew25d"
        ] if c in _df_sc.columns]

        _df_display = _df_sc[_display_cols].copy()
        # Format numeric columns
        for _col in ["Score","Slope(0-25)","RS(0-20)","RVol(0-20)","Setup(0-20)",
                     "RelVol","RSI","Pull%","Ret20d%","ATM_IV","PC_Ratio","Skew25d"]:
            if _col in _df_display.columns:
                _df_display[_col] = pd.to_numeric(_df_display[_col], errors="coerce").round(1)

        st.dataframe(
            _df_display.style.applymap(_score_color, subset=["Score"]),
            use_container_width=True,
            height=400,
        )

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
    # SECTION 5 — DRILL-DOWN
    # ─────────────────────────────────────────────────────────────────────────
    st.markdown("### Drill-Down")
    _tf_ticker_list = [r["Ticker"] for r in _tf_results] if _tf_results else []
    _tf_manual_tkr  = st.text_input("Or type any ticker:", key="tf_manual_ticker", placeholder="e.g. NVDA").strip().upper()

    _drill_tkr = None
    if _tf_manual_tkr:
        _drill_tkr = _tf_manual_tkr
    elif _tf_ticker_list:
        _drill_tkr = st.selectbox("Select from results:", _tf_ticker_list, key="tf_drill_select")

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
            _dscore, _dcomps = _tf_score_ticker(_full_df, _spy_ret20_dd, 0) if _full_df is not None else (0, {})
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
                _score_labels = ["Slope", "RS_SPY", "RelVol", "Setup", "News"]
                _score_vals   = [
                    _dcomps.get("Slope(0-25)", 0),
                    _dcomps.get("RS_SPY(0-20)", 0),
                    _dcomps.get("RelVol(0-20)", 0),
                    _dcomps.get("Setup(0-20)", 0),
                    _dcomps.get("News(0-15)", 0),
                ]
                _bar_colors   = ["#4caf50","#2196f3","#ff9800","#9c27b0","#f44336"]
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

            # Options snapshot
            _dopts = _tf_options_summary(_drill_tkr)
            if _dopts:
                st.markdown("**Options Snapshot (next 45 days)**")
                _oc = st.columns(4)
                _oc[0].metric("ATM IV", f"{_dopts.get('atm_iv','—')}%" if _dopts.get('atm_iv') else "—")
                _oc[1].metric("P/C Ratio (OI)", str(_dopts.get("pc_ratio","—")))
                _oc[2].metric("25d Put Skew", str(_dopts.get("skew_25d","—")))
                _oc[3].metric("Chain Count", str(_dopts.get("chain_count","—")))
            else:
                st.caption("Options data unavailable (check Polygon API key or click 'Fetch Options Data').")

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
                        f"Chain={_dopts.get('chain_count','N/A')} contracts"
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
                    "8. OPTIONS ANGLE (if IV data available — strategy suggestion)\n"
                    "9. CONFIDENCE (1–10 with reasoning)\n\n"
                    "Be specific with price levels. Avoid vague language. "
                    "If options data is not available, skip section 8."
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
                    f"Detected setups: {', '.join(_dsetups) if _dsetups else 'None'}",
                    f"Market: {regime_lbl} | {_vix_txt}",
                    f"Macro moves: {_macro_summary}",
                    _opts_txt,
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
