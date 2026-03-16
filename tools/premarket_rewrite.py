"""Script to replace the Premarket tab block in app18.py with a clean version."""
import sys

NEW_BLOCK = r"""if nav == 'Premarket':
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

    _pd1, _pd2, _pd3 = st.columns([3, 3, 1])
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
    with _pd3:
        run_pm = st.button("Scan", type="primary", use_container_width=True, key="pm_run_main")

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

"""

APP = r"C:\Users\David Alcosser\Documents\Visual Code\codex_ta_toolkit\app18.py"

with open(APP, encoding="utf-8") as f:
    lines = f.readlines()

# Premarket block: lines 8523-8826 (1-indexed) → 0-indexed 8522-8825
START = 8522   # 0-indexed first line of block (inclusive)
END   = 8826   # 0-indexed first line AFTER block (exclusive)

new_lines = lines[:START] + [NEW_BLOCK + "\n"] + lines[END:]

with open(APP, "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print(f"Done. Was {len(lines)} lines, now {len(new_lines)} lines.")
print(f"Block: {END - START} lines removed, replaced with {NEW_BLOCK.count(chr(10))} lines.")
