import streamlit as st
import pandas as pd
from pathlib import Path

from offopen_utils import (
    compute_offopen_for_dates,
    load_minute_dataframe,
)


st.title("Off-Open Returns (Minutes-only)")
default_dir = Path(r"C:\Users\David Alcosser\Documents\Visual Code\codex_ta_toolkit\per_ticker_minute")
minute_dir = Path(st.text_input("Minute parquet folder", value=str(default_dir))).expanduser()
ticker = st.text_input("Ticker", value="AAPL").strip().upper()
marks = st.multiselect("Minute marks", options=[1,2,3,5,10,15,30,60], default=[1,3,5,10,15])
tol = st.slider("Open tolerance (minutes)", min_value=0, max_value=15, value=5)
allow_nearest = st.checkbox("Allow nearest fallback beyond tolerance", value=True)
col1, col2 = st.columns(2)
with col1:
    d_start = st.date_input("Start date", value=pd.Timestamp('today').date())
with col2:
    d_end = st.date_input("End date", value=pd.Timestamp('today').date())

if st.button("Compute"):
    if not minute_dir.exists():
        st.error(f"Minute folder not found: {minute_dir}")
    else:
        intr_all = load_minute_dataframe(minute_dir, ticker)
        if intr_all is None or intr_all.empty:
            st.error(f"Minute parquet not found or empty for {ticker} under {minute_dir}")
        else:
            try:
                st.caption(f"Loaded minutes: {len(intr_all)} | Range ET: {intr_all.index.min()} -> {intr_all.index.max()} | Has Close: {'Close' in intr_all.columns}")
            except Exception:
                pass
            d0 = pd.Timestamp(d_start)
            d1 = pd.Timestamp(d_end)
            if d1 < d0:
                d0, d1 = d1, d0
            dates = list(pd.date_range(d0, d1, freq='D'))
            # Optional previews
            for dt in dates:
                try:
                    show_prev = st.checkbox(f"Preview 09:24-09:35 for {dt.date()}", value=False, key=f"pv_{dt.date()}")
                except Exception:
                    show_prev = False
                if show_prev:
                    seg = intr_all.loc[pd.Timestamp(dt).normalize(): pd.Timestamp(dt).normalize() + pd.Timedelta(days=1)]
                    win = seg[(seg.index.time >= pd.Timestamp('09:24').time()) & (seg.index.time <= pd.Timestamp('09:35').time())]
                    st.dataframe(win[['Open','High','Low','Close']].head(40), width='stretch')
            if not dates:
                st.info('No dates selected.')
            else:
                tbl = compute_offopen_for_dates(intr_all, dates, marks, int(tol), allow_nearest=bool(allow_nearest))
                if tbl.empty:
                    st.info('No rows computed (check date range or tolerance).')
                else:
                    st.dataframe(tbl, width='stretch')
                    try:
                        st.download_button('Download CSV', data=tbl.to_csv(), file_name=f'{ticker}_offopen_minutes_only.csv', mime='text/csv')
                    except Exception:
                        pass
                    try:
                        st.download_button('Download CSV', data=tbl.to_csv(), file_name=f'{ticker}_offopen_minutes_only.csv', mime='text/csv')
                    except Exception:
                        pass
