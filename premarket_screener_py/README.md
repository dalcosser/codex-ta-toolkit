# Pre-Market Screener (Python)

Translated from your R/Shiny workflow in `c:\\blp\\data\\app (1).R`.

## File

- `premarket_screener_dash.py`

## Setup

1. Install deps:
   `pip install -r requirements.txt`
2. Set API key:
   - env var: `MASSIVE_API_KEY`
   - or put key in `massive_api_key.txt` in the same folder
3. Run:
   `python premarket_screener_dash.py`
4. Open:
   `http://127.0.0.1:8050`

## Notes

- This version preserves core logic: snapshot, gainers/losers, historical gap matching, interval stats, and intraday charts.
- The heavy historical stats call many API requests and can take time.
- `Replay` mode rebuilds a past pre-market leaderboard (04:00-09:29 ET) from minute bars plus prior close.
- Replay speed depends on `Replay Universe` size; start with `200-500` for faster testing.
