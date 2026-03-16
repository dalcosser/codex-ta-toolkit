"""
Portable configuration layer for app18.py.

Replaces hardcoded local paths with env-var-driven defaults.
Import this at the top of app18.py and use the constants below
instead of inline path literals.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Data directories
# ---------------------------------------------------------------------------
PARQUET_DIR = os.environ.get(
    "PER_TICKER_PARQUET_DIR",
    str(Path.home() / "Documents/Visual Code/Polygon Data/per_ticker_daily_ohlcv"),
)
MINUTE_DIR = os.environ.get(
    "PER_TICKER_MINUTE_DIR",
    str(Path.home() / "Documents/Visual Code/Polygon Data/per_ticker_minute"),
)

# ---------------------------------------------------------------------------
# Bloomberg / spreadsheet paths  (Global tabs)
# Override via env vars when running outside your home machine.
# ---------------------------------------------------------------------------
FUTURES_SPREADSHEET = os.environ.get(
    "FUTURES_SPREADSHEET",
    str(Path.home() / "Documents/Futures Spread2xlsb (version 1).xlsx"),
)
TEN_DAY_SCREEN = os.environ.get(
    "TEN_DAY_SCREEN",
    str(Path.home() / "Documents/10Day Screen 2025v2.xls"),
)
PARITY_SHEET = os.environ.get(
    "PARITY_SHEET",
    str(Path.home() / "OneDrive - Merus Global/PARITY SHEET (9).xlsx"),
)
BETA_UNIVERSE = os.environ.get(
    "BETA_UNIVERSE",
    str(Path.home() / "Documents/Work/BETA Universe.xlsx"),
)

# ---------------------------------------------------------------------------
# API keys  (already env-driven in app18, listed here for documentation)
# ---------------------------------------------------------------------------
# POLYGON_API_KEY / MASSIVE_API_KEY
# ALPACA_API_KEY_ID / ALPACA_API_SECRET_KEY
# TIINGO_API_KEY
# OPENAI_API_KEY
# ANTHROPIC_API_KEY
# THETADATA_TERMINAL_URL
