# config/settings.py
# ─────────────────────────────────────────────
# Central configuration for the Stock Predictor
# ─────────────────────────────────────────────

import os
from datetime import date
from dotenv import load_dotenv

load_dotenv()

# ── PostgreSQL connection ──────────────────────
# Set DATABASE_URL env var to use a remote DB (e.g. Neon).
# Falls back to local PostgreSQL if not set.
_DATABASE_URL = os.environ.get("DATABASE_URL", "")

if _DATABASE_URL:
    import urllib.parse
    _u = urllib.parse.urlparse(_DATABASE_URL)
    DB_CONFIG = {
        "host":     _u.hostname,
        "port":     _u.port or 5432,
        "database": _u.path.lstrip("/"),
        "user":     _u.username,
        "password": _u.password,
        "sslmode":  "require",
    }
else:
    DB_CONFIG = {
        "host":     os.environ.get("PGHOST", "localhost"),
        "port":     int(os.environ.get("PGPORT", "5432")),
        "database": os.environ.get("PGDATABASE", "stock_predictor"),
        "user":     os.environ.get("PGUSER", "postgres"),
        "password": os.environ.get("PGPASSWORD", ""),
    }

# ── Stocks to track ────────────────────────────
STOCKS = ["AAPL", "TSLA", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "NFLX", "AMD", "BABA"]

# ── Date range for historical stock prices ─────
START_DATE = "2015-01-01"
END_DATE   = date.today().strftime("%Y-%m-%d")

# ── yfinance settings ─────────────────────────
YFINANCE_INTERVAL = "1d"

# ── API Keys ──────────────────────────────────
NEWSAPI_KEY    = os.environ.get("NEWSAPI_KEY", "")
GNEWS_KEY      = os.environ.get("GNEWS_KEY", "")
ALPHA_KEY      = os.environ.get("ALPHA_KEY", "")
ANTHROPIC_KEY  = os.environ.get("ANTHROPIC_KEY", "")
GROQ_KEY       = os.environ.get("GROQ_KEY", "")
TOGETHER_KEY   = os.environ.get("TOGETHER_KEY", "")
FRED_KEY       = os.environ.get("FRED_KEY", "")

# ── Alpha Vantage backfill date range ─────────
# 2024-01-01 → yesterday (GNews covers the live window)
ALPHA_BACKFILL_START = "2024-01-01"
ALPHA_BACKFILL_END   = "2026-04-25"   # covers up to today
