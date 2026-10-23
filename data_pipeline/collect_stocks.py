# data_pipeline/collect_stocks.py
# ─────────────────────────────────────────────
# Fetches daily stock data from Yahoo Finance
# and saves it to the stock_prices table in PostgreSQL
# ─────────────────────────────────────────────

import yfinance as yf
import pandas as pd
from psycopg2.extras import execute_values
import sys
import os
from datetime import datetime

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import STOCKS, START_DATE, END_DATE, YFINANCE_INTERVAL
from data_pipeline.db_connection import get_connection


def fetch_stock_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """
    Download daily OHLCV data for one ticker using yfinance.
    Returns a clean DataFrame.
    """
    print(f"  Fetching {ticker} from {start} to {end}...")

    try:
        df = yf.download(ticker, start=start, end=end,
                         interval=YFINANCE_INTERVAL,
                         auto_adjust=True, progress=False)
    except Exception as e:
        print(f"  [ERROR] yfinance failed for {ticker}: {e}")
        return pd.DataFrame()

    if df.empty:
        print(f"  [WARNING] No data returned for {ticker}")
        return pd.DataFrame()

    # yfinance 1.2+ may return MultiIndex columns for single tickers
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Keep only columns we need
    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    df.index = pd.to_datetime(df.index).date  # keep date only (no time)
    df["ticker"] = ticker

    # Rename columns to match DB schema
    df = df.rename(columns={
        "Open":   "open",
        "High":   "high",
        "Low":    "low",
        "Close":  "close",
        "Volume": "volume",
    })

    df = df.reset_index().rename(columns={"index": "date"})
    df = df.dropna(subset=["close"])  # drop rows with no close price

    print(f"  [OK] {len(df)} rows fetched for {ticker}")
    return df


def save_to_db(df: pd.DataFrame, conn) -> int:
    """
    Insert rows into stock_prices table.
    Uses ON CONFLICT DO NOTHING to skip duplicates.
    Returns number of rows inserted.
    """
    if df.empty:
        return 0

    rows = [
        (row["ticker"], row["date"], row["open"], row["high"],
         row["low"], row["close"], row["volume"])
        for _, row in df.iterrows()
    ]

    sql = """
        INSERT INTO stock_prices (ticker, date, open, high, low, close, volume)
        VALUES %s
        ON CONFLICT (ticker, date) DO NOTHING
    """

    with conn.cursor() as cur:
        execute_values(cur, sql, rows)
    conn.commit()

    return len(rows)


def collect_all_stocks():
    """
    Main function: fetch data for all tickers in settings.py
    and save to PostgreSQL.
    """
    print("=" * 50)
    print("Phase 1 — Stock Data Collection")
    print(f"Stocks : {STOCKS}")
    print(f"Period : {START_DATE} → {END_DATE}")
    print("=" * 50)

    conn = get_connection()
    total_inserted = 0

    for ticker in STOCKS:
        df = fetch_stock_data(ticker, START_DATE, END_DATE)
        if not df.empty:
            inserted = save_to_db(df, conn)
            total_inserted += inserted
            print(f"  Saved {inserted} rows for {ticker}")

    conn.close()

    print("=" * 50)
    print(f"Done! Total rows saved: {total_inserted}")
    print("=" * 50)


if __name__ == "__main__":
    collect_all_stocks()