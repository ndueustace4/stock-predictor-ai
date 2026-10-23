# data_pipeline/collect_alpha.py
# ─────────────────────────────────────────────
# Alpha Vantage news backfill  (2024-01-01 → 2025-12-31)
# Month-wide windows, VADER scoring, full deduplication.
# Called 3×/day by scheduler.py (07:00, 13:00, 20:00 UTC)
#
# Fix: if a month returns 0 articles (e.g. BABA April 2024),
# insert a sentinel row so the month is marked covered and
# the backfill moves on instead of retrying forever.
# ─────────────────────────────────────────────

import requests
import hashlib
import time
import sys
import os
from datetime import datetime, date
from calendar import monthrange

import pandas as pd
from psycopg2.extras import execute_values

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import STOCKS, ALPHA_KEY, ALPHA_BACKFILL_START, ALPHA_BACKFILL_END
from data_pipeline.db_connection import get_connection
from data_pipeline.sentiment_analysis import analyze_sentiment
from data_pipeline.monitoring import log_run

import nltk
nltk.download("vader_lexicon", quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer


# ─────────────────────────────────────────────
# Month-window helpers
# ─────────────────────────────────────────────

def all_month_windows(start: date, end: date) -> list[tuple[date, date]]:
    """Split [start, end] into full calendar-month windows."""
    windows = []
    cur = date(start.year, start.month, 1)
    while cur <= end:
        last_day = monthrange(cur.year, cur.month)[1]
        w_end   = min(date(cur.year, cur.month, last_day), end)
        w_start = max(cur, start)
        windows.append((w_start, w_end))
        cur = date(cur.year + (cur.month == 12), (cur.month % 12) + 1, 1)
    return windows


def get_covered_months(conn, ticker: str) -> set[tuple[int, int]]:
    """
    Return (year, month) pairs already fetched from Alpha Vantage.
    A month is covered if ANY row exists — including sentinel rows
    inserted for empty months.
    """
    sql = """
        SELECT DISTINCT EXTRACT(YEAR  FROM date)::int,
                        EXTRACT(MONTH FROM date)::int
        FROM news_sentiment
        WHERE ticker = %s
          AND source ILIKE 'AlphaVantage%%'
    """
    with conn.cursor() as cur:
        cur.execute(sql, (ticker,))
        return {(r[0], r[1]) for r in cur.fetchall()}


def get_missing_month_windows(conn, ticker: str) -> list[tuple[date, date]]:
    """Return month windows not yet fetched for this ticker."""
    start   = datetime.strptime(ALPHA_BACKFILL_START, "%Y-%m-%d").date()
    end     = datetime.strptime(ALPHA_BACKFILL_END,   "%Y-%m-%d").date()
    covered = get_covered_months(conn, ticker)
    return [
        (ws, we) for ws, we in all_month_windows(start, end)
        if (ws.year, ws.month) not in covered
    ]


def get_ticker_queue(conn) -> list[tuple[str, list]]:
    """Sort tickers by most-missing-first so all 10 progress evenly."""
    queue = []
    for ticker in STOCKS:
        windows = get_missing_month_windows(conn, ticker)
        if windows:
            queue.append((ticker, windows))
    queue.sort(key=lambda x: len(x[1]), reverse=True)
    return queue


# ─────────────────────────────────────────────
# Sentinel row — marks an empty month as covered
# ─────────────────────────────────────────────

def mark_month_covered(ticker: str, window_start: date, conn):
    """
    Insert a sentinel row so a month with 0 articles is marked
    as fetched and won't be retried on future runs.
    The row has NULL sentiment so it won't affect model training.
    """
    sql = """
        INSERT INTO news_sentiment
            (ticker, date, headline, sentiment_score, sentiment_label, source)
        VALUES (%s, %s, %s, NULL, NULL, %s)
        ON CONFLICT (ticker, date, headline) DO NOTHING
    """
    sentinel_headline = f"__NO_NEWS_{window_start.strftime('%Y%m')}__"
    with conn.cursor() as cur:
        cur.execute(sql, (
            ticker,
            window_start,
            sentinel_headline,
            f"AlphaVantage/NO_DATA",
        ))
    conn.commit()
    print(f"    → No articles found — month marked as covered (sentinel inserted)")


# ─────────────────────────────────────────────
# Alpha Vantage fetch
# ─────────────────────────────────────────────

def fetch_alpha_news(ticker: str, time_from: date, time_to: date,
                     limit: int = 1000) -> list[dict]:
    """Fetch up to 1000 articles for one ticker over a month window."""
    url = "https://www.alphavantage.co/query"
    params = {
        "function":  "NEWS_SENTIMENT",
        "tickers":   ticker,
        "time_from": time_from.strftime("%Y%m%dT0000"),
        "time_to":   time_to.strftime("%Y%m%dT2359"),
        "limit":     limit,
        "sort":      "EARLIEST",
        "apikey":    ALPHA_KEY,
    }

    try:
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()

        if "Note" in data:
            print(f"  [RATE LIMIT] Alpha Vantage throttled. Sleeping 65s...")
            time.sleep(65)
            return None   # None = rate limited, don't mark as covered

        if "Information" in data:
            print(f"  [QUOTA] Daily limit reached: {data['Information']}")
            return None   # None = quota hit, don't mark as covered

        if "Error Message" in data:
            print(f"  [API ERROR] {data['Error Message']}")
            return None

        items    = data.get("feed", [])
        reported = data.get("items", len(items))
        print(f"  [OK] {len(items)} articles (API total={reported}) "
              f"for {ticker} ({time_from} → {time_to})")
        return items   # may be empty list [] — that's fine, mark as covered

    except requests.exceptions.RequestException as e:
        print(f"  [ERROR] Network error for {ticker}: {e}")
        return None


# ─────────────────────────────────────────────
# Parse + VADER score
# ─────────────────────────────────────────────

def parse_alpha_articles(ticker: str, articles: list,
                          analyzer: SentimentIntensityAnalyzer) -> pd.DataFrame:
    """Parse articles, apply VADER, discard Alpha's own sentiment field."""
    if not articles:
        return pd.DataFrame()

    seen = set()
    rows = []

    for article in articles:
        try:
            pub_date = datetime.strptime(
                article.get("time_published", "")[:8], "%Y%m%d"
            ).date()
        except (ValueError, TypeError):
            continue

        if not any(
            ts.get("ticker", "").upper() == ticker.upper()
            for ts in article.get("ticker_sentiment", [])
        ):
            continue

        headline = (article.get("title", "") or "").strip()[:500]
        if not headline or headline == "[Removed]":
            continue

        h = hashlib.md5(f"{ticker}|{pub_date}|{headline}".encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)

        score, label = analyze_sentiment(headline, analyzer)
        source_name  = (article.get("source", "") or "Unknown")[:80]

        rows.append({
            "ticker":          ticker,
            "date":            pub_date,
            "headline":        headline,
            "sentiment_score": score,
            "sentiment_label": label,
            "source":          f"AlphaVantage/{source_name}",
        })

    return pd.DataFrame(rows)


def save_alpha_to_db(df: pd.DataFrame, conn) -> int:
    if df.empty:
        return 0
    rows = [
        (row["ticker"], row["date"], row["headline"],
         row["sentiment_score"], row["sentiment_label"], row["source"])
        for _, row in df.iterrows()
    ]
    sql = """
        INSERT INTO news_sentiment
            (ticker, date, headline, sentiment_score, sentiment_label, source)
        VALUES %s
        ON CONFLICT (ticker, date, headline) DO NOTHING
    """
    with conn.cursor() as cur:
        execute_values(cur, sql, rows)
    conn.commit()
    return len(rows)


# ─────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────

def run_alpha_backfill(max_windows_per_run: int = 5):
    """
    Process up to `max_windows_per_run` missing months per call.
    Returns None = rate limited (stop early).
    Empty month [] = insert sentinel and move on.
    """
    print("=" * 55)
    print("Alpha Vantage Backfill")
    print(f"Target  : {ALPHA_BACKFILL_START} → {ALPHA_BACKFILL_END}")
    print(f"Windows this run: up to {max_windows_per_run}")
    print("=" * 55)

    conn     = get_connection()
    analyzer = SentimentIntensityAnalyzer()
    queue    = get_ticker_queue(conn)

    if not queue:
        print("\n✓ All tickers fully covered — backfill complete!")
        conn.close()
        log_run(source="alpha_vantage", articles_collected=0,
                requests_made=0, windows_processed=0,
                notes="All complete")
        return

    total_saved   = 0
    total_windows = 0
    requests_used = 0

    for ticker, windows in queue:
        if requests_used >= max_windows_per_run:
            print(f"\n[LIMIT] Reached {max_windows_per_run}-window cap. Stopping.")
            break

        w_start, w_end = windows[0]
        print(f"\n  [{ticker}] {len(windows)} months remaining — "
              f"fetching {w_start:%Y-%m}")

        articles = fetch_alpha_news(ticker, w_start, w_end)
        requests_used += 1

        if articles is None:
            # Rate limited or quota hit — stop immediately, don't mark covered
            print("  [STOP] Quota or rate limit hit — stopping run early.")
            break

        if len(articles) == 0:
            # API responded but genuinely no news — mark month as covered
            # so we don't waste a request on it again tomorrow
            mark_month_covered(ticker, w_start, conn)
            total_windows += 1
        else:
            df    = parse_alpha_articles(ticker, articles, analyzer)
            saved = save_alpha_to_db(df, conn)
            total_saved   += saved
            total_windows += 1
            print(f"    → {saved} articles saved")

        if requests_used < max_windows_per_run:
            time.sleep(13)

    conn.close()

    log_run(
        source="alpha_vantage",
        articles_collected=total_saved,
        requests_made=requests_used,
        windows_processed=total_windows,
    )

    print("\n" + "=" * 55)
    print(f"Run complete: {total_windows} months processed, "
          f"{total_saved} articles saved.")
    print("=" * 55)


if __name__ == "__main__":
    run_alpha_backfill()