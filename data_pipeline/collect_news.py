# data_pipeline/collect_news.py
# ─────────────────────────────────────────────
# GNews live collector  (last 30 days, every 30 min via scheduler)
# GNews historical backfill (Essential plan, date-range queries)
# ─────────────────────────────────────────────
# Rules:
#   • ONE batch query per ticker using company name (not per-day loops)
#   • VADER applied immediately on fetch — no separate scoring pass needed
#   • ON CONFLICT DO NOTHING  +  MD5 hash guard = double dedup layer
#   • Monitoring logged after every run
# ─────────────────────────────────────────────

import argparse
import requests
import hashlib
import sys
import os
import time
from datetime import datetime, date, timedelta

import pandas as pd
from psycopg2.extras import execute_values

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import STOCKS, GNEWS_KEY
from data_pipeline.db_connection import get_connection
from data_pipeline.sentiment_analysis import analyze_sentiment
from data_pipeline.finbert_sentiment import score_headlines as finbert_score
from data_pipeline.monitoring import log_run

import nltk
nltk.download("vader_lexicon", quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer

TICKER_TO_COMPANY = {
    "AAPL":  "Apple",
    "TSLA":  "Tesla",
    "MSFT":  "Microsoft",
    "GOOGL": "Google",
    "AMZN":  "Amazon",
    "NVDA":  "Nvidia",
    "META":  "Meta",
    "NFLX":  "Netflix",
    "AMD":   "AMD",
    "BABA":  "Alibaba",
}


def fetch_news_gnews(ticker: str) -> list:
    """
    Fetch up to 10 recent headlines for a ticker from GNews.
    Free plan: 100 requests/day — called for each ticker every 30 min.
    10 tickers × 48 runs/day = 480 req — stay within limits by
    running the scheduler at 30-min intervals BUT only for 10 tickers
    per run (see scheduler.py for the daily budget guard).
    """
    company = TICKER_TO_COMPANY.get(ticker, ticker)
    url = "https://gnews.io/api/v4/search"
    params = {
        "q":       f"{company} stock",
        "lang":    "en",
        "country": "us",
        "max":     10,
        "sortby":  "publishedAt",
        "apikey":  GNEWS_KEY,
    }

    try:
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        data = resp.json()

        if "errors" in data:
            print(f"  [WARN] GNews error for {ticker}: {data['errors']}")
            return []

        articles = data.get("articles", [])
        print(f"  [OK] {len(articles)} articles for {ticker}")
        return articles

    except requests.exceptions.RequestException as e:
        print(f"  [ERROR] GNews network error for {ticker}: {e}")
        return []


def parse_and_score(ticker: str, articles: list,
                    analyzer: SentimentIntensityAnalyzer) -> pd.DataFrame:
    """
    Parse GNews articles, score with VADER + Groq in one pass.
    Deduplication guard: MD5 hash of (ticker|date|headline).
    """
    if not articles:
        return pd.DataFrame()

    seen = set()
    rows = []

    for article in articles:
        published = article.get("publishedAt", "")
        try:
            pub_date = datetime.strptime(published[:10], "%Y-%m-%d").date()
        except (ValueError, TypeError):
            continue

        headline = (article.get("title", "") or "").strip()
        if not headline or headline == "[Removed]":
            continue
        headline = headline[:500]

        # In-memory dedup before hitting DB
        h = hashlib.md5(f"{ticker}|{pub_date}|{headline}".encode()).hexdigest()
        if h in seen:
            continue
        seen.add(h)

        score, label = analyze_sentiment(headline, analyzer)

        rows.append({
            "ticker":          ticker,
            "date":            pub_date,
            "headline":        headline,
            "sentiment_score": score,
            "sentiment_label": label,
            "source":          (article.get("source", {}).get("name", "Unknown") or "Unknown")[:100],
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # Score all headlines for this ticker with FinBERT (local, free)
    df["groq_sentiment"] = finbert_score(df["headline"].tolist())

    return df


def save_to_db(df: pd.DataFrame, conn) -> int:
    """
    Insert rows with VADER + Groq sentiment scores.
    ON CONFLICT (ticker, date, headline) DO NOTHING — DB-level dedup.
    """
    if df.empty:
        return 0

    rows = [
        (
            row["ticker"], row["date"], row["headline"],
            row["sentiment_score"], row["sentiment_label"], row["source"],
            row.get("groq_sentiment"),
        )
        for _, row in df.iterrows()
    ]

    sql = """
        INSERT INTO news_sentiment
            (ticker, date, headline, sentiment_score, sentiment_label, source, groq_sentiment)
        VALUES %s
        ON CONFLICT (ticker, date, headline) DO NOTHING
    """
    with conn.cursor() as cur:
        execute_values(cur, sql, rows)
    conn.commit()
    return len(rows)


def collect_all_news():
    """
    Main entry point — called every 30 min by the scheduler.
    """
    print("=" * 55)
    print("GNews Live Collector")
    print(f"Tickers : {STOCKS}")
    print("=" * 55)

    conn          = get_connection()
    analyzer      = SentimentIntensityAnalyzer()
    total_saved   = 0
    total_fetched = 0

    for ticker in STOCKS:
        articles = fetch_news_gnews(ticker)
        total_fetched += len(articles)

        df = parse_and_score(ticker, articles, analyzer)
        saved = save_to_db(df, conn)
        total_saved += saved
        print(f"  {ticker}: {saved} new articles saved")
        time.sleep(1)  # avoid GNews rate limit

    conn.close()

    # ── Monitoring ─────────────────────────────
    log_run(
        source="gnews",
        articles_collected=total_saved,
        requests_made=len(STOCKS),
    )

    print(f"\nDone. Fetched {total_fetched}, saved {total_saved} new articles.")
    print("=" * 55)


def fetch_news_gnews_range(ticker: str, from_dt: str, to_dt: str,
                           max_articles: int = 100) -> list:
    """
    Fetch articles for a specific date range using GNews Essential `from`/`to` params.
    from_dt / to_dt must be ISO-8601: "YYYY-MM-DDTHH:MM:SSZ"
    """
    company = TICKER_TO_COMPANY.get(ticker, ticker)
    url = "https://gnews.io/api/v4/search"
    params = {
        "q":       f'"{company}" stock',
        "lang":    "en",
        "country": "us",
        "max":     max_articles,
        "sortby":  "publishedAt",
        "from":    from_dt,
        "to":      to_dt,
        "apikey":  GNEWS_KEY,
    }
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if "errors" in data:
            print(f"  [WARN] GNews error for {ticker}: {data['errors']}")
            return []
        return data.get("articles", [])
    except requests.exceptions.RequestException as e:
        print(f"  [ERROR] GNews range fetch for {ticker} [{from_dt[:10]}]: {e}")
        return []


def _week_already_filled(conn, ticker: str, week_start: date, week_end: date) -> bool:
    """Return True if we already have ≥1 article for this ticker/week in the DB."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM news_sentiment "
            "WHERE ticker=%s AND date BETWEEN %s AND %s LIMIT 1",
            (ticker, week_start, week_end),
        )
        return cur.fetchone() is not None


def backfill_gnews(
    start:       str   = "2021-01-01",
    end:         str   = "2023-12-31",
    tickers:     list  = None,
    chunk_days:  int   = 7,
    delay:       float = 1.5,
    daily_limit: int   = 950,
    skip_filled: bool  = True,
):
    """
    Backfill GNews articles week-by-week between `start` and `end`.

    Requires GNews Essential plan (historical data from 2020, 1,000 req/day).
    Uses ON CONFLICT DO NOTHING so re-running is safe.
    `skip_filled=True` skips weeks that already have ≥1 article in the DB,
    saving API quota when resuming an interrupted run.
    """
    tickers  = tickers or STOCKS
    conn     = get_connection()
    analyzer = SentimentIntensityAnalyzer()

    start_d = date.fromisoformat(start)
    end_d   = date.fromisoformat(end)

    chunks: list[tuple[date, date]] = []
    cur = start_d
    while cur <= end_d:
        chunk_end = min(cur + timedelta(days=chunk_days - 1), end_d)
        chunks.append((cur, chunk_end))
        cur = chunk_end + timedelta(days=1)

    total_req   = 0
    total_saved = 0
    skipped     = 0

    print("=" * 60)
    print("GNews Historical Backfill")
    print(f"  Range   : {start} to {end}")
    print(f"  Tickers : {tickers}")
    print(f"  Chunks  : {len(chunks)} weeks × {len(tickers)} tickers "
          f"= {len(chunks) * len(tickers)} max requests")
    print(f"  Limit   : {daily_limit} req/run  (~{len(chunks)*len(tickers)/daily_limit:.1f} days)")
    print("=" * 60)

    for week_start, week_end in chunks:
        from_dt = f"{week_start}T00:00:00Z"
        to_dt   = f"{week_end}T23:59:59Z"

        for ticker in tickers:
            if total_req >= daily_limit:
                print(f"\n[LIMIT] Reached {daily_limit} requests. "
                      f"Re-run tomorrow to continue from {week_start}.")
                conn.close()
                print(f"\nSaved {total_saved} new articles, skipped {skipped} filled weeks.")
                return total_saved

            if skip_filled and _week_already_filled(conn, ticker, week_start, week_end):
                skipped += 1
                continue

            articles = fetch_news_gnews_range(ticker, from_dt, to_dt)
            total_req += 1

            if articles:
                df   = parse_and_score(ticker, articles, analyzer)
                saved = save_to_db(df, conn)
                total_saved += saved
                print(f"  {ticker} [{week_start} to {week_end}]: "
                      f"{len(articles)} fetched, {saved} new saved")
            else:
                print(f"  {ticker} [{week_start} to {week_end}]: 0 articles")

            time.sleep(delay)

    conn.close()
    print(f"\nBackfill complete. Requests: {total_req}, "
          f"New articles: {total_saved}, Weeks skipped: {skipped}")
    return total_saved


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GNews collector / backfill")
    parser.add_argument("--backfill",    action="store_true",
                        help="Run historical backfill instead of live collection")
    parser.add_argument("--start",  default="2021-01-01",
                        help="Backfill start date (default: 2021-01-01)")
    parser.add_argument("--end",    default="2023-12-31",
                        help="Backfill end date   (default: 2023-12-31)")
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Tickers to backfill (default: all STOCKS)")
    parser.add_argument("--no-skip", action="store_true",
                        help="Re-fetch even weeks already in DB")
    args = parser.parse_args()

    if args.backfill:
        backfill_gnews(
            start=args.start,
            end=args.end,
            tickers=args.tickers,
            skip_filled=not args.no_skip,
        )
    else:
        collect_all_news()
