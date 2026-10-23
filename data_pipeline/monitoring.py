# data_pipeline/monitoring.py
# ─────────────────────────────────────────────
# Pipeline monitoring — tracks every collector run.
#
# Stored metrics (pipeline_monitor table):
#   requests_made      — API calls sent
#   articles_collected — new rows written to news_sentiment
#   windows_processed  — month-windows fetched (alpha only)
#
# Daily summary shows:
#   • Per-source request / article totals
#   • Alpha backfill progress (months covered out of 24)
#   • GNews activity over the last 7 days
# ─────────────────────────────────────────────

import sys
import os
from datetime import date, datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_pipeline.db_connection import get_connection


# ─────────────────────────────────────────────
# Schema bootstrap
# ─────────────────────────────────────────────

_SCHEMA_CREATED = False

def _ensure_schema():
    global _SCHEMA_CREATED
    if _SCHEMA_CREATED:
        return
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS pipeline_monitor (
                id                 SERIAL PRIMARY KEY,
                run_at             TIMESTAMP DEFAULT NOW(),
                source             VARCHAR(30)  NOT NULL,
                articles_collected INT          DEFAULT 0,
                requests_made      INT          DEFAULT 0,
                windows_processed  INT          DEFAULT 0,
                notes              TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_monitor_source ON pipeline_monitor(source);
            CREATE INDEX IF NOT EXISTS idx_monitor_run_at ON pipeline_monitor(run_at);
        """)
    conn.commit()
    conn.close()
    _SCHEMA_CREATED = True


# ─────────────────────────────────────────────
# Log a run
# ─────────────────────────────────────────────

def log_run(source: str, articles_collected: int = 0,
            requests_made: int = 0, windows_processed: int = 0,
            notes: str = None):
    """Insert one monitoring row after each collector run."""
    _ensure_schema()
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO pipeline_monitor
                (source, articles_collected, requests_made, windows_processed, notes)
            VALUES (%s, %s, %s, %s, %s)
        """, (source, articles_collected, requests_made, windows_processed, notes))
    conn.commit()
    conn.close()
    print(f"  [MONITOR] {source}: +{articles_collected} articles, "
          f"{requests_made} requests, {windows_processed} windows")


# ─────────────────────────────────────────────
# Budget guard
# ─────────────────────────────────────────────

def requests_today(source: str) -> int:
    """Total API requests already logged today for a source."""
    _ensure_schema()
    conn = get_connection()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT COALESCE(SUM(requests_made), 0)
            FROM pipeline_monitor
            WHERE source = %s AND DATE(run_at) = CURRENT_DATE
        """, (source,))
        total = cur.fetchone()[0]
    conn.close()
    return int(total)


# ─────────────────────────────────────────────
# Daily summary
# ─────────────────────────────────────────────

def daily_summary(target_date: date = None):
    """Print a full monitoring report for the given date (default: today)."""
    if target_date is None:
        target_date = date.today()

    _ensure_schema()
    conn = get_connection()

    print("\n" + "=" * 62)
    print(f"  Pipeline Monitor — {target_date}")
    print("=" * 62)

    # ── Per-source totals ──────────────────────
    with conn.cursor() as cur:
        cur.execute("""
            SELECT source,
                   COUNT(*)                AS runs,
                   SUM(requests_made)      AS total_requests,
                   SUM(articles_collected) AS total_articles
            FROM pipeline_monitor
            WHERE DATE(run_at) = %s
            GROUP BY source ORDER BY source
        """, (target_date,))
        rows = cur.fetchall()

    if rows:
        print(f"\n  {'Source':<18} {'Runs':>5} {'Requests':>10} {'Articles':>10}")
        print(f"  {'-'*18} {'-'*5} {'-'*10} {'-'*10}")
        for source, runs, reqs, arts in rows:
            print(f"  {source:<18} {runs:>5} {reqs:>10} {arts:>10}")
    else:
        print("\n  No runs recorded today.")

    # ── Alpha backfill progress (months, not days) ──
    from config.settings import STOCKS, ALPHA_BACKFILL_START, ALPHA_BACKFILL_END
    from calendar import monthrange

    start = datetime.strptime(ALPHA_BACKFILL_START, "%Y-%m-%d").date()
    end   = datetime.strptime(ALPHA_BACKFILL_END,   "%Y-%m-%d").date()

    # Count total months in range
    total_months = (end.year - start.year) * 12 + (end.month - start.month) + 1

    print(f"\n  Alpha Vantage backfill progress ({total_months} months total):")

    for ticker in STOCKS:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(DISTINCT
                    DATE_TRUNC('month', date)
                )
                FROM news_sentiment
                WHERE ticker = %s
                  AND source ILIKE 'AlphaVantage%%'
                  AND date BETWEEN %s AND %s
            """, (ticker, start, end))
            covered_months = cur.fetchone()[0]

        missing = total_months - covered_months
        pct     = (covered_months / total_months * 100) if total_months else 0
        filled  = int(pct / 5)
        bar     = "█" * filled + "░" * (20 - filled)
        print(f"  {ticker:<6} [{bar}] {pct:5.1f}%  "
              f"({covered_months}/{total_months} months, {missing} remaining)")

    # ── GNews recent activity ──────────────────
    week_ago = target_date - timedelta(days=7)
    print(f"\n  GNews articles (last 7 days):")
    for ticker in STOCKS:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT COUNT(*)
                FROM news_sentiment
                WHERE ticker = %s
                  AND source NOT ILIKE 'AlphaVantage%%'
                  AND date >= %s
            """, (ticker, week_ago))
            count = cur.fetchone()[0]
        bar   = "▪" * min(count, 30)
        print(f"  {ticker:<6} {bar:<30} {count:>3} articles")

    # ── All-time totals ────────────────────────
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                COUNT(*) FILTER (WHERE source ILIKE 'AlphaVantage%%') AS alpha_total,
                COUNT(*) FILTER (WHERE source NOT ILIKE 'AlphaVantage%%'
                                   AND source NOT ILIKE 'Kaggle%%')    AS gnews_total,
                COUNT(*) FILTER (WHERE source ILIKE 'Kaggle%%')         AS kaggle_total,
                COUNT(*)                                                 AS grand_total
            FROM news_sentiment
        """)
        row = cur.fetchone()

    print(f"\n  All-time totals in news_sentiment:")
    print(f"  {'Kaggle':<14} {row[2]:>8,} articles")
    print(f"  {'Alpha Vantage':<14} {row[0]:>8,} articles")
    print(f"  {'GNews':<14} {row[1]:>8,} articles")
    print(f"  {'TOTAL':<14} {row[3]:>8,} articles")

    conn.close()
    print("\n" + "=" * 62 + "\n")


if __name__ == "__main__":
    daily_summary()