# data_pipeline/backfill_sentiment.py
# ─────────────────────────────────────────────
# One-time script — rescores all existing news articles
# in the DB with FinBERT (local, free, no API limits).
#
# Run once:  python data_pipeline/backfill_sentiment.py
# ─────────────────────────────────────────────

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_pipeline.db_connection import get_connection
from data_pipeline.finbert_sentiment import score_headlines

BATCH_SIZE = 64    # FinBERT can handle large batches locally


def run_backfill():
    print("=" * 55)
    print("FinBERT Sentiment Backfill")
    print(f"Batch  : {BATCH_SIZE} articles per pass")
    print("=" * 55)

    conn = get_connection()

    # Add column if it doesn't exist
    with conn.cursor() as cur:
        cur.execute("""
            ALTER TABLE news_sentiment
            ADD COLUMN IF NOT EXISTS groq_sentiment NUMERIC(5, 4)
        """)
    conn.commit()
    print("  [OK] groq_sentiment column ready")

    # Fetch unscored articles (all dates — FinBERT has no API limits)
    with conn.cursor() as cur:
        cur.execute("""
            SELECT ticker, headline, date
            FROM news_sentiment
            WHERE groq_sentiment IS NULL
            ORDER BY date DESC
        """)
        rows = cur.fetchall()

    total = len(rows)
    if total == 0:
        print("  All articles already scored. Nothing to do.")
        conn.close()
        return

    print(f"  Articles to score : {total}\n")

    scored = 0
    for i in range(0, total, BATCH_SIZE):
        batch     = rows[i : i + BATCH_SIZE]
        headlines = [row[1] for row in batch]
        scores    = score_headlines(headlines)

        with conn.cursor() as cur:
            for (ticker, headline, date), score in zip(batch, scores):
                cur.execute(
                    """
                    UPDATE news_sentiment
                    SET groq_sentiment = %s
                    WHERE ticker = %s AND date = %s AND headline = %s
                    """,
                    (score, ticker, date, headline),
                )
        conn.commit()

        scored += len(batch)
        pct = scored / total * 100
        print(f"  [{pct:5.1f}%]  {scored}/{total} scored")

    conn.close()
    print(f"\n  Done! {scored} articles scored.")
    print("=" * 55)


if __name__ == "__main__":
    run_backfill()
