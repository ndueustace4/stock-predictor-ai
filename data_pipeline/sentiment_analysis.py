# data_pipeline/sentiment_analysis.py
# ─────────────────────────────────────────────
# Runs VADER sentiment analysis on all headlines
# in the news_sentiment table that don't have
# a sentiment score yet, then updates the DB.
# ─────────────────────────────────────────────

import sys
import os
import pandas as pd
from psycopg2.extras import execute_values

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_pipeline.db_connection import get_connection

# Download VADER lexicon on first run
import nltk
nltk.download("vader_lexicon", quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer


def get_unscored_headlines(conn) -> pd.DataFrame:
    """
    Fetch all headlines that don't have a sentiment score yet.
    """
    sql = """
        SELECT id, ticker, date, headline
        FROM news_sentiment
        WHERE sentiment_score IS NULL
        ORDER BY date DESC
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()

    df = pd.DataFrame(rows, columns=["id", "ticker", "date", "headline"])
    return df


def analyze_sentiment(text: str, analyzer: SentimentIntensityAnalyzer) -> tuple:
    """
    Run VADER on a single headline.
    Returns (score, label) where:
      score  = compound score between -1.0 and 1.0
      label  = 'positive', 'negative', or 'neutral'
    
    VADER compound score thresholds:
      >= 0.05  → positive
      <= -0.05 → negative
      between  → neutral
    """
    scores = analyzer.polarity_scores(text)
    compound = round(scores["compound"], 4)

    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"

    return compound, label


def update_sentiment_in_db(results: list, conn):
    """
    Bulk update sentiment scores and labels in the DB.
    results = list of (score, label, id) tuples
    """
    sql = """
        UPDATE news_sentiment
        SET sentiment_score = data.score,
            sentiment_label = data.label
        FROM (VALUES %s) AS data(score, label, id)
        WHERE news_sentiment.id = data.id::int
    """
    with conn.cursor() as cur:
        execute_values(cur, sql, results)
    conn.commit()


def run_sentiment_analysis():
    """
    Main function: score all unscored headlines and update the DB.
    """
    print("=" * 50)
    print("Phase 2 — Sentiment Analysis (VADER)")
    print("=" * 50)

    conn     = get_connection()
    analyzer = SentimentIntensityAnalyzer()

    # Get unscored headlines
    df = get_unscored_headlines(conn)

    if df.empty:
        print("No unscored headlines found. All sentiment scores are up to date!")
        conn.close()
        return

    print(f"Found {len(df)} headlines to score...\n")

    results = []
    pos_count = neg_count = neu_count = 0

    for _, row in df.iterrows():
        score, label = analyze_sentiment(row["headline"], analyzer)
        results.append((score, label, row["id"]))

        if label == "positive": pos_count += 1
        elif label == "negative": neg_count += 1
        else: neu_count += 1

    # Bulk update DB
    update_sentiment_in_db(results, conn)
    conn.close()

    print(f"Scored {len(results)} headlines:")
    print(f"  Positive : {pos_count}")
    print(f"  Negative : {neg_count}")
    print(f"  Neutral  : {neu_count}")
    print("\n" + "=" * 50)
    print("Sentiment analysis complete!")
    print("=" * 50)


def show_sample_results():
    """
    Print a sample of scored headlines from the DB.
    """
    conn = get_connection()
    sql = """
        SELECT ticker, date, sentiment_score, sentiment_label, headline
        FROM news_sentiment
        WHERE sentiment_score IS NOT NULL
        ORDER BY date DESC, ABS(sentiment_score) DESC
        LIMIT 10
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
    conn.close()

    print("\nSample scored headlines:")
    print("-" * 80)
    for row in rows:
        ticker, date, score, label, headline = row
        print(f"[{ticker}] {date} | {label:8s} ({score:+.4f}) | {headline[:60]}...")


if __name__ == "__main__":
    run_sentiment_analysis()
    show_sample_results()