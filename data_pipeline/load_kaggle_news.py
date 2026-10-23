# data_pipeline/load_kaggle_news.py
# ─────────────────────────────────────────────
# Loads financial news from Kaggle CSV dataset
# filters for our 10 stocks, runs VADER sentiment,
# and saves to the news_sentiment table
# ─────────────────────────────────────────────

import pandas as pd
import sys
import os
from psycopg2.extras import execute_values

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import STOCKS
from data_pipeline.db_connection import get_connection

import nltk
nltk.download("vader_lexicon", quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer

CSV_PATH = "data/analyst_ratings_processed.csv"


def load_and_filter_csv() -> pd.DataFrame:
    """
    Load the Kaggle CSV and filter for our 10 stocks.
    """
    print(f"Loading {CSV_PATH}...")
    print("(This may take 30 seconds — 1.4 million rows)\n")

    df = pd.read_csv(
        CSV_PATH,
        usecols=["title", "date", "stock"],
        dtype={"title": str, "stock": str},
    )

    # Filter for only our stocks
    df = df[df["stock"].isin(STOCKS)].copy()
    print(f"Found {len(df):,} articles for our 10 stocks")

    # Parse dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.date
    df = df.dropna(subset=["date", "title"])

    # Filter date range 2020-2023
    df = df[(df["date"] >= pd.Timestamp("2020-01-01").date()) &
            (df["date"] <= pd.Timestamp("2023-12-31").date())]

    # Clean up
    df = df.rename(columns={"title": "headline", "stock": "ticker"})
    df = df[["ticker", "date", "headline"]]
    df["headline"] = df["headline"].str.strip().str[:500]
    df = df[df["headline"].str.len() > 5]
    df = df.drop_duplicates(subset=["ticker", "date", "headline"])

    print(f"After filtering (2020-2023): {len(df):,} articles\n")
    return df


def score_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Run VADER sentiment on all headlines."""
    print("Running VADER sentiment analysis...")
    analyzer = SentimentIntensityAnalyzer()

    scores = df["headline"].apply(
        lambda text: analyzer.polarity_scores(text)["compound"]
    )
    df["sentiment_score"] = scores.round(4)
    df["sentiment_label"] = df["sentiment_score"].apply(
        lambda s: "positive" if s >= 0.05 else ("negative" if s <= -0.05 else "neutral")
    )

    pos = (df["sentiment_label"] == "positive").sum()
    neg = (df["sentiment_label"] == "negative").sum()
    neu = (df["sentiment_label"] == "neutral").sum()
    print(f"Scored {len(df):,} headlines → Positive: {pos:,} | Negative: {neg:,} | Neutral: {neu:,}\n")
    return df


def save_to_db_chunked(df: pd.DataFrame, conn, chunk_size: int = 1000) -> int:
    """
    Insert rows in chunks to avoid memory issues.
    ON CONFLICT DO NOTHING skips duplicates.
    """
    sql = """
        INSERT INTO news_sentiment (ticker, date, headline, sentiment_score, sentiment_label, source)
        VALUES %s
        ON CONFLICT (ticker, date, headline) DO NOTHING
    """

    total   = 0
    chunks  = [df.iloc[i:i+chunk_size] for i in range(0, len(df), chunk_size)]
    n       = len(chunks)

    for i, chunk in enumerate(chunks, 1):
        rows = [
            (row["ticker"], row["date"], row["headline"],
             row["sentiment_score"], row["sentiment_label"], "Kaggle")
            for _, row in chunk.iterrows()
        ]
        with conn.cursor() as cur:
            execute_values(cur, sql, rows)
        conn.commit()
        total += len(rows)

        # Progress every 10 chunks
        if i % 10 == 0 or i == n:
            print(f"  Inserted chunk {i}/{n} — {total:,} rows so far...")

    return total


def load_kaggle_news():
    """Main function."""
    print("=" * 50)
    print("Loading Kaggle News Dataset → PostgreSQL")
    print("=" * 50 + "\n")

    # Step 1: Load and filter CSV
    df = load_and_filter_csv()

    if df.empty:
        print("[ERROR] No matching articles found. Check your STOCKS list.")
        return

    # Show breakdown per ticker
    print("Articles per stock:")
    for ticker, count in df.groupby("ticker").size().items():
        print(f"  {ticker:<6} : {count:,}")
    print()

    # Step 2: Run sentiment
    df = score_sentiment(df)

    # Step 3: Save to DB
    print("Saving to PostgreSQL...")
    conn  = get_connection()
    total = save_to_db_chunked(df, conn)
    conn.close()

    print(f"\n{'='*50}")
    print(f"Done! {total:,} headlines saved to news_sentiment table")
    print(f"{'='*50}")
    print("\nNext: run feature engineering to merge sentiment with stock prices")


if __name__ == "__main__":
    load_kaggle_news()