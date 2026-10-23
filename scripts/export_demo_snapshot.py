"""Export a public, static dashboard snapshot from the configured database."""

import json
import math
import sys
from datetime import date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from config.settings import STOCKS
from data_pipeline.db_connection import get_connection
from ml.explain import explain_prediction_for_api

HORIZONS = ("1d", "5d", "21d", "252d")
OUTPUT_PATH = ROOT / "frontend" / "public" / "demo-data.json"


def _json_default(value):
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    raise TypeError(f"Unsupported JSON value: {type(value).__name__}")


def _clean_non_finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _clean_non_finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clean_non_finite(item) for item in value]
    return value


def export_snapshot():
    snapshot = {
        "metadata": {
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "price_as_of": None,
            "news_as_of": None,
        },
        "tickers": {},
    }

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT MAX(date) FROM stock_prices")
            snapshot["metadata"]["price_as_of"] = cur.fetchone()[0]
            cur.execute("SELECT MAX(date) FROM news_sentiment")
            snapshot["metadata"]["news_as_of"] = cur.fetchone()[0]

            for ticker in STOCKS:
                cur.execute(
                    """
                    SELECT date, open, high, low, close, volume
                    FROM (
                        SELECT date, open, high, low, close, volume
                        FROM stock_prices
                        WHERE ticker = %s
                        ORDER BY date DESC
                        LIMIT 252
                    ) recent
                    ORDER BY date
                    """,
                    (ticker,),
                )
                prices = [
                    {
                        "date": row[0].isoformat(),
                        "open": float(row[1]),
                        "high": float(row[2]),
                        "low": float(row[3]),
                        "close": float(row[4]),
                        "volume": int(row[5]),
                    }
                    for row in cur.fetchall()
                ]
                if not prices:
                    raise RuntimeError(f"No price data found for {ticker}")

                cur.execute(
                    """
                    WITH recent AS (
                        SELECT date, sentiment_score
                        FROM news_sentiment
                        WHERE ticker = %s AND sentiment_score IS NOT NULL
                        ORDER BY date DESC
                        LIMIT 15
                    )
                    SELECT date, AVG(sentiment_score), COUNT(*)
                    FROM recent
                    GROUP BY date
                    ORDER BY date DESC
                    """,
                    (ticker,),
                )
                sentiment_days = [
                    {
                        "date": row[0].isoformat(),
                        "average_sentiment": round(float(row[1]), 4),
                        "count": int(row[2]),
                    }
                    for row in cur.fetchall()
                ]
                sentiment_count = sum(day["count"] for day in sentiment_days)
                sentiment_average = (
                    sum(day["average_sentiment"] * day["count"] for day in sentiment_days)
                    / sentiment_count
                    if sentiment_count
                    else 0.0
                )

                predictions = {}
                for horizon in HORIZONS:
                    predictions[horizon] = {}
                    for use_sentiment, key in (
                        (False, "no_sentiment"),
                        (True, "with_sentiment"),
                    ):
                        result = explain_prediction_for_api(
                            ticker, use_sentiment=use_sentiment, horizon=horizon
                        )
                        if "error" in result:
                            raise RuntimeError(
                                f"Could not build {ticker} {horizon} prediction: "
                                f"{result['error']}"
                            )
                        predictions[horizon][key] = result
                    print(f"Prepared {ticker} {horizon}")

                snapshot["tickers"][ticker] = {
                    "prices": prices,
                    "sentiment": {
                        "count": sentiment_count,
                        "average_sentiment": round(sentiment_average, 4),
                        "sentiment_days": sentiment_days,
                    },
                    "predictions": predictions,
                }
    finally:
        conn.close()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp_path = OUTPUT_PATH.with_suffix(".json.tmp")
    with temp_path.open("w", encoding="utf-8") as output:
        json.dump(
            _clean_non_finite(snapshot),
            output,
            indent=2,
            default=_json_default,
            allow_nan=False,
        )
    temp_path.replace(OUTPUT_PATH)
    print(f"Snapshot written to {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    export_snapshot()
