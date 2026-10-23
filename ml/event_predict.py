# ml/event_predict.py
# ─────────────────────────────────────────────
# Event-driven prediction trigger.
#
# Called after every GNews fetch. If today's news exists for a ticker:
#   1. Compute avg FinBERT sentiment for today's articles
#   2. Patch sentiment_today in the features table (latest row for that ticker)
#   3. Re-run predictions for that ticker and UPSERT into predictions table
#
# This means a CEO sacking, fraud, or major event picked up by GNews
# will update the prediction within 30 minutes — not the next day.
# ─────────────────────────────────────────────

import os
import sys
import pickle
from datetime import date

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_pipeline.db_connection import get_connection
from ml.predict import load_latest_features, load_best_model, HORIZONS


def _fresh_conn():
    """Return a new DB connection, discarding any stale one."""
    return get_connection()


def trigger_event_predictions():
    """
    Query today's news sentiment per ticker.
    For any ticker that has articles today, patch sentiment_today and re-predict.
    Only updates predictions that haven't been resolved yet (actual_label IS NULL).
    """
    today = date.today()

    # Use a short-lived connection just for the initial SELECT
    conn = _fresh_conn()
    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                ticker,
                COUNT(*)                                          AS article_count,
                AVG(COALESCE(groq_sentiment, sentiment_score, 0)) AS avg_sentiment
            FROM news_sentiment
            WHERE date = %s
            GROUP BY ticker
        """, (today,))
        ticker_rows = cur.fetchall()
    conn.close()

    if not ticker_rows:
        return

    triggered = []

    for ticker, count, avg_sent in ticker_rows:
        if avg_sent is None:
            continue

        avg_sent = round(float(avg_sent), 4)

        # Fresh connection per ticker — avoids Neon SSL drop between queries
        conn = _fresh_conn()

        # Patch sentiment_today on the most recent features row for this ticker
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE features
                SET sentiment_today = %s
                WHERE ticker = %s
                  AND date = (
                      SELECT MAX(date) FROM features WHERE ticker = %s
                  )
            """, (avg_sent, ticker, ticker))

        conn.commit()

        # Re-load features (now includes updated sentiment_today)
        row = load_latest_features(conn, ticker)
        if row is None:
            continue

        pred_date  = row["date"]
        pred_close = float(row["raw_close"])
        updated    = 0

        for horizon, horizon_days in HORIZONS.items():
            model, features, source = load_best_model(ticker, horizon)
            if model is None:
                continue

            X     = pd.DataFrame([{f: row[f] for f in features}]).astype(float)
            label = int(model.predict(X)[0])
            prob  = float(model.predict_proba(X)[0][1])

            # UPSERT: update existing prediction unless it's already resolved
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO predictions
                        (ticker, horizon, horizon_days, prediction_date,
                         predicted_label, predicted_prob, prediction_close, model_source)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (ticker, horizon, prediction_date) DO UPDATE SET
                        predicted_label = EXCLUDED.predicted_label,
                        predicted_prob  = EXCLUDED.predicted_prob,
                        model_source    = EXCLUDED.model_source
                    WHERE predictions.actual_label IS NULL
                """, (
                    ticker, horizon, horizon_days, pred_date,
                    label, round(prob, 4), pred_close, source,
                ))
            updated += 1

        conn.commit()

        direction_1d = None
        for horizon, horizon_days in HORIZONS.items():
            if horizon == "1d":
                model, features, source = load_best_model(ticker, horizon)
                if model:
                    X = pd.DataFrame([{f: row[f] for f in features}]).astype(float)
                    label = int(model.predict(X)[0])
                    prob  = float(model.predict_proba(X)[0][1])
                    direction_1d = f"{'UP' if label == 1 else 'DOWN'} ({prob*100:.0f}%)"
                break

        conn.close()

        triggered.append(
            f"  {ticker:<6} {count} articles  sentiment={avg_sent:+.3f}"
            + (f"  → 1d: {direction_1d}" if direction_1d else "")
        )

    if triggered:
        print("[EVENT] Sentiment-triggered prediction update:")
        for line in triggered:
            print(line)
