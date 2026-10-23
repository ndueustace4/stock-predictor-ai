# ml/backfill_predictions.py
# Backfill predictions for dates where the scheduler was not running.
# Usage: python ml/backfill_predictions.py

import os
import sys
import pickle
from datetime import date

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_pipeline.db_connection import get_connection
from config.settings import STOCKS
from ml.predict import (
    HORIZONS, FEATURES_WITH_SENT, FEATURES_NO_SENT,
    FEATURES_252D_WITH_SENT,
    init_table, load_best_model, resolve_actuals, show_report
)

BACKFILL_START = date(2026, 5, 16)
BACKFILL_END   = date(2026, 5, 25)


def load_features_for_date(conn, ticker: str, pred_date: date) -> dict | None:
    sql = """
        SELECT
            close, volume, ma_5, ma_10, ma_20,
            volatility_7, momentum_5,
            COALESCE(price_lag1,  0) AS price_lag1,
            COALESCE(price_lag2,  0) AS price_lag2,
            COALESCE(price_lag3,  0) AS price_lag3,
            COALESCE(price_lag5,  0) AS price_lag5,
            COALESCE(rsi_14,      0) AS rsi_14,
            COALESCE(macd,        0) AS macd,
            COALESCE(macd_signal, 0) AS macd_signal,
            COALESCE(bollinger_width,  0) AS bollinger_width,
            COALESCE(volume_change,    0) AS volume_change,
            COALESCE(price_ma_ratio,   1) AS price_ma_ratio,
            COALESCE(return_1d,   0) AS return_1d,
            COALESCE(return_5d,   0) AS return_5d,
            COALESCE(return_10d,  0) AS return_10d,
            COALESCE(day_of_week, 0) AS day_of_week,
            COALESCE(atr_14,      0) AS atr_14,
            COALESCE(high_20d_ratio, 1) AS high_20d_ratio,
            COALESCE(low_20d_ratio,  1) AS low_20d_ratio,
            COALESCE(spy_return_1d,  0) AS spy_return_1d,
            COALESCE(qqq_return_1d,  0) AS qqq_return_1d,
            COALESCE(vix_level,  0) AS vix_level,
            COALESCE(vix_change, 0) AS vix_change,
            COALESCE(hsi_return_1d,  0) AS hsi_return_1d,
            COALESCE(mchi_return_1d, 0) AS mchi_return_1d,
            COALESCE(cny_usd_return, 0) AS cny_usd_return,
            COALESCE(xlk_return_1d,  0) AS xlk_return_1d,
            COALESCE(xlf_return_1d,  0) AS xlf_return_1d,
            COALESCE(xle_return_1d,  0) AS xle_return_1d,
            COALESCE(xlv_return_1d,  0) AS xlv_return_1d,
            COALESCE(xli_return_1d,  0) AS xli_return_1d,
            COALESCE(earnings_surprise, 0) AS earnings_surprise,
            COALESCE(price_ma252_ratio, 1) AS price_ma252_ratio,
            COALESCE(sentiment_today, 0) AS sentiment_today,
            COALESCE(sentiment_lag1,  0) AS sentiment_lag1,
            COALESCE(sentiment_lag2,  0) AS sentiment_lag2,
            COALESCE(sentiment_ma_7,  0) AS sentiment_ma_7,
            COALESCE(sentiment_std_7, 0) AS sentiment_std_7,
            COALESCE(sentiment_momentum, 0) AS sentiment_momentum,
            COALESCE(pe_ratio,           0) AS pe_ratio,
            COALESCE(revenue_growth_yoy, 0) AS revenue_growth_yoy,
            COALESCE(eps_growth_yoy,     0) AS eps_growth_yoy,
            COALESCE(treasury_10y,       0) AS treasury_10y,
            COALESCE(treasury_2y,        0) AS treasury_2y,
            COALESCE(yield_curve,        0) AS yield_curve,
            date, close AS raw_close
        FROM features
        WHERE ticker = %s
          AND date = %s
          AND close IS NOT NULL
          AND ma_20  IS NOT NULL
    """
    with conn.cursor() as cur:
        cur.execute(sql, (ticker, pred_date))
        row = cur.fetchone()

    if not row:
        return None

    cols = FEATURES_252D_WITH_SENT + ["date", "raw_close"]
    return dict(zip(cols, row))


def backfill():
    conn = get_connection()
    init_table(conn)

    # Get all trading days in the backfill range that exist in the features table
    with conn.cursor() as cur:
        cur.execute("""
            SELECT DISTINCT date FROM features
            WHERE date >= %s AND date <= %s
            ORDER BY date ASC
        """, (BACKFILL_START, BACKFILL_END))
        trading_days = [r[0] for r in cur.fetchall()]

    print("=" * 55)
    print(f"Backfilling predictions: {BACKFILL_START} to {BACKFILL_END}")
    print(f"Trading days found: {len(trading_days)}")
    print("=" * 55)

    saved = 0
    skipped = 0

    for pred_date in trading_days:
        for ticker in sorted(STOCKS):
            row = load_features_for_date(conn, ticker, pred_date)
            if row is None:
                continue

            pred_close = float(row["raw_close"])

            for horizon, horizon_days in HORIZONS.items():
                model, features, source = load_best_model(ticker, horizon)
                if model is None:
                    continue

                X = pd.DataFrame([{f: row[f] for f in features}]).astype(float)
                label = int(model.predict(X)[0])
                prob  = float(model.predict_proba(X)[0][1])

                sql = """
                    INSERT INTO predictions
                        (ticker, horizon, horizon_days, prediction_date,
                         predicted_label, predicted_prob, prediction_close, model_source)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (ticker, horizon, prediction_date) DO NOTHING
                """
                with conn.cursor() as cur:
                    cur.execute(sql, (
                        ticker, horizon, horizon_days, pred_date,
                        label, round(prob, 4), pred_close, source,
                    ))
                if cur.rowcount:
                    saved += 1
                else:
                    skipped += 1

        print(f"  {pred_date} — done")

    conn.commit()
    print(f"\nSaved {saved} predictions, {skipped} already existed.")

    print("\nResolving actuals...")
    resolve_actuals(conn)

    print()
    show_report(conn)
    conn.close()


if __name__ == "__main__":
    backfill()
