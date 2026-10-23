# ml/predict.py
# ─────────────────────────────────────────────
# Daily prediction logger + accuracy tracker.
#
# Usage:
#   python ml/predict.py              — log today's predictions
#   python ml/predict.py --resolve    — fill actuals for matured predictions
#   python ml/predict.py --report     — show accuracy comparison table
#   python ml/predict.py --all        — run + resolve + report
# ─────────────────────────────────────────────

import os
import sys
import pickle
import argparse
from datetime import datetime

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_pipeline.db_connection import get_connection
from config.settings import STOCKS

HORIZONS = {
    "1d":   1,
    "5d":   5,
    "21d":  21,
    "252d": 252,
}

FEATURES_NO_SENT = [
    "close", "volume",
    "ma_5", "ma_10", "ma_20",
    "volatility_7", "momentum_5",
    "price_lag1", "price_lag2", "price_lag3", "price_lag5",
    "rsi_14", "macd", "macd_signal",
    "bollinger_width", "volume_change", "price_ma_ratio",
    "return_1d", "return_5d", "return_10d",
    "day_of_week", "atr_14",
    "high_20d_ratio", "low_20d_ratio",
    "spy_return_1d", "qqq_return_1d",
    "vix_level", "vix_change",
    "hsi_return_1d", "mchi_return_1d", "cny_usd_return",
    "xlk_return_1d", "xlf_return_1d", "xle_return_1d",
    "xlv_return_1d", "xli_return_1d",
    "earnings_surprise", "price_ma252_ratio",
]

FEATURES_WITH_SENT = FEATURES_NO_SENT + [
    "sentiment_today",
    "sentiment_lag1", "sentiment_lag2",
    "sentiment_ma_7", "sentiment_std_7", "sentiment_momentum",
]

FEATURES_252D_EXTRA = [
    "pe_ratio", "revenue_growth_yoy", "eps_growth_yoy",
    "treasury_10y", "treasury_2y", "yield_curve",
]
FEATURES_252D_NO_SENT   = FEATURES_NO_SENT  + FEATURES_252D_EXTRA
FEATURES_252D_WITH_SENT = FEATURES_WITH_SENT + FEATURES_252D_EXTRA


# ─────────────────────────────────────────────
# DB setup
# ─────────────────────────────────────────────

def init_table(conn):
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id               SERIAL PRIMARY KEY,
                ticker           VARCHAR(10)  NOT NULL,
                horizon          VARCHAR(10)  NOT NULL,
                horizon_days     INTEGER      NOT NULL,
                prediction_date  DATE         NOT NULL,
                predicted_label  INTEGER      NOT NULL,
                predicted_prob   FLOAT,
                prediction_close FLOAT,
                model_source     VARCHAR(20),
                actual_close     FLOAT,
                actual_label     INTEGER,
                correct          BOOLEAN,
                resolved_at      TIMESTAMP,
                created_at       TIMESTAMP DEFAULT NOW(),
                UNIQUE(ticker, horizon, prediction_date)
            )
        """)
    conn.commit()


# ─────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────

def load_best_model(ticker: str, horizon: str):
    """
    Load the best available model for ticker/horizon.
    Tries: per-ticker with-sent → per-ticker no-sent → global with-sent → global no-sent.
    Returns (model, features_list, model_source).
    """
    h_suffix = "" if horizon == "1d" else f"_{horizon}"

    # Per-ticker 252d models use extended features (fundamentals + macro)
    # Global 252d fallback uses standard features (cross-ticker fundamentals add noise)
    if horizon == "252d":
        per_feat_ws, per_feat_ns = FEATURES_252D_WITH_SENT, FEATURES_252D_NO_SENT
        glob_feat_ws, glob_feat_ns = FEATURES_WITH_SENT, FEATURES_NO_SENT
    else:
        per_feat_ws = glob_feat_ws = FEATURES_WITH_SENT
        per_feat_ns = glob_feat_ns = FEATURES_NO_SENT

    candidates = [
        (f"ml/models/{ticker}_with_sentiment{h_suffix}.pkl", per_feat_ws,  "per_ticker"),
        (f"ml/models/{ticker}_no_sentiment{h_suffix}.pkl",   per_feat_ns,  "per_ticker"),
        (f"ml/models/model_with_sentiment{h_suffix}.pkl",    glob_feat_ws, "global"),
        (f"ml/models/model_no_sentiment{h_suffix}.pkl",      glob_feat_ns, "global"),
    ]

    for path, features, source in candidates:
        if os.path.exists(path):
            with open(path, "rb") as f:
                model = pickle.load(f)
            return model, features, source

    return None, None, None


# ─────────────────────────────────────────────
# Feature loading
# ─────────────────────────────────────────────

def load_latest_features(conn, ticker: str) -> dict | None:
    """Return the most recent features row for a ticker as a dict."""
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
            COALESCE(earnings_surprise,  0) AS earnings_surprise,
            COALESCE(price_ma252_ratio,  1) AS price_ma252_ratio,
            COALESCE(sentiment_today,    0) AS sentiment_today,
            COALESCE(sentiment_lag1,     0) AS sentiment_lag1,
            COALESCE(sentiment_lag2,     0) AS sentiment_lag2,
            COALESCE(sentiment_ma_7,     0) AS sentiment_ma_7,
            COALESCE(sentiment_std_7,    0) AS sentiment_std_7,
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
          AND close IS NOT NULL
          AND ma_20  IS NOT NULL
        ORDER BY date DESC
        LIMIT 1
    """
    with conn.cursor() as cur:
        cur.execute(sql, (ticker,))
        row = cur.fetchone()

    if not row:
        return None

    cols = FEATURES_252D_WITH_SENT + ["date", "raw_close"]
    return dict(zip(cols, row))


# ─────────────────────────────────────────────
# Core: run predictions
# ─────────────────────────────────────────────

def run_predictions(conn):
    """Log today's predictions for all tickers × horizons."""
    print("=" * 55)
    print(f"Prediction run  —  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 55)

    init_table(conn)
    saved = 0
    skipped = 0

    for ticker in sorted(STOCKS):
        row = load_latest_features(conn, ticker)
        if row is None:
            print(f"  [{ticker}] No features — skip")
            continue

        pred_date = row["date"]
        pred_close = float(row["raw_close"])

        for horizon, horizon_days in HORIZONS.items():
            model, features, source = load_best_model(ticker, horizon)
            if model is None:
                print(f"  [{ticker}/{horizon}] No model found — skip")
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
                direction = "UP  " if label == 1 else "DOWN"
                print(f"  {ticker:<5} {horizon:<5} {direction}  "
                      f"({prob*100:.1f}% conf)  [{source}]")
            else:
                skipped += 1

    conn.commit()
    print(f"\nDone. Saved {saved} new predictions, {skipped} already logged.")


# ─────────────────────────────────────────────
# Core: resolve actuals
# ─────────────────────────────────────────────

def resolve_actuals(conn):
    """
    For predictions where enough trading days have passed,
    fetch the actual close price and mark correct/incorrect.
    """
    print("=" * 55)
    print("Resolving actuals...")
    print("=" * 55)

    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, ticker, horizon_days, prediction_date, prediction_close,
                   predicted_label
            FROM predictions
            WHERE actual_label IS NULL
            ORDER BY prediction_date ASC
        """)
        pending = cur.fetchall()

    if not pending:
        print("Nothing to resolve.")
        return

    resolved = 0
    for pred_id, ticker, horizon_days, pred_date, pred_close, pred_label in pending:
        # Find the close price exactly horizon_days trading days after pred_date
        with conn.cursor() as cur:
            cur.execute("""
                SELECT date, close FROM stock_prices
                WHERE ticker = %s AND date > %s
                ORDER BY date ASC
                OFFSET %s LIMIT 1
            """, (ticker, pred_date, horizon_days - 1))
            result = cur.fetchone()

        if result is None:
            continue  # Not enough data yet

        actual_date, actual_close = result
        actual_close = float(actual_close)
        actual_label = 1 if actual_close > float(pred_close) else 0
        correct = (actual_label == pred_label)

        with conn.cursor() as cur:
            cur.execute("""
                UPDATE predictions
                SET actual_close = %s,
                    actual_label  = %s,
                    correct       = %s,
                    resolved_at   = NOW()
                WHERE id = %s
            """, (actual_close, actual_label, correct, pred_id))

        resolved += 1
        mark = "OK" if correct else "XX"
        print(f"  {mark} {ticker:<5} {str(pred_date):<12} "
              f"pred={'UP' if pred_label else 'DOWN'}  "
              f"actual={'UP' if actual_label else 'DOWN'}  "
              f"(resolved {actual_date})")

    conn.commit()
    print(f"\nResolved {resolved} predictions.")


# ─────────────────────────────────────────────
# Core: accuracy report
# ─────────────────────────────────────────────

def show_report(conn):
    """Print accuracy comparison table and pending predictions."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT ticker, horizon, COUNT(*) AS total,
                   SUM(correct::int) AS correct_count
            FROM predictions
            WHERE correct IS NOT NULL
            GROUP BY ticker, horizon
            ORDER BY ticker, horizon
        """)
        resolved = cur.fetchall()

    with conn.cursor() as cur:
        cur.execute("""
            SELECT ticker, horizon, prediction_date, predicted_label,
                   predicted_prob, model_source
            FROM predictions
            WHERE actual_label IS NULL
            ORDER BY prediction_date DESC, ticker, horizon
        """)
        pending = cur.fetchall()

    print("\n" + "=" * 65)
    print("  PREDICTION ACCURACY REPORT")
    print("=" * 65)

    if resolved:
        horizons = ["1d", "5d", "21d", "252d"]
        # Pivot into {ticker: {horizon: (total, correct)}}
        data: dict = {}
        for ticker, horizon, total, correct_count in resolved:
            data.setdefault(ticker, {})[horizon] = (total, correct_count or 0)

        header = f"  {'Ticker':<7}"
        for h in horizons:
            header += f"  {h:>14}"
        print(header)
        print("  " + "-" * 63)

        for ticker in sorted(data):
            row_str = f"  {ticker:<7}"
            for h in horizons:
                if h in data[ticker]:
                    total, correct_count = data[ticker][h]
                    acc = correct_count / total * 100
                    cell = f"{correct_count}/{total} ({acc:.0f}%)"
                else:
                    cell = "—"
                row_str += f"  {cell:>14}"
            print(row_str)
    else:
        print("  No resolved predictions yet.")

    print("\n" + "=" * 65)
    print("  PENDING PREDICTIONS (not yet matured)")
    print("=" * 65)

    if pending:
        print(f"  {'Ticker':<7} {'Horizon':<8} {'Date':<13} "
              f"{'Prediction':<11} {'Confidence':>10}  {'Model'}")
        print("  " + "-" * 63)
        for ticker, horizon, pred_date, pred_label, prob, source in pending:
            direction = "UP" if pred_label == 1 else "DOWN"
            print(f"  {ticker:<7} {horizon:<8} {str(pred_date):<13} "
                  f"{direction:<11} {prob*100:>9.1f}%  {source}")
    else:
        print("  No pending predictions.")

    print("=" * 65)


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stock prediction logger")
    parser.add_argument("--resolve", action="store_true",
                        help="Resolve actuals for matured predictions")
    parser.add_argument("--report",  action="store_true",
                        help="Show accuracy comparison report")
    parser.add_argument("--all",     action="store_true",
                        help="Run predictions + resolve + report")
    args = parser.parse_args()

    conn = get_connection()
    init_table(conn)

    if args.all:
        run_predictions(conn)
        resolve_actuals(conn)
        show_report(conn)
    elif args.resolve:
        resolve_actuals(conn)
    elif args.report:
        show_report(conn)
    else:
        run_predictions(conn)

    conn.close()


if __name__ == "__main__":
    main()
