# ml/evaluate_fnspid.py
# Tests our existing models using FNSPID news sentiment as an independent dataset.
# Compares accuracy: FNSPID sentiment vs no sentiment (baseline).
#
# Usage: python ml/evaluate_fnspid.py

import os
import sys
import pickle
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_pipeline.db_connection import get_connection
from ml.predict import FEATURES_NO_SENT, FEATURES_WITH_SENT, load_best_model

FNSPID_PATH = "ml/fnspid_filtered.csv"
TEST_START  = "2018-01-01"
TEST_END    = "2020-06-10"

# FB was Meta before Oct 2021
TICKER_MAP = {"FB": "META"}


# ─────────────────────────────────────────────
# Step 1 — Compute daily sentiment from FNSPID
# ─────────────────────────────────────────────

def compute_fnspid_sentiment(path: str) -> pd.DataFrame:
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError:
        print("Installing vaderSentiment...")
        os.system("pip install vaderSentiment -q")
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    print("Computing VADER sentiment on FNSPID article titles...")
    df = pd.read_csv(path)

    # Rename FB -> META
    df["Stock_symbol"] = df["Stock_symbol"].replace(TICKER_MAP)

    # Parse date to just the date portion
    df["date"] = pd.to_datetime(df["Date"], utc=True).dt.date

    # Score each title
    analyzer = SentimentIntensityAnalyzer()
    df["sentiment"] = df["Article_title"].fillna("").apply(
        lambda t: analyzer.polarity_scores(t)["compound"]
    )

    # Aggregate: mean sentiment per ticker per day
    daily = (
        df.groupby(["Stock_symbol", "date"])["sentiment"]
        .mean()
        .reset_index()
        .rename(columns={"Stock_symbol": "ticker"})
    )
    daily["date"] = pd.to_datetime(daily["date"])
    print(f"  Built {len(daily):,} ticker-date sentiment rows.")
    return daily


# ─────────────────────────────────────────────
# Step 2 — Load technical features from DB
# ─────────────────────────────────────────────

def load_tech_features(conn, ticker: str,
                       start: str, end: str) -> pd.DataFrame:
    sql = """
        SELECT date,
            close, volume, ma_5, ma_10, ma_20,
            volatility_7, momentum_5,
            COALESCE(price_lag1,  0) AS price_lag1,
            COALESCE(price_lag2,  0) AS price_lag2,
            COALESCE(price_lag3,  0) AS price_lag3,
            COALESCE(price_lag5,  0) AS price_lag5,
            COALESCE(rsi_14,      0) AS rsi_14,
            COALESCE(macd,        0) AS macd,
            COALESCE(macd_signal, 0) AS macd_signal,
            COALESCE(bollinger_width, 0) AS bollinger_width,
            COALESCE(volume_change,   0) AS volume_change,
            COALESCE(price_ma_ratio,  1) AS price_ma_ratio,
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
            target
        FROM features
        WHERE ticker = %s AND date >= %s AND date <= %s
          AND close IS NOT NULL AND ma_20 IS NOT NULL AND target IS NOT NULL
        ORDER BY date ASC
    """
    with conn.cursor() as cur:
        cur.execute(sql, (ticker, start, end))
        rows = cur.fetchall()

    cols = ["date"] + FEATURES_NO_SENT + ["target"]
    df = pd.DataFrame(rows, columns=cols)
    df["date"] = pd.to_datetime(df["date"])
    return df


# ─────────────────────────────────────────────
# Step 3 — Evaluate model on one ticker
# ─────────────────────────────────────────────

def evaluate_ticker(ticker: str, tech_df: pd.DataFrame,
                    sent_df: pd.DataFrame, model, features: list,
                    use_sentiment: bool) -> dict:
    if tech_df.empty:
        return None

    if use_sentiment:
        # Merge FNSPID sentiment onto technical features by date
        ticker_sent = sent_df[sent_df["ticker"] == ticker][["date", "sentiment"]].copy()
        df = tech_df.merge(ticker_sent, on="date", how="left")
        df["sentiment"] = df["sentiment"].fillna(0)

        # Build lagged sentiment features
        df = df.sort_values("date").reset_index(drop=True)
        df["sentiment_today"]    = df["sentiment"]
        df["sentiment_lag1"]     = df["sentiment"].shift(1).fillna(0)
        df["sentiment_lag2"]     = df["sentiment"].shift(2).fillna(0)
        df["sentiment_ma_7"]     = df["sentiment"].rolling(7, min_periods=1).mean()
        df["sentiment_std_7"]    = df["sentiment"].rolling(7, min_periods=1).std().fillna(0)
        df["sentiment_momentum"] = df["sentiment_today"] - df["sentiment_lag1"]
    else:
        df = tech_df.copy()

    X = df[features].astype(float)
    y = df["target"].astype(int)

    preds = model.predict(X)
    correct = (preds == y).sum()
    total   = len(y)
    acc     = correct / total if total > 0 else 0

    return {"ticker": ticker, "correct": int(correct),
            "total": int(total), "accuracy": round(acc, 4)}


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("FNSPID Independent Evaluation")
    print(f"Test period: {TEST_START} to {TEST_END}")
    print("=" * 60)

    sentiment_df = compute_fnspid_sentiment(FNSPID_PATH)
    conn = get_connection()

    tickers_with_sent  = sorted(sentiment_df["ticker"].unique())
    all_tickers        = sorted(set(FEATURES_NO_SENT) | set(tickers_with_sent))

    from config.settings import STOCKS
    all_tickers = sorted(STOCKS)

    results_no_sent   = []
    results_with_sent = []

    for ticker in all_tickers:
        tech_df = load_tech_features(conn, ticker, TEST_START, TEST_END)
        if tech_df.empty:
            print(f"  [{ticker}] No feature data in test period — skip")
            continue

        # No-sentiment model — load explicitly
        ns_path = f"ml/models/{ticker}_no_sentiment.pkl"
        if not os.path.exists(ns_path):
            ns_path = "ml/models/model_no_sentiment.pkl"
        if not os.path.exists(ns_path):
            print(f"  [{ticker}] No no-sentiment model — skip")
            continue
        with open(ns_path, "rb") as f:
            model_ns = pickle.load(f)

        r_ns = evaluate_ticker(ticker, tech_df, sentiment_df,
                               model_ns, FEATURES_NO_SENT, False)
        if r_ns:
            results_no_sent.append(r_ns)

        # With-sentiment model — load explicitly
        ws_path = f"ml/models/{ticker}_with_sentiment.pkl"
        if not os.path.exists(ws_path):
            ws_path = "ml/models/model_with_sentiment.pkl"
        if not os.path.exists(ws_path):
            print(f"  [{ticker}] No with-sentiment model — skip")
            continue
        with open(ws_path, "rb") as f:
            model_ws = pickle.load(f)

        r_ws = evaluate_ticker(ticker, tech_df, sentiment_df,
                               model_ws, FEATURES_WITH_SENT, True)
        if r_ws:
            results_with_sent.append(r_ws)

        print(f"  {ticker:<6}  no-sent: {r_ns['accuracy']*100:.1f}%  "
              f"({r_ns['correct']}/{r_ns['total']})  |  "
              f"fnspid-sent: {r_ws['accuracy']*100:.1f}%  "
              f"({r_ws['correct']}/{r_ws['total']})")

    conn.close()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY — 1-Day Prediction Accuracy")
    print("=" * 60)
    print(f"  {'Ticker':<8} {'No Sentiment':>14} {'FNSPID Sentiment':>18}")
    print("  " + "-" * 42)

    total_ns = total_ws = correct_ns = correct_ws = 0
    for ns, ws in zip(results_no_sent, results_with_sent):
        print(f"  {ns['ticker']:<8} "
              f"{ns['accuracy']*100:>12.1f}%  "
              f"{ws['accuracy']*100:>16.1f}%")
        total_ns   += ns["total"];   correct_ns += ns["correct"]
        total_ws   += ws["total"];   correct_ws += ws["correct"]

    print("  " + "-" * 42)
    overall_ns = correct_ns / total_ns * 100 if total_ns else 0
    overall_ws = correct_ws / total_ws * 100 if total_ws else 0
    print(f"  {'OVERALL':<8} {overall_ns:>12.1f}%  {overall_ws:>16.1f}%")
    print("=" * 60)

    verdict = "FNSPID sentiment IMPROVES accuracy" if overall_ws > overall_ns \
              else "FNSPID sentiment does NOT improve accuracy"
    print(f"\nVerdict: {verdict}")
    print("(If no-sent accuracy is similar to FNSPID-sent, the issue is the MODEL not the dataset)")


if __name__ == "__main__":
    main()
