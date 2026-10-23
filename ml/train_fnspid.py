# ml/train_fnspid.py
# Trains XGBoost models on FNSPID dataset and compares metrics with our existing models.
# Usage: python ml/train_fnspid.py

import os
import sys
import warnings
import pickle
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import STOCKS

PRICES_DIR  = "ml/fnspid_prices"
NEWS_PATH   = "ml/fnspid_filtered.csv"
OUT_DIR     = "ml/fnspid_models"
TRAIN_END   = "2022-12-31"
TEST_START  = "2023-01-01"
TICKER_MAP  = {"FB": "META"}

HORIZONS = {"1d": 1, "5d": 5, "21d": 21, "252d": 252}

os.makedirs(OUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# Technical indicator computation
# ─────────────────────────────────────────────

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").reset_index(drop=True)
    c = df["close"]
    v = df["volume"]

    df["ma_5"]  = c.rolling(5).mean()
    df["ma_10"] = c.rolling(10).mean()
    df["ma_20"] = c.rolling(20).mean()
    df["ma_252"] = c.rolling(252).mean()

    df["volatility_7"] = c.rolling(7).std()
    df["momentum_5"]   = c - c.shift(5)
    df["price_lag1"]   = c.shift(1)
    df["price_lag2"]   = c.shift(2)
    df["price_lag3"]   = c.shift(3)
    df["price_lag5"]   = c.shift(5)

    df["return_1d"]  = c.pct_change(1)
    df["return_5d"]  = c.pct_change(5)
    df["return_10d"] = c.pct_change(10)

    df["volume_change"]   = v.pct_change(1)
    df["price_ma_ratio"]  = c / df["ma_20"]
    df["price_ma252_ratio"] = c / df["ma_252"]
    df["day_of_week"]     = df["date"].dt.dayofweek

    # RSI-14
    delta = c.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema26 = c.ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # Bollinger width
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    df["bollinger_width"] = (2 * bb_std) / bb_mid

    # ATR-14
    high, low = df["high"], df["low"]
    tr = pd.concat([
        high - low,
        (high - c.shift(1)).abs(),
        (low  - c.shift(1)).abs()
    ], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean() / c

    # 20-day high/low ratios
    df["high_20d_ratio"] = c / c.rolling(20).max()
    df["low_20d_ratio"]  = c / c.rolling(20).min()

    return df


TECH_FEATURES = [
    "close", "volume", "ma_5", "ma_10", "ma_20",
    "volatility_7", "momentum_5",
    "price_lag1", "price_lag2", "price_lag3", "price_lag5",
    "rsi_14", "macd", "macd_signal",
    "bollinger_width", "volume_change", "price_ma_ratio",
    "return_1d", "return_5d", "return_10d",
    "day_of_week", "atr_14", "high_20d_ratio", "low_20d_ratio",
    "price_ma252_ratio",
]

SENT_FEATURES = [
    "sentiment_today", "sentiment_lag1", "sentiment_lag2",
    "sentiment_ma_7", "sentiment_std_7", "sentiment_momentum",
]


# ─────────────────────────────────────────────
# Sentiment computation
# ─────────────────────────────────────────────

def build_sentiment(news_path: str) -> pd.DataFrame:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()

    df = pd.read_csv(news_path)
    df["Stock_symbol"] = df["Stock_symbol"].replace(TICKER_MAP)
    df["date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None).dt.normalize()
    df["raw_sent"] = df["Article_title"].fillna("").apply(
        lambda t: analyzer.polarity_scores(t)["compound"]
    )
    daily = (
        df.groupby(["Stock_symbol", "date"])["raw_sent"]
        .mean().reset_index()
        .rename(columns={"Stock_symbol": "ticker", "raw_sent": "sentiment"})
    )
    return daily


def add_sentiment_features(df: pd.DataFrame,
                            sent_df: pd.DataFrame,
                            ticker: str) -> pd.DataFrame:
    t_sent = sent_df[sent_df["ticker"] == ticker][["date", "sentiment"]].copy()
    df = df.merge(t_sent, on="date", how="left")
    df["sentiment"] = df["sentiment"].fillna(0)
    df = df.sort_values("date").reset_index(drop=True)

    df["sentiment_today"]    = df["sentiment"]
    df["sentiment_lag1"]     = df["sentiment"].shift(1).fillna(0)
    df["sentiment_lag2"]     = df["sentiment"].shift(2).fillna(0)
    df["sentiment_ma_7"]     = df["sentiment"].rolling(7, min_periods=1).mean()
    df["sentiment_std_7"]    = df["sentiment"].rolling(7, min_periods=1).std().fillna(0)
    df["sentiment_momentum"] = df["sentiment_today"] - df["sentiment_lag1"]
    return df


# ─────────────────────────────────────────────
# Train + evaluate one ticker
# ─────────────────────────────────────────────

def train_and_eval(ticker: str, price_df: pd.DataFrame,
                   sent_df: pd.DataFrame, horizon: str,
                   horizon_days: int) -> dict:
    df = compute_features(price_df.copy())
    df = add_sentiment_features(df, sent_df, ticker)
    df = df.replace([np.inf, -np.inf], np.nan)

    # Target: 1 if close N trading days ahead > today's close
    c = df["close"]
    df["target"] = (c.shift(-horizon_days) > c).astype(int)

    all_features = TECH_FEATURES + SENT_FEATURES
    df = df.dropna(subset=all_features + ["target"])

    train = df[df["date"] <= TRAIN_END]
    test  = df[df["date"] >= TEST_START]

    if len(train) < 100 or len(test) < 20:
        return None

    X_train = train[all_features].astype(float)
    y_train = train["target"].astype(int)
    X_test  = test[all_features].astype(float)
    y_test  = test["target"].astype(int)

    model = XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric="logloss",
        random_state=42, verbosity=0,
    )
    model.fit(X_train, y_train)

    with open(os.path.join(OUT_DIR, f"{ticker}_{horizon}_fnspid.pkl"), "wb") as f:
        pickle.dump(model, f)

    preds = model.predict(X_test)
    return {
        "ticker":     ticker,
        "horizon":    horizon,
        "train_rows": len(train),
        "test_rows":  len(test),
        "accuracy":   round(accuracy_score(y_test, preds), 4),
        "precision":  round(precision_score(y_test, preds, zero_division=0), 4),
        "recall":     round(recall_score(y_test, preds, zero_division=0), 4),
        "f1":         round(f1_score(y_test, preds, zero_division=0), 4),
    }


# ─────────────────────────────────────────────
# Our existing model metrics (from last retrain)
# ─────────────────────────────────────────────

def load_existing_metrics(conn) -> dict:
    """Pull test accuracy from our DB predictions table (2025 out-of-sample)."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT ticker,
                   COUNT(*) AS total,
                   SUM(correct::int) AS correct_count
            FROM predictions
            WHERE correct IS NOT NULL AND horizon = '1d'
            GROUP BY ticker
        """)
        rows = cur.fetchall()
    return {ticker: (total, correct) for ticker, total, correct in rows}


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    print("=" * 65)
    print("FNSPID vs Our Dataset — Model Comparison")
    print(f"FNSPID train: up to {TRAIN_END}  |  test: {TEST_START}+")
    print("=" * 65)

    print("\nComputing FNSPID sentiment (VADER)...")
    sent_df = build_sentiment(NEWS_PATH)

    results = []

    for ticker in sorted(STOCKS):
        price_file = os.path.join(PRICES_DIR, f"{ticker}.pkl")
        fb_file    = os.path.join(PRICES_DIR, "FB.pkl")

        if not os.path.exists(price_file):
            if ticker == "META" and os.path.exists(fb_file):
                price_file = fb_file
            else:
                print(f"  [{ticker}] No price file — skip")
                continue

        price_df = pd.read_csv(price_file, parse_dates=["date"])
        price_df.columns = [c.lower() for c in price_df.columns]
        price_df = price_df.rename(columns={"adj close": "adj_close"})

        for horizon, horizon_days in HORIZONS.items():
            r = train_and_eval(ticker, price_df, sent_df, horizon, horizon_days)
            if r is None:
                continue
            results.append(r)
            print(f"  {ticker:<6} [{horizon:<4}]  "
                  f"acc={r['accuracy']*100:.1f}%  "
                  f"prec={r['precision']*100:.1f}%  "
                  f"rec={r['recall']*100:.1f}%  "
                  f"F1={r['f1']*100:.1f}%")

    # Load our existing live prediction accuracy
    from data_pipeline.db_connection import get_connection
    conn = get_connection()
    existing = load_existing_metrics(conn)
    conn.close()

    # Final comparison table per horizon
    print("\n" + "=" * 70)
    print("FNSPID MODEL RESULTS BY TICKER AND HORIZON")
    print("=" * 70)
    print(f"  {'Ticker':<7} {'1d Acc':>8} {'1d F1':>7} {'5d Acc':>8} {'5d F1':>7} "
          f"{'21d Acc':>8} {'21d F1':>7} {'252d Acc':>9} {'252d F1':>8}")
    print("  " + "-" * 68)

    for ticker in sorted(STOCKS):
        row_str = f"  {ticker:<7}"
        for h in ["1d", "5d", "21d", "252d"]:
            match = [r for r in results if r["ticker"] == ticker and r["horizon"] == h]
            if match:
                r = match[0]
                row_str += f"  {r['accuracy']*100:>6.1f}%  {r['f1']*100:>5.1f}%"
            else:
                row_str += f"  {'N/A':>6}   {'N/A':>5}"
        print(row_str)

    print("\n" + "=" * 70)
    print("OUR MODEL — Live out-of-sample accuracy (May 2026, 1d only)")
    print("=" * 70)
    for ticker, (total, correct) in sorted(existing.items()):
        acc = correct / total * 100 if total else 0
        print(f"  {ticker:<7}  {acc:.1f}%  ({correct}/{total})")
    print("=" * 70)


if __name__ == "__main__":
    main()
