# ml/eval_models.py
# Evaluate all saved models against the current test split.
# Produces a clean accuracy/F1 table and saves training_metrics.json.
# Usage: python3 ml/eval_models.py

import os
import sys
import pickle
import json
from datetime import datetime, timedelta

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_pipeline.db_connection import get_connection
from ml.train_model import (
    load_features, time_split, compute_target,
    HORIZONS, FEATURES_NO_SENT, FEATURES_WITH_SENT,
    FEATURES_252D_NO_SENT, FEATURES_252D_WITH_SENT,
)

MODELS_DIR = "ml/models"
SENT_COLS = ["sentiment_today", "sentiment_lag1", "sentiment_lag2",
             "sentiment_ma_7", "sentiment_std_7", "sentiment_momentum"]


def load_model(path):
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def eval_model(model, X_test, y_test):
    if len(X_test) == 0:
        return None
    expected = getattr(model, "n_features_in_", None)
    if expected is not None and expected != X_test.shape[1]:
        print(f"    [SKIP] feature mismatch — model expects {expected}, data has {X_test.shape[1]}")
        return None
    try:
        y_pred = model.predict(X_test)
    except Exception as e:
        print(f"    [SKIP] predict failed: {e}")
        return None
    return {
        "accuracy":  round(float(accuracy_score(y_test, y_pred)), 4),
        "f1":        round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "recall":    round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "test_rows": len(y_test),
        "pos_rate":  round(float(y_test.mean()), 3),
    }


def main():
    conn = get_connection()
    print("Loading features from DB...")
    df = load_features(conn)
    conn.close()
    print(f"  {len(df):,} rows | {df['ticker'].nunique()} tickers | "
          f"{df['date'].min().date()} to {df['date'].max().date()}")

    tickers = sorted(df["ticker"].unique())
    all_results = {}

    for horizon_key, horizon_cfg in HORIZONS.items():
        days   = horizon_cfg["days"]
        cutoff = horizon_cfg["cutoff"]
        h_sfx  = "" if horizon_key == "1d" else f"_{horizon_key}"

        feat_ns = FEATURES_252D_NO_SENT  if horizon_key == "252d" else FEATURES_NO_SENT
        feat_ws = FEATURES_252D_WITH_SENT if horizon_key == "252d" else FEATURES_WITH_SENT

        df_h, _ = compute_target(df, horizon_key, horizon_cfg, tickers, days, cutoff)
        df_h = df_h.dropna(subset=["target"])
        df_h["target"] = df_h["target"].astype(int)

        rows_ns, rows_ws = [], []

        for ticker in tickers:
            t_df = df_h[df_h["ticker"] == ticker].copy()

            # no-sentiment model
            df1 = t_df[feat_ns + ["date", "target"]].dropna()
            _, X1_te, _, y1_te = time_split(df1, feat_ns, cutoff)
            m1 = load_model(f"{MODELS_DIR}/{ticker}_no_sentiment{h_sfx}.pkl")
            r1 = eval_model(m1, X1_te, y1_te) if m1 is not None else None

            # with-sentiment model
            df2 = t_df[feat_ws + ["date", "target"]].copy()
            df2[SENT_COLS] = df2[SENT_COLS].fillna(0.0)
            df2 = df2.dropna()
            _, X2_te, _, y2_te = time_split(df2, feat_ws, cutoff)
            m2 = load_model(f"{MODELS_DIR}/{ticker}_with_sentiment{h_sfx}.pkl")
            r2 = eval_model(m2, X2_te, y2_te) if m2 is not None else None

            if r1:
                rows_ns.append({"ticker": ticker, **r1})
            if r2:
                rows_ws.append({"ticker": ticker, **r2})

        # Print results table
        print(f"\n{'='*70}")
        print(f"  HORIZON: {horizon_key}  |  test split: {cutoff} onward")
        print(f"  {'Ticker':<7} {'No-Sent Acc':>12} {'No-Sent F1':>11} "
              f"{'W-Sent Acc':>12} {'W-Sent F1':>10}")
        print(f"  {'-'*62}")

        for r1, r2 in zip(rows_ns, rows_ws):
            print(f"  {r1['ticker']:<7} "
                  f"{r1['accuracy']*100:>11.1f}% "
                  f"{r1['f1']*100:>10.1f}% "
                  f"{r2['accuracy']*100:>11.1f}% "
                  f"{r2['f1']*100:>9.1f}%")

        if rows_ns and rows_ws:
            def avg(lst, k): return sum(d[k] for d in lst) / len(lst)
            print(f"  {'AVG':<7} "
                  f"{avg(rows_ns,'accuracy')*100:>11.1f}% "
                  f"{avg(rows_ns,'f1')*100:>10.1f}% "
                  f"{avg(rows_ws,'accuracy')*100:>11.1f}% "
                  f"{avg(rows_ws,'f1')*100:>9.1f}%")
        print(f"{'='*70}")

        all_results[horizon_key] = {
            "cutoff":          cutoff,
            "no_sentiment":    rows_ns,
            "with_sentiment":  rows_ws,
        }

    out_path = f"{MODELS_DIR}/training_metrics.json"
    payload = {
        "generated_at": datetime.now().isoformat(),
        "test_cutoff":  _rolling_cutoff(),
        "horizons":     all_results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nSaved -> {out_path}")


def _rolling_cutoff():
    return (datetime.today() - timedelta(days=30)).strftime("%Y-%m-%d")


if __name__ == "__main__":
    main()
