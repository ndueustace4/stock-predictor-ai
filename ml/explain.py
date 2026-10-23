# ml/explain.py
# ─────────────────────────────────────────────
# Phase 5 — SHAP Explainability  (v2)
#
# Updated for new feature set:
#   Technical : RSI-14, MACD, MACD signal,
#               Bollinger width, volume change,
#               price/MA ratio
#   Sentiment : lag1, lag2, ma_7, std_7, momentum
# ─────────────────────────────────────────────

import pickle
import json
import os
import sys
import numpy as np
import pandas as pd
import shap


def _make_shap_explainer(model):
    """Create a SHAP TreeExplainer, fixing XGBoost 3.x base_score format incompatibility."""
    if hasattr(model, "get_booster"):
        try:
            booster = model.get_booster()
            config  = json.loads(booster.save_config())
            bs = config["learner"]["learner_model_param"].get("base_score", "0.5")
            if isinstance(bs, str) and bs.startswith("["):
                config["learner"]["learner_model_param"]["base_score"] = bs.strip("[]")
                booster.load_config(json.dumps(config))
        except Exception:
            pass
    return shap.TreeExplainer(model)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_pipeline.db_connection import get_connection


# ─────────────────────────────────────────────
# Feature lists — must match train_model.py exactly
# ─────────────────────────────────────────────

FEATURES_NO_SENT = [
    "close", "volume",
    "ma_5", "ma_10", "ma_20",
    "volatility_7", "momentum_5",
    "price_lag1", "price_lag2", "price_lag3", "price_lag5",
    "rsi_14", "macd", "macd_signal",
    "bollinger_width", "volume_change", "price_ma_ratio",
    # v3
    "return_1d", "return_5d", "return_10d",
    "day_of_week", "atr_14",
    "high_20d_ratio", "low_20d_ratio",
    # v4 — market context
    "spy_return_1d",
    "qqq_return_1d",
    "vix_level",
    "vix_change",
    # v5 — sector ETFs + Chinese market
    "hsi_return_1d",
    "mchi_return_1d",
    "cny_usd_return",
    "xlk_return_1d",
    "xlf_return_1d",
    "xle_return_1d",
    "xlv_return_1d",
    "xli_return_1d",
    # v6 — earnings
    "earnings_surprise",
    # v7 — long-term regime
    "price_ma252_ratio",
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

# ── Human-readable labels for the dashboard ──
FEATURE_LABELS = {
    "close":               "Current Price",
    "volume":              "Trading Volume",
    "ma_5":                "5-Day Moving Avg",
    "ma_10":               "10-Day Moving Avg",
    "ma_20":               "20-Day Moving Avg",
    "volatility_7":        "7-Day Volatility",
    "momentum_5":          "5-Day Momentum",
    "price_lag1":          "Yesterday's Price",
    "price_lag2":          "2 Days Ago Price",
    "price_lag3":          "3 Days Ago Price",
    "price_lag5":          "5 Days Ago Price",
    "rsi_14":              "RSI (14)",
    "macd":                "MACD",
    "macd_signal":         "MACD Signal",
    "bollinger_width":     "Bollinger Band Width",
    "volume_change":       "Volume Change %",
    "price_ma_ratio":      "Price / MA20 Ratio",
    "return_1d":           "1-Day Return %",
    "return_5d":           "5-Day Return %",
    "return_10d":          "10-Day Return %",
    "day_of_week":         "Day of Week",
    "atr_14":              "ATR (14) Normalised",
    "high_20d_ratio":      "Distance from 20d High",
    "low_20d_ratio":       "Distance from 20d Low",
    "spy_return_1d":       "S&P 500 Daily Return",
    "qqq_return_1d":       "Nasdaq-100 Daily Return",
    "vix_level":           "VIX Fear Index Level",
    "vix_change":          "VIX Change (Fear Rising?)",
    "hsi_return_1d":       "Hang Seng (China) Return",
    "mchi_return_1d":      "MSCI China ETF Return",
    "xlk_return_1d":       "Tech Sector Return",
    "xlf_return_1d":       "Finance Sector Return",
    "xle_return_1d":       "Energy Sector Return",
    "xlv_return_1d":       "Healthcare Sector Return",
    "xli_return_1d":       "Industrials Sector Return",
    "earnings_surprise":   "Earnings Surprise %",
    "cny_usd_return":      "CNY/USD Rate Change (Yuan Strength)",
    "price_ma252_ratio":   "Price vs 1-Year Average (Regime)",
    "sentiment_today":     "Today's News Sentiment",
    "sentiment_lag1":      "Yesterday's Sentiment",
    "sentiment_lag2":      "2-Day Sentiment",
    "sentiment_ma_7":      "7-Day Avg Sentiment",
    "sentiment_std_7":     "Sentiment Uncertainty",
    "sentiment_momentum":  "Sentiment Momentum",
}


# ─────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────

def load_model(path: str):
    """Load a saved XGBoost model from pickle."""
    with open(path, "rb") as f:
        return pickle.load(f)


def load_features_from_db(ticker: str, use_sentiment: bool) -> pd.DataFrame:
    """
    Load all feature rows for a ticker from the DB.
    Columns are selected in FEATURES_WITH_SENT order so the DataFrame
    is named correctly regardless of DB schema ordering.
    """
    conn = get_connection()
    sql = """
        SELECT
            COALESCE(close,          0.0) AS close,
            COALESCE(volume,         0.0) AS volume,
            COALESCE(ma_5,           0.0) AS ma_5,
            COALESCE(ma_10,          0.0) AS ma_10,
            COALESCE(ma_20,          0.0) AS ma_20,
            COALESCE(volatility_7,   0.0) AS volatility_7,
            COALESCE(momentum_5,     0.0) AS momentum_5,
            COALESCE(price_lag1,     0.0) AS price_lag1,
            COALESCE(price_lag2,     0.0) AS price_lag2,
            COALESCE(price_lag3,     0.0) AS price_lag3,
            COALESCE(price_lag5,     0.0) AS price_lag5,
            COALESCE(rsi_14,         0.0) AS rsi_14,
            COALESCE(macd,           0.0) AS macd,
            COALESCE(macd_signal,    0.0) AS macd_signal,
            COALESCE(bollinger_width, 0.0) AS bollinger_width,
            COALESCE(volume_change,  0.0) AS volume_change,
            COALESCE(price_ma_ratio, 1.0) AS price_ma_ratio,
            COALESCE(return_1d,      0.0) AS return_1d,
            COALESCE(return_5d,      0.0) AS return_5d,
            COALESCE(return_10d,     0.0) AS return_10d,
            COALESCE(day_of_week,    0)   AS day_of_week,
            COALESCE(atr_14,         0.0) AS atr_14,
            COALESCE(high_20d_ratio, 1.0) AS high_20d_ratio,
            COALESCE(low_20d_ratio,  1.0) AS low_20d_ratio,
            COALESCE(spy_return_1d,  0.0) AS spy_return_1d,
            COALESCE(qqq_return_1d,  0.0) AS qqq_return_1d,
            COALESCE(vix_level,      0.0) AS vix_level,
            COALESCE(vix_change,     0.0) AS vix_change,
            COALESCE(hsi_return_1d,  0.0) AS hsi_return_1d,
            COALESCE(mchi_return_1d, 0.0) AS mchi_return_1d,
            COALESCE(cny_usd_return, 0.0) AS cny_usd_return,
            COALESCE(xlk_return_1d,  0.0) AS xlk_return_1d,
            COALESCE(xlf_return_1d,  0.0) AS xlf_return_1d,
            COALESCE(xle_return_1d,  0.0) AS xle_return_1d,
            COALESCE(xlv_return_1d,  0.0) AS xlv_return_1d,
            COALESCE(xli_return_1d,  0.0) AS xli_return_1d,
            COALESCE(earnings_surprise,  0.0) AS earnings_surprise,
            COALESCE(price_ma252_ratio,  1.0) AS price_ma252_ratio,
            COALESCE(sentiment_today,    0.0) AS sentiment_today,
            COALESCE(sentiment_lag1,     0.0) AS sentiment_lag1,
            COALESCE(sentiment_lag2,     0.0) AS sentiment_lag2,
            COALESCE(sentiment_ma_7,     0.0) AS sentiment_ma_7,
            COALESCE(sentiment_std_7,    0.0) AS sentiment_std_7,
            COALESCE(sentiment_momentum, 0.0) AS sentiment_momentum,
            COALESCE(pe_ratio,           0.0) AS pe_ratio,
            COALESCE(revenue_growth_yoy, 0.0) AS revenue_growth_yoy,
            COALESCE(eps_growth_yoy,     0.0) AS eps_growth_yoy,
            COALESCE(treasury_10y,       0.0) AS treasury_10y,
            COALESCE(treasury_2y,        0.0) AS treasury_2y,
            COALESCE(yield_curve,        0.0) AS yield_curve,
            target, date
        FROM features
        WHERE ticker = %s
        ORDER BY date ASC
    """
    with conn.cursor() as cur:
        cur.execute(sql, (ticker,))
        col_names = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
    conn.close()

    df = pd.DataFrame(rows, columns=col_names)

    all_features = FEATURES_252D_WITH_SENT
    for col in all_features:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    features = FEATURES_WITH_SENT if use_sentiment else FEATURES_NO_SENT
    return df[features + ["target", "date"]]


# ─────────────────────────────────────────────
# SHAP helpers
# ─────────────────────────────────────────────

def compute_global_importance(model, X: pd.DataFrame,
                               feature_names: list) -> tuple:
    """Compute mean absolute SHAP values across all rows."""
    print("  Computing SHAP values...")
    explainer = _make_shap_explainer(model)
    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    mean_abs   = np.abs(shap_values).mean(axis=0)
    importance = dict(zip(feature_names, mean_abs.tolist()))
    importance = dict(sorted(importance.items(),
                             key=lambda x: x[1], reverse=True))
    return importance, explainer, shap_values


def explain_single_prediction(model, explainer, row: pd.Series,
                               feature_names: list,
                               use_sentiment: bool) -> dict:
    """Generate SHAP explanation for one row (the latest prediction)."""
    X_row = row[feature_names].astype(float).values.reshape(1, -1)
    X_df  = pd.DataFrame(X_row, columns=feature_names)

    pred = model.predict(X_df)[0]
    prob = model.predict_proba(X_df)[0]
    conf = float(prob[1]) if pred == 1 else float(prob[0])

    sv = explainer.shap_values(X_df)
    if isinstance(sv, list):
        sv = sv[1]
    sv = sv[0]

    factors = [
        {
            "feature":       feat,
            "label":         FEATURE_LABELS.get(feat, feat),
            "shap_value":    round(float(val), 4),
            "feature_value": round(float(row[feat]), 4),
            "direction":     "positive" if val > 0 else "negative",
        }
        for feat, val in zip(feature_names, sv)
    ]
    factors.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

    return {
        "prediction":    "UP" if pred == 1 else "DOWN",
        "confidence":    round(conf, 4),
        "factors":       factors[:5],
        "use_sentiment": use_sentiment,
    }


def build_plain_english(explanation: dict) -> str:
    """Convert SHAP explanation to plain English for the dashboard."""
    direction = explanation["prediction"]
    conf      = explanation["confidence"] * 100
    factors   = explanation["factors"]

    arrow = "↑" if direction == "UP" else "↓"
    lines = [
        f"Prediction: {arrow} Price {direction} "
        f"(Confidence: {conf:.1f}%)", ""
    ]
    lines.append("Main drivers:")

    for i, f in enumerate(factors[:3], 1):
        sign   = "+" if f["shap_value"] > 0 else ""
        impact = ("pushed prediction UP"
                  if f["shap_value"] > 0
                  else "pushed prediction DOWN")
        lines.append(
            f"  {i}. {f['label']} = {f['feature_value']:.4f}"
            f"  →  {impact} ({sign}{f['shap_value']:.4f})"
        )

    return "\n".join(lines)


# ─────────────────────────────────────────────
# Full analysis (called from CLI)
# ─────────────────────────────────────────────

def run_shap_analysis(ticker: str = "AAPL"):
    print("=" * 55)
    print(f"Phase 5 — SHAP Explainability  [{ticker}]")
    print("=" * 55)

    os.makedirs("ml/models", exist_ok=True)
    results = {}

    for label, model_path, use_sent, features in [
        ("no_sentiment",
         "ml/models/model_no_sentiment.pkl",   False, FEATURES_NO_SENT),
        ("with_sentiment",
         "ml/models/model_with_sentiment.pkl", True,  FEATURES_WITH_SENT),
    ]:
        print(f"\nModel: {label}  ({len(features)} features)")
        model = load_model(model_path)
        df    = load_features_from_db(ticker, use_sent)

        if df.empty:
            print(f"  [WARNING] No feature data for {ticker}")
            continue

        X = df[features]
        importance, explainer, _ = compute_global_importance(
            model, X, features)

        print(f"\n  Global Feature Importance ({ticker}):")
        print(f"  {'Feature':<26} {'SHAP':>10}")
        print(f"  {'-'*38}")
        for feat, val in importance.items():
            print(f"  {FEATURE_LABELS.get(feat, feat):<26} {val:>10.4f}")

        latest_row  = df.iloc[-1]
        explanation = explain_single_prediction(
            model, explainer, latest_row, features, use_sent)
        plain = build_plain_english(explanation)

        print(f"\n  Latest Prediction for {ticker}:")
        print(f"  {'-'*38}")
        for line in plain.split("\n"):
            print(f"  {line}")

        results[label] = {
            "ticker":               ticker,
            "global_importance":    importance,
            "latest_explanation":   explanation,
            "plain_english":        plain,
        }

    out_path = "ml/models/shap_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*55}")
    print(f"SHAP results saved → {out_path}")
    return results


# ─────────────────────────────────────────────
# Called by FastAPI
# ─────────────────────────────────────────────

def explain_prediction_for_api(ticker: str,
                                use_sentiment: bool = True,
                                horizon: str = "1d") -> dict:
    """
    Called by api/main.py for every prediction/explain request.
    Loads the per-ticker model first; falls back to global model if missing.
    Returns structured dict ready to serve as JSON.
    """
    suffix   = "with_sentiment" if use_sentiment else "no_sentiment"
    h_suffix = "" if horizon == "1d" else f"_{horizon}"

    per_ticker_path = f"ml/models/{ticker}_{suffix}{h_suffix}.pkl"
    global_path     = f"ml/models/model_{suffix}{h_suffix}.pkl"
    fallback_path   = f"ml/models/{ticker}_{suffix}.pkl"

    if os.path.exists(per_ticker_path):
        model_path = per_ticker_path
    elif os.path.exists(global_path):
        model_path = global_path
    else:
        model_path = fallback_path

    # 252d models were trained on extended features (fundamentals + macro)
    if horizon == "252d":
        features = FEATURES_252D_WITH_SENT if use_sentiment else FEATURES_252D_NO_SENT
    else:
        features = FEATURES_WITH_SENT if use_sentiment else FEATURES_NO_SENT

    model = load_model(model_path)
    model_features = getattr(model, "feature_names_in_", None)
    if model_features is None and hasattr(model, "get_booster"):
        model_features = model.get_booster().feature_names
    if model_features is None:
        model_features = getattr(model, "feature_name_", None)
    if model_features is not None:
        available_features = set(features)
        missing_features = set(model_features) - available_features
        if missing_features:
            raise ValueError(
                f"Model {model_path} requires unsupported features: "
                f"{', '.join(sorted(missing_features))}"
            )
        features = list(model_features)

    # Load only the single latest row — loading all history is wasteful for a live prediction
    conn = get_connection()
    feat_select = ", ".join(f"COALESCE({f}, 0) AS {f}" for f in features)
    with conn.cursor() as cur:
        cur.execute(f"""
            SELECT {feat_select}, date, close AS raw_close
            FROM features
            WHERE ticker = %s AND close IS NOT NULL AND ma_20 IS NOT NULL
            ORDER BY date DESC LIMIT 1
        """, (ticker,))
        row = cur.fetchone()

    if not row:
        conn.close()
        return {"error": f"No feature data found for {ticker}"}

    col_names = features + ["date", "raw_close"]
    row_dict  = dict(zip(col_names, row))

    # Override sentiment_today with live articles collected so far today
    if use_sentiment:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT AVG(COALESCE(groq_sentiment, sentiment_score))
                FROM news_sentiment
                WHERE ticker = %s AND date = CURRENT_DATE
                  AND COALESCE(groq_sentiment, sentiment_score) IS NOT NULL
            """, (ticker,))
            live_today = cur.fetchone()[0]
        if live_today is not None:
            row_dict["sentiment_today"] = float(live_today)

    conn.close()

    latest_row = pd.Series(row_dict)
    explainer  = _make_shap_explainer(model)

    explanation = explain_single_prediction(
        model, explainer, latest_row, features, use_sentiment)
    explanation["plain_english"] = build_plain_english(explanation)
    explanation["ticker"]  = ticker
    explanation["date"]    = str(row_dict["date"])
    explanation["horizon"] = horizon

    return explanation


if __name__ == "__main__":
    run_shap_analysis(ticker="AAPL")