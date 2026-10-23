# api/chat_context.py
# ─────────────────────────────────────────────
# Builds the data context injected into Claude's system prompt.
#
# Flow:
#   1. detect_tickers()  — scan message for known stock symbols
#   2. detect_intents()  — classify what the user is asking
#   3. fetch_*()         — pull relevant data from DB / ML model
#   4. build_context()   — assemble everything into a compact string
# ─────────────────────────────────────────────

import re
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_pipeline.db_connection import get_connection
from config.settings import STOCKS

# ─────────────────────────────────────────────
# Intent keyword map
# ─────────────────────────────────────────────

INTENT_KEYWORDS = {
    "prediction": [
        "predict", "prediction", "go up", "go down", "tomorrow",
        "bullish", "bearish", "will it", "direction", "forecast",
        "next day", "rise", "fall", "up or down",
    ],
    "sentiment": [
        "sentiment", "news", "headlines", "positive", "negative",
        "feeling", "media", "articles", "press", "coverage",
        "impactful", "week",
    ],
    "price": [
        "price", "trend", "performance", "performed", "rsi",
        "oversold", "overbought", "macd", "technical", "moving average",
        "bollinger", "momentum", "volume", "30 days", "last month",
    ],
    "explain": [
        "why", "explain", "reason", "factors", "driving",
        "cause", "what features", "behind", "shap",
    ],
    "compare": [
        "compare", " vs ", "versus", "better", "difference",
        "which is", "between",
    ],
    "metrics": [
        "accuracy", "accurate", "f1", "recall", "precision",
        "how good", "how well", "performance", "score", "correct",
        "wrong", "right", "reliable", "trust", "metric",
    ],
    "portfolio": [
        "all stocks", "portfolio", "summary", "all tickers",
        "overall", "everything", "every stock", "all of them",
        "most bullish", "most positive", "most negative",
        "which stocks", "oversold right now",
    ],
}

# ─────────────────────────────────────────────
# Detect tickers and intents
# ─────────────────────────────────────────────

_TICKER_RE = re.compile(
    r"\b(" + "|".join(re.escape(s) for s in STOCKS) + r")\b",
    re.IGNORECASE,
)

_NAME_TO_TICKER = {
    "apple":     "AAPL",
    "tesla":     "TSLA",
    "microsoft": "MSFT",
    "google":    "GOOGL",
    "alphabet":  "GOOGL",
    "amazon":    "AMZN",
    "nvidia":    "NVDA",
    "meta":      "META",
    "facebook":  "META",
    "netflix":   "NFLX",
    "alibaba":   "BABA",
    "amd":       "AMD",
}
_NAME_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _NAME_TO_TICKER) + r")\b",
    re.IGNORECASE,
)


def detect_tickers(message: str) -> list[str]:
    """Return list of STOCKS symbols mentioned in message (deduplicated, uppercased)."""
    found = _TICKER_RE.findall(message)
    for name in _NAME_RE.findall(message):
        ticker = _NAME_TO_TICKER.get(name.lower())
        if ticker:
            found.append(ticker)
    return list(dict.fromkeys(t.upper() for t in found))  # preserve order, dedupe


def detect_intents(message: str) -> list[str]:
    """Return list of intent labels that match keywords in message."""
    lower = message.lower()
    intents = [
        intent
        for intent, keywords in INTENT_KEYWORDS.items()
        if any(kw in lower for kw in keywords)
    ]
    return intents if intents else ["general"]


# ─────────────────────────────────────────────
# Data fetchers
# ─────────────────────────────────────────────

def _safe_float(val, default=None):
    if val is None:
        return default
    try:
        f = float(val)
        return default if f != f else f  # catches NaN
    except (TypeError, ValueError):
        return default


def fetch_prediction_context(ticker: str) -> dict:
    """
    Run the ML model for ticker across all horizons and return a compact summary.
    Returns {'error': ...} if model files are missing or data is unavailable.
    """
    try:
        from ml.explain import explain_prediction_for_api
        horizons = {}
        top_factors = []
        date = None
        for h in ["1d", "5d", "21d", "252d"]:
            result = explain_prediction_for_api(ticker, use_sentiment=True, horizon=h)
            if "error" in result:
                continue
            if date is None:
                date = result.get("date")
                top_factors = [
                    {
                        "label":     f["label"],
                        "value":     f["feature_value"],
                        "impact":    f["shap_value"],
                        "direction": f["direction"],
                    }
                    for f in result.get("factors", [])[:3]
                ]
            horizons[h] = {
                "prediction": result.get("prediction"),
                "confidence": result.get("confidence"),
            }
        if not horizons:
            return {"error": "No model results available"}
        return {
            "horizons":   horizons,
            "date":       date,
            "top_factors": top_factors,
        }
    except FileNotFoundError:
        return {"error": "Model files not found — run ml/train_model.py first"}
    except Exception as exc:
        return {"error": str(exc)}


def fetch_sentiment_context(ticker: str, limit: int = 15) -> dict:
    """
    Return recent news headlines + average sentiment score for a ticker.
    """
    conn = get_connection()
    sql = """
        SELECT date, headline, sentiment_score, sentiment_label
        FROM news_sentiment
        WHERE ticker = %s AND sentiment_score IS NOT NULL
        ORDER BY date DESC
        LIMIT %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (ticker, limit))
        rows = cur.fetchall()
    conn.close()

    if not rows:
        return {"ticker": ticker, "avg_sentiment": None, "headlines": []}

    scores = [float(r[2]) for r in rows]
    avg = round(sum(scores) / len(scores), 4)

    label = "POSITIVE" if avg > 0.05 else ("NEGATIVE" if avg < -0.05 else "NEUTRAL")

    headlines = [
        {
            "date":    str(r[0]),
            "text":    r[1],
            "score":   float(r[2]),
            "label":   r[3],
        }
        for r in rows[:5]  # most recent 5 headlines in context
    ]

    return {
        "ticker":           ticker,
        "avg_sentiment":    avg,
        "sentiment_label":  label,
        "recent_headlines": headlines,
    }


def fetch_price_context(ticker: str, days: int = 7) -> dict:
    """
    Return latest technical indicators + short price history for a ticker.
    """
    conn = get_connection()
    sql = """
        SELECT date, close,
               COALESCE(rsi_14, 0)          AS rsi_14,
               COALESCE(macd, 0)            AS macd,
               COALESCE(momentum_5, 0)      AS momentum_5,
               COALESCE(volume_change, 0)   AS volume_change,
               COALESCE(ma_20, 0)           AS ma_20
        FROM features
        WHERE ticker = %s
        ORDER BY date DESC
        LIMIT %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (ticker, days))
        rows = cur.fetchall()
    conn.close()

    if not rows:
        return {"ticker": ticker, "error": "No feature data in DB"}

    latest = rows[0]
    prices = [
        {"date": str(r[0]), "close": _safe_float(r[1], 0)}
        for r in reversed(rows)
    ]

    change_pct = 0.0
    if len(prices) >= 2 and prices[0]["close"] and prices[-1]["close"]:
        change_pct = round(
            (prices[-1]["close"] - prices[0]["close"]) / prices[0]["close"] * 100, 2
        )

    return {
        "ticker":             ticker,
        "latest_close":       _safe_float(latest[1]),
        "rsi_14":             _safe_float(latest[2]),
        "macd":               _safe_float(latest[3]),
        "momentum_5":         _safe_float(latest[4]),
        "volume_change_pct":  _safe_float(latest[5]),
        "ma_20":              _safe_float(latest[6]),
        "period_change_pct":  change_pct,
        "price_history":      prices,
    }


def fetch_portfolio_summary() -> list[dict]:
    """
    Lightweight summary for ALL 10 stocks: prediction + avg sentiment.
    Used when the user asks about the full portfolio.
    """
    summary = []
    for ticker in STOCKS:
        pred = fetch_prediction_context(ticker)
        sent = fetch_sentiment_context(ticker, limit=10)
        h1d = pred.get("horizons", {}).get("1d", {})
        summary.append({
            "ticker":          ticker,
            "prediction":      h1d.get("prediction", "N/A"),
            "confidence":      h1d.get("confidence"),
            "avg_sentiment":   sent.get("avg_sentiment"),
            "sentiment_label": sent.get("sentiment_label", "N/A"),
        })
    return summary


def fetch_metrics_context(ticker: str = None) -> str:
    """
    Load training_metrics.json and return a compact summary.
    If ticker is given, show that ticker only. Otherwise show averages.
    """
    import json
    metrics_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "ml", "models", "training_metrics.json")
    if not os.path.exists(metrics_path):
        return "Model metrics not available (run ml/eval_models.py to generate)."

    with open(metrics_path) as f:
        data = json.load(f)

    horizons = data.get("horizons", {})
    horizon_labels = {"1d": "1 day", "5d": "1 week", "21d": "1 month", "252d": "1 year"}
    lines = ["MODEL PERFORMANCE METRICS (accuracy = % of correct UP/DOWN calls on unseen test data):"]

    for h, lbl in horizon_labels.items():
        h_data = horizons.get(h, {})
        ns_rows = {r["ticker"]: r for r in h_data.get("no_sentiment", [])}
        ws_rows = {r["ticker"]: r for r in h_data.get("with_sentiment", [])}

        if ticker:
            ns = ns_rows.get(ticker)
            ws = ws_rows.get(ticker)
            if ns and ws:
                lines.append(
                    f"  {lbl}: Without news sentiment → {ns['accuracy']*100:.1f}% accurate, F1 {ns['f1']*100:.1f}% | "
                    f"With news sentiment → {ws['accuracy']*100:.1f}% accurate, F1 {ws['f1']*100:.1f}%"
                )
        else:
            if ns_rows and ws_rows:
                avg_ns_acc = sum(r["accuracy"] for r in ns_rows.values()) / len(ns_rows)
                avg_ws_acc = sum(r["accuracy"] for r in ws_rows.values()) / len(ws_rows)
                avg_ns_f1  = sum(r["f1"] for r in ns_rows.values()) / len(ns_rows)
                avg_ws_f1  = sum(r["f1"] for r in ws_rows.values()) / len(ws_rows)
                lines.append(
                    f"  {lbl} (avg across all stocks): Without sentiment → {avg_ns_acc*100:.1f}% acc, F1 {avg_ns_f1*100:.1f}% | "
                    f"With sentiment → {avg_ws_acc*100:.1f}% acc, F1 {avg_ws_f1*100:.1f}%"
                )

    lines.append("\nNote: Accuracy = how often the UP/DOWN call was correct. F1 = balance between catching UPs and DOWNs (0-100%, higher is better). These are measured on data the model has never seen before.")
    return "\n".join(lines)


# ─────────────────────────────────────────────
# Context builder — called by /chat endpoint
# ─────────────────────────────────────────────

def build_context(message: str) -> str:
    """
    Parse the user message, fetch relevant data, and return a
    compact text block that gets injected into the system prompt.
    """
    tickers = detect_tickers(message)
    intents = detect_intents(message)

    is_portfolio = "portfolio" in intents and not tickers

    sections: list[str] = []

    # ── Portfolio-level query ──────────────────
    if is_portfolio:
        rows = fetch_portfolio_summary()
        lines = ["ALL STOCKS PORTFOLIO SUMMARY:"]
        lines.append(f"{'Ticker':<6} {'Prediction':<10} {'Confidence':<12} {'Avg Sentiment':<16} {'Label'}")
        lines.append("-" * 55)
        for r in rows:
            conf_str = f"{r['confidence']*100:.1f}%" if r.get("confidence") else "N/A"
            sent_str = f"{r['avg_sentiment']:+.4f}" if r.get("avg_sentiment") is not None else "N/A"
            lines.append(
                f"{r['ticker']:<6} {r['prediction']:<10} {conf_str:<12} {sent_str:<16} {r['sentiment_label']}"
            )
        sections.append("\n".join(lines))
        return "\n\n".join(sections)

    # ── Per-ticker query ───────────────────────
    if not tickers:
        # No specific ticker found — fetch portfolio summary as fallback
        rows = fetch_portfolio_summary()
        lines = ["NO SPECIFIC TICKER DETECTED — SHOWING ALL STOCKS:"]
        for r in rows:
            conf_str = f"{r['confidence']*100:.1f}%" if r.get("confidence") else "N/A"
            sent_str = f"{r['avg_sentiment']:+.4f}" if r.get("avg_sentiment") is not None else "N/A"
            lines.append(
                f"  {r['ticker']}: {r['prediction']} ({conf_str}), "
                f"Sentiment={sent_str} ({r['sentiment_label']})"
            )
        sections.append("\n".join(lines))
        return "\n\n".join(sections)

    for ticker in tickers:
        block = [f"━━━ {ticker} ━━━"]

        # Prediction + explain
        if any(i in intents for i in ["prediction", "explain", "compare", "general"]):
            pred = fetch_prediction_context(ticker)
            if "error" in pred:
                block.append(f"PREDICTION: unavailable ({pred['error']})")
            else:
                block.append(f"PREDICTIONS (as of {pred['date']}):")
                labels = {"1d": "1 day", "5d": "1 week", "21d": "1 month", "252d": "1 year"}
                for h, lbl in labels.items():
                    h_data = pred.get("horizons", {}).get(h)
                    if h_data:
                        conf_pct = f"{h_data['confidence']*100:.1f}%" if h_data.get("confidence") else "N/A"
                        block.append(f"  {lbl} ({h}): {h_data['prediction']} | Confidence: {conf_pct}")
                for f in pred.get("top_factors", []):
                    sign = "+" if f["impact"] > 0 else ""
                    block.append(
                        f"  • {f['label']}: {f['value']:.4f}  "
                        f"(SHAP impact: {sign}{f['impact']:.4f}, pushes {f['direction']})"
                    )

        # Sentiment + headlines
        if any(i in intents for i in ["sentiment", "general"]):
            sent = fetch_sentiment_context(ticker)
            if sent.get("avg_sentiment") is not None:
                block.append(
                    f"SENTIMENT: avg={sent['avg_sentiment']:+.4f} ({sent['sentiment_label']})"
                )
                for h in sent.get("recent_headlines", []):
                    block.append(f"  [{h['date']}] {h['text'][:90]}  ({h['label']}, {h['score']:+.3f})")
            else:
                block.append("SENTIMENT: no data in DB")

        # Price + technicals
        if any(i in intents for i in ["price", "general"]):
            price = fetch_price_context(ticker)
            if "error" in price:
                block.append(f"PRICE: unavailable ({price['error']})")
            else:
                block.append(f"PRICE: ${price['latest_close']:.2f} | 7-day change: {price['period_change_pct']:+.2f}%")
                block.append(
                    f"RSI(14): {price['rsi_14']:.2f} | "
                    f"MACD: {price['macd']:.4f} | "
                    f"MA(20): ${price['ma_20']:.2f} | "
                    f"Volume Δ: {price['volume_change_pct']:+.2f}%"
                )

        # Model metrics
        if "metrics" in intents:
            block.append(fetch_metrics_context(ticker))

        sections.append("\n".join(block))

    return "\n\n".join(sections)
