# api/main.py
# ─────────────────────────────────────────────
# Phase 6 — FastAPI Backend
#
# Endpoints:
#   GET /stocks/{ticker}      → price history
#   GET /news/{ticker}        → sentiment headlines
#   GET /predict/{ticker}     → UP/DOWN prediction
#   GET /explain/{ticker}     → SHAP explanation
#   GET /compare/{ticker}     → both models side by side
#   GET /tickers              → list of available stocks
# ─────────────────────────────────────────────

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sys
import os
import time
import anthropic
from groq import Groq

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_pipeline.db_connection import get_connection
from ml.explain import explain_prediction_for_api, load_model, load_features_from_db
from config.settings import STOCKS, ANTHROPIC_KEY, GROQ_KEY
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__))))
from chat_context import build_context, detect_tickers, detect_intents, fetch_sentiment_context

# ── In-memory SHAP cache (30-min TTL) ─────────
# explain_prediction_for_api runs TreeExplainer on every call; results
# only change after nightly retrain (23:00 UTC), so caching 30 min is safe.
_EXPLAIN_CACHE: dict = {}
_EXPLAIN_TTL = 1800  # seconds

def _cached_explain(ticker: str, use_sentiment: bool, horizon: str) -> dict:
    key = (ticker, use_sentiment, horizon)
    entry = _EXPLAIN_CACHE.get(key)
    if entry and (time.time() - entry["ts"]) < _EXPLAIN_TTL:
        return entry["data"]
    result = explain_prediction_for_api(ticker, use_sentiment, horizon)
    _EXPLAIN_CACHE[key] = {"data": result, "ts": time.time()}
    return result

app = FastAPI(
    title="Stock Predictor API",
    description="AI-powered stock price direction predictor with SHAP explainability",
    version="1.0.0",
)

# ── Allow React frontend to call this API ─────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────

def validate_ticker(ticker: str):
    ticker = ticker.upper()
    if ticker not in STOCKS:
        raise HTTPException(
            status_code=404,
            detail=f"Ticker '{ticker}' not found. Available: {STOCKS}"
        )
    return ticker


def _sanitize(obj):
    """Recursively replace NaN/Inf floats with None so JSON serialisation never crashes."""
    import math
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


# ─────────────────────────────────────────────
# GET /tickers
# ─────────────────────────────────────────────

@app.get("/tickers")
def get_tickers():
    """Return list of all available stock tickers."""
    return {"tickers": STOCKS}


# ─────────────────────────────────────────────
# GET /stocks/{ticker}
# ─────────────────────────────────────────────

@app.get("/stocks/{ticker}")
def get_stock_prices(ticker: str, days: int = 90):
    """
    Return recent stock price history for a ticker.
    Query param: days (default 90)
    """
    ticker = validate_ticker(ticker)
    conn   = get_connection()

    sql = """
        SELECT date, open, high, low, close, volume
        FROM stock_prices
        WHERE ticker = %s
        ORDER BY date DESC
        LIMIT %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (ticker, days))
        rows = cur.fetchall()
    conn.close()

    if not rows:
        raise HTTPException(status_code=404, detail=f"No price data for {ticker}")

    data = [
        {
            "date":   str(r[0]),
            "open":   float(r[1]),
            "high":   float(r[2]),
            "low":    float(r[3]),
            "close":  float(r[4]),
            "volume": int(r[5]),
        }
        for r in reversed(rows)   # oldest first for charting
    ]

    return {
        "ticker": ticker,
        "days":   days,
        "count":  len(data),
        "prices": data,
    }


# ─────────────────────────────────────────────
# GET /news/{ticker}
# ─────────────────────────────────────────────

@app.get("/news/{ticker}")
def get_news_sentiment(ticker: str, days: int = 30):
    """
    Return recent news headlines with sentiment scores for a ticker.
    """
    ticker = validate_ticker(ticker)
    conn   = get_connection()

    sql = """
        SELECT date, headline, sentiment_score, sentiment_label, source
        FROM news_sentiment
        WHERE ticker = %s AND sentiment_score IS NOT NULL
        ORDER BY date DESC
        LIMIT %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (ticker, days))
        rows = cur.fetchall()
    conn.close()

    if not rows:
        return {"ticker": ticker, "count": 0, "news": []}

    news = [
        {
            "date":            str(r[0]),
            "headline":        r[1],
            "sentiment_score": float(r[2]),
            "sentiment_label": r[3],
            "source":          r[4],
        }
        for r in rows
    ]

    # Daily average sentiment
    from collections import defaultdict
    daily = defaultdict(list)
    for item in news:
        daily[item["date"]].append(item["sentiment_score"])
    daily_avg = [
        {"date": d, "avg_sentiment": round(sum(v)/len(v), 4)}
        for d, v in sorted(daily.items())
    ]

    return {
        "ticker":        ticker,
        "count":         len(news),
        "news":          news,
        "daily_average": daily_avg,
    }


# ─────────────────────────────────────────────
# GET /predict/{ticker}
# ─────────────────────────────────────────────

@app.get("/predict/{ticker}")
def get_prediction(ticker: str, use_sentiment: bool = True, horizon: str = "1d"):
    """
    Return UP/DOWN prediction for a ticker.
    Query params:
      use_sentiment : bool  (default True)
      horizon       : str   1d | 5d | 21d | 252d  (default 1d)
    """
    ticker = validate_ticker(ticker)

    try:
        result = explain_prediction_for_api(ticker, use_sentiment, horizon)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return _sanitize(result)
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="Model not found. Please run ml/train_model.py first."
        )


# ─────────────────────────────────────────────
# GET /explain/{ticker}
# ─────────────────────────────────────────────

@app.get("/explain/{ticker}")
def get_explanation(ticker: str, use_sentiment: bool = True):
    """
    Return full SHAP explanation for a ticker's latest prediction.
    Includes plain English summary and top feature drivers.
    """
    ticker = validate_ticker(ticker)

    try:
        result = _cached_explain(ticker, use_sentiment, "1d")
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return _sanitize(result)
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="Model not found. Please run ml/train_model.py first."
        )


# ─────────────────────────────────────────────
# GET /compare/{ticker}
# ─────────────────────────────────────────────

@app.get("/compare/{ticker}")
def compare_models(ticker: str, horizon: str = "1d"):
    """
    Return predictions from BOTH models side by side.
    Used by the React dashboard to show the thesis comparison.
    """
    ticker = validate_ticker(ticker)

    try:
        no_sent   = _cached_explain(ticker, use_sentiment=False, horizon=horizon)
        with_sent = _cached_explain(ticker, use_sentiment=True,  horizon=horizon)

        return _sanitize({
            "ticker": ticker,
            "horizon": horizon,
            "model_without_sentiment": {
                "prediction": no_sent.get("prediction"),
                "confidence": no_sent.get("confidence"),
                "top_factors": no_sent.get("factors", [])[:3],
                "plain_english": no_sent.get("plain_english"),
            },
            "model_with_sentiment": {
                "prediction": with_sent.get("prediction"),
                "confidence": with_sent.get("confidence"),
                "top_factors": with_sent.get("factors", [])[:3],
                "plain_english": with_sent.get("plain_english"),
            },
        })
    except FileNotFoundError:
        raise HTTPException(
            status_code=500,
            detail="Models not found. Please run ml/train_model.py first."
        )


# ─────────────────────────────────────────────
# GET /features/{ticker}
# ─────────────────────────────────────────────

@app.get("/features/{ticker}")
def get_features(ticker: str, days: int = 60):
    ticker = validate_ticker(ticker)
    conn   = get_connection()

    sql = """
        SELECT date, close, ma_5, ma_10, ma_20,
               volatility_7, momentum_5, sentiment_lag1, target
        FROM features
        WHERE ticker = %s
          AND ma_20 IS NOT NULL
          AND volatility_7 IS NOT NULL
        ORDER BY date DESC
        LIMIT %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (ticker, days))
        rows = cur.fetchall()
    conn.close()

    def safe_float(val):
        if val is None:
            return 0.0
        try:
            f = float(val)
            return 0.0 if f != f else f  # catches NaN
        except:
            return 0.0

    data = [
        {
            "date":         str(r[0]),
            "close":        safe_float(r[1]),
            "ma_5":         safe_float(r[2]),
            "ma_10":        safe_float(r[3]),
            "ma_20":        safe_float(r[4]),
            "volatility_7": safe_float(r[5]),
            "momentum_5":   safe_float(r[6]),
            "sentiment":    safe_float(r[7]),
            "target":       int(r[8]) if r[8] is not None else 0,
        }
        for r in reversed(rows)
    ]

    return {"ticker": ticker, "count": len(data), "features": data}


# ─────────────────────────────────────────────
# GET /analysis/{ticker}
# ─────────────────────────────────────────────

ANALYSIS_PROMPT = """You are a stock analyst writing a brief analysis for everyday people — not finance professionals.

Based on the data below, write an analysis in exactly this JSON format:
{{
  "main_reason": "1-2 sentences explaining the biggest reason for this prediction in simple terms anyone can understand. If you mention a technical indicator, explain what it means in plain words (e.g. 'the stock has been losing momentum over the past few days, meaning it has been slowing down').",
  "news_impact": "1-2 sentences explaining how recent news is helping or hurting the outlook. Refer to actual headlines and explain their effect simply.",
  "confidence_reason": "1 sentence explaining in plain terms why the model is very sure / fairly sure / unsure (e.g. 'The model is very confident because the same pattern has repeated consistently over the past few months')."
}}

Rules:
- No jargon without explanation — if you mention RSI, MACD, sentiment score, or SHAP, explain what it means in simple terms
- Do NOT give financial advice
- Return ONLY the JSON, no extra text

STOCK DATA:
Ticker     : {ticker}
Prediction : {direction} (confidence: {confidence}%)

TOP FACTORS (what drove the prediction):
{factors}

CURRENT INDICATORS:
{indicators}

RECENT NEWS (sentiment: {sentiment_score} — {sentiment_label}):
{headlines}"""


@app.get("/analysis/{ticker}")
def get_analysis(ticker: str, horizon: str = "1d"):
    """
    Auto-generate a plain English analysis for a ticker.
    Returns 3 sections: main_reason, news_impact, confidence_reason.
    """
    ticker = ticker.upper()
    if ticker not in STOCKS:
        raise HTTPException(status_code=404, detail=f"Unknown ticker: {ticker}")

    # 1. Get prediction + SHAP factors
    explanation = _cached_explain(ticker, use_sentiment=True, horizon=horizon)
    if "error" in explanation:
        raise HTTPException(status_code=500, detail=explanation["error"])

    direction  = explanation["prediction"]
    confidence = round(explanation["confidence"] * 100, 1)
    factors    = explanation.get("factors", [])

    # Format top 3 SHAP factors
    factors_text = "\n".join(
        f"  - {f['label']}: {f['feature_value']:.4f} "
        f"({'pushed UP' if f['shap_value'] > 0 else 'pushed DOWN'}, "
        f"impact: {f['shap_value']:+.4f})"
        for f in factors[:3]
    )

    # Format current indicators (from factors)
    indicator_map = {f["feature"]: f["feature_value"] for f in factors}
    indicators_text = "\n".join(
        f"  - {k}: {v:.4f}" for k, v in list(indicator_map.items())[:5]
    )

    # 2. Get recent news context
    sentiment_data = fetch_sentiment_context(ticker, limit=5)
    avg_sent  = sentiment_data.get("avg_sentiment") or 0
    sent_label = sentiment_data.get("sentiment_label", "NEUTRAL")
    headlines  = sentiment_data.get("recent_headlines", [])
    headlines_text = "\n".join(
        f"  - [{h['date']}] {h['text']}" for h in headlines[:4]
    ) or "  - No recent headlines available"

    # 3. Call Groq
    prompt = ANALYSIS_PROMPT.format(
        ticker=ticker,
        direction=direction,
        confidence=confidence,
        factors=factors_text,
        indicators=indicators_text,
        sentiment_score=round(avg_sent, 3),
        sentiment_label=sent_label,
        headlines=headlines_text,
    )

    try:
        groq_client = Groq(api_key=GROQ_KEY)
        resp = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=300,
            temperature=0.3,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.choices[0].message.content.strip()

        # Parse JSON response
        import json as _json
        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        analysis = _json.loads(raw.strip())

    except Exception as e:
        # Fallback to basic text if Groq fails
        analysis = {
            "main_reason":       explanation.get("plain_english", "Analysis unavailable."),
            "news_impact":       f"Sentiment is {sent_label.lower()} ({avg_sent:.2f}).",
            "confidence_reason": f"Model confidence is {confidence}%.",
        }

    return {
        "ticker":     ticker,
        "prediction": direction,
        "confidence": confidence,
        "analysis":   analysis,
    }


# ─────────────────────────────────────────────
# POST /chat
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are an AI stock analyst assistant for StockPredictor — a personal AI-powered trading dashboard.

You have access to LIVE data pulled directly from the user's database (prices, predictions, sentiment, SHAP explanations).
This data is injected below under "CURRENT DATA". Always base your answers on it.

Guidelines:
- Answer only using the data provided — never invent numbers or cite external sources
- Always explain in plain English that anyone can understand — avoid raw jargon
- When you must use a technical term, explain it simply in the same sentence:
    e.g. "The MACD (a momentum indicator — think of it as measuring whether a stock is speeding up or slowing down) is positive"
    e.g. "94% confidence means the model has seen very similar patterns before and they almost always ended this way"
    e.g. "Sentiment +0.15 means the recent news about this stock is slightly positive overall"
    e.g. "The key factors (the main clues the model used to make its decision) are..."
- Format predictions as: ↑ UP (confidence: X%) or ↓ DOWN (confidence: X%)
- Keep answers conversational and friendly — imagine explaining to someone with no finance background
- When comparing multiple stocks, use a short table or bullet list
- If data is missing for something the user asked, say so honestly
- Never provide financial advice — this is for educational/analytical purposes only
"""


class ChatMessage(BaseModel):
    role: str   # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []


@app.post("/chat")
def chat(req: ChatRequest):
    """
    Chat endpoint — answers questions about your real stock data using Claude.

    Body:
        message  : the user's latest question
        history  : prior conversation turns (role + content)

    Returns:
        reply    : Claude's response string
        tickers  : which tickers were detected in the message
        intents  : which intents were detected
    """
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # 1. Fetch live data context based on what the user asked
    tickers = detect_tickers(req.message)
    # If no ticker in current message, scan recent history for context
    if not tickers:
        for msg in reversed(req.history[-6:]):
            found = detect_tickers(msg.content)
            if found:
                tickers = found
                break
    intents = detect_intents(req.message)
    # Build context using current message, but inject history-detected tickers if needed
    effective_message = req.message if tickers == detect_tickers(req.message) else (req.message + " " + " ".join(tickers))
    data_context = build_context(effective_message)

    # 2. Build system prompt with injected data
    full_system = (
        SYSTEM_PROMPT
        + "\n\n─── CURRENT DATA ───\n"
        + data_context
        + "\n────────────────────"
    )

    # 3. Build messages array (history + new user message)
    messages = [
        {"role": msg.role, "content": msg.content}
        for msg in req.history
    ]
    messages.append({"role": "user", "content": req.message})

    # 4. Call Groq (Llama 3.3 70B — free, fast)
    client = Groq(api_key=GROQ_KEY)
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=1024,
            messages=[{"role": "system", "content": full_system}] + messages,
        )
    except Exception as exc:
        detail = str(exc)
        if "invalid_api_key" in detail.lower() or "authentication" in detail.lower():
            raise HTTPException(status_code=401, detail="Invalid Groq API key in config/settings.py")
        if "connection" in detail.lower():
            raise HTTPException(status_code=503, detail="Could not reach Groq API — check your internet connection")
        raise HTTPException(status_code=500, detail=f"Groq error: {detail}")

    reply = response.choices[0].message.content

    return {
        "reply":   reply,
        "tickers": tickers,
        "intents": intents,
    }


@app.get("/")
def root():
    return {
        "message": "Stock Predictor API is running",
        "version": "1.0.0",
        "endpoints": [
            "/tickers",
            "/stocks/{ticker}",
            "/news/{ticker}",
            "/predict/{ticker}",
            "/explain/{ticker}",
            "/compare/{ticker}",
            "/features/{ticker}",
        ]
    }