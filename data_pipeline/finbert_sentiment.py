# data_pipeline/finbert_sentiment.py
# ─────────────────────────────────────────────
# Shared FinBERT sentiment scorer.
# Loads the model once (lazy) and scores batches locally — no API, no limits.
#
# Install deps once:
#   pip install transformers torch
# ─────────────────────────────────────────────

from __future__ import annotations

_pipeline = None   # lazy-loaded singleton


def _get_pipeline():
    global _pipeline
    if _pipeline is None:
        from transformers import pipeline
        print("  [FinBERT] Loading model (first run — ~30 s) ...")
        _pipeline = pipeline(
            "text-classification",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            top_k=None,           # return all 3 label scores
            device=-1,            # CPU (-1); change to 0 for GPU
            truncation=True,
            max_length=512,
        )
        print("  [FinBERT] Model ready.")
    return _pipeline


def score_headlines(headlines: list[str]) -> list[float]:
    """
    Score a list of financial headlines with FinBERT.
    Returns float scores in [-1.0, +1.0]:
        score = P(positive) – P(negative)
    Falls back to 0.0 for any headline that errors.
    """
    if not headlines:
        return []

    nlp = _get_pipeline()
    scores: list[float] = []

    try:
        results = nlp(headlines, batch_size=16)
        for result in results:
            probs = {item["label"].lower(): item["score"] for item in result}
            s = probs.get("positive", 0.0) - probs.get("negative", 0.0)
            scores.append(round(max(-1.0, min(1.0, s)), 4))
    except Exception as e:
        print(f"  [FinBERT] Scoring error: {e} — returning zeros")
        scores = [0.0] * len(headlines)

    return scores
