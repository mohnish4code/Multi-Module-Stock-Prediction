"""Headline sentiment: FinBERT by default, keyword lexicon as fallback."""

from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

FINBERT_ID = "ProsusAI/finbert"


@lru_cache(maxsize=1)
def _finbert():
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(FINBERT_ID)
    model = AutoModelForSequenceClassification.from_pretrained(FINBERT_ID).eval()
    labels = {v.lower(): int(k) for k, v in model.config.id2label.items()}
    return tokenizer, model, labels


def finbert_scores(titles: list[str], batch_size: int = 64) -> np.ndarray:
    """P(positive) - P(negative) per headline, in [-1, 1]."""
    import torch

    if not titles:
        return np.zeros(0)
    tokenizer, model, labels = _finbert()
    out = []
    with torch.no_grad():
        for i in range(0, len(titles), batch_size):
            enc = tokenizer(titles[i : i + batch_size], padding=True, truncation=True, max_length=64,
                            return_tensors="pt")
            probs = torch.softmax(model(**enc).logits, dim=-1).numpy()
            out.append(probs[:, labels["positive"]] - probs[:, labels["negative"]])
    return np.concatenate(out)

POSITIVE = frozenset([
    "gain", "gains", "growth", "profit", "profits", "profitability", "surge", "surges", "rally", "rallies",
    "rise", "rises", "rising", "upgrade", "upgraded", "strong", "record", "beat", "beats", "outperform",
    "outperformance", "bullish", "positive", "expansion", "recovery", "rebound", "boost", "boosts", "higher",
    "improve", "improved", "improves", "success", "successful",
])

NEGATIVE = frozenset([
    "loss", "losses", "fall", "falls", "falling", "decline", "declines", "drop", "drops", "plunge", "plunges",
    "crash", "crashes", "weak", "downgrade", "downgraded", "bearish", "negative", "lower", "risk", "risks",
    "concern", "concerns", "warning", "slowdown", "lawsuit", "penalty", "fraud", "investigation", "debt",
    "pressure", "miss", "misses",
])


def score_headline(headline: str) -> tuple[str, int]:
    words = set(re.findall(r"[a-z]+", headline.lower()))
    score = len(words & POSITIVE) - len(words & NEGATIVE)
    return ("POSITIVE" if score > 0 else "NEGATIVE" if score < 0 else "NEUTRAL"), score


def _label(score: float, band: float) -> str:
    return "POSITIVE" if score > band else "NEGATIVE" if score < -band else "NEUTRAL"


def _finbert_or_none(titles: list[str]) -> list[float] | None:
    try:
        return [float(s) for s in finbert_scores(titles)]
    except Exception as exc:  # noqa: BLE001 - model download/transformers unavailable -> keyword fallback
        logger.warning("FinBERT unavailable (%s); using keyword sentiment", exc)
        return None


def summarise(items: list[dict[str, Any]], method: str = "finbert", band: float = 0.20) -> dict[str, Any]:
    """Labels every headline and aggregates them.

    finbert: per-headline score = P(pos) - P(neg); overall = mean score.
    keyword: per-headline score = positive - negative word count; overall = (pos - neg headlines) / total.
    Overall label is POSITIVE / NEGATIVE beyond +/- band, else NEUTRAL.
    """
    titles = [it["title"] for it in items]
    scores = _finbert_or_none(titles) if method == "finbert" and titles else None
    if scores is not None:
        scored = [{**it, "sentiment": _label(s, 0.3), "score": s} for it, s in zip(items, scores, strict=True)]
        used = "finbert"
    else:
        scored = [{**it, "sentiment": lab, "score": float(s)} for it in items
                  for lab, s in [score_headline(it["title"])]]
        used = "keyword"

    counts = {k: sum(s["sentiment"] == k for s in scored) for k in ("POSITIVE", "NEUTRAL", "NEGATIVE")}
    total = len(scored)
    if not total:
        overall = 0.0
    elif used == "finbert":
        overall = sum(s["score"] for s in scored) / total
    else:
        overall = (counts["POSITIVE"] - counts["NEGATIVE"]) / total
    label = _label(overall, band)
    return {
        "method": used,
        "total_news": total,
        "positive": counts["POSITIVE"],
        "neutral": counts["NEUTRAL"],
        "negative": counts["NEGATIVE"],
        "sentiment_score": overall,
        "sentiment_label": label if total else "NEUTRAL",
        "most_recent": scored[0]["published"] if scored else None,
        "newest_age_days": scored[0]["age_days"] if scored else None,
        "news": scored,
    }
