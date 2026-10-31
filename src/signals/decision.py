"""The single decision engine used by every entry point (CLI, app, API).

Combines the model forecast, the technical read and news sentiment into a
BUY / HOLD / SELL call. The agreement, confidence and risk scores are
rule-based heuristics, not calibrated probabilities.
"""

from __future__ import annotations

from typing import Any

SENTIMENT_TO_DIRECTION = {"POSITIVE": "BULLISH", "NEGATIVE": "BEARISH"}


def classify_direction(pct_change: float, threshold_pct: float) -> str:
    if pct_change > threshold_pct:
        return "BULLISH"
    if pct_change < -threshold_pct:
        return "BEARISH"
    return "NEUTRAL"


def normalise_sentiment(label: str | None) -> str:
    label = str(label or "").upper().strip()
    return label if label in ("POSITIVE", "NEGATIVE", "NEUTRAL") else "NEUTRAL"


def coordinate(model_direction: str, technical_signal: str, news_sentiment: str) -> dict[str, Any]:
    votes = [model_direction, technical_signal, SENTIMENT_TO_DIRECTION.get(news_sentiment, "NEUTRAL")]
    bull, bear = votes.count("BULLISH"), votes.count("BEARISH")
    signal = "BULLISH" if bull > bear else "BEARISH" if bear > bull else "NEUTRAL"
    agreement = "STRONG" if max(bull, bear) == 3 else "MODERATE" if max(bull, bear) == 2 else "MIXED"
    return {
        "coordinated_signal": signal,
        "agreement": agreement,
        "bullish_modules": bull,
        "bearish_modules": bear,
        "neutral_modules": 3 - bull - bear,
    }


def confidence(model_direction: str, technical_signal: str, news_sentiment: str, agreement: str) -> dict[str, Any]:
    score = 50
    if technical_signal == model_direction:
        score += 15
    elif technical_signal != "NEUTRAL":
        score -= 10

    news_direction = SENTIMENT_TO_DIRECTION.get(news_sentiment, "NEUTRAL")
    if news_direction != "NEUTRAL":
        score += 15 if news_direction == model_direction else -10

    score += {"STRONG": 20, "MODERATE": 10, "MIXED": -5}[agreement]
    score = max(0, min(100, score))
    level = "HIGH" if score >= 80 else "MODERATE" if score >= 60 else "LOW"
    return {"confidence_score": score, "confidence_level": level}


def risk(
    volatility_level: str, predicted_change_pct: float, technical_signal: str, confidence_score: int, agreement: str
) -> dict[str, Any]:
    score, factors = 0, []

    score += {"HIGH": 3, "MEDIUM": 2}.get(volatility_level, 1)
    if volatility_level in ("HIGH", "MEDIUM"):
        factors.append(f"{volatility_level.title()} market volatility is present.")

    move = abs(predicted_change_pct)
    if move >= 5:
        score += 3
        factors.append("A large predicted price movement is expected.")
    elif move >= 2:
        score += 2
        factors.append("A moderate predicted price movement is expected.")
    else:
        score += 1

    if technical_signal == "NEUTRAL":
        score += 2
        factors.append("Technical indicators provide mixed confirmation.")

    if confidence_score < 50:
        score += 3
        factors.append("Low signal agreement.")
    elif confidence_score < 70:
        score += 2
        factors.append("Moderate signal agreement.")

    if agreement == "MIXED":
        score += 2
        factors.append("The modules disagree.")
    elif agreement == "MODERATE":
        score += 1

    score = min(score, 10)
    level = "HIGH" if score >= 8 else "MEDIUM" if score >= 5 else "LOW"
    return {"risk_score": score, "risk_level": level, "risk_factors": factors}


def decide(
    model_direction: str,
    predicted_change_pct: float,
    technical_signal: str,
    news_sentiment: str,
    volatility_level: str,
) -> dict[str, Any]:
    """Full decision: coordination -> confidence -> risk -> BUY/HOLD/SELL with an explanation."""
    news_sentiment = normalise_sentiment(news_sentiment)
    coord = coordinate(model_direction, technical_signal, news_sentiment)
    conf = confidence(model_direction, technical_signal, news_sentiment, coord["agreement"])
    rsk = risk(volatility_level, predicted_change_pct, technical_signal, conf["confidence_score"], coord["agreement"])

    signal = coord["coordinated_signal"]
    if rsk["risk_level"] == "HIGH" or conf["confidence_score"] < 50 or signal == "NEUTRAL":
        recommendation = "HOLD"
    else:
        recommendation = "BUY" if signal == "BULLISH" else "SELL"

    explanation = (
        f"The model forecasts a {predicted_change_pct:+.2f}% move ({model_direction}). "
        f"Technical analysis is {technical_signal} and news sentiment is {news_sentiment}, "
        f"giving a {signal} coordinated signal with {coord['agreement'].lower()} agreement. "
        f"Agreement score {conf['confidence_score']}/100, risk {rsk['risk_level'].lower()} "
        f"({rsk['risk_score']}/10): {recommendation}."
    )
    if recommendation == "HOLD" and signal != "NEUTRAL":
        explanation += " The call is held back because agreement is low or risk is high."

    return {**coord, **conf, **rsk, "recommendation": recommendation, "explanation": explanation}
