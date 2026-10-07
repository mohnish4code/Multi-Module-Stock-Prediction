import re

from modules.news_collection import get_stock_news


# =================================================
# NEWS SENTIMENT  (keyword-based)
#
# get_stock_news() (modules/news_collection.py) does
# the real-time Google News RSS fetch. This module
# scores the headlines it returns.
# =================================================


# =================================================
# SIMPLE FINANCIAL SENTIMENT KEYWORDS
# =================================================

POSITIVE_KEYWORDS = [

    "gain",
    "gains",
    "growth",
    "profit",
    "profits",
    "profitability",
    "surge",
    "surges",
    "rally",
    "rallies",
    "rise",
    "rises",
    "rising",
    "upgrade",
    "upgraded",
    "strong",
    "record",
    "beat",
    "beats",
    "outperform",
    "outperformance",
    "bullish",
    "positive",
    "expansion",
    "recovery",
    "rebound",
    "boost",
    "boosts",
    "higher",
    "improve",
    "improved",
    "improves",
    "success",
    "successful"
]


NEGATIVE_KEYWORDS = [

    "loss",
    "losses",
    "fall",
    "falls",
    "falling",
    "decline",
    "declines",
    "drop",
    "drops",
    "plunge",
    "plunges",
    "crash",
    "crashes",
    "weak",
    "downgrade",
    "downgraded",
    "bearish",
    "negative",
    "lower",
    "risk",
    "risks",
    "concern",
    "concerns",
    "warning",
    "slowdown",
    "lawsuit",
    "penalty",
    "fraud",
    "investigation",
    "debt",
    "pressure",
    "miss",
    "misses"
]


# =================================================
# ANALYZE ONE HEADLINE
# =================================================

def analyze_headline_sentiment(
    headline
):
    """
    Performs keyword-based sentiment analysis
    on a single news headline.
    """

    headline_lower = headline.lower()


    # -------------------------------------------------
    # TOKENIZE HEADLINE
    # -------------------------------------------------

    words = re.findall(
        r"\b[a-zA-Z]+\b",
        headline_lower
    )


    positive_count = 0

    negative_count = 0


    # -------------------------------------------------
    # COUNT POSITIVE KEYWORDS
    # -------------------------------------------------

    for word in POSITIVE_KEYWORDS:

        if word in words:

            positive_count += 1


    # -------------------------------------------------
    # COUNT NEGATIVE KEYWORDS
    # -------------------------------------------------

    for word in NEGATIVE_KEYWORDS:

        if word in words:

            negative_count += 1


    # -------------------------------------------------
    # CALCULATE SCORE
    # -------------------------------------------------

    score = (
        positive_count
        - negative_count
    )


    # -------------------------------------------------
    # DETERMINE SENTIMENT
    # -------------------------------------------------

    if score > 0:

        sentiment = "POSITIVE"

    elif score < 0:

        sentiment = "NEGATIVE"

    else:

        sentiment = "NEUTRAL"


    return sentiment, score


# =================================================
# ANALYZE COMPANY NEWS SENTIMENT
# =================================================

def analyze_company_sentiment(
    company_name,
    ticker=None,
    max_results=None
):
    """
    Fetches the freshest available company news and
    runs keyword sentiment analysis over it.

    The returned dict carries `fetched_at` and
    `most_recent` so the UI can prove the feed is live.
    """

    import datetime as _dt

    # Prefer the ticker so the curated Google News query
    # is actually hit (callers pass the long display
    # name, which never matches the short keys).
    lookup = (
        ticker.replace(".NS", "") if ticker else company_name
    )

    news = get_stock_news(
        company_name=lookup,
        max_news=max_results,
    )

    fetched_at = _dt.datetime.now().strftime("%d %b %Y, %H:%M")

    # =================================================
    # HANDLE NO RELEVANT NEWS
    # =================================================

    if not news:

        return {

            "company": company_name,

            "total_news": 0,

            "positive": 0,

            "neutral": 0,

            "negative": 0,

            "sentiment_score": 0.0,

            "sentiment_label": "UNKNOWN",

            "fetched_at": fetched_at,

            "most_recent": None,

            "newest_age_days": None,

            "news": []

        }


    # =================================================
    # SENTIMENT COUNTERS
    # =================================================

    positive_count = 0

    neutral_count = 0

    negative_count = 0

    analyzed_news = []


    # =================================================
    # ANALYZE EACH HEADLINE
    # =================================================

    for item in news:

        headline = item.get(
            "title",
            ""
        )


        sentiment, score = (
            analyze_headline_sentiment(
                headline
            )
        )


        # -------------------------------------------------
        # COUNT SENTIMENT
        # -------------------------------------------------

        if sentiment == "POSITIVE":

            positive_count += 1


        elif sentiment == "NEGATIVE":

            negative_count += 1


        else:

            neutral_count += 1


        # -------------------------------------------------
        # SAVE RESULT
        # -------------------------------------------------

        analyzed_news.append(

            {

                "title": headline,

                "publisher": item.get(
                    "publisher",
                    "Unknown"
                ),

                "link": item.get(
                    "link",
                    ""
                ),

                "published_str": item.get(
                    "published_str",
                    "date unknown"
                ),

                "age_days": item.get("age_days"),

                "sentiment": sentiment,

                "score": score

            }

        )


    # =================================================
    # CALCULATE OVERALL SCORE
    # =================================================

    total_news = len(
        analyzed_news
    )


    overall_score = (

        (
            positive_count
            - negative_count
        )

        / total_news

    )


    # =================================================
    # DETERMINE OVERALL SENTIMENT
    # =================================================

    if overall_score > 0.20:

        overall_sentiment = "POSITIVE"


    elif overall_score < -0.20:

        overall_sentiment = "NEGATIVE"


    else:

        overall_sentiment = "NEUTRAL"


    # =================================================
    # RETURN RESULTS
    # =================================================

    newest = news[0] if news else None

    return {

        "company": company_name,

        "total_news": total_news,

        "positive": positive_count,

        "neutral": neutral_count,

        "negative": negative_count,

        "sentiment_score": overall_score,

        "sentiment_label": overall_sentiment,

        "fetched_at": fetched_at,

        "most_recent": newest.get("published_str") if newest else None,

        "newest_age_days": newest.get("age_days") if newest else None,

        "news": analyzed_news

    }