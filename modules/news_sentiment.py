import requests
import re
import feedparser
from urllib.parse import quote
from modules.news_collection import get_stock_news

# =================================================
# COMPANY SEARCH TERMS
# =================================================

COMPANY_SEARCH_TERMS = {

    # =================================================
    # TATA CONSULTANCY SERVICES
    # =================================================

    "TCS": {

        "query": "Tata Consultancy Services",

        "keywords": [

            "tata consultancy services",
            "tata consultancy",
            "tcs limited",
            "tcs ltd"

        ]

    },


    # =================================================
    # RELIANCE INDUSTRIES
    # =================================================

    "RELIANCE": {

        "query": "Reliance Industries",

        "keywords": [

            "reliance industries",
            "reliance industries limited",
            "reliance jio",
            "jio platforms",
            "mukesh ambani",
            "ril"

        ]

    },


    # =================================================
    # INFOSYS
    # =================================================

    "INFY": {

        "query": "Infosys Limited",

        "keywords": [

            "infosys limited",
            "infosys",
            "infy"

        ]

    },


    # =================================================
    # HDFC BANK
    # =================================================

    "HDFCBANK": {

        "query": "HDFC Bank",

        "keywords": [

            "hdfc bank limited",
            "hdfc bank"

        ]

    },


    # =================================================
    # ICICI BANK
    # =================================================

    "ICICIBANK": {

        "query": "ICICI Bank",

        "keywords": [

            "icici bank limited",
            "icici bank"

        ]

    },


    # =================================================
    # STATE BANK OF INDIA
    # =================================================

    "SBIN": {

        "query": "State Bank of India",

        "keywords": [

            "state bank of india",
            "state bank",
            "sbi bank"

        ]

    },


    # =================================================
    # WIPRO
    # =================================================

    "WIPRO": {

        "query": "Wipro Limited",

        "keywords": [

            "wipro limited",
            "wipro"

        ]

    },


    # =================================================
    # HINDUSTAN UNILEVER
    # =================================================

    "HINDUNILVR": {

        "query": "Hindustan Unilever",

        "keywords": [

            "hindustan unilever",
            "hindustan unilever limited",
            "hul india"

        ]

    },


    # =================================================
    # ITC LIMITED
    # =================================================

    "ITC": {

        "query": "ITC Limited India",

        "keywords": [

            "itc limited",
            "itc india"

        ]

    },


    # =================================================
    # BHARTI AIRTEL
    # =================================================

    "BHARTIARTL": {

        "query": "Bharti Airtel",

        "keywords": [

            "bharti airtel",
            "airtel india",
            "airtel limited"

        ]

    }

}

# =================================================
# GET COMPANY NEWS
# =================================================

def get_company_news(
    company_name,
    ticker=None,
    max_results=10
):
    """
    Fetches recent company-specific news using
    Google News RSS.
    """

    company_key = company_name.upper()


    # -------------------------------------------------
    # GET COMPANY SEARCH INFORMATION
    # -------------------------------------------------

    company_info = COMPANY_SEARCH_TERMS.get(
        company_key,
        {
            "query": company_name,
            "keywords": [
                company_name.lower()
            ]
        }
    )


    search_query = company_info["query"]


    # -------------------------------------------------
    # CREATE GOOGLE NEWS RSS URL
    # -------------------------------------------------

    encoded_query = quote(
        f'"{search_query}"'
    )


    url = (
        "https://news.google.com/rss/search?"
        f"q={encoded_query}"
        "&hl=en-IN"
        "&gl=IN"
        "&ceid=IN:en"
    )


    # -------------------------------------------------
    # FETCH RSS NEWS
    # -------------------------------------------------

    try:

        feed = feedparser.parse(
            url
        )


        processed_news = []


        # -------------------------------------------------
        # PROCESS NEWS
        # -------------------------------------------------

        for item in feed.entries:

            title = item.get(
                "title",
                ""
            )


            link = item.get(
                "link",
                ""
            )


            # Google News titles often look like:
            #
            # Headline - Source
            #
            # Extract source if available.
            source = item.get(
                "source",
                {}
            )


            if isinstance(source, dict):

                publisher = source.get(
                    "title",
                    "Unknown"
                )

            else:

                publisher = "Unknown"


            if not title:

                continue


            processed_news.append(

                {

                    "title": title,

                    "publisher": publisher,

                    "link": link

                }

            )


            # Stop after requested number
            if len(processed_news) >= max_results:

                break


        return processed_news


    except Exception as error:

        print(
            f"Error fetching news: {error}"
        )

        return []


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
    max_results=10
):
    """
    Fetches relevant company news and performs
    sentiment analysis.
    """


    news = get_stock_news(
    company_name=company_name,
    max_news=max_results
)

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

    return {

        "company": company_name,

        "total_news": total_news,

        "positive": positive_count,

        "neutral": neutral_count,

        "negative": negative_count,

        "sentiment_score": overall_score,

        "sentiment_label": overall_sentiment,

        "news": analyzed_news

    }