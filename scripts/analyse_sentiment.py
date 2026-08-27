import os
import sys


# =================================================
# ADD PROJECT ROOT TO PYTHON PATH
# =================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(
    0,
    PROJECT_ROOT
)


# =================================================
# IMPORT PROJECT MODULES
# =================================================

from modules.news_sentiment import (
    analyze_company_sentiment
)

from config import COMPANIES


# =================================================
# GET COMPANY FROM COMMAND LINE
# =================================================

if len(sys.argv) < 2:

    print(
        "\nPlease provide a company name."
    )

    print(
        "\nAvailable companies:"
    )

    for company in COMPANIES:

        display_name = company.replace(
            ".NS",
            ""
        )

        print(
            f"- {display_name}"
        )

    print(
        "\nExample:"
    )

    print(
        "python scripts/analyse_sentiment.py TCS"
    )

    sys.exit()


# =================================================
# GET COMPANY NAME
# =================================================

input_company = sys.argv[1].upper()


# =================================================
# FIND COMPANY
# =================================================

company_key = None


if input_company in COMPANIES:

    company_key = input_company


else:

    for key in COMPANIES:

        short_name = key.replace(
            ".NS",
            ""
        )

        if input_company == short_name:

            company_key = key

            break


# =================================================
# COMPANY NOT FOUND
# =================================================

if company_key is None:

    print(
        f"\nError: Company "
        f"'{input_company}' not found."
    )

    sys.exit()


# =================================================
# GET COMPANY INFORMATION
# =================================================

company = COMPANIES[
    company_key
]


if isinstance(company, dict):

    ticker = company.get(
        "ticker",
        company_key
    )

else:

    ticker = company


# =================================================
# CLEAN COMPANY NAME
# =================================================

company_name = company_key.replace(
    ".NS",
    ""
)


# =================================================
# DISPLAY INFORMATION
# =================================================

print(
    "\n" + "=" * 60
)

print(
    f"ANALYZING NEWS SENTIMENT FOR: "
    f"{company_name}"
)

print(
    f"STOCK TICKER: {ticker}"
)

print(
    "=" * 60
)


# =================================================
# ANALYZE COMPANY NEWS
# =================================================

print(
    "\nFetching recent company news..."
)

print(
    "Analyzing news sentiment..."
)


try:

    result = analyze_company_sentiment(

        company_name=company_name,

        ticker=ticker,

        max_results=10

    )


except Exception as error:

    print(
        "\nError while analyzing "
        "news sentiment:"
    )

    print(
        error
    )

    sys.exit()


# =================================================
# EXTRACT RESULT
# =================================================

overall_sentiment = result.get(
    "sentiment_label",
    "NEUTRAL"
)


sentiment_score = result.get(
    "sentiment_score",
    0.0
)


positive_count = result.get(
    "positive",
    0
)


neutral_count = result.get(
    "neutral",
    0
)


negative_count = result.get(
    "negative",
    0
)


total_news = result.get(
    "total_news",
    0
)


news_items = result.get(
    "news",
    []
)


# =================================================
# DISPLAY RESULTS
# =================================================

print(
    "\n" + "=" * 60
)

print(
    "NEWS SENTIMENT ANALYSIS"
)

print(
    "=" * 60
)


print(
    f"\nCompany: {company_name}"
)


print(
    f"Ticker: {ticker}"
)


print(
    f"\nOverall News Sentiment: "
    f"{overall_sentiment}"
)


print(
    f"Sentiment Score: "
    f"{sentiment_score:.3f}"
)


# =================================================
# SENTIMENT INTERPRETATION
# =================================================

print(
    "\n" + "-" * 60
)

print(
    "SENTIMENT INTERPRETATION"
)

print(
    "-" * 60
)


if overall_sentiment == "POSITIVE":

    print(
        "\nRecent company news shows "
        "a predominantly positive sentiment."
    )

    print(
        "Positive news coverage currently "
        "outweighs negative coverage."
    )


elif overall_sentiment == "NEGATIVE":

    print(
        "\nRecent company news shows "
        "a predominantly negative sentiment."
    )

    print(
        "Negative news coverage currently "
        "outweighs positive coverage."
    )


else:

    print(
        "\nRecent company news shows "
        "a neutral or mixed sentiment."
    )

    print(
        "No strong positive or negative "
        "news trend was identified."
    )


# =================================================
# SENTIMENT DISTRIBUTION
# =================================================

print(
    "\n" + "-" * 60
)

print(
    "SENTIMENT DISTRIBUTION"
)

print(
    "-" * 60
)


print(
    f"\nTotal News Articles Analyzed: "
    f"{total_news}"
)


print(
    f"Positive News: "
    f"{positive_count}"
)


print(
    f"Neutral News: "
    f"{neutral_count}"
)


print(
    f"Negative News: "
    f"{negative_count}"
)


# =================================================
# DISPLAY RECENT HEADLINES
# =================================================

if news_items:

    print(
        "\n" + "-" * 60
    )

    print(
        "RECENT NEWS HEADLINES"
    )

    print(
        "-" * 60
    )


    for index, news in enumerate(
        news_items[:5],
        start=1
    ):


        title = news.get(
            "title",
            "Headline not available"
        )


        publisher = news.get(
            "publisher",
            "Unknown"
        )


        sentiment = news.get(
            "sentiment",
            "NEUTRAL"
        )


        print(
            f"\n{index}. {title}"
        )


        print(
            f"   Source: {publisher}"
        )


        print(
            f"   Sentiment: {sentiment}"
        )


# =================================================
# SYSTEM EXPLANATION
# =================================================

print(
    "\n" + "-" * 60
)

print(
    "SYSTEM INTERPRETATION"
)

print(
    "-" * 60
)


print(
    "\nThe sentiment module evaluates recent "
    "company-specific news headlines and classifies "
    "their market sentiment."
)


print(
    "This analysis complements the LSTM market "
    "prediction by providing additional context "
    "about the current information environment."
)


print(
    "The historical price model and news analysis "
    "are maintained as separate modules, allowing "
    "their outputs to be compared independently."
)


# =================================================
# COMPLETION
# =================================================

print(
    "\n" + "=" * 60
)

print(
    "NEWS SENTIMENT ANALYSIS COMPLETED"
)

print(
    "=" * 60
)