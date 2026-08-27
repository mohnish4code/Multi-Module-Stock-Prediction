import os
import sys


# =================================================
# PROJECT ROOT SETUP
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
# IMPORT MODULES
# =================================================

from modules.news_collection import (
    get_stock_news
)

from modules.news_sentiment import (
    analyze_headline_sentiment
)


# =================================================
# SETTINGS
# =================================================

COMPANY_NAME = (
    "Tata Consultancy Services"
)


# =================================================
# FETCH NEWS
# =================================================

print(
    "\nFetching latest company news..."
)

headlines = get_stock_news(
    company_name=COMPANY_NAME,
    max_news=10
)


# =================================================
# CHECK NEWS AVAILABILITY
# =================================================

if not headlines:

    print(
        "\nNo news available for sentiment analysis."
    )


else:

    print(
        "\nAnalyzing financial news sentiment..."
    )


    # =================================================
    # SENTIMENT COUNTERS
    # =================================================

    positive_count = 0

    negative_count = 0

    neutral_count = 0


    # =================================================
    # STORE ANALYZED NEWS
    # =================================================

    analyzed_news = []


    # =================================================
    # ANALYZE EACH NEWS ARTICLE
    # =================================================

    for item in headlines:


        # -------------------------------------------------
        # EXTRACT HEADLINE
        # -------------------------------------------------

        headline = item.get(
            "title",
            ""
        )


        # Skip empty headlines

        if not headline:

            continue


        # -------------------------------------------------
        # ANALYZE HEADLINE SENTIMENT
        # -------------------------------------------------

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
        # SAVE ANALYSIS RESULT
        # -------------------------------------------------

        analyzed_news.append(

            {
                "headline": headline,

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
    # CHECK VALID ANALYZED NEWS
    # =================================================

    total_news = len(
        analyzed_news
    )


    if total_news == 0:

        print(
            "\nNo valid news headlines were available."
        )


    else:


        # =================================================
        # CALCULATE OVERALL SENTIMENT SCORE
        # =================================================

        sentiment_score = (

            (
                positive_count
                - negative_count
            )

            / total_news

        )


        # =================================================
        # DETERMINE OVERALL SENTIMENT
        # =================================================

        if sentiment_score > 0.20:

            overall_sentiment = (
                "POSITIVE"
            )


        elif sentiment_score < -0.20:

            overall_sentiment = (
                "NEGATIVE"
            )


        else:

            overall_sentiment = (
                "NEUTRAL"
            )


        # =================================================
        # DISPLAY RESULTS
        # =================================================

        print(
            "\n" + "=" * 65
        )

        print(
            "NEWS SENTIMENT ANALYSIS"
        )

        print(
            "=" * 65
        )


        print(
            f"\nCompany: "
            f"{COMPANY_NAME}"
        )


        print(
            f"\nOverall Sentiment: "
            f"{overall_sentiment}"
        )


        print(
            f"Sentiment Score: "
            f"{sentiment_score:.2f}"
        )


        # =================================================
        # SENTIMENT DISTRIBUTION
        # =================================================

        print(
            "\n" + "-" * 65
        )

        print(
            "SENTIMENT DISTRIBUTION"
        )

        print(
            "-" * 65
        )


        print(
            f"\nTotal Headlines: "
            f"{total_news}"
        )


        print(
            f"Positive News: "
            f"{positive_count}"
        )


        print(
            f"Negative News: "
            f"{negative_count}"
        )


        print(
            f"Neutral News: "
            f"{neutral_count}"
        )


        # =================================================
        # DETAILED NEWS ANALYSIS
        # =================================================

        print(
            "\n" + "-" * 65
        )

        print(
            "DETAILED NEWS ANALYSIS"
        )

        print(
            "-" * 65
        )


        for i, item in enumerate(
            analyzed_news,
            start=1
        ):


            print(
                f"\n{i}. "
                f"{item['headline']}"
            )


            print(
                f"Source: "
                f"{item['publisher']}"
            )


            print(
                f"Sentiment: "
                f"{item['sentiment']}"
            )


            print(
                f"Keyword Score: "
                f"{item['score']}"
            )


        # =================================================
        # FINAL RESULT
        # =================================================

        print(
            "\n" + "=" * 65
        )

        print(
            "NEWS SENTIMENT TEST COMPLETED SUCCESSFULLY"
        )

        print(
            "=" * 65
        )