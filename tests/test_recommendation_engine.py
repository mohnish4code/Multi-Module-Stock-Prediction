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
# IMPORT RECOMMENDATION ENGINE
# =================================================

from modules.recommendation_engine import (
    get_final_recommendation
)


# =================================================
# TEST ALL POSSIBLE COMBINATIONS
# =================================================

technical_signals = [
    "BUY",
    "HOLD",
    "SELL"
]


news_sentiments = [
    "POSITIVE",
    "NEUTRAL",
    "NEGATIVE"
]


print(
    "\n" + "=" * 60
)

print(
    "RECOMMENDATION ENGINE TEST"
)

print(
    "=" * 60
)


# =================================================
# TEST EACH COMBINATION
# =================================================

for technical_signal in technical_signals:

    for news_sentiment in news_sentiments:

        result = get_final_recommendation(

            technical_signal=technical_signal,

            news_sentiment=news_sentiment

        )


        print(
            "\n" + "-" * 60
        )


        print(
            f"Technical Signal: "
            f"{technical_signal}"
        )


        print(
            f"News Sentiment: "
            f"{news_sentiment}"
        )


        print(
            f"Final Recommendation: "
            f"{result['recommendation']}"
        )


        print(
            f"Confidence: "
            f"{result['confidence']}"
        )


        print(
            f"Signal Agreement: "
            f"{result['agreement']}"
        )


        print(
            f"Explanation: "
            f"{result['explanation']}"
        )


print(
    "\n" + "=" * 60
)

print(
    "RECOMMENDATION ENGINE TEST COMPLETED"
)

print(
    "=" * 60
)