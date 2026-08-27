import os
import sys


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


sys.path.insert(
    0,
    PROJECT_ROOT
)

from modules.confidence_engine import (
    calculate_confidence
)


result = calculate_confidence(

    prediction_direction="BEARISH",

    technical_signal="NEUTRAL",

    news_sentiment="NEUTRAL",

    agreement="MIXED"

)


print(
    "\n" + "=" * 60
)

print(
    "UPDATED CONFIDENCE ANALYSIS"
)

print(
    "=" * 60
)


print(
    f"\nConfidence Score: "
    f"{result['confidence_score']}%"
)


print(
    f"Confidence Level: "
    f"{result['confidence_level']}"
)


print(
    "\n" + "=" * 60
)