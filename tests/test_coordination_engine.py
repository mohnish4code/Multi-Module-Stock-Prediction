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

from modules.coordination_engine import (
    coordinate_modules
)


# -------------------------------------------------
# TEST MODULE OUTPUTS
# -------------------------------------------------

result = coordinate_modules(

    prediction_direction="BEARISH",

    technical_signal="NEUTRAL",

    news_sentiment="NEUTRAL"

)


# -------------------------------------------------
# DISPLAY RESULTS
# -------------------------------------------------

print(
    "\n" + "=" * 60
)

print(
    "MULTI-MODULE COORDINATION RESULTS"
)

print(
    "=" * 60
)


print(
    f"\nCoordinated Signal: "
    f"{result['coordinated_signal']}"
)


print(
    f"Agreement Level: "
    f"{result['agreement']}"
)


print(
    "\nMODULE DISTRIBUTION"
)


print(
    f"Bullish Modules: "
    f"{result['bullish_modules']}"
)


print(
    f"Bearish Modules: "
    f"{result['bearish_modules']}"
)


print(
    f"Neutral Modules: "
    f"{result['neutral_modules']}"
)


print(
    f"\nBullish Agreement: "
    f"{result['bullish_agreement']:.1f}%"
)


print(
    f"Bearish Agreement: "
    f"{result['bearish_agreement']:.1f}%"
)


print(
    "\n" + "=" * 60
)