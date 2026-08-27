# -------------------------------------------------
# SIGNAL CONVERSION UTILITIES
# -------------------------------------------------


def direction_to_recommendation(
    direction
):

    direction = (
        direction
        .upper()
        .strip()
    )


    if direction == "BULLISH":

        return "BUY"


    elif direction == "BEARISH":

        return "SELL"


    else:

        return "HOLD"


def recommendation_to_direction(
    recommendation
):

    recommendation = (
        recommendation
        .upper()
        .strip()
    )


    if recommendation == "BUY":

        return "BULLISH"


    elif recommendation == "SELL":

        return "BEARISH"


    else:

        return "NEUTRAL"