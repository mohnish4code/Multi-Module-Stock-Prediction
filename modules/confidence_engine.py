# -------------------------------------------------
# CONFIDENCE ANALYSIS MODULE
# -------------------------------------------------

def calculate_confidence(
    prediction_direction,
    technical_signal,
    news_sentiment,
    agreement
):

    # ---------------------------------------------
    # BASE CONFIDENCE
    # ---------------------------------------------

    confidence = 50


    # ---------------------------------------------
    # TECHNICAL AGREEMENT
    # ---------------------------------------------

    if prediction_direction == technical_signal:

        confidence += 15


    elif technical_signal == "NEUTRAL":

        confidence += 0


    else:

        confidence -= 10


    # ---------------------------------------------
    # NEWS AGREEMENT
    # ---------------------------------------------

    if (
        prediction_direction == "BULLISH"
        and news_sentiment == "POSITIVE"
    ):

        confidence += 15


    elif (
        prediction_direction == "BEARISH"
        and news_sentiment == "NEGATIVE"
    ):

        confidence += 15


    elif news_sentiment == "NEUTRAL":

        confidence += 0


    else:

        confidence -= 10


    # ---------------------------------------------
    # MODULE AGREEMENT
    # ---------------------------------------------

    if agreement == "STRONG":

        confidence += 20


    elif agreement == "MODERATE":

        confidence += 10


    elif agreement == "MIXED":

        confidence -= 5


    # ---------------------------------------------
    # LIMIT CONFIDENCE RANGE
    # ---------------------------------------------

    confidence = max(
        0,
        min(
            confidence,
            100
        )
    )


    # ---------------------------------------------
    # CONFIDENCE LEVEL
    # ---------------------------------------------

    if confidence >= 80:

        confidence_level = "HIGH"

    elif confidence >= 60:

        confidence_level = "MODERATE"

    else:

        confidence_level = "LOW"


    # ---------------------------------------------
    # RETURN RESULT
    # ---------------------------------------------

    return {

        "confidence_score":
        confidence,

        "confidence_level":
        confidence_level

    }