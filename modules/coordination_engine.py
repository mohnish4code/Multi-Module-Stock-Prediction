# -------------------------------------------------
# COORDINATION ENGINE
# -------------------------------------------------

def coordinate_modules(
    prediction_direction,
    technical_signal,
    news_sentiment
):

    bullish_score = 0
    bearish_score = 0


    # -------------------------------------------------
    # MODULE 1: LSTM PREDICTION
    # -------------------------------------------------

    if prediction_direction == "BULLISH":

        bullish_score += 1

    elif prediction_direction == "BEARISH":

        bearish_score += 1


    # -------------------------------------------------
    # MODULE 2: TECHNICAL ANALYSIS
    # -------------------------------------------------

    if technical_signal == "BULLISH":

        bullish_score += 1

    elif technical_signal == "BEARISH":

        bearish_score += 1


    # -------------------------------------------------
    # MODULE 3: NEWS SENTIMENT
    # -------------------------------------------------

    if news_sentiment == "POSITIVE":

        bullish_score += 1

    elif news_sentiment == "NEGATIVE":

        bearish_score += 1


    # -------------------------------------------------
    # DETERMINE AGREEMENT
    # -------------------------------------------------

    total_modules = 3


    bullish_agreement = (
        bullish_score / total_modules
    ) * 100


    bearish_agreement = (
        bearish_score / total_modules
    ) * 100


    neutral_count = (
        total_modules
        - bullish_score
        - bearish_score
    )


    # -------------------------------------------------
    # FINAL MARKET DIRECTION
    # -------------------------------------------------

    if bullish_score > bearish_score:

        coordinated_signal = "BULLISH"


    elif bearish_score > bullish_score:

        coordinated_signal = "BEARISH"


    else:

        coordinated_signal = "NEUTRAL"


    # -------------------------------------------------
    # AGREEMENT LEVEL
    # -------------------------------------------------

    if (
        bullish_score == 3
        or bearish_score == 3
    ):

        agreement = "STRONG"


    elif (
        bullish_score >= 2
        or bearish_score >= 2
    ):

        agreement = "MODERATE"


    else:

        agreement = "MIXED"


    # -------------------------------------------------
    # RETURN RESULTS
    # -------------------------------------------------

    return {

        "coordinated_signal":
        coordinated_signal,

        "agreement":
        agreement,

        "bullish_modules":
        bullish_score,

        "bearish_modules":
        bearish_score,

        "neutral_modules":
        neutral_count,

        "bullish_agreement":
        bullish_agreement,

        "bearish_agreement":
        bearish_agreement

    }