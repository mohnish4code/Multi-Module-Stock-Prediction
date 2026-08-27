# =================================================
# COMBINE LSTM AND NEWS SENTIMENT SIGNALS
# =================================================

def get_final_recommendation(
    technical_signal,
    news_sentiment
):
    """
    Combines:

    1. LSTM technical signal:
       BUY / HOLD / SELL

    2. News sentiment:
       POSITIVE / NEUTRAL / NEGATIVE

    Returns:
       - final recommendation
       - confidence
       - signal agreement
       - explanation
    """

    # =================================================
    # NORMALIZE INPUTS
    # =================================================

    technical_signal = (
        technical_signal
        .upper()
        .strip()
    )

    news_sentiment = (
        news_sentiment
        .upper()
        .strip()
    )


    # =================================================
    # VALIDATE TECHNICAL SIGNAL
    # =================================================

    valid_technical_signals = [
        "BUY",
        "HOLD",
        "SELL"
    ]

    if technical_signal not in valid_technical_signals:

        raise ValueError(
            "Invalid technical signal. "
            "Expected BUY, HOLD, or SELL."
        )


    # =================================================
    # VALIDATE NEWS SENTIMENT
    # =================================================

    valid_sentiments = [
        "POSITIVE",
        "NEUTRAL",
        "NEGATIVE"
    ]

    if news_sentiment not in valid_sentiments:

        raise ValueError(
            "Invalid news sentiment. "
            "Expected POSITIVE, NEUTRAL, "
            "or NEGATIVE."
        )


    # =================================================
    # BUY SIGNAL
    # =================================================

    if technical_signal == "BUY":

        # ---------------------------------------------
        # BUY + POSITIVE
        # ---------------------------------------------

        if news_sentiment == "POSITIVE":

            return {
                "recommendation": "BUY",
                "confidence": "HIGH",
                "agreement": "STRONG AGREEMENT",
                "explanation": (
                    "The technical model indicates "
                    "a positive market signal, and "
                    "recent company news also shows "
                    "positive sentiment."
                )
            }


        # ---------------------------------------------
        # BUY + NEUTRAL
        # ---------------------------------------------

        elif news_sentiment == "NEUTRAL":

            return {
                "recommendation": "BUY",
                "confidence": "MODERATE",
                "agreement": "PARTIAL AGREEMENT",
                "explanation": (
                    "The technical model indicates "
                    "a positive market signal while "
                    "recent news sentiment remains "
                    "neutral."
                )
            }


        # ---------------------------------------------
        # BUY + NEGATIVE
        # ---------------------------------------------

        else:

            return {
                "recommendation": "HOLD",
                "confidence": "LOW",
                "agreement": "CONFLICTING SIGNALS",
                "explanation": (
                    "The technical model indicates "
                    "positive market movement, but "
                    "recent news sentiment is "
                    "negative. The conflicting "
                    "signals suggest a cautious "
                    "market position."
                )
            }


    # =================================================
    # HOLD SIGNAL
    # =================================================

    elif technical_signal == "HOLD":

        # ---------------------------------------------
        # HOLD + POSITIVE
        # ---------------------------------------------

        if news_sentiment == "POSITIVE":

            return {
                "recommendation": "BUY",
                "confidence": "MODERATE",
                "agreement": "SUPPORTIVE SIGNALS",
                "explanation": (
                    "Technical indicators are stable, "
                    "while recent company news shows "
                    "positive sentiment. The positive "
                    "information environment supports "
                    "a moderately bullish outlook."
                )
            }


        # ---------------------------------------------
        # HOLD + NEUTRAL
        # ---------------------------------------------

        elif news_sentiment == "NEUTRAL":

            return {
                "recommendation": "HOLD",
                "confidence": "HIGH",
                "agreement": "STRONG AGREEMENT",
                "explanation": (
                    "Technical indicators and recent "
                    "news sentiment both indicate a "
                    "stable market environment with "
                    "no strong directional signal."
                )
            }


        # ---------------------------------------------
        # HOLD + NEGATIVE
        # ---------------------------------------------

        else:

            return {
                "recommendation": "HOLD",
                "confidence": "MODERATE",
                "agreement": "CAUTIOUS SIGNAL",
                "explanation": (
                    "Technical indicators remain "
                    "stable, but recent news "
                    "sentiment is negative. A "
                    "cautious market position is "
                    "recommended."
                )
            }


    # =================================================
    # SELL SIGNAL
    # =================================================

    elif technical_signal == "SELL":

        # ---------------------------------------------
        # SELL + POSITIVE
        # ---------------------------------------------

        if news_sentiment == "POSITIVE":

            return {
                "recommendation": "HOLD",
                "confidence": "LOW",
                "agreement": "CONFLICTING SIGNALS",
                "explanation": (
                    "The technical model indicates "
                    "a negative market signal, while "
                    "recent news sentiment is "
                    "positive. The conflicting "
                    "signals suggest waiting for "
                    "additional market confirmation."
                )
            }


        # ---------------------------------------------
        # SELL + NEUTRAL
        # ---------------------------------------------

        elif news_sentiment == "NEUTRAL":

            return {
                "recommendation": "SELL",
                "confidence": "MODERATE",
                "agreement": "PARTIAL AGREEMENT",
                "explanation": (
                    "The technical model indicates "
                    "negative market movement, while "
                    "recent news provides no strong "
                    "positive or negative signal."
                )
            }


        # ---------------------------------------------
        # SELL + NEGATIVE
        # ---------------------------------------------

        else:

            return {
                "recommendation": "SELL",
                "confidence": "HIGH",
                "agreement": "STRONG AGREEMENT",
                "explanation": (
                    "The technical model indicates "
                    "negative market movement, and "
                    "recent company news also shows "
                    "negative sentiment."
                )
            }