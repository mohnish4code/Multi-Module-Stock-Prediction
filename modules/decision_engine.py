def make_final_decision(
    coordinated_signal,
    agreement,
    confidence_score,
    risk_level,
    predicted_change_percent=0,
    technical_signal="NEUTRAL",
    news_sentiment="NEUTRAL"
):

    # =================================================
    # DETERMINE RECOMMENDATION
    # =================================================

    if risk_level == "HIGH" or confidence_score < 50:

        recommendation = "HOLD"

    elif coordinated_signal == "BULLISH":

        recommendation = "BUY"

    elif coordinated_signal == "BEARISH":

        recommendation = "SELL"

    else:

        recommendation = "HOLD"


    # =================================================
    # GENERATE DETAILED EXPLANATION
    # =================================================

    direction = (
        "an upward"
        if predicted_change_percent > 0
        else "a downward"
    )

    movement = abs(predicted_change_percent)


    # =================================================
    # HOLD EXPLANATION
    # =================================================

    if recommendation == "HOLD":

        explanation = (
            f"The LSTM model predicts {direction} price movement "
            f"of approximately {movement:.2f}%. "
        )

        if technical_signal == "NEUTRAL":

            explanation += (
                "However, technical indicators remain neutral and do not "
                "strongly confirm the predicted movement. "
            )

        elif technical_signal != coordinated_signal:

            explanation += (
                f"Technical analysis gives a {technical_signal} signal, "
                "which does not fully confirm the LSTM prediction. "
            )

        if news_sentiment == "NEUTRAL":

            explanation += (
                "Recent financial news also shows overall neutral sentiment. "
            )

        elif news_sentiment != coordinated_signal:

            explanation += (
                f"News sentiment is {news_sentiment}, creating additional "
                "uncertainty in the market outlook. "
            )

        explanation += (
            f"Because the modules show {agreement.lower()} agreement, "
            f"the confidence is {confidence_score}% and the risk level "
            f"is {risk_level}. Therefore, the system recommends HOLD rather "
            "than making a high-risk BUY or SELL decision."
        )


    # =================================================
    # BUY EXPLANATION
    # =================================================

    elif recommendation == "BUY":

        explanation = (
            f"The LSTM model predicts {direction} price movement "
            f"of approximately {movement:.2f}%. "
            f"The coordinated module signal is {coordinated_signal}, "
            f"supported by {agreement.lower()} agreement between the AI "
            f"modules. Technical analysis is {technical_signal}, while "
            f"news sentiment is {news_sentiment}. "
            f"With a confidence score of {confidence_score}% and "
            f"{risk_level.lower()} risk, the system recommends BUY."
        )


    # =================================================
    # SELL EXPLANATION
    # =================================================

    else:

        explanation = (
            f"The LSTM model predicts {direction} price movement "
            f"of approximately {movement:.2f}%. "
            f"The coordinated module signal is {coordinated_signal}, "
            f"with {agreement.lower()} agreement across the AI modules. "
            f"Technical analysis is {technical_signal}, while recent "
            f"news sentiment is {news_sentiment}. "
            f"With a confidence score of {confidence_score}% and "
            f"{risk_level.lower()} risk, the system recommends SELL."
        )


    return {
        "recommendation": recommendation,
        "explanation": explanation
    }