# =================================================
# PREDICTION SIGNAL MODULE
# =================================================

def get_prediction_signal(
    current_price,
    predicted_price,
    neutral_threshold=1.0
):
    """
    Converts an LSTM price prediction into:

    - Predicted percentage change
    - Prediction direction:
        BULLISH / NEUTRAL / BEARISH
    - Recommendation signal:
        BUY / HOLD / SELL
    """

    # -------------------------------------------------
    # VALIDATION
    # -------------------------------------------------

    current_price = float(current_price)
    predicted_price = float(predicted_price)

    if current_price <= 0:

        raise ValueError(
            "Current price must be greater than zero."
        )


    # -------------------------------------------------
    # CALCULATE PREDICTED CHANGE
    # -------------------------------------------------

    predicted_change = (
        (
            predicted_price
            - current_price
        )
        / current_price
    ) * 100


    # -------------------------------------------------
    # DETERMINE PREDICTION DIRECTION
    # -------------------------------------------------

    if predicted_change > neutral_threshold:

        prediction_direction = "BULLISH"

        recommendation_signal = "BUY"


    elif predicted_change < -neutral_threshold:

        prediction_direction = "BEARISH"

        recommendation_signal = "SELL"


    else:

        prediction_direction = "NEUTRAL"

        recommendation_signal = "HOLD"


    # -------------------------------------------------
    # RETURN RESULT
    # -------------------------------------------------

    return {

        "current_price":
        current_price,

        "predicted_price":
        predicted_price,

        "predicted_change_percent":
        predicted_change,

        "prediction_direction":
        prediction_direction,

        "recommendation_signal":
        recommendation_signal

    }