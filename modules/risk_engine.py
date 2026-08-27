# -------------------------------------------------
# RISK ANALYSIS MODULE
# -------------------------------------------------

def calculate_risk(

    volatility,
    predicted_change_percent,
    technical_signal,
    confidence_score,
    agreement

):

    risk_score = 0

    risk_factors = []


    # ---------------------------------------------
    # VOLATILITY RISK
    # ---------------------------------------------

    if volatility == "HIGH":

        risk_score += 3

        risk_factors.append(
            "High market volatility is present."
        )


    elif volatility == "MEDIUM":

        risk_score += 2

        risk_factors.append(
            "Medium market volatility is present."
        )


    else:

        risk_score += 1


    # ---------------------------------------------
    # PREDICTED PRICE MOVEMENT
    # ---------------------------------------------

    movement = abs(
        predicted_change_percent
    )


    if movement >= 5:

        risk_score += 3

        risk_factors.append(
            "A large predicted price movement is expected."
        )


    elif movement >= 2:

        risk_score += 2

        risk_factors.append(
            "A moderate predicted price movement is expected."
        )


    else:

        risk_score += 1


    # ---------------------------------------------
    # TECHNICAL CONFIRMATION
    # ---------------------------------------------

    if technical_signal == "NEUTRAL":

        risk_score += 2

        risk_factors.append(
            "Technical indicators provide mixed confirmation."
        )


    # ---------------------------------------------
    # CONFIDENCE RISK
    # ---------------------------------------------

    if confidence_score < 50:

        risk_score += 3

        risk_factors.append(
            "Low prediction confidence is present."
        )


    elif confidence_score < 70:

        risk_score += 2

        risk_factors.append(
            "Moderate prediction confidence is present."
        )


    # ---------------------------------------------
    # MODULE AGREEMENT RISK
    # ---------------------------------------------

    if agreement == "MIXED":

        risk_score += 2

        risk_factors.append(
            "AI modules show mixed agreement."
        )


    elif agreement == "MODERATE":

        risk_score += 1


    # ---------------------------------------------
    # LIMIT SCORE
    # ---------------------------------------------

    risk_score = min(
        risk_score,
        10
    )


    # ---------------------------------------------
    # DETERMINE RISK LEVEL
    # ---------------------------------------------

    if risk_score >= 8:

        risk_level = "HIGH"

    elif risk_score >= 5:

        risk_level = "MEDIUM"

    else:

        risk_level = "LOW"


    # ---------------------------------------------
    # RETURN RESULT
    # ---------------------------------------------

    return {

        "risk_score":
        risk_score,

        "risk_level":
        risk_level,

        "risk_factors":
        risk_factors

    }