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
# IMPORT PROJECT MODULES
# =================================================

from modules.predictions import (
    predict_next_day
)

from config import COMPANIES


# =================================================
# GET COMPANY FROM COMMAND LINE
# =================================================

if len(sys.argv) < 2:

    print(
        "\nPlease provide a company name."
    )

    print(
        "\nAvailable companies:"
    )

    for company in COMPANIES:

        display_name = company.replace(
            ".NS",
            ""
        )

        print(
            f"- {display_name}"
        )

    print(
        "\nExample:"
    )

    print(
        "python scripts/predict_stock.py TCS"
    )

    sys.exit()


# =================================================
# GET COMPANY NAME
# =================================================

input_company = sys.argv[1].upper()


# =================================================
# DISPLAY PREDICTION INFORMATION
# =================================================

print(
    "\n" + "=" * 60
)

print(
    f"ANALYZING STOCK: {input_company}"
)

print(
    "AI-BASED MARKET RECOMMENDATION"
)

print(
    "=" * 60
)


# =================================================
# GENERATE PREDICTION
# =================================================

print(
    "\nFetching latest market data..."
)

print(
    "Loading trained AI model..."
)

print(
    "Generating market analysis..."
)


try:

    prediction_result = predict_next_day(

        company_input=input_company,

        period="1y",

        sequence_length=60

    )


except Exception as error:

    print(
        "\nError while generating prediction:"
    )

    print(
        error
    )

    sys.exit()


# =================================================
# EXTRACT RESULTS
# =================================================

company_name = prediction_result.get(
    "company",
    input_company
)

ticker = prediction_result.get(
    "ticker",
    ""
)

latest_date = prediction_result.get(
    "latest_date"
)

current_price = prediction_result.get(
    "current_price"
)

predicted_price = prediction_result.get(
    "predicted_price"
)


# =================================================
# CHECK REQUIRED VALUES
# =================================================

if current_price is None:

    print(
        "\nError: Could not retrieve "
        "the latest stock price."
    )

    sys.exit()


if predicted_price is None:

    print(
        "\nError: Could not generate "
        "the stock prediction."
    )

    sys.exit()


# =================================================
# CONVERT TO FLOAT
# =================================================

current_price = float(
    current_price
)

predicted_price = float(
    predicted_price
)


# =================================================
# CALCULATE PRICE CHANGE
# =================================================

price_change = (
    predicted_price
    - current_price
)


percentage_change = (
    price_change
    / current_price
) * 100


# =================================================
# BUY / HOLD / SELL THRESHOLDS
# =================================================

BUY_THRESHOLD = 2.0

SELL_THRESHOLD = -2.0


# =================================================
# DETERMINE RECOMMENDATION
# =================================================

if percentage_change >= BUY_THRESHOLD:

    recommendation = "BUY"

    market_signal = (
        "Bullish"
    )

    reason = (
        "The predicted market movement is "
        "significantly positive."
    )


elif percentage_change <= SELL_THRESHOLD:

    recommendation = "SELL"

    market_signal = (
        "Bearish"
    )

    reason = (
        "The predicted market movement is "
        "significantly negative."
    )


else:

    recommendation = "HOLD"

    market_signal = (
        "Neutral / Stable"
    )

    reason = (
        "The predicted movement is within the "
        "neutral ±2% range."
    )


# =================================================
# DISPLAY RESULTS
# =================================================

print(
    "\n" + "=" * 60
)

print(
    "AI STOCK MARKET RECOMMENDATION"
)

print(
    "=" * 60
)


print(
    f"\nCompany: {company_name}"
)


print(
    f"Ticker: {ticker}"
)


if latest_date is not None:

    try:

        print(
            f"Latest Market Data: "
            f"{latest_date.date()}"
        )

    except AttributeError:

        print(
            f"Latest Market Data: "
            f"{latest_date}"
        )


print(
    f"\nCurrent Market Price: "
    f"₹{current_price:.2f}"
)


# =================================================
# PRIMARY RECOMMENDATION
# =================================================

print(
    "\n" + "-" * 60
)

print(
    "AI RECOMMENDATION"
)

print(
    "-" * 60
)


print(
    f"\nRecommendation: "
    f"{recommendation}"
)


print(
    f"Market Signal: "
    f"{market_signal}"
)


print(
    f"\nAnalysis: "
    f"{reason}"
)


# =================================================
# MODEL ANALYSIS
# =================================================

print(
    "\n" + "-" * 60
)

print(
    "MODEL ANALYSIS"
)

print(
    "-" * 60
)


print(
    f"\nCurrent Reference Price: "
    f"₹{current_price:.2f}"
)


print(
    f"Predicted Reference Price: "
    f"₹{predicted_price:.2f}"
)


print(
    f"\nEstimated Movement: "
    f"{percentage_change:.2f}%"
)


# =================================================
# RECOMMENDATION LOGIC
# =================================================

print(
    "\nRecommendation Thresholds:"
)


print(
    f"BUY  : Expected movement ≥ "
    f"+{BUY_THRESHOLD:.1f}%"
)


print(
    f"HOLD : Expected movement between "
    f"{SELL_THRESHOLD:.1f}% and "
    f"+{BUY_THRESHOLD:.1f}%"
)


print(
    f"SELL : Expected movement ≤ "
    f"{SELL_THRESHOLD:.1f}%"
)


# =================================================
# IMPORTANT NOTE
# =================================================

print(
    "\n" + "-" * 60
)

print(
    "MODEL INTERPRETATION"
)

print(
    "-" * 60
)


print(
    "The market recommendation is generated by the "
    "trained LSTM model using historical stock price "
    "patterns and technical indicators."
)

print(
    "The output represents the model's estimated market "
    "movement based on the available data and learned "
    "historical patterns."
)

print(
    "As stock markets are influenced by external events "
    "and changing market conditions, the prediction "
    "represents a probabilistic model-based estimate "
    "rather than a guaranteed future outcome."
)

print(
    "-" * 60
)


# =================================================
# FINAL OUTPUT
# =================================================

print(
    "\n" + "=" * 60
)

print(
    "MARKET ANALYSIS COMPLETED"
)

print(
    "=" * 60
)