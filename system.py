import os
import sys
import numpy as np
import joblib

from tensorflow.keras.models import load_model


# =================================================
# PROJECT ROOT
# =================================================

PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)

sys.path.insert(
    0,
    PROJECT_ROOT
)


# =================================================
# PROJECT MODULES
# =================================================

from modules.data_collection import (
    get_stock_data
)

from modules.feature_engineering import (
    add_technical_indicators
)

from modules.preprocessing import (
    FEATURE_COLUMNS
)

from modules.technical_analysis import (
    analyze_technical_indicators
)

from modules.news_collection import (
    get_stock_news
)

from modules.news_sentiment import (
    analyze_headline_sentiment
)

from modules.coordination_engine import (
    coordinate_modules
)

from modules.confidence_engine import (
    calculate_confidence
)

from modules.risk_engine import (
    calculate_risk
)

from modules.decision_engine import (
    make_final_decision
)


# =================================================
# COMPANY CONFIGURATION
# =================================================

COMPANIES = {
    "TCS.NS": {
        "name": "Tata Consultancy Services",
        "model_folder": "TCS"
    },

    "RELIANCE.NS": {
        "name": "Reliance Industries",
        "model_folder": "RELIANCE"
    },

    "INFY.NS": {
        "name": "Infosys",
        "model_folder": "INFY"
    },

    "HDFCBANK.NS": {
        "name": "HDFC Bank",
        "model_folder": "HDFCBANK"
    },

    "ICICIBANK.NS": {
        "name": "ICICI Bank",
        "model_folder": "ICICIBANK"
    },

    "SBIN.NS": {
        "name": "State Bank of India",
        "model_folder": "SBIN"
    },

    "BHARTIARTL.NS": {
        "name": "Bharti Airtel",
        "model_folder": "BHARTIARTL"
    },

    "LT.NS": {
        "name": "Larsen & Toubro",
        "model_folder": "LT"
    },

    "HINDUNILVR.NS": {
        "name": "Hindustan Unilever",
        "model_folder": "HINDUNILVR"
    },

    "ITC.NS": {
        "name": "ITC",
        "model_folder": "ITC"
    }
}


# =================================================
# MODEL SETTINGS
# =================================================

SEQUENCE_LENGTH = 60


# =================================================
# HELPER FUNCTION:
# GET TECHNICAL SIGNAL
# =================================================

def get_technical_signal(
    result
):

    possible_keys = [
        "technical_signal",
        "overall_signal",
        "overall_technical_signal",
        "signal"
    ]

    for key in possible_keys:

        if key in result:

            return (
                str(result[key])
                .upper()
                .strip()
            )

    raise KeyError(
        "Technical signal key not found. "
        f"Available keys: {list(result.keys())}"
    )


# =================================================
# HELPER FUNCTION:
# DETERMINE PREDICTION DIRECTION
# =================================================

def get_prediction_direction(
    percentage_change
):

    if percentage_change > 0.5:

        return "BULLISH"

    elif percentage_change < -0.5:

        return "BEARISH"

    else:

        return "NEUTRAL"


# =================================================
# HELPER FUNCTION:
# ANALYZE MULTIPLE NEWS HEADLINES
# =================================================

def analyze_news_headlines(
    headlines
):

    positive_news = 0

    negative_news = 0

    neutral_news = 0

    news_details = []


    # -------------------------------------------------
    # ANALYZE EACH NEWS ITEM
    # -------------------------------------------------

    for item in headlines:

        # ---------------------------------------------
        # EXTRACT HEADLINE
        # ---------------------------------------------

        if isinstance(
            item,
            dict
        ):

            headline = (
                item.get(
                    "title",
                    ""
                )
            )

            source = (
                item.get(
                    "source",
                    "Unknown"
                )
            )


        else:

            headline = str(
                item
            )

            source = (
                "Unknown"
            )


        # ---------------------------------------------
        # SKIP EMPTY HEADLINES
        # ---------------------------------------------

        if not headline:

            continue


        # ---------------------------------------------
        # ANALYZE SINGLE HEADLINE
        # ---------------------------------------------

        result = (
            analyze_headline_sentiment(
                headline
            )
        )


        # ---------------------------------------------
        # HANDLE DICTIONARY RESULT
        # ---------------------------------------------

        if isinstance(
            result,
            dict
        ):

            sentiment = (
                result.get(
                    "sentiment",
                    "NEUTRAL"
                )
            )

            keyword_score = (
                result.get(
                    "keyword_score",
                    0
                )
            )


        # ---------------------------------------------
        # HANDLE STRING RESULT
        # ---------------------------------------------

        else:

            sentiment = str(
                result
            )

            keyword_score = 0


        # ---------------------------------------------
        # NORMALIZE SENTIMENT
        # ---------------------------------------------

        sentiment = (
            str(sentiment)
            .upper()
            .strip()
        )


        # ---------------------------------------------
        # COUNT SENTIMENT
        # ---------------------------------------------

        if sentiment == "POSITIVE":

            positive_news += 1


        elif sentiment == "NEGATIVE":

            negative_news += 1


        else:

            sentiment = "NEUTRAL"

            neutral_news += 1


        # ---------------------------------------------
        # STORE NEWS DETAIL
        # ---------------------------------------------

        news_details.append(
            {
                "headline": headline,

                "source": source,

                "sentiment": sentiment,

                "keyword_score": keyword_score
            }
        )


    # =================================================
    # CALCULATE OVERALL SENTIMENT
    # =================================================

    total_news = len(
        news_details
    )


    if total_news == 0:

        sentiment_score = 0.0

        overall_sentiment = "NEUTRAL"


    else:

        sentiment_score = (

            positive_news
            -
            negative_news

        ) / total_news


        if sentiment_score > 0.1:

            overall_sentiment = "POSITIVE"


        elif sentiment_score < -0.1:

            overall_sentiment = "NEGATIVE"


        else:

            overall_sentiment = "NEUTRAL"


    # =================================================
    # RETURN RESULTS
    # =================================================

    return {

        "overall_sentiment":
        overall_sentiment,

        "sentiment_score":
        sentiment_score,

        "total_news":
        total_news,

        "positive_news":
        positive_news,

        "negative_news":
        negative_news,

        "neutral_news":
        neutral_news,

        "news_details":
        news_details
    }


# =================================================
# START SYSTEM
# =================================================

print(
    "\n" + "=" * 70
)

print(
    "MULTI-MODULE STOCK PREDICTION SYSTEM"
)

print(
    "=" * 70
)


# =================================================
# DISPLAY AVAILABLE COMPANIES
# =================================================

print(
    "\nAVAILABLE COMPANIES:\n"
)


company_items = list(
    COMPANIES.items()
)


for index, (
    ticker,
    company_info
) in enumerate(
    company_items,
    start=1
):

    print(
        f"{index}. "
        f"{company_info['name']} "
        f"({ticker})"
    )


# =================================================
# GET USER SELECTION
# =================================================

while True:

    try:

        choice = int(
            input(
                "\nEnter company number: "
            )
        )


        if (
            1
            <= choice
            <= len(company_items)
        ):

            break


        else:

            print(
                "Invalid company number. "
                "Please try again."
            )


    except ValueError:

        print(
            "Please enter a valid number."
        )


# =================================================
# GET SELECTED COMPANY
# =================================================

TICKER, company_info = (

    company_items[
        choice - 1
    ]

)


FULL_COMPANY_NAME = (

    company_info[
        "name"
    ]

)


MODEL_FOLDER = (

    company_info[
        "model_folder"
    ]

)


print(
    "\n" + "=" * 70
)

print(
    f"SELECTED COMPANY: "
    f"{FULL_COMPANY_NAME}"
)

print(
    f"TICKER: "
    f"{TICKER}"
)

print(
    "=" * 70
)


# =================================================
# MODEL PATHS
# =================================================

MODEL_PATH = os.path.join(

    PROJECT_ROOT,

    "models",

    MODEL_FOLDER,

    "lstm_model.keras"
)


SCALER_PATH = os.path.join(

    PROJECT_ROOT,

    "models",

    MODEL_FOLDER,

    "scaler.pkl"
)


# =================================================
# CHECK MODEL FILES
# =================================================

if not os.path.exists(
    MODEL_PATH
):

    raise FileNotFoundError(

        "\nTrained model not found:\n"
        f"{MODEL_PATH}"
    )


if not os.path.exists(
    SCALER_PATH
):

    raise FileNotFoundError(

        "\nScaler not found:\n"
        f"{SCALER_PATH}"
    )


# =================================================
# LOAD TRAINED MODEL
# =================================================

print(
    "\nLoading trained LSTM model..."
)


model = load_model(
    MODEL_PATH
)


print(
    "Model loaded successfully."
)


# =================================================
# LOAD SCALER
# =================================================

print(
    "Loading saved scaler..."
)


scaler = joblib.load(
    SCALER_PATH
)


print(
    "Scaler loaded successfully."
)


# =================================================
# FETCH LATEST STOCK DATA
# =================================================

print(
    "\nFetching latest stock data..."
)


df = get_stock_data(

    ticker=TICKER,

    period="1y",

    interval="1d"
)


if (
    df is None
    or df.empty
):

    raise ValueError(
        "No stock data was received."
    )


# =================================================
# FEATURE ENGINEERING
# =================================================

print(
    "Calculating technical indicators..."
)


df = add_technical_indicators(
    df
)


df = (
    df
    .dropna()
    .reset_index(
        drop=True
    )
)


if len(df) < SEQUENCE_LENGTH:

    raise ValueError(

        f"Not enough data for prediction. "
        f"Need at least "
        f"{SEQUENCE_LENGTH} valid records."
    )


# =================================================
# CHECK REQUIRED FEATURES
# =================================================

missing_features = [

    feature

    for feature in FEATURE_COLUMNS

    if feature not in df.columns
]


if missing_features:

    raise ValueError(

        "Missing required features:\n"
        f"{missing_features}"
    )


# =================================================
# CURRENT PRICE
# =================================================

current_price = float(

    df["Close"]
    .iloc[-1]
)


# =================================================
# TECHNICAL ANALYSIS
# =================================================

print(
    "Running technical analysis..."
)


technical_result = (

    analyze_technical_indicators(
        df
    )

)


technical_signal = (

    get_technical_signal(
        technical_result
    )

)


# =================================================
# PREPARE MODEL INPUT
# =================================================

features = (

    df[
        FEATURE_COLUMNS
    ]
    .copy()
)


scaled_data = (

    scaler.transform(
        features
    )

)


latest_sequence = (

    scaled_data[
        -SEQUENCE_LENGTH:
    ]

)


X_latest = np.expand_dims(

    latest_sequence,

    axis=0
)


# =================================================
# VERIFY MODEL INPUT
# =================================================

expected_sequence_length = (

    model.input_shape[1]
)


expected_features = (

    model.input_shape[2]
)


if (
    SEQUENCE_LENGTH
    != expected_sequence_length
):

    raise ValueError(

        "Sequence length mismatch. "
        f"Model expects "
        f"{expected_sequence_length}, "
        f"but system is using "
        f"{SEQUENCE_LENGTH}."
    )


if (
    len(FEATURE_COLUMNS)
    != expected_features
):

    raise ValueError(

        "Feature mismatch between "
        "trained model and "
        "current preprocessing.\n"

        f"Model expects: "
        f"{expected_features}\n"

        f"Current features: "
        f"{len(FEATURE_COLUMNS)}"
    )


print(
    "Model input prepared successfully."
)


print(
    f"Input Shape: "
    f"{X_latest.shape}"
)


# =================================================
# LSTM PREDICTION
# =================================================

print(
    "Making next-day prediction..."
)


scaled_prediction = (

    model.predict(

        X_latest,

        verbose=0
    )

)


scaled_prediction = float(

    scaled_prediction[
        0
    ][
        0
    ]

)


# =================================================
# INVERSE TRANSFORM PREDICTION
# =================================================

close_index = (

    FEATURE_COLUMNS.index(
        "Close"
    )

)


prediction_row = np.zeros(

    (

        1,

        len(
            FEATURE_COLUMNS
        )

    )

)


prediction_row[
    0,
    close_index
] = scaled_prediction


original_prediction = (

    scaler.inverse_transform(
        prediction_row
    )

)


predicted_price = float(

    original_prediction[
        0,
        close_index
    ]

)


# =================================================
# CALCULATE PRICE CHANGE
# =================================================

price_change = (

    predicted_price
    -
    current_price
)


percentage_change = (

    price_change
    /
    current_price

) * 100


# =================================================
# DETERMINE PREDICTION DIRECTION
# =================================================

prediction_direction = (

    get_prediction_direction(
        percentage_change
    )

)


# =================================================
# NEWS COLLECTION
# =================================================

print(
    "Fetching latest company news..."
)


headlines = get_stock_news(

    company_name=FULL_COMPANY_NAME,

    max_news=10
)


# =================================================
# NEWS SENTIMENT ANALYSIS
# =================================================

print(
    "Analyzing news sentiment..."
)


if not headlines:

    print(
        "No recent news available. "
        "Using NEUTRAL sentiment."
    )


    news_result = {

        "overall_sentiment":
        "NEUTRAL",

        "sentiment_score":
        0.0,

        "total_news":
        0,

        "positive_news":
        0,

        "negative_news":
        0,

        "neutral_news":
        0,

        "news_details":
        []
    }


else:

    news_result = (

        analyze_news_headlines(
            headlines
        )

    )


news_sentiment = (

    str(

        news_result[
            "overall_sentiment"
        ]

    )

    .upper()

    .strip()
)


# =================================================
# COORDINATION ENGINE
# =================================================

print(
    "Coordinating AI modules..."
)


coordination_result = (

    coordinate_modules(

        prediction_direction=
        prediction_direction,

        technical_signal=
        technical_signal,

        news_sentiment=
        news_sentiment
    )

)


# =================================================
# CONFIDENCE ENGINE
# =================================================

print(
    "Calculating confidence..."
)


confidence_result = (

    calculate_confidence(

        prediction_direction=
        prediction_direction,

        technical_signal=
        technical_signal,

        news_sentiment=
        news_sentiment,

        agreement=

        coordination_result[
            "agreement"
        ]
    )

)


# =================================================
# RISK ANALYSIS
# =================================================

print(
    "Calculating risk..."
)


risk_result = (

    calculate_risk(

        volatility=

        technical_result[
            "volatility_level"
        ],

        predicted_change_percent=
        percentage_change,

        technical_signal=
        technical_signal,

        confidence_score=

        confidence_result[
            "confidence_score"
        ],

        agreement=

        coordination_result[
            "agreement"
        ]
    )

)


# =================================================
# FINAL DECISION ENGINE
# =================================================

print(
    "Generating final decision..."
)


decision_result = (

    make_final_decision(

        coordinated_signal=

        coordination_result[
            "coordinated_signal"
        ],

        agreement=

        coordination_result[
            "agreement"
        ],

        confidence_score=

        confidence_result[
            "confidence_score"
        ],

        risk_level=

        risk_result[
            "risk_level"
        ],

        predicted_change_percent=
        percentage_change,

        technical_signal=
        technical_signal,

        news_sentiment=
        news_sentiment
    )

)


# =================================================
# FINAL OUTPUT
# =================================================

print(
    "\n" + "=" * 70
)

print(
    "LIVE MULTI-MODULE STOCK PREDICTION SYSTEM"
)

print(
    "=" * 70
)


# =================================================
# MARKET DATA
# =================================================

print(
    "\nMARKET DATA"
)

print(
    "-" * 70
)


print(
    f"Company: "
    f"{FULL_COMPANY_NAME}"
)


print(
    f"Ticker: "
    f"{TICKER}"
)


print(
    f"\nCurrent Price: "
    f"₹{current_price:.2f}"
)


print(
    f"Predicted Next Price: "
    f"₹{predicted_price:.2f}"
)


print(
    f"Expected Change: "
    f"₹{price_change:.2f}"
)


print(
    f"Expected Change (%): "
    f"{percentage_change:.2f}%"
)


# =================================================
# MODULE 1
# =================================================

print(
    "\nMODULE 1: LSTM PRICE PREDICTION"
)

print(
    "-" * 70
)


print(
    f"Prediction Direction: "
    f"{prediction_direction}"
)


# =================================================
# MODULE 2
# =================================================

print(
    "\nMODULE 2: TECHNICAL ANALYSIS"
)

print(
    "-" * 70
)


print(
    f"RSI: "
    f"{technical_result['rsi']:.2f}"
)


print(
    f"Trend: "
    f"{technical_result['trend']}"
)


print(
    f"MACD: "
    f"{technical_result['macd']:.4f}"
)


print(
    f"MACD Signal: "
    f"{technical_result['macd_signal']:.4f}"
)


print(
    f"Volatility: "
    f"{technical_result['volatility']:.4f}"
)


print(
    f"Volatility Level: "
    f"{technical_result['volatility_level']}"
)


print(
    f"Overall Technical Signal: "
    f"{technical_signal}"
)


# =================================================
# MODULE 3
# =================================================

print(
    "\nMODULE 3: NEWS SENTIMENT"
)

print(
    "-" * 70
)


print(
    f"Overall News Sentiment: "
    f"{news_sentiment}"
)


print(
    f"Sentiment Score: "
    f"{news_result['sentiment_score']:.2f}"
)


print(
    f"Total Headlines: "
    f"{news_result['total_news']}"
)


print(
    f"Positive Headlines: "
    f"{news_result['positive_news']}"
)


print(
    f"Negative Headlines: "
    f"{news_result['negative_news']}"
)


print(
    f"Neutral Headlines: "
    f"{news_result['neutral_news']}"
)


# =================================================
# COORDINATION
# =================================================

print(
    "\nMULTI-MODULE COORDINATION"
)

print(
    "-" * 70
)


print(
    f"Coordinated Signal: "
    f"{coordination_result['coordinated_signal']}"
)


print(
    f"Agreement Level: "
    f"{coordination_result['agreement']}"
)


print(
    f"Bullish Modules: "
    f"{coordination_result['bullish_modules']}"
)


print(
    f"Bearish Modules: "
    f"{coordination_result['bearish_modules']}"
)


print(
    f"Neutral Modules: "
    f"{coordination_result['neutral_modules']}"
)


# =================================================
# MODULE 4
# =================================================

print(
    "\nMODULE 4: CONFIDENCE ENGINE"
)

print(
    "-" * 70
)


print(
    f"Confidence Score: "
    f"{confidence_result['confidence_score']}%"
)


print(
    f"Confidence Level: "
    f"{confidence_result['confidence_level']}"
)


# =================================================
# MODULE 5
# =================================================

print(
    "\nMODULE 5: RISK ANALYSIS"
)

print(
    "-" * 70
)


print(
    f"Risk Score: "
    f"{risk_result['risk_score']}/10"
)


print(
    f"Risk Level: "
    f"{risk_result['risk_level']}"
)


print(
    "\nRisk Factors:"
)


for factor in risk_result[
    "risk_factors"
]:

    print(
        f"- {factor}"
    )


# =================================================
# FINAL DECISION
# =================================================

print(
    "\n" + "=" * 70
)

print(
    "FINAL AI DECISION"
)

print(
    "=" * 70
)


print(
    f"\nRECOMMENDATION: "
    f"{decision_result['recommendation']}"
)


print(
    f"CONFIDENCE: "
    f"{confidence_result['confidence_score']}%"
)


print(
    f"RISK LEVEL: "
    f"{risk_result['risk_level']}"
)


print(
    "\nEXPLANATION:"
)


print(
    decision_result[
        "explanation"
    ]
)


print(
    "\n" + "=" * 70
)

print(
    "SYSTEM EXECUTION COMPLETED SUCCESSFULLY"
)

print(
    "=" * 70
)