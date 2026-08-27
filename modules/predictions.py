import os
import sys
import numpy as np
import joblib

from tensorflow.keras.models import load_model


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

from modules.data_collection import get_stock_data

from modules.feature_engineering import (
    add_technical_indicators
)

from modules.preprocessing import (
    FEATURE_COLUMNS
)

from config import COMPANIES


# =================================================
# FIND COMPANY
# =================================================

def find_company(
    company_input
):
    """
    Finds the company from the COMPANIES configuration.
    Accepts both:
    TCS
    TCS.NS
    """

    input_company = company_input.upper()

    # ---------------------------------------------
    # CHECK DIRECT MATCH
    # ---------------------------------------------

    if input_company in COMPANIES:

        return input_company


    # ---------------------------------------------
    # CHECK WITHOUT .NS
    # ---------------------------------------------

    for company_key in COMPANIES:

        short_name = company_key.replace(
            ".NS",
            ""
        )

        if input_company == short_name:

            return company_key


    return None


# =================================================
# GET COMPANY TICKER
# =================================================

def get_company_ticker(
    company_key
):
    """
    Gets the ticker symbol from config.py.
    Supports both dictionary and simple string formats.
    """

    company = COMPANIES[
        company_key
    ]


    if isinstance(
        company,
        dict
    ):

        ticker = company.get(
            "ticker",
            company_key
        )


    else:

        ticker = company


    return ticker


# =================================================
# GET MODEL PATHS
# =================================================

def get_model_paths(
    company_name
):
    """
    Returns the saved model and scaler paths.
    """

    model_directory = os.path.join(
        PROJECT_ROOT,
        "models",
        company_name
    )


    model_path = os.path.join(
        model_directory,
        "lstm_model.keras"
    )


    scaler_path = os.path.join(
        model_directory,
        "scaler.pkl"
    )


    return (
        model_path,
        scaler_path
    )


# =================================================
# INVERSE TRANSFORM CLOSE PRICE
# =================================================

def inverse_close_price(
    scaled_value,
    scaler
):
    """
    Converts the scaled Close price back
    to the actual stock price.
    """

    # Position of Close column
    close_index = FEATURE_COLUMNS.index(
        "Close"
    )


    # Make sure value is 2D
    scaled_value = np.array(
        scaled_value
    ).reshape(
        -1
    )


    # Create dummy feature array
    dummy = np.zeros(
        (
            len(scaled_value),
            len(FEATURE_COLUMNS)
        )
    )


    # Put scaled Close values
    dummy[
        :,
        close_index
    ] = scaled_value


    # Inverse transform
    inverse_data = scaler.inverse_transform(
        dummy
    )


    # Return only Close prices
    return inverse_data[
        :,
        close_index
    ]


# =================================================
# PREPARE LATEST DATA
# =================================================

def prepare_latest_sequence(
    df,
    scaler,
    sequence_length=60
):
    """
    Prepares the latest stock data sequence
    for prediction.
    """

    # Keep only the features used during training
    data = df[
        FEATURE_COLUMNS
    ].copy()


    # Remove missing values
    data.dropna(
        inplace=True
    )


    # Check sufficient data
    if len(data) < sequence_length:

        raise ValueError(
            f"Not enough data available. "
            f"At least {sequence_length} "
            f"records are required."
        )


    # Scale data using the saved training scaler
    scaled_data = scaler.transform(
        data
    )


    # Get latest 60 days
    latest_sequence = scaled_data[
        -sequence_length:
    ]


    # Reshape for LSTM
    latest_sequence = np.reshape(
        latest_sequence,
        (
            1,
            sequence_length,
            len(FEATURE_COLUMNS)
        )
    )


    return latest_sequence


# =================================================
# PREDICT NEXT DAY
# =================================================

def predict_next_day(
    company_input,
    period="10y",
    sequence_length=60
):
    """
    Predicts the next trading day's closing price.

    Returns:
        Dictionary containing prediction details.
    """


    # =================================================
    # FIND COMPANY
    # =================================================

    company_key = find_company(
        company_input
    )


    if company_key is None:

        raise ValueError(
            f"Company '{company_input}' "
            f"not found in configuration."
        )


    # =================================================
    # GET COMPANY INFORMATION
    # =================================================

    ticker = get_company_ticker(
        company_key
    )


    company_name = company_key.replace(
        ".NS",
        ""
    )


    # =================================================
    # GET MODEL PATHS
    # =================================================

    (
        model_path,
        scaler_path
    ) = get_model_paths(
        company_name
    )


    # =================================================
    # CHECK MODEL EXISTS
    # =================================================

    if not os.path.exists(
        model_path
    ):

        raise FileNotFoundError(
            f"Trained model not found:\n"
            f"{model_path}"
        )


    # =================================================
    # CHECK SCALER EXISTS
    # =================================================

    if not os.path.exists(
        scaler_path
    ):

        raise FileNotFoundError(
            f"Scaler not found:\n"
            f"{scaler_path}"
        )


    # =================================================
    # FETCH LATEST DATA
    # =================================================

    print(
        f"\nFetching latest stock data "
        f"for {company_name}..."
    )


    df = get_stock_data(
        ticker=ticker,
        period=period,
        interval="1d"
    )


    if df is None or df.empty:

        raise ValueError(
            f"Could not fetch stock data "
            f"for {ticker}"
        )


    # =================================================
    # ADD TECHNICAL INDICATORS
    # =================================================

    print(
        "Adding technical indicators..."
    )


    df = add_technical_indicators(
        df
    )


    if df is None or df.empty:

        raise ValueError(
            "No data available after "
            "feature engineering."
        )


    # =================================================
    # LOAD SAVED MODEL
    # =================================================

    print(
        "Loading trained LSTM model..."
    )


    model = load_model(
        model_path
    )


    # =================================================
    # LOAD SAVED SCALER
    # =================================================

    print(
        "Loading saved scaler..."
    )


    scaler = joblib.load(
        scaler_path
    )


    # =================================================
    # PREPARE LATEST SEQUENCE
    # =================================================

    print(
        "Preparing latest 60-day sequence..."
    )


    latest_sequence = prepare_latest_sequence(
        df=df,
        scaler=scaler,
        sequence_length=sequence_length
    )


    # =================================================
    # GET CURRENT PRICE
    # =================================================

    current_price = float(
        df["Close"].iloc[-1]
    )


    latest_date = df.index[-1]


    # =================================================
    # MAKE PREDICTION
    # =================================================

    print(
        "Making next-day prediction..."
    )


    prediction_scaled = model.predict(
        latest_sequence,
        verbose=0
    )


    # =================================================
    # CONVERT PREDICTION TO ACTUAL PRICE
    # =================================================

    predicted_price = inverse_close_price(
        prediction_scaled,
        scaler
    )[0]


    # =================================================
    # CALCULATE PRICE CHANGE
    # =================================================

    predicted_change = (
        predicted_price -
        current_price
    )


    predicted_percentage_change = (
        (
            predicted_change /
            current_price
        )
        * 100
    )


    # =================================================
    # DETERMINE DIRECTION
    # =================================================

    if predicted_change > 0:

        direction = "UP"


    elif predicted_change < 0:

        direction = "DOWN"


    else:

        direction = "NO CHANGE"


    # =================================================
    # RETURN RESULTS
    # =================================================

    prediction_result = {

        "company": company_name,

        "ticker": ticker,

        "latest_data_date": str(
            latest_date.date()
        ),

        "current_price": round(
            current_price,
            2
        ),

        "predicted_price": round(
            float(predicted_price),
            2
        ),

        "predicted_change": round(
            float(predicted_change),
            2
        ),

        "predicted_percentage_change": round(
            float(
                predicted_percentage_change
            ),
            2
        ),

        "direction": direction,

        "model_path": model_path,

        "scaler_path": scaler_path
    }


    return prediction_result