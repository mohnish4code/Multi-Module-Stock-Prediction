import os
import sys
import joblib
import numpy as np

from tensorflow.keras.models import (
    load_model
)


# -------------------------------------------------
# PROJECT ROOT
# -------------------------------------------------

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


sys.path.insert(
    0,
    PROJECT_ROOT
)


# -------------------------------------------------
# PROJECT MODULES
# -------------------------------------------------

from config import (
    COMPANIES,
    SEQUENCE_LENGTH
)

from modules.data_collection import (
    get_stock_data
)

from modules.feature_engineering import (
    add_technical_indicators
)

from modules.preprocessing import (
    FEATURE_COLUMNS
)


# -------------------------------------------------
# START
# -------------------------------------------------

print(
    "\n" + "=" * 70
)

print(
    "TESTING ALL EXISTING TRAINED MODELS"
)

print(
    "=" * 70
)


# -------------------------------------------------
# RESULTS
# -------------------------------------------------

successful_companies = []

failed_companies = []


# -------------------------------------------------
# TEST EVERY COMPANY
# -------------------------------------------------

for ticker, company_info in COMPANIES.items():

    company_name = company_info[
        "name"
    ]

    model_folder = company_info[
        "model_folder"
    ]


    print(
        "\n" + "-" * 70
    )

    print(
        f"Testing: {company_name}"
    )

    print(
        f"Ticker: {ticker}"
    )

    print(
        "-" * 70
    )


    try:


        # -----------------------------------------
        # MODEL PATHS
        # -----------------------------------------

        model_path = os.path.join(

            PROJECT_ROOT,

            "models",

            model_folder,

            "lstm_model.keras"
        )


        scaler_path = os.path.join(

            PROJECT_ROOT,

            "models",

            model_folder,

            "scaler.pkl"
        )


        # -----------------------------------------
        # CHECK MODEL FILE
        # -----------------------------------------

        if not os.path.exists(
            model_path
        ):

            raise FileNotFoundError(
                f"Model not found: "
                f"{model_path}"
            )


        # -----------------------------------------
        # CHECK SCALER FILE
        # -----------------------------------------

        if not os.path.exists(
            scaler_path
        ):

            raise FileNotFoundError(
                f"Scaler not found: "
                f"{scaler_path}"
            )


        # -----------------------------------------
        # LOAD MODEL
        # -----------------------------------------

        print(
            "Loading model..."
        )


        model = load_model(
            model_path
        )


        # -----------------------------------------
        # LOAD SCALER
        # -----------------------------------------

        print(
            "Loading scaler..."
        )


        scaler = joblib.load(
            scaler_path
        )


        # -----------------------------------------
        # FETCH DATA
        # -----------------------------------------

        print(
            "Fetching stock data..."
        )


        df = get_stock_data(

            ticker=ticker,

            period="1y",

            interval="1d"
        )


        # -----------------------------------------
        # FEATURE ENGINEERING
        # -----------------------------------------

        print(
            "Adding technical indicators..."
        )


        df = add_technical_indicators(
            df
        )


        # -----------------------------------------
        # SELECT MODEL FEATURES
        # -----------------------------------------

        data = df[
            FEATURE_COLUMNS
        ].copy()


        # -----------------------------------------
        # REMOVE MISSING VALUES
        # -----------------------------------------

        data.dropna(
            inplace=True
        )


        # -----------------------------------------
        # VALIDATE DATA
        # -----------------------------------------

        if len(data) < SEQUENCE_LENGTH:

            raise ValueError(
                "Not enough data for "
                "prediction sequence."
            )


        # -----------------------------------------
        # SCALE DATA
        # -----------------------------------------

        scaled_data = scaler.transform(
            data
        )


        # -----------------------------------------
        # CREATE LATEST SEQUENCE
        # -----------------------------------------

        latest_sequence = scaled_data[
            -SEQUENCE_LENGTH:
        ]


        X_latest = np.expand_dims(

            latest_sequence,

            axis=0
        )


        # -----------------------------------------
        # CHECK INPUT SHAPE
        # -----------------------------------------

        expected_shape = (
            model.input_shape[1:]
        )


        actual_shape = (
            X_latest.shape[1:]
        )


        if actual_shape != expected_shape:

            raise ValueError(
                f"Input shape mismatch. "
                f"Expected {expected_shape}, "
                f"got {actual_shape}."
            )


        # -----------------------------------------
        # MAKE PREDICTION
        # -----------------------------------------

        prediction = model.predict(

            X_latest,

            verbose=0
        )


        # -----------------------------------------
        # SUCCESS
        # -----------------------------------------

        print(
            "SUCCESS"
        )

        print(
            f"Model Input Shape: "
            f"{X_latest.shape}"
        )

        print(
            f"Scaled Prediction: "
            f"{float(prediction[0][0]):.6f}"
        )


        successful_companies.append(
            company_name
        )


    except Exception as error:


        print(
            "FAILED"
        )

        print(
            f"Error: {error}"
        )


        failed_companies.append({

            "company":
            company_name,

            "error":
            str(error)

        })


# -------------------------------------------------
# FINAL SUMMARY
# -------------------------------------------------

print(
    "\n" + "=" * 70
)

print(
    "ALL MODEL TEST SUMMARY"
)

print(
    "=" * 70
)


print(
    f"\nTotal Companies: "
    f"{len(COMPANIES)}"
)


print(
    f"Successful: "
    f"{len(successful_companies)}"
)


print(
    f"Failed: "
    f"{len(failed_companies)}"
)


# -------------------------------------------------
# SUCCESSFUL MODELS
# -------------------------------------------------

if successful_companies:

    print(
        "\nSUCCESSFUL MODELS:"
    )


    for company in successful_companies:

        print(
            f"- {company}"
        )


# -------------------------------------------------
# FAILED MODELS
# -------------------------------------------------

if failed_companies:

    print(
        "\nFAILED MODELS:"
    )


    for item in failed_companies:

        print(
            f"- {item['company']}"
        )

        print(
            f"  Error: "
            f"{item['error']}"
        )


print(
    "\n" + "=" * 70
)