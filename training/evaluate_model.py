import os
import sys
import numpy as np
import joblib

from tensorflow.keras.models import load_model

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error
)


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
from modules.feature_engineering import add_technical_indicators

from modules.preprocessing import (
    prepare_train_val_test_data,
    FEATURE_COLUMNS
)

from config import COMPANIES


# =================================================
# GET COMPANY FROM COMMAND LINE
# =================================================

if len(sys.argv) < 2:

    print("\nPlease provide a company name.")

    print("\nExample:")

    print(
        "python training/evaluate_model.py TCS"
    )

    print("\nAvailable companies:")

    for company in COMPANIES:

        print(
            f"- {company.replace('.NS', '')}"
        )

    sys.exit()


# =================================================
# GET COMPANY NAME
# =================================================

input_company = sys.argv[1].upper()


# =================================================
# FIND COMPANY
# =================================================

company_key = None


# -------------------------------------------------
# CHECK DIRECT MATCH
# -------------------------------------------------

if input_company in COMPANIES:

    company_key = input_company


# -------------------------------------------------
# CHECK TICKER WITHOUT .NS
# -------------------------------------------------

else:

    for key in COMPANIES:

        short_name = key.replace(
            ".NS",
            ""
        )

        if input_company == short_name:

            company_key = key

            break


# =================================================
# COMPANY NOT FOUND
# =================================================

if company_key is None:

    print(
        f"\nCompany '{input_company}' not found."
    )

    print("\nAvailable companies:")

    for company in COMPANIES:

        print(
            f"- {company.replace('.NS', '')}"
        )

    sys.exit()


# =================================================
# COMPANY INFORMATION
# =================================================

company = COMPANIES[
    company_key
]


if isinstance(company, dict):

    ticker = company.get(
        "ticker",
        company_key
    )

else:

    ticker = company


company_name = company_key.replace(
    ".NS",
    ""
)


# =================================================
# DISPLAY COMPANY INFORMATION
# =================================================

print("\n" + "=" * 60)

print(
    f"EVALUATING MODEL FOR: {company_name}"
)

print(
    f"STOCK TICKER: {ticker}"
)

print("=" * 60)


# =================================================
# MODEL PATHS
# =================================================

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


# =================================================
# CHECK MODEL EXISTS
# =================================================

if not os.path.exists(model_path):

    print(
        "\nError: Trained model not found."
    )

    print(
        f"Expected location:\n{model_path}"
    )

    sys.exit()


# =================================================
# CHECK SCALER EXISTS
# =================================================

if not os.path.exists(scaler_path):

    print(
        "\nError: Saved scaler not found."
    )

    print(
        f"Expected location:\n{scaler_path}"
    )

    sys.exit()


# =================================================
# FETCH HISTORICAL STOCK DATA
# =================================================

print(
    "\nFetching 10 years of stock data..."
)


df = get_stock_data(
    ticker=ticker,
    period="10y",
    interval="1d"
)


# =================================================
# CHECK DATA
# =================================================

if df is None or df.empty:

    print(
        "\nError: Could not fetch stock data."
    )

    sys.exit()


# =================================================
# FEATURE ENGINEERING
# =================================================

print(
    "\nAdding technical indicators..."
)


df = add_technical_indicators(
    df
)


# =================================================
# CHECK DATA AFTER FEATURE ENGINEERING
# =================================================

if df is None or df.empty:

    print(
        "\nError: No data available "
        "after feature engineering."
    )

    sys.exit()


# =================================================
# PREPARE TRAIN / VALIDATION / TEST DATA
# =================================================

print(
    "Preparing evaluation data..."
)


(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    new_scaler
) = prepare_train_val_test_data(
    df,
    sequence_length=60,
    train_ratio=0.70,
    val_ratio=0.15
)


print("\nData split:")

print(
    f"Training samples: "
    f"{len(X_train)}"
)

print(
    f"Validation samples: "
    f"{len(X_val)}"
)

print(
    f"Final test samples: "
    f"{len(X_test)}"
)


# =================================================
# IMPORTANT
# =================================================

print(
    "\nUsing ONLY the final unseen test data "
    "for evaluation."
)


# =================================================
# LOAD TRAINED MODEL
# =================================================

print(
    "\nLoading trained LSTM model..."
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
# MAKE PREDICTIONS ON TEST DATA ONLY
# =================================================

print(
    "\nMaking predictions on unseen test data..."
)


predictions_scaled = model.predict(
    X_test,
    verbose=0
)


# =================================================
# GET CLOSE COLUMN INDEX
# =================================================

close_index = FEATURE_COLUMNS.index(
    "Close"
)


# =================================================
# CONVERT SCALED VALUES TO ACTUAL PRICES
# =================================================

def inverse_close_price(
    values,
    scaler
):

    dummy = np.zeros(
        (
            len(values),
            len(FEATURE_COLUMNS)
        )
    )


    dummy[
        :,
        close_index
    ] = values.flatten()


    inverse_data = scaler.inverse_transform(
        dummy
    )


    return inverse_data[
        :,
        close_index
    ]


# =================================================
# CONVERT PREDICTIONS TO ACTUAL PRICES
# =================================================

predictions = inverse_close_price(
    predictions_scaled,
    scaler
)


# =================================================
# CONVERT ACTUAL TEST VALUES TO PRICES
# =================================================

actual_prices = inverse_close_price(
    y_test,
    scaler
)


# =================================================
# CALCULATE PRICE PREDICTION METRICS
# =================================================

mae = mean_absolute_error(
    actual_prices,
    predictions
)


rmse = np.sqrt(
    mean_squared_error(
        actual_prices,
        predictions
    )
)


mape = (
    mean_absolute_percentage_error(
        actual_prices,
        predictions
    )
    * 100
)


# =================================================
# GET PREVIOUS ACTUAL PRICES
# =================================================

# The last Close value in each sequence is the
# previous known price before the target day.

previous_prices_scaled = X_test[
    :,
    -1,
    close_index
]


previous_prices = inverse_close_price(
    previous_prices_scaled,
    scaler
)


# =================================================
# DETERMINE ACTUAL DIRECTION
# =================================================

actual_direction = np.where(

    actual_prices > previous_prices,

    "UP",

    "DOWN"
)


# =================================================
# DETERMINE PREDICTED DIRECTION
# =================================================

predicted_direction = np.where(

    predictions > previous_prices,

    "UP",

    "DOWN"
)


# =================================================
# CALCULATE DIRECTIONAL ACCURACY
# =================================================

correct_predictions = (

    actual_direction
    ==
    predicted_direction

)


directional_accuracy = (
    np.mean(
        correct_predictions
    )
    * 100
)


# =================================================
# COUNT CORRECT UP PREDICTIONS
# =================================================

correct_up = np.sum(

    (actual_direction == "UP")
    &
    (predicted_direction == "UP")

)


# =================================================
# COUNT CORRECT DOWN PREDICTIONS
# =================================================

correct_down = np.sum(

    (actual_direction == "DOWN")
    &
    (predicted_direction == "DOWN")

)


# =================================================
# COUNT INCORRECT PREDICTIONS
# =================================================

incorrect_predictions = np.sum(

    actual_direction
    !=
    predicted_direction

)


# =================================================
# DISPLAY RESULTS
# =================================================

print("\n" + "=" * 60)

print(
    "FINAL MODEL EVALUATION RESULTS"
)

print("=" * 60)


print(
    "\nThe following results are based ONLY "
    "on the unseen 15% test data."
)


# -------------------------------------------------
# PRICE ACCURACY
# -------------------------------------------------

print("\nPRICE PREDICTION ACCURACY")

print("-" * 60)


print(
    f"MAE: ₹{mae:.2f}"
)


print(
    f"RMSE: ₹{rmse:.2f}"
)


print(
    f"MAPE: {mape:.2f}%"
)


# -------------------------------------------------
# DIRECTIONAL ACCURACY
# -------------------------------------------------

print("\nDIRECTIONAL ACCURACY")

print("-" * 60)


print(
    f"Directional Accuracy: "
    f"{directional_accuracy:.2f}%"
)


print(
    f"Correct UP Predictions: "
    f"{correct_up}"
)


print(
    f"Correct DOWN Predictions: "
    f"{correct_down}"
)


print(
    f"Incorrect Predictions: "
    f"{incorrect_predictions}"
)


print(
    f"Total Test Samples: "
    f"{len(X_test)}"
)


# =================================================
# SHOW LAST 10 TEST PREDICTIONS
# =================================================

print("\n" + "-" * 60)

print(
    "LAST 10 ACTUAL VS PREDICTED TEST PRICES"
)

print("-" * 60)


for (
    previous,
    actual,
    predicted,
    actual_dir,
    predicted_dir
) in zip(

    previous_prices[-10:],

    actual_prices[-10:],

    predictions[-10:],

    actual_direction[-10:],

    predicted_direction[-10:]
):

    correct = (
        "YES"
        if actual_dir == predicted_dir
        else "NO"
    )


    print(

        f"\nPrevious: ₹{previous:.2f}"

    )


    print(

        f"Actual: ₹{actual:.2f} "
        f"({actual_dir})"

    )


    print(

        f"Predicted: ₹{predicted:.2f} "
        f"({predicted_dir})"

    )


    print(

        f"Direction Correct: {correct}"

    )


# =================================================
# EVALUATION COMPLETE
# =================================================

print("\n" + "=" * 60)

print(
    "FINAL EVALUATION COMPLETED"
)

print("=" * 60)