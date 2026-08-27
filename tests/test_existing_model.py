import os
import sys
import joblib
import numpy as np

from tensorflow.keras.models import load_model


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
# IMPORT PROJECT MODULES
# -------------------------------------------------

from modules.data_collection import get_stock_data

from modules.feature_engineering import (
    add_technical_indicators
)

from modules.preprocessing import (
    FEATURE_COLUMNS
)


# -------------------------------------------------
# COMPANY SETTINGS
# -------------------------------------------------

COMPANY_NAME = "TCS"

TICKER = "TCS.NS"

SEQUENCE_LENGTH = 60


# -------------------------------------------------
# MODEL PATHS
# -------------------------------------------------

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    COMPANY_NAME,
    "lstm_model.keras"
)


SCALER_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    COMPANY_NAME,
    "scaler.pkl"
)


# -------------------------------------------------
# DISPLAY START
# -------------------------------------------------

print(
    "\n" + "=" * 60
)

print(
    "EXISTING TRAINED MODEL TEST"
)

print(
    "=" * 60
)


# -------------------------------------------------
# CHECK FILES
# -------------------------------------------------

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"\nModel not found:\n{MODEL_PATH}"
    )


if not os.path.exists(SCALER_PATH):

    raise FileNotFoundError(
        f"\nScaler not found:\n{SCALER_PATH}"
    )


# -------------------------------------------------
# LOAD EXISTING TRAINED MODEL
# -------------------------------------------------

print(
    "\nLoading existing TCS model..."
)


model = load_model(
    MODEL_PATH
)


print(
    "Model loaded successfully."
)


# -------------------------------------------------
# LOAD EXISTING SCALER
# -------------------------------------------------

print(
    "\nLoading existing TCS scaler..."
)


scaler = joblib.load(
    SCALER_PATH
)


print(
    "Scaler loaded successfully."
)


# -------------------------------------------------
# GET LATEST STOCK DATA
# -------------------------------------------------

print(
    "\nFetching latest TCS stock data..."
)


df = get_stock_data(

    ticker=TICKER,

    period="1y",

    interval="1d"
)


# -------------------------------------------------
# ADD TECHNICAL INDICATORS
# -------------------------------------------------

print(
    "Adding technical indicators..."
)


df = add_technical_indicators(
    df
)


# -------------------------------------------------
# KEEP TRAINING FEATURES
# -------------------------------------------------

data = df[
    FEATURE_COLUMNS
].copy()


# -------------------------------------------------
# REMOVE MISSING VALUES
# -------------------------------------------------

data.dropna(
    inplace=True
)


# -------------------------------------------------
# CHECK DATA
# -------------------------------------------------

if len(data) < SEQUENCE_LENGTH:

    raise ValueError(
        "Not enough data to create "
        "a prediction sequence."
    )


# -------------------------------------------------
# SAVE CURRENT PRICE
# -------------------------------------------------

current_price = float(
    data["Close"].iloc[-1]
)


# -------------------------------------------------
# SCALE DATA USING SAVED SCALER
# -------------------------------------------------

print(
    "Scaling data using existing scaler..."
)


scaled_data = scaler.transform(
    data
)


# -------------------------------------------------
# CREATE LATEST 60-DAY SEQUENCE
# -------------------------------------------------

latest_sequence = scaled_data[
    -SEQUENCE_LENGTH:
]


# -------------------------------------------------
# ADD BATCH DIMENSION
# -------------------------------------------------

X_latest = np.expand_dims(
    latest_sequence,
    axis=0
)


# -------------------------------------------------
# DISPLAY MODEL INPUT INFORMATION
# -------------------------------------------------

print(
    "\nMODEL INPUT INFORMATION"
)

print(
    "-" * 60
)

print(
    f"Sequence Length: "
    f"{SEQUENCE_LENGTH}"
)

print(
    f"Features Per Day: "
    f"{len(FEATURE_COLUMNS)}"
)

print(
    f"Input Shape: "
    f"{X_latest.shape}"
)


# -------------------------------------------------
# VERIFY MODEL INPUT SHAPE
# -------------------------------------------------

expected_shape = model.input_shape


print(
    f"Model Expected Shape: "
    f"{expected_shape}"
)


if X_latest.shape[1] != expected_shape[1]:

    raise ValueError(
        f"Sequence length mismatch. "
        f"Model expects {expected_shape[1]} "
        f"but received {X_latest.shape[1]}."
    )


if X_latest.shape[2] != expected_shape[2]:

    raise ValueError(
        f"Feature count mismatch. "
        f"Model expects {expected_shape[2]} "
        f"but received {X_latest.shape[2]}."
    )


# -------------------------------------------------
# MAKE PREDICTION
# -------------------------------------------------

print(
    "\nMaking prediction..."
)


scaled_prediction = model.predict(
    X_latest,
    verbose=0
)


scaled_prediction = float(
    scaled_prediction[0][0]
)


# -------------------------------------------------
# CONVERT PREDICTION BACK TO ORIGINAL PRICE
# -------------------------------------------------

close_index = FEATURE_COLUMNS.index(
    "Close"
)


# -------------------------------------------------
# CREATE DUMMY ROW
# -------------------------------------------------
#
# The scaler expects all 14 features when
# inverse_transform is called.
#
# We only have a prediction for the Close price,
# so the other values are placeholders.
#
# For MinMaxScaler, the Close feature is transformed
# independently, so this correctly restores the
# original Close price.
# -------------------------------------------------

prediction_row = np.zeros(
    (
        1,
        len(FEATURE_COLUMNS)
    )
)


prediction_row[
    0,
    close_index
] = scaled_prediction


# -------------------------------------------------
# REVERSE SCALING
# -------------------------------------------------

original_prediction = scaler.inverse_transform(
    prediction_row
)


predicted_price = float(
    original_prediction[
        0,
        close_index
    ]
)


# -------------------------------------------------
# CALCULATE PERCENTAGE CHANGE
# -------------------------------------------------

predicted_change_percent = (
    (
        predicted_price
        - current_price
    )
    / current_price
) * 100


# -------------------------------------------------
# DETERMINE PREDICTION DIRECTION
# -------------------------------------------------

if predicted_change_percent > 1:

    prediction_direction = "BULLISH"

    recommendation_signal = "BUY"


elif predicted_change_percent < -1:

    prediction_direction = "BEARISH"

    recommendation_signal = "SELL"


else:

    prediction_direction = "NEUTRAL"

    recommendation_signal = "HOLD"


# -------------------------------------------------
# FINAL RESULTS
# -------------------------------------------------

print(
    "\n" + "=" * 60
)

print(
    "EXISTING TCS MODEL PREDICTION"
)

print(
    "=" * 60
)


print(
    f"\nCompany: "
    f"{COMPANY_NAME}"
)


print(
    f"Ticker: "
    f"{TICKER}"
)


print(
    f"\nModel Path:"
)

print(
    MODEL_PATH
)


print(
    f"\nScaler Path:"
)

print(
    SCALER_PATH
)


print(
    f"\nLatest Available Close Price: "
    f"₹{current_price:.2f}"
)


print(
    f"Predicted Price: "
    f"₹{predicted_price:.2f}"
)


print(
    f"Predicted Change: "
    f"{predicted_change_percent:.2f}%"
)


print(
    f"Prediction Direction: "
    f"{prediction_direction}"
)


print(
    f"Recommendation Signal: "
    f"{recommendation_signal}"
)


print(
    "\n" + "=" * 60
)

print(
    "EXISTING MODEL TEST COMPLETED SUCCESSFULLY"
)

print(
    "=" * 60
)