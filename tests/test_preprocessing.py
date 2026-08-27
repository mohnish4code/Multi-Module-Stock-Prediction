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

from modules.data_collection import (
    get_stock_data
)

from modules.feature_engineering import (
    add_technical_indicators
)

from modules.preprocessing import (
    prepare_train_val_test_data
)


# =================================================
# GET STOCK DATA
# =================================================

print(
    "\nFetching stock data..."
)


df = get_stock_data(

    ticker="TCS.NS",

    period="1y",

    interval="1d"
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


# =================================================
# PREPARE DATA
# =================================================

print(
    "Preparing train, validation and test data..."
)


(
    X_train,
    y_train,

    X_val,
    y_val,

    X_test,
    y_test,

    scaler

) = prepare_train_val_test_data(

    df=df,

    sequence_length=60
)


# =================================================
# DISPLAY RESULTS
# =================================================

print(
    "\n" + "=" * 60
)

print(
    "PREPROCESSING TEST RESULTS"
)

print(
    "=" * 60
)


# -------------------------------------------------
# TRAINING DATA
# -------------------------------------------------

print(
    "\nTRAINING DATA"
)

print(
    f"X_train Shape: "
    f"{X_train.shape}"
)

print(
    f"y_train Shape: "
    f"{y_train.shape}"
)


# -------------------------------------------------
# VALIDATION DATA
# -------------------------------------------------

print(
    "\nVALIDATION DATA"
)

print(
    f"X_val Shape: "
    f"{X_val.shape}"
)

print(
    f"y_val Shape: "
    f"{y_val.shape}"
)


# -------------------------------------------------
# TEST DATA
# -------------------------------------------------

print(
    "\nTEST DATA"
)

print(
    f"X_test Shape: "
    f"{X_test.shape}"
)

print(
    f"y_test Shape: "
    f"{y_test.shape}"
)


# -------------------------------------------------
# SEQUENCE INFORMATION
# -------------------------------------------------

print(
    "\nSEQUENCE INFORMATION"
)

print(
    f"Sequence Length: "
    f"{X_train.shape[1]}"
)

print(
    f"Features Per Day: "
    f"{X_train.shape[2]}"
)


# -------------------------------------------------
# FIRST SAMPLE
# -------------------------------------------------

print(
    "\nFIRST TRAINING SEQUENCE"
)

print(
    f"Shape: "
    f"{X_train[0].shape}"
)


print(
    "\nFIRST TRAINING TARGET"
)

print(
    y_train[0]
)


# =================================================
# COMPLETED
# =================================================

print(
    "\n" + "=" * 60
)

print(
    "PREPROCESSING TEST COMPLETED SUCCESSFULLY"
)

print(
    "=" * 60
)