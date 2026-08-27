import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler


# -------------------------------------------------
# FEATURES USED BY THE LSTM
# -------------------------------------------------

FEATURE_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "SMA_20",
    "EMA_20",
    "RSI",
    "MACD",
    "MACD_Signal",
    "MACD_Histogram",
    "BB_Upper",
    "BB_Middle",
    "BB_Lower"
]


# -------------------------------------------------
# CREATE LSTM SEQUENCES
# -------------------------------------------------

def create_sequences(data, sequence_length=60):
    """
    Creates LSTM input sequences.

    Input:
        Previous 60 days of stock data

    Target:
        Current day's Close price
    """

    X = []
    y = []

    # Position of Close column
    close_index = FEATURE_COLUMNS.index(
        "Close"
    )

    for i in range(
        sequence_length,
        len(data)
    ):

        # Previous sequence_length days
        X.append(
            data[
                i - sequence_length:i
            ]
        )

        # Current day Close price
        y.append(
            data[
                i,
                close_index
            ]
        )

    return (
        np.array(X),
        np.array(y)
    )


# -------------------------------------------------
# PREPARE TRAIN / VALIDATION / TEST DATA
# -------------------------------------------------

def prepare_train_val_test_data(
    df,
    sequence_length=60,
    train_ratio=0.70,
    val_ratio=0.15
):
    """
    Prepares stock data for LSTM training.

    Data is split chronologically:

        70% Training
        15% Validation
        15% Testing

    The scaler is fitted ONLY on training data
    to prevent data leakage.
    """

    # -------------------------------------------------
    # KEEP SELECTED FEATURES
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
    # VALIDATE DATA
    # -------------------------------------------------

    if len(data) < sequence_length * 3:

        raise ValueError(
            "Not enough data to create "
            "training, validation and testing sets."
        )


    # -------------------------------------------------
    # CALCULATE SPLIT POSITIONS
    # -------------------------------------------------

    total_rows = len(data)


    train_end = int(
        total_rows * train_ratio
    )


    val_end = int(
        total_rows * (
            train_ratio + val_ratio
        )
    )


    # -------------------------------------------------
    # SPLIT DATA CHRONOLOGICALLY
    # -------------------------------------------------

    # Training data
    train_data = data.iloc[
        :train_end
    ]


    # Validation data
    #
    # Include previous sequence_length rows
    # from training data for historical context
    val_data = data.iloc[
        train_end - sequence_length:
        val_end
    ]


    # Testing data
    #
    # Include previous sequence_length rows
    # from validation data for historical context
    test_data = data.iloc[
        val_end - sequence_length:
    ]


    # -------------------------------------------------
    # CREATE SCALER
    # -------------------------------------------------

    scaler = MinMaxScaler()


    # -------------------------------------------------
    # FIT SCALER ONLY ON TRAINING DATA
    # -------------------------------------------------

    scaler.fit(
        train_data
    )


    # -------------------------------------------------
    # TRANSFORM ALL DATA USING TRAINING SCALER
    # -------------------------------------------------

    scaled_train = scaler.transform(
        train_data
    )


    scaled_val = scaler.transform(
        val_data
    )


    scaled_test = scaler.transform(
        test_data
    )


    # -------------------------------------------------
    # CREATE TRAINING SEQUENCES
    # -------------------------------------------------

    X_train, y_train = create_sequences(
        scaled_train,
        sequence_length
    )


    # -------------------------------------------------
    # CREATE VALIDATION SEQUENCES
    # -------------------------------------------------

    X_val, y_val = create_sequences(
        scaled_val,
        sequence_length
    )


    # -------------------------------------------------
    # CREATE TEST SEQUENCES
    # -------------------------------------------------

    X_test, y_test = create_sequences(
        scaled_test,
        sequence_length
    )


    # -------------------------------------------------
    # RETURN ALL DATASETS
    # -------------------------------------------------

    return (
        X_train,
        y_train,

        X_val,
        y_val,

        X_test,
        y_test,

        scaler
    )
