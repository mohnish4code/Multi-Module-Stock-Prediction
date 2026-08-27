import os
import sys
import joblib

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint
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
    prepare_train_val_test_data
)

from models.model import build_lstm_model

from config import COMPANIES


# =================================================
# GET COMPANY FROM COMMAND LINE
# =================================================

if len(sys.argv) < 2:

    print("\nPlease provide a company name.")

    print("\nAvailable companies:")

    for company_name in COMPANIES:

        display_name = company_name.replace(
            ".NS",
            ""
        )

        print(f"- {display_name}")

    print("\nExample:")

    print(
        "python training/train_model.py TCS"
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
        f"\nError: Company "
        f"'{input_company}' not found."
    )

    print("\nAvailable companies:")

    for company in COMPANIES:

        display_name = company.replace(
            ".NS",
            ""
        )

        print(f"- {display_name}")

    sys.exit()


# =================================================
# GET COMPANY INFORMATION
# =================================================

company = COMPANIES[
    company_key
]


# -------------------------------------------------
# HANDLE CONFIG STRUCTURE
# -------------------------------------------------

if isinstance(company, dict):

    ticker = company.get(
        "ticker",
        company_key
    )

else:

    ticker = company


# =================================================
# CREATE CLEAN COMPANY NAME
# =================================================

company_name = company_key.replace(
    ".NS",
    ""
)


# =================================================
# DISPLAY TRAINING INFORMATION
# =================================================

print("\n" + "=" * 60)

print(
    f"TRAINING STOCK MODEL FOR: {company_name}"
)

print(
    f"STOCK TICKER: {ticker}"
)

print("=" * 60)


# =================================================
# CREATE COMPANY MODEL DIRECTORY
# =================================================

model_directory = os.path.join(
    PROJECT_ROOT,
    "models",
    company_name
)

os.makedirs(
    model_directory,
    exist_ok=True
)


# =================================================
# FETCH 10 YEARS OF STOCK DATA
# =================================================

print(
    f"\nFetching 10 years of stock data "
    f"for {company_name}..."
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
        "\nError: No stock data was fetched."
    )

    print(
        "Please check the ticker symbol "
        "and internet connection."
    )

    sys.exit()


print(
    f"\nTotal raw records fetched: "
    f"{len(df)}"
)


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


print(
    f"Records after feature engineering: "
    f"{len(df)}"
)


# =================================================
# PREPARE TRAIN / VALIDATION / TEST DATA
# =================================================

print(
    "\nPreparing training, validation "
    "and testing data..."
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

    df,

    sequence_length=60,

    train_ratio=0.70,

    val_ratio=0.15
)


# =================================================
# DISPLAY DATA SHAPES
# =================================================

print(
    "\nData Preparation Completed!"
)


print(
    "\nDATA SPLIT:"
)

print(
    "Training:   70%"
)

print(
    "Validation: 15%"
)

print(
    "Testing:    15%"
)


print(
    f"\nTraining data shape: "
    f"{X_train.shape}"
)


print(
    f"Validation data shape: "
    f"{X_val.shape}"
)


print(
    f"Testing data shape: "
    f"{X_test.shape}"
)


print(
    f"\nTraining samples: "
    f"{len(X_train)}"
)


print(
    f"Validation samples: "
    f"{len(X_val)}"
)


print(
    f"Testing samples: "
    f"{len(X_test)}"
)


# =================================================
# BUILD LSTM MODEL
# =================================================

print(
    "\nBuilding LSTM model..."
)


model = build_lstm_model(

    sequence_length=X_train.shape[1],

    num_features=X_train.shape[2]
)


# =================================================
# DISPLAY MODEL ARCHITECTURE
# =================================================

print(
    "\nModel Architecture:\n"
)


model.summary()


# =================================================
# MODEL PATH
# =================================================

model_path = os.path.join(
    model_directory,
    "lstm_model.keras"
)


# =================================================
# EARLY STOPPING
# =================================================

early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=10,

    restore_best_weights=True,

    verbose=1
)


# =================================================
# MODEL CHECKPOINT
# =================================================

model_checkpoint = ModelCheckpoint(

    filepath=model_path,

    monitor="val_loss",

    save_best_only=True,

    mode="min",

    verbose=1
)


# =================================================
# TRAIN MODEL
# =================================================

print(
    "\nStarting model training..."
)

print(
    "Training data: 70%"
)

print(
    "Validation data: 15%"
)

print(
    "Final test data: 15% "
    "(not used during training)"
)

print(
    "\nThe best model will be selected "
    "using validation loss.\n"
)


history = model.fit(

    X_train,

    y_train,

    validation_data=(

        X_val,

        y_val
    ),

    epochs=100,

    batch_size=32,

    callbacks=[

        early_stopping,

        model_checkpoint
    ],

    verbose=1
)


# =================================================
# TRAINING COMPLETION
# =================================================

print(
    "\nTraining completed."
)

print(
    "The best model was selected "
    "using validation data only."
)

print(
    "The test data remains completely "
    "unseen for final evaluation."
)


# =================================================
# SCALER PATH
# =================================================

scaler_path = os.path.join(
    model_directory,
    "scaler.pkl"
)


# =================================================
# SAVE SCALER
# =================================================

print(
    "\nSaving data scaler..."
)


joblib.dump(
    scaler,
    scaler_path
)


# =================================================
# SAVE TRAINING INFORMATION
# =================================================

training_info_path = os.path.join(
    model_directory,
    "training_info.txt"
)


with open(
    training_info_path,
    "w"
) as file:

    file.write(
        f"Company: {company_name}\n"
    )

    file.write(
        f"Ticker: {ticker}\n\n"
    )

    file.write(
        "DATA SPLIT\n"
    )

    file.write(
        "Training: 70%\n"
    )

    file.write(
        "Validation: 15%\n"
    )

    file.write(
        "Testing: 15%\n\n"
    )

    file.write(
        f"Training samples: "
        f"{len(X_train)}\n"
    )

    file.write(
        f"Validation samples: "
        f"{len(X_val)}\n"
    )

    file.write(
        f"Testing samples: "
        f"{len(X_test)}\n\n"
    )

    file.write(
        f"Total epochs trained: "
        f"{len(history.history['loss'])}\n"
    )

    file.write(
        f"Best validation loss: "
        f"{min(history.history['val_loss']):.6f}\n"
    )

    file.write(
        f"Best validation MAE: "
        f"{min(history.history['val_mae']):.6f}\n"
    )


# =================================================
# FINAL RESULTS
# =================================================

print(
    "\n" + "=" * 60
)

print(
    "MODEL TRAINING COMPLETED SUCCESSFULLY"
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


print(
    "\nData Split:"
)


print(
    f"Training: "
    f"{len(X_train)} samples"
)


print(
    f"Validation: "
    f"{len(X_val)} samples"
)


print(
    f"Testing: "
    f"{len(X_test)} samples"
)


print(
    "\nBest Model saved to:"
)


print(
    model_path
)


print(
    "\nScaler saved to:"
)


print(
    scaler_path
)


print(
    "\nTraining information saved to:"
)


print(
    training_info_path
)


print(
    f"\nTotal epochs trained: "
    f"{len(history.history['loss'])}"
)


print(
    f"Best validation loss: "
    f"{min(history.history['val_loss']):.6f}"
)


print(
    f"Best validation MAE: "
    f"{min(history.history['val_mae']):.6f}"
)


print(
    "\nFinal test evaluation "
    "has not yet been performed."
)


print(
    "=" * 60
)
