from data_collection import get_stock_data
from feature_engineering import add_technical_indicators
from preprocessing import prepare_train_test_data


# -------------------------------------------------
# GET REAL STOCK DATA
# -------------------------------------------------

df = get_stock_data(
    ticker="TCS.NS",
    period="1y",
    interval="1d"
)


# -------------------------------------------------
# ADD TECHNICAL INDICATORS
# -------------------------------------------------

df = add_technical_indicators(df)


# -------------------------------------------------
# PREPARE TRAIN AND TEST DATA
# -------------------------------------------------

(
    X_train,
    y_train,
    X_test,
    y_test,
    scaler
) = prepare_train_test_data(
    df,
    sequence_length=60,
    train_ratio=0.80
)


# -------------------------------------------------
# PRINT RESULTS
# -------------------------------------------------

print("\nData Preparation Successful!\n")

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

print()

print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

print()

print(
    "Each input sample:",
    X_train.shape[1],
    "days ×",
    X_train.shape[2],
    "features"
)

