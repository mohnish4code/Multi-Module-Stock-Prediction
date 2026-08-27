import os
import sys


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


sys.path.insert(
    0,
    PROJECT_ROOT
)

from modules.data_collection import get_stock_data
from modules.feature_engineering import add_technical_indicators


# Fetch TCS data
df = get_stock_data(
    ticker="TCS.NS",
    period="1y",
    interval="1d"
)


# Add technical indicators
df = add_technical_indicators(df)


# Print all column names
print("\nColumns:\n")

print(df.columns)


# Print latest rows
print("\nLatest Data with Indicators:\n")

print(
    df[
        [
            "Close",
            "SMA_20",
            "EMA_20",
            "RSI",
            "MACD",
            "MACD_Signal",
            "BB_Upper",
            "BB_Middle",
            "BB_Lower"
        ]
    ].tail()
)