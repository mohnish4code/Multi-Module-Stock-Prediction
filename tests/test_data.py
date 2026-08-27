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

ticker = "TCS.NS"

df = get_stock_data(ticker)

print(df.tail())

print("\nLatest Close Price:")

print(df["Close"].iloc[-1])