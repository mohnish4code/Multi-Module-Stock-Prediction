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


from modules.data_collection import (
    get_stock_data
)

from modules.technical_analysis import (
    analyze_technical_indicators
)


# -------------------------------------------------
# SETTINGS
# -------------------------------------------------

TICKER = "TCS.NS"


# -------------------------------------------------
# FETCH STOCK DATA
# -------------------------------------------------

print(
    "\nFetching stock data..."
)


df = get_stock_data(

    ticker=TICKER,

    period="1y",

    interval="1d"
)


# -------------------------------------------------
# RUN TECHNICAL ANALYSIS
# -------------------------------------------------

print(
    "Running technical analysis..."
)


result = analyze_technical_indicators(
    df
)


# -------------------------------------------------
# DISPLAY RESULTS
# -------------------------------------------------

print(
    "\n" + "=" * 60
)

print(
    "TECHNICAL ANALYSIS TEST"
)

print(
    "=" * 60
)


print(
    f"\nRSI: "
    f"{result['rsi']:.2f}"
)


print(
    f"RSI Signal: "
    f"{result['rsi_signal']}"
)


print(
    f"\nSMA 20: "
    f"{result['sma_20']:.2f}"
)


print(
    f"SMA 50: "
    f"{result['sma_50']:.2f}"
)


print(
    f"Trend: "
    f"{result['trend']}"
)


print(
    f"\nMACD: "
    f"{result['macd']:.4f}"
)


print(
    f"MACD Signal: "
    f"{result['macd_signal']:.4f}"
)


print(
    f"MACD Trend: "
    f"{result['macd_trend']}"
)


print(
    f"\nVolatility: "
    f"{result['volatility']:.4f}"
)


print(
    f"Volatility Level: "
    f"{result['volatility_level']}"
)


print(
    f"\nBullish Signals: "
    f"{result['bullish_signals']}"
)


print(
    f"Bearish Signals: "
    f"{result['bearish_signals']}"
)


print(
    f"Neutral Signals: "
    f"{result['neutral_signals']}"
)


print(
    f"\nOverall Technical Signal: "
    f"{result['technical_signal']}"
)


print(
    "\nSIGNALS:"
)


for signal in result["signals"]:

    print(
        f"- {signal}"
    )


print(
    "\n" + "=" * 60
)

print(
    "TECHNICAL ANALYSIS TEST COMPLETED"
)

print(
    "=" * 60
)