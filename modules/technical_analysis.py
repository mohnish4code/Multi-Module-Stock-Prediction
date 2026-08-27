import pandas as pd

from modules.feature_engineering import (
    add_technical_indicators
)


# -------------------------------------------------
# TECHNICAL ANALYSIS MODULE
# -------------------------------------------------

def analyze_technical_indicators(df):
    """
    Analyzes the latest technical indicators.

    This function uses the same indicator calculations
    from feature_engineering.py so the LSTM model and
    technical analysis are based on consistent data.
    """


    # ---------------------------------------------
    # SAFETY CHECK
    # ---------------------------------------------

    df = df.copy()


    if "Close" not in df.columns:

        raise ValueError(
            "The DataFrame must contain "
            "a 'Close' column."
        )


    # ---------------------------------------------
    # CALCULATE TECHNICAL INDICATORS
    #
    # We use the shared feature engineering module.
    # ---------------------------------------------

    df = add_technical_indicators(
        df
    )


    # ---------------------------------------------
    # GET LATEST MARKET VALUES
    # ---------------------------------------------

    latest = df.iloc[-1]


    signals = []

    bullish_signals = 0

    bearish_signals = 0

    neutral_signals = 0


    # ---------------------------------------------
    # RSI ANALYSIS
    # ---------------------------------------------

    rsi = float(
        latest["RSI"]
    )


    if rsi < 30:

        rsi_signal = "BULLISH"

        signals.append(
            f"RSI is {rsi:.2f}, indicating "
            "an oversold market."
        )

        bullish_signals += 1


    elif rsi > 70:

        rsi_signal = "BEARISH"

        signals.append(
            f"RSI is {rsi:.2f}, indicating "
            "an overbought market."
        )

        bearish_signals += 1


    else:

        rsi_signal = "NEUTRAL"

        signals.append(
            f"RSI is {rsi:.2f}, indicating "
            "neutral market conditions."
        )

        neutral_signals += 1


    # ---------------------------------------------
    # SMA TREND ANALYSIS
    # ---------------------------------------------

    sma_20 = float(
        latest["SMA_20"]
    )


    sma_50 = float(
        latest["SMA_50"]
    )


    if sma_20 > sma_50:

        trend = "BULLISH"

        signals.append(
            "Short-term trend is bullish because "
            "SMA 20 is above SMA 50."
        )

        bullish_signals += 1


    elif sma_20 < sma_50:

        trend = "BEARISH"

        signals.append(
            "Short-term trend is bearish because "
            "SMA 20 is below SMA 50."
        )

        bearish_signals += 1


    else:

        trend = "NEUTRAL"

        signals.append(
            "SMA 20 and SMA 50 are equal, "
            "indicating a neutral trend."
        )

        neutral_signals += 1


    # ---------------------------------------------
    # MACD ANALYSIS
    # ---------------------------------------------

    macd = float(
        latest["MACD"]
    )


    macd_signal = float(
        latest["MACD_Signal"]
    )


    if macd > macd_signal:

        macd_trend = "BULLISH"

        signals.append(
            "MACD indicates positive "
            "market momentum."
        )

        bullish_signals += 1


    elif macd < macd_signal:

        macd_trend = "BEARISH"

        signals.append(
            "MACD indicates negative "
            "market momentum."
        )

        bearish_signals += 1


    else:

        macd_trend = "NEUTRAL"

        signals.append(
            "MACD indicates neutral "
            "market momentum."
        )

        neutral_signals += 1


    # ---------------------------------------------
    # VOLATILITY ANALYSIS
    # ---------------------------------------------

    volatility = float(
        latest["Volatility"]
    )


    if volatility < 0.01:

        volatility_level = "LOW"


    elif volatility < 0.03:

        volatility_level = "MEDIUM"


    else:

        volatility_level = "HIGH"


    signals.append(
        f"Market volatility is "
        f"{volatility_level} "
        f"({volatility:.4f})."
    )


    # ---------------------------------------------
    # OVERALL TECHNICAL SIGNAL
    # ---------------------------------------------

    if (
        bullish_signals
        > bearish_signals
    ):

        technical_signal = "BULLISH"


    elif (
        bearish_signals
        > bullish_signals
    ):

        technical_signal = "BEARISH"


    else:

        technical_signal = "NEUTRAL"


    # ---------------------------------------------
    # RETURN RESULTS
    # ---------------------------------------------

    return {

        "rsi":
        rsi,

        "rsi_signal":
        rsi_signal,

        "sma_20":
        sma_20,

        "sma_50":
        sma_50,

        "trend":
        trend,

        "macd":
        macd,

        "macd_signal":
        macd_signal,

        "macd_trend":
        macd_trend,

        "volatility":
        volatility,

        "volatility_level":
        volatility_level,

        "technical_signal":
        technical_signal,

        "bullish_signals":
        bullish_signals,

        "bearish_signals":
        bearish_signals,

        "neutral_signals":
        neutral_signals,

        "signals":
        signals

    }