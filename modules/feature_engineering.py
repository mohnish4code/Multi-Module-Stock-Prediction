import pandas as pd

from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands


# -------------------------------------------------
# ADD TECHNICAL INDICATORS
# -------------------------------------------------

def add_technical_indicators(df):
    """
    Adds all technical indicators required by:

    1. LSTM preprocessing
    2. Technical analysis

    The existing trained LSTM models use the original
    14 FEATURE_COLUMNS, so those column names must
    remain unchanged.
    """

    # ---------------------------------------------
    # SAFETY COPY
    # ---------------------------------------------

    df = df.copy()


    # ---------------------------------------------
    # FIX MULTI-LEVEL YFINANCE COLUMNS
    # ---------------------------------------------

    if isinstance(
        df.columns,
        pd.MultiIndex
    ):

        df.columns = (
            df.columns
            .get_level_values(0)
        )


    # ---------------------------------------------
    # VALIDATE REQUIRED COLUMNS
    # ---------------------------------------------

    required_columns = [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]


    missing_columns = [

        column

        for column in required_columns

        if column not in df.columns
    ]


    if missing_columns:

        raise ValueError(
            "Missing required columns: "
            f"{missing_columns}"
        )


    # ---------------------------------------------
    # MAKE SURE COLUMNS ARE NUMERIC
    # ---------------------------------------------

    for column in required_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )


    # ---------------------------------------------
    # CLOSE PRICE SERIES
    # ---------------------------------------------

    close = df["Close"].squeeze()


    # ---------------------------------------------
    # SMA 20
    # ---------------------------------------------

    sma_20_indicator = SMAIndicator(

        close=close,

        window=20
    )


    df["SMA_20"] = (
        sma_20_indicator
        .sma_indicator()
    )


    # ---------------------------------------------
    # SMA 50
    #
    # Used by technical analysis only.
    # Not added to the LSTM FEATURE_COLUMNS.
    # ---------------------------------------------

    sma_50_indicator = SMAIndicator(

        close=close,

        window=50
    )


    df["SMA_50"] = (
        sma_50_indicator
        .sma_indicator()
    )


    # ---------------------------------------------
    # EMA 20
    # ---------------------------------------------

    ema_20_indicator = EMAIndicator(

        close=close,

        window=20
    )


    df["EMA_20"] = (
        ema_20_indicator
        .ema_indicator()
    )


    # ---------------------------------------------
    # RSI
    # ---------------------------------------------

    rsi_indicator = RSIIndicator(

        close=close,

        window=14
    )


    df["RSI"] = (
        rsi_indicator
        .rsi()
    )


    # ---------------------------------------------
    # MACD
    # ---------------------------------------------

    macd_indicator = MACD(

        close=close,

        window_slow=26,

        window_fast=12,

        window_sign=9
    )


    df["MACD"] = (
        macd_indicator
        .macd()
    )


    df["MACD_Signal"] = (
        macd_indicator
        .macd_signal()
    )


    df["MACD_Histogram"] = (
        macd_indicator
        .macd_diff()
    )


    # ---------------------------------------------
    # BOLLINGER BANDS
    # ---------------------------------------------

    bollinger = BollingerBands(

        close=close,

        window=20,

        window_dev=2
    )


    df["BB_Upper"] = (
        bollinger
        .bollinger_hband()
    )


    df["BB_Middle"] = (
        bollinger
        .bollinger_mavg()
    )


    df["BB_Lower"] = (
        bollinger
        .bollinger_lband()
    )


    # ---------------------------------------------
    # DAILY RETURNS
    #
    # Used by technical analysis only.
    # Not part of the trained LSTM features.
    # ---------------------------------------------

    df["Returns"] = (
        df["Close"]
        .pct_change()
    )


    # ---------------------------------------------
    # 20-DAY VOLATILITY
    #
    # Used by technical analysis and risk engine.
    # Not part of the trained LSTM features.
    # ---------------------------------------------

    df["Volatility"] = (

        df["Returns"]

        .rolling(
            window=20
        )

        .std()
    )


    # ---------------------------------------------
    # REMOVE NaN VALUES
    #
    # SMA 50 requires at least 50 rows.
    # ---------------------------------------------

    df.dropna(

        inplace=True
    )


    # ---------------------------------------------
    # FINAL CHECK
    # ---------------------------------------------

    if len(df) == 0:

        raise ValueError(
            "Not enough data after calculating "
            "technical indicators."
        )


    return df