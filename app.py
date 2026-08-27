# =========================================================
# MULTI-MODULE STOCK PREDICTION SYSTEM
# PROFESSIONAL AI MARKET INTELLIGENCE TERMINAL
# =========================================================

import os
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from tensorflow.keras.models import load_model


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Market Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS
# =========================================================

st.markdown(
    """
    <style>

    /* =====================================================
       GLOBAL
    ===================================================== */

    .stApp {
        background:
            radial-gradient(
                circle at 88% 0%,
                rgba(37, 99, 235, 0.10),
                transparent 30%
            ),
            radial-gradient(
                circle at 5% 95%,
                rgba(16, 185, 129, 0.04),
                transparent 30%
            ),
            #08111f;

        color: #e5e7eb;
    }

    .block-container {
        max-width: 1600px;
        padding-top: 1.5rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3, h4 {
        color: #f8fafc !important;
    }

    p, span, label {
        color: #cbd5e1;
    }


    /* =====================================================
       HIDE DEFAULT STREAMLIT ELEMENTS
    ===================================================== */

    #MainMenu {
        visibility: hidden;
    }

    footer {
        visibility: hidden;
    }

    header {
        background: transparent !important;
    }


    /* =====================================================
       SIDEBAR
    ===================================================== */

    [data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #0b1423 0%,
                #08111f 50%,
                #050a13 100%
            );

        border-right: 1px solid #1d2b3f;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1.2rem;
    }

    [data-testid="stSidebar"] * {
        color: #dbe5f0;
    }

    [data-testid="stSidebar"] hr {
        border-color: #1d2b3f;
        margin-top: 1.4rem;
        margin-bottom: 1.4rem;
    }


    /* =====================================================
       METRIC CARDS
    ===================================================== */

    [data-testid="stMetric"] {
        background:
            linear-gradient(
                145deg,
                rgba(17, 29, 46, 0.98),
                rgba(12, 22, 37, 0.98)
            );

        border: 1px solid #263a54;

        border-radius: 14px;

        padding: 1rem 1.1rem;

        min-height: 120px;

        box-shadow:
            0 10px 28px
            rgba(0, 0, 0, 0.18);

        transition:
            transform 0.2s ease,
            border-color 0.2s ease;
    }

    [data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: #3b82f6;
    }

    [data-testid="stMetricLabel"] {
        color: #8fa4bc !important;

        font-size: 0.72rem !important;

        font-weight: 750 !important;

        text-transform: uppercase;

        letter-spacing: 1px;
    }

    [data-testid="stMetricValue"] {
        color: #f8fafc !important;

        font-size: 1.5rem !important;

        font-weight: 800 !important;
    }

    [data-testid="stMetricDelta"] {
        font-size: 0.8rem !important;
    }


    /* =====================================================
       BUTTONS
    ===================================================== */

    .stButton > button {
        width: 100%;

        min-height: 2.9rem;

        background:
            linear-gradient(
                90deg,
                #2563eb,
                #4f46e5
            ) !important;

        color: #ffffff !important;

        border:
            1px solid
            rgba(96, 165, 250, 0.40) !important;

        border-radius: 10px !important;

        font-weight: 750 !important;

        letter-spacing: 0.2px;

        box-shadow:
            0 8px 20px
            rgba(37, 99, 235, 0.22);

        transition:
            transform 0.2s ease,
            box-shadow 0.2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-1px);

        box-shadow:
            0 12px 28px
            rgba(37, 99, 235, 0.35);
    }


    /* =====================================================
       SELECTBOX
    ===================================================== */

    [data-baseweb="select"] > div {
        background-color: #0d1828 !important;

        border-color: #263a54 !important;

        color: #e5e7eb !important;

        border-radius: 10px !important;
    }


    /* =====================================================
       EXPANDERS
    ===================================================== */

    [data-testid="stExpander"] {
        background:
            linear-gradient(
                145deg,
                rgba(13, 24, 40, 0.96),
                rgba(10, 19, 32, 0.96)
            );

        border: 1px solid #263a54;

        border-radius: 13px;

        overflow: hidden;

        margin-bottom: 0.9rem;
    }

    [data-testid="stExpander"] summary {
        color: #e5e7eb !important;

        font-weight: 750 !important;

        font-size: 0.95rem !important;
    }


    /* =====================================================
       PROGRESS
    ===================================================== */

    [data-testid="stProgressBar"] {
        background-color: #15243a;
    }

    [data-testid="stProgressBar"] > div > div {
        background:
            linear-gradient(
                90deg,
                #2563eb,
                #38bdf8
            );
    }


    /* =====================================================
       ALERTS
    ===================================================== */

    [data-testid="stAlert"] {
        border-radius: 10px;
        border: 1px solid #2a3d56;
    }


    /* =====================================================
       DIVIDERS
    ===================================================== */

    hr {
        border-color: #1e2f45 !important;
    }


    /* =====================================================
       SIDEBAR CAPTION
    ===================================================== */

    [data-testid="stSidebar"] .stCaption {
        color: #6f8298 !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# PROJECT ROOT
# =========================================================

PROJECT_ROOT = os.path.dirname(
    os.path.abspath(__file__)
)

if PROJECT_ROOT not in sys.path:

    sys.path.insert(
        0,
        PROJECT_ROOT
    )


# =========================================================
# IMPORT PROJECT MODULES
# =========================================================

from modules.data_collection import (
    get_stock_data
)

from modules.feature_engineering import (
    add_technical_indicators
)

from modules.preprocessing import (
    FEATURE_COLUMNS
)

from modules.technical_analysis import (
    analyze_technical_indicators
)

from modules.prediction_signal import (
    get_prediction_signal
)

from modules.news_sentiment import (
    analyze_company_sentiment
)

from modules.coordination_engine import (
    coordinate_modules
)

from modules.confidence_engine import (
    calculate_confidence
)

from modules.risk_engine import (
    calculate_risk
)

from modules.recommendation_engine import (
    get_final_recommendation
)

from config import (
    COMPANIES,
    SEQUENCE_LENGTH
)


# =========================================================
# SESSION STATE
# =========================================================

if "analysis_requested" not in st.session_state:

    st.session_state.analysis_requested = False

if "last_ticker" not in st.session_state:

    st.session_state.last_ticker = None


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def get_model_paths(ticker):

    company_key = ticker.replace(
        ".NS",
        ""
    )

    model_path = os.path.join(
        PROJECT_ROOT,
        "models",
        company_key,
        "lstm_model.keras"
    )

    scaler_path = os.path.join(
        PROJECT_ROOT,
        "models",
        company_key,
        "scaler.pkl"
    )

    return (
        model_path,
        scaler_path
    )


# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def load_company_model(ticker):

    model_path, scaler_path = get_model_paths(
        ticker
    )

    if not os.path.exists(model_path):

        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    if not os.path.exists(scaler_path):

        raise FileNotFoundError(
            f"Scaler not found: {scaler_path}"
        )

    model = load_model(
        model_path
    )

    scaler = joblib.load(
        scaler_path
    )

    return (
        model,
        scaler
    )


# =========================================================
# SIGNAL HELPERS
# =========================================================

def get_signal_emoji(signal):

    signal = str(
        signal
    ).upper()

    if signal in [
        "BUY",
        "BULLISH",
        "POSITIVE"
    ]:

        return "🟢"

    elif signal in [
        "SELL",
        "BEARISH",
        "NEGATIVE"
    ]:

        return "🔴"

    return "🟡"


def sentiment_badge(sentiment):

    sentiment = str(
        sentiment
    ).upper()

    if sentiment == "POSITIVE":

        return "🟢 Positive"

    elif sentiment == "NEGATIVE":

        return "🔴 Negative"

    return "🟡 Neutral"


# =========================================================
# SIDEBAR BRAND
# =========================================================

st.sidebar.title(
    "📈 Market Terminal"
)

st.sidebar.caption(
    "MULTI-MODULE STOCK INTELLIGENCE"
)

st.sidebar.divider()


# =========================================================
# MARKET EXPLORER
# =========================================================

st.sidebar.subheader(
    "Market Explorer"
)

company_options = {}

for ticker_key, details in COMPANIES.items():

    display_name = (
        f"{details['name']} "
        f"({ticker_key})"
    )

    company_options[
        display_name
    ] = ticker_key


selected_company = st.sidebar.selectbox(
    "Select Stock",
    list(company_options.keys())
)


ticker = company_options[
    selected_company
]

company_name = COMPANIES[
    ticker
]["name"]


# =========================================================
# ACTIVE INSTRUMENT
# =========================================================

with st.sidebar.container(
    border=True
):

    st.caption(
        "ACTIVE INSTRUMENT"
    )

    st.subheader(
        company_name
    )

    st.code(
        ticker,
        language=None
    )


# =========================================================
# RUN ANALYSIS
# =========================================================

if st.sidebar.button(
    "⚡ Run AI Market Analysis",
    use_container_width=True
):

    st.session_state.analysis_requested = True

    st.session_state.last_ticker = ticker


# If user selects a different stock,
# return to the welcome screen until they run analysis again.

if (
    st.session_state.last_ticker is not None
    and
    st.session_state.last_ticker != ticker
):

    st.session_state.analysis_requested = False


st.sidebar.divider()


# =========================================================
# INTELLIGENCE MODULES
# =========================================================

st.sidebar.subheader(
    "Intelligence Modules"
)

st.sidebar.markdown(
    """
- 🤖 **LSTM Forecasting**
- 📊 **Technical Analysis**
- 📰 **News Sentiment**
- 🔗 **Signal Coordination**
- 🎯 **Confidence Engine**
- ⚠️ **Risk Engine**
"""
)

st.sidebar.divider()

st.sidebar.caption(
    "Forecast target: next available trading-day close."
)


# =========================================================
# TOP HEADER
# =========================================================

header_left, header_right = st.columns(
    [5, 1]
)

with header_left:

    st.title(
        "Market Intelligence Terminal"
    )

    st.caption(
        "AI-powered forecasting, technical signals and "
        "financial intelligence in one decision dashboard."
    )


with header_right:

    st.metric(
        "System Status",
        "READY",
        "● Online"
    )


st.divider()


# =========================================================
# INITIAL SCREEN
# =========================================================

if not st.session_state.analysis_requested:

    st.header(
        "AI Trading Intelligence"
    )

    st.write(
        "Select a company from the Market Explorer and "
        "run the multi-module market analysis."
    )

    st.divider()


    # =====================================================
    # PIPELINE
    # =====================================================

    st.subheader(
        "Analysis Pipeline"
    )

    pipeline_cols = st.columns(3)


    with pipeline_cols[0]:

        with st.container(
            border=True
        ):

            st.markdown(
                "### 🤖 LSTM Forecasting"
            )

            st.write(
                "Uses the previous "
                f"{SEQUENCE_LENGTH} trading days and "
                "engineered market features to estimate "
                "the next available trading-day closing price."
            )


    with pipeline_cols[1]:

        with st.container(
            border=True
        ):

            st.markdown(
                "### 📊 Technical Intelligence"
            )

            st.write(
                "Evaluates RSI, moving averages, MACD, "
                "market trends and volatility to determine "
                "the current technical market signal."
            )


    with pipeline_cols[2]:

        with st.container(
            border=True
        ):

            st.markdown(
                "### 📰 News Intelligence"
            )

            st.write(
                "Analyzes recent company-specific financial "
                "news to identify positive, negative or "
                "neutral market sentiment."
            )


    st.divider()

    st.subheader(
        "Intelligence Architecture"
    )

    architecture_cols = st.columns(3)


    with architecture_cols[0]:

        with st.container(
            border=True
        ):

            st.markdown(
                "### 🔗 Signal Coordination"
            )

            st.write(
                "Combines prediction direction, technical "
                "signals and financial news sentiment."
            )


    with architecture_cols[1]:

        with st.container(
            border=True
        ):

            st.markdown(
                "### 🎯 Confidence Engine"
            )

            st.write(
                "Measures agreement between independent "
                "intelligence modules to estimate forecast confidence."
            )


    with architecture_cols[2]:

        with st.container(
            border=True
        ):

            st.markdown(
                "### ⚠️ Risk Engine"
            )

            st.write(
                "Evaluates volatility, signal agreement and "
                "forecast movement to determine market risk."
            )


    st.info(
        "⚡ Ready for analysis. Select an instrument and "
        "click **Run AI Market Analysis**."
    )

    st.stop()


# =========================================================
# SYSTEM EXECUTION
# =========================================================

try:

    progress = st.progress(0)

    status = st.empty()


    # =====================================================
    # STEP 1 - LOAD MODEL
    # =====================================================

    status.info(
        "🤖 Initializing LSTM prediction model..."
    )

    model, scaler = load_company_model(
        ticker
    )

    progress.progress(10)


    # =====================================================
    # STEP 2 - FETCH DATA
    # =====================================================

    status.info(
        "📡 Fetching latest market data..."
    )

    df = get_stock_data(
        ticker=ticker,
        period="1y",
        interval="1d"
    )

    if df is None or df.empty:

        raise ValueError(
            "No stock data was received."
        )

    progress.progress(20)


    # =====================================================
    # STEP 3 - FEATURES
    # =====================================================

    status.info(
        "⚙️ Calculating technical indicators..."
    )

    df_features = add_technical_indicators(
        df
    )

    progress.progress(30)


    # =====================================================
    # STEP 4 - TECHNICAL ANALYSIS
    # =====================================================

    status.info(
        "📊 Running technical analysis..."
    )

    technical_result = (
        analyze_technical_indicators(
            df_features
        )
    )

    progress.progress(40)


    # =====================================================
    # STEP 5 - PREPARE MODEL INPUT
    # =====================================================

    status.info(
        "🧠 Preparing LSTM input sequence..."
    )

    data = df_features[
        FEATURE_COLUMNS
    ].copy()

    data.dropna(
        inplace=True
    )

    if len(data) < SEQUENCE_LENGTH:

        raise ValueError(
            "Not enough stock data to "
            "create prediction sequence."
        )


    current_price = float(
        data["Close"].iloc[-1]
    )


    # =====================================================
    # STEP 6 - SCALE DATA
    # =====================================================

    scaled_data = scaler.transform(
        data
    )

    latest_sequence = scaled_data[
        -SEQUENCE_LENGTH:
    ]

    X_latest = np.expand_dims(
        latest_sequence,
        axis=0
    )

    progress.progress(50)


    # =====================================================
    # STEP 7 - PREDICTION
    # =====================================================

    status.info(
        "📈 Generating next trading-day forecast..."
    )

    scaled_prediction = model.predict(
        X_latest,
        verbose=0
    )

    scaled_prediction = float(
        scaled_prediction[0][0]
    )


    close_index = FEATURE_COLUMNS.index(
        "Close"
    )


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


    original_prediction = (
        scaler.inverse_transform(
            prediction_row
        )
    )


    predicted_price = float(
        original_prediction[
            0,
            close_index
        ]
    )


    predicted_change_percent = (
        (
            predicted_price
            - current_price
        )
        / current_price
    ) * 100


    price_change = (
        predicted_price
        - current_price
    )

    progress.progress(60)


    # =====================================================
    # STEP 8 - PREDICTION SIGNAL
    # =====================================================

    status.info(
        "🤖 Generating LSTM market signal..."
    )

    prediction_result = (
        get_prediction_signal(
            current_price,
            predicted_price
        )
    )


    prediction_direction = (
        prediction_result[
            "prediction_direction"
        ]
    )


    recommendation_signal = (
        prediction_result[
            "recommendation_signal"
        ]
    )

    progress.progress(68)


    # =====================================================
    # STEP 9 - NEWS SENTIMENT
    # =====================================================

    status.info(
        "📰 Analyzing latest company news..."
    )

    news_result = (
        analyze_company_sentiment(
            company_name=company_name,
            ticker=ticker,
            max_results=10
        )
    )


    news_sentiment = (
        news_result.get(
            "sentiment_label",
            "UNKNOWN"
        )
        .upper()
        .strip()
    )

    progress.progress(75)


    # =====================================================
    # STEP 10 - COORDINATION
    # =====================================================

    status.info(
        "🔗 Coordinating AI modules..."
    )


    technical_signal = (
        technical_result[
            "technical_signal"
        ]
    )


    coordination_result = (
        coordinate_modules(
            prediction_direction,
            technical_signal,
            news_sentiment
        )
    )


    agreement = (
        coordination_result[
            "agreement"
        ]
    )

    progress.progress(82)


    # =====================================================
    # STEP 11 - CONFIDENCE
    # =====================================================

    status.info(
        "🎯 Calculating forecast confidence..."
    )


    confidence_result = (
        calculate_confidence(
            prediction_direction,
            technical_signal,
            news_sentiment,
            agreement
        )
    )


    confidence_score = float(
        confidence_result[
            "confidence_score"
        ]
    )


    confidence_level = (
        confidence_result[
            "confidence_level"
        ]
    )

    progress.progress(88)


    # =====================================================
    # STEP 12 - RISK
    # =====================================================

    status.info(
        "⚠️ Calculating market risk..."
    )


    risk_result = (
        calculate_risk(

            volatility=
            technical_result[
                "volatility_level"
            ],

            predicted_change_percent=
            predicted_change_percent,

            technical_signal=
            technical_signal,

            confidence_score=
            confidence_score,

            agreement=
            agreement
        )
    )


    risk_score = (
        risk_result[
            "risk_score"
        ]
    )


    risk_level = (
        risk_result[
            "risk_level"
        ]
    )

    progress.progress(94)


    # =====================================================
    # STEP 13 - FINAL RECOMMENDATION
    # =====================================================

    status.info(
        "🎯 Generating final AI decision..."
    )


    final_result = (
        get_final_recommendation(
            recommendation_signal,
            news_sentiment
        )
    )

    progress.progress(100)

    status.empty()

    progress.empty()


    # =====================================================
    # RESULTS HEADER
    # =====================================================

    st.header(
        company_name
    )

    result_col1, result_col2, result_col3 = (
        st.columns(
            [2, 1, 1]
        )
    )


    with result_col1:

        st.caption(
            f"Instrument: {ticker}"
        )


    with result_col2:

        st.caption(
            "● ANALYSIS COMPLETE"
        )


    with result_col3:

        st.caption(
            "NEXT TRADING DAY FORECAST"
        )


    st.divider()


    # =====================================================
    # MARKET OVERVIEW
    # =====================================================

    st.subheader(
        "Market Overview"
    )

    st.caption(
        "Latest market close and AI-generated forecast "
        "for the next available trading day."
    )


    col1, col2, col3, col4 = st.columns(4)


    col1.metric(
        "Current Price",
        f"₹{current_price:.2f}"
    )


    col2.metric(
        "AI Forecast",
        f"₹{predicted_price:.2f}"
    )


    col3.metric(
        "Expected Move",
        f"₹{price_change:.2f}",
        f"{predicted_change_percent:.2f}%"
    )


    col4.metric(
        "Forecast Direction",
        prediction_direction
    )


    # =====================================================
    # FINAL DECISION
    # =====================================================

    st.divider()

    st.subheader(
        "Final AI Decision"
    )


    final_recommendation = (
        final_result.get(
            "recommendation",
            recommendation_signal
        )
    )


    recommendation_display = (
        f"{get_signal_emoji(final_recommendation)} "
        f"{final_recommendation}"
    )


    decision_col1, decision_col2, decision_col3 = (
        st.columns(3)
    )


    decision_col1.metric(
        "Recommendation",
        recommendation_display
    )


    decision_col2.metric(
        "Confidence",
        f"{confidence_score:.0f}%"
    )


    decision_col3.metric(
        "Risk Level",
        risk_level
    )


    explanation = final_result.get(
        "explanation",
        (
            "The system combined the available "
            "LSTM, technical and news signals."
        )
    )


    if str(final_recommendation).upper() == "BUY":

        st.success(
            f"**AI Explanation:** {explanation}"
        )

    elif str(final_recommendation).upper() == "SELL":

        st.error(
            f"**AI Explanation:** {explanation}"
        )

    else:

        st.info(
            f"**AI Explanation:** {explanation}"
        )


    # =====================================================
    # PRICE CHART
    # =====================================================

    st.divider()

    st.subheader(
        "Price Trend"
    )

    st.caption(
        "Historical closing prices for the most recent "
        "90 available trading days. The AI forecast is "
        "displayed separately because only one future "
        "prediction is generated."
    )


    chart_data = (
        df_features["Close"]
        .dropna()
        .tail(90)
        .copy()
    )


    if isinstance(
        chart_data.index,
        pd.DatetimeIndex
    ):

        chart_data.index = (
            pd.to_datetime(
                chart_data.index
            )
        )


    st.line_chart(
        chart_data,
        use_container_width=True
    )


    forecast_col1, forecast_col2, forecast_col3 = (
        st.columns(3)
    )


    forecast_col1.metric(
        "Last Market Close",
        f"₹{current_price:.2f}"
    )


    forecast_col2.metric(
        "Predicted Next Close",
        f"₹{predicted_price:.2f}"
    )


    forecast_col3.metric(
        "Forecast Movement",
        f"{predicted_change_percent:.2f}%"
    )


    # =====================================================
    # AI MODULE INTELLIGENCE
    # =====================================================

    st.divider()

    st.subheader(
        "AI Module Intelligence"
    )


    # =====================================================
    # MODULE 01 - LSTM
    # =====================================================

    with st.expander(
        "🤖 Module 01 — LSTM Price Forecasting",
        expanded=True
    ):


        col1, col2, col3 = st.columns(3)


        col1.metric(
            "Prediction Direction",
            prediction_direction
        )


        col2.metric(
            "Recommendation Signal",
            recommendation_signal
        )


        col3.metric(
            "Expected Change",
            f"{predicted_change_percent:.2f}%"
        )


        st.markdown(
            "### Forecast Details"
        )


        st.write(
            f"The LSTM model uses the previous "
            f"{SEQUENCE_LENGTH} trading days and "
            f"{len(FEATURE_COLUMNS)} engineered features "
            f"to forecast the next available trading-day "
            f"closing price."
        )


        st.caption(
            f"Latest Close: ₹{current_price:.2f} "
            f"• Predicted Next Close: ₹{predicted_price:.2f}"
        )


    # =====================================================
    # MODULE 02 - TECHNICAL ANALYSIS
    # =====================================================

    with st.expander(
        "📊 Module 02 — Technical Analysis",
        expanded=False
    ):


        col1, col2, col3, col4 = st.columns(4)


        col1.metric(
            "RSI",
            f"{technical_result['rsi']:.2f}"
        )


        col2.metric(
            "Trend",
            technical_result[
                "trend"
            ]
        )


        col3.metric(
            "Technical Signal",
            technical_signal
        )


        col4.metric(
            "Volatility",
            technical_result[
                "volatility_level"
            ]
        )


        col5, col6, col7 = st.columns(3)


        col5.metric(
            "MACD",
            f"{technical_result['macd']:.4f}"
        )


        col6.metric(
            "MACD Signal",
            f"{technical_result['macd_signal']:.4f}"
        )


        col7.metric(
            "MACD Trend",
            technical_result.get(
                "macd_trend",
                "UNKNOWN"
            )
        )


        signals = technical_result.get(
            "signals",
            []
        )


        if signals:

            st.markdown(
                "### Technical Signals"
            )

            for signal in signals:

                st.write(
                    f"• {signal}"
                )


    # =====================================================
    # MODULE 03 - NEWS
    # =====================================================

    with st.expander(
        "📰 Module 03 — Financial News Sentiment",
        expanded=True
    ):


        col1, col2, col3, col4 = st.columns(4)


        col1.metric(
            "Overall Sentiment",
            news_sentiment
        )


        col2.metric(
            "Sentiment Score",
            f"{news_result.get('sentiment_score', 0.0):.2f}"
        )


        col3.metric(
            "Positive News",
            news_result.get(
                "positive",
                0
            )
        )


        col4.metric(
            "Negative News",
            news_result.get(
                "negative",
                0
            )
        )


        total_news = news_result.get(
            "total_news",
            0
        )


        neutral_news = news_result.get(
            "neutral",
            0
        )


        st.caption(
            f"Total Headlines: {total_news} "
            f"• Neutral: {neutral_news}"
        )


        news_items = news_result.get(
            "news",
            []
        )


        if news_items:

            st.markdown(
                "### Latest Company Headlines"
            )


            for index, item in enumerate(
                news_items,
                start=1
            ):


                headline = item.get(
                    "title",
                    "Unknown headline"
                )


                publisher = item.get(
                    "publisher",
                    "Unknown"
                )


                sentiment = item.get(
                    "sentiment",
                    "NEUTRAL"
                )


                score = item.get(
                    "score",
                    0
                )


                link = item.get(
                    "link",
                    ""
                )


                with st.container(
                    border=True
                ):

                    st.markdown(
                        f"**{index}. {headline}**"
                    )

                    st.caption(
                        f"📰 {publisher}  •  "
                        f"{sentiment_badge(sentiment)}  •  "
                        f"Score: {score}"
                    )

                    if link:

                        st.link_button(
                            f"Read Article {index} ↗",
                            link
                        )


        else:

            st.info(
                "No detailed news headlines are "
                "currently available."
            )


    # =====================================================
    # MODULE 04 - COORDINATION
    # =====================================================

    with st.expander(
        "🔗 Module 04 — Coordination Engine",
        expanded=False
    ):


        col1, col2, col3 = st.columns(3)


        col1.metric(
            "Coordinated Signal",
            coordination_result[
                "coordinated_signal"
            ]
        )


        col2.metric(
            "Agreement",
            agreement
        )


        col3.metric(
            "Bullish Modules",
            coordination_result[
                "bullish_modules"
            ]
        )


        col4, col5 = st.columns(2)


        col4.metric(
            "Bearish Modules",
            coordination_result[
                "bearish_modules"
            ]
        )


        col5.metric(
            "Neutral Modules",
            coordination_result[
                "neutral_modules"
            ]
        )


    # =====================================================
    # MODULE 05 - CONFIDENCE
    # =====================================================

    with st.expander(
        "🎯 Module 05 — Confidence Analysis",
        expanded=False
    ):


        st.metric(
            "Confidence Score",
            f"{confidence_score:.0f}%"
        )


        safe_confidence = int(
            max(
                0,
                min(
                    100,
                    confidence_score
                )
            )
        )


        st.progress(
            safe_confidence
        )


        st.caption(
            f"Confidence Level: {confidence_level}"
        )


        st.write(
            "The confidence engine measures agreement "
            "between LSTM forecasting, technical analysis "
            "and financial news sentiment."
        )


    # =====================================================
    # MODULE 06 - RISK
    # =====================================================

    with st.expander(
        "⚠️ Module 06 — Risk Analysis",
        expanded=True
    ):


        col1, col2, col3 = st.columns(3)


        col1.metric(
            "Risk Score",
            f"{risk_score}/10"
        )


        col2.metric(
            "Risk Level",
            risk_level
        )


        col3.metric(
            "Market Volatility",
            technical_result[
                "volatility_level"
            ]
        )


        risk_factors = risk_result.get(
            "risk_factors",
            []
        )


        if risk_factors:

            st.markdown(
                "### Risk Factors"
            )

            for factor in risk_factors:

                st.write(
                    f"• {factor}"
                )


        else:

            st.success(
                "No major risk factors were returned."
            )


    # =====================================================
    # SYSTEM INFORMATION
    # =====================================================

    with st.expander(
        "ℹ️ System & Forecast Information",
        expanded=False
    ):


        st.markdown(
            "### Prediction Target"
        )

        st.write(
            "The system predicts the next available "
            "trading-day closing price."
        )


        st.markdown(
            "### Model Input"
        )

        st.write(
            f"The LSTM model uses the previous "
            f"{SEQUENCE_LENGTH} trading days."
        )


        st.markdown(
            "### Forecast Timing"
        )

        st.write(
            "The system is designed for daily closing-price "
            "prediction and does not predict an exact intraday time."
        )


        st.markdown(
            "### AI Pipeline"
        )


        pipeline = [
            "Historical Market Data",
            "Technical Indicator Engineering",
            "LSTM Price Forecasting",
            "Technical Analysis",
            "Financial News Sentiment",
            "Module Coordination",
            "Confidence + Risk Analysis",
            "Final BUY / SELL / HOLD Recommendation"
        ]


        for step in pipeline:

            st.write(
                f"↓  {step}"
            )


    # =====================================================
    # FOOTER
    # =====================================================

    st.divider()

    footer_col1, footer_col2 = st.columns(
        [3, 1]
    )


    with footer_col1:

        st.caption(
            "Multi-Module Stock Prediction System • "
            "LSTM Forecasting • Technical Analysis • "
            "Financial News Sentiment • Risk Intelligence"
        )


    with footer_col2:

        st.caption(
            "⚠️ Not financial advice"
        )


# =========================================================
# ERROR HANDLING
# =========================================================

except Exception as error:

    st.error(
        "The system encountered an error."
    )

    st.exception(
        error
    )