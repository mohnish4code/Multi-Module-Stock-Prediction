# =================================================
# COMPANY CONFIGURATION
# =================================================

COMPANIES = {

    "TCS.NS": {
        "name": "Tata Consultancy Services",
        "model_folder": "TCS"
    },


    "RELIANCE.NS": {
        "name": "Reliance Industries",
        "model_folder": "RELIANCE"
    },


    "INFY.NS": {
        "name": "Infosys",
        "model_folder": "INFY"
    },


    "HDFCBANK.NS": {
        "name": "HDFC Bank",
        "model_folder": "HDFCBANK"
    },


    "ICICIBANK.NS": {
        "name": "ICICI Bank",
        "model_folder": "ICICIBANK"
    },


    "SBIN.NS": {
        "name": "State Bank of India",
        "model_folder": "SBIN"
    },


    "BHARTIARTL.NS": {
        "name": "Bharti Airtel",
        "model_folder": "BHARTIARTL"
    },


    "LT.NS": {
        "name": "Larsen & Toubro",
        "model_folder": "LT"
    },


    "HINDUNILVR.NS": {
        "name": "Hindustan Unilever",
        "model_folder": "HINDUNILVR"
    },


    "ITC.NS": {
        "name": "ITC",
        "model_folder": "ITC"
    }

}


# =================================================
# DATA SETTINGS
# =================================================

DATA_PERIOD = "10y"

DATA_INTERVAL = "1d"

# Period used by the live app / system for a single
# next-day forecast. Needs SEQUENCE_LENGTH + a warm-up
# buffer for the 252-day rolling momentum features.
LIVE_DATA_PERIOD = "5y"


# =================================================
# MODEL SETTINGS
# =================================================

SEQUENCE_LENGTH = 60

# Chronological split. Fitted scalers use TRAIN only.
TRAIN_RATIO = 0.70

VAL_RATIO = 0.15

# TEST_RATIO is the remainder (0.15).

EPOCHS = 120

BATCH_SIZE = 32

EARLY_STOPPING_PATIENCE = 15


# =================================================
# SIGNAL SETTINGS
# =================================================

# Predicted next-day % move inside +/- this band is
# treated as NEUTRAL / HOLD. Used by every entry
# point so the definition of BULLISH/BEARISH is
# identical across app.py, system.py and predictions.
NEUTRAL_THRESHOLD_PCT = 1.0


# =================================================
# NEWS SETTINGS
# =================================================

# Preferred freshness window. Headlines newer than
# this are always used.
NEWS_RECENCY_DAYS = 21

# If the recency window yields fewer than this many
# headlines, fall back to the newest available for
# that company regardless of age.
NEWS_MIN_ITEMS = 4

# Hard cap on how many headlines to analyse.
NEWS_MAX_ITEMS = 15
