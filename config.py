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
# MODEL SETTINGS
# =================================================

SEQUENCE_LENGTH = 60

TRAIN_RATIO = 0.80

DATA_PERIOD = "10y"

DATA_INTERVAL = "1d"

EPOCHS = 30

BATCH_SIZE = 32