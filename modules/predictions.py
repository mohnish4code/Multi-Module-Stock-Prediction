import os
import sys

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from modules.data_collection import get_stock_data
from modules.inference import (
    load_artifacts,
    load_metrics,
    artifact_paths,
    predict_next_day as _predict_next_day,
)
from config import COMPANIES, SEQUENCE_LENGTH, LIVE_DATA_PERIOD


# =================================================
# RESOLVE COMPANY
# =================================================

def find_company(company_input):
    """'TCS' / 'tcs' / 'TCS.NS' -> canonical ticker key, or None."""

    key = str(company_input).upper().strip()

    if key in COMPANIES:
        return key

    for ticker, info in COMPANIES.items():
        if key == ticker.replace(".NS", ""):
            return ticker
        if key == info["model_folder"].upper():
            return ticker

    return None


# =================================================
# PREDICT NEXT DAY  (thin wrapper over modules.inference)
# =================================================

def predict_next_day(company_input, period=None, sequence_length=SEQUENCE_LENGTH):
    """
    Predicts the next trading day's closing price for a
    configured company. Returns a plain dict.
    """

    ticker = find_company(company_input)
    if ticker is None:
        raise ValueError(f"Company '{company_input}' not found in configuration.")

    info = COMPANIES[ticker]
    model_folder = info["model_folder"]

    artifacts = load_artifacts(model_folder)

    df = get_stock_data(
        ticker=ticker,
        period=period or LIVE_DATA_PERIOD,
        interval="1d",
    )
    if df is None or df.empty:
        raise ValueError(f"Could not fetch stock data for {ticker}")

    latest_date = df.index[-1]

    pred = _predict_next_day(df, artifacts, sequence_length=sequence_length)

    metrics = load_metrics(model_folder) or {}
    paths = artifact_paths(model_folder)

    return {
        "company": info["name"],
        "ticker": ticker,
        "latest_data_date": str(getattr(latest_date, "date", lambda: latest_date)()),
        "latest_date": str(getattr(latest_date, "date", lambda: latest_date)()),
        "current_price": round(pred["current_price"], 2),
        "predicted_price": round(pred["predicted_price"], 2),
        "predicted_change": round(pred["predicted_change"], 2),
        "predicted_percentage_change": round(pred["predicted_change_percent"], 2),
        "predicted_return_percent": round(pred["predicted_return"] * 100, 3),
        "direction": pred["prediction_direction"],
        "recommendation_signal": pred["recommendation_signal"],
        "model_directional_accuracy": metrics.get("directional_accuracy"),
        "model_conviction_accuracy": metrics.get("conviction_accuracy"),
        "model_mape": metrics.get("mape"),
        "model_path": paths["model"],
        "feature_scaler_path": paths["feature_scaler"],
        "target_scaler_path": paths["target_scaler"],
    }
