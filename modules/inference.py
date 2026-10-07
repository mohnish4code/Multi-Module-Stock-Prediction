import os
import json

import numpy as np
import joblib

from tensorflow.keras.models import load_model

from modules.feature_engineering import add_technical_indicators
from modules.preprocessing import (
    MODEL_FEATURE_COLUMNS,
    build_feature_frame,
)


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

try:
    from config import NEUTRAL_THRESHOLD_PCT
except Exception:
    NEUTRAL_THRESHOLD_PCT = 1.0


# =================================================
# DIRECTION CLASSIFIER  (single source of truth)
# =================================================

def classify_direction(pct_change, threshold=NEUTRAL_THRESHOLD_PCT):
    """
    One definition of BULLISH / BEARISH / NEUTRAL used
    by app.py, system.py and modules.predictions so the
    three entry points never disagree.
    """

    if pct_change > threshold:
        return "BULLISH", "BUY"

    if pct_change < -threshold:
        return "BEARISH", "SELL"

    return "NEUTRAL", "HOLD"


# =================================================
# MODEL ARTIFACT PATHS / LOADING
# =================================================

def get_model_dir(model_folder):
    return os.path.join(PROJECT_ROOT, "models", model_folder)


def artifact_paths(model_folder):
    d = get_model_dir(model_folder)
    return {
        "model": os.path.join(d, "lstm_model.keras"),
        "feature_scaler": os.path.join(d, "feature_scaler.pkl"),
        "target_scaler": os.path.join(d, "target_scaler.pkl"),
        "metrics": os.path.join(d, "metrics.json"),
    }


def load_artifacts(model_folder):
    """
    Loads model + both scalers. Raises FileNotFoundError
    with a clear message if the company has not been
    (re)trained with the current pipeline.
    """

    paths = artifact_paths(model_folder)

    for key in ("model", "feature_scaler", "target_scaler"):
        if not os.path.exists(paths[key]):
            raise FileNotFoundError(
                f"{key} not found for '{model_folder}'.\n"
                f"Expected: {paths[key]}\n"
                "Run:  python training/retrain_all.py"
            )

    return {
        "model": load_model(paths["model"]),
        "feature_scaler": joblib.load(paths["feature_scaler"]),
        "target_scaler": joblib.load(paths["target_scaler"]),
    }


def load_metrics(model_folder):
    """Returns the saved metrics dict or None."""

    path = artifact_paths(model_folder)["metrics"]

    if not os.path.exists(path):
        return None

    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


# =================================================
# NEXT-DAY PREDICTION  (shared by every caller)
# =================================================

def predict_next_day(df_ohlcv, artifacts, sequence_length=60,
                     neutral_threshold=NEUTRAL_THRESHOLD_PCT):
    """
    df_ohlcv   : raw OHLCV DataFrame (yfinance output)
    artifacts  : dict from load_artifacts()

    Returns a dict with current_price, predicted_price,
    predicted_change, predicted_change_percent,
    predicted_return, prediction_direction,
    recommendation_signal, df_features (indicator frame).
    """

    df_ind = add_technical_indicators(df_ohlcv)

    feat_frame = build_feature_frame(df_ind, include_target=False)

    if len(feat_frame) < sequence_length:
        raise ValueError(
            "Not enough usable rows after feature engineering "
            f"({len(feat_frame)} < {sequence_length})."
        )

    window = feat_frame[MODEL_FEATURE_COLUMNS].to_numpy(dtype="float64")
    window = window[-sequence_length:]

    scaled = artifacts["feature_scaler"].transform(window)
    X = np.expand_dims(scaled, axis=0)

    scaled_pred = artifacts["model"].predict(X, verbose=0)
    pred_return = float(
        artifacts["target_scaler"].inverse_transform(
            np.reshape(scaled_pred, (1, 1))
        )[0, 0]
    )

    current_price = float(feat_frame["_close"].iloc[-1])
    predicted_price = float(current_price * np.exp(pred_return))

    price_change = predicted_price - current_price
    pct_change = (price_change / current_price) * 100.0

    direction, signal = classify_direction(pct_change, neutral_threshold)

    return {
        "current_price": current_price,
        "predicted_price": predicted_price,
        "predicted_return": pred_return,
        "predicted_change": price_change,
        "predicted_change_percent": pct_change,
        "prediction_direction": direction,
        "recommendation_signal": signal,
        "df_features": df_ind,
    }
