import os
import sys
import json
import datetime

import joblib

from tensorflow.keras.models import load_model


PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from modules.data_collection import get_stock_data
from modules.feature_engineering import add_technical_indicators
from modules.preprocessing import prepare_train_val_test_data
from modules.evaluation import evaluate_return_predictions
from training.train_model import resolve_company

from config import (
    DATA_PERIOD,
    DATA_INTERVAL,
    SEQUENCE_LENGTH,
    TRAIN_RATIO,
    VAL_RATIO,
)


def evaluate_company(company_input, write=True):
    """
    Recomputes hold-out metrics for an already-trained
    company and (optionally) rewrites metrics.json.
    """

    ticker, model_folder, name = resolve_company(company_input)
    model_dir = os.path.join(PROJECT_ROOT, "models", model_folder)

    model_path = os.path.join(model_dir, "lstm_model.keras")
    fs_path = os.path.join(model_dir, "feature_scaler.pkl")
    ts_path = os.path.join(model_dir, "target_scaler.pkl")

    for p in (model_path, fs_path, ts_path):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Missing artifact: {p}\nRun training/retrain_all.py first."
            )

    model = load_model(model_path)
    target_scaler = joblib.load(ts_path)

    df = get_stock_data(ticker=ticker, period=DATA_PERIOD, interval=DATA_INTERVAL)
    df = add_technical_indicators(df)

    (
        _, _, _, _,
        X_test, y_test,
        _feature_scaler, _target_scaler,
        meta,
    ) = prepare_train_val_test_data(
        df,
        sequence_length=SEQUENCE_LENGTH,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
    )

    pred_scaled = model.predict(X_test, verbose=0)
    pred_return = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
    true_return = target_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()

    metrics = evaluate_return_predictions(
        y_true_return=true_return,
        y_pred_return=pred_return,
        prev_close=meta["test_prev_close"],
    )
    metrics.update({
        "company": name,
        "ticker": ticker,
        "model_folder": model_folder,
        "evaluated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "sequence_length": SEQUENCE_LENGTH,
        "target": "next_day_log_return",
    })

    print("\n" + "=" * 60)
    print(f"EVALUATION  {name}  ({ticker})")
    print("=" * 60)
    print(f"Test samples:          {metrics['n_samples']}")
    print(f"Directional accuracy:  {metrics['directional_accuracy']}%")
    print(f"  baseline always-up:  {metrics['baseline_always_up']}%")
    print(f"  baseline persistence:{metrics['baseline_persistence']}%")
    print(f"Up-day accuracy:       {metrics['up_accuracy']}%")
    print(f"Down-day accuracy:     {metrics['down_accuracy']}%")
    print(f"MAPE:                  {metrics['mape']}%")
    print(f"MAE:                   Rs {metrics['mae']}")
    print(f"RMSE:                  Rs {metrics['rmse']}")

    if write:
        existing = {}
        mpath = os.path.join(model_dir, "metrics.json")
        if os.path.exists(mpath):
            try:
                with open(mpath, "r", encoding="utf-8") as fh:
                    existing = json.load(fh)
            except Exception:
                existing = {}
        existing.update(metrics)
        with open(mpath, "w", encoding="utf-8") as fh:
            json.dump(existing, fh, indent=2)
        print(f"\nUpdated {mpath}")

    return metrics


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python training/evaluate_model.py <COMPANY>")
        sys.exit(1)
    evaluate_company(sys.argv[1])
