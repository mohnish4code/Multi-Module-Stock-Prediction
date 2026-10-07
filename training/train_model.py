import os
import sys
import json
import datetime

import joblib
import numpy as np

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)


# =================================================
# PROJECT ROOT
# =================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from modules.data_collection import get_stock_data
from modules.feature_engineering import add_technical_indicators
from modules.preprocessing import prepare_train_val_test_data
from modules.evaluation import evaluate_return_predictions
from models.model import build_lstm_model

from config import (
    COMPANIES,
    DATA_PERIOD,
    DATA_INTERVAL,
    SEQUENCE_LENGTH,
    TRAIN_RATIO,
    VAL_RATIO,
    EPOCHS,
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
)


# =================================================
# RESOLVE COMPANY
# =================================================

def resolve_company(company_input):
    """Accepts 'TCS', 'tcs', 'TCS.NS' -> (ticker, model_folder, name)."""

    key = str(company_input).upper().strip()

    if key in COMPANIES:
        info = COMPANIES[key]
        return key, info["model_folder"], info["name"]

    for ticker, info in COMPANIES.items():
        if key == ticker.replace(".NS", ""):
            return ticker, info["model_folder"], info["name"]
        if key == info["model_folder"].upper():
            return ticker, info["model_folder"], info["name"]

    raise ValueError(f"Company '{company_input}' not found in config.COMPANIES")


# =================================================
# TRAIN ONE COMPANY
# =================================================

def train_company(company_input, verbose=1):
    """
    Full pipeline for a single company:
    fetch -> features -> split -> train -> evaluate -> save.

    Saves into models/<folder>/:
        lstm_model.keras
        feature_scaler.pkl
        target_scaler.pkl
        metrics.json
        training_info.txt

    Returns the metrics dict.
    """

    ticker, model_folder, name = resolve_company(company_input)

    print("\n" + "=" * 62)
    print(f"TRAINING  {name}  ({ticker})  ->  models/{model_folder}")
    print("=" * 62)

    model_dir = os.path.join(PROJECT_ROOT, "models", model_folder)
    os.makedirs(model_dir, exist_ok=True)

    # ---------------- data ----------------
    df = get_stock_data(
        ticker=ticker,
        period=DATA_PERIOD,
        interval=DATA_INTERVAL,
    )

    if df is None or df.empty:
        raise RuntimeError(f"No data downloaded for {ticker}")

    df = add_technical_indicators(df)

    (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        feature_scaler,
        target_scaler,
        meta,
    ) = prepare_train_val_test_data(
        df,
        sequence_length=SEQUENCE_LENGTH,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
    )

    print(
        f"samples  train={len(X_train)}  "
        f"val={len(X_val)}  test={len(X_test)}  "
        f"features={X_train.shape[2]}"
    )

    # ---------------- model ----------------
    model = build_lstm_model(
        sequence_length=X_train.shape[1],
        num_features=X_train.shape[2],
    )

    model_path = os.path.join(model_dir, "lstm_model.keras")

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=verbose,
        ),
        ModelCheckpoint(
            filepath=model_path,
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=0,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=6,
            min_lr=1e-5,
            verbose=verbose,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=verbose,
    )

    # ---------------- evaluate on unseen test ----------------
    pred_scaled = model.predict(X_test, verbose=0)
    pred_return = target_scaler.inverse_transform(
        pred_scaled.reshape(-1, 1)
    ).ravel()
    true_return = target_scaler.inverse_transform(
        y_test.reshape(-1, 1)
    ).ravel()

    metrics = evaluate_return_predictions(
        y_true_return=true_return,
        y_pred_return=pred_return,
        prev_close=meta["test_prev_close"],
    )

    metrics.update({
        "company": name,
        "ticker": ticker,
        "model_folder": model_folder,
        "trained_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "data_period": DATA_PERIOD,
        "sequence_length": SEQUENCE_LENGTH,
        "n_features": int(X_train.shape[2]),
        "epochs_ran": len(history.history["loss"]),
        "best_val_loss": round(float(min(history.history["val_loss"])), 6),
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "test_samples": int(len(X_test)),
        "target": "next_day_log_return",
    })

    # ---------------- save ----------------
    joblib.dump(feature_scaler, os.path.join(model_dir, "feature_scaler.pkl"))
    joblib.dump(target_scaler, os.path.join(model_dir, "target_scaler.pkl"))

    with open(os.path.join(model_dir, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    with open(os.path.join(model_dir, "training_info.txt"), "w", encoding="utf-8") as fh:
        fh.write(f"Company: {name}\n")
        fh.write(f"Ticker: {ticker}\n")
        fh.write(f"Trained: {metrics['trained_at']}\n\n")
        fh.write(f"Target: next-day log return\n")
        fh.write(f"Features: {X_train.shape[2]} stationary transforms\n")
        fh.write(f"Sequence length: {SEQUENCE_LENGTH}\n\n")
        fh.write("DATA SPLIT (chronological)\n")
        fh.write(f"Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}\n\n")
        fh.write("HOLD-OUT TEST METRICS\n")
        fh.write(f"Directional accuracy: {metrics['directional_accuracy']}%\n")
        fh.write(f"  (baseline always-up: {metrics['baseline_always_up']}%)\n")
        fh.write(
            f"High-conviction accuracy: {metrics['conviction_accuracy']}% "
            f"(top {metrics['conviction_coverage']}% of days)\n"
        )
        fh.write(f"MAPE: {metrics['mape']}%\n")
        fh.write(f"MAE:  Rs {metrics['mae']}\n")
        fh.write(f"RMSE: Rs {metrics['rmse']}\n")

    print(
        f"\nDONE {name}: dir-acc {metrics['directional_accuracy']}%  "
        f"(base {metrics['baseline_always_up']}%)  "
        f"conviction {metrics['conviction_accuracy']}%  "
        f"MAPE {metrics['mape']}%  epochs {metrics['epochs_ran']}"
    )

    return metrics


# =================================================
# CLI
# =================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("\nUsage: python training/train_model.py <COMPANY>\n")
        print("Available:")
        for t, i in COMPANIES.items():
            print(f"  {i['model_folder']:<12} {t}")
        sys.exit(1)

    train_company(sys.argv[1])
