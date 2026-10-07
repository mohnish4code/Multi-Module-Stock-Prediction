import numpy as np


# =================================================
# MODEL EVALUATION METRICS
#
# Everything is computed from RETURNS and the
# reconstructed price, so the numbers are directly
# comparable across companies and across time.
# =================================================

def evaluate_return_predictions(
    y_true_return,
    y_pred_return,
    prev_close,
):
    """
    y_true_return : realised next-day log returns (1D)
    y_pred_return : predicted next-day log returns (1D)
    prev_close    : close price on the day the
                    prediction was made (1D, same len)

    Returns a dict of metrics.
    """

    y_true_return = np.asarray(y_true_return, dtype="float64").ravel()
    y_pred_return = np.asarray(y_pred_return, dtype="float64").ravel()
    prev_close = np.asarray(prev_close, dtype="float64").ravel()

    n = len(y_true_return)

    actual_price = prev_close * np.exp(y_true_return)
    predicted_price = prev_close * np.exp(y_pred_return)

    abs_err = np.abs(predicted_price - actual_price)

    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(np.mean((predicted_price - actual_price) ** 2)))
    mape = float(np.mean(abs_err / actual_price) * 100.0)

    # ---- directional accuracy (the number that matters) ----
    true_dir = np.sign(y_true_return)
    pred_dir = np.sign(y_pred_return)

    # treat exact-zero true moves as their own class so a
    # flat prediction is not "free"
    directional_hits = (true_dir == pred_dir)
    directional_accuracy = float(np.mean(directional_hits) * 100.0)

    # accuracy on the days the model is most sure about
    # (top third by |predicted return|) - this is the
    # number that actually matters for acting on a signal
    conviction_accuracy = None
    conviction_coverage = None
    if n >= 9:
        k = max(1, n // 3)
        order = np.argsort(-np.abs(y_pred_return))
        top = order[:k]
        conviction_accuracy = float(
            np.mean(np.sign(y_pred_return[top]) == np.sign(y_true_return[top])) * 100.0
        )
        conviction_coverage = round(k / n * 100.0, 1)

    up_mask = y_true_return > 0
    down_mask = y_true_return < 0

    up_accuracy = (
        float(np.mean(pred_dir[up_mask] > 0) * 100.0)
        if up_mask.any() else None
    )
    down_accuracy = (
        float(np.mean(pred_dir[down_mask] < 0) * 100.0)
        if down_mask.any() else None
    )

    # ---- naive baselines for context ----
    # "always predict up" accuracy = share of up days
    base_rate_up = float(np.mean(up_mask) * 100.0)
    always_up_acc = base_rate_up
    # "predict yesterday's direction again"
    if n > 1:
        persistence_acc = float(
            np.mean(np.sign(y_true_return[1:]) == np.sign(y_true_return[:-1])) * 100.0
        )
    else:
        persistence_acc = None

    return {
        "n_samples": int(n),
        "directional_accuracy": round(directional_accuracy, 2),
        "conviction_accuracy": round(conviction_accuracy, 2) if conviction_accuracy is not None else None,
        "conviction_coverage": conviction_coverage,
        "up_accuracy": round(up_accuracy, 2) if up_accuracy is not None else None,
        "down_accuracy": round(down_accuracy, 2) if down_accuracy is not None else None,
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mape": round(mape, 2),
        "baseline_always_up": round(always_up_acc, 2),
        "baseline_persistence": round(persistence_acc, 2) if persistence_acc is not None else None,
    }
