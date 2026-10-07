import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler


# =================================================
# STATIONARY MODEL FEATURES
#
# The LSTM is trained on *stationary* transforms of
# the raw OHLCV + indicator columns, NOT on absolute
# price levels. Absolute prices drift permanently
# upward, so a MinMaxScaler fitted on old data puts
# every recent day out of range and the model is
# forced to extrapolate. Returns / ratios stay in a
# stable band for the life of the model.
# =================================================

MODEL_FEATURE_COLUMNS = [
    "ret_1",         # log return of Close vs previous Close
    "ret_open",      # overnight gap: log(Open / prev Close)
    "hl_range",      # (High - Low) / prev Close
    "co_range",      # (Close - Open) / Open
    "vol_ratio",     # Volume / 20d average Volume  (log1p)
    "rsi",           # RSI / 100
    "macd_n",        # MACD / Close
    "macd_sig_n",    # MACD signal / Close
    "macd_hist_n",   # MACD histogram / Close
    "sma20_gap",     # (Close - SMA_20) / SMA_20
    "ema20_gap",     # (Close - EMA_20) / EMA_20
    "sma_cross",     # (SMA_20 - SMA_50) / SMA_50
    "bb_pctb",       # position of Close within the Bollinger band
    "volatility",    # 20d rolling std of daily returns
    # --- momentum / regime context (the real signal) ---
    "mom_5",         # 5-day cumulative log return
    "mom_10",        # 10-day cumulative log return
    "mom_20",        # 20-day cumulative log return
    "dist_hi_252",   # Close / 252-day high  - 1   (<= 0)
    "dist_lo_252",   # Close / 252-day low   - 1   (>= 0)
    "dow_sin",       # day-of-week, cyclical
    "dow_cos",
]

TARGET_COLUMN = "target_return"


# Kept as an alias so older imports do not explode.
FEATURE_COLUMNS = MODEL_FEATURE_COLUMNS


# -------------------------------------------------
# BUILD STATIONARY FEATURE FRAME
# -------------------------------------------------

def build_feature_frame(df, include_target=True):
    """
    Turns an indicator-enriched DataFrame (output of
    add_technical_indicators) into the stationary
    feature matrix the LSTM consumes.

    include_target=True adds `target_return`, the
    NEXT day's log return of Close. The final row
    (no next day yet) is dropped in that case.

    Returns a DataFrame containing exactly
    MODEL_FEATURE_COLUMNS (+ TARGET_COLUMN) with all
    NaN rows removed.
    """

    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    required = [
        "Open", "High", "Low", "Close", "Volume",
        "SMA_20", "SMA_50", "EMA_20", "RSI",
        "MACD", "MACD_Signal", "MACD_Histogram",
        "BB_Upper", "BB_Middle", "BB_Lower",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"build_feature_frame missing columns: {missing}. "
            "Call add_technical_indicators first."
        )

    close = df["Close"].astype(float)
    prev_close = close.shift(1)

    volume = df["Volume"].astype(float)
    volume_avg = volume.rolling(window=20).mean()

    bb_width = (df["BB_Upper"] - df["BB_Lower"]).replace(0, np.nan)

    out = pd.DataFrame(index=df.index)

    out["ret_1"] = np.log(close / prev_close)
    out["ret_open"] = np.log(df["Open"].astype(float) / prev_close)
    out["hl_range"] = (df["High"].astype(float) - df["Low"].astype(float)) / prev_close
    out["co_range"] = (close - df["Open"].astype(float)) / df["Open"].astype(float)
    out["vol_ratio"] = np.log1p(volume / volume_avg)
    out["rsi"] = df["RSI"].astype(float) / 100.0
    out["macd_n"] = df["MACD"].astype(float) / close
    out["macd_sig_n"] = df["MACD_Signal"].astype(float) / close
    out["macd_hist_n"] = df["MACD_Histogram"].astype(float) / close
    out["sma20_gap"] = (close - df["SMA_20"].astype(float)) / df["SMA_20"].astype(float)
    out["ema20_gap"] = (close - df["EMA_20"].astype(float)) / df["EMA_20"].astype(float)
    out["sma_cross"] = (df["SMA_20"].astype(float) - df["SMA_50"].astype(float)) / df["SMA_50"].astype(float)
    out["bb_pctb"] = (close - df["BB_Lower"].astype(float)) / bb_width
    out["volatility"] = df["Volatility"].astype(float) if "Volatility" in df.columns else out["ret_1"].rolling(20).std()

    # ---- momentum / regime context ----
    log_close = np.log(close)
    out["mom_5"] = log_close - log_close.shift(5)
    out["mom_10"] = log_close - log_close.shift(10)
    out["mom_20"] = log_close - log_close.shift(20)

    roll_hi = close.rolling(window=252, min_periods=60).max()
    roll_lo = close.rolling(window=252, min_periods=60).min()
    out["dist_hi_252"] = close / roll_hi - 1.0
    out["dist_lo_252"] = close / roll_lo - 1.0

    dow = pd.Series(df.index.dayofweek, index=df.index).astype(float)
    out["dow_sin"] = np.sin(2.0 * np.pi * dow / 5.0)
    out["dow_cos"] = np.cos(2.0 * np.pi * dow / 5.0)

    # Keep the actual Close so callers can rebuild a
    # price from a predicted return. Dropped before
    # scaling.
    out["_close"] = close

    if include_target:
        out[TARGET_COLUMN] = np.log(close.shift(-1) / close)

    out = out.replace([np.inf, -np.inf], np.nan).dropna()

    return out


# -------------------------------------------------
# CREATE LSTM SEQUENCES
# -------------------------------------------------

def create_sequences(features, target, sequence_length=60):
    """
    features : 2D array (rows x MODEL_FEATURE_COLUMNS)
    target   : 1D array aligned row-for-row with features

    X[i] = features[i-sequence_length : i]
    y[i] = target[i-1]

    i.e. the last row of each window is the "today"
    whose NEXT-day return we predict.
    """

    X = []
    y = []

    for i in range(sequence_length, len(features) + 1):

        X.append(features[i - sequence_length:i])
        y.append(target[i - 1])

    return np.array(X), np.array(y)


# -------------------------------------------------
# PREPARE TRAIN / VALIDATION / TEST DATA
# -------------------------------------------------

def prepare_train_val_test_data(
    df,
    sequence_length=60,
    train_ratio=0.70,
    val_ratio=0.15,
):
    """
    Chronological split. Scalers are fitted on the
    TRAINING rows only (no leakage).

    Returns:
        X_train, y_train,
        X_val,   y_val,
        X_test,  y_test,
        feature_scaler,
        target_scaler,
        meta   (dict: test_prev_close, test_actual_close,
                feature_names, n_rows)
    """

    frame = build_feature_frame(df, include_target=True)

    if len(frame) < sequence_length * 3:
        raise ValueError(
            "Not enough data to build train/val/test sets "
            f"({len(frame)} usable rows)."
        )

    feats = frame[MODEL_FEATURE_COLUMNS].to_numpy(dtype="float64")
    targets = frame[TARGET_COLUMN].to_numpy(dtype="float64")
    closes = frame["_close"].to_numpy(dtype="float64")

    n = len(frame)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    # -------- fit scalers on TRAIN only --------
    feature_scaler = StandardScaler().fit(feats[:train_end])
    target_scaler = StandardScaler().fit(targets[:train_end].reshape(-1, 1))

    feats_s = feature_scaler.transform(feats)
    targets_s = target_scaler.transform(targets.reshape(-1, 1)).ravel()

    # -------- slice with sequence_length warm-up --------
    def _slice(lo, hi):
        lo_w = max(0, lo - sequence_length)
        return (
            feats_s[lo_w:hi],
            targets_s[lo_w:hi],
            closes[lo_w:hi],
        )

    f_tr, t_tr, _ = _slice(0, train_end)
    f_va, t_va, _ = _slice(train_end, val_end)
    f_te, t_te, c_te = _slice(val_end, n)

    X_train, y_train = create_sequences(f_tr, t_tr, sequence_length)
    X_val, y_val = create_sequences(f_va, t_va, sequence_length)
    X_test, y_test = create_sequences(f_te, t_te, sequence_length)

    # For each test sample: close on the window's last
    # day (prev) and the realised next-day close.
    test_prev_close = c_te[sequence_length - 1:len(c_te)]
    test_actual_close = test_prev_close * np.exp(
        target_scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
    )

    meta = {
        "feature_names": list(MODEL_FEATURE_COLUMNS),
        "n_rows": int(n),
        "test_prev_close": test_prev_close,
        "test_actual_close": test_actual_close,
    }

    return (
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        feature_scaler,
        target_scaler,
        meta,
    )
