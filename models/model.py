from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber


def build_lstm_model(
    sequence_length,
    num_features,
):
    """
    LSTM regressor for the next-day *log return* of Close.

    Notes
    -----
    * Huber loss instead of plain MSE: standardised
      daily returns have fat tails (+/- 6 sigma days),
      and MSE lets those few days dominate the gradient
      so the model collapses to predicting the mean.
    * Recurrent dropout for a little regularisation on
      the ~1600 training sequences.
    """

    # Deliberately small. On ~1600 noisy daily sequences
    # a 64/32 LSTM finds its best val loss within a few
    # epochs and then just memorises noise. A compact
    # net generalises better on this signal-to-noise.
    model = Sequential([
        Input(shape=(sequence_length, num_features)),

        LSTM(units=24),
        Dropout(0.35),

        Dense(units=8, activation="relu"),
        Dense(units=1),
    ])

    model.compile(
        optimizer=Adam(learning_rate=7e-4),
        loss=Huber(delta=1.0),
        metrics=["mae"],
    )

    return model
