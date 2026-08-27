from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam


def build_lstm_model(
    sequence_length,
    num_features
):
    """
    Builds an LSTM model for stock price prediction.
    """

    model = Sequential()

    # First LSTM layer
    model.add(
        LSTM(
            units=64,
            return_sequences=True,
            input_shape=(
                sequence_length,
                num_features
            )
        )
    )

    model.add(
        Dropout(0.2)
    )

    # Second LSTM layer
    model.add(
        LSTM(
            units=32
        )
    )

    model.add(
        Dropout(0.2)
    )

    # Dense layers
    model.add(
        Dense(
            units=16,
            activation="relu"
        )
    )

    # Final prediction layer
    model.add(
        Dense(
            units=1
        )
    )

    # Compile model
    model.compile(
        optimizer=Adam(
            learning_rate=0.001
        ),
        loss="mean_squared_error",
        metrics=["mae"]
    )

    return model