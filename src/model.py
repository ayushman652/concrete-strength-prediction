
from tensorflow import keras
from tensorflow.keras import layers

from .config import (
    HIDDEN_LAYERS,
    UNITS_PER_LAYER,
    LEARNING_RATE,
)


def build_model(n_features: int) -> keras.Model:
    """Build and compile a neural network for concrete-strength regression."""

    model = keras.Sequential()
    model.add(keras.Input(shape=(n_features,)))

    # Hidden layers
    for _ in range(HIDDEN_LAYERS):
        model.add(
            layers.Dense(
                UNITS_PER_LAYER,
                activation="relu",
            )
        )

    # Output layer: linear activation for regression
    model.add(layers.Dense(1))

    # Configure the learning process
    optimizer = keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    )

    model.compile(
        optimizer=optimizer,
        loss="mean_squared_error",
        metrics=["mean_absolute_error"],
    )

    return model
