
from tensorflow import keras

from .config import (
    EPOCHS,
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
)


def train_model(model, X_train, y_train, X_val, y_val):
    """Train the model and monitor validation performance."""

    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=1,
    )

    return history
