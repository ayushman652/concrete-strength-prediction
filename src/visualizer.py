
import matplotlib.pyplot as plt
from pathlib import Path

from .config import OUTPUTS_DIR

def plot_training_history(history):
    """Plot training and validation loss across epochs."""

    train_loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, label="Training Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")

    plt.title("Training and Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUTS_DIR / "Train_Val_loss.png", dpi=300, bbox_inches="tight")

    plt.show()


def plot_predictions(y_test, predictions):
    """Compare actual and predicted concrete strength."""

    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, predictions, alpha=0.7)

    # Reference line: perfect predictions lie on this line.
    minimum = min(y_test.min(), predictions.min())
    maximum = max(y_test.max(), predictions.max())

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--",
        label="Perfect Predictions",
    )

    plt.title("Actual vs. Predicted Concrete Strength")
    plt.xlabel("Actual Strength (MPa)")
    plt.ylabel("Predicted Strength (MPa)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUTS_DIR / "Actual_vs_Predicted.png", dpi=300, bbox_inches="tight")
    plt.show()
