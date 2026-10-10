import tensorflow as tf

from src.data_loader import load_dataset
from src.preprocessing import preprocess_data
from src.model import build_model
from src.trainer import train_model
from src.evaluator import evaluate_model
from src.visualizer import plot_training_history, plot_predictions


def main():
    
    # Set random seed first
    tf.keras.utils.set_random_seed(42)
    
    # 1. Load the dataset
    df = load_dataset()
    print(f"Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")

    # 2. Split the data and scale the features
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        scaler,
    ) = preprocess_data(df)

    print(f"Training samples: {len(y_train)}")
    print(f"Validation samples: {len(y_val)}")
    print(f"Test samples: {len(y_test)}")

    # 3. Build the neural network
    model = build_model(n_features=X_train.shape[1])
    model.summary()

    # 4. Train the model
    history = train_model(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
    )

    # 5. Evaluate on unseen test data
    results, predictions = evaluate_model(
        model,
        X_test,
        y_test,
    )

    # 6. Visualize training and prediction performance
    plot_training_history(history)
    plot_predictions(y_test, predictions)


if __name__ == "__main__":
    main()
