
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(model, X_test, y_test):

    # Generate predictions
    predictions = model.predict(X_test, verbose=0).ravel()

    # Calculate regression metrics
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions)

    results = {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
    }

    print("\nTest Set Evaluation")
    print("-" * 30)
    print(f"MAE:  {mae:.3f} MPa")
    print(f"MSE:  {mse:.3f}")
    print(f"RMSE: {rmse:.3f} MPa")
    print(f"R²:   {r2:.3f}")

    return results, predictions
