
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    RANDOM_STATE,
    TEST_SIZE,
    VALIDATION_SIZE,
)


def preprocess_data(df):

    # Separate input features and prediction target
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()

    # Reserve the final test set first
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    # Split the remaining data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_STATE,
    )

    # Fit the scaler using training features only
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Apply the same training-derived transformation elsewhere
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        y_train,
        y_val,
        y_test,
        scaler,
    )
