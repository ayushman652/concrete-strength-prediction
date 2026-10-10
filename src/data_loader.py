
import pandas as pd

from .config import DATA_PATH, FEATURE_COLUMNS, TARGET_COLUMN


def load_dataset() -> pd.DataFrame:
    """Load the concrete-strength dataset and validate its columns."""

    if not DATA_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}\n"
            "Check the shared Deep learning/datasets folder."
        )

    try:
        df = pd.read_csv(DATA_PATH)
    except pd.errors.EmptyDataError as exc:
        raise ValueError(
            f"The dataset file is empty: {DATA_PATH}"
        ) from exc

    if df.empty:
        raise ValueError("The dataset contains no data rows.")

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}\n"
            f"Columns found: {df.columns.tolist()}"
        )

    return df
