
from pathlib import Path


# Project paths
# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEEP_LEARNING_ROOT = PROJECT_ROOT.parent
WORKSPACE_ROOT = DEEP_LEARNING_ROOT.parent

DATA_PATH = DEEP_LEARNING_ROOT / "datasets" / "concrete_data.csv"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

# Dataset configuration
FEATURE_COLUMNS = [
    "Cement",
    "Blast Furnace Slag",
    "Fly Ash",
    "Water",
    "Superplasticizer",
    "Coarse Aggregate",
    "Fine Aggregate",
    "Age",
]

TARGET_COLUMN = "Strength"


# Data splitting
RANDOM_STATE = 42
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20  # Fraction of the data remaining after the test split


# Neural network configuration
HIDDEN_LAYERS = 2
UNITS_PER_LAYER = 50


# Training configuration
EPOCHS = 100
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 15
