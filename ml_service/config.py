# ml_service/config.py

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent  # repo root/ml_service
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
BEST_CONFIG_PATH = ARTIFACTS_DIR / "best_config.json"
OOF_PROBS_PATH = ARTIFACTS_DIR / "oof_probs_train.npy"
OOF_LABELS_PATH = ARTIFACTS_DIR / "oof_labels_train.npy"

# Raw dataset path
CSV_PATH = DATA_DIR / "final_dataset_processed.csv"

# Target column name
TARGET_COL = "Is Fraud?"

# Train/test split
TEST_SIZE = 0.20
RANDOM_STATE = 42

# Preprocessing-related filenames
SCALER_FILENAME = "seconds_1990_scaler.pkl"
FEATURES_FILENAME = "feature_names.json"

# Ensure dirs exist (safe if they already exist)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
