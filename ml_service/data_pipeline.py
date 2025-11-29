# ml_service/data_pipeline.py

from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


from config import (
    CSV_PATH,
    TARGET_COL,
    TEST_SIZE,
    RANDOM_STATE
)

@dataclass
class DatasetSplits:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]


def load_processed_dataframe(csv_path: Path = CSV_PATH) -> pd.DataFrame:
    """
    Load the already-preprocessed dataset (seconds_1990 is already normalized).
    """
    df = pd.read_csv(csv_path)
    return df


def _encode_target(df: pd.DataFrame, target_col: str = TARGET_COL) -> pd.DataFrame:
    """
    Ensure the target column is 0/1 (int8). Handles 'Yes'/'No' if they exist.
    """
    df = df.copy()

    if df[target_col].dtype == object:
        df[target_col] = (
            df[target_col]
            .map({"Yes": 1, "No": 0})
            .fillna(df[target_col])
        )

    df[target_col] = df[target_col].astype(int).astype("int8")
    return df


def _convert_numeric_to_float32(df: pd.DataFrame, exclude_cols: List[str]) -> pd.DataFrame:
    """
    Convert all numeric columns except excluded ones to float32.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
    numeric_cols = [c for c in numeric_cols if c not in exclude_cols]

    for col in numeric_cols:
        df[col] = df[col].astype("float32")

    return df


def prepare_train_test_splits(
    csv_path: Path = CSV_PATH,
    target_col: str = TARGET_COL,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> DatasetSplits:
    """
    Offline data prep for training:

    1. Load processed CSV (seconds_1990 already normalized once).
    2. Encode target to 0/1 int8.
    3. Separate X / y.
    4. Stratified train/test split.
    5. Convert features to float32.
    """
    # 1. Load
    df = load_processed_dataframe(csv_path)

    # 2. Encode target
    df = _encode_target(df, target_col=target_col)

    # 3. Split features / target
    feature_cols = [c for c in df.columns if c != target_col]
    X_df = df[feature_cols]
    y = df[target_col].values.astype("int8")

    # 4. Stratified split
    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X_df,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # 5. Convert numeric features to float32
    X_train_df = _convert_numeric_to_float32(X_train_df, exclude_cols=[])
    X_test_df = _convert_numeric_to_float32(X_test_df, exclude_cols=[])

    # To numpy
    X_train = X_train_df.values
    X_test = X_test_df.values

    return DatasetSplits(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_cols,
    )