from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET_COLUMN = "target"


def load_dataset(data_path: str | Path) -> pd.DataFrame:
    """Load heart disease dataset from CSV."""
    dataset = pd.read_csv(data_path)
    if TARGET_COLUMN not in dataset.columns:
        raise ValueError(f"Required target column '{TARGET_COLUMN}' not found in dataset.")
    return dataset


def normalize_target(df: pd.DataFrame) -> pd.DataFrame:
    """Convert target to binary classes (0=no disease, 1=disease)."""
    normalized = df.copy()
    normalized[TARGET_COLUMN] = normalized[TARGET_COLUMN].apply(lambda value: 0 if value == 0 else 1)
    return normalized


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Split dataset into features and target."""
    features = df.drop(columns=[TARGET_COLUMN])
    target = df[TARGET_COLUMN]
    return features, target


def build_preprocessor(numeric_features: Iterable[str]) -> ColumnTransformer:
    """Create sklearn preprocessing pipeline for numeric features."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    return ColumnTransformer(
        transformers=[("num", numeric_pipeline, list(numeric_features))],
        remainder="drop",
    )


def make_train_test_data(
    features: pd.DataFrame,
    target: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create stratified train/test split."""
    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )
