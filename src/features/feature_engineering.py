from __future__ import annotations

import pandas as pd


def engineer_features(features: pd.DataFrame) -> pd.DataFrame:
    """
    Build model-ready features.

    Current feature engineering remains intentionally lightweight and deterministic:
    - creates clinically inspired ratio/index features
    - avoids leakage and target-dependent transforms
    """
    engineered = features.copy()

    if {"trestbps", "chol"}.issubset(engineered.columns):
        engineered["bp_chol_ratio"] = engineered["trestbps"] / (engineered["chol"].replace(0, 1))

    if {"age", "thalach"}.issubset(engineered.columns):
        engineered["age_thalach_interaction"] = engineered["age"] * engineered["thalach"]

    if {"oldpeak", "thalach"}.issubset(engineered.columns):
        engineered["st_depression_per_hr"] = engineered["oldpeak"] / (engineered["thalach"].replace(0, 1))

    return engineered
