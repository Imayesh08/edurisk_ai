"""
Preprocessing and Feature Engineering Pipeline for EduRisk AI.
Implements modular Scikit-learn ColumnTransformer, feature engineering,
and leakage-free transformations.
"""

import sys
from pathlib import Path
from typing import Tuple, List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.utils.config import load_config


class EducationalFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Domain-specific feature engineering for educational risk indicators.
    Engineers compound interaction terms strictly from early-stage indicators.
    """

    def __init__(self):
        pass

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_out = X.copy()

        # 1. Parental education average & differential
        if "Medu" in X_out.columns and "Fedu" in X_out.columns:
            X_out["parent_edu_avg"] = (X_out["Medu"] + X_out["Fedu"]) / 2.0
            X_out["parent_edu_diff"] = (X_out["Medu"] - X_out["Fedu"]).abs()

        # 2. Total alcohol consumption (workday + weekend)
        if "Dalc" in X_out.columns and "Walc" in X_out.columns:
            X_out["total_alcohol"] = X_out["Dalc"] + X_out["Walc"]

        # 3. Study to recreation balance
        if "studytime" in X_out.columns and "freetime" in X_out.columns:
            X_out["study_to_free_ratio"] = X_out["studytime"] / (X_out["freetime"] + 0.1)

        # 4. Social distraction / risk index
        if "goout" in X_out.columns and "total_alcohol" in X_out.columns:
            X_out["social_distraction_index"] = X_out["goout"] * X_out["total_alcohol"]

        # 5. Prior academic friction index
        if "failures" in X_out.columns and "absences" in X_out.columns:
            X_out["academic_friction_index"] = (X_out["failures"] * 3.0) + (X_out["absences"] / 4.0)

        return X_out


def create_preprocessor(numerical_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    """
    Builds a Scikit-learn ColumnTransformer with median imputation, standard scaling,
    and one-hot encoding with handle_unknown='ignore'.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    return preprocessor


def prepare_and_split_data(
    csv_path: str = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, Dict[str, Any]]:
    """
    Loads raw dataset, creates academic risk target, removes target leakage columns,
    performs stratified train/val/test split, and applies feature engineering.
    """
    config = load_config()
    raw_path = Path(csv_path) if csv_path else PROJECT_ROOT / config["data"]["raw_file"]
    df = pd.read_csv(raw_path, sep=";")

    # Target definition: G3 < 10 represents academic failure / risk
    risk_threshold = config["target"]["risk_threshold"]
    df["at_risk"] = (df["G3"] < risk_threshold).astype(int)

    # Exclude leakage features (final outcome G3 and mid-term exam G2)
    leakage_cols = config["target"]["leakage_columns_to_drop"]
    feature_cols = [c for c in df.columns if c not in leakage_cols and c != "at_risk"]

    X = df[feature_cols].copy()
    y = df["at_risk"].copy()

    # Engineer educational features
    engineer = EducationalFeatureEngineer()
    X = engineer.fit_transform(X)

    # Update numerical features list to include engineered columns
    engineered_num_cols = [
        "parent_edu_avg",
        "parent_edu_diff",
        "total_alcohol",
        "study_to_free_ratio",
        "social_distraction_index",
        "academic_friction_index",
    ]
    base_num_cols = config["features"]["numerical"]
    all_num_cols = list(set(base_num_cols + engineered_num_cols))
    all_cat_cols = config["features"]["categorical"]

    # Stratified Split: 65% train, 15% validation, 20% test
    random_state = config["split"]["random_state"]
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=config["split"]["test_size"], random_state=random_state, stratify=y
    )

    val_relative_size = config["split"]["val_size"] / (1.0 - config["split"]["test_size"])
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_relative_size, random_state=random_state, stratify=y_train_val
    )

    # Save processed splits for reproducibility
    processed_dir = PROJECT_ROOT / config["data"]["processed_dir"]
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_df = X_train.copy()
    train_df["at_risk"] = y_train
    val_df = X_val.copy()
    val_df["at_risk"] = y_val
    test_df = X_test.copy()
    test_df["at_risk"] = y_test

    train_df.to_csv(processed_dir / "train.csv", index=False)
    val_df.to_csv(processed_dir / "val.csv", index=False)
    test_df.to_csv(processed_dir / "test.csv", index=False)

    metadata = {
        "num_features": all_num_cols,
        "cat_features": all_cat_cols,
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "train_risk_rate": float(y_train.mean()),
        "val_risk_rate": float(y_val.mean()),
        "test_risk_rate": float(y_test.mean()),
    }

    print(f"[INFO] Data prepared and split successfully:")
    print(f" - Train samples: {len(X_train)} (At-risk: {y_train.sum()} / {y_train.mean():.1%})")
    print(f" - Validation samples: {len(X_val)} (At-risk: {y_val.sum()} / {y_val.mean():.1%})")
    print(f" - Test samples: {len(X_test)} (At-risk: {y_test.sum()} / {y_test.mean():.1%})")

    return X_train, X_val, X_test, y_train, y_val, y_test, metadata


if __name__ == "__main__":
    prepare_and_split_data()
