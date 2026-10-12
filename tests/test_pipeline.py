"""
Tests for preprocessing and feature engineering pipeline.
"""

import pytest
import pandas as pd
import numpy as np
import joblib
from src.utils.config import load_config, get_project_root
from src.features.preprocessing import EducationalFeatureEngineer


def test_feature_engineering_creates_expected_columns():
    engineer = EducationalFeatureEngineer()
    sample_df = pd.DataFrame([{
        "Medu": 3,
        "Fedu": 2,
        "Dalc": 1,
        "Walc": 2,
        "studytime": 2,
        "freetime": 3,
        "goout": 4,
        "failures": 1,
        "absences": 6
    }])
    transformed = engineer.transform(sample_df)
    
    assert "parent_edu_avg" in transformed.columns
    assert transformed["parent_edu_avg"].iloc[0] == 2.5
    assert "total_alcohol" in transformed.columns
    assert transformed["total_alcohol"].iloc[0] == 3
    assert "academic_friction_index" in transformed.columns
    assert transformed["academic_friction_index"].iloc[0] == (1 * 3.0 + 6 / 4.0)


def test_preprocessor_artifact_loads_and_transforms():
    config = load_config()
    p_path = get_project_root() / config["paths"]["preprocessor_path"]
    assert p_path.exists(), "Preprocessor artifact must exist on disk"
    
    preprocessor = joblib.load(p_path)
    features = preprocessor.get_feature_names_out()
    assert len(features) > 30, "Preprocessor must produce one-hot encoded feature matrix"
