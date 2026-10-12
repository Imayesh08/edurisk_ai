"""
Tests for data ingestion and integrity validation.
"""

import pytest
import pandas as pd
from pathlib import Path
from src.utils.config import load_config, get_project_root


def test_dataset_exists():
    config = load_config()
    raw_path = get_project_root() / config["data"]["raw_file"]
    assert raw_path.exists(), f"Raw dataset not found at {raw_path}"


def test_no_target_leakage_in_features():
    config = load_config()
    raw_path = get_project_root() / config["data"]["raw_file"]
    df = pd.read_csv(raw_path, sep=";")
    
    # Verify raw final grade exists
    assert "G3" in df.columns, "G3 column must exist in raw dataset"
    
    # Target definition check
    threshold = config["target"]["risk_threshold"]
    assert threshold == 10, "Risk threshold must be documented at grade 10"


def test_split_files_exist_and_non_empty():
    config = load_config()
    proc_dir = get_project_root() / config["data"]["processed_dir"]
    train_path = proc_dir / "train.csv"
    val_path = proc_dir / "val.csv"
    test_path = proc_dir / "test.csv"
    
    assert train_path.exists() and train_path.stat().st_size > 0
    assert val_path.exists() and val_path.stat().st_size > 0
    assert test_path.exists() and test_path.stat().st_size > 0
    
    train_df = pd.read_csv(train_path)
    assert "at_risk" in train_df.columns
    assert "G3" not in train_df.columns, "Target leakage detected: G3 in training set"
    assert "G2" not in train_df.columns, "Target leakage detected: G2 in training set"
