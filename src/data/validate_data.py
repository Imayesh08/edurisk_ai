"""
Data validation and exploratory inspection module for EduRisk AI.
Validates raw dataset integrity, checks for anomalies, missing values, duplicates,
and calculates class distribution statistics.
"""

import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.utils.config import load_config


def validate_raw_dataset(csv_path: str = None) -> dict:
    """
    Performs comprehensive validation of the student dataset.
    """
    config = load_config()
    file_path = Path(csv_path) if csv_path else PROJECT_ROOT / config["data"]["raw_file"]

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found at {file_path}. Run download_data.py first.")

    # UCI Student Performance is semicolon-separated
    df = pd.read_csv(file_path, sep=";")
    print(f"\n================ DATASET VALIDATION REPORT ================")
    print(f"File Path: {file_path}")
    print(f"Total Observations (Rows): {len(df)}")
    print(f"Total Raw Features (Columns): {len(df.columns)}")

    # 1. Check Missing Values
    null_counts = df.isnull().sum()
    total_nulls = int(null_counts.sum())
    print(f"\nMissing Values Check:")
    print(f" - Total Missing Values: {total_nulls}")
    if total_nulls > 0:
        print(" - Columns with nulls:")
        print(null_counts[null_counts > 0])
    else:
        print(" - Data integrity verified: 0 missing cells.")

    # 2. Check Duplicates
    num_duplicates = int(df.duplicated().sum())
    print(f"\nDuplicate Rows Check:")
    print(f" - Total Duplicate Rows: {num_duplicates}")

    # 3. Target Distribution Analysis (G3 and derived academic risk)
    threshold = config["target"]["risk_threshold"]
    if "G3" not in df.columns:
        raise KeyError("Expected final grade 'G3' column not found in dataset.")

    df["at_risk"] = (df["G3"] < threshold).astype(int)
    risk_counts = df["at_risk"].value_counts().to_dict()
    at_risk_students = risk_counts.get(1, 0)
    lower_risk_students = risk_counts.get(0, 0)
    imbalance_ratio = lower_risk_students / max(1, at_risk_students)

    print(f"\nAcademic Risk Target Analysis (Threshold G3 < {threshold}):")
    print(f" - Lower Risk (G3 >= {threshold}): {lower_risk_students} ({lower_risk_students / len(df):.1%})")
    print(f" - At Risk    (G3 < {threshold}):  {at_risk_students} ({at_risk_students / len(df):.1%})")
    print(f" - Class Imbalance Ratio: {imbalance_ratio:.2f}:1")

    # 4. Feature Types Breakdown
    num_cols = config["features"]["numerical"]
    cat_cols = config["features"]["categorical"]
    leakage_cols = config["target"]["leakage_columns_to_drop"]

    missing_expected_num = [c for c in num_cols if c not in df.columns and c != "at_risk"]
    missing_expected_cat = [c for c in cat_cols if c not in df.columns]

    if missing_expected_num or missing_expected_cat:
        print(f"\n[WARNING] Expected columns missing from dataset:")
        print(f" - Missing numerical: {missing_expected_num}")
        print(f" - Missing categorical: {missing_expected_cat}")

    # 5. Summary Statistics for Key Indicators
    print(f"\nKey Educational Feature Statistics:")
    print(f" - Age: Mean {df['age'].mean():.2f}, Range [{df['age'].min()}, {df['age'].max()}]")
    print(f" - Absences: Mean {df['absences'].mean():.2f}, Median {df['absences'].median():.0f}, Max {df['absences'].max()}")
    print(f" - Prior Failures: Mean {df['failures'].mean():.2f}, Max {df['failures'].max()}")
    print(f" - Study Time (1-4 scale): Mean {df['studytime'].mean():.2f}")
    if "G1" in df.columns:
        print(f" - Early Assessment Score (G1): Mean {df['G1'].mean():.2f}, Std {df['G1'].std():.2f}")
    print(f" - Final Grade (G3): Mean {df['G3'].mean():.2f}, Std {df['G3'].std():.2f}")

    report = {
        "file_path": str(file_path),
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "total_nulls": total_nulls,
        "total_duplicates": num_duplicates,
        "risk_threshold": threshold,
        "at_risk_count": at_risk_students,
        "lower_risk_count": lower_risk_students,
        "at_risk_percentage": float(at_risk_students / len(df)),
        "imbalance_ratio": float(imbalance_ratio),
        "columns": list(df.columns)
    }

    report_path = PROJECT_ROOT / "reports" / "data_validation_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\n[SUCCESS] Validation report written to {report_path}")
    print(f"===========================================================\n")
    return report


if __name__ == "__main__":
    validate_raw_dataset()
