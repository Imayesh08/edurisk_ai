"""
Data acquisition module for EduRisk AI.
Downloads and unpacks the UCI Student Performance dataset.
"""

import os
import sys
import zipfile
import urllib.request
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.utils.config import load_config


def download_dataset(force: bool = False) -> Path:
    """
    Downloads and extracts the UCI Student Performance dataset.
    Source: Cortez, P., & Silva, A. (2008). Using data mining to predict secondary school student performance.
    UCI ML Repository ID: 320
    """
    config = load_config()
    raw_dir = PROJECT_ROOT / config["data"]["raw_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    dest_csv = PROJECT_ROOT / config["data"]["raw_file"]
    math_csv = PROJECT_ROOT / config["data"]["math_file"]

    if dest_csv.exists() and math_csv.exists() and not force:
        print(f"[INFO] Dataset files already present in {raw_dir}")
        return dest_csv

    url = config["data"]["uci_url"]
    zip_path = raw_dir / "student_performance.zip"

    print(f"[INFO] Downloading dataset from {url}...")
    import requests
    import certifi
    
    headers = {"User-Agent": "Mozilla/5.0 (EduRisk-AI Data Loader; Research Project)"}
    response = requests.get(url, headers=headers, verify=certifi.where(), timeout=30)
    response.raise_for_status()
    with open(zip_path, "wb") as out_file:
        out_file.write(response.content)

    print(f"[INFO] Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(raw_dir)

    # Some versions have nested zip files (e.g. student.zip inside)
    nested_zip = raw_dir / "student.zip"
    if nested_zip.exists():
        print(f"[INFO] Extracting nested {nested_zip}...")
        with zipfile.ZipFile(nested_zip, "r") as zip_ref:
            zip_ref.extractall(raw_dir)

    # Verify extracted CSV files
    if not dest_csv.exists():
        raise FileNotFoundError(f"Expected file {dest_csv} was not found after extraction.")

    print(f"[SUCCESS] Dataset successfully extracted to {raw_dir}:")
    print(f" - Portuguese course data: {dest_csv.name} ({dest_csv.stat().st_size} bytes)")
    if math_csv.exists():
        print(f" - Mathematics course data: {math_csv.name} ({math_csv.stat().st_size} bytes)")

    return dest_csv


if __name__ == "__main__":
    download_dataset()
