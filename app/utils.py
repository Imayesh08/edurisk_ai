"""
Utility functions for EduRisk AI Streamlit Dashboard.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from src.utils.config import load_config
from src.models.predict import EduRiskPredictor


def get_cached_predictor() -> EduRiskPredictor:
    """Returns singleton EduRiskPredictor instance."""
    return EduRiskPredictor()


def load_metrics_and_metadata() -> Dict[str, Any]:
    """Loads saved model metrics and evaluation bundle."""
    config = load_config()
    metrics_path = PROJECT_ROOT / config["paths"]["metrics_path"]
    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics not found at {metrics_path}")
    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_sample_student_profiles() -> Dict[str, Dict[str, Any]]:
    """Provides representative archetypes for rapid testing."""
    return {
        "Profile A: Student in Need of Urgent Academic Support": {
            "school": "GP",
            "sex": "M",
            "age": 18,
            "address": "R",
            "famsize": "GT3",
            "Pstatus": "A",
            "Medu": 1,
            "Fedu": 1,
            "Mjob": "at_home",
            "Fjob": "other",
            "reason": "course",
            "guardian": "other",
            "traveltime": 3,
            "studytime": 1,
            "failures": 2,
            "schoolsup": "no",
            "famsup": "no",
            "paid": "no",
            "activities": "no",
            "nursery": "no",
            "higher": "no",
            "internet": "no",
            "romantic": "yes",
            "famrel": 2,
            "freetime": 4,
            "goout": 4,
            "Dalc": 3,
            "Walc": 4,
            "health": 3,
            "absences": 12,
            "G1": 7,
        },
        "Profile B: Average / Borderline Student": {
            "school": "GP",
            "sex": "F",
            "age": 16,
            "address": "U",
            "famsize": "GT3",
            "Pstatus": "T",
            "Medu": 2,
            "Fedu": 2,
            "Mjob": "services",
            "Fjob": "services",
            "reason": "home",
            "guardian": "mother",
            "traveltime": 2,
            "studytime": 2,
            "failures": 0,
            "schoolsup": "yes",
            "famsup": "yes",
            "paid": "no",
            "activities": "yes",
            "nursery": "yes",
            "higher": "yes",
            "internet": "yes",
            "romantic": "no",
            "famrel": 4,
            "freetime": 3,
            "goout": 3,
            "Dalc": 1,
            "Walc": 2,
            "health": 4,
            "absences": 4,
            "G1": 10,
        },
        "Profile C: Resilient / High-Performing Student": {
            "school": "GP",
            "sex": "F",
            "age": 16,
            "address": "U",
            "famsize": "LE3",
            "Pstatus": "T",
            "Medu": 4,
            "Fedu": 4,
            "Mjob": "health",
            "Fjob": "teacher",
            "reason": "reputation",
            "guardian": "mother",
            "traveltime": 1,
            "studytime": 3,
            "failures": 0,
            "schoolsup": "no",
            "famsup": "yes",
            "paid": "yes",
            "activities": "yes",
            "nursery": "yes",
            "higher": "yes",
            "internet": "yes",
            "romantic": "no",
            "famrel": 5,
            "freetime": 2,
            "goout": 2,
            "Dalc": 1,
            "Walc": 1,
            "health": 5,
            "absences": 1,
            "G1": 15,
        },
    }
