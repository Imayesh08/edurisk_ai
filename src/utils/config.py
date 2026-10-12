"""
Configuration loader and utility functions for EduRisk AI.
"""

from pathlib import Path
from typing import Any, Dict
import yaml


def get_project_root() -> Path:
    """Returns the absolute root directory of the project."""
    return Path(__file__).resolve().parent.parent.parent


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Loads YAML configuration file."""
    root = get_project_root()
    path = root / config_path
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
