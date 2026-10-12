"""
FastAPI application dependencies and singleton predictor loader.
"""

from functools import lru_cache
from src.models.predict import EduRiskPredictor


@lru_cache()
def get_predictor() -> EduRiskPredictor:
    """Provides a cached singleton instance of EduRiskPredictor."""
    return EduRiskPredictor()
